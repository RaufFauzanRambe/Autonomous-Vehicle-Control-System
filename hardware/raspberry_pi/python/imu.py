"""
File:        python/imu.py
Brief:       ImuModule — driver for the Bosch BNO055 9-DOF IMU over I²C
             on the Raspberry Pi. Provides quaternion, Euler-angle,
             accelerometer, gyroscope and magnetometer readings, plus a
             complementary filter fallback and calibration helpers.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger

try:
    import board
    import busio
    from adafruit_bno055 import BNO055_I2C, _BNO055_ADDRESS
    _BNO055_AVAILABLE = True
except ImportError:  # pragma: no cover - dev workstation
    board = None  # type: ignore
    busio = None  # type: ignore
    BNO055_I2C = None  # type: ignore
    _BNO055_AVAILABLE = False


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ImuData:
    """One IMU sample."""

    # Orientation (quaternion w, x, y, z)
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float
    # Euler angles (degrees)
    roll: float
    pitch: float
    yaw: float
    # Linear acceleration (m/s²) in body frame
    accel_x: float
    accel_y: float
    accel_z: float
    # Angular velocity (rad/s) in body frame
    gyro_x: float
    gyro_y: float
    gyro_z: float
    # Magnetic field (micro-tesla)
    mag_x: float
    mag_y: float
    mag_z: float
    # Calibration status (0..3 each)
    calib_sys: int
    calib_gyro: int
    calib_accel: int
    calib_mag: int
    timestamp: float
    temperature_c: float = 0.0


# ----------------------------------------------------------------------
# Complementary filter (fallback)
# ----------------------------------------------------------------------
class ComplementaryFilter:
    """Fuse accel + gyro + mag into a roll/pitch/yaw estimate.

    Used as a fallback when the BNO055's internal fusion is unavailable
    (e.g. calibration incomplete), or to smooth out drift on the yaw axis.
    """

    def __init__(self, alpha: float = 0.98,
                 gyro_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.alpha = alpha
        self.gyro_bias = np.array(gyro_bias, dtype=np.float32)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._last_t: Optional[float] = None

    def update(self, accel: np.ndarray, gyro: np.ndarray,
               mag: Optional[np.ndarray] = None,
               timestamp: Optional[float] = None) -> Tuple[float, float, float]:
        """Push a new sample and return (roll, pitch, yaw) in degrees."""
        t = timestamp if timestamp is not None else time.monotonic()
        dt = (t - self._last_t) if self._last_t else 0.0
        self._last_t = t

        # Tilt from accelerometer (roll = phi, pitch = theta)
        ax, ay, az = accel
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        # Integrate gyro
        gx, gy, gz = gyro - self.gyro_bias
        if dt > 0:
            self.roll += math.degrees(gx) * dt
            self.pitch += math.degrees(gy) * dt
            self.yaw += math.degrees(gz) * dt

        # Complementary blend
        self.roll = self.alpha * self.roll + (1 - self.alpha) * math.degrees(roll_acc)
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * math.degrees(pitch_acc)

        # Yaw from magnetometer (optional)
        if mag is not None and dt > 0:
            mx, my, mz = mag
            # Compensate for tilt
            cr, sr = math.cos(math.radians(self.roll)), math.sin(math.radians(self.roll))
            cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
            mxh = mx * cp + mz * sp
            myh = mx * sr * sp + my * cp - mz * sr * cp
            yaw_mag = math.degrees(math.atan2(-myh, mxh))
            self.yaw = self.alpha * self.yaw + (1 - self.alpha) * yaw_mag

        return self.roll, self.pitch, self.yaw

    def reset(self) -> None:
        self.roll = self.pitch = self.yaw = 0.0
        self._last_t = None


# ----------------------------------------------------------------------
# ImuModule
# ----------------------------------------------------------------------
class ImuModule:
    """BNO055 IMU driver with a background sampling thread.

    The BNO055 runs its own sensor fusion; this class reads the fused
    output at a configurable rate and stores the latest sample in a
    thread-safe single-slot buffer.
    """

    DEFAULT_RATE_HZ: int = 50
    DEFAULT_I2C_ADDRESS: int = 0x28

    def __init__(self, config: dict) -> None:
        self.address: int = int(config.get("address", self.DEFAULT_I2C_ADDRESS), 0) \
            if isinstance(config.get("address"), str) else int(config.get("address", self.DEFAULT_I2C_ADDRESS))
        self.rate_hz: int = int(config.get("rate_hz", self.DEFAULT_RATE_HZ))
        self.use_internal_fusion: bool = bool(config.get("internal_fusion", True))
        self.alpha: float = float(config.get("complementary_alpha", 0.98))

        self._sensor = None
        self._i2c = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._latest: Optional[ImuData] = None
        self._sample_count: int = 0
        self._error_count: int = 0
        self._filter = ComplementaryFilter(alpha=self.alpha)

        self._open()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _open(self) -> None:
        if not _BNO055_AVAILABLE:
            logger.warning("adafruit_bno055 not installed — running in stub mode")
            return
        try:
            logger.info("Opening BNO055 at I²C 0x{:02x}", self.address)
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = BNO055_I2C(self._i2c, address=self.address)
            if self.use_internal_fusion:
                # NDOF mode = internal 9-DOF fusion
                try:
                    self._sensor.mode = 0x0C  # BNO055_OPERATION_MODE_NDOF
                except Exception:  # noqa: BLE001
                    pass
            logger.info("BNO055 initialized")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open BNO055: {}", exc)
            self._sensor = None
            raise

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop,
                                        name="imu-sample", daemon=True)
        self._thread.start()
        logger.info("IMU sample thread started ({} Hz)", self.rate_hz)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_loop(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                sample = self._read_sample()
                if sample is not None:
                    with self._lock:
                        self._latest = sample
                    self._sample_count += 1
            except Exception as exc:  # noqa: BLE001
                self._error_count += 1
                if self._error_count % 50 == 0:
                    logger.warning("IMU read errors: {}", self._error_count)
            elapsed = time.monotonic() - loop_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _read_sample(self) -> Optional[ImuData]:
        """Read a single IMU sample (stub if no hardware)."""
        if self._sensor is None:
            return self._read_stub_sample()

        # Read each field individually; adafruit-bno055 returns tuples.
        quaternion = self._sensor.quaternion              # (w, x, y, z)
        euler = self._sensor.euler                         # (yaw, pitch, roll) in deg
        accel = self._sensor.acceleration                  # (x, y, z) m/s²
        gyro = self._sensor.gyro                           # (x, y, z) rad/s
        mag = self._sensor.magnetic                        # (x, y, z) µT
        calib = self._sensor.calibration_status            # (sys, gyro, accel, mag)
        temperature = float(self._sensor.temperature)

        w, x, y, z = (float(v) for v in quaternion)
        yaw_deg, pitch_deg, roll_deg = (float(v or 0.0) for v in euler)
        ax, ay, az = (float(v or 0.0) for v in accel)
        gx, gy, gz = (float(v or 0.0) for v in gyro)
        mx, my, mz = (float(v or 0.0) for v in mag)
        cs, cg, ca, cm = calib

        # Optionally run complementary filter for smoothing
        if not self.use_internal_fusion:
            roll_deg, pitch_deg, yaw_deg = self._filter.update(
                np.array([ax, ay, az]),
                np.array([gx, gy, gz]),
                np.array([mx, my, mz]),
                timestamp=time.monotonic(),
            )

        return ImuData(
            quat_w=w, quat_x=x, quat_y=y, quat_z=z,
            roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            mag_x=mx, mag_y=my, mag_z=mz,
            calib_sys=cs, calib_gyro=cg, calib_accel=ca, calib_mag=cm,
            timestamp=time.monotonic(),
            temperature_c=temperature,
        )

    def _read_stub_sample(self) -> ImuData:
        """Generate a synthetic sample for testing without hardware."""
        self._sample_count += 1
        t = self._sample_count / self.rate_hz
        return ImuData(
            quat_w=1.0, quat_x=0.0, quat_y=0.0, quat_z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            accel_x=0.0, accel_y=0.0, accel_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            mag_x=0.0, mag_y=0.0, mag_z=0.0,
            calib_sys=3, calib_gyro=3, calib_accel=3, calib_mag=3,
            timestamp=time.monotonic(),
            temperature_c=25.0 + math.sin(t) * 0.1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read(self) -> Optional[ImuData]:
        """Return the most recent sample."""
        with self._lock:
            return self._latest

    def is_calibrated(self) -> bool:
        """True if all calibration values are >= 2 (acceptable accuracy)."""
        sample = self.read()
        if sample is None:
            return False
        return (sample.calib_sys >= 2 and sample.calib_gyro >= 2
                and sample.calib_accel >= 2 and sample.calib_mag >= 2)

    def calibration_status(self) -> Tuple[int, int, int, int]:
        """Return (sys, gyro, accel, mag) calibration levels 0..3."""
        sample = self.read()
        if sample is None:
            return (0, 0, 0, 0)
        return (sample.calib_sys, sample.calib_gyro,
                sample.calib_accel, sample.calib_mag)

    def run_calibration(self, timeout_s: float = 30.0) -> bool:
        """Block until the BNO055 is calibrated or timeout elapses.

        Calibration procedure:
          * Gyro: leave the sensor still.
          * Accel: hold the sensor at several angles for a few seconds each.
          * Mag: wave the sensor in a figure-8 pattern.

        Returns ``True`` if calibrated within the timeout.
        """
        logger.info("Starting BNO055 calibration (timeout={}s) — perform the "
                    "gyro/accel/mag motions now", timeout_s)
        end_time = time.monotonic() + timeout_s
        while time.monotonic() < end_time:
            if self.is_calibrated():
                logger.info("BNO055 calibrated")
                return True
            time.sleep(0.5)
        logger.warning("BNO055 calibration timed out")
        return False

    # ------------------------------------------------------------------
    # Statistics / cleanup
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "sample_count": self._sample_count,
            "error_count": self._error_count,
            "calibrated": self.is_calibrated(),
            "rate_hz": self.rate_hz,
        }

    def close(self) -> None:
        self.stop()
        if self._sensor is not None:
            try:
                self._sensor.mode = 0x00  # CONFIG_MODE
            except Exception:  # noqa: BLE001
                pass
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:  # noqa: BLE001
                pass
        logger.info("IMU closed")
