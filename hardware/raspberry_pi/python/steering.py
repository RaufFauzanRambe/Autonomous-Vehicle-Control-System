"""
File:        python/steering.py
Brief:       Steering — controls a steering servo via PCA9685 (or RPi
             GPIO software PWM), providing angle setpoints, calibration,
             and a simple Ackermann steering model for differential
             wheel speeds.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from utils import clamp

try:
    import board, busio
    from adafruit_pca9685 import PCA9685
    _PCA9685_AVAILABLE = True
except Exception:  # pragma: no cover
    PCA9685 = None  # type: ignore
    _PCA9685_AVAILABLE = False

try:
    import RPi.GPIO as RPi_GPIO
    _RPI_GPIO_AVAILABLE = True
except Exception:  # pragma: no cover
    RPi_GPIO = None  # type: ignore
    _RPI_GPIO_AVAILABLE = False


# ----------------------------------------------------------------------
# Calibration data
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ServoCalibration:
    """Maps physical steering angles to servo PWM pulse widths."""

    center_pulse_us: float = 1500.0     # 0°
    min_angle_deg: float = -30.0
    max_angle_deg: float = 30.0
    min_pulse_us: float = 1000.0        # at min_angle_deg
    max_pulse_us: float = 2000.0        # at max_angle_deg
    invert: bool = False

    def angle_to_pulse_us(self, angle_deg: float) -> float:
        """Linear interpolation from angle to pulse width."""
        if self.invert:
            angle_deg = -angle_deg
        angle_clamped = clamp(angle_deg, self.min_angle_deg, self.max_angle_deg)
        span = self.max_angle_deg - self.min_angle_deg
        if span == 0:
            return self.center_pulse_us
        t = (angle_clamped - self.min_angle_deg) / span
        return self.min_pulse_us + t * (self.max_pulse_us - self.min_pulse_us)


# ----------------------------------------------------------------------
# Ackermann steering model
# ----------------------------------------------------------------------
@dataclass(slots=True)
class AckermannParams:
    """Ackermann geometry parameters (meters, degrees)."""

    wheelbase_m: float = 0.30
    track_width_m: float = 0.26
    max_steering_angle_deg: float = 27.0

    def turning_radius_m(self, steering_angle_deg: float) -> float:
        """Turning radius of the vehicle center for a given steering angle."""
        if abs(steering_angle_deg) < 0.5:
            return float("inf")
        theta = math.radians(steering_angle_deg)
        return self.wheelbase_m / math.tan(theta)

    def wheel_speed_factors(self, steering_angle_deg: float
                            ) -> tuple[float, float]:
        """Return (left, right) wheel speed multipliers for Ackermann.

        For a left turn (positive angle), the inner (left) wheel slows
        and the outer (right) wheel speeds up. Multipliers are
        normalized so the average is 1.0.
        """
        if abs(steering_angle_deg) < 0.5:
            return (1.0, 1.0)
        r = self.turning_radius_m(steering_angle_deg)
        r_left = r - math.copysign(self.track_width_m / 2.0,
                                   steering_angle_deg)
        r_right = r + math.copysign(self.track_width_m / 2.0,
                                    steering_angle_deg)
        v_avg = (abs(r_left) + abs(r_right)) / 2.0
        return r_left / v_avg, r_right / v_avg


# ----------------------------------------------------------------------
# Steering module
# ----------------------------------------------------------------------
class Steering:
    """Steering servo driver with calibration and Ackermann model.

    The PCA9685 is the recommended backend because it provides a stable
    50 Hz PWM without consuming CPU cycles. RPi GPIO software PWM works
    but may jitter under load.

    Example:
        >>> steer = Steering({"backend": "pca9685", "channel": 0})
        >>> steer.set_angle(15.0)        # turn right 15°
        >>> steer.set_angle(0.0)         # center
    """

    SERVO_FREQ_HZ: int = 50           # standard hobby servo PWM
    SERVO_PERIOD_US: float = 1_000_000 / SERVO_FREQ_HZ

    def __init__(self, config: dict) -> None:
        self.channel: int = int(config.get("channel", 0))
        self.pca_address: int = int(config.get("pca_address", 0x40), 0) \
            if isinstance(config.get("pca_address"), str) \
            else int(config.get("pca_address", 0x40))
        self.rpi_gpio_pin: int = int(config.get("rpi_gpio_pin", 18))

        self.calibration = ServoCalibration(
            center_pulse_us=float(config.get("center_pulse_us", 1500.0)),
            min_angle_deg=float(config.get("min_angle_deg", -30.0)),
            max_angle_deg=float(config.get("max_angle_deg", 30.0)),
            min_pulse_us=float(config.get("min_pulse_us", 1000.0)),
            max_pulse_us=float(config.get("max_pulse_us", 2000.0)),
            invert=bool(config.get("invert", False)),
        )
        self.ackermann = AckermannParams(
            wheelbase_m=float(config.get("wheelbase_m", 0.30)),
            track_width_m=float(config.get("track_width_m", 0.26)),
            max_steering_angle_deg=float(
                config.get("max_steering_angle_deg",
                           self.calibration.max_angle_deg)
            ),
        )

        self.backend: str = config.get("backend", "pca9685")
        self._pca: Optional[object] = None
        self._pwm = None
        self._lock = threading.RLock()
        self._current_angle_deg: float = 0.0

        self._init_backend()
        # Center the steering on startup
        self.set_angle(0.0, smooth=True)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        if self.backend == "pca9685":
            if not _PCA9685_AVAILABLE:
                logger.warning("PCA9685 unavailable — running in stub mode")
                return
            i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(i2c, address=self.pca_address)
            self._pca.frequency = self.SERVO_FREQ_HZ
            logger.info("Steering servo on PCA9685 channel {}", self.channel)
        elif self.backend == "rpi_gpio":
            if not _RPI_GPIO_AVAILABLE:
                logger.warning("RPi.GPIO unavailable — running in stub mode")
                return
            RPi_GPIO.setmode(RPi_GPIO.BCM)
            RPi_GPIO.setwarnings(False)
            RPi_GPIO.setup(self.rpi_gpio_pin, RPi_GPIO.OUT)
            self._pwm = RPi_GPIO.PWM(self.rpi_gpio_pin, self.SERVO_FREQ_HZ)
            self._pwm.start(self._duty_for_pulse_us(self.calibration.center_pulse_us))
            logger.info("Steering servo on GPIO {}", self.rpi_gpio_pin)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Angle control
    # ------------------------------------------------------------------
    def set_angle(self, angle_deg: float, smooth: bool = False,
                  step_deg: float = 1.0, delay_s: float = 0.02) -> None:
        """Command a new steering angle.

        Args:
            angle_deg: Desired steering angle in degrees. Negative =
                       left, positive = right.
            smooth:    If True, ramp from current to target in
                       ``step_deg`` increments with ``delay_s`` between
                       them (avoids mechanical shock).
        """
        target = clamp(angle_deg,
                       self.ackermann.max_steering_angle_deg
                       if self.ackermann.max_steering_angle_deg < 0
                       else -self.ackermann.max_steering_angle_deg,
                       self.ackermann.max_steering_angle_deg)
        target = clamp(target, self.calibration.min_angle_deg,
                       self.calibration.max_angle_deg)

        with self._lock:
            if not smooth or not _PCA9685_AVAILABLE and not _RPI_GPIO_AVAILABLE:
                self._write_angle(target)
                self._current_angle_deg = target
                return
            cur = self._current_angle_deg
            step = step_deg if target > cur else -step_deg
            steps = int(abs(target - cur) / step_deg) + 1
            for _ in range(steps):
                cur = clamp(cur + step,
                            min(cur, target), max(cur, target))
                self._write_angle(cur)
                self._current_angle_deg = cur
                time.sleep(delay_s)
            self._write_angle(target)
            self._current_angle_deg = target

    def get_angle(self) -> float:
        """Return the last commanded steering angle (degrees)."""
        with self._lock:
            return self._current_angle_deg

    def center(self) -> None:
        """Set steering to 0°."""
        self.set_angle(0.0)

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    def calibrate(self, center_pulse_us: Optional[float] = None,
                  endpoints: Optional[tuple[float, float, float, float]] = None
                  ) -> None:
        """Update calibration values at runtime.

        Args:
            center_pulse_us: New pulse width for 0°.
            endpoints: Tuple (min_angle_deg, max_angle_deg,
                              min_pulse_us, max_pulse_us).
        """
        if center_pulse_us is not None:
            self.calibration.center_pulse_us = center_pulse_us
        if endpoints is not None:
            (self.calibration.min_angle_deg,
             self.calibration.max_angle_deg,
             self.calibration.min_pulse_us,
             self.calibration.max_pulse_us) = endpoints
        logger.info("Steering recalibrated: {}", self.calibration)
        self.set_angle(0.0)

    # ------------------------------------------------------------------
    # Ackermann helpers
    # ------------------------------------------------------------------
    def wheel_speed_factors(self, steering_angle_deg: Optional[float] = None
                            ) -> tuple[float, float]:
        """Return (left, right) wheel speed multipliers for Ackermann."""
        angle = (steering_angle_deg if steering_angle_deg is not None
                 else self._current_angle_deg)
        return self.ackermann.wheel_speed_factors(angle)

    def turning_radius_m(self, steering_angle_deg: Optional[float] = None
                         ) -> float:
        """Return turning radius of the vehicle center (m)."""
        angle = (steering_angle_deg if steering_angle_deg is not None
                 else self._current_angle_deg)
        return self.ackermann.turning_radius_m(angle)

    # ------------------------------------------------------------------
    # Low-level write
    # ------------------------------------------------------------------
    def _write_angle(self, angle_deg: float) -> None:
        pulse_us = self.calibration.angle_to_pulse_us(angle_deg)
        if self.backend == "pca9685" and self._pca is not None:
            duty_16 = self._duty_16_for_pulse_us(pulse_us)
            self._pca.channels[self.channel].duty_cycle = duty_16
        elif self.backend == "rpi_gpio" and self._pwm is not None:
            self._pwm.ChangeDutyCycle(self._duty_for_pulse_us(pulse_us))

    def _duty_for_pulse_us(self, pulse_us: float) -> float:
        """RPi.GPIO PWM duty cycle (0..100) for a given pulse width."""
        return 100.0 * pulse_us / self.SERVO_PERIOD_US

    def _duty_16_for_pulse_us(self, pulse_us: float) -> int:
        """PCA9685 16-bit duty value for a given pulse width."""
        return int(65535 * pulse_us / self.SERVO_PERIOD_US)

    # ------------------------------------------------------------------
    # Stats / cleanup
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "current_angle_deg": self._current_angle_deg,
            "backend": self.backend,
            "channel": self.channel,
            "calibration": {
                "center_pulse_us": self.calibration.center_pulse_us,
                "min_angle_deg": self.calibration.min_angle_deg,
                "max_angle_deg": self.calibration.max_angle_deg,
            },
            "turning_radius_m": self.turning_radius_m(),
            "wheel_factors": self.wheel_speed_factors(),
        }

    def close(self) -> None:
        """Center the steering and release the backend."""
        try:
            self.center()
        except Exception:  # noqa: BLE001
            pass
        if self._pwm is not None:
            try:
                self._pwm.stop()
            except Exception:  # noqa: BLE001
                pass
        if _RPI_GPIO_AVAILABLE and self.backend == "rpi_gpio":
            try:
                RPi_GPIO.cleanup(self.rpi_gpio_pin)
            except Exception:  # noqa: BLE001
                pass
        logger.info("Steering closed")
