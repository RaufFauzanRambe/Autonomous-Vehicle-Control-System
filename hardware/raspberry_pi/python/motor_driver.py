"""
File:        python/motor_driver.py
Brief:       MotorDriver — PWM output to a DC motor via H-bridge (L298N,
             DRV8833, TB6612FNG) controlled by either RPi GPIO hardware
             PWM or a PCA9685 I²C PWM driver. Includes a closed-loop
             speed PID controller with optional quadrature-encoder
             feedback.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from utils import PIDController, clamp

try:
    import RPi.GPIO as RPi_GPIO
    _RPI_GPIO_AVAILABLE = True
except Exception:  # pragma: no cover
    RPi_GPIO = None  # type: ignore
    _RPI_GPIO_AVAILABLE = False

try:
    import board, busio
    from adafruit_pca9685 import PCA9685
    _PCA9685_AVAILABLE = True
except Exception:  # pragma: no cover
    PCA9685 = None  # type: ignore
    _PCA9685_AVAILABLE = False


# ----------------------------------------------------------------------
# Enums & dataclasses
# ----------------------------------------------------------------------
class MotorBackend(str, Enum):
    """Where the PWM signal is generated."""

    RPI_GPIO = "rpi_gpio"
    PCA9685 = "pca9685"


class MotorDirection(Enum):
    """H-bridge direction state."""

    FORWARD = 1
    REVERSE = -1
    BRAKE = 0


@dataclass(slots=True)
class EncoderFeedback:
    """Quadrature-encoder feedback state."""

    pin_a: int
    pin_b: int
    counts_per_rev: int = 20
    gear_ratio: float = 30.0
    wheel_radius_m: float = 0.05
    _count: int = 0
    _last_count: int = 0
    _last_timestamp: float = 0.0
    _last_velocity_mps: float = 0.0


# ----------------------------------------------------------------------
# MotorDriver
# ----------------------------------------------------------------------
class MotorDriver:
    """H-bridge DC motor driver with optional speed PID.

    Pin layout (L298N style):
        in_a, in_b  : direction GPIOs (low/low = coast, high/high = brake)
        pwm         : PWM channel (RPi GPIO 12/13/18/19, or PCA9685 channel)

    The PID controller runs at ``pid_rate_hz`` and uses the measured
    wheel speed from the encoder. Without an encoder, the controller
    is bypassed and the driver just sets a duty cycle proportional to
    the commanded speed (open loop).
    """

    RPI_PWM_FREQ_HZ: int = 1000   # hardware PWM freq on RPi
    PCA9685_FREQ_HZ: int = 1000   # PCA9685 freq

    def __init__(self, config: dict) -> None:
        self.backend: MotorBackend = MotorBackend(
            config.get("backend", "rpi_gpio")
        )
        self.in_a_pin: int = int(config.get("in_a_pin", 17))
        self.in_b_pin: int = int(config.get("in_b_pin", 27))
        self.pwm_pin: int = int(config.get("pwm_pin", 18))
        self.pca_channel: int = int(config.get("pca_channel", 0))
        self.pca_address: int = int(config.get("pca_address", 0x40), 0) \
            if isinstance(config.get("pca_address"), str) \
            else int(config.get("pca_address", 0x40))

        self.max_speed_mps: float = float(config.get("max_speed_mps", 1.5))
        self.invert: bool = bool(config.get("invert", False))
        self.use_pid: bool = bool(config.get("use_pid", True))

        # PID gains
        self.pid = PIDController(
            kp=float(config.get("pid_kp", 0.8)),
            ki=float(config.get("pid_ki", 0.05)),
            kd=float(config.get("pid_kd", 0.01)),
            setpoint=0.0,
            output_limits=(-1.0, 1.0),
        )

        # Encoder (optional)
        encoder_cfg = config.get("encoder")
        self.encoder: Optional[EncoderFeedback] = None
        if encoder_cfg:
            self.encoder = EncoderFeedback(
                pin_a=int(encoder_cfg["pin_a"]),
                pin_b=int(encoder_cfg["pin_b"]),
                counts_per_rev=int(encoder_cfg.get("counts_per_rev", 20)),
                gear_ratio=float(encoder_cfg.get("gear_ratio", 30.0)),
                wheel_radius_m=float(encoder_cfg.get("wheel_radius_m", 0.05)),
            )

        # Internal state
        self._pwm = None
        self._pca = None
        self._target_speed_mps: float = 0.0
        self._duty: float = 0.0
        self._direction: MotorDirection = MotorDirection.BRAKE
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pid_thread: Optional[threading.Thread] = None
        self.pid_rate_hz: int = int(config.get("pid_rate_hz", 50))

        self._init_backend()
        if self.use_pid and self.encoder is not None:
            self._start_pid_thread()

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        if self.backend == MotorBackend.RPI_GPIO:
            if not _RPI_GPIO_AVAILABLE:
                logger.warning("RPi.GPIO unavailable — running in stub mode")
                return
            RPi_GPIO.setmode(RPi_GPIO.BCM)
            RPi_GPIO.setwarnings(False)
            RPi_GPIO.setup(self.in_a_pin, RPi_GPIO.OUT, initial=RPi_GPIO.LOW)
            RPi_GPIO.setup(self.in_b_pin, RPi_GPIO.OUT, initial=RPi_GPIO.LOW)
            RPi_GPIO.setup(self.pwm_pin, RPi_GPIO.OUT)
            self._pwm = RPi_GPIO.PWM(self.pwm_pin, self.RPI_PWM_FREQ_HZ)
            self._pwm.start(0.0)
            logger.info("Motor PWM on GPIO {} @ {} Hz", self.pwm_pin,
                        self.RPI_PWM_FREQ_HZ)
        elif self.backend == MotorBackend.PCA9685:
            if not _PCA9685_AVAILABLE:
                logger.warning("PCA9685 unavailable — running in stub mode")
                return
            i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(i2c, address=self.pca_address)
            self._pca.frequency = self.PCA9685_FREQ_HZ
            # Direction pins via GPIO (or extra PCA channels)
            if _RPI_GPIO_AVAILABLE:
                RPi_GPIO.setmode(RPi_GPIO.BCM)
                RPi_GPIO.setup(self.in_a_pin, RPi_GPIO.OUT, initial=RPi_GPIO.LOW)
                RPi_GPIO.setup(self.in_b_pin, RPi_GPIO.OUT, initial=RPi_GPIO.LOW)
            logger.info("Motor PWM on PCA9685 channel {}", self.pca_channel)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Encoder interrupt
        if self.encoder and _RPI_GPIO_AVAILABLE:
            RPi_GPIO.setup(self.encoder.pin_a, RPi_GPIO.IN)
            RPi_GPIO.setup(self.encoder.pin_b, RPi_GPIO.IN)
            RPi_GPIO.add_event_detect(
                self.encoder.pin_a, RPi_GPIO.RISING,
                callback=self._encoder_isr, bouncetime=1,
            )

    # ------------------------------------------------------------------
    # Encoder interrupt
    # ------------------------------------------------------------------
    def _encoder_isr(self, pin: int) -> None:
        assert self.encoder is not None
        a = RPi_GPIO.input(self.encoder.pin_a)
        b = RPi_GPIO.input(self.encoder.pin_b)
        # Quadrature: if A leads B, increment; else decrement.
        if a == b:
            self.encoder._count += 1
        else:
            self.encoder._count -= 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_speed(self, speed_mps: float) -> None:
        """Command a target wheel speed in m/s.

        Negative values reverse the motor. Speeds are clamped to
        ``±max_speed_mps``.
        """
        speed_mps = clamp(speed_mps, -self.max_speed_mps, self.max_speed_mps)
        if self.invert:
            speed_mps = -speed_mps
        with self._lock:
            self._target_speed_mps = speed_mps
            if self.use_pid and self.encoder is not None:
                self.pid.setpoint = speed_mps
            else:
                # Open-loop: duty proportional to speed
                self._set_duty_open_loop(speed_mps)

    def get_target_speed(self) -> float:
        return self._target_speed_mps

    def get_measured_speed(self) -> float:
        """Return the latest measured wheel speed in m/s (or 0)."""
        if self.encoder is None:
            return 0.0
        return self.encoder._last_velocity_mps

    def brake(self) -> None:
        """Short both H-bridge inputs (active brake)."""
        with self._lock:
            self._direction = MotorDirection.BRAKE
            self._write_direction(MotorDirection.BRAKE)
            self._write_duty(0.0)
            self._duty = 0.0
            self._target_speed_mps = 0.0
            self.pid.reset()

    def coast(self) -> None:
        """Release both H-bridge inputs (free-wheel)."""
        with self._lock:
            self._direction = MotorDirection.BRAKE
            self._write_direction(MotorDirection.BRAKE)
            self._write_duty(0.0)
            self._duty = 0.0

    # ------------------------------------------------------------------
    # Open-loop
    # ------------------------------------------------------------------
    def _set_duty_open_loop(self, speed_mps: float) -> None:
        """Set duty cycle proportional to commanded speed."""
        duty = abs(speed_mps) / self.max_speed_mps
        duty = clamp(duty, 0.0, 1.0)
        direction = (MotorDirection.FORWARD if speed_mps >= 0
                     else MotorDirection.REVERSE)
        self._write_direction(direction)
        self._write_duty(duty)
        self._direction = direction
        self._duty = duty

    # ------------------------------------------------------------------
    # Low-level writes
    # ------------------------------------------------------------------
    def _write_direction(self, direction: MotorDirection) -> None:
        if not _RPI_GPIO_AVAILABLE:
            return
        if direction == MotorDirection.FORWARD:
            RPi_GPIO.output(self.in_a_pin, RPi_GPIO.HIGH)
            RPi_GPIO.output(self.in_b_pin, RPi_GPIO.LOW)
        elif direction == MotorDirection.REVERSE:
            RPi_GPIO.output(self.in_a_pin, RPi_GPIO.LOW)
            RPi_GPIO.output(self.in_b_pin, RPi_GPIO.HIGH)
        else:  # BRAKE
            RPi_GPIO.output(self.in_a_pin, RPi_GPIO.LOW)
            RPi_GPIO.output(self.in_b_pin, RPi_GPIO.LOW)

    def _write_duty(self, duty: float) -> None:
        duty = clamp(duty, 0.0, 1.0)
        if self.backend == MotorBackend.RPI_GPIO and self._pwm is not None:
            # RPi.GPIO PWM expects 0..100
            self._pwm.ChangeDutyCycle(duty * 100.0)
        elif self.backend == MotorBackend.PCA9685 and self._pca is not None:
            # 16-bit duty (0..65535)
            self._pca.channels[self.pca_channel].duty_cycle = int(duty * 65535)

    # ------------------------------------------------------------------
    # Closed-loop PID thread
    # ------------------------------------------------------------------
    def _start_pid_thread(self) -> None:
        self._pid_thread = threading.Thread(target=self._pid_loop,
                                            name="motor-pid", daemon=True)
        self._pid_thread.start()
        logger.info("Motor PID thread started ({} Hz)", self.pid_rate_hz)

    def _pid_loop(self) -> None:
        period = 1.0 / self.pid_rate_hz
        while not self._stop_event.is_set():
            try:
                self._pid_step()
            except Exception as exc:  # noqa: BLE001
                logger.exception("PID step failed: {}", exc)
            time.sleep(period)

    def _pid_step(self) -> None:
        """Run one PID iteration."""
        assert self.encoder is not None
        now = time.monotonic()
        # Update measured velocity (rotations/sec → m/s)
        count_delta = self.encoder._count - self.encoder._last_count
        dt = now - self.encoder._last_timestamp
        if dt > 0 and self.encoder._last_timestamp > 0:
            counts_per_sec = count_delta / dt
            rev_per_sec = counts_per_sec / (
                self.encoder.counts_per_rev * self.encoder.gear_ratio
            )
            self.encoder._last_velocity_mps = rev_per_sec * (
                2.0 * 3.141592653589793 * self.encoder.wheel_radius_m
            )
        self.encoder._last_count = self.encoder._count
        self.encoder._last_timestamp = now

        measured = self.encoder._last_velocity_mps
        control = self.pid.update(measured, dt=dt if dt > 0 else 1e-3)
        # control is in [-1, 1]; sign dictates direction
        direction = (MotorDirection.FORWARD if control >= 0
                     else MotorDirection.REVERSE)
        duty = abs(control)
        with self._lock:
            self._write_direction(direction)
            self._write_duty(duty)
            self._direction = direction
            self._duty = duty

    # ------------------------------------------------------------------
    # Stats & cleanup
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "target_speed_mps": self._target_speed_mps,
            "measured_speed_mps": self.get_measured_speed(),
            "duty": self._duty,
            "direction": self._direction.name,
            "encoder_count": self.encoder._count if self.encoder else 0,
            "backend": self.backend.value,
            "pid": self.pid.stats() if self.use_pid and self.encoder else None,
        }

    def close(self) -> None:
        """Stop the PID thread, brake the motor, and release GPIOs."""
        self._stop_event.set()
        if self._pid_thread:
            self._pid_thread.join(timeout=2.0)
        self.brake()
        if self._pwm is not None:
            try:
                self._pwm.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.encoder is not None and _RPI_GPIO_AVAILABLE:
            try:
                RPi_GPIO.remove_event_detect(self.encoder.pin_a)
            except Exception:  # noqa: BLE001
                pass
        if _RPI_GPIO_AVAILABLE:
            try:
                RPi_GPIO.cleanup((self.in_a_pin, self.in_b_pin, self.pwm_pin))
            except Exception:  # noqa: BLE001
                pass
        logger.info("MotorDriver closed")
