"""
File:        python/ultrasonic.py
Brief:       UltrasonicModule — multi-sensor HC-SR04 driver using RPi
             GPIO (or gpiozero). Supports arrays of sensors with
             median filtering, timeout handling and threaded reads.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

# Try gpiozero first (cleaner API), fall back to RPi.GPIO.
try:
    from gpiozero import DistanceSensor as GpiozeroDistanceSensor
    _GPIOZERO_AVAILABLE = True
except Exception:  # pragma: no cover - dev workstation
    GpiozeroDistanceSensor = None  # type: ignore
    _GPIOZERO_AVAILABLE = False

try:
    import RPi.GPIO as RPi_GPIO
    _RPI_GPIO_AVAILABLE = True
except Exception:  # pragma: no cover - dev workstation
    RPi_GPIO = None  # type: ignore
    _RPI_GPIO_AVAILABLE = False


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SPEED_OF_SOUND_M_PER_S: float = 343.0     # at 20 °C
MAX_RANGE_M: float = 4.0                  # HC-SR04 max
MIN_RANGE_M: float = 0.02                 # HC-SR04 min
DEFAULT_TIMEOUT_S: float = 0.030          # 30 ms ~ 5 m round trip
DEFAULT_SAMPLES: int = 5                  # median filter window


# ----------------------------------------------------------------------
# Median filter
# ----------------------------------------------------------------------
class MedianFilter:
    """Rolling median filter for distance samples."""

    def __init__(self, window: int = DEFAULT_SAMPLES) -> None:
        self.window = max(1, window)
        self._samples: List[float] = []

    def push(self, value: float) -> float:
        """Add a sample and return the current median."""
        self._samples.append(value)
        if len(self._samples) > self.window:
            self._samples.pop(0)
        return statistics.median(self._samples)

    def reset(self) -> None:
        self._samples.clear()


# ----------------------------------------------------------------------
# Sensor definition
# ----------------------------------------------------------------------
@dataclass(slots=True)
class UltrasonicSensorConfig:
    """One HC-SR04 sensor definition."""

    name: str
    trigger_pin: int          # BCM numbering
    echo_pin: int             # BCM numbering
    median_window: int = DEFAULT_SAMPLES
    timeout_s: float = DEFAULT_TIMEOUT_S


# ----------------------------------------------------------------------
# Single sensor driver (using RPi.GPIO directly)
# ----------------------------------------------------------------------
class _HCSR04:
    """Low-level driver for a single HC-SR04 sensor."""

    def __init__(self, cfg: UltrasonicSensorConfig) -> None:
        self.cfg = cfg
        self._filter = MedianFilter(cfg.median_window)
        self._lock = threading.Lock()
        if not _RPI_GPIO_AVAILABLE:
            logger.warning("RPi.GPIO unavailable — sensor '%s' in stub mode",
                           cfg.name)
        else:
            RPi_GPIO.setup(cfg.trigger_pin, RPi_GPIO.OUT, initial=RPi_GPIO.LOW)
            RPi_GPIO.setup(cfg.echo_pin, RPi_GPIO.IN, pull_up_down=RPi_GPIO.PUD_DOWN)
            time.sleep(0.05)  # let the sensor settle

    def read_m(self) -> float:
        """Measure distance in meters. Returns ``inf`` on timeout."""
        if not _RPI_GPIO_AVAILABLE:
            # Stub: return a plausible constant for testing
            return self._filter.push(0.50)

        with self._lock:
            # Send 10 µs trigger pulse
            RPi_GPIO.output(self.cfg.trigger_pin, RPi_GPIO.HIGH)
            time.sleep(1e-5)
            RPi_GPIO.output(self.cfg.trigger_pin, RPi_GPIO.LOW)

            pulse_start: Optional[float] = None
            pulse_end: Optional[float] = None
            deadline = time.monotonic() + self.cfg.timeout_s

            # Wait for echo rise
            while RPi_GPIO.input(self.cfg.echo_pin) == 0:
                if time.monotonic() > deadline:
                    return self._filter.push(float("inf"))
                pulse_start = time.monotonic()

            # Wait for echo fall
            while RPi_GPIO.input(self.cfg.echo_pin) == 1:
                if time.monotonic() > deadline:
                    return self._filter.push(float("inf"))
                pulse_end = time.monotonic()

        if pulse_start is None or pulse_end is None:
            return self._filter.push(float("inf"))

        elapsed = pulse_end - pulse_start
        distance_m = (elapsed * SPEED_OF_SOUND_M_PER_S) / 2.0
        if distance_m < MIN_RANGE_M or distance_m > MAX_RANGE_M:
            distance_m = float("inf")
        return self._filter.push(distance_m)

    def cleanup(self) -> None:
        if _RPI_GPIO_AVAILABLE:
            try:
                RPi_GPIO.cleanup((self.cfg.trigger_pin, self.cfg.echo_pin))
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------
# Ultrasonic module (array of sensors)
# ----------------------------------------------------------------------
class UltrasonicModule:
    """HC-SR04 multi-sensor array with sequential + threaded reads.

    Sensors are fired **sequentially** by default to avoid echo
    cross-talk. Use ``parallel=True`` to fire them all at once — only
    do this if the sensors are well separated physically.

    Example:
        >>> mod = UltrasonicModule({"sensors": [
        ...     {"name": "front", "trigger_pin": 23, "echo_pin": 24},
        ...     {"name": "left",  "trigger_pin": 5,  "echo_pin": 6},
        ... ]})
        >>> distances = mod.read_all()
        >>> distances["front"]
        0.42
    """

    def __init__(self, config: dict) -> None:
        self.parallel: bool = bool(config.get("parallel", False))
        self.settle_s: float = float(config.get("settle_s", 0.05))

        sensor_cfgs: list[dict] = config.get("sensors", [])
        if not sensor_cfgs:
            # Default 4-sensor layout: front, rear, left, right
            sensor_cfgs = [
                {"name": "front", "trigger_pin": 23, "echo_pin": 24},
                {"name": "rear",  "trigger_pin": 20, "echo_pin": 21},
                {"name": "left",  "trigger_pin": 5,  "echo_pin": 6},
                {"name": "right", "trigger_pin": 16, "echo_pin": 12},
            ]

        self._sensors: Dict[str, _HCSR04] = {}
        for cfg_dict in sensor_cfgs:
            cfg = UltrasonicSensorConfig(
                name=cfg_dict["name"],
                trigger_pin=int(cfg_dict["trigger_pin"]),
                echo_pin=int(cfg_dict["echo_pin"]),
                median_window=int(cfg_dict.get("median_window",
                                               DEFAULT_SAMPLES)),
                timeout_s=float(cfg_dict.get("timeout_s",
                                             DEFAULT_TIMEOUT_S)),
            )
            self._sensors[cfg.name] = _HCSR04(cfg)

        if _RPI_GPIO_AVAILABLE:
            RPi_GPIO.setmode(RPi_GPIO.BCM)
            RPi_GPIO.setwarnings(False)

        logger.info("UltrasonicModule initialized with {} sensors",
                    len(self._sensors))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def read_all(self) -> Dict[str, float]:
        """Return a dict {sensor_name: distance_m}.

        Distances are ``float('inf')`` on timeout.
        """
        if self.parallel:
            return self._read_all_parallel()
        return self._read_all_sequential()

    def read(self, name: str) -> float:
        """Read a single sensor by name."""
        sensor = self._sensors.get(name)
        if sensor is None:
            raise KeyError(f"Unknown sensor: {name}")
        return sensor.read_m()

    def _read_all_sequential(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, sensor in self._sensors.items():
            results[name] = sensor.read_m()
            time.sleep(self.settle_s)  # avoid cross-talk
        return results

    def _read_all_parallel(self) -> Dict[str, float]:
        """Fire all sensors simultaneously and join results."""
        results: Dict[str, float] = {}
        threads: List[threading.Thread] = []
        lock = threading.Lock()

        def _read(name: str, sensor: _HCSR04) -> None:
            value = sensor.read_m()
            with lock:
                results[name] = value

        for name, sensor in self._sensors.items():
            t = threading.Thread(target=_read, args=(name, sensor),
                                 daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=1.0)
        return results

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------
    def closest_obstacle(self) -> tuple[str, float]:
        """Return ``(name, distance)`` of the closest reading."""
        results = self.read_all()
        valid = {n: d for n, d in results.items() if d != float("inf")}
        if not valid:
            return ("none", float("inf"))
        name = min(valid, key=valid.get)  # type: ignore[arg-type]
        return name, valid[name]

    def is_clear(self, threshold_m: float = 0.30) -> bool:
        """True if all sensors report > ``threshold_m``."""
        return all(d > threshold_m
                   for d in self.read_all().values()
                   if d != float("inf"))

    def stats(self) -> dict:
        return {
            "num_sensors": len(self._sensors),
            "sensor_names": list(self._sensors.keys()),
            "parallel": self.parallel,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        for sensor in self._sensors.values():
            sensor.cleanup()
        logger.info("UltrasonicModule closed")
