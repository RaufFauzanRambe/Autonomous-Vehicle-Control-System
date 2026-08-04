"""
File:        python/utils.py
Brief:       Helper utilities: PID controller, moving average, low-pass
             filter, math helpers (clamp, map_range, haversine, bearing),
             JSON helpers and a YAML-backed Config loader.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Optional, TypeVar

from loguru import logger

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False

T = TypeVar("T", int, float)


# ----------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------
def clamp(value: T, low: T, high: T) -> T:
    """Clamp ``value`` to ``[low, high]``."""
    if low > high:
        low, high = high, low
    if value < low:
        return low
    if value > high:
        return high
    return value


def map_range(value: float, in_min: float, in_max: float,
              out_min: float, out_max: float, clamp_output: bool = True
              ) -> float:
    """Linearly map a value from one range to another.

    Args:
        value:        Input value.
        in_min, in_max: Input range.
        out_min, out_max: Output range.
        clamp_output: If True, clamp to ``[out_min, out_max]``.

    Returns:
        Mapped value.
    """
    if in_max == in_min:
        return out_min
    mapped = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    if clamp_output:
        return clamp(mapped, min(out_min, out_max), max(out_min, out_max))
    return mapped


def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in meters."""
    r = 6_371_000.0  # mean Earth radius
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
    return 2.0 * r * math.asin(math.sqrt(a))


def bearing_to_waypoint_deg(lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
    """Initial great-circle bearing (deg, 0=N, 90=E) from A to B."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360.0) % 360.0


def normalize_angle_deg(angle: float) -> float:
    """Wrap an angle to ``[-180, 180)``."""
    return ((angle + 180.0) % 360.0) - 180.0


def angular_error_deg(setpoint_deg: float, current_deg: float) -> float:
    """Shortest signed angular error ``setpoint - current`` in degrees."""
    return normalize_angle_deg(setpoint_deg - current_deg)


# ----------------------------------------------------------------------
# PID controller
# ----------------------------------------------------------------------
class PIDController:
    """Discrete PID controller with anti-windup and derivative filter.

    Example:
        >>> pid = PIDController(kp=1.0, ki=0.1, kd=0.01, setpoint=1.0)
        >>> for _ in range(100):
        ...     u = pid.update(measured=0.5, dt=0.01)
    """

    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 setpoint: float = 0.0,
                 output_limits: Optional[tuple[float, float]] = (-1.0, 1.0),
                 integral_limit: Optional[float] = None,
                 derivative_filter_alpha: float = 0.1) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.output_limits = output_limits
        self.integral_limit = integral_limit
        self.derivative_filter_alpha = float(derivative_filter_alpha)

        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._last_derivative: float = 0.0
        self._last_output: float = 0.0
        self._last_timestamp: Optional[float] = None
        self._lock = threading.RLock()

    def update(self, measured: float, dt: Optional[float] = None) -> float:
        """Compute the next control output.

        Args:
            measured: Current process value.
            dt:       Delta time in seconds. If ``None``, use monotonic clock.

        Returns:
            Control output (clamped to ``output_limits``).
        """
        now = time.monotonic()
        if dt is None:
            if self._last_timestamp is None:
                dt = 0.0
            else:
                dt = max(now - self._last_timestamp, 1e-6)
        self._last_timestamp = now

        error = self.setpoint - measured
        with self._lock:
            # Integral with anti-windup
            self._integral += error * dt
            if self.integral_limit is not None:
                self._integral = clamp(self._integral,
                                       -self.integral_limit,
                                       self.integral_limit)
            elif self.output_limits is not None:
                # Clamp integral so ki*integral can't exceed output range
                if self.ki != 0:
                    lim = (self.output_limits[1] - self.output_limits[0]) \
                          / (2.0 * self.ki)
                    self._integral = clamp(self._integral, -lim, lim)

            # Derivative with low-pass filter
            if dt > 0:
                raw_d = (error - self._last_error) / dt
            else:
                raw_d = 0.0
            alpha = self.derivative_filter_alpha
            self._last_derivative = (alpha * raw_d
                                     + (1.0 - alpha) * self._last_derivative)

            # Output
            output = (self.kp * error
                      + self.ki * self._integral
                      - self.kd * self._last_derivative)

            # Clamp output
            if self.output_limits is not None:
                output = clamp(output, self.output_limits[0],
                               self.output_limits[1])
            self._last_error = error
            self._last_output = output
            return output

    def reset(self) -> None:
        """Zero out the integral and derivative state."""
        with self._lock:
            self._integral = 0.0
            self._last_error = 0.0
            self._last_derivative = 0.0
            self._last_output = 0.0
            self._last_timestamp = None

    def set_gains(self, kp: Optional[float] = None,
                  ki: Optional[float] = None,
                  kd: Optional[float] = None) -> None:
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)

    def stats(self) -> dict:
        return {
            "kp": self.kp, "ki": self.ki, "kd": self.kd,
            "setpoint": self.setpoint,
            "integral": self._integral,
            "last_error": self._last_error,
            "last_output": self._last_output,
        }


# ----------------------------------------------------------------------
# Moving average
# ----------------------------------------------------------------------
class MovingAverage:
    """Rolling mean of the last N samples."""

    def __init__(self, window: int = 10) -> None:
        self.window = max(1, window)
        self._samples: Deque[float] = deque(maxlen=self.window)

    def push(self, value: float) -> float:
        self._samples.append(float(value))
        return sum(self._samples) / len(self._samples)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)


# ----------------------------------------------------------------------
# Low-pass filter (1st-order, exponential)
# ----------------------------------------------------------------------
class LowPassFilter:
    """First-order IIR low-pass filter.

    ``y[n] = alpha * x[n] + (1 - alpha) * y[n-1]``
    """

    def __init__(self, alpha: float = 0.2,
                 initial: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._value: float = float(initial)

    def update(self, value: float) -> float:
        self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self, value: float = 0.0) -> None:
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------
def to_json(obj: Any, indent: Optional[int] = None) -> str:
    """Serialize ``obj`` to JSON, handling ``bytes``/``Path``/``set``."""
    return json.dumps(obj, default=_json_default, indent=indent,
                      separators=None if indent else (",", ":"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, deque):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def from_json(text: str) -> Any:
    """Deserialize JSON text."""
    return json.loads(text)


# ----------------------------------------------------------------------
# Config loader
# ----------------------------------------------------------------------
class Config:
    """Hierarchical config loader backed by YAML.

    Supports dotted-key access (``config.get("control.rate_hz", 30)``),
    nested sections (``config.section("control")`` returns a new
    ``Config`` rooted at that subtree), and runtime overrides via
    environment variables (e.g. ``AV_CONTROL_RATE_HZ=20`` overrides
    ``control.rate_hz``).
    """

    ENV_PREFIX: str = "AV_"

    def __init__(self, data: Optional[Dict[str, Any]] = None,
                 env_overrides: bool = True) -> None:
        self._data: Dict[str, Any] = data or {}
        if env_overrides:
            self._apply_env_overrides()

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load a YAML or JSON config file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise RuntimeError("pyyaml not installed; cannot load YAML")
            data = yaml.safe_load(text) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            # Try YAML first (it's a superset of JSON)
            if _YAML_AVAILABLE:
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data)}")
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(data)

    def _apply_env_overrides(self) -> None:
        """Walk the env vars and override matching dotted keys."""
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue
            stripped = key[len(self.ENV_PREFIX):].lower()
            parts = stripped.split("__")
            if len(parts) < 2:
                continue
            dotted = ".".join(parts)
            self.set(dotted, _coerce_env_value(value))

    def get(self, key: str, default: Any = None) -> Any:
        """Dotted-key getter: ``get("a.b.c", 0)``."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """Dotted-key setter (creates intermediate dicts as needed)."""
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def section(self, key: str) -> "Config":
        """Return a new ``Config`` rooted at ``key`` (or empty if missing)."""
        sub = self.get(key, {})
        if not isinstance(sub, dict):
            sub = {}
        return Config(sub, env_overrides=False)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def save(self, path: str | Path) -> None:
        """Persist config back to disk as YAML."""
        if not _YAML_AVAILABLE:
            raise RuntimeError("pyyaml not installed; cannot save YAML")
        Path(path).write_text(
            yaml.safe_dump(self._data, sort_keys=False),
            encoding="utf-8",
        )

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _coerce_env_value(value: str) -> Any:
    """Try to coerce an env var string to int/float/bool."""
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------
def rate_limit(rate_hz: float) -> Callable[[], None]:
    """Return a callable that sleeps to maintain the given rate.

    Usage:
        wait = rate_limit(30)
        while True:
            do_work()
            wait()
    """
    period = 1.0 / rate_hz if rate_hz > 0 else 0.0
    last = [time.monotonic()]

    def _wait() -> None:
        now = time.monotonic()
        elapsed = now - last[0]
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        last[0] = time.monotonic()

    return _wait


def debounce(period_s: float) -> Callable[[Callable], Callable]:
    """Decorator that ensures the wrapped function is not called more
    than once per ``period_s`` seconds."""
    def decorator(fn: Callable) -> Callable:
        last_call = [0.0]
        lock = threading.Lock()

        def wrapper(*args, **kwargs):
            now = time.monotonic()
            with lock:
                if now - last_call[0] < period_s:
                    return None
                last_call[0] = now
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def now_iso8601() -> str:
    """Current UTC time in ISO 8601 string form."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
