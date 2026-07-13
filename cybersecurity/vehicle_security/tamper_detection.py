"""Physical and cryptographic tamper detection.

The :class:`TamperDetector` aggregates tamper signals from multiple sources:

  - Chassis intrusion switches (GPIO lines on the ECU enclosure).
  - Case-open sensors (magnetic reed switches).
  - Accelerometer anomaly detection (someone trying to extract the ECU).
  - TPM 2.0 counters (NV counter increments when the case is opened).
  - Tamper-evident seals with GPIO continuity sensors.

When any source reports a tamper event, the detector records it with a
timestamp and forwards it to the safety monitor + event logger.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .constants import Severity
from .utils import format_timestamp
from .security_event_logger import SecurityEventLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enums and dataclasses
# --------------------------------------------------------------------------- #


class TamperSource(str, Enum):
    INTRUSION_SWITCH = "intrusion_switch"
    CASE_OPEN = "case_open"
    ACCEL_ANOMALY = "accel_anomaly"
    TPM_COUNTER = "tpm_counter"
    SEAL_BROKEN = "seal_broken"
    MANUAL = "manual"


@dataclass
class TamperEvent:
    timestamp: str
    source: TamperSource
    severity: Severity
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        d["severity"] = self.severity.value
        return d


@dataclass
class AccelerometerSample:
    """One 3-axis accelerometer sample (in m/s^2)."""

    timestamp: float
    x: float
    y: float
    z: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


# --------------------------------------------------------------------------- #
# Hardware abstractions
# --------------------------------------------------------------------------- #


GPIOSnapshot = Dict[str, bool]
GPIOReader = Callable[[], GPIOSnapshot]
AccelReader = Callable[[], AccelerometerSample]
TPMCounterReader = Callable[[], Dict[int, int]]


def _default_gpio_reader() -> GPIOSnapshot:
    return {"chassis_intrusion": False, "case_open": False, "seal_intact": True}


def _default_accel_reader() -> AccelerometerSample:
    return AccelerometerSample(timestamp=time.time(), x=0.0, y=0.0, z=9.81)


def _default_tpm_counter_reader() -> Dict[int, int]:
    return {0: 0, 1: 0}


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


class TamperDetector:
    """Aggregates tamper signals from GPIO, accelerometer, and TPM."""

    def __init__(
        self,
        gpio_reader: Optional[GPIOReader] = None,
        accel_reader: Optional[AccelReader] = None,
        tpm_counter_reader: Optional[TPMCounterReader] = None,
        accel_window: int = 100,
        accel_magnitude_threshold: float = 20.0,  # m/s^2
        accel_jerk_threshold: float = 50.0,  # m/s^3
        logger_: Optional[SecurityEventLogger] = None,
    ) -> None:
        self.gpio_reader = gpio_reader or _default_gpio_reader
        self.accel_reader = accel_reader or _default_accel_reader
        self.tpm_counter_reader = tpm_counter_reader or _default_tpm_counter_reader
        self.accel_window = accel_window
        self.accel_magnitude_threshold = accel_magnitude_threshold
        self.accel_jerk_threshold = accel_jerk_threshold

        self._lock = threading.RLock()
        self._events: List[TamperEvent] = []
        self._accel_history: Deque[AccelerometerSample] = deque(maxlen=accel_window)
        self._last_gpio: GPIOSnapshot = {}
        self._last_tpm_counters: Dict[int, int] = {}
        self._armed = True
        self.event_logger = logger_

    # ------------------------------------------------------------------ #
    # GPIO-based checks
    # ------------------------------------------------------------------ #

    def check_intrusion_switches(self) -> Optional[TamperEvent]:
        """Poll GPIO lines for chassis intrusion and case-open signals."""
        snapshot = self.gpio_reader()
        with self._lock:
            previous = self._last_gpio
            self._last_gpio = snapshot

        event: Optional[TamperEvent] = None
        if snapshot.get("chassis_intrusion") and not previous.get("chassis_intrusion", False):
            event = TamperEvent(
                timestamp=format_timestamp(),
                source=TamperSource.INTRUSION_SWITCH,
                severity=Severity.CRITICAL,
                description="chassis intrusion switch tripped",
                details={"gpio": snapshot},
            )
        elif snapshot.get("case_open") and not previous.get("case_open", False):
            event = TamperEvent(
                timestamp=format_timestamp(),
                source=TamperSource.CASE_OPEN,
                severity=Severity.CRITICAL,
                description="case-open sensor triggered",
                details={"gpio": snapshot},
            )
        elif not snapshot.get("seal_intact", True) and previous.get("seal_intact", True):
            event = TamperEvent(
                timestamp=format_timestamp(),
                source=TamperSource.SEAL_BROKEN,
                severity=Severity.HIGH,
                description="tamper-evident seal broken",
                details={"gpio": snapshot},
            )
        if event:
            self.register_tamper_event(event)
        return event

    # ------------------------------------------------------------------ #
    # Accelerometer checks
    # ------------------------------------------------------------------ #

    def check_accel_anomaly(self) -> Optional[TamperEvent]:
        """Detect anomalous acceleration (physical extraction attempt)."""
        sample = self.accel_reader()
        with self._lock:
            self._accel_history.append(sample)
            history = list(self._accel_history)

        # Magnitude spike
        if sample.magnitude > self.accel_magnitude_threshold:
            event = TamperEvent(
                timestamp=format_timestamp(),
                source=TamperSource.ACCEL_ANOMALY,
                severity=Severity.HIGH,
                description=f"acceleration magnitude spike: {sample.magnitude:.2f} m/s^2",
                details={"sample": asdict(sample)},
            )
            self.register_tamper_event(event)
            return event

        # Jerk (derivative of magnitude) - looks for sudden impact
        if len(history) >= 2:
            prev = history[-2]
            dt = max(sample.timestamp - prev.timestamp, 1e-6)
            jerk = abs(sample.magnitude - prev.magnitude) / dt
            if jerk > self.accel_jerk_threshold:
                event = TamperEvent(
                    timestamp=format_timestamp(),
                    source=TamperSource.ACCEL_ANOMALY,
                    severity=Severity.MEDIUM,
                    description=f"acceleration jerk anomaly: {jerk:.2f} m/s^3",
                    details={"jerk": jerk, "sample": asdict(sample)},
                )
                self.register_tamper_event(event)
                return event
        return None

    # ------------------------------------------------------------------ #
    # TPM counter checks
    # ------------------------------------------------------------------ #

    def check_tpm_counters(self) -> Optional[TamperEvent]:
        """Detect increments of TPM NV counters (e.g. case-open counter)."""
        counters = self.tpm_counter_reader()
        with self._lock:
            previous = self._last_tpm_counters
            self._last_tpm_counters = counters

        for idx, value in counters.items():
            prev_value = previous.get(idx, value)
            if value > prev_value:
                event = TamperEvent(
                    timestamp=format_timestamp(),
                    source=TamperSource.TPM_COUNTER,
                    severity=Severity.CRITICAL,
                    description=f"TPM NV counter {idx} incremented ({prev_value} -> {value})",
                    details={"counter_index": idx, "previous": prev_value, "current": value},
                )
                self.register_tamper_event(event)
                return event
        return None

    # ------------------------------------------------------------------ #
    # Event registration
    # ------------------------------------------------------------------ #

    def register_tamper_event(self, event: TamperEvent) -> None:
        with self._lock:
            self._events.append(event)
        if self.event_logger:
            self.event_logger.log_event(
                event_type=f"tamper.{event.source.value}",
                severity=event.severity.value,
                source="tamper_detector",
                message=event.description,
                details=event.details,
            )
        logger.warning("tamper event: %s (%s)", event.description, event.source.value)

    def manual_tamper(self, description: str, severity: Severity = Severity.HIGH) -> TamperEvent:
        """Allow an operator or other module to register a tamper event."""
        event = TamperEvent(
            timestamp=format_timestamp(),
            source=TamperSource.MANUAL,
            severity=severity,
            description=description,
        )
        self.register_tamper_event(event)
        return event

    # ------------------------------------------------------------------ #
    # Polling helper
    # ------------------------------------------------------------------ #

    def poll_all(self) -> List[TamperEvent]:
        """Run all three checks; return any events triggered."""
        events: List[TamperEvent] = []
        for check in (self.check_intrusion_switches, self.check_accel_anomaly, self.check_tpm_counters):
            ev = check()
            if ev is not None:
                events.append(ev)
        return events

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_events(self, since: Optional[float] = None) -> List[TamperEvent]:
        with self._lock:
            if since is None:
                return list(self._events)
            # since is interpreted as POSIX seconds
            return [
                e for e in self._events
                if _parse_iso(e.timestamp) >= since
            ]

    def get_event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear_events(self) -> int:
        with self._lock:
            count = len(self._events)
            self._events.clear()
            return count

    def is_armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        self._armed = True
        # Re-snapshot baseline so we don't double-fire on existing states
        self._last_gpio = self.gpio_reader()
        self._last_tpm_counters = self.tpm_counter_reader()

    def disarm(self) -> None:
        self._armed = False


def _parse_iso(text: str) -> float:
    from datetime import datetime
    cleaned = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        return 0.0
