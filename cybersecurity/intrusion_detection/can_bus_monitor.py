"""CAN bus monitor for in-vehicle intrusion detection.

Wraps the :mod:`can` (python-can) library to sniff CAN frames and applies a
suite of detectors that flag:

* **Unauthorized CAN IDs** — IDs not on the allow-list.
* **Message injection** — abnormal message rates per CAN ID.
* **Replay attacks** — repeated identical payloads at suspicious intervals.
* **Fuzzing attacks** — rapidly changing payload bytes per ID.
* **Bus flooding / DoS** — aggregate rate exceeding a safe threshold.
* **Diagnostic abuse** — unsolicited UDS / OBD-II messages outside a service bay.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

try:  # python-can is the canonical backend
    import can  # type: ignore
    _HAS_CAN = True
except ImportError:  # pragma: no cover - python-can optional
    can = None  # type: ignore
    _HAS_CAN = False

from .constants import (
    DEFAULT_CAN_INTERFACE,
    DEFAULT_CAPTURE_INTERVAL_SEC,
    MAX_CAN_FRAME_RATE_PER_ID,
    AlertSeverity,
    ThreatType,
)
from .utils import CANFrame, parse_can_frame, timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CANFrameEvent:
    """Normalized representation of an observed CAN frame."""

    arbitration_id: int
    is_extended: bool
    is_remote: bool
    dlc: int
    data: bytes
    timestamp: float
    interface: str

    @property
    def id_hex(self) -> str:
        width = 8 if self.is_extended else 3
        return f"0x{self.arbitration_id:0{width}X}"


@dataclass
class CANAlert:
    """An alert raised by the CAN bus monitor."""

    timestamp: float
    alert_type: str
    severity: AlertSeverity
    threat_type: ThreatType
    can_id: Optional[int]
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _IDState:
    """Per-CAN-ID tracking state."""

    can_id: int
    count: int = 0
    last_seen: float = 0.0
    rate_window: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    payloads: Deque[Tuple[float, bytes]] = field(default_factory=lambda: deque(maxlen=100))
    inter_arrivals: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, payload: bytes) -> None:
        now = timestamp_now()
        if self.last_seen:
            self.inter_arrivals.append(now - self.last_seen)
        self.last_seen = now
        self.count += 1
        self.rate_window.append(now)
        self.payloads.append((now, payload))


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class CANBusMonitor:
    """Thread-safe CAN bus sniffer and intrusion detector.

    Parameters
    ----------
    interface:
        python-can bus type (e.g. ``"socketcan"``).
    channel:
        Bus channel (e.g. ``"can0"``).
    allowed_ids:
        Allow-list of arbitration IDs considered legitimate. Frames with IDs
        outside this set (when non-empty) raise ``unauthorized_id`` alerts.
    per_id_rate_limit:
        Maximum frames per second per arbitration ID before ``injection`` alert.
    global_rate_limit:
        Maximum frames per second on the bus before ``bus_flooding`` alert.
    """

    def __init__(
        self,
        interface: str = "socketcan",
        channel: str = DEFAULT_CAN_INTERFACE,
        allowed_ids: Optional[Set[int]] = None,
        per_id_rate_limit: float = float(MAX_CAN_FRAME_RATE_PER_ID),
        global_rate_limit: float = 20_000.0,
        capture_interval: float = DEFAULT_CAPTURE_INTERVAL_SEC,
        diagnostic_window: bool = False,
    ) -> None:
        self.interface = interface
        self.channel = channel
        self.allowed_ids: Set[int] = set(allowed_ids) if allowed_ids else set()
        self.per_id_rate_limit = float(per_id_rate_limit)
        self.global_rate_limit = float(global_rate_limit)
        self.capture_interval = float(capture_interval)
        self.diagnostic_window = bool(diagnostic_window)
        self._bus: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[CANFrameEvent], None]] = []
        self._alert_callbacks: List[Callable[[CANAlert], None]] = []
        self._id_states: Dict[int, _IDState] = {}
        self._global_times: Deque[float] = deque(maxlen=100_000)
        self._alerts: Deque[CANAlert] = deque(maxlen=10_000)
        self._stats = {"frames_processed": 0, "alerts_raised": 0}
        self._fuzz_threshold = 0.7  # fraction of bytes changing between frames

    # ------------------------------------------------------------------
    # Allow-list management
    # ------------------------------------------------------------------

    def add_allowed_id(self, can_id: int) -> None:
        """Add ``can_id`` to the allow-list."""
        with self._lock:
            self.allowed_ids.add(int(can_id))

    def remove_allowed_id(self, can_id: int) -> None:
        with self._lock:
            self.allowed_ids.discard(int(can_id))

    def get_allowed_ids(self) -> Set[int]:
        with self._lock:
            return set(self.allowed_ids)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_callback(self, cb: Callable[[CANFrameEvent], None]) -> None:
        """Register a frame callback invoked for every observed frame."""
        with self._lock:
            self._callbacks.append(cb)

    def register_alert_callback(self, cb: Callable[[CANAlert], None]) -> None:
        """Register an alert callback invoked whenever an alert is raised."""
        with self._lock:
            self._alert_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_capture(self) -> None:
        """Start the background capture thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("CAN capture already running")
            return
        self._stop_event.clear()
        if _HAS_CAN:
            try:
                self._bus = can.Bus(interface=self.interface, channel=self.channel)
                logger.info("Opened CAN bus %s:%s", self.interface, self.channel)
            except Exception as exc:
                logger.error("Failed to open CAN bus %s:%s: %s", self.interface, self.channel, exc)
                self._bus = None
        else:
            logger.warning("python-can not installed; running in dry/test mode")
        self._thread = threading.Thread(
            target=self._capture_loop, name="can-monitor", daemon=True
        )
        self._thread.start()

    def stop_capture(self) -> None:
        """Stop the background capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
        logger.info("CAN capture stopped")

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._bus is None:
                # In dry mode, just sleep to avoid busy loop.
                self._stop_event.wait(self.capture_interval)
                continue
            try:
                msg = self._bus.recv(timeout=self.capture_interval)
            except Exception as exc:
                logger.error("CAN recv error: %s", exc)
                self._stop_event.wait(self.capture_interval)
                continue
            if msg is None:
                continue
            frame = self._normalize(msg)
            self.process_frame(frame)
        logger.debug("CAN capture loop exiting")

    def _normalize(self, msg: Any) -> CANFrameEvent:
        return CANFrameEvent(
            arbitration_id=int(getattr(msg, "arbitration_id", 0)),
            is_extended=bool(getattr(msg, "is_extended_id", False)),
            is_remote=bool(getattr(msg, "is_rx", False)) and bool(getattr(msg, "is_remote_frame", False)),
            dlc=int(getattr(msg, "dlc", len(getattr(msg, "data", b"")))),
            data=bytes(getattr(msg, "data", b"")),
            timestamp=float(getattr(msg, "timestamp", timestamp_now())),
            interface=self.channel,
        )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: CANFrameEvent) -> None:
        """Process a single frame through the detection pipeline."""
        with self._lock:
            self._stats["frames_processed"] += 1
            self._global_times.append(frame.timestamp)
            state = self._id_states.get(frame.arbitration_id)
            if state is None:
                state = _IDState(can_id=frame.arbitration_id)
                self._id_states[frame.arbitration_id] = state
            state.record(frame.data)
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(frame)
            except Exception as exc:
                logger.error("Frame callback raised: %s", exc)
        self.detect_anomalies(frame, state)

    def ingest_raw(self, raw: bytes) -> Optional[CANFrameEvent]:
        """Parse a raw SocketCAN frame and feed it through the pipeline.

        Useful for testing or when reading from a log file.
        """
        parsed = parse_can_frame(raw, interface=self.channel)
        if parsed is None:
            return None
        event = CANFrameEvent(
            arbitration_id=parsed.arbitration_id,
            is_extended=parsed.is_extended,
            is_remote=parsed.is_remote,
            dlc=parsed.dlc,
            data=parsed.data,
            timestamp=parsed.timestamp,
            interface=parsed.interface,
        )
        self.process_frame(event)
        return event

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_anomalies(self, frame: CANFrameEvent, state: _IDState) -> List[CANAlert]:
        """Run all detectors for the given frame; return any alerts raised."""
        alerts: List[CANAlert] = []
        # 1. Unauthorized CAN ID
        if self.allowed_ids and frame.arbitration_id not in self.allowed_ids:
            alerts.append(CANAlert(
                timestamp=frame.timestamp,
                alert_type="unauthorized_can_id",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.CAN_INJECTION,
                can_id=frame.arbitration_id,
                description=f"CAN ID {frame.id_hex} not in allow-list",
                evidence={"allow_list_size": len(self.allowed_ids)},
            ))
        # 2. Per-ID rate (injection)
        rate = self._rate(state.rate_window, frame.timestamp)
        if rate > self.per_id_rate_limit:
            alerts.append(CANAlert(
                timestamp=frame.timestamp,
                alert_type="message_injection",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.CAN_INJECTION,
                can_id=frame.arbitration_id,
                description=f"CAN ID {frame.id_hex} rate {rate:.1f}/s exceeds {self.per_id_rate_limit}/s",
                evidence={"rate": rate, "limit": self.per_id_rate_limit},
            ))
        # 3. Global rate (bus flooding / DoS)
        global_rate = self._rate(self._global_times, frame.timestamp, window=1.0)
        if global_rate > self.global_rate_limit:
            alerts.append(CANAlert(
                timestamp=frame.timestamp,
                alert_type="bus_flooding",
                severity=AlertSeverity.CRITICAL,
                threat_type=ThreatType.DOS,
                can_id=None,
                description=f"Bus-wide rate {global_rate:.1f}/s exceeds {self.global_rate_limit}/s",
                evidence={"rate": global_rate, "limit": self.global_rate_limit},
            ))
        # 4. Replay detection (same payload repeatedly at fixed intervals)
        if len(state.payloads) >= 4:
            alerts.extend(self._detect_replay(frame, state))
        # 5. Fuzzing (rapid payload variation)
        if len(state.payloads) >= 3:
            alerts.extend(self._detect_fuzzing(frame, state))
        # 6. Diagnostic abuse
        if 0x7DF <= frame.arbitration_id <= 0x7EF and not self.diagnostic_window:
            alerts.append(CANAlert(
                timestamp=frame.timestamp,
                alert_type="unsolicited_diagnostic",
                severity=AlertSeverity.MEDIUM,
                threat_type=ThreatType.INTRUSION,
                can_id=frame.arbitration_id,
                description=f"UDS/OBD-II message {frame.id_hex} outside diagnostic window",
                evidence={"dlc": frame.dlc, "data_hex": frame.data.hex()},
            ))
        for a in alerts:
            self._raise_alert(a)
        return alerts

    def _detect_replay(self, frame: CANFrameEvent, state: _IDState) -> List[CANAlert]:
        alerts: List[CANAlert] = []
        recent = list(state.payloads)[-4:]
        payloads = [p for _, p in recent]
        if len(set(payloads)) == 1 and len(payloads) >= 3:
            intervals = [recent[i + 1][0] - recent[i][0] for i in range(len(recent) - 1)]
            if intervals and statistics.pstdev(intervals) < 0.05:
                alerts.append(CANAlert(
                    timestamp=frame.timestamp,
                    alert_type="replay_attack",
                    severity=AlertSeverity.HIGH,
                    threat_type=ThreatType.CAN_INJECTION,
                    can_id=frame.arbitration_id,
                    description=f"Repeating payload on {frame.id_hex} with low interval jitter",
                    evidence={"interval_mean": statistics.mean(intervals),
                              "interval_stdev": statistics.pstdev(intervals),
                              "payload_hex": payloads[0].hex()},
                ))
        return alerts

    def _detect_fuzzing(self, frame: CANFrameEvent, state: _IDState) -> List[CANAlert]:
        alerts: List[CANAlert] = []
        recent = list(state.payloads)[-3:]
        if len(recent) < 3:
            return alerts
        a, b, c = recent[-3][1], recent[-2][1], recent[-1][1]
        if not (len(a) == len(b) == len(c)):
            return alerts
        diff = sum(
            self._hamming_ratio(a, b),
            self._hamming_ratio(b, c),
            self._hamming_ratio(a, c),
        ) / 3.0
        if diff >= self._fuzz_threshold:
            alerts.append(CANAlert(
                timestamp=frame.timestamp,
                alert_type="fuzzing_attack",
                severity=AlertSeverity.MEDIUM,
                threat_type=ThreatType.CAN_INJECTION,
                can_id=frame.arbitration_id,
                description=f"Rapid payload variation on {frame.id_hex}",
                evidence={"change_ratio": diff, "threshold": self._fuzz_threshold},
            ))
        return alerts

    @staticmethod
    def _hamming_ratio(a: bytes, b: bytes) -> float:
        if not a:
            return 0.0
        diff = sum(1 for x, y in zip(a, b) if x != y)
        return diff / len(a)

    def _rate(self, timestamps: Deque[float], now: float, window: float = 1.0) -> float:
        """Compute rate (events/sec) within the last ``window`` seconds."""
        if not timestamps:
            return 0.0
        cutoff = now - window
        # Count entries after cutoff (timestamps are roughly sorted).
        count = 0
        for t in reversed(timestamps):
            if t >= cutoff:
                count += 1
            else:
                break
        return count / window if window > 0 else float(count)

    # ------------------------------------------------------------------
    # Alerts & stats
    # ------------------------------------------------------------------

    def _raise_alert(self, alert: CANAlert) -> None:
        with self._lock:
            self._alerts.append(alert)
            self._stats["alerts_raised"] += 1
            cbs = list(self._alert_callbacks)
        for cb in cbs:
            try:
                cb(alert)
            except Exception as exc:
                logger.error("Alert callback raised: %s", exc)

    def get_alerts(self, limit: int = 100) -> List[CANAlert]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "unique_ids": len(self._id_states),
                "allowed_ids": len(self.allowed_ids),
                "alerts_in_buffer": len(self._alerts),
            }


__all__ = ["CANBusMonitor", "CANFrameEvent", "CANAlert"]
