"""Real-time vehicle security monitor.

The :class:`VehicleMonitor` runs a background thread that periodically samples
the state of every ECU on the CAN bus, counts CAN bus traffic, lists network
connections and running processes, and re-checks file integrity baselines.
Detected anomalies are emitted as :class:`MonitorEvent` instances via
registered callbacks.

Hardware access (CAN sockets, ``/proc``, ``/sys``) is abstracted behind
``reader`` callables that can be replaced with mocks in tests.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .constants import ECU_IDS, EventType, Severity
from .utils import format_timestamp, now_monotonic

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


class ECUState(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    RESTARTING = "restarting"


@dataclass
class ECUStatus:
    name: str
    ecu_id: int
    state: ECUState
    last_seen: float
    firmware_version: str = ""
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class CANBusStats:
    received_frames: int = 0
    transmitted_frames: int = 0
    error_frames: int = 0
    bus_load_percent: float = 0.0
    drops: int = 0


@dataclass
class NetworkConnection:
    proto: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    state: str


@dataclass
class MonitorEvent:
    """An event emitted by the monitor."""

    timestamp: str
    event_type: EventType
    severity: Severity
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class MonitorState:
    """Snapshot of the monitor's current state."""

    running: bool = False
    last_poll: float = 0.0
    poll_count: int = 0
    ecu_status: Dict[str, ECUStatus] = field(default_factory=dict)
    can_stats: CANBusStats = field(default_factory=CANBusStats)
    connections: List[NetworkConnection] = field(default_factory=list)
    process_count: int = 0
    file_integrity_violations: List[str] = field(default_factory=list)
    events_emitted: int = 0


# --------------------------------------------------------------------------- #
# Readers (hardware abstraction)
# --------------------------------------------------------------------------- #

ECUReader = Callable[[], List[ECUStatus]]
CANStatsReader = Callable[[], CANBusStats]
NetworkReader = Callable[[], List[NetworkConnection]]
ProcessReader = Callable[[], int]
IntegrityReader = Callable[[], List[str]]


def _default_ecu_reader() -> List[ECUStatus]:
    # In production this would query CAN via python-can or socketcan.
    now = time.time()
    return [
        ECUStatus(name=name, ecu_id=ecu_id, state=ECUState.ONLINE, last_seen=now)
        for name, ecu_id in ECU_IDS.items()
    ]


def _default_can_stats_reader() -> CANBusStats:
    return CANBusStats()


def _default_network_reader() -> List[NetworkConnection]:
    return []


def _default_process_reader() -> int:
    return 0


def _default_integrity_reader() -> List[str]:
    return []


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #


class VehicleMonitor:
    """Background-threaded real-time security monitor."""

    def __init__(
        self,
        poll_interval: float = 0.5,
        max_events_per_second: int = 1000,
        ecu_reader: Optional[ECUReader] = None,
        can_stats_reader: Optional[CANStatsReader] = None,
        network_reader: Optional[NetworkReader] = None,
        process_reader: Optional[ProcessReader] = None,
        integrity_reader: Optional[IntegrityReader] = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.max_events_per_second = max_events_per_second
        self.state = MonitorState()
        self._state_lock = threading.RLock()

        self._ecu_reader = ecu_reader or _default_ecu_reader
        self._can_stats_reader = can_stats_reader or _default_can_stats_reader
        self._network_reader = network_reader or _default_network_reader
        self._process_reader = process_reader or _default_process_reader
        self._integrity_reader = integrity_reader or _default_integrity_reader

        self._callbacks: List[Callable[[MonitorEvent], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._event_queue: "queue.Queue[MonitorEvent]" = queue.Queue(maxsize=10000)
        self._dispatch_thread: Optional[threading.Thread] = None

        # rate limiter for events
        self._event_times: List[float] = []

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def add_callback(self, cb: Callable[[MonitorEvent], None]) -> None:
        self._callbacks.append(cb)

    def remove_callback(self, cb: Callable[[MonitorEvent], None]) -> None:
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self.state.running:
            logger.warning("VehicleMonitor already running")
            return
        self._stop_event.clear()
        self.state.running = True
        self._thread = threading.Thread(target=self._run_loop, name="vehicle-monitor", daemon=True)
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="vehicle-monitor-dispatch", daemon=True)
        self._thread.start()
        self._dispatch_thread.start()
        logger.info("VehicleMonitor started (poll=%.2fs)", self.poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        if not self.state.running:
            return
        self._stop_event.set()
        self.state.running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=timeout)
        logger.info("VehicleMonitor stopped")

    # ------------------------------------------------------------------ #
    # Event emission
    # ------------------------------------------------------------------ #

    def _emit(self, event: MonitorEvent) -> None:
        # rate limit
        now = now_monotonic()
        self._event_times = [t for t in self._event_times if now - t < 1.0]
        if len(self._event_times) >= self.max_events_per_second:
            logger.warning("event rate limit reached; dropping %s", event.event_type)
            return
        self._event_times.append(now)
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.error("event queue full; dropping %s", event.event_type)

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set() or not self._event_queue.empty():
            try:
                event = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._state_lock:
                self.state.events_emitted += 1
            for cb in list(self._callbacks):
                try:
                    cb(event)
                except Exception:  # noqa: BLE001 - callback must not kill dispatch
                    logger.exception("monitor callback raised")

    # ------------------------------------------------------------------ #
    # Main poll loop
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001
                logger.exception("monitor poll failed")
            self._stop_event.wait(self.poll_interval)

    def _poll_once(self) -> None:
        now = time.time()
        with self._state_lock:
            self.state.last_poll = now
            self.state.poll_count += 1

        # ECU status
        new_status = {s.name: s for s in self._ecu_reader()}
        self._check_ecu_state_changes(new_status)
        with self._state_lock:
            self.state.ecu_status = new_status

        # CAN bus stats
        new_can = self._can_stats_reader()
        self._check_can_anomaly(new_can)
        with self._state_lock:
            self.state.can_stats = new_can

        # Network connections
        conns = self._network_reader()
        with self._state_lock:
            self.state.connections = conns

        # Process count
        procs = self._process_reader()
        with self._state_lock:
            self.state.process_count = procs

        # File integrity
        violations = self._integrity_reader()
        if violations:
            with self._state_lock:
                self.state.file_integrity_violations = violations
            self._emit(MonitorEvent(
                timestamp=format_timestamp(now),
                event_type=EventType.INTEGRITY_VIOLATION,
                severity=Severity.HIGH,
                source="vehicle_monitor",
                message="file integrity violations detected",
                details={"files": violations},
            ))

    # ------------------------------------------------------------------ #
    # Anomaly detection
    # ------------------------------------------------------------------ #

    def _check_ecu_state_changes(self, new_status: Dict[str, ECUStatus]) -> None:
        with self._state_lock:
            old = self.state.ecu_status
        for name, status in new_status.items():
            prev = old.get(name)
            if prev is None:
                continue
            if prev.state == ECUState.ONLINE and status.state != ECUState.ONLINE:
                self._emit(MonitorEvent(
                    timestamp=format_timestamp(),
                    event_type=EventType.ECU_OFFLINE,
                    severity=Severity.HIGH if name in {"BRAKES", "STEERING"} else Severity.MEDIUM,
                    source=name,
                    message=f"ECU {name} went {status.state.value}",
                    details=status.to_dict(),
                ))
            elif status.state == ECUState.RESTARTING:
                self._emit(MonitorEvent(
                    timestamp=format_timestamp(),
                    event_type=EventType.ECU_RESTART,
                    severity=Severity.MEDIUM,
                    source=name,
                    message=f"ECU {name} restarting",
                    details=status.to_dict(),
                ))

    def _check_can_anomaly(self, stats: CANBusStats) -> None:
        if stats.error_frames > 0:
            self._emit(MonitorEvent(
                timestamp=format_timestamp(),
                event_type=EventType.CAN_BUS_ANOMALY,
                severity=Severity.MEDIUM,
                source="can_bus",
                message=f"{stats.error_frames} CAN error frames detected",
                details={"error_frames": stats.error_frames, "drops": stats.drops},
            ))
        if stats.bus_load_percent > 80.0:
            self._emit(MonitorEvent(
                timestamp=format_timestamp(),
                event_type=EventType.CAN_BUS_ANOMALY,
                severity=Severity.MEDIUM,
                source="can_bus",
                message=f"high CAN bus load: {stats.bus_load_percent:.1f}%",
                details={"bus_load_percent": stats.bus_load_percent},
            ))

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            snap = MonitorState(
                running=self.state.running,
                last_poll=self.state.last_poll,
                poll_count=self.state.poll_count,
                ecu_status=dict(self.state.ecu_status),
                can_stats=self.state.can_stats,
                connections=list(self.state.connections),
                process_count=self.state.process_count,
                file_integrity_violations=list(self.state.file_integrity_violations),
                events_emitted=self.state.events_emitted,
            )
        return {
            "running": snap.running,
            "last_poll": format_timestamp(snap.last_poll) if snap.last_poll else None,
            "poll_count": snap.poll_count,
            "ecus": [s.to_dict() for s in snap.ecu_status.values()],
            "can_stats": asdict(snap.can_stats),
            "connections": [asdict(c) for c in snap.connections],
            "process_count": snap.process_count,
            "file_integrity_violations": snap.file_integrity_violations,
            "events_emitted": snap.events_emitted,
        }
