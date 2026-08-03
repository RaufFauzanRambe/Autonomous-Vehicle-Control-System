"""Real-time monitor: aggregates IDS state into WebSocket-friendly snapshots.

The :class:`RealtimeMonitor` periodically polls the various IDS components
(alert manager, detectors, response engine, incident logger) and publishes a
compact JSON snapshot to subscribers (e.g. a websocket server, a SIEM
forwarder, or an in-process dashboard).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from .constants import AlertSeverity, AlertStatus, DEFAULT_STATS_INTERVAL_SEC
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MonitorSnapshot:
    """A single snapshot of the IDS state, suitable for JSON serialization."""

    timestamp: float
    instance_id: str
    uptime_sec: float
    alerts_active: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)
    alerts_by_status: Dict[str, int] = field(default_factory=dict)
    alerts_recent: List[Dict[str, Any]] = field(default_factory=list)
    detector_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    response_stats: Dict[str, Any] = field(default_factory=dict)
    incident_stats: Dict[str, Any] = field(default_factory=dict)
    threat_classification_stats: Dict[str, Any] = field(default_factory=dict)
    event_throughput_per_sec: float = 0.0
    error_count: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "instance_id": self.instance_id,
            "uptime_sec": round(self.uptime_sec, 3),
            "alerts_active": self.alerts_active,
            "alerts_by_severity": dict(self.alerts_by_severity),
            "alerts_by_status": dict(self.alerts_by_status),
            "alerts_recent": list(self.alerts_recent),
            "detector_stats": dict(self.detector_stats),
            "response_stats": dict(self.response_stats),
            "incident_stats": dict(self.incident_stats),
            "threat_classification_stats": dict(self.threat_classification_stats),
            "event_throughput_per_sec": round(self.event_throughput_per_sec, 3),
            "error_count": self.error_count,
            "extras": dict(self.extras),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, sort_keys=True)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


SubscriberCallback = Callable[[MonitorSnapshot], None]


class RealtimeMonitor:
    """Background thread that aggregates IDS state and publishes snapshots.

    Parameters
    ----------
    instance_id:
        Identifier of the IDS instance, included in every snapshot.
    interval_sec:
        Snapshot publication interval.
    history_size:
        How many past snapshots to keep in memory.
    """

    def __init__(
        self,
        instance_id: str = "avcs-ids-default",
        interval_sec: float = DEFAULT_STATS_INTERVAL_SEC,
        history_size: int = 1440,
    ) -> None:
        self.instance_id = instance_id
        self.interval_sec = float(interval_sec)
        self.history_size = int(history_size)
        self._lock = threading.RLock()
        self._subscribers: List[SubscriberCallback] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._history: Deque[MonitorSnapshot] = deque(maxlen=self.history_size)
        self._sources: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._start_time = timestamp_now()
        self._error_count = 0
        self._events_counter = 0
        self._events_last_ts = timestamp_now()
        self._current: Optional[MonitorSnapshot] = None

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def register_source(self, name: str, getter: Callable[[], Dict[str, Any]]) -> None:
        """Register a callable that returns stats for a named component."""
        with self._lock:
            self._sources[name] = getter

    def unregister_source(self, name: str) -> None:
        with self._lock:
            self._sources.pop(name, None)

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def register_subscriber(self, cb: SubscriberCallback) -> None:
        """Register a callback that receives every published snapshot."""
        with self._lock:
            self._subscribers.append(cb)

    def unregister_subscriber(self, cb: SubscriberCallback) -> None:
        with self._lock:
            try:
                self._subscribers.remove(cb)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("Realtime monitor already running")
            return
        self._stop_event.clear()
        self._start_time = timestamp_now()
        self._thread = threading.Thread(
            target=self._loop, name="rt-monitor", daemon=True
        )
        self._thread.start()
        logger.info("Realtime monitor started (interval=%.1fs)", self.interval_sec)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Realtime monitor stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.publish_update()
            except Exception as exc:
                self._error_count += 1
                logger.error("Realtime monitor publish error: %s", exc)
            self._stop_event.wait(self.interval_sec)

    # ------------------------------------------------------------------
    # Snapshot production
    # ------------------------------------------------------------------

    def _gather_sources(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            sources = list(self._sources.items())
        for name, getter in sources:
            try:
                out[name] = getter() or {}
            except Exception as exc:
                self._error_count += 1
                logger.error("Source %s raised: %s", name, exc)
                out[name] = {"error": str(exc)}
        return out

    def record_event(self, count: int = 1) -> None:
        """Account for ``count`` processed events (used for throughput metrics)."""
        with self._lock:
            self._events_counter += int(count)

    def publish_update(self) -> MonitorSnapshot:
        """Build and publish a snapshot to all subscribers."""
        now = timestamp_now()
        uptime = now - self._start_time
        with self._lock:
            events = self._events_counter
            last_ts = self._events_last_ts
            self._events_counter = 0
            self._events_last_ts = now
        elapsed = max(now - last_ts, 1e-6)
        throughput = events / elapsed
        detector_stats = self._gather_sources()
        # Pull well-known keys out of detector stats
        alerts_stats = detector_stats.get("alert_manager", {})
        response_stats = detector_stats.get("response_engine", {})
        incident_stats = detector_stats.get("incident_logger", {})
        tc_stats = detector_stats.get("threat_classifier", {})
        snapshot = MonitorSnapshot(
            timestamp=now,
            instance_id=self.instance_id,
            uptime_sec=uptime,
            alerts_active=int(alerts_stats.get("active", alerts_stats.get("total", 0)) or 0),
            alerts_by_severity=alerts_stats.get("by_severity", {}) or {},
            alerts_by_status=alerts_stats.get("by_status", {}) or {},
            alerts_recent=alerts_stats.get("recent_alerts", []) or [],
            detector_stats=detector_stats,
            response_stats=response_stats,
            incident_stats=incident_stats,
            threat_classification_stats=tc_stats,
            event_throughput_per_sec=throughput,
            error_count=self._error_count,
        )
        with self._lock:
            self._history.append(snapshot)
            self._current = snapshot
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(snapshot)
            except Exception as exc:
                logger.error("Subscriber raised: %s", exc)
        return snapshot

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_current_state(self) -> Optional[MonitorSnapshot]:
        """Return the most recent snapshot, or None if none has been produced."""
        with self._lock:
            return self._current

    def get_history(self, limit: int = 100) -> List[MonitorSnapshot]:
        with self._lock:
            return list(self._history)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "instance_id": self.instance_id,
                "uptime_sec": timestamp_now() - self._start_time,
                "history_size": len(self._history),
                "subscribers": len(self._subscribers),
                "sources": len(self._sources),
                "error_count": self._error_count,
            }


__all__ = ["RealtimeMonitor", "MonitorSnapshot", "SubscriberCallback"]
