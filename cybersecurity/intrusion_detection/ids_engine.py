"""IDS engine: core event pipeline (ingest → preprocess → rules → anomalies).

The :class:`IDSEngine` is the heart of the intrusion detection subsystem. It
runs a background worker thread that pulls events from an internal queue,
applies preprocessing, runs each registered detector's rule matcher, scores
anomalies, and emits alerts via a callback.

Designed to be detector-agnostic: detectors (CAN bus monitor, network
monitor, log analyzer, packet analyzer, malware detector, ...) are registered
via :meth:`register_detector` and contribute both events and matchers.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol

from .anomaly_detection import AnomalyDetector
from .attack_signature import AttackSignatureDB
from .constants import (
    DEFAULT_CAPTURE_INTERVAL_SEC,
    EventType,
    MAX_EVENT_RATE_PER_SEC,
    AlertSeverity,
    ThreatType,
)
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event data structure
# ---------------------------------------------------------------------------


@dataclass
class IDSEvent:
    """A single event flowing through the IDS pipeline."""

    event_id: str
    event_type: EventType
    source: str  # detector / producer name
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------


class DetectorProtocol(Protocol):
    """Structural type implemented by all registered detectors."""

    name: str

    def get_statistics(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Pipeline stage stats
# ---------------------------------------------------------------------------


@dataclass
class PipelineStats:
    """Per-stage counters for the event pipeline."""

    ingested: int = 0
    preprocessed: int = 0
    rule_matches: int = 0
    anomaly_emissions: int = 0
    alerts_emitted: int = 0
    errors: int = 0
    dropped: int = 0
    queue_depth: int = 0
    throughput_per_sec: float = 0.0


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass
class IDSRule:
    """A rule evaluated against each event."""

    id: str
    name: str
    event_types: List[EventType]  # empty means "any"
    matcher: Callable[[IDSEvent], bool]
    severity: AlertSeverity = AlertSeverity.MEDIUM
    threat_type: ThreatType = ThreatType.UNKNOWN
    description: str = ""

    def applies(self, event: IDSEvent) -> bool:
        if self.event_types and event.event_type not in self.event_types:
            return False
        try:
            return bool(self.matcher(event))
        except Exception as exc:
            logger.error("Rule %s matcher raised: %s", self.id, exc)
            return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IDSEngine:
    """Core IDS engine: event pipeline + rule matching + anomaly scoring."""

    def __init__(
        self,
        instance_id: str = "avcs-ids-default",
        max_queue_size: int = 100_000,
        worker_count: int = 1,
        batch_size: int = 200,
        enable_anomaly: bool = True,
    ) -> None:
        self.instance_id = instance_id
        self.max_queue_size = int(max_queue_size)
        self.worker_count = max(1, int(worker_count))
        self.batch_size = int(batch_size)
        self.enable_anomaly = bool(enable_anomaly)

        self._queue: "queue.Queue[Optional[IDSEvent]]" = queue.Queue(maxsize=self.max_queue_size)
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self._rules: Dict[str, IDSRule] = {}
        self._detectors: Dict[str, DetectorProtocol] = {}
        self._signature_db: Optional[AttackSignatureDB] = None
        self._anomaly: Optional[AnomalyDetector] = None
        if self.enable_anomaly:
            self._anomaly = AnomalyDetector()

        self._alert_callbacks: List[Callable[[IDSEvent, List[IDSRule], Dict[str, Any]], None]] = []
        self._event_callbacks: List[Callable[[IDSEvent], None]] = []
        self._stats = PipelineStats()
        self._event_history: Deque[IDSEvent] = deque(maxlen=10_000)
        self._alerts_history: Deque[Dict[str, Any]] = deque(maxlen=10_000)
        self._event_id_counter = 0
        self._throughput_window: Deque[float] = deque(maxlen=10_000)
        self._started_at = 0.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_signature_db(self, db: AttackSignatureDB) -> None:
        """Attach an attack-signature database for payload matching."""
        with self._lock:
            self._signature_db = db

    def set_anomaly_detector(self, detector: AnomalyDetector) -> None:
        with self._lock:
            self._anomaly = detector
            self.enable_anomaly = True

    # ------------------------------------------------------------------
    # Rule & detector registration
    # ------------------------------------------------------------------

    def register_rule(self, rule: IDSRule) -> bool:
        """Register a rule. Returns True on success."""
        if not rule.id:
            raise ValueError("IDSRule requires an id")
        with self._lock:
            self._rules[rule.id] = rule
            logger.debug("Registered rule %s (%s)", rule.id, rule.name)
            return True

    def unregister_rule(self, rule_id: str) -> bool:
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def list_rules(self, enabled_only: bool = False) -> List[IDSRule]:
        with self._lock:
            return list(self._rules.values())

    def register_detector(self, name: str, detector: DetectorProtocol) -> None:
        """Register a named detector for stats aggregation."""
        with self._lock:
            self._detectors[name] = detector
            logger.info("Registered detector '%s'", name)

    def unregister_detector(self, name: str) -> None:
        with self._lock:
            self._detectors.pop(name, None)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_alert_callback(
        self,
        cb: Callable[[IDSEvent, List[IDSRule], Dict[str, Any]], None],
    ) -> None:
        """Register a callback invoked when rules match an event.

        Signature: ``cb(event, matched_rules, anomaly_info)``
        """
        with self._lock:
            self._alert_callbacks.append(cb)

    def register_event_callback(self, cb: Callable[[IDSEvent], None]) -> None:
        """Register a callback invoked for every preprocessed event."""
        with self._lock:
            self._event_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._workers:
            logger.warning("IDS engine already running")
            return
        self._stop_event.clear()
        self._started_at = timestamp_now()
        for i in range(self.worker_count):
            t = threading.Thread(
                target=self._worker_loop, name=f"ids-worker-{i}", daemon=True
            )
            t.start()
            self._workers.append(t)
        logger.info("IDS engine started (%d workers)", self.worker_count)

    def stop(self) -> None:
        self._stop_event.set()
        # Drain the queue by sending sentinel values
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()
        logger.info("IDS engine stopped")

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def ingest_event(
        self,
        event_type: EventType,
        source: str,
        payload: Dict[str, Any],
        raw: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[IDSEvent]:
        """Push an event onto the processing queue.

        Returns the constructed :class:`IDSEvent` (with assigned id), or
        ``None`` if the queue was full and the event was dropped.
        """
        with self._lock:
            self._event_id_counter += 1
            eid = f"evt-{self._event_id_counter:010d}"
        event = IDSEvent(
            event_id=eid,
            event_type=event_type,
            source=source,
            timestamp=timestamp or timestamp_now(),
            payload=dict(payload),
            raw=raw,
            metadata=dict(metadata or {}),
        )
        try:
            self._queue.put_nowait(event)
            return event
        except queue.Full:
            with self._lock:
                self._stats.dropped += 1
            logger.warning("IDS event queue full; dropping event %s", eid)
            return None

    def ingest_event_object(self, event: IDSEvent) -> bool:
        """Push a pre-constructed :class:`IDSEvent` onto the queue."""
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            with self._lock:
                self._stats.dropped += 1
            return False

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        batch: List[IDSEvent] = []
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=DEFAULT_CAPTURE_INTERVAL_SEC)
            except queue.Empty:
                continue
            if event is None:
                # Sentinel
                self._queue.task_done()
                break
            batch.append(event)
            if len(batch) >= self.batch_size:
                self._process_batch(batch)
                batch.clear()
            self._queue.task_done()
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[IDSEvent]) -> None:
        for event in batch:
            try:
                self._process_one(event)
            except Exception as exc:
                with self._lock:
                    self._stats.errors += 1
                logger.exception("Error processing event %s: %s", event.event_id, exc)

    def _process_one(self, event: IDSEvent) -> None:
        # 1. Ingestion counter
        with self._lock:
            self._stats.ingested += 1
            self._event_history.append(event)
            self._throughput_window.append(event.timestamp)
            event_cbs = list(self._event_callbacks)
            rules = list(self._rules.values())
        # 2. Preprocess (no-op default; subclasses can override)
        self._preprocess(event)
        with self._lock:
            self._stats.preprocessed += 1
        for cb in event_cbs:
            try:
                cb(event)
            except Exception as exc:
                logger.error("Event callback raised: %s", exc)
        # 3. Rule matching
        matched = [r for r in rules if r.applies(event)]
        if matched:
            with self._lock:
                self._stats.rule_matches += len(matched)
        # 4. Signature DB matching (if available) — applied to payload only
        if self._signature_db is not None:
            payload_str = str(event.payload)
            sig_matches = self._signature_db.match(payload_str)
            if sig_matches:
                logger.debug("Event %s matched %d signatures", event.event_id, len(sig_matches))
        # 5. Anomaly scoring
        anomaly_info: Dict[str, Any] = {}
        if self.enable_anomaly and self._anomaly is not None:
            anomaly_info = self._score_anomaly(event)
            with self._lock:
                if anomaly_info.get("is_anomalous"):
                    self._stats.anomaly_emissions += 1
        # 6. Emit alerts if any matches or anomalies
        if matched or anomaly_info.get("is_anomalous"):
            with self._lock:
                self._stats.alerts_emitted += 1
                self._alerts_history.append({
                    "event_id": event.event_id,
                    "rules": [r.id for r in matched],
                    "anomaly": anomaly_info,
                    "timestamp": event.timestamp,
                })
                alert_cbs = list(self._alert_callbacks)
            for cb in alert_cbs:
                try:
                    cb(event, matched, anomaly_info)
                except Exception as exc:
                    logger.error("Alert callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Preprocessing & anomaly scoring
    # ------------------------------------------------------------------

    def _preprocess(self, event: IDSEvent) -> None:
        """Normalize event fields. Default implementation is a no-op; override
        in subclasses for domain-specific normalization."""
        # Add an "hour_of_day" feature useful for time-based anomaly detection
        ts = event.timestamp
        event.metadata.setdefault(
            "hour_of_day",
            int(time.gmtime(ts).tm_hour + time.gmtime(ts).tm_min / 60.0),
        )

    def _score_anomaly(self, event: IDSEvent) -> Dict[str, Any]:
        """Feed event metrics into the anomaly detector and return a summary."""
        if self._anomaly is None:
            return {}
        # Map a few common payload fields into metrics
        metrics: Dict[str, float] = {}
        for key in ("rate", "byte_count", "payload_len", "ttl", "dlc", "entropy"):
            if key in event.payload:
                try:
                    metrics[key] = float(event.payload[key])
                except (TypeError, ValueError):
                    pass
        results = {}
        is_anomalous = False
        for name, value in metrics.items():
            self._anomaly.add_observation(name, value)
            res = self._anomaly.detect_anomaly(name, value)
            if res.is_anomalous:
                is_anomalous = True
            results[name] = {
                "value": res.value,
                "score": res.score,
                "method": res.method,
                "is_anomalous": res.is_anomalous,
            }
        return {"is_anomalous": is_anomalous, "metrics": results}

    # ------------------------------------------------------------------
    # Stats & queries
    # ------------------------------------------------------------------

    def get_event_pipeline_stats(self) -> Dict[str, Any]:
        """Return detailed per-stage pipeline statistics."""
        with self._lock:
            now = timestamp_now()
            # Compute throughput over the last 5s window
            cutoff = now - 5.0
            recent = [t for t in self._throughput_window if t >= cutoff]
            throughput = len(recent) / 5.0 if recent else 0.0
            stats = PipelineStats(
                ingested=self._stats.ingested,
                preprocessed=self._stats.preprocessed,
                rule_matches=self._stats.rule_matches,
                anomaly_emissions=self._stats.anomaly_emissions,
                alerts_emitted=self._stats.alerts_emitted,
                errors=self._stats.errors,
                dropped=self._stats.dropped,
                queue_depth=self._queue.qsize(),
                throughput_per_sec=throughput,
            )
            detector_stats = {name: det.get_statistics() for name, det in self._detectors.items()}
            return {
                "instance_id": self.instance_id,
                "uptime_sec": now - self._started_at if self._started_at else 0.0,
                "pipeline": stats.__dict__,
                "detectors": detector_stats,
                "rules_count": len(self._rules),
                "workers": len(self._workers),
            }

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts_history)[-limit:]

    def get_recent_events(self, limit: int = 100) -> List[IDSEvent]:
        with self._lock:
            return list(self._event_history)[-limit:]


__all__ = ["IDSEngine", "IDSEvent", "IDSRule", "PipelineStats", "DetectorProtocol"]
