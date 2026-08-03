"""Alert manager: lifecycle, deduplication, correlation, escalation.

The :class:`AlertManager` is the central registry for all IDS alerts. It
provides:

* **Lifecycle management** — new → acknowledged → investigating → resolved /
  false_positive, with explicit transitions.
* **Deduplication** — alerts with the same fingerprint within a configurable
  window increment a counter rather than creating duplicates.
* **Correlation** — alerts within a time window sharing one or more keys
  (e.g. ``src_ip``) are grouped into a *correlation set*.
* **Escalation** — alerts that remain NEW for too long are auto-escalated by a
  background thread.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from .constants import (
    AlertSeverity,
    AlertStatus,
    DEFAULT_DEDUP_WINDOW_SEC,
    DEFAULT_ESCALATION_TIME_SEC,
    ThreatType,
)
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """A single alert managed by :class:`AlertManager`."""

    id: str
    fingerprint: str
    title: str
    description: str
    severity: AlertSeverity
    threat_type: ThreatType
    source: str  # detector / analyzer that raised it
    status: AlertStatus = AlertStatus.NEW
    created_at: float = field(default_factory=timestamp_now)
    updated_at: float = field(default_factory=timestamp_now)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    count: int = 1  # number of duplicates absorbed
    correlation_id: Optional[str] = None
    mitre_attack_ids: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.name,
            "threat_type": self.threat_type.value,
            "source": self.source,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "count": self.count,
            "correlation_id": self.correlation_id,
            "mitre_attack_ids": list(self.mitre_attack_ids),
            "evidence": dict(self.evidence),
            "tags": list(self.tags),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class AlertManager:
    """Thread-safe alert lifecycle manager."""

    VALID_TRANSITIONS: Dict[AlertStatus, Set[AlertStatus]] = {
        AlertStatus.NEW: {AlertStatus.ACKNOWLEDGED, AlertStatus.INVESTIGATING,
                          AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE,
                          AlertStatus.ESCALATED},
        AlertStatus.ACKNOWLEDGED: {AlertStatus.INVESTIGATING, AlertStatus.RESOLVED,
                                   AlertStatus.FALSE_POSITIVE, AlertStatus.ESCALATED},
        AlertStatus.INVESTIGATING: {AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE,
                                    AlertStatus.ESCALATED, AlertStatus.ACKNOWLEDGED},
        AlertStatus.ESCALATED: {AlertStatus.INVESTIGATING, AlertStatus.RESOLVED,
                                AlertStatus.FALSE_POSITIVE},
        AlertStatus.RESOLVED: set(),
        AlertStatus.FALSE_POSITIVE: set(),
    }

    def __init__(
        self,
        dedup_window_sec: float = DEFAULT_DEDUP_WINDOW_SEC,
        escalation_sec: int = DEFAULT_ESCALATION_TIME_SEC,
        enable_correlation: bool = True,
        correlation_window_sec: float = 120.0,
    ) -> None:
        self.dedup_window_sec = float(dedup_window_sec)
        self.escalation_sec = int(escalation_sec)
        self.enable_correlation = bool(enable_correlation)
        self.correlation_window_sec = float(correlation_window_sec)
        self._alerts: Dict[str, Alert] = {}
        self._fingerprints: Dict[str, Tuple[str, float]] = {}  # fp -> (alert_id, last_ts)
        self._correlations: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._state_callbacks: List[Callable[[Alert, AlertStatus, AlertStatus], None]] = []
        self._next_id = 1
        self._escalator_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats = {"raised": 0, "deduplicated": 0, "escalated": 0,
                       "resolved": 0, "false_positives": 0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background escalation thread."""
        if self._escalator_thread and self._escalator_thread.is_alive():
            return
        self._stop_event.clear()
        self._escalator_thread = threading.Thread(
            target=self._escalation_loop, name="alert-escalator", daemon=True
        )
        self._escalator_thread.start()

    def stop(self) -> None:
        """Stop the background escalation thread."""
        self._stop_event.set()
        if self._escalator_thread:
            self._escalator_thread.join(timeout=3.0)
            self._escalator_thread = None

    def _escalation_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._escalate_stale_alerts()
            except Exception as exc:
                logger.error("Escalation loop error: %s", exc)
            self._stop_event.wait(5.0)

    def _escalate_stale_alerts(self) -> None:
        now = timestamp_now()
        with self._lock:
            stale = [
                a for a in self._alerts.values()
                if a.status == AlertStatus.NEW
                and now - a.created_at >= self.escalation_sec
            ]
        for a in stale:
            self._transition(a, AlertStatus.ESCALATED)

    # ------------------------------------------------------------------
    # Raising alerts
    # ------------------------------------------------------------------

    def raise_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        threat_type: ThreatType,
        source: str,
        evidence: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        mitre_attack_ids: Optional[List[str]] = None,
    ) -> Alert:
        """Raise a new alert (or absorb into an existing duplicate)."""
        fingerprint = self._compute_fingerprint(
            title, threat_type, source, evidence or {}
        )
        now = timestamp_now()
        with self._lock:
            existing = self._fingerprints.get(fingerprint)
            if existing is not None:
                alert_id, last_ts = existing
                if now - last_ts <= self.dedup_window_sec:
                    alert = self._alerts[alert_id]
                    alert.count += 1
                    alert.updated_at = now
                    self._fingerprints[fingerprint] = (alert_id, now)
                    self._stats["deduplicated"] += 1
                    return alert
            alert_id = f"alert-{self._next_id:08d}"
            self._next_id += 1
            alert = Alert(
                id=alert_id, fingerprint=fingerprint, title=title,
                description=description, severity=severity,
                threat_type=threat_type, source=source,
                evidence=dict(evidence or {}),
                tags=list(tags or []),
                mitre_attack_ids=list(mitre_attack_ids or []),
            )
            self._alerts[alert_id] = alert
            self._fingerprints[fingerprint] = (alert_id, now)
            self._stats["raised"] += 1
            callbacks = list(self._alert_callbacks)
            if self.enable_correlation:
                self._correlate(alert)
        for cb in callbacks:
            try:
                cb(alert)
            except Exception as exc:
                logger.error("Alert callback raised: %s", exc)
        return alert

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: str, note: Optional[str] = None) -> bool:
        return self._transition_by_id(alert_id, AlertStatus.ACKNOWLEDGED, note)

    def resolve_alert(self, alert_id: str, note: Optional[str] = None) -> bool:
        ok = self._transition_by_id(alert_id, AlertStatus.RESOLVED, note)
        if ok:
            with self._lock:
                self._stats["resolved"] += 1
        return ok

    def mark_false_positive(self, alert_id: str, note: Optional[str] = None) -> bool:
        ok = self._transition_by_id(alert_id, AlertStatus.FALSE_POSITIVE, note)
        if ok:
            with self._lock:
                self._stats["false_positives"] += 1
        return ok

    def escalate_alert(self, alert_id: str, note: Optional[str] = None) -> bool:
        ok = self._transition_by_id(alert_id, AlertStatus.ESCALATED, note)
        if ok:
            with self._lock:
                self._stats["escalated"] += 1
        return ok

    def investigate_alert(self, alert_id: str, note: Optional[str] = None) -> bool:
        return self._transition_by_id(alert_id, AlertStatus.INVESTIGATING, note)

    def _transition_by_id(self, alert_id: str, new_state: AlertStatus,
                          note: Optional[str] = None) -> bool:
        with self._lock:
            alert = self._alerts.get(alert_id)
        if alert is None:
            logger.warning("Transition requested for unknown alert %s", alert_id)
            return False
        return self._transition(alert, new_state, note)

    def _transition(self, alert: Alert, new_state: AlertStatus,
                    note: Optional[str] = None) -> bool:
        with self._lock:
            old_state = alert.status
            if new_state == old_state:
                return True
            allowed = self.VALID_TRANSITIONS.get(old_state, set())
            if new_state not in allowed:
                logger.warning("Invalid transition %s -> %s for alert %s",
                               old_state.value, new_state.value, alert.id)
                return False
            alert.status = new_state
            alert.updated_at = timestamp_now()
            if new_state == AlertStatus.ACKNOWLEDGED and alert.acknowledged_at is None:
                alert.acknowledged_at = alert.updated_at
            if new_state in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE):
                alert.resolved_at = alert.updated_at
            if note:
                alert.notes.append(f"[{alert.updated_at:.0f}] {note}")
            callbacks = list(self._state_callbacks)
        for cb in callbacks:
            try:
                cb(alert, old_state, new_state)
            except Exception as exc:
                logger.error("State callback raised: %s", exc)
        return True

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def _correlate(self, alert: Alert) -> None:
        """Group alerts that share at least one correlation key within the window."""
        keys = []
        for field_name in ("src_ip", "dst_ip", "can_id", "process", "file"):
            val = alert.evidence.get(field_name)
            if val is not None:
                keys.append(f"{field_name}:{val}")
        if not keys:
            return
        now = alert.created_at
        for k in keys:
            cid = f"corr-{k}"
            related = self._correlations[cid]
            # Drop entries outside the window
            related[:] = [aid for aid in related
                          if aid in self._alerts
                          and now - self._alerts[aid].created_at <= self.correlation_window_sec]
            related.append(alert.id)
            if len(related) > 1:
                alert.correlation_id = cid

    def correlate_alerts(self, window_sec: Optional[float] = None) -> Dict[str, List[str]]:
        """Return all active correlation sets."""
        window = window_sec or self.correlation_window_sec
        now = timestamp_now()
        with self._lock:
            out: Dict[str, List[str]] = {}
            for cid, aids in self._correlations.items():
                recent = [
                    aid for aid in aids
                    if aid in self._alerts
                    and now - self._alerts[aid].created_at <= window
                ]
                if len(recent) >= 2:
                    out[cid] = recent
            return out

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        threat_type: Optional[ThreatType] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Return alerts that are not yet resolved/false_positive."""
        with self._lock:
            alerts = list(self._alerts.values())
        active = [
            a for a in alerts
            if a.status not in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE)
        ]
        if severity is not None:
            active = [a for a in active if a.severity == severity]
        if threat_type is not None:
            active = [a for a in active if a.threat_type == threat_type]
        if source is not None:
            active = [a for a in active if a.source == source]
        active.sort(key=lambda a: (int(a.severity), a.created_at), reverse=True)
        return active[:limit]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            return self._alerts.get(alert_id)

    def get_all_alerts(self, limit: int = 1000) -> List[Alert]:
        with self._lock:
            alerts = list(self._alerts.values())
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            by_status: Dict[str, int] = defaultdict(int)
            by_severity: Dict[str, int] = defaultdict(int)
            for a in self._alerts.values():
                by_status[a.status.value] += 1
                by_severity[a.severity.name] += 1
            return {
                **self._stats,
                "total": len(self._alerts),
                "by_status": dict(by_status),
                "by_severity": dict(by_severity),
                "active_correlations": sum(
                    1 for v in self._correlations.values() if len(v) >= 2
                ),
            }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_alert_callback(self, cb: Callable[[Alert], None]) -> None:
        with self._lock:
            self._alert_callbacks.append(cb)

    def register_state_callback(
        self, cb: Callable[[Alert, AlertStatus, AlertStatus], None]
    ) -> None:
        with self._lock:
            self._state_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fingerprint(
        title: str, threat_type: ThreatType, source: str,
        evidence: Dict[str, Any],
    ) -> str:
        """Compute a stable fingerprint used for deduplication."""
        # Use a small subset of evidence for stability (avoid volatile fields
        # like timestamps, counters, random nonces).
        stable_keys = ("can_id", "src_ip", "dst_ip", "dst_port", "rule_id",
                       "pattern_id", "signature", "hash", "yara_rule")
        stable = {k: evidence[k] for k in stable_keys if k in evidence}
        raw = f"{title}|{threat_type.value}|{source}|{json_stable(stable)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def json_stable(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True, default=str)


__all__ = ["AlertManager", "Alert"]
