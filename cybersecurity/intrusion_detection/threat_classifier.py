"""Threat classification: type + severity assignment for detected events.

The :class:`ThreatClassifier` combines a deterministic rule-based classifier
(with ATT&CK-aligned mappings) and an optional scikit-learn classifier
(e.g. ``RandomForestClassifier`` or ``GradientBoostingClassifier``) trained on
labelled features. It also exposes a small threat-intel lookup table.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

from .constants import AlertSeverity, ThreatType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ThreatClassification:
    """The output of classifying a single event/alert."""

    threat_type: ThreatType
    severity: AlertSeverity
    confidence: float  # 0.0 - 1.0
    method: str  # "rules" | "ml" | "hybrid"
    mitre_attack_ids: List[str] = field(default_factory=list)
    description: str = ""
    features_used: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIntelEntry:
    """A threat-intel entry (IoC or known-actor reference)."""

    id: str
    indicator: str
    indicator_type: str  # "ip" | "domain" | "hash" | "url" | "yara_rule"
    threat_type: ThreatType = ThreatType.UNKNOWN
    severity: AlertSeverity = AlertSeverity.MEDIUM
    description: str = ""
    source: str = ""
    mitre_attack_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------


@dataclass
class _ClassificationRule:
    """An internal rule that maps a feature-based predicate to a classification."""

    id: str
    predicate: Any  # callable[Dict[str, Any]] -> bool
    threat_type: ThreatType
    severity: AlertSeverity
    mitre_attack_ids: List[str] = field(default_factory=list)
    description: str = ""


def _always_false(_: Dict[str, Any]) -> bool:
    return False


# ---------------------------------------------------------------------------
# ThreatClassifier
# ---------------------------------------------------------------------------


class ThreatClassifier:
    """Classify threats by type and severity using rules + optional ML model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rules: List[_ClassificationRule] = []
        self._ml_model: Optional[Any] = None
        self._feature_names: List[str] = []
        self._threat_intel: Dict[str, ThreatIntelEntry] = {}
        self._stats = {"classified": 0, "by_type": {}, "by_severity": {}}
        self._install_default_rules()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule_id: str, predicate, threat_type: ThreatType,
                 severity: AlertSeverity, mitre_attack_ids: Optional[List[str]] = None,
                 description: str = "") -> None:
        """Register a custom classification rule."""
        with self._lock:
            self._rules.append(_ClassificationRule(
                id=rule_id, predicate=predicate, threat_type=threat_type,
                severity=severity,
                mitre_attack_ids=list(mitre_attack_ids or []),
                description=description,
            ))

    def _install_default_rules(self) -> None:
        # Rule helpers
        def has(event: Dict[str, Any], key: str, val: Any) -> bool:
            return event.get(key) == val

        def has_any(event: Dict[str, Any], key: str, vals) -> bool:
            return event.get(key) in vals

        def contains(event: Dict[str, Any], key: str, val: Any) -> bool:
            actual = event.get(key)
            if actual is None:
                return False
            if isinstance(actual, (list, tuple, set, dict)):
                return val in actual
            return val in str(actual)

        # CAN injection rules
        self.add_rule(
            "can-unauth-id",
            lambda e: e.get("alert_type") == "unauthorized_can_id",
            ThreatType.CAN_INJECTION, AlertSeverity.HIGH,
            ["T0817"],  # MITRE ICS: Modify Program / Fake Sensor/Messages
            "Unauthorized CAN ID observed on the bus",
        )
        self.add_rule(
            "can-injection-rate",
            lambda e: e.get("alert_type") == "message_injection",
            ThreatType.CAN_INJECTION, AlertSeverity.HIGH,
            ["T0817"],
            "Abnormal CAN message rate for a single arbitration ID",
        )
        self.add_rule(
            "can-flooding",
            lambda e: e.get("alert_type") == "bus_flooding",
            ThreatType.DOS, AlertSeverity.CRITICAL,
            ["T0814"],  # Denial of Service
            "Bus-wide flooding: aggregate rate exceeds safe threshold",
        )
        self.add_rule(
            "can-replay",
            lambda e: e.get("alert_type") == "replay_attack",
            ThreatType.CAN_INJECTION, AlertSeverity.HIGH,
            ["T0857"],  # Replay
            "Replay-style CAN injection",
        )
        # Network rules
        self.add_rule(
            "net-port-scan",
            lambda e: e.get("alert_type") == "port_scan",
            ThreatType.RECON, AlertSeverity.HIGH,
            ["T1046"],  # Network Service Discovery
            "Port scan detected",
        )
        self.add_rule(
            "net-c2-beacon",
            lambda e: e.get("alert_type") == "c2_beaconing",
            ThreatType.DATA_EXFIL, AlertSeverity.CRITICAL,
            ["T1071.001"],  # Application Layer Protocol: Web Protocols
            "C2 beaconing detected",
        )
        self.add_rule(
            "net-syn-flood",
            lambda e: e.get("alert_type") == "syn_flood",
            ThreatType.DOS, AlertSeverity.HIGH,
            ["T1498.001"],  # Network DoS: Direct Network Flood
            "SYN flood",
        )
        # Process / privilege rules
        self.add_rule(
            "auth-su-root",
            lambda e: e.get("pattern_id") == "auth-su-root",
            ThreatType.PRIVILEGE_ESCALATION, AlertSeverity.HIGH,
            ["T1548.003"],  # Setuid/Setgid
            "Privilege escalation via su",
        )
        self.add_rule(
            "auth-brute-force",
            lambda e: e.get("pattern_id") == "auth-brute-force",
            ThreatType.INTRUSION, AlertSeverity.HIGH,
            ["T1110.001"],  # Brute Force: Password Guessing
            "Authentication brute force",
        )
        # Malware rules
        self.add_rule(
            "malware-yara-hit",
            lambda e: e.get("type") == "yara_match",
            ThreatType.MALWARE, AlertSeverity.HIGH,
            ["T1059"],  # Command and Scripting Interpreter
            "YARA signature match",
        )
        # V2X misbehavior
        self.add_rule(
            "v2x-misbehavior",
            lambda e: e.get("alert_type") == "v2x_misbehavior",
            ThreatType.V2X_MISBEHAVIOR, AlertSeverity.CRITICAL,
            ["T0817"],
            "V2X message misbehavior",
        )

    # ------------------------------------------------------------------
    # Threat intel
    # ------------------------------------------------------------------

    def add_threat_intel(self, entry: ThreatIntelEntry) -> None:
        """Register a threat-intel entry (IoC lookup)."""
        with self._lock:
            self._threat_intel[entry.indicator] = entry

    def get_threat_intel(self, indicator: str) -> Optional[ThreatIntelEntry]:
        """Look up a threat-intel entry by indicator value."""
        with self._lock:
            return self._threat_intel.get(indicator)

    def import_threat_intel(self, path: str) -> int:
        """Import threat-intel entries from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
        count = 0
        for item in data:
            try:
                item.setdefault("threat_type", ThreatType.UNKNOWN.value)
                if isinstance(item["threat_type"], str):
                    item["threat_type"] = ThreatType(item["threat_type"])
                if isinstance(item.get("severity"), str):
                    item["severity"] = AlertSeverity.from_str(item["severity"])
                entry = ThreatIntelEntry(
                    id=item["id"], indicator=item["indicator"],
                    indicator_type=item["indicator_type"],
                    threat_type=item.get("threat_type", ThreatType.UNKNOWN),
                    severity=item.get("severity", AlertSeverity.MEDIUM),
                    description=item.get("description", ""),
                    source=item.get("source", ""),
                    mitre_attack_ids=item.get("mitre_attack_ids", []),
                )
                self.add_threat_intel(entry)
                count += 1
            except Exception as exc:
                logger.error("Failed to import threat intel entry: %s", exc)
        return count

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, event: Dict[str, Any]) -> ThreatClassification:
        """Classify a single event/alert.

        Parameters
        ----------
        event:
            A dict describing the event. Common keys: ``alert_type``,
            ``pattern_id``, ``type``, ``severity``, ``src_ip``, ``can_id``, ...
        """
        with self._lock:
            rules = list(self._rules)
            self._stats["classified"] += 1
        matched: List[_ClassificationRule] = []
        for r in rules:
            try:
                if r.predicate(event):
                    matched.append(r)
            except Exception as exc:
                logger.debug("Rule %s predicate error: %s", r.id, exc)
        # Also consult threat intel for indicators present in the event
        ti_hit: Optional[ThreatIntelEntry] = None
        for key in ("src_ip", "dst_ip", "hash", "url", "domain"):
            val = event.get(key)
            if val:
                ti = self.get_threat_intel(str(val))
                if ti is not None:
                    ti_hit = ti
                    break
        # Determine the most severe match
        if matched:
            best = max(matched, key=lambda r: int(r.severity))
            tc = ThreatClassification(
                threat_type=best.threat_type,
                severity=best.severity,
                confidence=0.9,
                method="rules",
                mitre_attack_ids=list(best.mitre_attack_ids),
                description=best.description,
                features_used=list(event.keys()),
            )
        elif ti_hit is not None:
            tc = ThreatClassification(
                threat_type=ti_hit.threat_type,
                severity=ti_hit.severity,
                confidence=0.75,
                method="rules",
                mitre_attack_ids=list(ti_hit.mitre_attack_ids),
                description=f"Threat intel match: {ti_hit.description}",
                features_used=["indicator"],
            )
        elif self._ml_model is not None and self._feature_names:
            tc = self._classify_ml(event)
        else:
            # Fallback: derive severity from event field if present
            sev = AlertSeverity.LOW
            if "severity" in event:
                try:
                    sev = AlertSeverity.from_str(event["severity"]) if isinstance(event["severity"], str) \
                        else AlertSeverity(int(event["severity"]))
                except (ValueError, KeyError):
                    pass
            tc = ThreatClassification(
                threat_type=ThreatType.UNKNOWN,
                severity=sev,
                confidence=0.3,
                method="rules",
                description="Unclassified event",
                features_used=list(event.keys()),
            )
        # Update stats
        with self._lock:
            tname = tc.threat_type.value
            sname = tc.severity.name
            self._stats["by_type"][tname] = self._stats["by_type"].get(tname, 0) + 1
            self._stats["by_severity"][sname] = self._stats["by_severity"].get(sname, 0) + 1
        return tc

    def _classify_ml(self, event: Dict[str, Any]) -> ThreatClassification:
        """Run the ML model on the event features."""
        row = [float(event.get(f, 0.0) or 0.0) for f in self._feature_names]
        try:
            import numpy as np
            pred = int(self._ml_model.predict([row])[0])
            proba = self._ml_model.predict_proba([row])[0]
            confidence = float(max(proba))
        except Exception as exc:
            logger.error("ML classifier inference failed: %s", exc)
            return ThreatClassification(
                threat_type=ThreatType.UNKNOWN, severity=AlertSeverity.LOW,
                confidence=0.0, method="ml", features_used=list(self._feature_names),
                extras={"error": str(exc)},
            )
        try:
            threat_type = ThreatType(self._label_map[pred])  # type: ignore[attr-defined]
        except Exception:
            threat_type = ThreatType.UNKNOWN
        return ThreatClassification(
            threat_type=threat_type,
            severity=self._severity_for_type(threat_type),
            confidence=confidence,
            method="ml",
            features_used=list(self._feature_names),
        )

    def _severity_for_type(self, threat_type: ThreatType) -> AlertSeverity:
        """Default severity per threat type (when ML is used without rules)."""
        defaults = {
            ThreatType.DOS: AlertSeverity.HIGH,
            ThreatType.CAN_INJECTION: AlertSeverity.HIGH,
            ThreatType.V2X_MISBEHAVIOR: AlertSeverity.CRITICAL,
            ThreatType.MALWARE: AlertSeverity.HIGH,
            ThreatType.DATA_EXFIL: AlertSeverity.HIGH,
            ThreatType.PRIVILEGE_ESCALATION: AlertSeverity.HIGH,
            ThreatType.INTRUSION: AlertSeverity.MEDIUM,
            ThreatType.RECON: AlertSeverity.LOW,
            ThreatType.LATERAL_MOVEMENT: AlertSeverity.HIGH,
            ThreatType.SUPPLY_CHAIN: AlertSeverity.CRITICAL,
            ThreatType.UNKNOWN: AlertSeverity.LOW,
        }
        return defaults.get(threat_type, AlertSeverity.MEDIUM)

    # ------------------------------------------------------------------
    # Severity assessment (override from raw confidence + impact)
    # ------------------------------------------------------------------

    def assess_severity(
        self,
        threat_type: ThreatType,
        confidence: float,
        impact: Optional[Dict[str, Any]] = None,
    ) -> AlertSeverity:
        """Assess severity from confidence + optional impact dict."""
        base = self._severity_for_type(threat_type)
        if impact and impact.get("safety_critical"):
            return AlertSeverity.CRITICAL
        if confidence < 0.4:
            # Degrade one level
            return AlertSeverity(max(int(base) - 1, 0))
        if confidence > 0.85 and impact and impact.get("widespread"):
            return AlertSeverity.CRITICAL
        return base

    # ------------------------------------------------------------------
    # ML model management
    # ------------------------------------------------------------------

    def train_ml(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[str],
        feature_names: Sequence[str],
    ) -> None:
        """Train the ML classifier from labelled data."""
        if not _HAS_SKLEARN:
            logger.warning("scikit-learn not installed; cannot train ML classifier")
            return
        import numpy as np
        labels = sorted(set(y))
        self._label_map = {i: lbl for i, lbl in enumerate(labels)}
        # sklearn expects numeric labels; remap
        y_num = [labels.index(v) for v in y]
        self._ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._ml_model.fit(np.asarray(X, dtype=float), np.asarray(y_num))
        self._feature_names = list(feature_names)
        logger.info("Trained ML classifier on %d samples, %d classes",
                    len(X), len(labels))

    def save_ml(self, path: str) -> None:
        if self._ml_model is None:
            raise RuntimeError("No ML model to save")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "model": self._ml_model,
                "feature_names": self._feature_names,
                "label_map": getattr(self, "_label_map", {}),
            }, fh)

    def load_ml(self, path: str) -> None:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self._ml_model = payload["model"]
        self._feature_names = payload["feature_names"]
        self._label_map = payload.get("label_map", {})

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


__all__ = ["ThreatClassifier", "ThreatClassification", "ThreatIntelEntry"]
