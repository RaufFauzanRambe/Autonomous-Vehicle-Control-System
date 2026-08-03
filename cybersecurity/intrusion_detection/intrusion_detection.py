"""Intrusion detection orchestrator.

The :class:`IntrusionDetectionSystem` class is the top-level entry point for
the AVCS intrusion detection subsystem. It wires together all detectors,
analyzers, the alert manager, the response engine, the incident logger, the
forensic tools and the realtime monitor. It exposes a small surface
(``start``, ``stop``, ``register_detector``, ``process_event``, ``get_alerts``,
``get_statistics``) suitable for embedding in a ROS2 node or a standalone
service.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .alert_manager import Alert, AlertManager
from .anomaly_detection import AnomalyDetector
from .attack_signature import AttackSignatureDB
from .behavior_analysis import BehaviorAnalyzer
from .can_bus_monitor import CANBusMonitor
from .config import IDSConfig, default_config, load_config
from .constants import (
    AlertSeverity,
    AlertStatus,
    EventType,
    ResponseActionType,
    ThreatType,
)
from .forensic_tools import ForensicTools
from .ids_engine import IDSEngine, IDSEvent, IDSRule
from .incident_logger import IncidentLogger
from .log_analyzer import LogAnalyzer
from .malware_detector import MalwareDetector
from .network_monitor import NetworkMonitor
from .packet_analyzer import PacketAnalyzer
from .realtime_monitor import RealtimeMonitor
from .response_engine import ResponseEngine
from .threat_classifier import ThreatClassification, ThreatClassifier
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class IntrusionDetectionSystem:
    """Top-level IDS orchestrator for the AVCS cybersecurity module."""

    def __init__(self, config: Optional[Union[IDSConfig, str, Dict[str, Any]]] = None) -> None:
        if config is None:
            self.config: IDSConfig = default_config()
        elif isinstance(config, IDSConfig):
            self.config = config
        else:
            self.config = load_config(config)

        # Configure logging level from config
        if self.config.debug or self.config.log_level.upper() == "DEBUG":
            logging.getLogger("avcs.ids").setLevel(logging.DEBUG)

        # --- Core engine ---
        self.engine = IDSEngine(instance_id=self.config.instance_id)

        # --- Detectors / analyzers ---
        self.anomaly_detector = AnomalyDetector(
            zscore_threshold=self.config.anomaly.zscore_threshold,
            iqr_multiplier=self.config.anomaly.iqr_multiplier,
            min_samples=self.config.anomaly.training_samples,
            isolation_forest_contamination=self.config.anomaly.isolation_forest_contamination,
        )
        self.engine.set_anomaly_detector(self.anomaly_detector)

        self.signature_db = AttackSignatureDB()
        self.engine.set_signature_db(self.signature_db)

        self.malware_detector = MalwareDetector(
            yara_rules_dir=self.config.rules.yara_rules_dir or None,
        )
        self.can_bus_monitor = CANBusMonitor(
            interface="socketcan",
            channel=self.config.capture.can_interface,
        )
        self.network_monitor = NetworkMonitor(
            interface=self.config.capture.network_interface,
            bpf_filter=self.config.capture.bpf_filter,
            snaplen=self.config.capture.pcap_snaplen,
            timeout_ms=self.config.capture.pcap_timeout_ms,
            promiscuous=self.config.capture.promiscuous,
        )
        self.packet_analyzer = PacketAnalyzer()
        self.log_analyzer = LogAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.threat_classifier = ThreatClassifier()

        # --- Lifecycle managers ---
        self.alert_manager = AlertManager(
            dedup_window_sec=self.config.alerts.dedup_window_sec,
            escalation_sec=self.config.alerts.escalation_sec,
            enable_correlation=self.config.alerts.enable_correlation,
            correlation_window_sec=self.config.alerts.correlation_window_sec,
        )
        self.response_engine = ResponseEngine(
            enabled=self.config.response.enabled,
            dry_run=self.config.response.dry_run,
            max_actions_per_min=self.config.response.max_actions_per_min,
            cooldown_sec=self.config.response.cooldown_sec,
        )
        self.incident_logger = IncidentLogger(
            log_path=self.config.incident_log.log_path,
            hash_algorithm=self.config.incident_log.hash_algorithm,
            retention_days=self.config.incident_log.retention_days,
        )
        self.forensic_tools = ForensicTools(
            evidence_dir=self.config.forensic.evidence_dir,
            hash_algorithm=self.config.forensic.hash_algorithm,
            max_size_mb=self.config.forensic.max_evidence_size_mb,
            compress=self.config.forensic.compress_evidence,
        )
        self.realtime_monitor = RealtimeMonitor(
            instance_id=self.config.instance_id,
        )

        # --- State ---
        self._lock = threading.RLock()
        self._started = False
        self._started_at = 0.0
        self._detectors: Dict[str, Any] = {}

        # Wire sources into the realtime monitor
        self._wire_realtime_sources()
        # Wire alert callbacks
        self._wire_alert_pipeline()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_realtime_sources(self) -> None:
        self.realtime_monitor.register_source("alert_manager", self._alert_source)
        self.realtime_monitor.register_source("ids_engine", self.engine.get_event_pipeline_stats)
        self.realtime_monitor.register_source("response_engine", self.response_engine.get_statistics)
        self.realtime_monitor.register_source("incident_logger", self.incident_logger.get_statistics)
        self.realtime_monitor.register_source("threat_classifier", self.threat_classifier.get_statistics)
        self.realtime_monitor.register_source("can_bus", self.can_bus_monitor.get_statistics)
        self.realtime_monitor.register_source("network", self.network_monitor.get_statistics)
        self.realtime_monitor.register_source("packet_analyzer", self.packet_analyzer.get_statistics)
        self.realtime_monitor.register_source("log_analyzer", self.log_analyzer.get_statistics)
        self.realtime_monitor.register_source("malware_detector", self.malware_detector.stats)
        self.realtime_monitor.register_source("behavior", self.behavior_analyzer.get_statistics)

    def _alert_source(self) -> Dict[str, Any]:
        stats = self.alert_manager.get_statistics()
        recent = self.alert_manager.get_active_alerts(limit=10)
        stats["recent_alerts"] = [a.to_dict() for a in recent]
        return stats

    def _wire_alert_pipeline(self) -> None:
        """Forward detector alerts → alert manager → classifier → response."""
        def can_alert_handler(can_alert) -> None:
            self._process_detector_alert(
                title=f"CAN alert: {can_alert.alert_type}",
                description=can_alert.description,
                evidence=dict(can_alert.evidence, can_id=can_alert.can_id,
                              alert_type=can_alert.alert_type),
                source="can_bus_monitor",
            )

        def net_alert_handler(net_alert) -> None:
            self._process_detector_alert(
                title=f"Network alert: {net_alert.alert_type}",
                description=net_alert.description,
                evidence=dict(net_alert.evidence, src_ip=net_alert.src_ip,
                              dst_ip=net_alert.dst_ip, alert_type=net_alert.alert_type),
                source="network_monitor",
            )

        def dpi_alert_handler(dpi_alert) -> None:
            self._process_detector_alert(
                title=f"DPI alert: {dpi_alert.rule_id}",
                description=dpi_alert.description,
                evidence=dpi_alert.metadata,
                source="packet_analyzer",
            )

        def log_finding_handler(finding) -> None:
            self._process_detector_alert(
                title=f"Log finding: {finding.pattern_id}",
                description=finding.description,
                evidence={"source": finding.source, "raw_line": finding.raw_line,
                          "groups": finding.groups, "pattern_id": finding.pattern_id},
                source="log_analyzer",
            )

        self.can_bus_monitor.register_alert_callback(can_alert_handler)
        self.network_monitor.register_alert_callback(net_alert_handler)
        self.packet_analyzer.register_alert_callback(dpi_alert_handler)
        self.log_analyzer.register_finding_callback(log_finding_handler)
        # Engine alerts (rule matches / anomalies)
        def engine_alert_handler(event: IDSEvent, rules: List[IDSRule], anomaly: Dict[str, Any]) -> None:
            self._process_detector_alert(
                title=f"IDS rule(s) matched: {', '.join(r.id for r in rules)}" if rules else "Anomaly detected",
                description=" / ".join(r.description for r in rules) or "Anomaly score exceeded threshold",
                evidence={"event": event.to_dict(), "rules": [r.id for r in rules], "anomaly": anomaly},
                source="ids_engine",
            )
        self.engine.register_alert_callback(engine_alert_handler)

    # ------------------------------------------------------------------
    # Detector alert → alert manager → classifier → response / incident
    # ------------------------------------------------------------------

    def _process_detector_alert(
        self,
        title: str,
        description: str,
        evidence: Dict[str, Any],
        source: str,
    ) -> None:
        """Run the full alert pipeline for a detector-emitted alert."""
        # Classify first (use evidence dict as the event)
        classification: ThreatClassification = self.threat_classifier.classify({
            "alert_type": evidence.get("alert_type"),
            "pattern_id": evidence.get("pattern_id"),
            "type": evidence.get("type"),
            "severity": evidence.get("severity"),
            "src_ip": evidence.get("src_ip"),
            "dst_ip": evidence.get("dst_ip"),
            "can_id": evidence.get("can_id"),
            "hash": evidence.get("hash"),
            "rule_id": evidence.get("rule_id"),
        })
        severity = classification.severity
        threat_type = classification.threat_type
        # Raise the alert (deduplicated)
        alert = self.alert_manager.raise_alert(
            title=title,
            description=description,
            severity=severity,
            threat_type=threat_type,
            source=source,
            evidence=evidence,
            mitre_attack_ids=classification.mitre_attack_ids,
        )
        # Only act on NEW alerts (not deduplicated ones)
        if alert.count == 1 and alert.status == AlertStatus.NEW:
            # Log to incident log
            self.incident_logger.log_incident(
                title=title,
                description=description,
                severity=severity,
                threat_type=threat_type,
                source=source,
                alert_ids=[alert.id],
                evidence=evidence,
            )
            # Trigger automated response
            self.response_engine.execute_response(
                severity=severity,
                target=self._response_target(evidence),
                alert_id=alert.id,
                parameters=dict(evidence),
            )

    @staticmethod
    def _response_target(evidence: Dict[str, Any]) -> str:
        return (
            evidence.get("src_ip")
            or evidence.get("can_id")
            or evidence.get("target")
            or "unknown"
        )

    # ------------------------------------------------------------------
    # Detector registration
    # ------------------------------------------------------------------

    def register_detector(self, name: str, detector: Any) -> None:
        """Register an external detector (any object with ``get_statistics``)."""
        with self._lock:
            self._detectors[name] = detector
        self.engine.register_detector(name, detector)
        self.realtime_monitor.register_source(name, detector.get_statistics)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the IDS: engine, monitors, alert manager, realtime monitor."""
        if self._started:
            logger.warning("IDS already started")
            return
        logger.info("Starting IDS instance '%s'", self.config.instance_id)
        self._started_at = timestamp_now()
        self.alert_manager.start()
        self.engine.start()
        self.can_bus_monitor.start_capture()
        self.network_monitor.start_capture()
        self.realtime_monitor.start()
        self._started = True
        logger.info("IDS started")

    def stop(self) -> None:
        """Stop the IDS and all background threads."""
        if not self._started:
            return
        logger.info("Stopping IDS instance '%s'", self.config.instance_id)
        self.realtime_monitor.stop()
        self.network_monitor.stop_capture()
        self.can_bus_monitor.stop_capture()
        self.engine.stop()
        self.alert_manager.stop()
        self._started = False
        logger.info("IDS stopped")

    # ------------------------------------------------------------------
    # Event ingestion (public API)
    # ------------------------------------------------------------------

    def process_event(
        self,
        event_type: EventType,
        source: str,
        payload: Dict[str, Any],
        raw: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[IDSEvent]:
        """Submit an event to the IDS engine for processing."""
        event = self.engine.ingest_event(
            event_type=event_type,
            source=source,
            payload=payload,
            raw=raw,
            metadata=metadata,
        )
        if event is not None:
            self.realtime_monitor.record_event()
        return event

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_alerts(self, limit: int = 100, active_only: bool = True) -> List[Alert]:
        if active_only:
            return self.alert_manager.get_active_alerts(limit=limit)
        return self.alert_manager.get_all_alerts(limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        snapshot = self.realtime_monitor.get_current_state()
        if snapshot is not None:
            return snapshot.to_dict()
        # Fall back to building a fresh snapshot synchronously
        return self.realtime_monitor.publish_update().to_dict()

    def get_current_state(self) -> Dict[str, Any]:
        """Alias for :meth:`get_statistics`."""
        return self.get_statistics()

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the incident log chain."""
        ok, errors = self.incident_logger.verify_chain()
        return {"intact": ok, "errors": errors}

    def collect_forensics(self, case_name: Optional[str] = None) -> Dict[str, Any]:
        """Trigger a forensic snapshot of the running system."""
        items = self.forensic_tools.snapshot_system(
            case_name=case_name or f"ids-{int(timestamp_now())}",
            collect_memory=False,
            collect_disk=False,
            collect_capture=True,
            collect_can=True,
        )
        return {
            "case_items": [i.to_dict() for i in items],
            "stats": self.forensic_tools.get_statistics(),
        }

    def __enter__(self) -> "IntrusionDetectionSystem":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


__all__ = ["IntrusionDetectionSystem"]
