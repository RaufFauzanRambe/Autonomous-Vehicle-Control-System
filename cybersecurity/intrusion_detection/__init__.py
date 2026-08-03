"""Intrusion detection subpackage for the AVCS cybersecurity module.

Re-exports the main orchestrator and the most commonly used dataclasses for
convenience::

    from cybersecurity.intrusion_detection import IntrusionDetectionSystem
    from cybersecurity.intrusion_detection import IDSConfig, AlertSeverity, ThreatType
"""

from .alert_manager import Alert, AlertManager
from .anomaly_detection import AnomalyDetector, AnomalyResult
from .attack_signature import AttackSignature, AttackSignatureDB
from .behavior_analysis import BehaviorAnalyzer, BehaviorScore
from .can_bus_monitor import CANBusMonitor, CANFrameEvent
from .config import IDSConfig, default_config, load_config
from .constants import (
    AlertSeverity,
    AlertStatus,
    EventType,
    ResponseActionType,
    ThreatType,
)
from .forensic_tools import EvidenceItem, ForensicTools
from .ids_engine import IDSEngine, IDSEvent, IDSRule
from .incident_logger import IncidentLogger, IncidentRecord
from .intrusion_detection import IntrusionDetectionSystem
from .log_analyzer import LogAnalyzer, LogFinding, LogPattern
from .malware_detector import IoCSignature, MalwareDetector, ScanResult
from .network_monitor import NetworkMonitor
from .packet_analyzer import PacketAnalyzer, PacketMetadata, PacketRule
from .realtime_monitor import RealtimeMonitor
from .response_engine import ResponseAction, ResponseEngine
from .threat_classifier import ThreatClassification, ThreatClassifier

__all__ = [
    "IntrusionDetectionSystem",
    "IDSConfig",
    "default_config",
    "load_config",
    "IDSEngine",
    "IDSEvent",
    "IDSRule",
    "AnomalyDetector",
    "AnomalyResult",
    "AttackSignature",
    "AttackSignatureDB",
    "MalwareDetector",
    "IoCSignature",
    "ScanResult",
    "CANBusMonitor",
    "CANFrameEvent",
    "NetworkMonitor",
    "PacketAnalyzer",
    "PacketMetadata",
    "PacketRule",
    "LogAnalyzer",
    "LogPattern",
    "LogFinding",
    "BehaviorAnalyzer",
    "BehaviorScore",
    "ThreatClassifier",
    "ThreatClassification",
    "AlertManager",
    "Alert",
    "IncidentLogger",
    "IncidentRecord",
    "ResponseEngine",
    "ResponseAction",
    "ForensicTools",
    "EvidenceItem",
    "RealtimeMonitor",
    "AlertSeverity",
    "AlertStatus",
    "EventType",
    "ResponseActionType",
    "ThreatType",
]
