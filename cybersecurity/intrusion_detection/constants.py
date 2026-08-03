"""Constants, enumerations, and default values for the intrusion detection module.

This module centralizes all magic numbers, enumerations, and default
configuration values used across the intrusion detection subsystem of the
Autonomous Vehicle Control System (AVCS). Centralizing these values keeps the
detection logic consistent across the IDS engine, anomaly detector, CAN bus
monitor, network monitor, alert manager, response engine, and forensic tools.
"""

from __future__ import annotations

import enum
from typing import Final


# ---------------------------------------------------------------------------
# Default timing and rate-limit values
# ---------------------------------------------------------------------------

DEFAULT_CAPTURE_INTERVAL_SEC: Final[float] = 1.0
"""Default polling interval (seconds) for capture loops (CAN, network, logs)."""

DEFAULT_STATS_INTERVAL_SEC: Final[float] = 5.0
"""Default interval for publishing aggregated statistics."""

DEFAULT_HEARTBEAT_INTERVAL_SEC: Final[float] = 10.0
"""Default interval for component heartbeat messages."""

MAX_EVENT_RATE_PER_SEC: Final[int] = 50_000
"""Maximum event ingest rate (events/sec) before back-pressure is applied."""

MAX_CAN_FRAME_RATE_PER_ID: Final[int] = 1_000
"""Per-CAN-ID message rate above which injection is suspected."""

DEFAULT_PORT_SCAN_THRESHOLD: Final[int] = 10
"""Number of distinct ports contacted by a single host in a window to flag a scan."""

DEFAULT_PORT_SCAN_WINDOW_SEC: Final[float] = 60.0
"""Time window for port-scan detection."""

DEFAULT_C2_BEACON_INTERVAL_SEC: Final[float] = 60.0
"""Typical C2 beaconing interval — used as a baseline."""

DEFAULT_C2_BEACON_JITTER_SEC: Final[float] = 5.0
"""Acceptable jitter around C2 beacon interval."""

DEFAULT_ANOMALY_ZSCORE_THRESHOLD: Final[float] = 3.0
"""Z-score above which a metric is considered anomalous."""

DEFAULT_ANOMALY_IQR_MULTIPLIER: Final[float] = 1.5
"""IQR multiplier for outlier detection (Tukey's fence)."""

DEFAULT_ALERT_TTL_SEC: Final[int] = 86_400
"""Default alert retention (24 hours) before automatic expiry."""

DEFAULT_INCIDENT_RETENTION_DAYS: Final[int] = 365
"""Default retention for incident log entries."""

DEFAULT_BASELINE_TRAINING_SAMPLES: Final[int] = 1_000
"""Minimum number of samples to train a baseline model."""

DEFAULT_DEDUP_WINDOW_SEC: Final[float] = 30.0
"""Window for alert deduplication."""

DEFAULT_ESCALATION_TIME_SEC: Final[int] = 900
"""Time without acknowledgement before an alert is escalated (15 min)."""


# ---------------------------------------------------------------------------
# Network defaults
# ---------------------------------------------------------------------------

DEFAULT_PCAP_SNAPLEN: Final[int] = 65535
"""Default pcap snapshot length."""

DEFAULT_PCAP_TIMEOUT_MS: Final[int] = 100
"""Default pcap read timeout in milliseconds."""

DEFAULT_CAN_INTERFACE: Final[str] = "can0"
"""Default CAN interface name on Linux."""

DEFAULT_NETWORK_INTERFACE: Final[str] = "eth0"
"""Default network capture interface."""

COMMON_VEHICLE_PORTS: Final[tuple[int, ...]] = (
    15765,  # Unified Diagnostic Services (UDS) over DoIP
    13400,  # DoIP
    30490,  # SOME-IP
    18801,  # SOME-IP SD
    1883,   # MQTT
    5353,   # mDNS (used by V2X discovery)
    53,     # DNS
)
"""TCP/UDP ports commonly seen in in-vehicle networks."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AlertSeverity(enum.IntEnum):
    """Severity levels for IDS alerts, ordered from least to most severe."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_str(cls, name: str) -> "AlertSeverity":
        """Parse a severity from a case-insensitive string."""
        return cls[name.strip().upper()]


class AlertStatus(enum.Enum):
    """Lifecycle states for an alert managed by :class:`AlertManager`."""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"


class ThreatType(enum.Enum):
    """High-level taxonomy of threats the IDS can detect.

    Aligns loosely with the MITRE ATT&CK for ICS / Automotive matrices.
    """

    RECON = "recon"
    INTRUSION = "intrusion"
    MALWARE = "malware"
    DOS = "dos"
    DATA_EXFIL = "data_exfil"
    SUPPLY_CHAIN = "supply_chain"
    LATERAL_MOVEMENT = "lateral_movement"
    CAN_INJECTION = "can_injection"
    V2X_MISBEHAVIOR = "v2x_misbehavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNKNOWN = "unknown"


class EventType(enum.Enum):
    """Categories of events processed by the IDS event pipeline."""

    CAN_FRAME = "can_frame"
    NETWORK_PACKET = "network_packet"
    LOG_ENTRY = "log_entry"
    PROCESS_EVENT = "process_event"
    FILE_EVENT = "file_event"
    SENSOR_READING = "sensor_reading"
    SYSTEM_CALL = "system_call"
    USER_ACTION = "user_action"
    DIAGNOSTIC = "diagnostic"
    V2X_MESSAGE = "v2x_message"


class ResponseActionType(enum.Enum):
    """Automated response actions the :class:`ResponseEngine` can execute."""

    BLOCK_IP = "block_ip"
    BLOCK_PORT = "block_port"
    DISABLE_ECU = "disable_ecu"
    ISOLATE_NETWORK = "isolate_network"
    KILL_PROCESS = "kill_process"
    QUARANTINE_FILE = "quarantine_file"
    TRIGGER_LOCKDOWN = "trigger_lockdown"
    NOTIFY_SIEM = "notify_siem"
    COLLECT_EVIDENCE = "collect_evidence"
    RATE_LIMIT = "rate_limit"
    DISABLE_INTERFACE = "disable_interface"


class IncidentStatus(enum.Enum):
    """Lifecycle states for an incident recorded by :class:`IncidentLogger`."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"


class DetectorState(enum.Enum):
    """State of a detector component in the IDS engine."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class EvidenceType(enum.Enum):
    """Types of forensic evidence collected by :class:`ForensicTools`."""

    MEMORY_DUMP = "memory_dump"
    DISK_IMAGE = "disk_image"
    PACKET_CAPTURE = "packet_capture"
    CAN_LOG = "can_log"
    PROCESS_LIST = "process_list"
    NETWORK_STATE = "network_state"
    LOG_SNAPSHOT = "log_snapshot"
    YARA_HIT = "yara_hit"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Default alert TTLs per severity (seconds)
# ---------------------------------------------------------------------------

ALERT_TTL_BY_SEVERITY: Final[dict[AlertSeverity, int]] = {
    AlertSeverity.INFO: 3600,
    AlertSeverity.LOW: 6 * 3600,
    AlertSeverity.MEDIUM: 24 * 3600,
    AlertSeverity.HIGH: 7 * 24 * 3600,
    AlertSeverity.CRITICAL: 30 * 24 * 3600,
}
"""Retention policy for alerts, indexed by severity."""


# ---------------------------------------------------------------------------
# Default response policies per severity
# ---------------------------------------------------------------------------

DEFAULT_RESPONSE_POLICY: Final[dict[AlertSeverity, tuple[ResponseActionType, ...]]] = {
    AlertSeverity.INFO: (ResponseActionType.NOTIFY_SIEM,),
    AlertSeverity.LOW: (ResponseActionType.NOTIFY_SIEM, ResponseActionType.RATE_LIMIT),
    AlertSeverity.MEDIUM: (
        ResponseActionType.NOTIFY_SIEM,
        ResponseActionType.RATE_LIMIT,
        ResponseActionType.COLLECT_EVIDENCE,
    ),
    AlertSeverity.HIGH: (
        ResponseActionType.NOTIFY_SIEM,
        ResponseActionType.BLOCK_IP,
        ResponseActionType.COLLECT_EVIDENCE,
        ResponseActionType.DISABLE_INTERFACE,
    ),
    AlertSeverity.CRITICAL: (
        ResponseActionType.NOTIFY_SIEM,
        ResponseActionType.BLOCK_IP,
        ResponseActionType.DISABLE_ECU,
        ResponseActionType.ISOLATE_NETWORK,
        ResponseActionType.TRIGGER_LOCKDOWN,
        ResponseActionType.COLLECT_EVIDENCE,
    ),
}
"""Default automated response actions keyed by alert severity."""


__all__ = [
    "DEFAULT_CAPTURE_INTERVAL_SEC",
    "DEFAULT_STATS_INTERVAL_SEC",
    "DEFAULT_HEARTBEAT_INTERVAL_SEC",
    "MAX_EVENT_RATE_PER_SEC",
    "MAX_CAN_FRAME_RATE_PER_ID",
    "DEFAULT_PORT_SCAN_THRESHOLD",
    "DEFAULT_PORT_SCAN_WINDOW_SEC",
    "DEFAULT_C2_BEACON_INTERVAL_SEC",
    "DEFAULT_C2_BEACON_JITTER_SEC",
    "DEFAULT_ANOMALY_ZSCORE_THRESHOLD",
    "DEFAULT_ANOMALY_IQR_MULTIPLIER",
    "DEFAULT_ALERT_TTL_SEC",
    "DEFAULT_INCIDENT_RETENTION_DAYS",
    "DEFAULT_BASELINE_TRAINING_SAMPLES",
    "DEFAULT_DEDUP_WINDOW_SEC",
    "DEFAULT_ESCALATION_TIME_SEC",
    "DEFAULT_PCAP_SNAPLEN",
    "DEFAULT_PCAP_TIMEOUT_MS",
    "DEFAULT_CAN_INTERFACE",
    "DEFAULT_NETWORK_INTERFACE",
    "COMMON_VEHICLE_PORTS",
    "ALERT_TTL_BY_SEVERITY",
    "DEFAULT_RESPONSE_POLICY",
    "AlertSeverity",
    "AlertStatus",
    "ThreatType",
    "EventType",
    "ResponseActionType",
    "IncidentStatus",
    "DetectorState",
    "EvidenceType",
]
