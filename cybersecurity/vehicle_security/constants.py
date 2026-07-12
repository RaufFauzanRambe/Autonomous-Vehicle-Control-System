"""Constants for the vehicle_security subsystem.

Defines severity levels, security event types, ECU identifiers, CAN bus IDs,
ASIL safety goals, default network ports, filesystem paths, and version
strings used across all vehicle-security modules. Keeping these in one module
makes it easy to audit configuration and to keep enumerations consistent.

References:
  - ISO 26262 (ASIL classification)
  - ISO/SAE 21434 (vehicle cybersecurity engineering)
  - AUTOSAR Secure Onboard Communication
"""

from __future__ import annotations

import enum
from typing import Final, Dict, Tuple

# --------------------------------------------------------------------------- #
# Version
# --------------------------------------------------------------------------- #

MODULE_VERSION: Final[str] = "1.4.0"
SCHEMA_VERSION: Final[str] = "1.0"
PROTOCOL_VERSION: Final[str] = "sec-v2x-1.2"

# --------------------------------------------------------------------------- #
# Severity levels
# --------------------------------------------------------------------------- #


class Severity(str, enum.Enum):
    """Severity levels used for events, incidents, and findings."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def weight(cls, sev: "Severity") -> int:
        order = {cls.INFO: 0, cls.LOW: 1, cls.MEDIUM: 2, cls.HIGH: 3, cls.CRITICAL: 4}
        return order[sev]


SEVERITY_ORDER: Final[Dict[str, int]] = {s.value: Severity.weight(s) for s in Severity}

# --------------------------------------------------------------------------- #
# Security event types
# --------------------------------------------------------------------------- #


class EventType(str, enum.Enum):
    FIREWALL_BLOCK = "firewall.block"
    FIREWALL_RATE_LIMIT = "firewall.rate_limit"
    CAN_BUS_ANOMALY = "can.anomaly"
    ECU_OFFLINE = "ecu.offline"
    ECU_RESTART = "ecu.restart"
    TAMPER_DETECTED = "tamper.detected"
    INTRUSION_CASE_OPEN = "tamper.case_open"
    BOOT_ATTESTATION_FAIL = "boot.attestation_fail"
    BOOT_STAGE_OK = "boot.stage_ok"
    UPDATE_STARTED = "update.started"
    UPDATE_VERIFIED = "update.verified"
    UPDATE_FAILED = "update.failed"
    UPDATE_ROLLED_BACK = "update.rolled_back"
    FIRMWARE_INVALID = "firmware.invalid"
    INTEGRITY_VIOLATION = "integrity.violation"
    SAFETY_VIOLATION = "safety.violation"
    SAFETY_STATE_CHANGE = "safety.state_change"
    VULN_FOUND = "vuln.found"
    CVE_INDEX_UPDATED = "vuln.index_updated"
    LOCKDOWN_TRIGGERED = "lockdown.triggered"
    LOCKDOWN_RELEASED = "lockdown.released"
    INCIDENT_REPORTED = "incident.reported"
    INCIDENT_CONTAINED = "incident.contained"
    KEY_ROTATED = "storage.key_rotated"
    CONFIG_CHANGED = "config.changed"


# --------------------------------------------------------------------------- #
# ECU identifiers
# --------------------------------------------------------------------------- #

ECU_IDS: Final[Dict[str, int]] = {
    "POWERTRAIN": 0x01,
    "BRAKES": 0x02,
    "STEERING": 0x03,
    "BODY": 0x04,
    "INFOTAINMENT": 0x05,
    "GATEWAY": 0x10,
    "ADAS_FRONT": 0x20,
    "ADAS_REAR": 0x21,
    "ADAS_SIDE_L": 0x22,
    "ADAS_SIDE_R": 0x23,
    "V2X": 0x30,
    "TELEMETRY": 0x40,
    "OBD": 0x50,
    "TPMS": 0x60,
}

CRITICAL_ECUS: Final[Tuple[str, ...]] = ("BRAKES", "STEERING", "POWERTRAIN", "ADAS_FRONT")

# --------------------------------------------------------------------------- #
# CAN bus identifiers (sample standard IDs used for firewall allowlisting)
# --------------------------------------------------------------------------- #

CAN_ID_BRAKE_COMMAND: Final[int] = 0x100
CAN_ID_STEERING_COMMAND: Final[int] = 0x101
CAN_ID_THROTTLE_COMMAND: Final[int] = 0x102
CAN_ID_WHEEL_SPEED: Final[int] = 0x110
CAN_ID_STEERING_ANGLE: Final[int] = 0x111
CAN_ID_BRAKE_PRESSURE: Final[int] = 0x112
CAN_ID_ODOMETER: Final[int] = 0x120
CAN_ID_TIRE_PRESSURE: Final[int] = 0x130
CAN_ID_DIAGNOSTIC: Final[int] = 0x7DF
CAN_ID_UDS_RESPONSE: Final[int] = 0x7E8

SAFETY_CRITICAL_CAN_IDS: Final[Tuple[int, ...]] = (
    CAN_ID_BRAKE_COMMAND,
    CAN_ID_STEERING_COMMAND,
    CAN_ID_THROTTLE_COMMAND,
)

# --------------------------------------------------------------------------- #
# ASIL safety goals
# --------------------------------------------------------------------------- #


class ASILLevel(str, enum.Enum):
    QM = "QM"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


# ASIL weight for comparison (higher is more safety-critical)
ASIL_WEIGHT: Final[Dict[str, int]] = {"QM": 0, "A": 1, "B": 2, "C": 3, "D": 4}

SAFETY_GOAL_BRAKING: Final[str] = "SG-BRAKE-001"
SAFETY_GOAL_STEERING: Final[str] = "SG-STEER-001"
SAFETY_GOAL_POWERTRAIN: Final[str] = "SG-PT-001"
SAFETY_GOAL_MOTION_CONTROL: Final[str] = "SG-MOTION-001"

# --------------------------------------------------------------------------- #
# Default network ports
# --------------------------------------------------------------------------- #

DEFAULT_PORT_V2X: Final[int] = 47900  # SAE J2735 / WAVE
DEFAULT_PORT_SOMEIP: Final[int] = 30490  # SOME/IP-SD
DEFAULT_PORT_DOIP: Final[int] = 13400  # Diagnostics over IP
DEFAULT_PORT_OTA: Final[int] = 443
DEFAULT_PORT_TELEMETRY: Final[int] = 8443
DEFAULT_PORT_OBD2: Final[int] = 23

BLOCKED_PORTS_BY_DEFAULT: Final[Tuple[int, ...]] = (
    23,    # telnet
    21,    # ftp
    3389,  # RDP
)

# --------------------------------------------------------------------------- #
# Filesystem paths (defaults; overridable via config)
# --------------------------------------------------------------------------- #

DEFAULT_BASE_DIR: Final[str] = "/etc/avcs/security"
DEFAULT_LOG_PATH: Final[str] = "/var/log/avcs/security_events.jsonl"
DEFAULT_SECURE_STORE_PATH: Final[str] = "/var/lib/avcs/secure_store.db"
DEFAULT_BASELINE_DB_PATH: Final[str] = "/var/lib/avcs/integrity_baseline.json"
DEFAULT_CVE_INDEX_PATH: Final[str] = "/var/lib/avcs/cve_index.json"
DEFAULT_FIREWALL_RULES_PATH: Final[str] = "/etc/avcs/security/firewall_rules.yaml"
DEFAULT_MANIFEST_PATH: Final[str] = "/etc/avcs/security/boot_manifest.json"
DEFAULT_TPM_DEVICE: Final[str] = "/dev/tpmrmis0"

# --------------------------------------------------------------------------- #
# Polling intervals (seconds)
# --------------------------------------------------------------------------- #

POLL_INTERVAL_MONITOR: Final[float] = 0.5
POLL_INTERVAL_INTEGRITY: Final[float] = 30.0
POLL_INTERVAL_TAMPER: Final[float] = 1.0
POLL_INTERVAL_VULN: Final[float] = 3600.0
POLL_INTERVAL_SAFETY: Final[float] = 0.1

# --------------------------------------------------------------------------- #
# TPM PCR allocation (measured boot)
# --------------------------------------------------------------------------- #

PCR_ROM: Final[int] = 0
PCR_BOOTLOADER: Final[int] = 2
PCR_KERNEL: Final[int] = 4
PCR_INITRAMFS: Final[int] = 5
PCR_APP: Final[int] = 7
PCR_NUMBER: Final[int] = 24  # TPM 2.0 default bank size (SHA-256)

# --------------------------------------------------------------------------- #
# Crypto defaults
# --------------------------------------------------------------------------- #

AES_KEY_SIZE_BYTES: Final[int] = 32       # AES-256
AES_GCM_NONCE_SIZE: Final[int] = 12
AES_GCM_TAG_SIZE: Final[int] = 16
RSA_KEY_BITS: Final[int] = 3072
ECDSA_CURVE: Final[str] = "secp384r1"     # P-384
HASH_ALG: Final[str] = "sha256"

MAX_CAN_FRAME_BYTES: Final[int] = 8
MAX_CAN_FD_BYTES: Final[int] = 64
