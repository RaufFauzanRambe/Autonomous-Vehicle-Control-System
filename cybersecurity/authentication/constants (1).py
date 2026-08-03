"""
constants.py
============

Centralised enumerations and default constants for the authentication
sub-system of the Autonomous-Vehicle-Control-System.

Defining all authentication-related enumerations in a single module keeps
the rest of the package free of magic strings, makes the security policy
easy to audit, and gives downstream tooling (e.g. dashboards, audit-log
consumers) a stable vocabulary to consume.
"""

from __future__ import annotations

import enum
from datetime import timedelta


# ---------------------------------------------------------------------------
# Authentication methods
# ---------------------------------------------------------------------------
class AuthMethod(str, enum.Enum):
    """Top-level authentication methods supported by the platform."""

    PASSWORD = "password"
    PIN = "pin"
    BIOMETRIC = "biometric"
    OTP_TOTP = "otp_totp"
    HARDWARE_TOKEN = "hardware_token"  # FIDO2 / U2F
    PUSH_NOTIFICATION = "push_notification"
    X509_CERTIFICATE = "x509_certificate"
    VEHICLE_CERTIFICATE = "vehicle_certificate"  # IEEE 1609.2 SCMS
    TPM_ATTESTATION = "tpm_attestation"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    MUTUAL_TLS = "mutual_tls"


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
class Role(str, enum.Enum):
    """First-class roles in the AV platform role hierarchy."""

    DRIVER = "driver"
    PASSENGER = "passenger"
    FLEET_MANAGER = "fleet_manager"
    SERVICE_TECHNICIAN = "service_technician"
    SECURITY_ADMIN = "security_admin"
    EMERGENCY_OVERRIDE = "emergency_override"
    OTA_OPERATOR = "ota_operator"
    AUDITOR = "auditor"


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------
class Permission(str, enum.Enum):
    """Granular permissions used by the RBAC/ABAC layer.

    The string namespace is ``<domain>:<action>`` so that policies can be
    written declaratively, e.g. ``permit(role, "vehicle:control_steering")``.
    """

    # Vehicle telemetry / state
    VEHICLE_READ_TELEMETRY = "vehicle:read_telemetry"
    VEHICLE_READ_LOCATION = "vehicle:read_location"
    VEHICLE_CONTROL_STEERING = "vehicle:control_steering"
    VEHICLE_CONTROL_BRAKING = "vehicle:control_braking"
    VEHICLE_CONTROL_THROTTLE = "vehicle:control_throttle"
    VEHICLE_EMERGENCY_STOP = "vehicle:emergency_stop"
    VEHICLE_EMERGENCY_OVERRIDE = "vehicle:emergency_override"

    # OTA updates
    OTA_READ_MANIFEST = "ota:read_manifest"
    OTA_INSTALL = "ota:install"
    OTA_ROLLBACK = "ota:rollback"

    # Security / key management
    SECURITY_MANAGE_KEYS = "security:manage_keys"
    SECURITY_ROTATE_CERTS = "security:rotate_certs"
    SECURITY_REVOKE_DEVICE = "security:revoke_device"
    SECURITY_VIEW_AUDIT_LOG = "security:view_audit_log"
    SECURITY_MANAGE_RBAC = "security:manage_rbac"

    # Fleet / operational
    FLEET_VIEW_ALL = "fleet:view_all"
    FLEET_DISPATCH = "fleet:dispatch"
    FLEET_DECOMMISSION = "fleet:decommission"

    # Diagnostics / maintenance
    DIAG_READ_DTC = "diag:read_dtc"
    DIAG_CLEAR_DTC = "diag:clear_dtc"
    DIAG_FLASH_ECU = "diag:flash_ecu"


# ---------------------------------------------------------------------------
# Audit / auth events
# ---------------------------------------------------------------------------
class AuthEvent(str, enum.Enum):
    """Events recorded by the tamper-evident audit log."""

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"
    TOKEN_VALIDATION_FAILED = "token_validation_failed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    MFA_INITIATED = "mfa_initiated"
    MFA_FACTOR_VERIFIED = "mfa_factor_verified"
    MFA_FAILED = "mfa_failed"
    MFA_COMPLETE = "mfa_complete"
    BIOMETRIC_ENROLLED = "biometric_enrolled"
    BIOMETRIC_AUTHENTICATED = "biometric_authenticated"
    DEVICE_REGISTERED = "device_registered"
    DEVICE_REVOKED = "device_revoked"
    DEVICE_ATTESTED = "device_attested"
    VEHICLE_AUTHENTICATED = "vehicle_authenticated"
    VEHICLE_AUTH_FAILED = "vehicle_auth_failed"
    SESSION_CREATED = "session_created"
    SESSION_DESTROYED = "session_destroyed"
    LOCKOUT = "lockout"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET = "password_reset"


class AuthStatus(str, enum.Enum):
    """Outcome of an authentication / authorisation decision."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    CHALLENGED = "challenged"  # MFA / additional factor required
    LOCKED_OUT = "locked_out"
    EXPIRED = "expired"
    REVOKED = "revoked"


# ---------------------------------------------------------------------------
# Default TTLs and lockout policy
# ---------------------------------------------------------------------------
DEFAULT_ACCESS_TOKEN_TTL = timedelta(minutes=15)
DEFAULT_REFRESH_TOKEN_TTL = timedelta(days=7)
DEFAULT_SESSION_TTL = timedelta(hours=12)
DEFAULT_MFA_CHALLENGE_TTL = timedelta(minutes=5)
DEFAULT_OTP_WINDOW = 1  # ±1 step TOTP tolerance
DEFAULT_TOTP_STEP = 30  # seconds
DEFAULT_TOTP_DIGITS = 6

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = timedelta(minutes=15)

# Biometric match thresholds (cosine similarity)
BIOMETRIC_FACE_THRESHOLD = 0.92
BIOMETRIC_FINGERPRINT_THRESHOLD = 0.85
BIOMETRIC_VOICE_THRESHOLD = 0.80

# Token types
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"
TOKEN_TYPE_SESSION = "session"
TOKEN_TYPE_MFA = "mfa"

# Default JWT issuer / audience
DEFAULT_ISSUER = "avcs-auth"
DEFAULT_AUDIENCE = "avcs-platform"

# Bcrypt cost factor (12 ≈ ~250ms on commodity hardware)
DEFAULT_BCRYPT_ROUNDS = 12

__all__ = [
    "AuthMethod",
    "Role",
    "Permission",
    "AuthEvent",
    "AuthStatus",
    "DEFAULT_ACCESS_TOKEN_TTL",
    "DEFAULT_REFRESH_TOKEN_TTL",
    "DEFAULT_SESSION_TTL",
    "DEFAULT_MFA_CHALLENGE_TTL",
    "DEFAULT_OTP_WINDOW",
    "DEFAULT_TOTP_STEP",
    "DEFAULT_TOTP_DIGITS",
    "MAX_LOGIN_ATTEMPTS",
    "LOCKOUT_DURATION",
    "BIOMETRIC_FACE_THRESHOLD",
    "BIOMETRIC_FINGERPRINT_THRESHOLD",
    "BIOMETRIC_VOICE_THRESHOLD",
    "TOKEN_TYPE_ACCESS",
    "TOKEN_TYPE_REFRESH",
    "TOKEN_TYPE_SESSION",
    "TOKEN_TYPE_MFA",
    "DEFAULT_ISSUER",
    "DEFAULT_AUDIENCE",
    "DEFAULT_BCRYPT_ROUNDS",
]
