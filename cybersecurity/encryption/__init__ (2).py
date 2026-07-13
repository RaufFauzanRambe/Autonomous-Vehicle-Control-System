"""vehicle_security package.

A comprehensive cybersecurity subsystem for an Autonomous-Vehicle-Control-System.

The package exposes the orchestrator :class:`VehicleSecuritySystem` plus every
sub-module's primary class for direct use:

    from vehicle_security.vehicle_security import VehicleSecuritySystem
    from vehicle_security.vehicle_firewall import VehicleFirewall
"""

from __future__ import annotations

from .constants import MODULE_VERSION, Severity, EventType

__version__ = MODULE_VERSION

__all__ = [
    "MODULE_VERSION",
    "Severity",
    "EventType",
    "VehicleSecuritySystem",
    "VehicleFirewall",
    "VehicleMonitor",
    "SecureBootManager",
    "SecureUpdateManager",
    "SecureStorage",
    "FirmwareValidator",
    "SoftwareIntegrityMonitor",
    "SafetyMonitor",
    "TamperDetector",
    "VulnerabilityScanner",
    "SecurityEventLogger",
    "EmergencyLockdown",
    "IncidentResponseManager",
    "SecurityDiagnostics",
    "VehicleSecurityConfig",
    "load_config",
]

# Lazy imports to avoid pulling in optional deps at package import time.
def __getattr__(name: str):  # PEP 562
    if name == "VehicleSecuritySystem":
        from .vehicle_security import VehicleSecuritySystem
        return VehicleSecuritySystem
    if name == "VehicleFirewall":
        from .vehicle_firewall import VehicleFirewall
        return VehicleFirewall
    if name == "VehicleMonitor":
        from .vehicle_monitor import VehicleMonitor
        return VehicleMonitor
    if name == "SecureBootManager":
        from .secure_boot import SecureBootManager
        return SecureBootManager
    if name == "SecureUpdateManager":
        from .secure_update import SecureUpdateManager
        return SecureUpdateManager
    if name == "SecureStorage":
        from .secure_storage import SecureStorage
        return SecureStorage
    if name == "FirmwareValidator":
        from .firmware_validation import FirmwareValidator
        return FirmwareValidator
    if name == "SoftwareIntegrityMonitor":
        from .software_integrity import SoftwareIntegrityMonitor
        return SoftwareIntegrityMonitor
    if name == "SafetyMonitor":
        from .safety_monitor import SafetyMonitor
        return SafetyMonitor
    if name == "TamperDetector":
        from .tamper_detection import TamperDetector
        return TamperDetector
    if name == "VulnerabilityScanner":
        from .vulnerability_scanner import VulnerabilityScanner
        return VulnerabilityScanner
    if name == "SecurityEventLogger":
        from .security_event_logger import SecurityEventLogger
        return SecurityEventLogger
    if name == "EmergencyLockdown":
        from .emergency_lockdown import EmergencyLockdown
        return EmergencyLockdown
    if name == "IncidentResponseManager":
        from .incident_response import IncidentResponseManager
        return IncidentResponseManager
    if name == "SecurityDiagnostics":
        from .diagnostics import SecurityDiagnostics
        return SecurityDiagnostics
    if name == "VehicleSecurityConfig":
        from .config import VehicleSecurityConfig
        return VehicleSecurityConfig
    if name == "load_config":
        from .config import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
