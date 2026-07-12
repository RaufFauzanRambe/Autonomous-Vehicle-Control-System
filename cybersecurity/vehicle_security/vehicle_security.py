"""Top-level orchestrator for the vehicle_security subsystem.

The :class:`VehicleSecuritySystem` wires together every security sub-module
(firewall, monitor, secure boot, secure update, secure storage, firmware
validation, software integrity, safety monitor, tamper detection,
vulnerability scanner, security event logger, emergency lockdown, incident
response, diagnostics) into a single coordinated system.

A typical lifecycle:

    cfg = load_config("/etc/avcs/security/config.yaml")
    system = VehicleSecuritySystem(cfg)
    system.start()
    ...
    status = system.get_security_status()
    ...
    system.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .config import VehicleSecurityConfig
from .constants import ECU_IDS, Severity
from .diagnostics import DiagnosticsReport, HealthStatus, SecurityDiagnostics
from .emergency_lockdown import EmergencyLockdown, hash_admin_pin
from .firmware_validation import FirmwareValidator
from .incident_response import IncidentResponseManager, IncidentStatus, Playbook
from .safety_monitor import SafetyMonitor, SafetyState
from .secure_boot import SecureBootManager, SoftwareTPM
from .secure_storage import SecureStorage
from .secure_update import InMemoryTransport, SecureUpdateManager, SlotBackend, UpdateTransport
from .security_event_logger import SecurityEventLogger
from .software_integrity import SoftwareIntegrityMonitor
from .tamper_detection import TamperDetector
from .utils import format_timestamp
from .vehicle_firewall import VehicleFirewall
from .vehicle_monitor import MonitorEvent, VehicleMonitor
from .vulnerability_scanner import VulnerabilityScanner

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Status dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class SecurityStatus:
    """Snapshot of the entire vehicle security system's state."""

    running: bool = False
    started_at: float = 0.0
    uptime_seconds: float = 0.0
    subsystems_enabled: Dict[str, bool] = field(default_factory=dict)
    subsystems_healthy: Dict[str, str] = field(default_factory=dict)
    firewall_stats: Dict[str, int] = field(default_factory=dict)
    monitor_running: bool = False
    safety_state: str = SafetyState.NORMAL.value
    lockdown_active: bool = False
    lockdown_reason: str = ""
    open_incidents: int = 0
    tamper_events: int = 0
    integrity_violations: int = 0
    vuln_findings: int = 0
    last_scan: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityConfig:
    """Runtime override wrapper around :class:`VehicleSecurityConfig`."""

    config: VehicleSecurityConfig
    admin_pin: str = ""
    master_key_hex: str = ""

    @classmethod
    def from_file(cls, path: str, admin_pin: str = "") -> "SecurityConfig":
        from .config import load_config
        return cls(config=load_config(path), admin_pin=admin_pin)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class VehicleSecuritySystem:
    """Coordinates every vehicle security sub-module."""

    SUBSYSTEMS = (
        "firewall", "monitor", "secure_boot", "secure_update", "secure_storage",
        "firmware_validation", "software_integrity", "safety_monitor",
        "tamper_detection", "vulnerability_scanner", "event_logger",
        "emergency_lockdown", "incident_response", "diagnostics",
    )

    def __init__(
        self,
        config: Optional[VehicleSecurityConfig] = None,
        admin_pin: str = "",
        master_key: Optional[bytes] = None,
    ) -> None:
        self.config = config or VehicleSecurityConfig()
        self._started_at = 0.0
        self._running = False
        self._lock = threading.RLock()

        # ---- Build sub-modules ----
        self.event_logger = SecurityEventLogger(self.config.log_path)

        self.firewall = VehicleFirewall(
            can_allowlist=self.config.firewall.can_allowlist,
            can_denylist=self.config.firewall.can_denylist,
            ip_allowlist=self.config.firewall.ip_allowlist,
            ip_denylist=self.config.firewall.ip_denylist,
            blocked_ports=self.config.firewall.blocked_ports,
            rate_limit_per_second=self.config.firewall.rate_limit_per_second,
            dpi_enabled=self.config.firewall.dpi_enabled,
            stateful_tracking=self.config.firewall.stateful_tracking,
        )

        self.secure_storage = SecureStorage(
            store_path=self.config.storage.store_path,
            master_key=master_key,
            kdf_iterations=self.config.storage.kdf_iterations,
        )

        self.secure_boot = SecureBootManager(
            manifest_path=self.config.boot.manifest_path,
            tpm=SoftwareTPM(),
            logger_=self.event_logger,
        )

        self.slot_backend = SlotBackend()
        self.update_transport: UpdateTransport = InMemoryTransport(b"", {})
        self.secure_update = SecureUpdateManager(
            transport=self.update_transport,
            slot_backend=self.slot_backend,
            logger_=self.event_logger,
            max_retries=self.config.update.max_retries,
        )

        self.firmware_validator = FirmwareValidator()

        self.integrity_monitor = SoftwareIntegrityMonitor(
            baseline_db_path=self.config.baseline_db_path,
            logger_=self.event_logger,
        )

        self.safety_monitor = SafetyMonitor(logger=self.event_logger)

        self.tamper_detector = TamperDetector(logger_=self.event_logger)

        self.vuln_scanner = VulnerabilityScanner(
            cve_index_path=self.config.cve_index_path,
        )

        admin_hash = hash_admin_pin(admin_pin) if admin_pin else hash_admin_pin("default-pin-change-me")
        self.lockdown = EmergencyLockdown(
            admin_pin_hash=admin_hash,
            logger_=self.event_logger,
        )

        self.incident_response = IncidentResponseManager(
            logger_=self.event_logger,
        )
        # Wire lockdown into the incident-response playbook executor
        self.incident_response.executor.lockdown_manager = self.lockdown

        self.monitor = VehicleMonitor(
            poll_interval=self.config.monitor.monitor_interval,
            max_events_per_second=self.config.monitor.max_events_per_second,
            integrity_reader=lambda: [v.path for v in self.integrity_monitor.verify_integrity()],
        )
        self.monitor.add_callback(self._on_monitor_event)

        self.diagnostics = SecurityDiagnostics()
        self._register_diagnostics()

    # ------------------------------------------------------------------ #
    # Diagnostics registration
    # ------------------------------------------------------------------ #

    def _register_diagnostics(self) -> None:
        d = self.diagnostics
        d.register_check("firewall", "stats", lambda: _check_firewall(self.firewall))
        d.register_check("monitor", "running", lambda: _check_monitor(self.monitor))
        d.register_check("secure_boot", "attestation", lambda: _check_secure_boot(self.secure_boot))
        d.register_check("secure_storage", "backend", lambda: _check_secure_storage(self.secure_storage))
        d.register_check("secure_update", "state", lambda: _check_update(self.secure_update))
        d.register_check("firmware_validation", "ready", lambda: _check_fw_validator(self.firmware_validator))
        d.register_check("integrity", "baseline_size", lambda: _check_integrity(self.integrity_monitor))
        d.register_check("safety", "state", lambda: _check_safety(self.safety_monitor))
        d.register_check("tamper", "events", lambda: _check_tamper(self.tamper_detector))
        d.register_check("vuln_scanner", "index_size", lambda: _check_vuln(self.vuln_scanner))
        d.register_check("event_logger", "chain", lambda: _check_event_log(self.event_logger))
        d.register_check("lockdown", "state", lambda: _check_lockdown(self.lockdown))
        d.register_check("incident_response", "open_incidents", lambda: _check_incidents(self.incident_response))

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        with self._lock:
            if self._running:
                logger.warning("VehicleSecuritySystem already running")
                return
            self._running = True
            self._started_at = time.time()

        if self.config.enable_monitor:
            self.monitor.start()
        logger.info("VehicleSecuritySystem started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self.monitor.stop()
        self.event_logger.flush()
        logger.info("VehicleSecuritySystem stopped")

    # ------------------------------------------------------------------ #
    # Scans & status
    # ------------------------------------------------------------------ #

    def run_security_scan(self) -> Dict[str, Any]:
        """Run integrity + vulnerability scans; return combined results."""
        integrity_violations = []
        if self.config.enable_integrity_monitor:
            integrity_violations = [v.to_dict() for v in self.integrity_monitor.verify_integrity()]

        vuln_findings = []
        if self.config.enable_vuln_scanner:
            vuln_findings = [f.to_dict() for f in self.vuln_scanner.get_findings()]

        diag_report = self.diagnostics.run_full_diagnostics()
        return {
            "timestamp": format_timestamp(),
            "integrity_violations": integrity_violations,
            "vuln_findings": vuln_findings,
            "diagnostics": diag_report.to_dict(),
        }

    def get_security_status(self) -> SecurityStatus:
        with self._lock:
            running = self._running
            started = self._started_at
        enabled = {
            name: getattr(self.config, f"enable_{name}", True) for name in self.SUBSYSTEMS
        }
        # Map subsystem name -> config flag name (special cases)
        flag_map = {
            "firewall": "enable_firewall",
            "monitor": "enable_monitor",
            "secure_boot": "enable_secure_boot",
            "secure_update": "enable_secure_update",
            "secure_storage": "enable_secure_storage",
            "firmware_validation": "enable_firmware_validation",
            "software_integrity": "enable_integrity_monitor",
            "safety_monitor": "enable_safety_monitor",
            "tamper_detection": "enable_tamper_detection",
            "vulnerability_scanner": "enable_vuln_scanner",
            "event_logger": "enable_event_logger",
            "emergency_lockdown": "enable_emergency_lockdown",
            "incident_response": "enable_incident_response",
            "diagnostics": "enable_diagnostics",
        }
        enabled = {name: getattr(self.config, flag, True) for name, flag in flag_map.items()}

        diag = self.diagnostics.get_report()
        healthy: Dict[str, str] = {}
        if diag:
            for r in diag.get("results", []):
                healthy.setdefault(r["subsystem"], r["status"])
                healthy[r["subsystem"] + "." + r["check"]] = r["status"]

        return SecurityStatus(
            running=running,
            started_at=started,
            uptime_seconds=max(0.0, time.time() - started) if started else 0.0,
            subsystems_enabled=enabled,
            subsystems_healthy=healthy,
            firewall_stats=self.firewall.get_stats(),
            monitor_running=self.monitor.state.running,
            safety_state=self.safety_monitor.get_state().value,
            lockdown_active=self.lockdown.is_locked_down(),
            lockdown_reason=self.lockdown.get_lockdown_reason(),
            open_incidents=len(self.incident_response.list_incidents()),
            tamper_events=self.tamper_detector.get_event_count(),
            integrity_violations=len(self.integrity_monitor.get_violations()),
            vuln_findings=len(self.vuln_scanner.get_findings()),
            last_scan=format_timestamp(),
        )

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #

    def handle_security_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Central event handler: log + safety check + incident classification."""
        sev = Severity(severity)
        self.event_logger.log_event(event_type, severity, source, message, details or {})

        # Safety impact check
        if self.config.enable_safety_monitor:
            violation = self.safety_monitor.check_safety_violation(event_type, sev, source, details)
            if violation and sev in (Severity.HIGH, Severity.CRITICAL):
                self._auto_report_incident(event_type, sev, source, message, details)

        # Critical tamper -> trigger lockdown
        if (
            event_type.startswith("tamper.")
            and sev == Severity.CRITICAL
            and self.config.enable_emergency_lockdown
            and not self.lockdown.is_locked_down()
        ):
            self.lockdown.trigger_lockdown(
                reason=f"critical tamper: {message}",
                triggered_by="vehicle_security.auto",
            )

    def _on_monitor_event(self, event: MonitorEvent) -> None:
        self.handle_security_event(
            event_type=event.event_type.value,
            severity=event.severity.value,
            source=event.source,
            message=event.message,
            details=event.details,
        )

    def _auto_report_incident(
        self,
        event_type: str,
        severity: Severity,
        source: str,
        message: str,
        details: Optional[Dict[str, Any]],
    ) -> None:
        inc_sev, playbook = self.incident_response.auto_classify(event_type, message)
        inc = self.incident_response.report_incident(
            title=f"{event_type} from {source}",
            description=message,
            severity=severity,
            source=source,
            playbook=playbook,
            indicators=details or {},
        )
        if severity in (Severity.CRITICAL, Severity.HIGH):
            self.incident_response.run_playbook(inc.incident_id, only_steps=["contain"])


# --------------------------------------------------------------------------- #
# Diagnostic check helpers
# --------------------------------------------------------------------------- #


def _check_firewall(fw: VehicleFirewall):
    from .diagnostics import CheckResult, HealthStatus
    stats = fw.get_stats()
    return CheckResult(
        subsystem="firewall", check="stats", status=HealthStatus.PASS,
        details=stats, message=f"{stats['can_inspected']} CAN frames inspected",
    )


def _check_monitor(mon: VehicleMonitor):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="monitor", check="running",
        status=HealthStatus.PASS if mon.state.running else HealthStatus.WARN,
        message="running" if mon.state.running else "not running",
    )


def _check_secure_boot(sb: SecureBootManager):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="secure_boot", check="attestation",
        status=HealthStatus.PASS,
        details={"pcr_count": len(sb.get_pcr_values()), "stages": len(sb.get_stage_results())},
    )


def _check_secure_storage(store: SecureStorage):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="secure_storage", check="backend",
        status=HealthStatus.PASS if store.backend.available else HealthStatus.WARN,
        message="cryptography backend available" if store.backend.available else "fallback (insecure)",
        details={"secrets": store.count()},
    )


def _check_update(upd: SecureUpdateManager):
    from .diagnostics import CheckResult, HealthStatus
    status = upd.get_update_status()
    return CheckResult(
        subsystem="secure_update", check="state",
        status=HealthStatus.PASS if status["status"] != "failed" else HealthStatus.FAIL,
        details=status,
    )


def _check_fw_validator(fv: FirmwareValidator):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="firmware_validation", check="ready",
        status=HealthStatus.PASS,
        details={"known_bad_hashes": len(fv.known_bad_hashes), "tracked_versions": len(fv.current_versions)},
    )


def _check_integrity(im: SoftwareIntegrityMonitor):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="integrity", check="baseline_size",
        status=HealthStatus.PASS if im.get_baseline_size() > 0 else HealthStatus.WARN,
        details={"baseline_size": im.get_baseline_size(), "violations": len(im.get_violations())},
    )


def _check_safety(sm: SafetyMonitor):
    from .diagnostics import CheckResult, HealthStatus
    state = sm.get_state()
    return CheckResult(
        subsystem="safety", check="state",
        status=HealthStatus.PASS if state == SafetyState.NORMAL else HealthStatus.WARN,
        details={"state": state.value},
    )


def _check_tamper(td: TamperDetector):
    from .diagnostics import CheckResult, HealthStatus
    count = td.get_event_count()
    return CheckResult(
        subsystem="tamper", check="events",
        status=HealthStatus.PASS if count == 0 else HealthStatus.WARN,
        details={"events": count},
    )


def _check_vuln(vs: VulnerabilityScanner):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="vuln_scanner", check="index_size",
        status=HealthStatus.PASS if vs.get_index_size() > 0 else HealthStatus.WARN,
        details=vs.summary(),
    )


def _check_event_log(el: SecurityEventLogger):
    from .diagnostics import CheckResult, HealthStatus
    broken = el.verify_chain()
    return CheckResult(
        subsystem="event_logger", check="chain",
        status=HealthStatus.PASS if broken is None else HealthStatus.FAIL,
        message="chain intact" if broken is None else f"chain broken at seq {broken}",
        details={"entries": el.count()},
    )


def _check_lockdown(ld: EmergencyLockdown):
    from .diagnostics import CheckResult, HealthStatus
    return CheckResult(
        subsystem="lockdown", check="state",
        status=HealthStatus.PASS if not ld.is_locked_down() else HealthStatus.WARN,
        details=ld.get_state(),
    )


def _check_incidents(ir: IncidentResponseManager):
    from .diagnostics import CheckResult, HealthStatus
    open_count = len([i for i in ir.list_incidents()
                      if i.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE)])
    return CheckResult(
        subsystem="incident_response", check="open_incidents",
        status=HealthStatus.PASS if open_count == 0 else HealthStatus.WARN,
        details=ir.summary(),
    )
