"""Pytest suite for the vehicle_security package.

Tests cover:
  - utility helpers (hashing, hex, CAN id parsing, retry)
  - firewall rule matching (allowlist/denylist, rate limiting, DPI)
  - secure storage encrypt/decrypt + key rotation
  - firmware validation (header parsing, version check, known-bad hash)
  - software integrity monitor (baseline, tamper detection)
  - security event logger (hash chain integrity, query)
  - emergency lockdown (trigger/release flow, PIN)
  - incident response (classification, playbook execution)
  - safety monitor (state transitions)
  - tamper detection (GPIO + accel + TPM)
  - vulnerability scanner (CVE matching, severity filtering)
  - secure boot (PCR extension, attestation)
  - diagnostics (full report, fail propagation)
  - orchestrator end-to-end

All hardware (TPM, GPIO, CAN sockets, network) is mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Make the package importable when run directly via `pytest <this file>`
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PKG_PARENT = _THIS_DIR.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from vehicle_security import (  # noqa: E402
    constants,
    utils,
)
from vehicle_security.vehicle_firewall import (  # noqa: E402
    FirewallRule, Protocol, VehicleFirewall, Verdict,
)
from vehicle_security.vehicle_monitor import (  # noqa: E402
    ECUStatus, ECUState, MonitorEvent, VehicleMonitor, CANBusStats,
)
from vehicle_security.secure_storage import (  # noqa: E402
    SecureStorage, derive_master_key,
)
from vehicle_security.firmware_validation import (  # noqa: E402
    FirmwareValidator, FirmwareHeader, FirmwareFlags, ValidationCode,
)
from vehicle_security.software_integrity import (  # noqa: E402
    SoftwareIntegrityMonitor,
)
from vehicle_security.security_event_logger import (  # noqa: E402
    SecurityEventLogger, GENESIS_HASH,
)
from vehicle_security.emergency_lockdown import (  # noqa: E402
    EmergencyLockdown, hash_admin_pin, verify_admin_pin,
)
from vehicle_security.incident_response import (  # noqa: E402
    IncidentResponseManager, IncidentStatus, Playbook, PlaybookExecutor,
)
from vehicle_security.safety_monitor import (  # noqa: E402
    SafetyMonitor, SafetyGoal, SafetyState,
)
from vehicle_security.tamper_detection import (  # noqa: E402
    TamperDetector, TamperSource, AccelerometerSample,
)
from vehicle_security.vulnerability_scanner import (  # noqa: E402
    VulnerabilityScanner, InstalledPackage, CVEEntry,
    version_in_range, severity_from_cvss,
)
from vehicle_security.secure_boot import (  # noqa: E402
    SecureBootManager, SoftwareTPM, BootStage,
)
from vehicle_security.diagnostics import (  # noqa: E402
    SecurityDiagnostics, HealthStatus, CheckResult,
)
from vehicle_security.config import (  # noqa: E402
    VehicleSecurityConfig, load_config,
)
from vehicle_security.vehicle_security import (  # noqa: E402
    VehicleSecuritySystem, SecurityStatus,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    return tmp_path / "secure_store.json"


@pytest.fixture
def tmp_log_path(tmp_path: Path) -> Path:
    return tmp_path / "events.jsonl"


@pytest.fixture
def tmp_baseline_path(tmp_path: Path) -> Path:
    return tmp_path / "baseline.json"


@pytest.fixture
def tmp_cve_index(tmp_path: Path) -> Path:
    path = tmp_path / "cve_index.json"
    path.write_text(json.dumps({
        "cves": [
            {
                "cve_id": "CVE-2024-0001",
                "package": "openssl",
                "affected_ranges": [">=1.0,<1.1.1k"],
                "fixed_version": "1.1.1k",
                "cvss_v3": 9.8,
                "severity": "critical",
                "description": "buffer overflow in openssl",
            },
            {
                "cve_id": "CVE-2024-0002",
                "package": "curl",
                "affected_ranges": [">=7.0,<7.68.0"],
                "fixed_version": "7.68.0",
                "cvss_v3": 7.5,
                "severity": "high",
                "description": "curl heap overflow",
            },
        ]
    }), encoding="utf-8")
    return path


@pytest.fixture
def secure_storage(tmp_store_path: Path) -> SecureStorage:
    # Use a fixed master key so the test is deterministic.
    key = derive_master_key(b"test-salt", b"test-password", iterations=1000)
    return SecureStorage(store_path=str(tmp_store_path), master_key=key)


# --------------------------------------------------------------------------- #
# Utility tests
# --------------------------------------------------------------------------- #


class TestUtils:
    def test_compute_sha256(self) -> None:
        h = utils.compute_sha256(b"hello")
        assert h.hex() == hashlib.sha256(b"hello").hexdigest()

    def test_compute_sha512(self) -> None:
        h = utils.compute_sha512(b"hello")
        assert h.hex() == hashlib.sha512(b"hello").hexdigest()

    def test_hex_encode_decode_roundtrip(self) -> None:
        original = b"\x00\x01\xff\x10"
        encoded = utils.hex_encode(original)
        assert utils.hex_decode(encoded) == original

    def test_hex_decode_with_0x_prefix(self) -> None:
        assert utils.hex_decode("0xDEADBEEF") == b"\xde\xad\xbe\xef"

    def test_safe_compare_equal(self) -> None:
        assert utils.safe_compare(b"abc", b"abc") is True

    def test_safe_compare_not_equal(self) -> None:
        assert utils.safe_compare(b"abc", b"abd") is False

    def test_safe_compare_wrong_type(self) -> None:
        with pytest.raises(TypeError):
            utils.safe_compare("abc", b"abc")

    def test_parse_can_id_int(self) -> None:
        assert utils.parse_can_id(0x100) == 0x100

    def test_parse_can_id_hex_string(self) -> None:
        assert utils.parse_can_id("0x7FF") == 0x7FF
        assert utils.parse_can_id("100") == 0x100

    def test_parse_can_id_invalid_string(self) -> None:
        with pytest.raises(ValueError):
            utils.parse_can_id("not-a-cid")

    def test_parse_can_id_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            utils.parse_can_id(0x20000000)

    def test_format_timestamp_iso8601(self) -> None:
        ts = utils.format_timestamp(0)
        assert ts.endswith("Z")
        assert "1970-01-01T00:00:00" in ts

    def test_retry_succeeds_after_failure(self) -> None:
        calls: List[int] = []

        @utils.retry(retries=3, delay=0)
        def flaky() -> int:
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("nope")
            return 42

        assert flaky() == 42
        assert len(calls) == 2

    def test_retry_exhausts(self) -> None:
        @utils.retry(retries=2, delay=0, exceptions=(ValueError,))
        def always_fails() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fails()


# --------------------------------------------------------------------------- #
# Firewall tests
# --------------------------------------------------------------------------- #


class TestVehicleFirewall:
    def test_denylist_blocks_can_id(self) -> None:
        fw = VehicleFirewall(can_denylist=[0x7DF])
        assert fw.inspect_can_frame(0x7DF, b"\x01\x02") == Verdict.DENY

    def test_allowlist_blocks_unlisted(self) -> None:
        fw = VehicleFirewall(can_allowlist=[0x100, 0x101])
        assert fw.inspect_can_frame(0x100, b"\x00") == Verdict.ALLOW
        assert fw.inspect_can_frame(0x200, b"\x00") == Verdict.DENY

    def test_rate_limiting(self) -> None:
        fw = VehicleFirewall(rate_limit_per_second=2)
        assert fw.inspect_can_frame(0x100, b"\x00") == Verdict.ALLOW
        assert fw.inspect_can_frame(0x100, b"\x00") == Verdict.ALLOW
        assert fw.inspect_can_frame(0x100, b"\x00") == Verdict.RATE_LIMIT

    def test_ethernet_denylist(self) -> None:
        fw = VehicleFirewall(ip_denylist=["10.0.0.99"])
        v = fw.inspect_ethernet_packet("10.0.0.99", "10.0.0.1", 1234, 80, Protocol.TCP)
        assert v == Verdict.DENY

    def test_blocked_ports(self) -> None:
        fw = VehicleFirewall(blocked_ports=[23])
        v = fw.inspect_ethernet_packet("10.0.0.1", "10.0.0.2", 50000, 23, Protocol.TCP)
        assert v == Verdict.DENY

    def test_block_source_persists(self) -> None:
        fw = VehicleFirewall()
        fw.block_source("10.1.1.1")
        blocked = fw.get_blocked_sources()
        assert "10.1.1.1" in blocked

    def test_block_source_expires(self) -> None:
        fw = VehicleFirewall()
        fw.block_source("10.1.1.1", duration=0.0)
        # immediately expired
        assert "10.1.1.1" not in fw.get_blocked_sources()

    def test_dpi_hook_denies(self) -> None:
        from vehicle_security.vehicle_firewall import DPIResult
        fw = VehicleFirewall(can_allowlist=[], dpi_enabled=True)
        fw.add_dpi_hook(lambda payload: DPIResult(verdict=Verdict.DENY, reason="bad"))
        assert fw.inspect_can_frame(0x100, b"\x00") == Verdict.DENY

    def test_add_remove_rule(self) -> None:
        fw = VehicleFirewall(can_allowlist=[])  # no allowlist so rule can fire
        rule = FirewallRule(verdict=Verdict.DENY, can_id=0x200)
        fw.add_rule(rule)
        assert fw.inspect_can_frame(0x200, b"\x00") == Verdict.DENY
        assert fw.remove_rule(rule) is True
        assert fw.inspect_can_frame(0x200, b"\x00") == Verdict.ALLOW

    def test_stateful_connection_tracking(self) -> None:
        fw = VehicleFirewall(stateful_tracking=True)
        fw.inspect_ethernet_packet("1.1.1.1", "2.2.2.2", 1000, 80, Protocol.TCP, b"GET / HTTP/1.1")
        assert fw.get_connection_count() == 1


# --------------------------------------------------------------------------- #
# Secure storage tests
# --------------------------------------------------------------------------- #


class TestSecureStorage:
    def test_store_and_retrieve(self, secure_storage: SecureStorage) -> None:
        secure_storage.store("api_key", b"super-secret-value")
        assert secure_storage.retrieve("api_key") == b"super-secret-value"

    def test_retrieve_missing_raises(self, secure_storage: SecureStorage) -> None:
        with pytest.raises(KeyError):
            secure_storage.retrieve("nonexistent")

    def test_delete(self, secure_storage: SecureStorage) -> None:
        secure_storage.store("k", b"v")
        assert secure_storage.delete("k") is True
        assert secure_storage.delete("k") is False

    def test_list_keys_sorted(self, secure_storage: SecureStorage) -> None:
        secure_storage.store("b", b"1")
        secure_storage.store("a", b"2")
        secure_storage.store("c", b"3")
        assert secure_storage.list_keys() == ["a", "b", "c"]

    def test_persistence_across_instances(self, tmp_store_path: Path) -> None:
        key = derive_master_key(b"salt", b"pw", iterations=1000)
        s1 = SecureStorage(store_path=str(tmp_store_path), master_key=key)
        s1.store("token", b"abc123")
        s2 = SecureStorage(store_path=str(tmp_store_path), master_key=key)
        assert s2.retrieve("token") == b"abc123"

    def test_rotate_master_key(self, secure_storage: SecureStorage) -> None:
        secure_storage.store("k1", b"v1")
        secure_storage.store("k2", b"v2")
        new_key = os.urandom(32)
        count = secure_storage.rotate_master_key(new_key)
        assert count == 2
        # Old key can no longer decrypt; new key can (handled internally)
        assert secure_storage.retrieve("k1") == b"v1"
        assert secure_storage.retrieve("k2") == b"v2"

    def test_store_rejects_empty_key(self, secure_storage: SecureStorage) -> None:
        with pytest.raises(ValueError):
            secure_storage.store("", b"v")


# --------------------------------------------------------------------------- #
# Firmware validation tests
# --------------------------------------------------------------------------- #


class TestFirmwareValidator:
    def test_parse_header_roundtrip(self) -> None:
        payload = b"hello firmware payload"
        image = FirmwareValidator.build_image(
            target_ecu=0x02, version=(1, 2, 3), payload=payload,
        )
        header = FirmwareValidator.parse_header(image)
        assert header.target_ecu == 0x02
        assert header.version_str == "1.2.3"
        assert header.payload_size == len(payload)
        assert header.is_signed is True

    def test_validate_image_ok(self) -> None:
        validator = FirmwareValidator()
        payload = b"some payload"
        image = FirmwareValidator.build_image(
            target_ecu=0x02, version=(1, 0, 0), payload=payload,
        )
        result = validator.validate_image(image, require_signature=False)
        assert result.success
        assert result.code == ValidationCode.OK

    def test_validate_image_bad_magic(self) -> None:
        bad = b"BADMAGIC" + b"\x00" * 100
        validator = FirmwareValidator()
        result = validator.validate_image(bad, require_signature=False)
        assert result.code == ValidationCode.HEADER_INVALID

    def test_version_check_blocks_downgrade(self) -> None:
        validator = FirmwareValidator()
        validator.set_current_version(0x02, 2, 0, 0)
        assert validator.check_version(0x02, 1, 0, 0) is False
        assert validator.check_version(0x02, 2, 0, 0) is True
        assert validator.check_version(0x02, 3, 0, 0) is True

    def test_known_bad_hash_detected(self) -> None:
        validator = FirmwareValidator()
        payload = b"bad payload"
        validator.add_known_bad_hash(hashlib.sha256(payload).hexdigest())
        image = FirmwareValidator.build_image(
            target_ecu=0x02, version=(1, 0, 0), payload=payload,
        )
        result = validator.validate_image(image, require_signature=False)
        assert result.code == ValidationCode.KNOWN_BAD_HASH
        assert not result.hash_scan_ok


# --------------------------------------------------------------------------- #
# Software integrity tests
# --------------------------------------------------------------------------- #


class TestSoftwareIntegrity:
    def test_build_baseline_and_verify(self, tmp_baseline_path: Path, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"contents-a")
        f2.write_bytes(b"contents-b")
        mon = SoftwareIntegrityMonitor(baseline_db_path=str(tmp_baseline_path))
        assert mon.add_file(str(f1)) is True
        assert mon.add_file(str(f2)) is True
        assert mon.get_baseline_size() == 2
        assert mon.verify_integrity() == []

    def test_detects_modification(self, tmp_baseline_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"original")
        mon = SoftwareIntegrityMonitor(baseline_db_path=str(tmp_baseline_path))
        mon.add_file(str(f))
        # Modify the file
        f.write_bytes(b"tampered")
        violations = mon.verify_integrity()
        assert len(violations) == 1
        assert violations[0].kind == "modified"
        assert violations[0].path == str(f)

    def test_detects_missing_file(self, tmp_baseline_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "gone.bin"
        f.write_bytes(b"temp")
        mon = SoftwareIntegrityMonitor(baseline_db_path=str(tmp_baseline_path))
        mon.add_file(str(f))
        f.unlink()
        violations = mon.verify_integrity()
        assert len(violations) == 1
        assert violations[0].kind == "missing"

    def test_persistence_with_hmac(self, tmp_baseline_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"data")
        mon = SoftwareIntegrityMonitor(baseline_db_path=str(tmp_baseline_path))
        mon.add_file(str(f))
        mon._save()
        assert mon.verify_baseline_signature() is True
        # Reload via new instance
        mon2 = SoftwareIntegrityMonitor(baseline_db_path=str(tmp_baseline_path))
        assert mon2.get_baseline_size() == 1

    def test_remove_file(self, tmp_baseline_path: Path, tmp_path: Path) -> None:
        f = tmp_path / "a.bin"
        f.write_bytes(b"data")
        mon = SoftwareIntegrityMonitor(baseline_db_path=str(tmp_baseline_path))
        mon.add_file(str(f))
        assert mon.remove_file(str(f)) is True
        assert mon.remove_file(str(f)) is False


# --------------------------------------------------------------------------- #
# Event logger tests
# --------------------------------------------------------------------------- #


class TestSecurityEventLogger:
    def test_log_and_count(self, tmp_log_path: Path) -> None:
        log = SecurityEventLogger(str(tmp_log_path))
        log.log_event("test.event", "info", "unit-test", "hello")
        log.flush()
        assert log.count() == 1

    def test_chain_intact(self, tmp_log_path: Path) -> None:
        log = SecurityEventLogger(str(tmp_log_path))
        for i in range(5):
            log.log_event("test.event", "info", "unit-test", f"msg-{i}")
        log.flush()
        assert log.verify_chain() is None

    def test_chain_broken_by_tampering(self, tmp_log_path: Path) -> None:
        log = SecurityEventLogger(str(tmp_log_path))
        log.log_event("test.event", "info", "unit-test", "msg1")
        log.log_event("test.event", "info", "unit-test", "msg2")
        log.flush()
        # Tamper: rewrite the file with a corrupted message
        with open(tmp_log_path, "r") as fh:
            lines = fh.readlines()
        first = json.loads(lines[0])
        first["message"] = "TAMPERED"
        lines[0] = json.dumps(first) + "\n"
        with open(tmp_log_path, "w") as fh:
            fh.writelines(lines)
        broken = log.verify_chain()
        assert broken == 1  # second entry's prev_hash no longer matches

    def test_query_filter(self, tmp_log_path: Path) -> None:
        log = SecurityEventLogger(str(tmp_log_path))
        log.log_event("a.b", "info", "src1", "m1")
        log.log_event("a.b", "medium", "src2", "m2")
        log.log_event("x.y", "critical", "src1", "m3")
        log.flush()
        results = log.query(event_type="a.b")
        assert len(results) == 2
        results = log.query(source="src1")
        assert len(results) == 2

    def test_export_log(self, tmp_log_path: Path, tmp_path: Path) -> None:
        log = SecurityEventLogger(str(tmp_log_path))
        for i in range(3):
            log.log_event("test.event", "info", "src", f"m{i}")
        log.flush()
        export_path = tmp_path / "export.jsonl"
        count = log.export_log(str(export_path))
        assert count == 3
        assert export_path.exists()

    def test_invalid_severity_rejected(self, tmp_log_path: Path) -> None:
        log = SecurityEventLogger(str(tmp_log_path))
        with pytest.raises(ValueError):
            log.log_event("x.y", "bogus", "src", "m")


# --------------------------------------------------------------------------- #
# Lockdown tests
# --------------------------------------------------------------------------- #


class TestEmergencyLockdown:
    def test_trigger_and_release(self, tmp_log_path: Path) -> None:
        logger_ = SecurityEventLogger(str(tmp_log_path))
        pin_hash = hash_admin_pin("1234")
        actions: List[str] = []
        ld = EmergencyLockdown(
            admin_pin_hash=pin_hash,
            action_callback=lambda scope, enable: actions.append(f"{'+' if enable else '-'}{scope}") or True,
            logger_=logger_,
        )
        assert ld.is_locked_down() is False
        ld.trigger_lockdown(reason="test", triggered_by="tester")
        assert ld.is_locked_down() is True
        assert ld.get_lockdown_reason() == "test"
        # Wrong PIN
        assert ld.release_lockdown("0000") is False
        assert ld.is_locked_down() is True
        # Correct PIN
        assert ld.release_lockdown("1234") is True
        assert ld.is_locked_down() is False

    def test_double_trigger_ignored(self) -> None:
        ld = EmergencyLockdown(admin_pin_hash=hash_admin_pin("p"))
        ld.trigger_lockdown("r1")
        result = ld.trigger_lockdown("r2")
        assert result is False
        assert ld.get_lockdown_reason() == "r1"

    def test_verify_admin_pin(self) -> None:
        h = hash_admin_pin("secret")
        assert verify_admin_pin("secret", h) is True
        assert verify_admin_pin("wrong", h) is False


# --------------------------------------------------------------------------- #
# Incident response tests
# --------------------------------------------------------------------------- #


class TestIncidentResponse:
    def test_report_and_list(self, tmp_log_path: Path) -> None:
        logger_ = SecurityEventLogger(str(tmp_log_path))
        ir = IncidentResponseManager(logger_=logger_)
        inc = ir.report_incident(
            title="test incident", description="desc",
            severity=constants.Severity.MEDIUM, source="tester",
        )
        assert inc.incident_id.startswith("INC-")
        assert len(ir.list_incidents()) == 1
        fetched = ir.get_incident(inc.incident_id)
        assert fetched is not None
        assert fetched.title == "test incident"

    def test_auto_classify(self, tmp_log_path: Path) -> None:
        ir = IncidentResponseManager()
        sev, pb = ir.auto_classify("CAN injection detected", "brake-by-wire CAN injection")
        assert sev == constants.Severity.CRITICAL
        assert pb == Playbook.CAN_BUS_INJECTION

    def test_run_playbook_default(self, tmp_log_path: Path) -> None:
        logger_ = SecurityEventLogger(str(tmp_log_path))
        ir = IncidentResponseManager(logger_=logger_)
        inc = ir.report_incident(
            title="low severity thing", description="some issue",
            severity=constants.Severity.LOW, source="test",
            playbook=Playbook.DEFAULT,
        )
        ir.run_playbook(inc.incident_id)
        inc2 = ir.get_incident(inc.incident_id)
        assert inc2 is not None
        assert inc2.status == IncidentStatus.RESOLVED
        assert len(inc2.steps) == 4  # contain, eradicate, recover, lessons_learned

    def test_close_incident(self, tmp_log_path: Path) -> None:
        ir = IncidentResponseManager()
        inc = ir.report_incident("t", "d", constants.Severity.LOW, "src")
        ir.close_incident(inc.incident_id)
        assert ir.get_incident(inc.incident_id).status == IncidentStatus.CLOSED

    def test_lockdown_triggered_on_critical(self, tmp_log_path: Path) -> None:
        logger_ = SecurityEventLogger(str(tmp_log_path))
        ld = EmergencyLockdown(admin_pin_hash=hash_admin_pin("p"), logger_=logger_)
        ir = IncidentResponseManager(logger_=logger_)
        ir.executor.lockdown_manager = ld
        inc = ir.report_incident(
            title="malware", description="malware detected",
            severity=constants.Severity.CRITICAL, source="test",
            playbook=Playbook.MALWARE,
        )
        ir.run_playbook(inc.incident_id, only_steps=["contain"])
        assert ld.is_locked_down() is True


# --------------------------------------------------------------------------- #
# Safety monitor tests
# --------------------------------------------------------------------------- #


class TestSafetyMonitor:
    def test_default_goals_registered(self) -> None:
        sm = SafetyMonitor()
        goals = sm.get_goals()
        ids = {g.goal_id for g in goals}
        assert constants.SAFETY_GOAL_BRAKING in ids
        assert constants.SAFETY_GOAL_STEERING in ids

    def test_violation_triggers_degradation(self) -> None:
        sm = SafetyMonitor()
        initial = sm.get_state()
        assert initial == SafetyState.NORMAL
        v = sm.check_safety_violation(
            event_type="can.anomaly",
            severity=constants.Severity.CRITICAL,
            source="BRAKES",
            details={"can_id": 0x100},
        )
        assert v is not None
        assert sm.get_state() in (SafetyState.LIMP_HOME, SafetyState.FAIL_SAFE)

    def test_trigger_safe_state(self) -> None:
        sm = SafetyMonitor()
        sm.trigger_safe_state("manual")
        assert sm.get_state() == SafetyState.FAIL_SAFE

    def test_reset_requires_satisfied_goals(self) -> None:
        sm = SafetyMonitor()
        sm.check_safety_violation("x", constants.Severity.HIGH, "BRAKES", {})
        assert sm.reset_to_normal() is False  # goals unsatisfied


# --------------------------------------------------------------------------- #
# Tamper detection tests
# --------------------------------------------------------------------------- #


class TestTamperDetector:
    def test_intrusion_switch_triggers(self) -> None:
        state = {"chassis_intrusion": False, "case_open": False, "seal_intact": True}

        def reader():
            return dict(state)

        td = TamperDetector(gpio_reader=reader)
        td.check_intrusion_switches()  # baseline
        state["chassis_intrusion"] = True
        ev = td.check_intrusion_switches()
        assert ev is not None
        assert ev.source == TamperSource.INTRUSION_SWITCH
        assert td.get_event_count() == 1

    def test_accel_magnitude_spike(self) -> None:
        sample_iter = iter([
            AccelerometerSample(time.time(), 0.0, 0.0, 9.81),
            AccelerometerSample(time.time(), 0.0, 0.0, 50.0),  # spike
        ])

        def reader():
            return next(sample_iter)

        td = TamperDetector(accel_reader=reader, accel_magnitude_threshold=20.0)
        td.check_accel_anomaly()  # normal sample
        ev = td.check_accel_anomaly()
        assert ev is not None
        assert ev.source == TamperSource.ACCEL_ANOMALY

    def test_tpm_counter_increment(self) -> None:
        counters = {0: 0, 1: 0}

        def reader():
            return dict(counters)

        td = TamperDetector(tpm_counter_reader=reader)
        td.check_tpm_counters()
        counters[1] = 5
        ev = td.check_tpm_counters()
        assert ev is not None
        assert ev.source == TamperSource.TPM_COUNTER

    def test_manual_tamper(self) -> None:
        td = TamperDetector()
        ev = td.manual_tamper("operator reported")
        assert ev.source == TamperSource.MANUAL
        assert td.get_event_count() == 1


# --------------------------------------------------------------------------- #
# Vulnerability scanner tests
# --------------------------------------------------------------------------- #


class TestVulnerabilityScanner:
    def test_version_in_range(self) -> None:
        assert version_in_range("1.1.0", ">=1.0,<1.2.5") is True
        assert version_in_range("1.2.5", ">=1.0,<1.2.5") is False
        assert version_in_range("2.0.0", ">=2.0,<2.0.3") is True
        assert version_in_range("2.0.5", ">=2.0,<2.0.3") is False

    def test_severity_from_cvss(self) -> None:
        assert severity_from_cvss(9.5) == constants.Severity.CRITICAL
        assert severity_from_cvss(7.0) == constants.Severity.HIGH
        assert severity_from_cvss(5.0) == constants.Severity.MEDIUM
        assert severity_from_cvss(2.0) == constants.Severity.LOW

    def test_scan_packages(self, tmp_cve_index: Path) -> None:
        scanner = VulnerabilityScanner(cve_index_path=str(tmp_cve_index))
        pkgs = [
            InstalledPackage(name="openssl", version="1.1.0"),
            InstalledPackage(name="curl", version="7.50.0"),
            InstalledPackage(name="curl", version="7.80.0"),  # patched
        ]
        findings = scanner.scan_packages(pkgs)
        assert len(findings) == 2
        cves = {f.cve_id for f in findings}
        assert "CVE-2024-0001" in cves
        assert "CVE-2024-0002" in cves

    def test_severity_filter(self, tmp_cve_index: Path) -> None:
        scanner = VulnerabilityScanner(cve_index_path=str(tmp_cve_index))
        scanner.scan_packages([InstalledPackage(name="openssl", version="1.1.0"),
                               InstalledPackage(name="curl", version="7.50.0")])
        critical = scanner.severity_filter(constants.Severity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].cve_id == "CVE-2024-0001"

    def test_scan_configs_finds_issues(self, tmp_cve_index: Path) -> None:
        scanner = VulnerabilityScanner(cve_index_path=str(tmp_cve_index))
        findings = scanner.scan_configs([
            ("/etc/ssh/sshd_config", {"permit_root_login": "yes", "ssl_min_version": "TLSv1.3"}),
            ("/etc/telnet.conf", {"telnet_enabled": True, "ssl_min_version": "TLSv1.3"}),
        ])
        assert len(findings) == 2

    def test_update_cve_index(self, tmp_cve_index: Path) -> None:
        scanner = VulnerabilityScanner(cve_index_path=str(tmp_cve_index))
        original_size = scanner.get_index_size()
        new_data = {"cves": [{"cve_id": "CVE-2025-X", "package": "x", "affected_ranges": [">=1.0"],
                              "fixed_version": "1.1", "cvss_v3": 5.0, "severity": "medium",
                              "description": "test"}]}
        scanner.update_cve_index(new_data)
        assert scanner.get_index_size() == 1
        assert original_size != scanner.get_index_size()


# --------------------------------------------------------------------------- #
# Secure boot tests
# --------------------------------------------------------------------------- #


class TestSecureBoot:
    def test_pcr_extension_deterministic(self) -> None:
        tpm = SoftwareTPM()
        h1 = tpm.extend(0, b"hello")
        tpm2 = SoftwareTPM()
        h2 = tpm2.extend(0, b"hello")
        assert h1 == h2

    def test_pcr_extension_changes_value(self) -> None:
        tpm = SoftwareTPM()
        before = tpm.read(0)
        tpm.extend(0, b"hello")
        after = tpm.read(0)
        assert before != after

    def test_quote_deterministic(self) -> None:
        tpm = SoftwareTPM()
        tpm.extend(0, b"x")
        q1 = tpm.quote([0, 1], b"nonce")
        tpm2 = SoftwareTPM()
        tpm2.extend(0, b"x")
        q2 = tpm2.quote([0, 1], b"nonce")
        assert q1 == q2

    def test_verify_boot_stage_extends_pcr(self, tmp_path: Path) -> None:
        # No manifest file -> permissive mode
        sb = SecureBootManager(manifest_path=str(tmp_path / "missing.json"))
        before = sb.tpm.read(constants.PCR_KERNEL)
        sb.verify_boot_stage(BootStage.KERNEL, b"kernel image")
        after = sb.tpm.read(constants.PCR_KERNEL)
        assert before != after

    def test_attestation_report(self, tmp_path: Path) -> None:
        sb = SecureBootManager(manifest_path=str(tmp_path / "missing.json"))
        sb.verify_boot_stage(BootStage.ROM, b"rom")
        sb.verify_boot_stage(BootStage.BOOTLOADER, b"bootloader")
        report = sb.attest_state()
        assert report.overall_success is True
        assert len(report.stage_results) == 2
        assert len(report.pcr_values) == constants.PCR_NUMBER


# --------------------------------------------------------------------------- #
# Diagnostics tests
# --------------------------------------------------------------------------- #


class TestDiagnostics:
    def test_pass_status(self) -> None:
        d = SecurityDiagnostics()
        d.register_check("x", "always_ok", lambda: CheckResult("x", "ok", HealthStatus.PASS))
        report = d.run_full_diagnostics()
        assert report.overall_status == HealthStatus.PASS

    def test_fail_propagates(self) -> None:
        d = SecurityDiagnostics()
        d.register_check("x", "ok", lambda: CheckResult("x", "ok", HealthStatus.PASS))
        d.register_check("x", "bad", lambda: CheckResult("x", "bad", HealthStatus.FAIL))
        report = d.run_full_diagnostics()
        assert report.overall_status == HealthStatus.FAIL

    def test_warn_does_not_fail(self) -> None:
        d = SecurityDiagnostics()
        d.register_check("x", "ok", lambda: CheckResult("x", "ok", HealthStatus.PASS))
        d.register_check("x", "warn", lambda: CheckResult("x", "warn", HealthStatus.WARN))
        report = d.run_full_diagnostics()
        assert report.overall_status == HealthStatus.WARN

    def test_export_report(self, tmp_path: Path) -> None:
        d = SecurityDiagnostics()
        d.register_check("x", "ok", lambda: CheckResult("x", "ok", HealthStatus.PASS))
        d.run_full_diagnostics()
        out = tmp_path / "diag.json"
        n = d.export_report(str(out))
        assert n == 1
        assert out.exists()


# --------------------------------------------------------------------------- #
# Config tests
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_defaults(self) -> None:
        cfg = VehicleSecurityConfig()
        assert cfg.enable_firewall is True
        assert cfg.firewall.rate_limit_per_second == 500

    def test_load_config_missing_file(self) -> None:
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.enable_firewall is True

    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "enable_firewall: false\n"
            "firewall:\n"
            "  rate_limit_per_second: 100\n",
            encoding="utf-8",
        )
        cfg = load_config(str(yaml_path))
        assert cfg.enable_firewall is False
        assert cfg.firewall.rate_limit_per_second == 100


# --------------------------------------------------------------------------- #
# Orchestrator tests
# --------------------------------------------------------------------------- #


class TestVehicleSecuritySystem:
    def test_instantiation(self, tmp_path: Path) -> None:
        cfg = VehicleSecurityConfig()
        cfg.log_path = str(tmp_path / "log.jsonl")
        cfg.storage.store_path = str(tmp_path / "store.json")
        cfg.baseline_db_path = str(tmp_path / "baseline.json")
        cfg.cve_index_path = str(tmp_path / "cve.json")
        cfg.boot.manifest_path = str(tmp_path / "missing.json")
        system = VehicleSecuritySystem(config=cfg, admin_pin="test-pin")
        assert system.firewall is not None
        assert system.event_logger is not None
        assert system.safety_monitor is not None

    def test_start_stop(self, tmp_path: Path) -> None:
        cfg = VehicleSecurityConfig()
        cfg.log_path = str(tmp_path / "log.jsonl")
        cfg.storage.store_path = str(tmp_path / "store.json")
        cfg.baseline_db_path = str(tmp_path / "baseline.json")
        cfg.cve_index_path = str(tmp_path / "cve.json")
        cfg.boot.manifest_path = str(tmp_path / "missing.json")
        cfg.monitor.monitor_interval = 0.05
        system = VehicleSecuritySystem(config=cfg, admin_pin="test-pin")
        system.start()
        time.sleep(0.1)
        assert system.monitor.state.running is True
        system.stop()
        assert system.monitor.state.running is False

    def test_get_security_status(self, tmp_path: Path) -> None:
        cfg = VehicleSecurityConfig()
        cfg.log_path = str(tmp_path / "log.jsonl")
        cfg.storage.store_path = str(tmp_path / "store.json")
        cfg.baseline_db_path = str(tmp_path / "baseline.json")
        cfg.cve_index_path = str(tmp_path / "cve.json")
        cfg.boot.manifest_path = str(tmp_path / "missing.json")
        system = VehicleSecuritySystem(config=cfg, admin_pin="test-pin")
        status = system.get_security_status()
        assert isinstance(status, SecurityStatus)
        d = status.to_dict()
        assert "firewall_stats" in d
        assert d["safety_state"] == SafetyState.NORMAL.value

    def test_handle_security_event_logs(self, tmp_path: Path) -> None:
        cfg = VehicleSecurityConfig()
        cfg.log_path = str(tmp_path / "log.jsonl")
        cfg.storage.store_path = str(tmp_path / "store.json")
        cfg.baseline_db_path = str(tmp_path / "baseline.json")
        cfg.cve_index_path = str(tmp_path / "cve.json")
        cfg.boot.manifest_path = str(tmp_path / "missing.json")
        system = VehicleSecuritySystem(config=cfg, admin_pin="test-pin")
        system.handle_security_event("test.event", "info", "tester", "hello", {"k": "v"})
        system.event_logger.flush()
        entries = system.event_logger.query(event_type="test.event")
        assert len(entries) == 1

    def test_critical_tamper_triggers_lockdown(self, tmp_path: Path) -> None:
        cfg = VehicleSecurityConfig()
        cfg.log_path = str(tmp_path / "log.jsonl")
        cfg.storage.store_path = str(tmp_path / "store.json")
        cfg.baseline_db_path = str(tmp_path / "baseline.json")
        cfg.cve_index_path = str(tmp_path / "cve.json")
        cfg.boot.manifest_path = str(tmp_path / "missing.json")
        system = VehicleSecuritySystem(config=cfg, admin_pin="test-pin")
        system.handle_security_event(
            "tamper.intrusion_switch", "critical", "tamper_detector",
            "chassis intrusion detected", {},
        )
        assert system.lockdown.is_locked_down() is True

    def test_run_security_scan(self, tmp_path: Path) -> None:
        cfg = VehicleSecurityConfig()
        cfg.log_path = str(tmp_path / "log.jsonl")
        cfg.storage.store_path = str(tmp_path / "store.json")
        cfg.baseline_db_path = str(tmp_path / "baseline.json")
        cfg.cve_index_path = str(tmp_path / "cve.json")
        cfg.boot.manifest_path = str(tmp_path / "missing.json")
        system = VehicleSecuritySystem(config=cfg, admin_pin="test-pin")
        result = system.run_security_scan()
        assert "integrity_violations" in result
        assert "vuln_findings" in result
        assert "diagnostics" in result
        assert "summary" in result["diagnostics"]
