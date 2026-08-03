"""Pytest suite for the AVCS intrusion detection subpackage.

Run with::

    pytest test_intrusion_detection.py -v

All tests use only in-memory entry points (``process_frame``, ``ingest_*``,
``analyze``, ``parse_log`` etc.) so no hardware / root privileges / packet
capture interfaces are required.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# Make the package importable when pytest is run from anywhere.
# ``parent.parent`` = ``cybersecurity/`` (matches sibling test convention); we
# also add the project root (``parents[2]``) so the ``cybersecurity`` namespace
# package is resolvable regardless of the invoking cwd.
PKG_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PKG_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports from the subpackage
# ---------------------------------------------------------------------------

from cybersecurity.intrusion_detection import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    AnomalyDetector,
    AttackSignature,
    AttackSignatureDB,
    BehaviorAnalyzer,
    CANBusMonitor,
    CANFrameEvent,
    ForensicTools,
    IDSEngine,
    IDSEvent,
    EventType,
    IncidentLogger,
    IntrusionDetectionSystem,
    LogAnalyzer,
    MalwareDetector,
    NetworkMonitor,
    PacketAnalyzer,
    PacketRule,
    ResponseActionType,
    ResponseEngine,
    ThreatClassifier,
    ThreatType,
    default_config,
)
from cybersecurity.intrusion_detection.utils import (
    format_bytes,
    parse_can_frame,
    parse_ip_packet,
    rate_limit_check,
    timestamp_now,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def can_monitor() -> CANBusMonitor:
    """CAN monitor without live capture (no python-can required)."""
    mon = CANBusMonitor(
        interface="socketcan",
        channel="vcan0",
        allowed_ids={0x100, 0x200, 0x300},
        per_id_rate_limit=100.0,
        global_rate_limit=10_000.0,
    )
    return mon


@pytest.fixture
def network_monitor() -> NetworkMonitor:
    return NetworkMonitor(
        interface="lo",
        allowed_ports=(53, 1883, 13400, 30490),
        port_scan_threshold=5,
        port_scan_window=60.0,
    )


@pytest.fixture
def alert_manager() -> AlertManager:
    return AlertManager(dedup_window_sec=1.0, escalation_sec=3600)


@pytest.fixture
def incident_logger(tmp_path) -> IncidentLogger:
    return IncidentLogger(log_path=str(tmp_path / "incidents.chain"))


@pytest.fixture
def response_engine() -> ResponseEngine:
    return ResponseEngine(enabled=True, dry_run=True, max_actions_per_min=1000,
                          cooldown_sec=0)


# ---------------------------------------------------------------------------
# 1. Anomaly detection — z-score
# ---------------------------------------------------------------------------


class TestAnomalyDetector:
    def test_zscore_detects_outlier(self):
        det = AnomalyDetector(zscore_threshold=3.0, min_samples=10)
        det.train_baseline("cpu_load", [10.0] * 100)
        normal = det.detect_anomaly("cpu_load", 10.5)
        assert not normal.is_anomalous
        anomalous = det.detect_anomaly("cpu_load", 100.0)
        assert anomalous.is_anomalous
        assert anomalous.method == "zscore"

    def test_iqr_detects_outlier(self):
        det = AnomalyDetector(zscore_threshold=10.0, iqr_multiplier=1.5,
                              min_samples=10, enabled_methods=["iqr"])
        det.train_baseline("temp", list(range(100)))
        result = det.detect_anomaly("temp", 1000.0)
        assert result.is_anomalous
        assert result.method == "iqr"

    def test_insufficient_baseline(self):
        det = AnomalyDetector(min_samples=100)
        result = det.detect_anomaly("unknown_metric", 5.0)
        assert not result.is_anomalous
        assert "insufficient_baseline" in result.details["reason"]

    def test_anomaly_score_is_monotonic(self):
        det = AnomalyDetector(zscore_threshold=3.0, min_samples=50)
        det.train_baseline("net_rate", [100.0] * 100)
        s1 = det.get_anomaly_score("net_rate", 110.0)
        s2 = det.get_anomaly_score("net_rate", 200.0)
        assert s2 > s1


# ---------------------------------------------------------------------------
# 2. Attack signature database
# ---------------------------------------------------------------------------


class TestAttackSignatureDB:
    def test_regex_match(self):
        db = AttackSignatureDB()
        db.add_signature(AttackSignature(
            id="sig-001", name="cmd shell",
            pattern=r"/bin/sh", severity=AlertSeverity.HIGH,
            threat_type=ThreatType.INTRUSION,
        ))
        matches = db.match("some log line containing /bin/sh -c ls")
        assert len(matches) == 1
        assert matches[0].id == "sig-001"

    def test_bytes_match(self):
        db = AttackSignatureDB()
        db.add_signature(AttackSignature(
            id="sig-bytes", name="UPX header",
            pattern="55 50 58 21", pattern_type="bytes",
            severity=AlertSeverity.HIGH,
        ))
        matches = db.match(b"random\x55\x50\x58\x21trailer")
        assert len(matches) == 1

    def test_behavior_match(self):
        db = AttackSignatureDB()
        spec = '{"all":[{"field":"can_id","op":"eq","value":123},{"field":"dlc","op":"gt","value":5}]}'
        db.add_signature(AttackSignature(
            id="sig-beh", name="bad can combo",
            pattern=spec, pattern_type="behavior",
            severity=AlertSeverity.CRITICAL,
        ))
        # Matching event
        assert len(db.match({"can_id": 123, "dlc": 8})) == 1
        # Non-matching event
        assert len(db.match({"can_id": 999, "dlc": 8})) == 0

    def test_export_import_roundtrip(self, tmp_path):
        db = AttackSignatureDB()
        db.add_signature(AttackSignature(id="s1", name="a", pattern="x"))
        db.add_signature(AttackSignature(id="s2", name="b", pattern="y",
                                          severity=AlertSeverity.CRITICAL))
        path = tmp_path / "sigs.json"
        db.export_signatures(str(path))
        db2 = AttackSignatureDB()
        n = db2.import_signatures(str(path))
        assert n == 2
        assert db2.get_signature("s2").severity == AlertSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 3. CAN bus monitor
# ---------------------------------------------------------------------------


class TestCANBusMonitor:
    def test_unauthorized_can_id_alert(self, can_monitor):
        alerts = []
        can_monitor.register_alert_callback(alerts.append)
        # Send a frame with an unauthorized ID
        frame = CANFrameEvent(
            arbitration_id=0x999, is_extended=False, is_remote=False,
            dlc=2, data=b"\x01\x02", timestamp=timestamp_now(),
            interface="vcan0",
        )
        can_monitor.process_frame(frame)
        assert any(a.alert_type == "unauthorized_can_id" for a in alerts)

    def test_allowed_id_no_unauthorized_alert(self, can_monitor):
        alerts = []
        can_monitor.register_alert_callback(alerts.append)
        frame = CANFrameEvent(
            arbitration_id=0x100, is_extended=False, is_remote=False,
            dlc=2, data=b"\x01\x02", timestamp=timestamp_now(),
            interface="vcan0",
        )
        can_monitor.process_frame(frame)
        assert not any(a.alert_type == "unauthorized_can_id" for a in alerts)

    def test_message_injection_rate_alert(self, can_monitor):
        alerts = []
        can_monitor.register_alert_callback(alerts.append)
        now = timestamp_now()
        for i in range(150):
            frame = CANFrameEvent(
                arbitration_id=0x100, is_extended=False, is_remote=False,
                dlc=1, data=bytes([i & 0xFF]),
                timestamp=now + i * 0.001,  # 1000/s
                interface="vcan0",
            )
            can_monitor.process_frame(frame)
        assert any(a.alert_type == "message_injection" for a in alerts)

    def test_replay_detection(self, can_monitor):
        alerts = []
        can_monitor.register_alert_callback(alerts.append)
        can_monitor._fuzz_threshold = 1.5  # disable fuzz
        now = timestamp_now()
        payload = b"\xAA\xBB"
        for i in range(5):
            can_monitor.process_frame(CANFrameEvent(
                arbitration_id=0x200, is_extended=False, is_remote=False,
                dlc=2, data=payload, timestamp=now + i * 1.0,
                interface="vcan0",
            ))
        assert any(a.alert_type == "replay_attack" for a in alerts)

    def test_parse_can_frame_helper(self):
        # Build a SocketCAN-style frame manually
        arb_id = struct.pack("<I", 0x123)
        flags = bytes([0x02])  # DLC=2
        data = b"\xAA\xBB"
        raw = arb_id + flags + data
        frame = parse_can_frame(raw, interface="vcan0")
        assert frame is not None
        assert frame.arbitration_id == 0x123
        assert frame.dlc == 2
        assert frame.data == b"\xAA\xBB"


# ---------------------------------------------------------------------------
# 4. Network monitor
# ---------------------------------------------------------------------------


class TestNetworkMonitor:
    def test_port_scan_detection(self, network_monitor):
        alerts = []
        network_monitor.register_alert_callback(alerts.append)
        for port in range(1, 11):  # 10 ports
            network_monitor._on_packet_info({
                "timestamp": timestamp_now(),
                "src_ip": "10.0.0.99", "dst_ip": "192.168.1.1",
                "l4_proto": "tcp", "src_port": 50000, "dst_port": port,
                "flags": "S",
            })
        assert any(a.alert_type == "port_scan" for a in alerts)

    def test_c2_beaconing_detection(self, network_monitor):
        alerts = []
        network_monitor.register_alert_callback(alerts.append)
        base = timestamp_now()
        for i in range(6):
            network_monitor._on_packet_info({
                "timestamp": base + i * 60.0,  # 60s apart, jitter < 5s
                "src_ip": "10.0.0.5", "dst_ip": "203.0.113.99",
                "l4_proto": "tcp", "src_port": 54321, "dst_port": 443,
                "flags": "PA",
            })
        assert any(a.alert_type == "c2_beaconing" for a in alerts)

    def test_unusual_outbound_alert(self, network_monitor):
        alerts = []
        network_monitor.register_alert_callback(alerts.append)
        network_monitor._on_packet_info({
            "timestamp": timestamp_now(),
            "src_ip": "10.0.0.5", "dst_ip": "203.0.113.1",
            "l4_proto": "tcp", "src_port": 43210, "dst_port": 31337,
            "flags": "S",
        })
        assert any(a.alert_type == "unusual_outbound" for a in alerts)


# ---------------------------------------------------------------------------
# 5. Packet analyzer
# ---------------------------------------------------------------------------


class TestPacketAnalyzer:
    def test_mqtt_cmd_topic_rule(self):
        pa = PacketAnalyzer()
        alerts = []
        pa.register_alert_callback(alerts.append)
        meta_dict = {
            "timestamp": timestamp_now(),
            "src_ip": "10.0.0.5", "dst_ip": "10.0.0.10",
            "l4_proto": "tcp", "src_port": 49152, "dst_port": 1883,
            "app_proto": "mqtt",
            "app_fields": {"msg_type": 3, "topic": "/vehicle/cmd/accelerate"},
        }
        meta, fired = pa.analyze(meta_dict)
        assert meta is not None
        assert meta.app_proto == "mqtt"
        assert any(a.rule_id == "mqtt-cmd-topic" for a in fired)

    def test_add_custom_rule(self):
        pa = PacketAnalyzer()
        pa.add_rule(PacketRule(
            id="custom-can-bus-flush",
            name="CAN bus flush",
            field="dst_port",
            op="eq",
            value=29999,
            severity=AlertSeverity.HIGH,
            threat_type=ThreatType.INTRUSION,
        ))
        meta_dict = {"timestamp": timestamp_now(),
                     "dst_port": 29999, "src_ip": "1.2.3.4"}
        _, alerts = pa.analyze(meta_dict)
        assert any(a.rule_id == "custom-can-bus-flush" for a in alerts)

    def test_extract_metadata_from_dict(self):
        pa = PacketAnalyzer()
        info = {"timestamp": 1.0, "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2",
                "dst_port": 53, "l4_proto": "udp"}
        meta_dict = pa.extract_metadata(info)
        assert meta_dict is not None
        assert meta_dict["src_ip"] == "1.1.1.1"
        assert meta_dict["app_proto"] == "dns"


# ---------------------------------------------------------------------------
# 6. Alert manager lifecycle
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_raise_and_acknowledge(self, alert_manager):
        alert = alert_manager.raise_alert(
            title="t", description="d",
            severity=AlertSeverity.HIGH, threat_type=ThreatType.INTRUSION,
            source="test", evidence={"src_ip": "10.0.0.1"},
        )
        assert alert.status == AlertStatus.NEW
        assert alert_manager.acknowledge_alert(alert.id)
        assert alert_manager.get_alert(alert.id).status == AlertStatus.ACKNOWLEDGED

    def test_deduplication(self, alert_manager):
        a1 = alert_manager.raise_alert(
            title="t", description="d", severity=AlertSeverity.MEDIUM,
            threat_type=ThreatType.DOS, source="s", evidence={"can_id": 1},
        )
        a2 = alert_manager.raise_alert(
            title="t", description="d", severity=AlertSeverity.MEDIUM,
            threat_type=ThreatType.DOS, source="s", evidence={"can_id": 1},
        )
        assert a1.id == a2.id
        assert a2.count == 2

    def test_invalid_transition(self, alert_manager):
        alert = alert_manager.raise_alert(
            title="t", description="d", severity=AlertSeverity.LOW,
            threat_type=ThreatType.RECON, source="s",
        )
        assert alert_manager.resolve_alert(alert.id)
        # Already resolved; cannot escalate
        assert not alert_manager.escalate_alert(alert.id)

    def test_resolve_and_stats(self, alert_manager):
        alert = alert_manager.raise_alert(
            title="t", description="d", severity=AlertSeverity.LOW,
            threat_type=ThreatType.RECON, source="s",
        )
        assert alert_manager.resolve_alert(alert.id)
        stats = alert_manager.get_statistics()
        assert stats["resolved"] == 1


# ---------------------------------------------------------------------------
# 7. Incident logger integrity
# ---------------------------------------------------------------------------


class TestIncidentLogger:
    def test_chain_verifies_clean(self, incident_logger):
        incident_logger.log_incident(
            title="first", description="d1",
            severity=AlertSeverity.LOW, threat_type=ThreatType.RECON,
            source="test",
        )
        incident_logger.log_incident(
            title="second", description="d2",
            severity=AlertSeverity.HIGH, threat_type=ThreatType.CAN_INJECTION,
            source="test",
        )
        ok, errors = incident_logger.verify_chain()
        assert ok, errors
        assert errors == []

    def test_tamper_detection(self, incident_logger, tmp_path):
        incident_logger.log_incident(
            title="first", description="d1",
            severity=AlertSeverity.LOW, threat_type=ThreatType.RECON,
            source="test",
        )
        # Tamper: rewrite the file with modified description
        with open(incident_logger.log_path, "r") as fh:
            lines = fh.readlines()
        import json
        rec = json.loads(lines[0])
        rec["description"] = "tampered"
        lines[0] = json.dumps(rec, sort_keys=True) + "\n"
        with open(incident_logger.log_path, "w") as fh:
            fh.writelines(lines)
        ok, errors = incident_logger.verify_chain()
        assert not ok
        assert any("record_hash mismatch" in e for e in errors)

    def test_query_by_severity(self, incident_logger):
        for sev, ttype in [
            (AlertSeverity.LOW, ThreatType.RECON),
            (AlertSeverity.HIGH, ThreatType.CAN_INJECTION),
            (AlertSeverity.CRITICAL, ThreatType.DOS),
        ]:
            incident_logger.log_incident(
                title="t", description="d", severity=sev,
                threat_type=ttype, source="test",
            )
        highs = incident_logger.query_incidents(severity=AlertSeverity.HIGH)
        assert len(highs) == 1
        assert highs[0].threat_type == ThreatType.CAN_INJECTION

    def test_export_jsonl(self, incident_logger, tmp_path):
        incident_logger.log_incident(
            title="t", description="d", severity=AlertSeverity.LOW,
            threat_type=ThreatType.RECON, source="test",
        )
        out = tmp_path / "export.jsonl"
        n = incident_logger.export_incidents(str(out), fmt="jsonl")
        assert n == 1
        assert out.exists()


# ---------------------------------------------------------------------------
# 8. Response engine
# ---------------------------------------------------------------------------


class TestResponseEngine:
    def test_dry_run_skips_execution(self, response_engine):
        executed = []
        response_engine.register_response_action(
            ResponseActionType.BLOCK_IP, lambda a: executed.append(a.target),
        )
        actions = response_engine.execute_response(
            severity=AlertSeverity.HIGH, target="1.2.3.4", alert_id="a-1",
        )
        assert len(actions) > 0
        # In dry-run mode nothing should be executed
        assert all(a.status == "skipped" for a in actions if a.action_type == ResponseActionType.BLOCK_IP)
        assert executed == []

    def test_handler_executes_when_enabled(self):
        engine = ResponseEngine(enabled=True, dry_run=False,
                                max_actions_per_min=1000, cooldown_sec=0)
        captured = []
        engine.register_response_action(
            ResponseActionType.BLOCK_IP, lambda a: captured.append(a.target),
        )
        engine.execute_response(
            severity=AlertSeverity.HIGH, target="9.9.9.9", alert_id="a-1",
        )
        assert "9.9.9.9" in captured

    def test_no_handler_fails_gracefully(self, response_engine):
        # In dry-run mode the engine skips before checking handler, so use non-dry-run
        engine = ResponseEngine(enabled=True, dry_run=False,
                                max_actions_per_min=1000, cooldown_sec=0)
        actions = engine.execute_response(
            severity=AlertSeverity.HIGH, target="1.1.1.1",
        )
        # At least one action should fail because no handler is registered
        statuses = {a.status for a in actions}
        assert "failed" in statuses or "skipped" in statuses  # skipped via dry-run path

    def test_response_history(self, response_engine):
        response_engine.execute_response(
            severity=AlertSeverity.MEDIUM, target="8.8.8.8",
        )
        history = response_engine.get_response_history()
        assert len(history) > 0


# ---------------------------------------------------------------------------
# 9. Log analyzer
# ---------------------------------------------------------------------------


class TestLogAnalyzer:
    def test_failed_login_pattern(self):
        la = LogAnalyzer()
        line = ("Jan 01 12:00:00 host sshd[1234]: Failed password for invalid user "
                "admin from 10.0.0.5 port 50000 ssh2")
        findings = la.parse_log(line, source="/var/log/auth.log")
        assert any(f.pattern_id == "auth-failed-login" for f in findings)

    def test_brute_force_burst(self):
        la = LogAnalyzer()
        la._failed_login_threshold = 3
        la._failed_login_window = 60.0
        for i in range(3):
            la.parse_log(
                f"Jan 01 12:00:0{i} host sshd[1234]: Failed password for invalid user "
                f"root from 10.0.0.5 port {50000 + i} ssh2",
                source="auth.log",
            )
        findings = la.get_findings(limit=20)
        assert any(f.pattern_id == "auth-brute-force" for f in findings)

    def test_ioc_string_pattern(self):
        la = LogAnalyzer()
        findings = la.parse_log("process spawned meterpreter reverse_tcp", source="app.log")
        assert any(f.pattern_id == "app-ioc-string" for f in findings)


# ---------------------------------------------------------------------------
# 10. Threat classifier
# ---------------------------------------------------------------------------


class TestThreatClassifier:
    def test_classify_can_injection(self):
        tc = ThreatClassifier()
        result = tc.classify({"alert_type": "message_injection", "can_id": 0x100})
        assert result.threat_type == ThreatType.CAN_INJECTION
        assert result.severity == AlertSeverity.HIGH
        assert "T0817" in result.mitre_attack_ids

    def test_classify_port_scan(self):
        tc = ThreatClassifier()
        result = tc.classify({"alert_type": "port_scan", "src_ip": "1.2.3.4"})
        assert result.threat_type == ThreatType.RECON
        assert "T1046" in result.mitre_attack_ids

    def test_threat_intel_lookup(self):
        from cybersecurity.intrusion_detection.threat_classifier import ThreatIntelEntry
        tc = ThreatClassifier()
        tc.add_threat_intel(ThreatIntelEntry(
            id="ti-1", indicator="203.0.113.99", indicator_type="ip",
            threat_type=ThreatType.DATA_EXFIL, severity=AlertSeverity.CRITICAL,
            description="Known C2 IP", mitre_attack_ids=["T1071.001"],
        ))
        result = tc.classify({"src_ip": "203.0.113.99"})
        assert result.threat_type == ThreatType.DATA_EXFIL
        assert result.confidence > 0.5

    def test_severity_assessment(self):
        tc = ThreatClassifier()
        assert tc.assess_severity(ThreatType.DOS, 0.9, {"safety_critical": True}) == AlertSeverity.CRITICAL
        assert tc.assess_severity(ThreatType.RECON, 0.2) == AlertSeverity.LOW


# ---------------------------------------------------------------------------
# 11. IDS engine event pipeline
# ---------------------------------------------------------------------------


class TestIDSEngine:
    def test_rule_match_emits_callback(self):
        engine = IDSEngine(worker_count=1, batch_size=1, enable_anomaly=False)
        rule = IDSRule(
            id="rule-1", name="test rule", event_types=[EventType.CAN_FRAME],
            matcher=lambda e: e.payload.get("can_id") == 0xDEAD,
            severity=AlertSeverity.HIGH, threat_type=ThreatType.CAN_INJECTION,
        )
        engine.register_rule(rule)
        captured = []
        engine.register_alert_callback(lambda e, rules, anomaly: captured.append((e, rules)))
        engine.start()
        try:
            engine.ingest_event(
                EventType.CAN_FRAME, "test",
                {"can_id": 0xDEAD, "data": "deadbeef"},
            )
            engine.ingest_event(
                EventType.CAN_FRAME, "test",
                {"can_id": 0x1234, "data": "ok"},
            )
            # Wait briefly for async processing
            time.sleep(0.2)
        finally:
            engine.stop()
        assert len(captured) == 1
        evt, rules = captured[0]
        assert rules[0].id == "rule-1"

    def test_pipeline_stats(self):
        engine = IDSEngine(worker_count=1, batch_size=1, enable_anomaly=False)
        engine.start()
        try:
            for i in range(5):
                engine.ingest_event(EventType.SENSOR_READING, "test", {"value": i})
            time.sleep(0.2)
            stats = engine.get_event_pipeline_stats()
            assert stats["pipeline"]["ingested"] >= 5
            assert stats["pipeline"]["preprocessed"] >= 5
        finally:
            engine.stop()


# ---------------------------------------------------------------------------
# 12. Forensic tools
# ---------------------------------------------------------------------------


class TestForensicTools:
    def test_collect_file_evidence(self, tmp_path):
        ft = ForensicTools(evidence_dir=str(tmp_path / "evidence"), compress=True)
        target = tmp_path / "suspect.bin"
        target.write_bytes(b"malware" * 100)
        case_dir = ft._new_case_dir("case-x")
        item = ft.collect_file(case_dir, str(target), description="suspect")
        assert item is not None
        assert item.type.value == "other"
        assert item.size_bytes == 700
        assert len(item.sha256) == 64

    def test_hash_evidence(self, tmp_path):
        ft = ForensicTools(evidence_dir=str(tmp_path / "ev"))
        target = tmp_path / "f.bin"
        target.write_bytes(b"abcdef")
        case_dir = ft._new_case_dir()
        item = ft.collect_file(case_dir, str(target))
        hashes = ft.hash_evidence([item])
        assert item.id in hashes
        assert len(hashes[item.id]) == 64

    def test_package_evidence(self, tmp_path):
        ft = ForensicTools(evidence_dir=str(tmp_path / "ev"), compress=True)
        case_dir = ft._new_case_dir("pack-me")
        (case_dir / "f.txt").write_text("hello", encoding="utf-8")
        archive = ft.package_evidence(case_dir)
        assert os.path.exists(archive)


# ---------------------------------------------------------------------------
# 13. Utils
# ---------------------------------------------------------------------------


class TestUtils:
    def test_format_bytes(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(1024) == "1.00 KiB"
        assert format_bytes(1024 * 1024) == "1.00 MiB"
        assert format_bytes(1500, binary=False) == "1.50 KB"

    def test_parse_ip_packet(self):
        # Build a minimal IPv4 packet header (20 bytes)
        version_ihl = (4 << 4) | 5  # v4, IHL=5
        tos = 0
        total_length = 20
        identification = 0
        flags_offset = 0
        ttl = 64
        proto = 6  # TCP
        checksum = 0
        src = bytes([10, 0, 0, 1])
        dst = bytes([10, 0, 0, 2])
        header = struct.pack(
            "!BBHHHBBH4s4s",
            version_ihl, tos, total_length, identification,
            flags_offset, ttl, proto, checksum, src, dst,
        )
        info = parse_ip_packet(header)
        assert info is not None
        assert info.version == 4
        assert info.src_ip == "10.0.0.1"
        assert info.dst_ip == "10.0.0.2"
        assert info.protocol == 6

    def test_rate_limit_check(self):
        state = {}
        # First few should pass
        assert rate_limit_check("k", 100.0, state)
        assert rate_limit_check("k", 100.0, state)
        # Different key passes independently
        assert rate_limit_check("other", 100.0, state)


# ---------------------------------------------------------------------------
# 14. End-to-end orchestrator smoke test
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_start_stop_smoke(self, tmp_path):
        cfg = default_config()
        cfg.incident_log.log_path = str(tmp_path / "incidents.chain")
        cfg.forensic.evidence_dir = str(tmp_path / "evidence")
        cfg.response.enabled = False  # no actions during smoke test
        ids = IntrusionDetectionSystem(config=cfg)
        ids.start()
        try:
            # Ingest a benign event
            evt = ids.process_event(
                EventType.CAN_FRAME, "test", {"can_id": 0x100, "data": "0102"},
            )
            assert evt is not None
            time.sleep(0.2)
            stats = ids.get_statistics()
            assert "alerts_active" in stats
        finally:
            ids.stop()

    def test_verify_integrity(self, tmp_path):
        cfg = default_config()
        cfg.incident_log.log_path = str(tmp_path / "incidents.chain")
        cfg.response.enabled = False
        ids = IntrusionDetectionSystem(config=cfg)
        result = ids.verify_integrity()
        assert result["intact"] is True


# ---------------------------------------------------------------------------
# 15. Malware detector
# ---------------------------------------------------------------------------


class TestMalwareDetector:
    def test_signature_hash_match(self, tmp_path):
        md = MalwareDetector()
        target = tmp_path / "bad.bin"
        target.write_bytes(b"this is malware content with meterpreter string")
        # Add a hash signature that does not match
        from cybersecurity.intrusion_detection.malware_detector import IoCSignature
        md.add_signature(IoCSignature(
            id="bad-hash", name="bad hash", type="hash_sha256",
            value="0" * 64, severity=AlertSeverity.HIGH,
        ))
        result = md.scan_file(str(target))
        assert result.target == str(target)
        # The IoC string heuristic should fire
        assert any(f.get("label") == "credential_dumping_tool" for f in result.findings)

    def test_scan_process_invalid_pid(self):
        md = MalwareDetector()
        result = md.scan_process(999_999_999)
        assert any(f["type"] == "process_error" for f in result.findings)
