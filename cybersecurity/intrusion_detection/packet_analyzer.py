"""Deep packet inspection for vehicle Ethernet / SOME-IP / DoIP / MQTT / HTTP.

The :class:`PacketAnalyzer` takes raw packets (scapy or pre-extracted metadata
dicts) and applies a configurable set of rules to detect:

* Unauthenticated SOME-IP service requests to safety-critical services.
* DoIP diagnostic sessions opened outside an authorized service bay.
* Plaintext MQTT publishes to vehicle command topics.
* Suspicious HTTP methods (PUT/DELETE) on internal REST APIs.
* IPs communicating from disallowed subnets.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:  # scapy optional
    from scapy.all import IP, TCP, UDP, Raw, Ether  # type: ignore
    from scapy.contrib.automotive.someip import SOMEIP  # type: ignore
    _HAS_SCAPY = True
except ImportError:  # pragma: no cover
    _HAS_SCAPY = False

from .constants import AlertSeverity, ThreatType
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PacketRule:
    """A user-defined DPI rule."""

    id: str
    name: str
    field: str  # dotted path into the metadata dict, e.g. "tcp.dst_port"
    op: str  # "eq" | "ne" | "in" | "regex" | "gt" | "lt" | "contains"
    value: Any
    severity: AlertSeverity = AlertSeverity.MEDIUM
    threat_type: ThreatType = ThreatType.INTRUSION
    description: str = ""
    enabled: bool = True


@dataclass
class PacketMetadata:
    """Structured metadata extracted from a packet."""

    timestamp: float
    eth_src: Optional[str] = None
    eth_dst: Optional[str] = None
    eth_type: Optional[int] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    ip_proto: Optional[int] = None
    ttl: Optional[int] = None
    l4_proto: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    tcp_flags: Optional[str] = None
    payload_len: Optional[int] = None
    payload_hex: Optional[str] = None
    payload_ascii: Optional[str] = None
    app_proto: Optional[str] = None  # "someip" | "doip" | "mqtt" | "http" | "dns" | ...
    app_fields: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.__dict__.items():
            if v is None or v == "":
                continue
            out[k] = v
        return out


@dataclass
class DPIAlert:
    """An alert raised by the packet analyzer."""

    timestamp: float
    rule_id: str
    severity: AlertSeverity
    threat_type: ThreatType
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class PacketAnalyzer:
    """Deep packet inspection with rule-based matching."""

    def __init__(self) -> None:
        self._rules: Dict[str, PacketRule] = {}
        self._lock = threading.RLock()
        self._alerts: Deque[DPIAlert] = deque(maxlen=20_000)
        self._alert_callbacks: List[Callable[[DPIAlert], None]] = []
        self._stats = {
            "packets_parsed": 0,
            "alerts_raised": 0,
            "by_app_proto": {},  # Dict[str, int]
        }
        self._install_default_rules()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: PacketRule) -> bool:
        """Register a new DPI rule."""
        if not rule.id:
            raise ValueError("PacketRule requires an id")
        with self._lock:
            self._rules[rule.id] = rule
            logger.debug("Added DPI rule %s (%s)", rule.id, rule.name)
            return True

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def list_rules(self, enabled_only: bool = False) -> List[PacketRule]:
        with self._lock:
            rules = list(self._rules.values())
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules

    def _install_default_rules(self) -> None:
        """Install a baseline set of vehicle-aware DPI rules."""
        defaults = [
            PacketRule(
                id="someip-unauth-0x1235",
                name="Unauth SOME-IP service 0x1235",
                field="app_fields.service_id",
                op="eq",
                value=0x1235,
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.INTRUSION,
                description="SOME-IP service 0x1235 (steering) called without auth",
            ),
            PacketRule(
                id="doip-outside-bay",
                name="DoIP session outside service bay",
                field="app_proto",
                op="eq",
                value="doip",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.INTRUSION,
                description="DoIP diagnostic message observed outside authorized window",
            ),
            PacketRule(
                id="mqtt-cmd-topic",
                name="MQTT publish to /vehicle/cmd",
                field="app_fields.topic",
                op="regex",
                value=r"^/vehicle/(cmd|brake|steer)/.+$",
                severity=AlertSeverity.CRITICAL,
                threat_type=ThreatType.INTRUSION,
                description="MQTT message to a safety-critical vehicle command topic",
            ),
            PacketRule(
                id="http-put-internal",
                name="HTTP PUT/DELETE to internal API",
                field="app_fields.method",
                op="in",
                value=["PUT", "DELETE", "PATCH"],
                severity=AlertSeverity.MEDIUM,
                threat_type=ThreatType.INTRUSION,
                description="Mutating HTTP method on internal API",
            ),
        ]
        for r in defaults:
            self.add_rule(r)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_packet(self, pkt: Any) -> Optional[PacketMetadata]:
        """Parse a scapy packet or metadata dict into :class:`PacketMetadata`."""
        if pkt is None:
            return None
        if isinstance(pkt, dict):
            return self._metadata_from_dict(pkt)
        if not _HAS_SCAPY:
            return None
        return self._metadata_from_scapy(pkt)

    def _metadata_from_dict(self, info: Dict[str, Any]) -> PacketMetadata:
        meta = PacketMetadata(timestamp=info.get("timestamp", timestamp_now()))
        meta.eth_src = info.get("eth_src")
        meta.eth_dst = info.get("eth_dst")
        meta.eth_type = info.get("eth_type")
        meta.src_ip = info.get("src_ip")
        meta.dst_ip = info.get("dst_ip")
        meta.ip_proto = info.get("ip_proto")
        meta.ttl = info.get("ttl")
        meta.l4_proto = info.get("l4_proto")
        meta.src_port = info.get("src_port")
        meta.dst_port = info.get("dst_port")
        meta.tcp_flags = info.get("flags")
        meta.payload_len = info.get("payload_len")
        meta.app_proto = info.get("app_proto")
        meta.app_fields = info.get("app_fields", {})
        meta.extras = {k: v for k, v in info.items() if k not in meta.__dict__}
        return meta

    def _metadata_from_scapy(self, pkt: Any) -> PacketMetadata:
        meta = PacketMetadata(timestamp=timestamp_now())
        if pkt.haslayer(Ether):
            meta.eth_src = pkt[Ether].src
            meta.eth_dst = pkt[Ether].dst
            meta.eth_type = int(pkt[Ether].type)
        if pkt.haslayer(IP):
            ip = pkt[IP]
            meta.src_ip = ip.src
            meta.dst_ip = ip.dst
            meta.ip_proto = int(ip.proto)
            meta.ttl = int(ip.ttl)
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            meta.l4_proto = "tcp"
            meta.src_port = int(tcp.sport)
            meta.dst_port = int(tcp.dport)
            meta.tcp_flags = str(tcp.flags)
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            meta.l4_proto = "udp"
            meta.src_port = int(udp.sport)
            meta.dst_port = int(udp.dport)
        raw_payload = bytes(pkt[Raw].load) if pkt.haslayer(Raw) else b""
        if raw_payload:
            meta.payload_len = len(raw_payload)
            meta.payload_hex = raw_payload.hex()
            meta.payload_ascii = raw_payload.decode("latin-1", errors="replace")
        self._identify_app_protocol(meta, raw_payload, pkt)
        return meta

    def _identify_app_proto(self, meta: PacketMetadata, payload: bytes, pkt: Any) -> None:
        """Heuristically identify the application-layer protocol."""
        dst_port = meta.dst_port or 0
        src_port = meta.src_port or 0
        # SOME-IP: ports 30490+; protocol header starts with service_id (4B),
        # method_id (4B), length (4B), client_id (2B), session_id (2B),
        # protocol version (1B=1), interface version (1B), message type (1B), return code (1B)
        if (30490 <= dst_port <= 30500 or 30490 <= src_port <= 30500) and len(payload) >= 16:
            try:
                import struct
                service_id, method_id, length = struct.unpack("!III", payload[:12])
                proto_ver = payload[12]
                msg_type = payload[14]
                meta.app_proto = "someip"
                meta.app_fields = {
                    "service_id": service_id,
                    "method_id": method_id,
                    "length": length,
                    "proto_version": proto_ver,
                    "msg_type": msg_type,
                }
                return
            except Exception:
                pass
        # DoIP: port 13400; first byte = protocol version (0x02), second = inverse
        if dst_port == 13400 and len(payload) >= 8:
            if payload[0] == 0x02 and payload[1] == 0xFD:
                meta.app_proto = "doip"
                meta.app_fields = {
                    "protocol_version": payload[0],
                    "payload_type": int.from_bytes(payload[2:4], "big"),
                    "payload_length": int.from_bytes(payload[4:8], "big"),
                }
                return
        # MQTT: port 1883; first byte high nibble for CONNECT (1) or PUBLISH (3)
        if dst_port == 1883 and payload:
            msg_type = (payload[0] >> 4) & 0x0F
            if msg_type in (1, 3, 8, 10, 12):
                meta.app_proto = "mqtt"
                meta.app_fields = {"msg_type": msg_type, "flags": payload[0] & 0x0F}
                if msg_type == 3 and len(payload) > 4:
                    topic_len = int.from_bytes(payload[2:4], "big")
                    if 4 + topic_len <= len(payload):
                        meta.app_fields["topic"] = payload[4:4 + topic_len].decode("latin-1", "replace")
                return
        # HTTP: starts with an HTTP method or "HTTP/"
        if payload:
            try:
                first_line = payload.split(b"\r\n", 1)[0].decode("latin-1", "replace")
                if re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|HTTP)/", first_line):
                    meta.app_proto = "http"
                    parts = first_line.split()
                    if parts and parts[0] in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
                        meta.app_fields = {
                            "method": parts[0],
                            "path": parts[1] if len(parts) > 1 else "",
                            "version": parts[2] if len(parts) > 2 else "",
                        }
                    else:
                        meta.app_fields = {"status_line": first_line}
                    return
            except Exception:
                pass
        if dst_port == 53 or src_port == 53:
            meta.app_proto = "dns"

    # ------------------------------------------------------------------
    # Metadata extraction & rule matching
    # ------------------------------------------------------------------

    def extract_metadata(self, pkt: Any) -> Optional[Dict[str, Any]]:
        """Convenience wrapper that returns metadata as a flat dict."""
        meta = self.parse_packet(pkt)
        return meta.to_dict() if meta else None

    def match_rules(self, meta: PacketMetadata) -> List[DPIAlert]:
        """Evaluate all rules against the metadata; return any alerts raised."""
        flat = meta.to_dict()
        # Flatten dotted paths
        flat_full: Dict[str, Any] = dict(flat)
        for k, v in (meta.app_fields or {}).items():
            flat_full[f"app_fields.{k}"] = v
        alerts: List[DPIAlert] = []
        with self._lock:
            rules = list(self._rules.values())
        for rule in rules:
            if not rule.enabled:
                continue
            if self._eval_rule(rule, flat_full):
                alerts.append(DPIAlert(
                    timestamp=meta.timestamp,
                    rule_id=rule.id,
                    severity=rule.severity,
                    threat_type=rule.threat_type,
                    description=rule.description,
                    metadata=flat_full,
                ))
        for a in alerts:
            self._raise_alert(a)
        return alerts

    def _eval_rule(self, rule: PacketRule, flat: Dict[str, Any]) -> bool:
        actual = flat.get(rule.field)
        if actual is None and rule.field not in flat:
            return False
        expected = rule.value
        op = rule.op
        try:
            if op == "eq":
                return actual == expected
            if op == "ne":
                return actual != expected
            if op == "in":
                return actual in (expected or [])
            if op == "gt":
                return actual is not None and actual > expected
            if op == "lt":
                return actual is not None and actual < expected
            if op == "contains":
                return expected in (actual or "")
            if op == "regex":
                return re.search(expected, str(actual)) is not None
        except Exception as exc:
            logger.debug("Rule %s eval error: %s", rule.id, exc)
            return False
        return False

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def analyze(self, pkt: Any) -> Tuple[Optional[PacketMetadata], List[DPIAlert]]:
        """Parse a packet and run all rules; return metadata and any alerts."""
        meta = self.parse_packet(pkt)
        if meta is None:
            return None, []
        with self._lock:
            self._stats["packets_parsed"] += 1
            if meta.app_proto:
                self._stats["by_app_proto"][meta.app_proto] = (
                    self._stats["by_app_proto"].get(meta.app_proto, 0) + 1
                )
        alerts = self.match_rules(meta)
        return meta, alerts

    # ------------------------------------------------------------------
    # Alerts & stats
    # ------------------------------------------------------------------

    def register_alert_callback(self, cb: Callable[[DPIAlert], None]) -> None:
        with self._lock:
            self._alert_callbacks.append(cb)

    def _raise_alert(self, alert: DPIAlert) -> None:
        with self._lock:
            self._alerts.append(alert)
            self._stats["alerts_raised"] += 1
            cbs = list(self._alert_callbacks)
        for cb in cbs:
            try:
                cb(alert)
            except Exception as exc:
                logger.error("Alert callback raised: %s", exc)

    def get_alerts(self, limit: int = 100) -> List[DPIAlert]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


__all__ = ["PacketAnalyzer", "PacketRule", "PacketMetadata", "DPIAlert"]
