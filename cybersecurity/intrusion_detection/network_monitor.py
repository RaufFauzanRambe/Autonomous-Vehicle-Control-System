"""Network monitor: packet capture, connection tracking, attack detection.

Wraps a scapy or pyshark L2 sniffing backend, maintains a rolling table of
TCP/UDP/ICMP connections, and applies detectors for:

* **Port scans** — many distinct destination ports from one source in a window.
* **C2 beaconing** — repeated outbound connections at near-constant intervals.
* **Unusual outbound traffic** — connections to rare ASNs / non-vehicle ports.
* **TCP SYN floods** — many half-open SYNs without completion.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:  # scapy is the recommended backend
    from scapy.all import IP, TCP, UDP, ICMP, sniff as scapy_sniff  # type: ignore
    _HAS_SCAPY = True
except ImportError:  # pragma: no cover - scapy optional
    scapy_sniff = None  # type: ignore
    _HAS_SCAPY = False

from .constants import (
    DEFAULT_NETWORK_INTERFACE,
    DEFAULT_PCAP_SNAPLEN,
    DEFAULT_PCAP_TIMEOUT_MS,
    DEFAULT_PORT_SCAN_THRESHOLD,
    DEFAULT_PORT_SCAN_WINDOW_SEC,
    DEFAULT_C2_BEACON_INTERVAL_SEC,
    DEFAULT_C2_BEACON_JITTER_SEC,
    COMMON_VEHICLE_PORTS,
    AlertSeverity,
    ThreatType,
)
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Connection:
    """A 5-tuple connection tracked by the monitor."""

    proto: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    first_seen: float
    last_seen: float
    packet_count: int = 0
    byte_count: int = 0
    syn_count: int = 0
    fin_count: int = 0
    state: str = "new"

    @property
    def key(self) -> Tuple[str, str, int, str, int]:
        return (self.proto, self.src_ip, self.src_port, self.dst_ip, self.dst_port)


@dataclass
class NetworkAlert:
    """An alert raised by the network monitor."""

    timestamp: float
    alert_type: str
    severity: AlertSeverity
    threat_type: ThreatType
    src_ip: Optional[str]
    dst_ip: Optional[str]
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class NetworkMonitor:
    """Thread-safe network capture and intrusion detection.

    Parameters
    ----------
    interface:
        Capture interface (e.g. ``"eth0"``).
    bpf_filter:
        Optional BPF filter to apply at the kernel level.
    allowed_ports:
        TCP/UDP ports considered legitimate for the vehicle. Connections to
        ports outside this set (when non-empty) raise ``unusual_outbound`` alerts.
    """

    def __init__(
        self,
        interface: str = DEFAULT_NETWORK_INTERFACE,
        bpf_filter: str = "",
        snaplen: int = DEFAULT_PCAP_SNAPLEN,
        timeout_ms: int = DEFAULT_PCAP_TIMEOUT_MS,
        promiscuous: bool = True,
        allowed_ports: Optional[Tuple[int, ...]] = COMMON_VEHICLE_PORTS,
        port_scan_threshold: int = DEFAULT_PORT_SCAN_THRESHOLD,
        port_scan_window: float = DEFAULT_PORT_SCAN_WINDOW_SEC,
        c2_interval: float = DEFAULT_C2_BEACON_INTERVAL_SEC,
        c2_jitter: float = DEFAULT_C2_BEACON_JITTER_SEC,
        connection_ttl: float = 600.0,
    ) -> None:
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.snaplen = snaplen
        self.timeout_ms = timeout_ms
        self.promiscuous = promiscuous
        self.allowed_ports: Tuple[int, ...] = tuple(allowed_ports) if allowed_ports else ()
        self.port_scan_threshold = int(port_scan_threshold)
        self.port_scan_window = float(port_scan_window)
        self.c2_interval = float(c2_interval)
        self.c2_jitter = float(c2_jitter)
        self.connection_ttl = float(connection_ttl)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._connections: Dict[Tuple, Connection] = {}
        self._src_ports: Dict[str, Deque[Tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=5000)
        )
        self._dst_pairs: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=5000)
        )
        self._alerts: Deque[NetworkAlert] = deque(maxlen=20_000)
        self._alert_callbacks: List[Callable[[NetworkAlert], None]] = []
        self._packet_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._stats = {
            "packets_captured": 0,
            "alerts_raised": 0,
            "connections_active": 0,
            "connections_total": 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_capture(self) -> None:
        """Start the background capture thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Network capture already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="net-monitor", daemon=True
        )
        self._thread.start()

    def stop_capture(self) -> None:
        """Stop the background capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Network capture stopped")

    def _capture_loop(self) -> None:
        if not _HAS_SCAPY:
            logger.warning("scapy not installed; network monitor idle (test mode)")
            while not self._stop_event.is_set():
                self._stop_event.wait(0.5)
            return
        kwargs = dict(
            iface=self.interface,
            prn=self._on_packet,
            store=False,
            count=0,
            timeout=self.timeout_ms / 1000.0,
        )
        if self.bpf_filter:
            kwargs["filter"] = self.bpf_filter
        try:
            scapy_sniff(**kwargs)
        except Exception as exc:
            logger.error("scapy sniff error: %s", exc)

    # ------------------------------------------------------------------
    # Packet handling
    # ------------------------------------------------------------------

    def _on_packet(self, pkt: Any) -> None:
        """Callback invoked by scapy for each captured packet."""
        with self._lock:
            self._stats["packets_captured"] += 1
        info = self._extract_packet_info(pkt)
        for cb in list(self._packet_callbacks):
            try:
                cb(info)
            except Exception as exc:
                logger.error("Packet callback raised: %s", exc)
        self._track_connection(info)
        self._detect_attacks(info)

    def _extract_packet_info(self, pkt: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {"timestamp": timestamp_now()}
        if not _HAS_SCAPY or pkt is None:
            return info
        if pkt.haslayer(IP):
            ip = pkt[IP]
            info.update({
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "ttl": ip.ttl,
                "proto": int(ip.proto),
                "payload_len": int(ip.len) - int(ip.ihl * 4) if ip.ihl else 0,
            })
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            info.update({
                "l4_proto": "tcp",
                "src_port": int(tcp.sport),
                "dst_port": int(tcp.dport),
                "flags": str(tcp.flags),
                "seq": int(tcp.seq),
                "ack": int(tcp.ack),
            })
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            info.update({
                "l4_proto": "udp",
                "src_port": int(udp.sport),
                "dst_port": int(udp.dport),
                "payload_len": int(udp.len) - 8,
            })
        elif pkt.haslayer(ICMP):
            info.update({"l4_proto": "icmp", "type": int(pkt[ICMP].type)})
        return info

    def _track_connection(self, info: Dict[str, Any]) -> None:
        proto = info.get("l4_proto")
        if proto is None or "src_ip" not in info:
            return
        key = (proto, info["src_ip"], info.get("src_port", 0),
               info["dst_ip"], info.get("dst_port", 0))
        now = info["timestamp"]
        with self._lock:
            conn = self._connections.get(key)
            if conn is None:
                conn = Connection(
                    proto=proto,
                    src_ip=info["src_ip"],
                    src_port=info.get("src_port", 0),
                    dst_ip=info["dst_ip"],
                    dst_port=info.get("dst_port", 0),
                    first_seen=now,
                    last_seen=now,
                )
                self._connections[key] = conn
                self._stats["connections_total"] += 1
            conn.last_seen = now
            conn.packet_count += 1
            conn.byte_count += int(info.get("payload_len", 0) or 0)
            flags = info.get("flags", "")
            if "S" in flags and "A" not in flags:
                conn.syn_count += 1
            if "F" in flags:
                conn.fin_count += 1
            if conn.syn_count and conn.fin_count:
                conn.state = "closed"
            elif conn.syn_count and not conn.fin_count:
                conn.state = "established" if "A" in flags else "syn_sent"

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_attacks(self, info: Dict[str, Any]) -> None:
        if "src_ip" not in info or "dst_port" not in info:
            return
        alerts: List[NetworkAlert] = []
        alerts.extend(self._check_port_scan(info))
        alerts.extend(self._check_unusual_outbound(info))
        alerts.extend(self._check_c2_beaconing(info))
        alerts.extend(self._check_syn_flood(info))
        for a in alerts:
            self._raise_alert(a)

    def _check_port_scan(self, info: Dict[str, Any]) -> List[NetworkAlert]:
        src = info["src_ip"]
        dst_port = info["dst_port"]
        now = info["timestamp"]
        with self._lock:
            history = self._src_ports[src]
            history.append((now, dst_port))
            cutoff = now - self.port_scan_window
            recent_ports = {p for t, p in history if t >= cutoff}
        if len(recent_ports) >= self.port_scan_threshold:
            return [NetworkAlert(
                timestamp=now,
                alert_type="port_scan",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.RECON,
                src_ip=src,
                dst_ip=info.get("dst_ip"),
                description=f"{src} contacted {len(recent_ports)} ports in {self.port_scan_window}s",
                evidence={"ports": sorted(recent_ports)[:50], "threshold": self.port_scan_threshold},
            )]
        return []

    def _check_unusual_outbound(self, info: Dict[str, Any]) -> List[NetworkAlert]:
        if not self.allowed_ports:
            return []
        dst_port = info.get("dst_port", 0)
        if dst_port in self.allowed_ports:
            return []
        return [NetworkAlert(
            timestamp=info["timestamp"],
            alert_type="unusual_outbound",
            severity=AlertSeverity.MEDIUM,
            threat_type=ThreatType.DATA_EXFIL,
            src_ip=info["src_ip"],
            dst_ip=info.get("dst_ip"),
            description=f"Outbound connection to non-vehicle port {dst_port}",
            evidence={"dst_port": dst_port, "allowed": self.allowed_ports},
        )]

    def _check_c2_beaconing(self, info: Dict[str, Any]) -> List[NetworkAlert]:
        src, dst = info["src_ip"], info["dst_ip"]
        now = info["timestamp"]
        pair = f"{src}->{dst}"
        with self._lock:
            times = self._dst_pairs[pair]
            times.append(now)
            if len(times) < 5:
                return []
            intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        mean_int = statistics.mean(intervals)
        stdev_int = statistics.pstdev(intervals)
        # Check if intervals cluster around the configured C2 beacon period.
        if (abs(mean_int - self.c2_interval) <= self.c2_jitter
                and stdev_int <= self.c2_jitter):
            return [NetworkAlert(
                timestamp=now,
                alert_type="c2_beaconing",
                severity=AlertSeverity.CRITICAL,
                threat_type=ThreatType.DATA_EXFIL,
                src_ip=src,
                dst_ip=dst,
                description=f"Beaconing to {dst} every ~{mean_int:.1f}s",
                evidence={"interval_mean": mean_int, "interval_stdev": stdev_int,
                          "expected": self.c2_interval, "jitter": self.c2_jitter},
            )]
        return []

    def _check_syn_flood(self, info: Dict[str, Any]) -> List[NetworkAlert]:
        flags = info.get("flags", "")
        if "S" not in flags or "A" in flags:
            return []
        dst = info["dst_ip"]
        now = info["timestamp"]
        # Count SYNs in the last 5s to the same destination.
        with self._lock:
            recent = sum(
                1 for c in self._connections.values()
                if c.dst_ip == dst and c.proto == "tcp"
                and now - c.last_seen < 5.0
            )
        if recent >= 200:
            return [NetworkAlert(
                timestamp=now,
                alert_type="syn_flood",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.DOS,
                src_ip=info["src_ip"],
                dst_ip=dst,
                description=f"SYN flood against {dst} ({recent} connections/5s)",
                evidence={"conn_count": recent},
            )]
        return []

    # ------------------------------------------------------------------
    # Public detection helpers (sync, callable without live capture)
    # ------------------------------------------------------------------

    def detect_port_scan(self, src_ip: str, since: Optional[float] = None) -> List[int]:
        """Return the list of distinct ports contacted by ``src_ip`` since ``since``."""
        with self._lock:
            history = self._src_ports.get(src_ip, deque())
            cutoff = since or (timestamp_now() - self.port_scan_window)
            return sorted({p for t, p in history if t >= cutoff})

    def detect_c2_traffic(self, src_ip: str, dst_ip: str) -> Optional[Dict[str, float]]:
        """Return C2-beaconing statistics for a pair, or None."""
        pair = f"{src_ip}->{dst_ip}"
        with self._lock:
            times = list(self._dst_pairs.get(pair, []))
        if len(times) < 5:
            return None
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        return {
            "interval_mean": statistics.mean(intervals),
            "interval_stdev": statistics.pstdev(intervals),
            "sample_count": len(intervals),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_connections(self, limit: int = 100, stale_after: Optional[float] = None) -> List[Connection]:
        """Return tracked connections, optionally filtered by freshness."""
        now = timestamp_now()
        with self._lock:
            conns = list(self._connections.values())
            if stale_after is not None:
                conns = [c for c in conns if now - c.last_seen < stale_after]
            conns.sort(key=lambda c: c.last_seen, reverse=True)
            return conns[:limit]

    def register_alert_callback(self, cb: Callable[[NetworkAlert], None]) -> None:
        with self._lock:
            self._alert_callbacks.append(cb)

    def register_packet_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._packet_callbacks.append(cb)

    def ingest_packet_info(self, info: Dict[str, Any]) -> None:
        """Manually feed a packet-info dict into the pipeline (for testing)."""
        self._on_packet_info(info)

    def _on_packet_info(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self._stats["packets_captured"] += 1
        for cb in list(self._packet_callbacks):
            try:
                cb(info)
            except Exception as exc:
                logger.error("Packet callback raised: %s", exc)
        self._track_connection(info)
        self._detect_attacks(info)

    def _raise_alert(self, alert: NetworkAlert) -> None:
        with self._lock:
            self._alerts.append(alert)
            self._stats["alerts_raised"] += 1
            cbs = list(self._alert_callbacks)
        for cb in cbs:
            try:
                cb(alert)
            except Exception as exc:
                logger.error("Alert callback raised: %s", exc)

    def get_alerts(self, limit: int = 100) -> List[NetworkAlert]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            self._stats["connections_active"] = len(self._connections)
            return dict(self._stats)


__all__ = ["NetworkMonitor", "Connection", "NetworkAlert"]
