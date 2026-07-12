"""CAN/Ethernet firewall for the in-vehicle network.

The :class:`VehicleFirewall` enforces an allowlist/denylist policy on both the
CAN bus (per arbitration ID) and Ethernet (per source IP / port). It performs
rate limiting, stateful connection tracking for TCP/UDP traffic, and provides
hooks for deep packet inspection (DPI) callbacks that can be plugged in by the
caller.

The firewall is intentionally pure-Python so that it can be unit-tested
without any kernel netfilter/iptables dependencies; the integration layer
(e.g. ``can-utils`` / ``iptables`` wrappers) calls into :meth:`inspect_can_frame`
and :meth:`inspect_ethernet_packet` for each frame/packet.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

from .utils import now_monotonic, parse_can_id

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    RATE_LIMIT = "rate_limit"


class Protocol(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


@dataclass
class FirewallRule:
    """A single firewall rule.

    A rule matches when every non-None field matches the inspected frame or
    packet. ``verdict`` determines whether matching traffic is allowed or
    denied. ``direction`` may be ``in`` (received), ``out`` (transmitted), or
    ``*`` (both).
    """

    verdict: Verdict
    direction: str = "*"
    can_id: Optional[int] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: Optional[Protocol] = None
    description: str = ""
    enabled: bool = True
    priority: int = 100  # lower = higher priority

    def matches_packet(
        self,
        direction: str,
        src_ip: Optional[str],
        dst_ip: Optional[str],
        src_port: Optional[int],
        dst_port: Optional[int],
        protocol: Optional[Protocol],
    ) -> bool:
        if not self.enabled:
            return False
        if self.direction != "*" and self.direction != direction:
            return False
        if self.src_ip is not None and self.src_ip != src_ip:
            return False
        if self.dst_ip is not None and self.dst_ip != dst_ip:
            return False
        if self.src_port is not None and self.src_port != src_port:
            return False
        if self.dst_port is not None and self.dst_port != dst_port:
            return False
        if self.protocol is not None and self.protocol != protocol:
            return False
        return True

    def matches_can(self, direction: str, can_id: int) -> bool:
        if not self.enabled:
            return False
        if self.direction != "*" and self.direction != direction:
            return False
        return self.can_id is not None and self.can_id == can_id


# --------------------------------------------------------------------------- #
# DPI hooks
# --------------------------------------------------------------------------- #


@dataclass
class DPIResult:
    verdict: Verdict
    reason: str = ""


DPIHook = Callable[[bytes], DPIResult]


# --------------------------------------------------------------------------- #
# Firewall
# --------------------------------------------------------------------------- #


@dataclass
class Connection:
    """A tracked TCP/UDP flow (stateful)."""

    key: Tuple[str, int, str, int, Protocol]
    state: str = "new"  # new, established, closed
    last_seen: float = field(default_factory=now_monotonic)
    packets: int = 0
    bytes_seen: int = 0


class VehicleFirewall:
    """CAN/Ethernet firewall with allowlist, rate-limit, stateful, DPI hooks."""

    def __init__(
        self,
        can_allowlist: Optional[List[int]] = None,
        can_denylist: Optional[List[int]] = None,
        ip_allowlist: Optional[List[str]] = None,
        ip_denylist: Optional[List[str]] = None,
        blocked_ports: Optional[List[int]] = None,
        rate_limit_per_second: int = 500,
        dpi_enabled: bool = True,
        stateful_tracking: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._rules: List[FirewallRule] = []
        self._dpi_hooks: List[DPIHook] = []
        self.dpi_enabled = dpi_enabled
        self.stateful_tracking = stateful_tracking
        self.rate_limit_per_second = rate_limit_per_second

        # Compiled allow/deny lists (fast membership tests)
        self._can_allow: Set[int] = set(can_allowlist or [])
        self._can_deny: Set[int] = set(can_denylist or [])
        self._ip_allow: Set[str] = set(ip_allowlist or [])
        self._ip_deny: Set[str] = set(ip_denylist or [])
        self._blocked_ports: Set[int] = set(blocked_ports or [])

        # Rate limiting state per source
        self._can_rate: Dict[int, Deque[float]] = defaultdict(deque)
        self._eth_rate: Dict[str, Deque[float]] = defaultdict(deque)

        # Stateful connections: keyed by (src_ip, src_port, dst_ip, dst_port, proto)
        self._connections: Dict[Tuple, Connection] = {}

        # Blocked sources with expiry timestamps (None = permanent)
        self._blocked_sources: Dict[str, Optional[float]] = {}

        # Counters
        self.stats: Dict[str, int] = {
            "can_inspected": 0,
            "can_blocked": 0,
            "eth_inspected": 0,
            "eth_blocked": 0,
            "rate_limited": 0,
            "dpi_denied": 0,
        }

        logger.info("VehicleFirewall initialized (allowlist=%d denylist=%d)",
                    len(self._can_allow), len(self._can_deny))

    # ------------------------------------------------------------------ #
    # Rule management
    # ------------------------------------------------------------------ #

    def add_rule(self, rule: FirewallRule) -> None:
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority)
        logger.debug("added firewall rule: %s", rule)

    def remove_rule(self, rule: FirewallRule) -> bool:
        with self._lock:
            try:
                self._rules.remove(rule)
                return True
            except ValueError:
                return False

    def clear_rules(self) -> None:
        with self._lock:
            self._rules.clear()

    def add_dpi_hook(self, hook: DPIHook) -> None:
        with self._lock:
            self._dpi_hooks.append(hook)

    # ------------------------------------------------------------------ #
    # Blocking helpers
    # ------------------------------------------------------------------ #

    def block_source(self, source: str, duration: Optional[float] = None) -> None:
        """Block a source IP (Ethernet) or CAN source identifier (str of hex).

        Args:
            source: An IPv4/IPv6 string or a hex CAN-id string.
            duration: Seconds to block for; ``None`` means permanent.
        """
        with self._lock:
            expiry = None if duration is None else now_monotonic() + duration
            self._blocked_sources[source] = expiry
        logger.warning("blocking source %s (duration=%s)", source, duration)

    def unblock_source(self, source: str) -> bool:
        with self._lock:
            return self._blocked_sources.pop(source, None) is not None

    def get_blocked_sources(self) -> Dict[str, Optional[float]]:
        with self._lock:
            now = now_monotonic()
            active: Dict[str, Optional[float]] = {}
            expired: List[str] = []
            for src, expiry in self._blocked_sources.items():
                if expiry is not None and expiry <= now:
                    expired.append(src)
                else:
                    active[src] = expiry
            for src in expired:
                self._blocked_sources.pop(src, None)
            return active

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #

    def _rate_check(self, bucket: Dict, key, limit: int) -> bool:
        now = now_monotonic()
        dq = bucket[key]
        while dq and dq[0] < now - 1.0:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True

    # ------------------------------------------------------------------ #
    # CAN frame inspection
    # ------------------------------------------------------------------ #

    def inspect_can_frame(
        self,
        can_id: int,
        payload: bytes,
        direction: str = "in",
        source: Optional[str] = None,
    ) -> Verdict:
        """Inspect a CAN frame and return a verdict."""
        can_id = parse_can_id(can_id)
        self.stats["can_inspected"] += 1

        with self._lock:
            if source is not None and source in self.get_blocked_sources():
                self.stats["can_blocked"] += 1
                return Verdict.DENY

            # Denylist first (always), then allowlist (deny if not allowed)
            if can_id in self._can_deny:
                self.stats["can_blocked"] += 1
                return Verdict.DENY
            if self._can_allow and can_id not in self._can_allow:
                self.stats["can_blocked"] += 1
                return Verdict.DENY

            # Custom rules (priority order)
            for rule in self._rules:
                if rule.matches_can(direction, can_id):
                    if rule.verdict == Verdict.DENY:
                        self.stats["can_blocked"] += 1
                        return Verdict.DENY
                    if rule.verdict == Verdict.ALLOW:
                        break

            # DPI hooks
            if self.dpi_enabled and self._dpi_hooks:
                for hook in self._dpi_hooks:
                    result = hook(payload)
                    if result.verdict == Verdict.DENY:
                        self.stats["dpi_denied"] += 1
                        return Verdict.DENY

            # Rate limit per CAN id
            if not self._rate_check(self._can_rate, can_id, self.rate_limit_per_second):
                self.stats["rate_limited"] += 1
                return Verdict.RATE_LIMIT

        return Verdict.ALLOW

    # ------------------------------------------------------------------ #
    # Ethernet packet inspection
    # ------------------------------------------------------------------ #

    def inspect_ethernet_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: Optional[int],
        dst_port: Optional[int],
        protocol: Protocol,
        payload: bytes = b"",
        direction: str = "in",
    ) -> Verdict:
        """Inspect an Ethernet/IP packet and return a verdict."""
        self.stats["eth_inspected"] += 1

        with self._lock:
            if src_ip in self.get_blocked_sources():
                self.stats["eth_blocked"] += 1
                return Verdict.DENY

            if src_ip in self._ip_deny or dst_ip in self._ip_deny:
                self.stats["eth_blocked"] += 1
                return Verdict.DENY

            if self._ip_allow:
                if src_ip not in self._ip_allow and dst_ip not in self._ip_allow:
                    self.stats["eth_blocked"] += 1
                    return Verdict.DENY

            if dst_port is not None and dst_port in self._blocked_ports:
                self.stats["eth_blocked"] += 1
                return Verdict.DENY

            # Custom rules
            for rule in self._rules:
                if rule.matches_packet(direction, src_ip, dst_ip, src_port, dst_port, protocol):
                    if rule.verdict == Verdict.DENY:
                        self.stats["eth_blocked"] += 1
                        return Verdict.DENY
                    if rule.verdict == Verdict.ALLOW:
                        break

            # DPI hooks
            if self.dpi_enabled and self._dpi_hooks and payload:
                for hook in self._dpi_hooks:
                    result = hook(payload)
                    if result.verdict == Verdict.DENY:
                        self.stats["dpi_denied"] += 1
                        return Verdict.DENY

            # Rate limit per source IP
            if not self._rate_check(self._eth_rate, src_ip, self.rate_limit_per_second):
                self.stats["rate_limited"] += 1
                return Verdict.RATE_LIMIT

            # Stateful tracking
            if self.stateful_tracking:
                key = (src_ip, src_port or 0, dst_ip, dst_port or 0, protocol)
                conn = self._connections.get(key)
                if conn is None:
                    conn = Connection(key=key, state="new")
                    self._connections[key] = conn
                conn.last_seen = now_monotonic()
                conn.packets += 1
                conn.bytes_seen += len(payload)
                if protocol == Protocol.TCP and conn.packets > 3:
                    conn.state = "established"

        return Verdict.ALLOW

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def gc_connections(self, max_age: float = 300.0) -> int:
        """Reap idle stateful connections older than *max_age* seconds."""
        now = now_monotonic()
        with self._lock:
            stale = [k for k, c in self._connections.items() if now - c.last_seen > max_age]
            for k in stale:
                del self._connections[k]
        return len(stale)

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.stats)

    def get_connection_count(self) -> int:
        with self._lock:
            return len(self._connections)
