"""Utility functions for the intrusion detection module.

This module collects small, reusable helpers used across detectors, analyzers,
and forensic tools: packet and CAN-frame parsing, timestamp helpers, hashing,
byte formatting, and a simple token-bucket rate limiter.
"""

from __future__ import annotations

import binascii
import hashlib
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__) if False else None  # placeholder for typing
import logging  # noqa: E402
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def timestamp_now() -> float:
    """Return the current UTC timestamp as a UNIX float (seconds since epoch)."""
    return time.time()


def iso_timestamp(ts: Optional[float] = None) -> str:
    """Return an ISO-8601 UTC timestamp string."""
    import datetime as _dt
    if ts is None:
        ts = time.time()
    return _dt.datetime.utcfromtimestamp(ts).isoformat() + "Z"


# ---------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------

def format_bytes(num_bytes: int, binary: bool = True) -> str:
    """Format a byte count as a human-readable string.

    Parameters
    ----------
    num_bytes:
        The byte count to format (must be >= 0).
    binary:
        If True, use 1024-based (KiB/MiB/...); otherwise 1000-based (KB/MB/...).
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    base = 1024 if binary else 1000
    suffixes = (
        ["B", "KiB", "MiB", "GiB", "TiB", "PiB"] if binary
        else ["B", "KB", "MB", "GB", "TB", "PB"]
    )
    size = float(num_bytes)
    idx = 0
    while size >= base and idx < len(suffixes) - 1:
        size /= base
        idx += 1
    if idx == 0:
        return f"{int(size)} {suffixes[idx]}"
    return f"{size:.2f} {suffixes[idx]}"


def hex_dump(data: bytes, prefix: str = "") -> str:
    """Return a classic hex+ASCII dump of the given bytes."""
    if not data:
        return f"{prefix}(empty)"
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{prefix}{i:08x}  {hex_part:<48}  {ascii_part}")
    return "\n".join(lines)


def extract_payload(frame: bytes, offset: int = 0, length: Optional[int] = None) -> bytes:
    """Extract a payload slice from a raw frame buffer."""
    if length is None:
        return frame[offset:]
    return frame[offset:offset + length]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_packet_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Compute a hex digest of the packet/frame bytes."""
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def hash_file(path: str, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Stream-hash a file and return the hex digest."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# IP packet parsing (lightweight, no scapy required for basic fields)
# ---------------------------------------------------------------------------


@dataclass
class IPPacketInfo:
    """Minimal structured representation of an IPv4 packet."""

    version: int
    ihl: int
    total_length: int
    ttl: int
    protocol: int
    src_ip: str
    dst_ip: str
    payload: bytes
    checksum: int


def _ip_to_str(packed: bytes) -> str:
    return ".".join(str(b) for b in packed)


def parse_ip_packet(raw: bytes) -> Optional[IPPacketInfo]:
    """Parse a raw Ethernet payload as an IPv4 packet.

    Returns None when the buffer is too short or the IP version is unsupported.
    """
    if raw is None or len(raw) < 20:
        return None
    version_ihl = raw[0]
    version = version_ihl >> 4
    if version != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or len(raw) < ihl:
        return None
    total_length = struct.unpack("!H", raw[2:4])[0]
    ttl = raw[8]
    protocol = raw[9]
    checksum = struct.unpack("!H", raw[10:12])[0]
    src_ip = _ip_to_str(raw[12:16])
    dst_ip = _ip_to_str(raw[16:20])
    payload = raw[ihl:total_length] if total_length >= ihl else raw[ihl:]
    return IPPacketInfo(
        version=version,
        ihl=ihl,
        total_length=total_length,
        ttl=ttl,
        protocol=protocol,
        src_ip=src_ip,
        dst_ip=dst_ip,
        payload=payload,
        checksum=checksum,
    )


# ---------------------------------------------------------------------------
# CAN frame parsing (classic CAN 11/29-bit IDs)
# ---------------------------------------------------------------------------


@dataclass
class CANFrame:
    """Structured representation of a classic CAN frame."""

    arbitration_id: int
    is_extended: bool
    is_remote: bool
    is_error: bool
    dlc: int
    data: bytes
    timestamp: float = field(default_factory=timestamp_now)
    interface: str = "can0"

    @property
    def id_hex(self) -> str:
        width = 8 if self.is_extended else 3
        return f"0x{self.arbitration_id:0{width}X}"


def parse_can_frame(raw: bytes, interface: str = "can0") -> Optional[CANFrame]:
    """Parse a raw socketcan-style frame buffer.

    Expected layout (Linux SocketCAN):
        byte 0-3: arbitration ID (little-endian, low 29 bits used)
        byte 4:   DLC in low nibble, flags in high nibble
                  bit 4 (0x10) = RTR, bit 6 (0x40) = EFF (extended),
                  bit 5 (0x20) = ERR, bit 7 (0x80) = FDF (CAN-FD)
        byte 5-12: up to 8 data bytes
    """
    if raw is None or len(raw) < 5:
        return None
    arb_id = struct.unpack("<I", raw[0:4])[0]
    flags = raw[4]
    dlc = flags & 0x0F
    is_extended = bool(flags & 0x40)
    is_remote = bool(flags & 0x10)
    is_error = bool(flags & 0x20)
    if is_extended:
        arb_id &= 0x1FFFFFFF
    else:
        arb_id &= 0x7FF
    data = bytes(raw[5:5 + min(dlc, 8)])
    return CANFrame(
        arbitration_id=arb_id,
        is_extended=is_extended,
        is_remote=is_remote,
        is_error=is_error,
        dlc=dlc,
        data=data,
        interface=interface,
    )


def can_id_is_diagnostic(can_id: int) -> bool:
    """Return True if the CAN ID falls in the UDS/diagnostic range.

    Standard OBD-II / UDS request IDs are 0x7DF (broadcast) and 0x7E0-0x7E7
    (functional). Responses are 0x7E8-0x7EF.
    """
    return can_id == 0x7DF or 0x7E0 <= can_id <= 0x7EF


# ---------------------------------------------------------------------------
# Rate limiting (token bucket)
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Simple token-bucket rate limiter.

    Parameters
    ----------
    capacity:
        Maximum number of tokens in the bucket.
    refill_rate:
        Tokens added per second.
    """

    capacity: float
    refill_rate: float
    _tokens: float = field(init=False)
    _last: float = field(default_factory=timestamp_now)

    def __post_init__(self) -> None:
        self._tokens = self.capacity

    def consume(self, amount: float = 1.0) -> bool:
        """Consume ``amount`` tokens; return True if allowed, False if rejected."""
        now = timestamp_now()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        if self._tokens >= amount:
            self._tokens -= amount
            return True
        return False


def rate_limit_check(
    key: str,
    limit_per_sec: float,
    state: Optional[Dict[str, TokenBucket]] = None,
) -> bool:
    """Check a rate limit for ``key``.

    Parameters
    ----------
    key:
        Identifier of the rate-limited entity (e.g. ``"192.168.1.5"``).
    limit_per_sec:
        Maximum allowed operations per second.
    state:
        Optional shared dict mapping keys to buckets. If omitted, a fresh
        bucket is created each call (useful only for one-off checks).
    """
    if state is None:
        bucket = TokenBucket(capacity=limit_per_sec, refill_rate=limit_per_sec)
        return bucket.consume()
    bucket = state.get(key)
    if bucket is None or bucket.refill_rate != limit_per_sec:
        bucket = TokenBucket(capacity=max(1.0, limit_per_sec), refill_rate=limit_per_sec)
        state[key] = bucket
    return bucket.consume()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def safe_inet_aton(ip_str: str) -> Optional[int]:
    """Convert a dotted-quad IP string to an int; return None on failure."""
    try:
        return struct.unpack("!I", __import__("socket").inet_aton(ip_str))[0]
    except (OSError, ValueError):
        return None


def truncate_middle(s: str, max_len: int = 80) -> str:
    """Truncate a string in the middle if it exceeds ``max_len`` characters."""
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    keep = (max_len - 3) // 2
    return f"{s[:keep]}...{s[-keep:]}"


def crc16_ccitt(data: bytes, polynomial: int = 0x1021, init: int = 0xFFFF) -> int:
    """Compute a CRC-16/CCITT-FALSE checksum over ``data``."""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ polynomial) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


__all__ = [
    "IPPacketInfo",
    "CANFrame",
    "TokenBucket",
    "timestamp_now",
    "iso_timestamp",
    "format_bytes",
    "hex_dump",
    "extract_payload",
    "compute_packet_hash",
    "hash_file",
    "parse_ip_packet",
    "parse_can_frame",
    "can_id_is_diagnostic",
    "rate_limit_check",
    "safe_inet_aton",
    "truncate_middle",
    "crc16_ccitt",
]
