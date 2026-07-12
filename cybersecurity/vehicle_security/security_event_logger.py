"""Tamper-evident, append-only security event log.

The :class:`SecurityEventLogger` writes JSONL-formatted log entries to disk.
Each entry contains a hash of the previous entry, producing a cryptographic
chain that makes silent retroactive modification detectable. Verification
(:meth:`verify_chain`) re-computes the chain from the genesis entry and
returns the index of the first broken link (or ``None`` if the chain is
intact).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .constants import Severity
from .utils import compute_sha256, format_timestamp, hex_encode, safe_compare

logger = logging.getLogger(__name__)


GENESIS_HASH = "0" * 64  # 32-byte all-zero hash in hex


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class LogEntry:
    """One entry in the hash-chained log."""

    seq: int
    timestamp: str
    event_type: str
    severity: str
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    current_hash: str = ""

    def body_bytes(self) -> bytes:
        """Bytes that are hashed to produce current_hash."""
        body = {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_hash(self) -> str:
        return hex_encode(compute_sha256(self.body_bytes()))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["current_hash"] = self.current_hash or self.compute_hash()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        return cls(
            seq=int(data["seq"]),
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            severity=data["severity"],
            source=data["source"],
            message=data["message"],
            details=data.get("details", {}),
            prev_hash=data.get("prev_hash", GENESIS_HASH),
            current_hash=data.get("current_hash", ""),
        )


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #


class SecurityEventLogger:
    """Append-only, hash-chained JSONL log."""

    def __init__(self, log_path: str, flush_interval: float = 1.0, max_entries: int = 1_000_000) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._buffer: List[LogEntry] = []
        self._last_flush = 0.0
        self._seq = 0
        self._last_hash = GENESIS_HASH
        self._load_tail()

    # ------------------------------------------------------------------ #
    # Loading existing chain
    # ------------------------------------------------------------------ #

    def _load_tail(self) -> None:
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = LogEntry.from_dict(json.loads(line))
                    self._seq = entry.seq
                    self._last_hash = entry.current_hash or entry.compute_hash()
            logger.info("resumed event log at seq=%d", self._seq)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load log tail: %s", exc)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        if severity not in {s.value for s in Severity}:
            raise ValueError(f"invalid severity: {severity}")
        with self._lock:
            self._seq += 1
            entry = LogEntry(
                seq=self._seq,
                timestamp=format_timestamp(),
                event_type=event_type,
                severity=severity,
                source=source,
                message=message,
                details=details or {},
                prev_hash=self._last_hash,
            )
            entry.current_hash = entry.compute_hash()
            self._last_hash = entry.current_hash
            self._buffer.append(entry)

            if (
                len(self._buffer) >= 100
                or (time.monotonic() - self._last_flush) >= self.flush_interval
            ):
                self._flush_locked()
        return entry

    # ------------------------------------------------------------------ #
    # Flushing
    # ------------------------------------------------------------------ #

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        # Atomic append via tmp + rename is overkill for append-only; use 'a'
        with open(self.log_path, "a", encoding="utf-8") as fh:
            for entry in self._buffer:
                fh.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
        self._buffer.clear()
        self._last_flush = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #

    def verify_chain(self) -> Optional[int]:
        """Verify the entire on-disk chain.

        Returns the seq number of the first broken entry, or ``None`` if the
        entire chain is intact. The in-memory buffer is included in the check.
        """
        prev_hash = GENESIS_HASH
        seq = 0
        with self._lock:
            buffer_snapshot = list(self._buffer)

        def _check(entry: LogEntry) -> bool:
            nonlocal prev_hash
            if entry.prev_hash != prev_hash:
                return False
            if entry.current_hash and entry.current_hash != entry.compute_hash():
                return False
            prev_hash = entry.current_hash or entry.compute_hash()
            return True

        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = LogEntry.from_dict(json.loads(line))
                    if entry.seq != seq + 1:
                        return entry.seq
                    if not _check(entry):
                        return entry.seq
                    seq = entry.seq

        for entry in buffer_snapshot:
            if entry.seq != seq + 1:
                return entry.seq
            if not _check(entry):
                return entry.seq
            seq = entry.seq
        return None

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def query(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        since_seq: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[LogEntry]:
        """Read matching entries from disk + in-memory buffer."""
        results: List[LogEntry] = []
        with self._lock:
            buffer_snapshot = list(self._buffer)

        def _match(entry: LogEntry) -> bool:
            if event_type and entry.event_type != event_type:
                return False
            if severity and entry.severity != severity:
                return False
            if source and entry.source != source:
                return False
            if since_seq is not None and entry.seq < since_seq:
                return False
            return True

        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = LogEntry.from_dict(json.loads(line))
                    if _match(entry):
                        results.append(entry)
                        if limit and len(results) >= limit:
                            return results

        for entry in buffer_snapshot:
            if _match(entry):
                results.append(entry)
                if limit and len(results) >= limit:
                    break
        return results

    def iter_entries(self) -> Iterator[LogEntry]:
        """Iterate every entry on disk + in-memory buffer in order."""
        with self._lock:
            buffer_snapshot = list(self._buffer)
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield LogEntry.from_dict(json.loads(line))
        for entry in buffer_snapshot:
            yield entry

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export_log(self, dest_path: str, since_seq: Optional[int] = None) -> int:
        """Export entries (optionally since *since_seq*) to a JSONL file."""
        count = 0
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            for entry in self.iter_entries():
                if since_seq is not None and entry.seq < since_seq:
                    continue
                fh.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
                count += 1
        logger.info("exported %d entries to %s", count, dest_path)
        return count

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        return self._seq

    def last_hash(self) -> str:
        with self._lock:
            return self._last_hash
