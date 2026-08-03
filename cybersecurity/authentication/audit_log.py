"""
audit_log.py
============

Tamper-evident audit logging for the authentication sub-system.

Every audit entry is hash-chained to the previous one (each entry's
``prev_hash`` is the SHA-256 of the prior entry's canonical serialised
form). This means a single byte flip anywhere in the log breaks the
chain and is detected by :meth:`AuditLogger.verify_chain`.

The log is intentionally append-only and line-delimited JSON, so it can
be tailed with standard tooling (``tail -f``) and shipped to a SIEM
without bespoke parsers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .constants import AuthEvent, AuthStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class AuditEvent:
    """A single audit-log entry."""

    event_id: str
    timestamp: int
    event_type: AuthEvent
    actor: str  # user_id / device_id / "system"
    status: AuthStatus
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    target: Optional[str] = None  # resource acted upon
    method: Optional[str] = None  # AuthMethod, kept as str for portability
    detail: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = "0" * 64
    entry_hash: str = ""

    def canonical(self) -> bytes:
        """Serialise the entry without ``entry_hash`` for hashing."""
        payload = asdict(self)
        payload.pop("entry_hash", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def compute_hash(self) -> str:
        """Return the SHA-256 of this entry's canonical form."""
        return hashlib.sha256(self.canonical()).hexdigest()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class AuditLogger:
    """Append-only, hash-chained audit log.

    Parameters
    ----------
    log_path:
        Filesystem path to the audit log. The file is created (and
        pre-pended with a synthetic genesis entry) if it does not exist.
    hash_algorithm:
        Only SHA-256 is currently supported; the parameter exists so that
        the algorithm can be upgraded in a backward-compatible way later.
    flush_immediately:
        If ``True`` (default) the file is flushed + ``fsync``-ed after
        every entry, which is what you want for forensic durability.
    """

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        log_path: Path,
        hash_algorithm: str = "sha256",
        flush_immediately: bool = True,
        max_entry_size: int = 16_384,
    ) -> None:
        if hash_algorithm.lower() != "sha256":
            raise ValueError("Only sha256 is supported for hash chaining")
        self.log_path = Path(log_path)
        self.hash_algorithm = hash_algorithm.lower()
        self.flush_immediately = flush_immediately
        self.max_entry_size = max_entry_size
        self._lock = threading.Lock()
        self._last_hash = self.GENESIS_HASH
        self._counter = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
        else:
            self._last_hash = self._tail_hash()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tail_hash(self) -> str:
        last_hash = self.GENESIS_HASH
        try:
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        last_hash = entry.get("entry_hash", last_hash)
                    except json.JSONDecodeError:
                        logger.error("Corrupt audit entry encountered; chain broken")
                        return last_hash
        except FileNotFoundError:
            pass
        return last_hash

    def _next_event_id(self) -> str:
        self._counter += 1
        return f"{int(time.time())}-{self._counter:08d}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_event(
        self,
        event_type: AuthEvent,
        actor: str,
        status: AuthStatus,
        *,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        target: Optional[str] = None,
        method: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Append a new entry to the audit log and return it."""
        event = AuditEvent(
            event_id=self._next_event_id(),
            timestamp=int(time.time()),
            event_type=event_type,
            actor=actor,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            target=target,
            method=method,
            detail=detail or {},
            prev_hash=self._last_hash,
        )
        event.entry_hash = event.compute_hash()
        if len(event.canonical()) > self.max_entry_size:
            raise ValueError("Audit entry exceeds max_entry_size; truncation refused")

        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
                if self.flush_immediately:
                    fh.flush()
                    os.fsync(fh.fileno())
            self._last_hash = event.entry_hash
        logger.debug(
            "audit event id=%s type=%s actor=%s status=%s",
            event.event_id,
            event.event_type.value,
            event.actor,
            event.status.value,
        )
        return event

    def iter_entries(self) -> Iterator[AuditEvent]:
        """Yield every :class:`AuditEvent` in chronological order."""
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield AuditEvent(
                    event_id=data["event_id"],
                    timestamp=data["timestamp"],
                    event_type=AuthEvent(data["event_type"]),
                    actor=data["actor"],
                    status=AuthStatus(data["status"]),
                    ip_address=data.get("ip_address"),
                    user_agent=data.get("user_agent"),
                    target=data.get("target"),
                    method=data.get("method"),
                    detail=data.get("detail", {}),
                    prev_hash=data.get("prev_hash", self.GENESIS_HASH),
                    entry_hash=data.get("entry_hash", ""),
                )

    def query(
        self,
        *,
        event_type: Optional[AuthEvent] = None,
        actor: Optional[str] = None,
        status: Optional[AuthStatus] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Filter entries by attribute / time range."""
        results: List[AuditEvent] = []
        for entry in self.iter_entries():
            if event_type is not None and entry.event_type != event_type:
                continue
            if actor is not None and entry.actor != actor:
                continue
            if status is not None and entry.status != status:
                continue
            if since is not None and entry.timestamp < since:
                continue
            if until is not None and entry.timestamp > until:
                continue
            results.append(entry)
            if limit is not None and len(results) >= limit:
                break
        return results

    def verify_chain(self) -> bool:
        """Return ``True`` iff every ``prev_hash`` chains correctly."""
        prev = self.GENESIS_HASH
        for entry in self.iter_entries():
            if entry.prev_hash != prev:
                logger.error(
                    "Audit chain broken at event_id=%s: expected prev_hash=%s got %s",
                    entry.event_id,
                    prev,
                    entry.prev_hash,
                )
                return False
            recomputed = entry.compute_hash()
            if recomputed != entry.entry_hash:
                logger.error(
                    "Audit entry hash mismatch at event_id=%s", entry.event_id
                )
                return False
            prev = entry.entry_hash
        return True

    def export_log(self, destination: Path) -> int:
        """Export the entire log to ``destination``; returns entry count."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with destination.open("w", encoding="utf-8") as out:
            for entry in self.iter_entries():
                out.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
                count += 1
        return count

    def count(self) -> int:
        """Return total number of entries."""
        return sum(1 for _ in self.iter_entries())


__all__ = ["AuditEvent", "AuditLogger"]
