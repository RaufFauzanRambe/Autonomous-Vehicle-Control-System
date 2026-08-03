"""Tamper-evident incident logger with hash-chained entries.

Each :class:`IncidentRecord` written by :class:`IncidentLogger` is appended to
an append-only chain file. Every record carries a SHA-256 hash of its
serialized content concatenated with the previous record's hash, so any
after-the-fact tampering can be detected by :meth:`IncidentLogger.verify_chain`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .constants import AlertSeverity, IncidentStatus, ThreatType
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IncidentRecord:
    """A single tamper-evident incident log entry."""

    seq: int
    id: str
    timestamp: float
    title: str
    description: str
    severity: AlertSeverity
    threat_type: ThreatType
    status: IncidentStatus = IncidentStatus.OPEN
    source: str = ""
    alert_ids: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[str] = field(default_factory=list)
    previous_hash: str = ""  # hash of the previous record
    record_hash: str = ""    # hash of this record
    operator: str = ""
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_payload_dict(self) -> Dict[str, Any]:
        """Return the dict form used to compute the record hash (excludes
        ``record_hash`` itself)."""
        d = asdict(self)
        d.pop("record_hash", None)
        d["severity"] = self.severity.name
        d["threat_type"] = self.threat_type.value
        d["status"] = self.status.value
        return d

    def to_dict(self) -> Dict[str, Any]:
        d = self.to_payload_dict()
        d["record_hash"] = self.record_hash
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentRecord":
        data = dict(data)
        if isinstance(data.get("severity"), str):
            data["severity"] = AlertSeverity.from_str(data["severity"])
        if isinstance(data.get("threat_type"), str):
            try:
                data["threat_type"] = ThreatType(data["threat_type"])
            except ValueError:
                data["threat_type"] = ThreatType.UNKNOWN
        if isinstance(data.get("status"), str):
            try:
                data["status"] = IncidentStatus(data["status"])
            except ValueError:
                data["status"] = IncidentStatus.OPEN
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class IncidentLogger:
    """Append-only, hash-chained incident log.

    Parameters
    ----------
    log_path:
        Path to the chain file. Each line is a JSON-encoded
        :class:`IncidentRecord`. The file is created if missing.
    hash_algorithm:
        Hash algorithm name accepted by :mod:`hashlib` (default ``sha256``).
    """

    GENESIS_HASH: str = "0" * 64

    def __init__(
        self,
        log_path: str,
        hash_algorithm: str = "sha256",
        retention_days: int = 365,
    ) -> None:
        self.log_path = str(log_path)
        self.hash_algorithm = hash_algorithm
        self.retention_days = int(retention_days)
        self._lock = threading.RLock()
        self._last_seq = 0
        self._last_hash = self.GENESIS_HASH
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._load_existing_chain()

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def _load_existing_chain(self) -> None:
        """Read the existing chain to determine the last seq and hash."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = IncidentRecord.from_dict(json.loads(line))
                    self._last_seq = max(self._last_seq, rec.seq)
                    self._last_hash = rec.record_hash or self.GENESIS_HASH
            logger.info("Loaded incident chain with %d entries", self._last_seq)
        except Exception as exc:
            logger.error("Failed to load existing incident chain: %s", exc)

    def _compute_hash(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        h = hashlib.new(self.hash_algorithm)
        h.update(raw)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_incident(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        threat_type: ThreatType,
        source: str = "",
        alert_ids: Optional[List[str]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        response_actions: Optional[List[str]] = None,
        operator: str = "",
        notes: Optional[List[str]] = None,
    ) -> IncidentRecord:
        """Append a new incident record to the chain."""
        with self._lock:
            seq = self._last_seq + 1
            ts = timestamp_now()
            rec = IncidentRecord(
                seq=seq,
                id=f"inc-{seq:08d}",
                timestamp=ts,
                title=title,
                description=description,
                severity=severity,
                threat_type=threat_type,
                source=source,
                alert_ids=list(alert_ids or []),
                evidence=dict(evidence or {}),
                response_actions=list(response_actions or []),
                previous_hash=self._last_hash,
                operator=operator,
                notes=list(notes or []),
            )
            payload = rec.to_payload_dict()
            rec.record_hash = self._compute_hash(payload)
            self._append_record(rec)
            self._last_seq = seq
            self._last_hash = rec.record_hash
        logger.info("Logged incident %s: %s", rec.id, rec.title)
        return rec

    def _append_record(self, rec: IncidentRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True, default=str) + "\n")

    def add_note(self, incident_id: str, note: str, operator: str = "") -> bool:
        """Append a note to an existing incident.

        This appends a new entry to the chain referencing the original
        incident, preserving immutability of past records.
        """
        target = self.query_incidents(incident_id=incident_id, limit=1)
        if not target:
            return False
        original = target[0]
        new_notes = list(original.notes) + [note]
        self.log_incident(
            title=f"Update: {original.title}",
            description=f"Note added to {incident_id} by {operator or 'system'}",
            severity=original.severity,
            threat_type=original.threat_type,
            source="incident_logger",
            alert_ids=[incident_id],
            operator=operator,
            notes=new_notes,
        )
        return True

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the chain.

        Returns ``(True, [])`` if every record's hash matches its content and
        the previous-hash linkage is intact. Otherwise returns ``(False, errors)``.
        """
        errors: List[str] = []
        prev_hash = self.GENESIS_HASH
        seen_seq = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = IncidentRecord.from_dict(json.loads(line))
                    except Exception as exc:
                        errors.append(f"line {line_no}: parse error: {exc}")
                        break
                    if rec.seq != seen_seq + 1:
                        errors.append(
                            f"line {line_no}: seq gap/break (expected {seen_seq + 1}, got {rec.seq})"
                        )
                    if rec.previous_hash != prev_hash:
                        errors.append(
                            f"line {line_no}: previous_hash mismatch (expected {prev_hash[:16]}, "
                            f"got {rec.previous_hash[:16]})"
                        )
                    expected_hash = self._compute_hash(rec.to_payload_dict())
                    if rec.record_hash != expected_hash:
                        errors.append(
                            f"line {line_no}: record_hash mismatch (expected {expected_hash[:16]}, "
                            f"got {rec.record_hash[:16]})"
                        )
                    prev_hash = rec.record_hash
                    seen_seq = rec.seq
        except FileNotFoundError:
            return True, []  # empty chain is valid
        return (len(errors) == 0), errors

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_incidents(
        self,
        incident_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        threat_type: Optional[ThreatType] = None,
        source: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
    ) -> List[IncidentRecord]:
        """Query the incident log by various filters."""
        results: List[IncidentRecord] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = IncidentRecord.from_dict(json.loads(line))
                    except Exception:
                        continue
                    if incident_id and rec.id != incident_id:
                        continue
                    if severity and rec.severity != severity:
                        continue
                    if threat_type and rec.threat_type != threat_type:
                        continue
                    if source and rec.source != source:
                        continue
                    if since and rec.timestamp < since:
                        continue
                    if until and rec.timestamp > until:
                        continue
                    results.append(rec)
        except FileNotFoundError:
            return []
        return results[-limit:]

    def iterate(self) -> Iterator[IncidentRecord]:
        """Yield every record in the chain in order."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield IncidentRecord.from_dict(json.loads(line))
                    except Exception:
                        continue
        except FileNotFoundError:
            return

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_incidents(
        self,
        dest: str,
        fmt: str = "json",
        since: Optional[float] = None,
    ) -> int:
        """Export incident records to a file (JSON or JSONL).

        Returns the number of records exported.
        """
        records = self.query_incidents(since=since, limit=10**9)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump([r.to_dict() for r in records], fh, indent=2, default=str)
        elif fmt == "jsonl":
            with open(dest, "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r.to_dict(), default=str) + "\n")
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        logger.info("Exported %d incidents to %s (%s)", len(records), dest, fmt)
        return len(records)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        records = self.query_incidents(limit=10**9)
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for r in records:
            by_severity[r.severity.name] = by_severity.get(r.severity.name, 0) + 1
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
            by_type[r.threat_type.value] = by_type.get(r.threat_type.value, 0) + 1
        return {
            "total": len(records),
            "by_severity": by_severity,
            "by_status": by_status,
            "by_threat_type": by_type,
            "last_seq": self._last_seq,
        }


__all__ = ["IncidentLogger", "IncidentRecord"]
