"""Attack signature database with regex / byte-pattern / behavioral matching.

The :class:`AttackSignatureDB` holds an in-memory collection of
:class:`AttackSignature` records and supports fast matching against event
payloads, byte buffers, or structured event metadata. Records can be imported
from and exported to JSON for sharing across fleet vehicles or with a SIEM.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .constants import AlertSeverity, ThreatType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AttackSignature:
    """A single attack signature (rule).

    A signature is considered to match an event when:

    * For ``pattern_type == "regex"``: the ``pattern`` (compiled) is found
      anywhere in the textual representation of the event payload.
    * For ``pattern_type == "bytes"``: the raw byte pattern (parsed from a
      hex string) appears in the byte buffer.
    * For ``pattern_type == "behavior"``: the ``pattern`` is a JSON-encoded
      expression of field matches evaluated against an event dict.
    """

    id: str
    name: str
    pattern: str
    severity: AlertSeverity = AlertSeverity.MEDIUM
    description: str = ""
    threat_type: ThreatType = ThreatType.UNKNOWN
    pattern_type: str = "regex"  # "regex" | "bytes" | "behavior"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    false_positive_rate: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    # Cached compiled forms (not serialised)
    _compiled_regex: Optional[re.Pattern] = field(default=None, repr=False, compare=False)
    _compiled_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)
    _behavior_spec: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._compile()

    def _compile(self) -> None:
        try:
            if self.pattern_type == "regex":
                self._compiled_regex = re.compile(self.pattern, re.MULTILINE | re.DOTALL)
            elif self.pattern_type == "bytes":
                # Allow spaces / 0x prefixes; strip them.
                cleaned = self.pattern.replace(" ", "").replace("0x", "")
                self._compiled_bytes = bytes.fromhex(cleaned)
            elif self.pattern_type == "behavior":
                self._behavior_spec = json.loads(self.pattern)
            else:
                logger.warning("Unknown pattern_type %r for signature %s", self.pattern_type, self.id)
        except (re.error, ValueError, json.JSONDecodeError) as exc:
            logger.error("Failed to compile signature %s: %s", self.id, exc)
            self.enabled = False

    def matches(self, candidate: Union[str, bytes, Dict[str, Any]]) -> bool:
        """Test whether this signature matches the supplied candidate."""
        if not self.enabled:
            return False
        try:
            if self.pattern_type == "regex" and self._compiled_regex is not None:
                if isinstance(candidate, bytes):
                    candidate = candidate.decode("latin-1", errors="replace")
                return self._compiled_regex.search(str(candidate)) is not None
            if self.pattern_type == "bytes" and self._compiled_bytes is not None:
                data = candidate if isinstance(candidate, bytes) else str(candidate).encode("utf-8")
                return self._compiled_bytes in data
            if self.pattern_type == "behavior" and self._behavior_spec is not None:
                if not isinstance(candidate, dict):
                    return False
                return _match_behavior(self._behavior_spec, candidate)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Signature %s match error: %s", self.id, exc)
            return False
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this signature to a JSON-friendly dict."""
        d = asdict(self)
        # Drop private cached fields
        for k in list(d.keys()):
            if k.startswith("_"):
                d.pop(k)
        d["severity"] = self.severity.name
        d["threat_type"] = self.threat_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackSignature":
        """Construct a signature from a serialized dict."""
        data = dict(data)
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = AlertSeverity.from_str(data["severity"])
        if "threat_type" in data and isinstance(data["threat_type"], str):
            try:
                data["threat_type"] = ThreatType(data["threat_type"])
            except ValueError:
                data["threat_type"] = ThreatType.UNKNOWN
        # Strip unknown keys defensively
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def _match_behavior(spec: Dict[str, Any], event: Dict[str, Any]) -> bool:
    """Evaluate a simple behavioral spec against an event dict.

    Spec format::

        {"all": [{"field": "can_id", "op": "eq", "value": 0x123},
                 {"field": "dlc", "op": "gt", "value": 5}]}
        {"any": [...]}
        {"field": "src_ip", "op": "in", "value": ["10.0.0.1","10.0.0.2"]}
    """
    if "all" in spec:
        return all(_match_behavior(s, event) for s in spec["all"])
    if "any" in spec:
        return any(_match_behavior(s, event) for s in spec["any"])
    field_name = spec.get("field")
    if field_name is None:
        return False
    actual = event.get(field_name)
    op = spec.get("op", "eq")
    expected = spec.get("value")
    try:
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "gt":
            return actual is not None and actual > expected
        if op == "gte":
            return actual is not None and actual >= expected
        if op == "lt":
            return actual is not None and actual < expected
        if op == "lte":
            return actual is not None and actual <= expected
        if op == "in":
            return actual in (expected or [])
        if op == "contains":
            return expected in (actual or "")
        if op == "regex":
            return re.search(expected, str(actual)) is not None
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class AttackSignatureDB:
    """In-memory, thread-safe database of :class:`AttackSignature` records."""

    def __init__(self) -> None:
        self._signatures: Dict[str, AttackSignature] = {}
        self._lock = threading.RLock()
        self._match_counter: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_signature(self, sig: AttackSignature) -> bool:
        """Add or replace a signature. Returns True on success."""
        if not isinstance(sig, AttackSignature):
            raise TypeError("Expected AttackSignature instance")
        with self._lock:
            if not sig.id:
                logger.error("Signature missing id; refusing to add")
                return False
            self._signatures[sig.id] = sig
            self._match_counter.setdefault(sig.id, 0)
            logger.debug("Added signature %s (%s)", sig.id, sig.name)
            return True

    def remove_signature(self, sig_id: str) -> bool:
        """Remove a signature by id. Returns True if it existed."""
        with self._lock:
            existed = self._signatures.pop(sig_id, None) is not None
            self._match_counter.pop(sig_id, None)
            return existed

    def get_signature(self, sig_id: str) -> Optional[AttackSignature]:
        """Retrieve a signature by id."""
        with self._lock:
            return self._signatures.get(sig_id)

    def list_signatures(self, enabled_only: bool = False) -> List[AttackSignature]:
        """Return all (optionally enabled-only) signatures as a list."""
        with self._lock:
            sigs = list(self._signatures.values())
        if enabled_only:
            sigs = [s for s in sigs if s.enabled]
        return sigs

    def __len__(self) -> int:
        with self._lock:
            return len(self._signatures)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        candidate: Union[str, bytes, Dict[str, Any]],
        threat_type_filter: Optional[ThreatType] = None,
    ) -> List[AttackSignature]:
        """Return all signatures matching ``candidate``."""
        matches: List[AttackSignature] = []
        with self._lock:
            sigs = list(self._signatures.values())
        for sig in sigs:
            if not sig.enabled:
                continue
            if threat_type_filter is not None and sig.threat_type != threat_type_filter:
                continue
            if sig.matches(candidate):
                matches.append(sig)
                with self._lock:
                    self._match_counter[sig.id] = self._match_counter.get(sig.id, 0) + 1
        if matches:
            logger.debug("Matched %d signatures for candidate", len(matches))
        return matches

    def match_any(self, candidates: Iterable[Union[str, bytes, Dict[str, Any]]]) -> List[AttackSignature]:
        """Return all signatures matching ANY of the provided candidates."""
        out: List[AttackSignature] = []
        seen_ids: set = set()
        for cand in candidates:
            for sig in self.match(cand):
                if sig.id not in seen_ids:
                    seen_ids.add(sig.id)
                    out.append(sig)
        return out

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    def import_signatures(self, source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """Import signatures from a file path or parsed JSON structure.

        Returns the number of signatures successfully imported.
        """
        data: Any
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Signature file not found: {path}")
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = source
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise TypeError("Signatures file must contain a list or single object")
        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                sig = AttackSignature.from_dict(item)
                if self.add_signature(sig):
                    count += 1
            except Exception as exc:
                logger.error("Failed to import signature: %s", exc)
        logger.info("Imported %d signatures", count)
        return count

    def export_signatures(self, dest: Optional[Union[str, Path]] = None) -> Union[str, List[Dict[str, Any]]]:
        """Export signatures to a JSON file (if dest provided) or return a list."""
        with self._lock:
            data = [s.to_dict() for s in self._signatures.values()]
        if dest is None:
            return data
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        logger.info("Exported %d signatures to %s", len(data), path)
        return str(path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the database and its usage."""
        with self._lock:
            by_severity: Dict[str, int] = {}
            by_type: Dict[str, int] = {}
            for s in self._signatures.values():
                by_severity[s.severity.name] = by_severity.get(s.severity.name, 0) + 1
                by_type[s.threat_type.value] = by_type.get(s.threat_type.value, 0) + 1
            return {
                "total": len(self._signatures),
                "by_severity": by_severity,
                "by_threat_type": by_type,
                "match_counts": dict(self._match_counter),
            }


__all__ = ["AttackSignature", "AttackSignatureDB"]
