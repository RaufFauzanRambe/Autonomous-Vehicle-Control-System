"""Software integrity baseline monitor.

The :class:`SoftwareIntegrityMonitor` maintains a baseline hash database for
every critical binary, shared library, and configuration file on the vehicle
compute platform. It periodically re-hashes those files and compares against
the baseline, emitting violations for any that have changed.

The baseline itself is signed (HMAC-SHA-256 over the JSON body) so that an
attacker who replaces both the file and the baseline is still detected
(unless they also possess the HMAC key, which lives in :class:`SecureStorage`).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .constants import Severity
from .utils import compute_hmac_sha256, compute_sha256, hex_encode, safe_compare
from .security_event_logger import SecurityEventLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class BaselineEntry:
    path: str
    sha256: str
    size: int
    mtime: float
    mode: int
    added_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrityViolation:
    path: str
    kind: str  # modified | missing | added | mode_changed
    expected_sha256: str = ""
    actual_sha256: str = ""
    expected_size: int = 0
    actual_size: int = 0
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# File reader abstraction (mockable for tests)
# --------------------------------------------------------------------------- #


FileReader = Callable[[str], Optional[bytes]]


def default_file_reader(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None


def file_stat(path: str) -> Optional[Dict[str, Any]]:
    try:
        st = os.stat(path)
        return {"size": st.st_size, "mtime": st.st_mtime, "mode": st.st_mode}
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #


class SoftwareIntegrityMonitor:
    """Builds and verifies a SHA-256 baseline of critical files."""

    def __init__(
        self,
        baseline_db_path: str,
        hmac_key: Optional[bytes] = None,
        file_reader: Optional[FileReader] = None,
        logger_: Optional[SecurityEventLogger] = None,
    ) -> None:
        self.baseline_db_path = Path(baseline_db_path)
        self.hmac_key = hmac_key or compute_sha256(b"avcs-integrity-default-key")
        self.file_reader = file_reader or default_file_reader
        self.event_logger = logger_
        self._baseline: Dict[str, BaselineEntry] = {}
        self._watch_paths: List[str] = []
        self._violations: List[IntegrityViolation] = []
        self._lock = threading.RLock()
        self._last_scan = 0.0
        self._load()

    # ------------------------------------------------------------------ #
    # Baseline management
    # ------------------------------------------------------------------ #

    def add_file(self, path: str) -> bool:
        """Add a single file to the baseline."""
        data = self.file_reader(path)
        if data is None:
            logger.warning("cannot add file to baseline: %s", path)
            return False
        stat = file_stat(path) or {"size": len(data), "mtime": 0.0, "mode": 0}
        with self._lock:
            self._baseline[path] = BaselineEntry(
                path=path,
                sha256=hex_encode(compute_sha256(data)),
                size=stat["size"],
                mtime=stat["mtime"],
                mode=stat["mode"],
            )
            if path not in self._watch_paths:
                self._watch_paths.append(path)
        return True

    def add_directory(self, dir_path: str, recursive: bool = True) -> int:
        """Add every regular file in *dir_path* to the baseline."""
        added = 0
        root = Path(dir_path)
        if not root.exists():
            logger.warning("directory %s does not exist", dir_path)
            return 0
        iterator = root.rglob("*") if recursive else root.iterdir()
        for p in iterator:
            if p.is_file() and not p.is_symlink():
                if self.add_file(str(p)):
                    added += 1
        logger.info("added %d files from %s to baseline", added, dir_path)
        return added

    def remove_file(self, path: str) -> bool:
        with self._lock:
            existed = self._baseline.pop(path, None) is not None
            if path in self._watch_paths:
                self._watch_paths.remove(path)
            return existed

    def build_baseline(self) -> int:
        """Re-hash every watched path and rebuild the baseline in memory."""
        with self._lock:
            paths = list(self._watch_paths)
        for path in paths:
            self.add_file(path)  # add_file overwrites existing entry
        self._save()
        return len(self._baseline)

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #

    def verify_integrity(self) -> List[IntegrityViolation]:
        """Re-hash all baseline files; return the list of violations."""
        new_violations: List[IntegrityViolation] = []
        with self._lock:
            baseline_snapshot = dict(self._baseline)
            self._violations.clear()

        for path, entry in baseline_snapshot.items():
            data = self.file_reader(path)
            if data is None:
                violation = IntegrityViolation(
                    path=path,
                    kind="missing",
                    expected_sha256=entry.sha256,
                    expected_size=entry.size,
                )
                new_violations.append(violation)
                continue
            stat = file_stat(path) or {"size": len(data), "mode": entry.mode}
            actual_hash = hex_encode(compute_sha256(data))
            if not safe_compare(bytes.fromhex(actual_hash), bytes.fromhex(entry.sha256)):
                new_violations.append(IntegrityViolation(
                    path=path,
                    kind="modified",
                    expected_sha256=entry.sha256,
                    actual_sha256=actual_hash,
                    expected_size=entry.size,
                    actual_size=stat["size"],
                ))
            elif stat["size"] != entry.size:
                new_violations.append(IntegrityViolation(
                    path=path,
                    kind="modified",
                    expected_sha256=entry.sha256,
                    actual_sha256=actual_hash,
                    expected_size=entry.size,
                    actual_size=stat["size"],
                ))
            elif stat["mode"] != entry.mode:
                new_violations.append(IntegrityViolation(
                    path=path,
                    kind="mode_changed",
                    expected_sha256=entry.sha256,
                    actual_sha256=actual_hash,
                    expected_size=entry.size,
                    actual_size=stat["size"],
                ))

        with self._lock:
            self._violations = new_violations
            self._last_scan = time.time()

        if new_violations and self.event_logger:
            self.event_logger.log_event(
                event_type="integrity.violation",
                severity=Severity.HIGH.value,
                source="software_integrity",
                message=f"{len(new_violations)} integrity violations detected",
                details={"violations": [v.to_dict() for v in new_violations]},
            )
        logger.info("integrity scan complete: %d violations", len(new_violations))
        return new_violations

    def get_violations(self) -> List[IntegrityViolation]:
        with self._lock:
            return list(self._violations)

    def get_baseline_size(self) -> int:
        with self._lock:
            return len(self._baseline)

    def get_last_scan(self) -> float:
        with self._lock:
            return self._last_scan

    def list_watched(self) -> List[str]:
        with self._lock:
            return sorted(self._watch_paths)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        with self._lock:
            entries = [e.to_dict() for e in self._baseline.values()]
        body = json.dumps({"entries": entries}, sort_keys=True)
        hmac_hex = hex_encode(compute_hmac_sha256(self.hmac_key, body.encode("utf-8")))
        payload = {"hmac": hmac_hex, "body": body}
        self.baseline_db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.baseline_db_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.baseline_db_path)

    def _load(self) -> None:
        if not self.baseline_db_path.exists():
            return
        try:
            payload = json.loads(self.baseline_db_path.read_text(encoding="utf-8"))
            body = payload["body"]
            expected_hmac = payload["hmac"]
            actual_hmac = hex_encode(compute_hmac_sha256(self.hmac_key, body.encode("utf-8")))
            if not safe_compare(bytes.fromhex(expected_hmac), bytes.fromhex(actual_hmac)):
                logger.error("baseline HMAC verification failed; refusing to load")
                return
            entries = json.loads(body)["entries"]
            for e in entries:
                entry = BaselineEntry(**e)
                self._baseline[entry.path] = entry
                if entry.path not in self._watch_paths:
                    self._watch_paths.append(entry.path)
            logger.info("loaded %d baseline entries from %s", len(entries), self.baseline_db_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load baseline: %s", exc)

    def verify_baseline_signature(self) -> bool:
        """Return True if the on-disk baseline HMAC is valid."""
        if not self.baseline_db_path.exists():
            return False
        try:
            payload = json.loads(self.baseline_db_path.read_text(encoding="utf-8"))
            expected = bytes.fromhex(payload["hmac"])
            actual = compute_hmac_sha256(self.hmac_key, payload["body"].encode("utf-8"))
            return safe_compare(expected, actual)
        except Exception:  # noqa: BLE001
            return False
