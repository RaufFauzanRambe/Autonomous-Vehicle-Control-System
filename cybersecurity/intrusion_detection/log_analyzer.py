"""Log analyzer for OS, application, and CAN logs.

The :class:`LogAnalyzer` parses textual log lines (syslog, journald exports,
auth.log, application logs) and matches them against a set of named patterns
that surface:

* Failed-login bursts (account brute force).
* Privilege escalation (sudo, su, setuid transitions).
* Unexpected process spawns.
* Kernel module loading / unloading.
* Service crashes and restarts.
* Indicator strings (IoCs) in app logs.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .constants import AlertSeverity, ThreatType
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LogPattern:
    """A regex pattern that flags suspicious log lines."""

    id: str
    name: str
    pattern: str
    severity: AlertSeverity = AlertSeverity.MEDIUM
    threat_type: ThreatType = ThreatType.INTRUSION
    description: str = ""
    enabled: bool = True
    _compiled: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            self._compiled = re.compile(self.pattern)
        except re.error as exc:
            logger.error("Invalid log pattern %s: %s", self.id, exc)
            self.enabled = False

    def matches(self, line: str) -> Optional[Dict[str, str]]:
        if not self.enabled or self._compiled is None:
            return None
        m = self._compiled.search(line)
        if not m:
            return None
        return m.groupdict()


@dataclass
class LogFinding:
    """A finding raised by the log analyzer."""

    timestamp: float
    pattern_id: str
    severity: AlertSeverity
    threat_type: ThreatType
    description: str
    source: str
    raw_line: str
    groups: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class LogAnalyzer:
    """Pattern-based log scanner with burst detection."""

    def __init__(self) -> None:
        self._patterns: Dict[str, LogPattern] = {}
        self._lock = threading.RLock()
        self._findings: Deque[LogFinding] = deque(maxlen=50_000)
        self._finding_callbacks: List[Callable[[LogFinding], None]] = []
        self._failed_login_buckets: Dict[Tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=5000)
        )
        self._failed_login_threshold = 5
        self._failed_login_window = 60.0
        self._stats = {
            "lines_parsed": 0,
            "findings_raised": 0,
            "files_scanned": 0,
        }
        self._install_default_patterns()

    # ------------------------------------------------------------------
    # Pattern management
    # ------------------------------------------------------------------

    def add_pattern(self, pattern: LogPattern) -> bool:
        if not pattern.id:
            raise ValueError("LogPattern requires an id")
        with self._lock:
            self._patterns[pattern.id] = pattern
            return True

    def remove_pattern(self, pattern_id: str) -> bool:
        with self._lock:
            return self._patterns.pop(pattern_id, None) is not None

    def list_patterns(self, enabled_only: bool = False) -> List[LogPattern]:
        with self._lock:
            patterns = list(self._patterns.values())
        if enabled_only:
            patterns = [p for p in patterns if p.enabled]
        return patterns

    def _install_default_patterns(self) -> None:
        defaults = [
            LogPattern(
                id="auth-failed-login",
                name="Failed SSH/password login",
                pattern=r"(?P<timestamp>\w{3}\s+\d+\s[\d:]+).*"
                        r"(?:Failed password|authentication failure|Invalid user).*"
                        r"(?:from|for)\s+(?P<user>\S+)",
                severity=AlertSeverity.LOW,
                threat_type=ThreatType.INTRUSION,
                description="Failed login attempt",
            ),
            LogPattern(
                id="auth-sudo-success",
                name="Successful sudo invocation",
                pattern=r"(?P<timestamp>\w{3}\s+\d+\s[\d:]+).*"
                        r"(?P<user>\S+)\s+:\s+TTY=\S+\s+;\s+PWD=\S+\s+;\s+USER=root\s+;\s+COMMAND=(?P<command>.+)",
                severity=AlertSeverity.LOW,
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                description="sudo command execution",
            ),
            LogPattern(
                id="auth-su-root",
                name="Root shell via su",
                pattern=r"su\[\d+\]:\s+(?P<user>\S+)\s+\S+\s+root",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                description="Privilege escalation via su to root",
            ),
            LogPattern(
                id="kernel-module-load",
                name="Kernel module load",
                pattern=r"kernel:\s+\S+\s+loading out-of-tree module",
                severity=AlertSeverity.MEDIUM,
                threat_type=ThreatType.SUPPLY_CHAIN,
                description="Out-of-tree kernel module load",
            ),
            LogPattern(
                id="process-spawn-shell",
                name="Process spawned a shell",
                pattern=r"(?P<proc>\S+).*execved?\s+.*/bin/(?:bash|sh)\b",
                severity=AlertSeverity.MEDIUM,
                threat_type=ThreatType.INTRUSION,
                description="Process spawned a shell (possible RCE)",
            ),
            LogPattern(
                id="service-crash",
                name="Service crashed and will be restarted",
                pattern=r"systemd\[\d+\]:\s+(?P<service>\S+)\s+failed.*"
                        r"(?:Restarting|Main process exited)",
                severity=AlertSeverity.MEDIUM,
                threat_type=ThreatType.DOS,
                description="Service crash detected",
            ),
            LogPattern(
                id="app-ioc-string",
                name="Application IoC string",
                pattern=r"\b(meterpreter|mimikatz|cobalt\s*strike|reverse_tcp|nc\s+-l)\b",
                severity=AlertSeverity.HIGH,
                threat_type=ThreatType.MALWARE,
                description="Known IoC string in application log",
            ),
        ]
        for p in defaults:
            self.add_pattern(p)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_log(self, line: str, source: str = "unknown") -> List[LogFinding]:
        """Parse a single log line and return any findings."""
        findings: List[LogFinding] = []
        with self._lock:
            patterns = list(self._patterns.values())
            self._stats["lines_parsed"] += 1
        for pat in patterns:
            groups = pat.matches(line)
            if groups is None:
                continue
            finding = LogFinding(
                timestamp=timestamp_now(),
                pattern_id=pat.id,
                severity=pat.severity,
                threat_type=pat.threat_type,
                description=pat.description,
                source=source,
                raw_line=line.rstrip("\n")[:500],
                groups=groups,
            )
            findings.append(finding)
            # Burst detection on failed logins
            if pat.id == "auth-failed-login" and "user" in groups:
                self._record_failed_login(groups["user"], source)
        for f in findings:
            self._raise_finding(f)
        return findings

    def _record_failed_login(self, user: str, source: str) -> None:
        key = (user, source)
        now = timestamp_now()
        with self._lock:
            bucket = self._failed_login_buckets[key]
            bucket.append(now)
            cutoff = now - self._failed_login_window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._failed_login_threshold:
                self._raise_finding(LogFinding(
                    timestamp=now,
                    pattern_id="auth-brute-force",
                    severity=AlertSeverity.HIGH,
                    threat_type=ThreatType.INTRUSION,
                    description=f"Brute force on user '{user}' ({len(bucket)} failures in {self._failed_login_window}s)",
                    source=source,
                    raw_line="",
                    groups={"user": user, "count": str(len(bucket))},
                ))
                # Reset to avoid spamming
                bucket.clear()

    # ------------------------------------------------------------------
    # Bulk scanning
    # ------------------------------------------------------------------

    def scan_logs(self, paths: List[str], max_lines: Optional[int] = None) -> List[LogFinding]:
        """Scan one or more log files; return all findings."""
        all_findings: List[LogFinding] = []
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning("Log file %s does not exist", path)
                continue
            try:
                with self._lock:
                    self._stats["files_scanned"] += 1
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if max_lines is not None and i >= max_lines:
                            break
                        findings = self.parse_log(line, source=str(path))
                        all_findings.extend(findings)
            except OSError as exc:
                logger.error("Cannot read log file %s: %s", path, exc)
        logger.info("Scanned %d log file(s), %d finding(s) raised",
                    len(paths), len(all_findings))
        return all_findings

    def scan_lines(self, lines: List[str], source: str = "stream") -> List[LogFinding]:
        """Scan a list of log lines (useful for in-memory log streams)."""
        out: List[LogFinding] = []
        for line in lines:
            out.extend(self.parse_log(line, source=source))
        return out

    # ------------------------------------------------------------------
    # Findings & stats
    # ------------------------------------------------------------------

    def register_finding_callback(self, cb: Callable[[LogFinding], None]) -> None:
        with self._lock:
            self._finding_callbacks.append(cb)

    def _raise_finding(self, finding: LogFinding) -> None:
        with self._lock:
            self._findings.append(finding)
            self._stats["findings_raised"] += 1
            cbs = list(self._finding_callbacks)
        for cb in cbs:
            try:
                cb(finding)
            except Exception as exc:
                logger.error("Finding callback raised: %s", exc)

    def get_findings(self, limit: int = 100, since: Optional[float] = None) -> List[LogFinding]:
        with self._lock:
            findings = list(self._findings)
        if since is not None:
            findings = [f for f in findings if f.timestamp >= since]
        return findings[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


__all__ = ["LogAnalyzer", "LogPattern", "LogFinding"]
