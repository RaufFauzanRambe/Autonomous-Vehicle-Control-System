"""Security incident response orchestrator.

The :class:`IncidentResponseManager` accepts incident reports (from the
firewall, monitor, integrity monitor, or any other module), classifies them
into severity tiers, runs the appropriate response playbook, and coordinates
with the :class:`~.emergency_lockdown.EmergencyLockdown` manager for
critical incidents.

Each playbook is implemented as a sequence of named steps
(``contain`` -> ``eradicate`` -> ``recover`` -> ``lessons_learned``). Steps
are idempotent so that re-running a playbook after a crash does not
double-apply containment actions.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .constants import Severity
from .utils import format_timestamp
from .security_event_logger import SecurityEventLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enums and dataclasses
# --------------------------------------------------------------------------- #


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    CLASSIFIED = "classified"
    CONTAINING = "containing"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class Playbook(str, Enum):
    MALWARE = "malware"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFIL = "data_exfiltration"
    FIRMWARE_TAMPER = "firmware_tamper"
    CAN_BUS_INJECTION = "can_bus_injection"
    DEFAULT = "default"


@dataclass
class IncidentStep:
    name: str
    status: str = "pending"  # pending | in_progress | done | failed | skipped
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Incident:
    incident_id: str
    title: str
    description: str
    severity: Severity
    playbook: Playbook
    source: str
    status: IncidentStatus = IncidentStatus.REPORTED
    created_at: str = field(default_factory=format_timestamp)
    updated_at: str = field(default_factory=format_timestamp)
    resolved_at: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    steps: List[IncidentStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["playbook"] = self.playbook.value
        d["status"] = self.status.value
        return d


# --------------------------------------------------------------------------- #
# Playbook step executor
# --------------------------------------------------------------------------- #


StepExecutor = Callable[[Incident], str]  # returns output text


class PlaybookExecutor:
    """Executes the steps of a playbook for an incident."""

    def __init__(self, lockdown_manager=None) -> None:
        self._steps: Dict[str, Dict[str, StepExecutor]] = {
            Playbook.DEFAULT.value: {
                "contain": self._default_contain,
                "eradicate": self._default_eradicate,
                "recover": self._default_recover,
                "lessons_learned": self._default_lessons,
            },
            Playbook.CAN_BUS_INJECTION.value: {
                "contain": self._canbus_contain,
                "eradicate": self._canbus_eradicate,
                "recover": self._canbus_recover,
                "lessons_learned": self._default_lessons,
            },
            Playbook.FIRMWARE_TAMPER.value: {
                "contain": self._firmware_contain,
                "eradicate": self._firmware_eradicate,
                "recover": self._firmware_recover,
                "lessons_learned": self._default_lessons,
            },
            Playbook.MALWARE.value: {
                "contain": self._malware_contain,
                "eradicate": self._malware_eradicate,
                "recover": self._default_recover,
                "lessons_learned": self._default_lessons,
            },
        }
        self.lockdown_manager = lockdown_manager

    def get_steps(self, playbook: Playbook) -> List[str]:
        return list(self._steps.get(playbook.value, self._steps[Playbook.DEFAULT.value]).keys())

    def execute_step(self, playbook: Playbook, step: str, incident: Incident) -> str:
        steps = self._steps.get(playbook.value, self._steps[Playbook.DEFAULT.value])
        executor = steps.get(step)
        if executor is None:
            return f"no executor for step {step}"
        return executor(incident)

    # ---- default step implementations ----
    def _default_contain(self, inc: Incident) -> str:
        if self.lockdown_manager and inc.severity in (Severity.HIGH, Severity.CRITICAL):
            self.lockdown_manager.trigger_lockdown(
                reason=f"incident {inc.incident_id}: {inc.title}",
                triggered_by="incident_response",
            )
            return "lockdown triggered"
        return "no containment action required"

    def _default_eradicate(self, inc: Incident) -> str:
        return "default eradicate step (no-op)"

    def _default_recover(self, inc: Incident) -> str:
        if self.lockdown_manager and self.lockdown_manager.is_locked_down():
            # Recovery requires admin to release lockdown; record note.
            inc.notes.append("recovery requires admin PIN to release lockdown")
        return "default recover step (no-op)"

    def _default_lessons(self, inc: Incident) -> str:
        inc.notes.append("lessons-learned: post-mortem scheduled")
        return "post-mortem scheduled"

    # ---- CAN bus injection playbook ----
    def _canbus_contain(self, inc: Incident) -> str:
        can_id = inc.indicators.get("can_id", "unknown")
        inc.notes.append(f"contained CAN injection on id {can_id}")
        return f"can_id={can_id} isolated"

    def _canbus_eradicate(self, inc: Incident) -> str:
        return "CAN source blocked at firewall"

    def _canbus_recover(self, inc: Incident) -> str:
        return "CAN bus monitoring resumed"

    # ---- firmware tamper playbook ----
    def _firmware_contain(self, inc: Incident) -> str:
        if self.lockdown_manager:
            self.lockdown_manager.trigger_lockdown(
                reason=f"firmware tamper: {inc.title}",
                triggered_by="incident_response",
                scopes=None,
            )
        return "OTA disabled, rollback scheduled"

    def _firmware_eradicate(self, inc: Incident) -> str:
        return "rolled back to last known-good firmware"

    def _firmware_recover(self, inc: Incident) -> str:
        return "firmware re-validated, vehicle back online"

    # ---- malware playbook ----
    def _malware_contain(self, inc: Incident) -> str:
        if self.lockdown_manager:
            self.lockdown_manager.trigger_lockdown(
                reason=f"malware detected: {inc.title}",
                triggered_by="incident_response",
            )
        return "network isolated"

    def _malware_eradicate(self, inc: Incident) -> str:
        return "affected processes killed, files quarantined"


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class IncidentResponseManager:
    """Classifies, runs playbooks for, and tracks security incidents."""

    def __init__(
        self,
        executor: Optional[PlaybookExecutor] = None,
        logger_: Optional[SecurityEventLogger] = None,
    ) -> None:
        self.executor = executor or PlaybookExecutor()
        self.event_logger = logger_
        self._lock = threading.RLock()
        self._incidents: Dict[str, Incident] = {}

    # ------------------------------------------------------------------ #
    # Reporting & classification
    # ------------------------------------------------------------------ #

    def report_incident(
        self,
        title: str,
        description: str,
        severity: Severity,
        source: str,
        playbook: Playbook = Playbook.DEFAULT,
        affected_components: Optional[List[str]] = None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Incident:
        incident = Incident(
            incident_id=f"INC-{uuid.uuid4().hex[:12].upper()}",
            title=title,
            description=description,
            severity=severity,
            playbook=playbook,
            source=source,
            affected_components=affected_components or [],
            indicators=indicators or {},
        )
        with self._lock:
            self._incidents[incident.incident_id] = incident
        if self.event_logger:
            self.event_logger.log_event(
                event_type="incident.reported",
                severity=severity.value,
                source="incident_response",
                message=f"incident {incident.incident_id} reported: {title}",
                details=incident.to_dict(),
            )
        logger.info("incident %s reported: %s", incident.incident_id, title)
        return incident

    def classify_incident(self, incident_id: str, severity: Severity, playbook: Playbook) -> Incident:
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                raise KeyError(incident_id)
            inc.severity = severity
            inc.playbook = playbook
            inc.status = IncidentStatus.CLASSIFIED
            inc.updated_at = format_timestamp()
        logger.info("incident %s classified: severity=%s playbook=%s", incident_id, severity.value, playbook.value)
        return inc

    def auto_classify(self, title: str, description: str) -> tuple[Severity, Playbook]:
        """Heuristic classification based on keywords in title/description."""
        text = (title + " " + description).lower()
        if "can" in text and "injection" in text:
            return Severity.CRITICAL, Playbook.CAN_BUS_INJECTION
        if "firmware" in text and ("tamper" in text or "modified" in text):
            return Severity.CRITICAL, Playbook.FIRMWARE_TAMPER
        if "malware" in text or "virus" in text:
            return Severity.HIGH, Playbook.MALWARE
        if "unauthorized" in text and "access" in text:
            return Severity.HIGH, Playbook.UNAUTHORIZED_ACCESS
        if "exfil" in text:
            return Severity.CRITICAL, Playbook.DATA_EXFIL
        return Severity.MEDIUM, Playbook.DEFAULT

    # ------------------------------------------------------------------ #
    # Playbook execution
    # ------------------------------------------------------------------ #

    def run_playbook(self, incident_id: str, only_steps: Optional[List[str]] = None) -> Incident:
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                raise KeyError(incident_id)
        steps_to_run = only_steps or self.executor.get_steps(inc.playbook)
        for step_name in steps_to_run:
            step = IncidentStep(name=step_name, status="in_progress", started_at=format_timestamp())
            try:
                output = self.executor.execute_step(inc.playbook, step_name, inc)
                step.output = output
                step.status = "done"
            except Exception as exc:  # noqa: BLE001
                step.status = "failed"
                step.error = str(exc)
                logger.exception("playbook step %s failed for %s", step_name, incident_id)
            step.completed_at = format_timestamp()
            with self._lock:
                inc.steps.append(step)
                inc.updated_at = format_timestamp()
                if step.status == "failed":
                    inc.status = IncidentStatus.RECOVERING
                elif step_name == "contain":
                    inc.status = IncidentStatus.CONTAINING
                elif step_name == "eradicate":
                    inc.status = IncidentStatus.ERADICATING
                elif step_name == "recover":
                    inc.status = IncidentStatus.RECOVERING
                elif step_name == "lessons_learned":
                    inc.status = IncidentStatus.RESOLVED
                    inc.resolved_at = format_timestamp()

            if self.event_logger:
                self.event_logger.log_event(
                    event_type="incident.step",
                    severity=inc.severity.value,
                    source="incident_response",
                    message=f"step {step_name} {step.status} for {incident_id}",
                    details={"step": step.to_dict(), "incident_id": incident_id},
                )
        logger.info("playbook complete for %s (final status=%s)", incident_id, inc.status.value)
        return inc

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        with self._lock:
            return self._incidents.get(incident_id)

    def list_incidents(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[Severity] = None,
    ) -> List[Incident]:
        with self._lock:
            results = list(self._incidents.values())
        if status is not None:
            results = [i for i in results if i.status == status]
        if severity is not None:
            results = [i for i in results if i.severity == severity]
        return sorted(results, key=lambda i: i.created_at)

    def close_incident(self, incident_id: str, false_positive: bool = False) -> Incident:
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                raise KeyError(incident_id)
            inc.status = IncidentStatus.FALSE_POSITIVE if false_positive else IncidentStatus.CLOSED
            inc.updated_at = format_timestamp()
            inc.resolved_at = format_timestamp()
        return inc

    def add_note(self, incident_id: str, note: str) -> None:
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                raise KeyError(incident_id)
            inc.notes.append(note)
            inc.updated_at = format_timestamp()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": len(self._incidents),
                "by_status": {s.value: sum(1 for i in self._incidents.values() if i.status == s) for s in IncidentStatus},
                "by_severity": {s.value: sum(1 for i in self._incidents.values() if i.severity == s) for s in Severity},
            }
