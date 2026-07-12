"""Safety-critical security monitor.

The :class:`SafetyMonitor` bridges cybersecurity and functional safety. It
tracks ISO 26262 ASIL-D safety goals (e.g. "the braking system shall never
apply unintended brake torque") and detects security events that could
compromise those goals (e.g. CAN bus tampering with brake-by-wire frames).

When a security event affects a safety goal, the monitor triggers a
degraded safety state (``SAFE`` -> ``DEGRADED`` -> ``LIMP_HOME`` ->
``FAIL_SAFE``) and notifies the chassis safety controller to take
appropriate action (e.g. switch to mechanical fallback, request minimum
risk maneuver).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .constants import (
    ASILLevel,
    ASIL_WEIGHT,
    CRITICAL_ECUS,
    SAFETY_CRITICAL_CAN_IDS,
    SAFETY_GOAL_BRAKING,
    SAFETY_GOAL_MOTION_CONTROL,
    SAFETY_GOAL_POWERTRAIN,
    SAFETY_GOAL_STEERING,
    Severity,
)
from .utils import format_timestamp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class SafetyState(str, Enum):
    """The vehicle's safety mode.

    Ordered from least restrictive (NORMAL) to most restrictive (FAIL_SAFE).
    """

    NORMAL = "normal"
    DEGRADED = "degraded"
    LIMP_HOME = "limp_home"
    FAIL_SAFE = "fail_safe"
    EMERGENCY_STOP = "emergency_stop"

    @classmethod
    def weight(cls, state: "SafetyState") -> int:
        order = {
            cls.NORMAL: 0,
            cls.DEGRADED: 1,
            cls.LIMP_HOME: 2,
            cls.FAIL_SAFE: 3,
            cls.EMERGENCY_STOP: 4,
        }
        return order[state]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class SafetyGoal:
    goal_id: str
    description: str
    asil: ASILLevel
    related_can_ids: List[int] = field(default_factory=list)
    related_ecus: List[str] = field(default_factory=list)
    satisfied: bool = True
    last_checked: float = field(default_factory=time.time)
    violations: int = 0

    @property
    def weight(self) -> int:
        return ASIL_WEIGHT.get(self.asil.value, 0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["asil"] = self.asil.value
        return d


@dataclass
class SafetyViolation:
    goal_id: str
    timestamp: str
    severity: Severity
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


SafetyStateCallback = Callable[[SafetyState, SafetyState], None]


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #


class SafetyMonitor:
    """Tracks ASIL safety goals and triggers safety-state degradation."""

    DEFAULT_GOALS: List[SafetyGoal] = [
        SafetyGoal(
            goal_id=SAFETY_GOAL_BRAKING,
            description="Brake-by-wire shall never apply unintended brake torque",
            asil=ASILLevel.D,
            related_can_ids=[0x100, 0x112],
            related_ecus=["BRAKES"],
        ),
        SafetyGoal(
            goal_id=SAFETY_GOAL_STEERING,
            description="Steering-by-wire shall not produce uncommanded wheel angles",
            asil=ASILLevel.D,
            related_can_ids=[0x101, 0x111],
            related_ecus=["STEERING"],
        ),
        SafetyGoal(
            goal_id=SAFETY_GOAL_POWERTRAIN,
            description="Powertrain shall not produce uncommanded propulsion torque",
            asil=ASILLevel.C,
            related_can_ids=[0x102],
            related_ecus=["POWERTRAIN"],
        ),
        SafetyGoal(
            goal_id=SAFETY_GOAL_MOTION_CONTROL,
            description="Vehicle motion controller shall maintain stable trajectory",
            asil=ASILLevel.D,
            related_can_ids=list(SAFETY_CRITICAL_CAN_IDS),
            related_ecus=list(CRITICAL_ECUS),
        ),
    ]

    def __init__(
        self,
        goals: Optional[List[SafetyGoal]] = None,
        logger=None,  # Optional[SecurityEventLogger]
    ) -> None:
        self._lock = threading.RLock()
        self._goals: Dict[str, SafetyGoal] = {g.goal_id: g for g in (goals or list(self.DEFAULT_GOALS))}
        self._violations: List[SafetyViolation] = []
        self._state = SafetyState.NORMAL
        self._callbacks: List[SafetyStateCallback] = []
        self._state_history: List[Dict[str, Any]] = [
            {"timestamp": format_timestamp(), "state": self._state.value, "reason": "init"}
        ]
        self.event_logger = logger

    # ------------------------------------------------------------------ #
    # Goal management
    # ------------------------------------------------------------------ #

    def register_safety_goal(self, goal: SafetyGoal) -> None:
        with self._lock:
            self._goals[goal.goal_id] = goal
        logger.info("registered safety goal %s (ASIL %s)", goal.goal_id, goal.asil.value)

    def get_goals(self) -> List[SafetyGoal]:
        with self._lock:
            return list(self._goals.values())

    # ------------------------------------------------------------------ #
    # Violation checking
    # ------------------------------------------------------------------ #

    def check_safety_violation(
        self,
        event_type: str,
        severity: Severity,
        source: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SafetyViolation]:
        """Check whether a security event impacts a registered safety goal.

        Returns the :class:`SafetyViolation` if one was created, else ``None``.
        """
        details = details or {}
        affected_goals: List[SafetyGoal] = []
        with self._lock:
            for goal in self._goals.values():
                if source in goal.related_ecus:
                    affected_goals.append(goal)
                    continue
                # CAN ID hint in details
                can_id = details.get("can_id")
                if can_id is not None and can_id in goal.related_can_ids:
                    affected_goals.append(goal)

        if not affected_goals:
            return None

        # Pick the most safety-critical affected goal
        target_goal = max(affected_goals, key=lambda g: g.weight)
        violation = SafetyViolation(
            goal_id=target_goal.goal_id,
            timestamp=format_timestamp(),
            severity=severity,
            description=f"security event '{event_type}' from '{source}' may compromise {target_goal.goal_id}: {target_goal.description}",
            details=details,
        )
        with self._lock:
            target_goal.satisfied = False
            target_goal.violations += 1
            target_goal.last_checked = time.time()
            self._violations.append(violation)

        # Determine the new safety state
        new_state = self._compute_new_state(severity, target_goal)
        if SafetyState.weight(new_state) > SafetyState.weight(self._state):
            self._set_state(new_state, reason=f"safety violation on {target_goal.goal_id}")

        if self.event_logger:
            self.event_logger.log_event(
                event_type="safety.violation",
                severity=severity.value,
                source="safety_monitor",
                message=violation.description,
                details=violation.to_dict(),
            )
        logger.warning("safety violation: %s", violation.description)
        return violation

    def _compute_new_state(self, severity: Severity, goal: SafetyGoal) -> SafetyState:
        """Map (severity, ASIL) to a target safety state."""
        if severity == Severity.CRITICAL and goal.asil in (ASILLevel.D, ASILLevel.C):
            return SafetyState.FAIL_SAFE
        if severity == Severity.HIGH and goal.asil == ASILLevel.D:
            return SafetyState.LIMP_HOME
        if severity == Severity.HIGH:
            return SafetyState.DEGRADED
        if severity == Severity.MEDIUM:
            return SafetyState.DEGRADED
        return SafetyState.NORMAL

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #

    def _set_state(self, new_state: SafetyState, reason: str) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
            self._state_history.append({
                "timestamp": format_timestamp(),
                "state": new_state.value,
                "reason": reason,
            })
            callbacks = list(self._callbacks)
        logger.warning("safety state transition: %s -> %s (reason=%s)", old.value, new_state.value, reason)
        if self.event_logger:
            self.event_logger.log_event(
                event_type="safety.state_change",
                severity=Severity.HIGH.value,
                source="safety_monitor",
                message=f"safety state {old.value} -> {new_state.value}: {reason}",
                details={"old": old.value, "new": new_state.value},
            )
        for cb in callbacks:
            try:
                cb(old, new_state)
            except Exception:  # noqa: BLE001
                logger.exception("safety state callback raised")

    def trigger_safe_state(self, reason: str = "manual trigger") -> SafetyState:
        """Force the most restrictive safety state immediately."""
        self._set_state(SafetyState.FAIL_SAFE, reason=reason)
        return self._state

    def trigger_emergency_stop(self, reason: str = "emergency") -> SafetyState:
        self._set_state(SafetyState.EMERGENCY_STOP, reason=reason)
        return self._state

    def reset_to_normal(self, reason: str = "all clear") -> bool:
        """Return to NORMAL only if all goals are satisfied."""
        with self._lock:
            if not all(g.satisfied for g in self._goals.values()):
                logger.info("cannot reset: not all goals satisfied")
                return False
        self._set_state(SafetyState.NORMAL, reason=reason)
        return True

    def acknowledge_violation(self, idx: int) -> bool:
        with self._lock:
            if 0 <= idx < len(self._violations):
                self._violations[idx].acknowledged = True
                return True
            return False

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_safety_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "goals": [g.to_dict() for g in self._goals.values()],
                "violations": [v.to_dict() for v in self._violations],
                "history": list(self._state_history),
            }

    def get_state(self) -> SafetyState:
        with self._lock:
            return self._state

    def get_violations(self) -> List[SafetyViolation]:
        with self._lock:
            return list(self._violations)

    def add_state_callback(self, cb: SafetyStateCallback) -> None:
        self._callbacks.append(cb)
