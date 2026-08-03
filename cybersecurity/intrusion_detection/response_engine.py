"""Automated response engine for the IDS.

The :class:`ResponseEngine` translates classified alerts into concrete
mitigation actions (block IP, disable ECU, isolate network, trigger lockdown,
...). Each action is represented by a :class:`ResponseAction`; user code
registers handlers (callables) per :class:`ResponseActionType` and the engine
applies them when an alert's severity triggers the configured policy.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .constants import (
    AlertSeverity,
    DEFAULT_RESPONSE_POLICY,
    ResponseActionType,
    ThreatType,
)
from .utils import TokenBucket, timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResponseAction:
    """A response action to execute (or that was executed)."""

    id: str
    action_type: ResponseActionType
    target: str  # e.g. "192.168.1.5", "can0", "ECU:BCM", "/proc/1234"
    severity: AlertSeverity
    triggered_by_alert: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # "pending" | "executing" | "success" | "failed" | "skipped"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "target": self.target,
            "severity": self.severity.name,
            "triggered_by_alert": self.triggered_by_alert,
            "parameters": dict(self.parameters),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


ResponseHandler = Callable[[ResponseAction], Any]


class ResponseEngine:
    """Maps alert severity → response actions, executes them with rate limiting.

    Parameters
    ----------
    enabled:
        Master switch. When False, no actions are executed.
    dry_run:
        When True, actions are recorded as ``skipped`` without execution.
    max_actions_per_min:
        Hard rate limit on action execution (token bucket).
    allowed_actions:
        Allow-list. If non-empty, only these action types may execute.
    blocked_actions:
        Block-list. Action types here will be skipped.
    cooldown_sec:
        Minimum time between two actions of the same type on the same target.
    policy:
        Mapping from severity → list of action types to execute.
    """

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        max_actions_per_min: int = 30,
        allowed_actions: Optional[List[ResponseActionType]] = None,
        blocked_actions: Optional[List[ResponseActionType]] = None,
        cooldown_sec: int = 60,
        policy: Optional[Dict[AlertSeverity, Tuple[ResponseActionType, ...]]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.dry_run = bool(dry_run)
        self.max_actions_per_min = int(max_actions_per_min)
        self.allowed_actions: List[ResponseActionType] = list(allowed_actions or [])
        self.blocked_actions: List[ResponseActionType] = list(blocked_actions or [])
        self.cooldown_sec = int(cooldown_sec)
        self.policy: Dict[AlertSeverity, Tuple[ResponseActionType, ...]] = (
            dict(policy) if policy else dict(DEFAULT_RESPONSE_POLICY)
        )
        self._handlers: Dict[ResponseActionType, ResponseHandler] = {}
        self._lock = threading.RLock()
        self._history: Deque[ResponseAction] = deque(maxlen=10_000)
        self._last_run: Dict[Tuple[str, str], float] = {}
        self._next_id = 1
        self._bucket = TokenBucket(
            capacity=float(self.max_actions_per_min),
            refill_rate=self.max_actions_per_min / 60.0,
        )
        self._stats = {"evaluated": 0, "executed": 0, "skipped": 0, "failed": 0}

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_response_action(
        self,
        action_type: ResponseActionType,
        handler: ResponseHandler,
    ) -> None:
        """Register a handler that executes ``action_type`` actions."""
        with self._lock:
            self._handlers[action_type] = handler
        logger.info("Registered handler for %s", action_type.value)

    def unregister_response_action(self, action_type: ResponseActionType) -> None:
        with self._lock:
            self._handlers.pop(action_type, None)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def set_policy(self, severity: AlertSeverity, actions: Tuple[ResponseActionType, ...]) -> None:
        with self._lock:
            self.policy[severity] = tuple(actions)

    def get_policy(self) -> Dict[AlertSeverity, Tuple[ResponseActionType, ...]]:
        with self._lock:
            return dict(self.policy)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_response(
        self,
        severity: AlertSeverity,
        target: str,
        alert_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[ResponseAction]:
        """Compute the list of response actions an alert of the given severity
        would trigger. Does **not** execute them — see :meth:`execute_response`.
        """
        actions: List[ResponseAction] = []
        for atype in self.policy.get(severity, ()):
            with self._lock:
                aid = f"resp-{self._next_id:08d}"
                self._next_id += 1
            actions.append(ResponseAction(
                id=aid,
                action_type=atype,
                target=target,
                severity=severity,
                triggered_by_alert=alert_id,
                parameters=dict(parameters or {}),
                dry_run=self.dry_run,
            ))
        return actions

    def execute_response(
        self,
        severity: AlertSeverity,
        target: str,
        alert_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[ResponseAction]:
        """Evaluate and execute the response actions for an alert.

        Returns the list of actions (with their final status).
        """
        if not self.enabled:
            logger.info("Response engine disabled; skipping response for %s", target)
            return []
        with self._lock:
            self._stats["evaluated"] += 1
        actions = self.evaluate_response(severity, target, alert_id, parameters)
        executed: List[ResponseAction] = []
        for action in actions:
            self._execute_one(action)
            executed.append(action)
        return executed

    def _execute_one(self, action: ResponseAction) -> None:
        # Allow-list / block-list
        if self.allowed_actions and action.action_type not in self.allowed_actions:
            self._skip(action, "action not in allow-list")
            return
        if action.action_type in self.blocked_actions:
            self._skip(action, "action in block-list")
            return
        # Cooldown
        key = (action.action_type.value, action.target)
        now = timestamp_now()
        with self._lock:
            last = self._last_run.get(key, 0.0)
            if now - last < self.cooldown_sec:
                self._skip(action, f"cooldown ({self.cooldown_sec - int(now - last)}s left)")
                return
        # Rate limit
        if not self._bucket.consume():
            self._skip(action, "rate-limit exceeded")
            return
        # Dry run
        if self.dry_run:
            action.status = "skipped"
            action.started_at = now
            action.finished_at = timestamp_now()
            action.result = "dry_run"
            self._record(action)
            return
        # Dispatch
        with self._lock:
            handler = self._handlers.get(action.action_type)
        action.status = "executing"
        action.started_at = now
        if handler is None:
            action.status = "failed"
            action.error = f"no handler registered for {action.action_type.value}"
            action.finished_at = timestamp_now()
            self._record(action)
            logger.warning("No handler for %s; action %s failed",
                           action.action_type.value, action.id)
            with self._lock:
                self._stats["failed"] += 1
            return
        try:
            action.result = handler(action)
            action.status = "success"
        except Exception as exc:
            action.status = "failed"
            action.error = str(exc)
            logger.error("Response handler %s raised: %s", action.action_type.value, exc)
        action.finished_at = timestamp_now()
        with self._lock:
            self._last_run[key] = action.finished_at or timestamp_now()
            if action.status == "success":
                self._stats["executed"] += 1
            else:
                self._stats["failed"] += 1
        self._record(action)

    def _skip(self, action: ResponseAction, reason: str) -> None:
        action.status = "skipped"
        action.error = reason
        action.started_at = timestamp_now()
        action.finished_at = action.started_at
        self._record(action)
        with self._lock:
            self._stats["skipped"] += 1
        logger.info("Skipping action %s (%s): %s",
                    action.id, action.action_type.value, reason)

    def _record(self, action: ResponseAction) -> None:
        with self._lock:
            self._history.append(action)

    # ------------------------------------------------------------------
    # Manual execution
    # ------------------------------------------------------------------

    def execute_action(
        self,
        action_type: ResponseActionType,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        alert_id: Optional[str] = None,
    ) -> ResponseAction:
        """Manually trigger a single action (bypassing the policy)."""
        with self._lock:
            aid = f"resp-{self._next_id:08d}"
            self._next_id += 1
        action = ResponseAction(
            id=aid, action_type=action_type, target=target,
            severity=severity, triggered_by_alert=alert_id,
            parameters=dict(parameters or {}), dry_run=self.dry_run,
        )
        self._execute_one(action)
        return action

    # ------------------------------------------------------------------
    # History & stats
    # ------------------------------------------------------------------

    def get_response_history(self, limit: int = 100) -> List[ResponseAction]:
        with self._lock:
            return list(self._history)[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            stats["history_size"] = len(self._history)
            stats["handlers_registered"] = len(self._handlers)
            return stats


__all__ = ["ResponseEngine", "ResponseAction", "ResponseHandler"]
