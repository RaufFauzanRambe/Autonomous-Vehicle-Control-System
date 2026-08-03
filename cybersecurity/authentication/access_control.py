"""
access_control.py
=================

Attribute-Based + Role-Based Access Control (ABAC + RBAC) policy engine.

While :class:`RBACManager` answers "does this subject have this
permission?", :class:`AccessControlManager` answers the richer question
"given this subject, this resource, this action, and this environment
(time-of-day, vehicle speed, geo-fence, …), is the request allowed?".

Policies are simple Python dicts that are evaluated against an
:class:`AccessRequest`. A policy is a list of conditions; all must match
for the policy to apply. If the policy applies, its ``decision`` (``allow``
or ``deny``) is returned. The first matching policy wins. If no policy
matches, the configured default decision is returned.
"""

from __future__ import annotations

import logging
import operator
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .constants import AuthStatus, Permission, Role

logger = logging.getLogger(__name__)

# Condition operators
_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "contains": lambda a, b: b in a,
    "starts_with": lambda a, b: isinstance(a, str) and a.startswith(b),
    "ends_with": lambda a, b: isinstance(a, str) and a.endswith(b),
    "between": lambda a, b: b[0] <= a <= b[1],
    "time_between": lambda a, b: b[0] <= a <= b[1],
}


@dataclass
class AccessRequest:
    """Everything the PDP needs to make a decision."""

    subject_id: str
    action: str  # the requested action / verb
    resource: str  # resource identifier (e.g. "vehicle:VIN-123")
    permission: Permission
    roles: Set[Role] = field(default_factory=set)
    environment: Dict[str, Any] = field(default_factory=dict)
    # Examples of environment attributes:
    #   - vehicle_speed_kph: float
    #   - in_geofence: bool
    #   - time_of_day: int (hour 0-23)
    #   - is_emergency: bool
    #   - ip_address: str

    def attr(self, name: str, default: Any = None) -> Any:
        """Fetch a request attribute by dotted name.

        ``subject.id`` -> subject_id; ``env.vehicle_speed_kph`` -> environment entry.
        """
        if name == "subject_id":
            return self.subject_id
        if name == "action":
            return self.action
        if name == "resource":
            return self.resource
        if name == "permission":
            return self.permission.value
        if name.startswith("env."):
            return self.environment.get(name[4:], default)
        if name.startswith("role:"):
            wanted = name[5:]
            return any(r.value == wanted for r in self.roles)
        return self.environment.get(name, default)


@dataclass
class AccessDecision:
    """Outcome of an access evaluation."""

    status: AuthStatus  # SUCCESS (allow) or DENIED
    matched_policy: Optional[str] = None
    reason: str = ""
    evaluated_at: int = field(
        default_factory=lambda: int(datetime.now(tz=timezone.utc).timestamp())
    )

    @property
    def allowed(self) -> bool:
        return self.status == AuthStatus.SUCCESS


@dataclass
class Policy:
    """A single ABAC policy entry."""

    name: str
    description: str = ""
    effect: str = "allow"  # "allow" | "deny"
    priority: int = 100  # lower = higher priority
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    # Each condition: {"attr": "env.vehicle_speed_kph", "op": "lt", "value": 5}

    def matches(self, request: AccessRequest) -> bool:
        for cond in self.conditions:
            attr = cond["attr"]
            op_name = cond.get("op", "eq")
            expected = cond.get("value")
            actual = request.attr(attr)
            op = _OPS.get(op_name)
            if op is None:
                logger.warning("Unknown operator %s in policy %s", op_name, self.name)
                return False
            try:
                if not op(actual, expected):
                    return False
            except TypeError:
                return False
        return True


class AccessControlManager:
    """Policy-based access decision engine (PDP)."""

    def __init__(
        self,
        default_decision: AuthStatus = AuthStatus.DENIED,
        rbac=None,  # RBACManager - imported lazily to avoid cycles
        permission_manager=None,
    ) -> None:
        self.default_decision = default_decision
        self.rbac = rbac
        self.permission_manager = permission_manager
        self._lock = threading.RLock()
        self._policies: List[Policy] = []
        self._load_defaults()

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------
    def _load_defaults(self) -> None:
        """Install a small set of platform-default policies."""
        self._policies.extend([
            Policy(
                name="deny-emergency-override-outside-emergency",
                description="Emergency-override permission requires env.is_emergency=true",
                effect="deny",
                priority=10,
                conditions=[
                    {"attr": "permission", "op": "eq",
                     "value": Permission.VEHICLE_EMERGENCY_OVERRIDE.value},
                    {"attr": "env.is_emergency", "op": "eq", "value": False},
                ],
            ),
            Policy(
                name="deny-ota-install-while-driving",
                description="OTA install forbidden when vehicle is in motion",
                effect="deny",
                priority=20,
                conditions=[
                    {"attr": "permission", "op": "eq",
                     "value": Permission.OTA_INSTALL.value},
                    {"attr": "env.vehicle_speed_kph", "op": "gt", "value": 0},
                ],
            ),
            Policy(
                name="deny-flash-ecu-while-driving",
                description="ECU flash forbidden when vehicle is in motion",
                effect="deny",
                priority=20,
                conditions=[
                    {"attr": "permission", "op": "eq",
                     "value": Permission.DIAG_FLASH_ECU.value},
                    {"attr": "env.vehicle_speed_kph", "op": "gt", "value": 0},
                ],
            ),
            Policy(
                name="allow-emergency-stop-any-state",
                description="Emergency stop is always allowed if role grants the perm",
                effect="allow",
                priority=30,
                conditions=[
                    {"attr": "permission", "op": "eq",
                     "value": Permission.VEHICLE_EMERGENCY_STOP.value},
                ],
            ),
        ])

    def add_policy(self, policy: Policy) -> None:
        with self._lock:
            self._policies.append(policy)
            self._policies.sort(key=lambda p: (p.priority, p.name))
        logger.info("Added policy %s (effect=%s, priority=%d)",
                    policy.name, policy.effect, policy.priority)

    def load_policy(self, policy_dict: Dict[str, Any]) -> Policy:
        """Construct and install a policy from a dict (e.g. from YAML)."""
        policy = Policy(
            name=policy_dict["name"],
            description=policy_dict.get("description", ""),
            effect=policy_dict.get("effect", "allow"),
            priority=int(policy_dict.get("priority", 100)),
            conditions=policy_dict.get("conditions", []),
        )
        self.add_policy(policy)
        return policy

    def remove_policy(self, name: str) -> bool:
        with self._lock:
            before = len(self._policies)
            self._policies = [p for p in self._policies if p.name != name]
            return len(self._policies) < before

    def list_policies(self) -> List[Policy]:
        with self._lock:
            return list(self._policies)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def evaluate_access(self, request: AccessRequest) -> AccessDecision:
        """Run the policy engine over ``request``.

        The first matching policy (sorted by priority) wins. If no policy
        matches, the manager still requires RBAC permission to allow; if
        RBAC denies, the default decision applies.
        """
        with self._lock:
            policies = list(self._policies)

        for policy in policies:
            if policy.matches(request):
                if policy.effect == "deny":
                    logger.info(
                        "Policy %s DENIED request subject=%s perm=%s",
                        policy.name, request.subject_id, request.permission.value,
                    )
                    return AccessDecision(
                        status=AuthStatus.DENIED,
                        matched_policy=policy.name,
                        reason=f"Denied by policy {policy.name}",
                    )
                # allow policy -> still require RBAC perm
                if self._rbac_allows(request):
                    return AccessDecision(
                        status=AuthStatus.SUCCESS,
                        matched_policy=policy.name,
                        reason=f"Allowed by policy {policy.name}",
                    )
                return AccessDecision(
                    status=AuthStatus.DENIED,
                    matched_policy=policy.name,
                    reason=f"Policy {policy.name} allows but RBAC denies",
                )

        # No policy matched: fall through to RBAC + default
        if self._rbac_allows(request):
            return AccessDecision(
                status=AuthStatus.SUCCESS,
                matched_policy=None,
                reason="Allowed by RBAC (no ABAC policy matched)",
            )
        return AccessDecision(
            status=self.default_decision,
            matched_policy=None,
            reason="No matching policy and RBAC denied",
        )

    def get_decision(self, request: AccessRequest) -> AccessDecision:
        """Alias for :meth:`evaluate_access`."""
        return self.evaluate_access(request)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _rbac_allows(self, request: AccessRequest) -> bool:
        if self.rbac is None:
            # Without RBAC wired up, only allow if permission_manager grants directly.
            if self.permission_manager is None:
                return False
            return self.permission_manager.check_permission(
                request.subject_id, request.permission, request.roles
            )
        return self.rbac.check_access(request.subject_id, request.permission)


__all__ = [
    "AccessRequest",
    "AccessDecision",
    "Policy",
    "AccessControlManager",
]
