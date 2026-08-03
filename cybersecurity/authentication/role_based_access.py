"""
role_based_access.py
====================

Role-Based Access Control (RBAC) layer for the AV platform.

Wraps :class:`PermissionManager` with a user-facing role assignment
registry so that the orchestrator can answer questions like "what roles
does driver X have?" without poking the permission store directly.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Set

from .constants import Role
from .permission_manager import PermissionManager

logger = logging.getLogger(__name__)


@dataclass
class RoleAssignment:
    """A role bound to a subject, optionally scoped to a resource."""

    subject_id: str
    role: Role
    assigned_at: int
    assigned_by: str
    expires_at: Optional[int] = None  # None = does not expire
    scope: Optional[str] = None  # e.g. "vehicle:VIN-123" / "fleet:west"
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_active(self, now: Optional[int] = None) -> bool:
        ts = now or int(datetime.now(tz=timezone.utc).timestamp())
        return self.expires_at is None or ts < self.expires_at


class RBACManager:
    """Role assignment registry layered on top of :class:`PermissionManager`."""

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager
        self._lock = threading.RLock()
        # subject_id -> list of RoleAssignment
        self._assignments: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Assignment lifecycle
    # ------------------------------------------------------------------
    def assign_role(
        self,
        subject_id: str,
        role: Role,
        assigned_by: str = "system",
        expires_at: Optional[int] = None,
        scope: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RoleAssignment:
        assignment = RoleAssignment(
            subject_id=subject_id,
            role=role,
            assigned_at=int(datetime.now(tz=timezone.utc).timestamp()),
            assigned_by=assigned_by,
            expires_at=expires_at,
            scope=scope,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            lst = self._assignments.setdefault(subject_id, [])
            # de-duplicate identical (role, scope) assignments
            for existing in lst:
                if existing.role == role and existing.scope == scope and existing.is_active():
                    logger.info(
                        "Role %s already assigned to %s; ignoring re-assignment",
                        role.value, subject_id,
                    )
                    return existing
            lst.append(assignment)
        logger.info(
            "Assigned role %s to %s (by=%s, scope=%s)",
            role.value, subject_id, assigned_by, scope,
        )
        return assignment

    def revoke_role(
        self,
        subject_id: str,
        role: Role,
        scope: Optional[str] = None,
    ) -> bool:
        with self._lock:
            lst = self._assignments.get(subject_id)
            if not lst:
                return False
            before = len(lst)
            lst = [
                a for a in lst
                if not (a.role == role and a.scope == scope)
            ]
            self._assignments[subject_id] = lst
            if not lst:
                self._assignments.pop(subject_id, None)
            removed = before - len(lst) > 0
        if removed:
            logger.info("Revoked role %s from %s (scope=%s)",
                        role.value, subject_id, scope)
        return removed

    def revoke_all_roles(self, subject_id: str) -> int:
        with self._lock:
            lst = self._assignments.pop(subject_id, [])
        if lst:
            logger.info("Revoked all %d roles from %s", len(lst), subject_id)
        return len(lst)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_roles(
        self,
        subject_id: str,
        scope: Optional[str] = None,
        include_expired: bool = False,
    ) -> Set[Role]:
        now = int(datetime.now(tz=timezone.utc).timestamp())
        with self._lock:
            assignments = list(self._assignments.get(subject_id, []))
        roles: Set[Role] = set()
        for a in assignments:
            if not include_expired and not a.is_active(now):
                continue
            if scope is not None and a.scope is not None and a.scope != scope:
                continue
            roles.add(a.role)
        return roles

    def get_assignments(self, subject_id: str) -> list:
        with self._lock:
            return list(self._assignments.get(subject_id, []))

    def get_permissions(
        self,
        subject_id: str,
        scope: Optional[str] = None,
    ) -> Set[str]:
        """Return the string permission set for ``subject_id``."""
        roles = self.get_roles(subject_id, scope=scope)
        from .constants import Permission  # local to avoid cycle
        perms = self.permission_manager.effective_permissions(subject_id, roles)
        return {p.value for p in perms}

    def check_access(
        self,
        subject_id: str,
        permission,  # Permission | str
        scope: Optional[str] = None,
    ) -> bool:
        """Convenience: check access through the underlying permission manager."""
        from .constants import Permission
        if isinstance(permission, str):
            try:
                permission = Permission(permission)
            except ValueError:
                logger.warning("Unknown permission string: %s", permission)
                return False
        roles = self.get_roles(subject_id, scope=scope)
        return self.permission_manager.check_permission(subject_id, permission, roles)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def purge_expired(self) -> int:
        now = int(datetime.now(tz=timezone.utc).timestamp())
        purged = 0
        with self._lock:
            for subject_id in list(self._assignments.keys()):
                lst = self._assignments[subject_id]
                kept = [a for a in lst if a.is_active(now)]
                purged += len(lst) - len(kept)
                if kept:
                    self._assignments[subject_id] = kept
                else:
                    self._assignments.pop(subject_id, None)
        if purged:
            logger.info("Purged %d expired role assignments", purged)
        return purged

    def subjects_with_role(self, role: Role) -> Set[str]:
        out: Set[str] = set()
        with self._lock:
            for subject_id, lst in self._assignments.items():
                if any(a.role == role and a.is_active() for a in lst):
                    out.add(subject_id)
        return out


__all__ = ["RBACManager", "RoleAssignment"]
