"""
permission_manager.py
=====================

Granular permission registry for the AV platform.

Permissions are namespaced strings (``<domain>:<action>``) — see
:class:`constants.Permission` — and can be granted either directly to a
subject or to a role. :class:`PermissionManager` keeps both mappings and
exposes a single ``check_permission`` that consults both layers.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Iterable, Optional, Set

from .constants import Permission, Role

logger = logging.getLogger(__name__)


class PermissionManager:
    """In-memory permission registry.

    Two mappings are maintained:

    * ``role_permissions``: ``Role -> Set[Permission]``
    * ``subject_permissions``: ``subject_id -> Set[Permission]``
      (direct grants; used for ad-hoc, fine-grained overrides)
    """

    # Default role -> permission mapping used when a fresh manager is
    # instantiated. Callers can override via :meth:`set_role_permissions`.
    DEFAULT_ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.DRIVER: {
            Permission.VEHICLE_READ_TELEMETRY,
            Permission.VEHICLE_READ_LOCATION,
            Permission.VEHICLE_CONTROL_STEERING,
            Permission.VEHICLE_CONTROL_BRAKING,
            Permission.VEHICLE_CONTROL_THROTTLE,
            Permission.VEHICLE_EMERGENCY_STOP,
        },
        Role.PASSENGER: {
            Permission.VEHICLE_READ_LOCATION,
        },
        Role.FLEET_MANAGER: {
            Permission.VEHICLE_READ_TELEMETRY,
            Permission.VEHICLE_READ_LOCATION,
            Permission.FLEET_VIEW_ALL,
            Permission.FLEET_DISPATCH,
            Permission.FLEET_DECOMMISSION,
            Permission.OTA_READ_MANIFEST,
            Permission.DIAG_READ_DTC,
        },
        Role.SERVICE_TECHNICIAN: {
            Permission.VEHICLE_READ_TELEMETRY,
            Permission.DIAG_READ_DTC,
            Permission.DIAG_CLEAR_DTC,
            Permission.DIAG_FLASH_ECU,
            Permission.OTA_READ_MANIFEST,
        },
        Role.SECURITY_ADMIN: {
            Permission.SECURITY_MANAGE_KEYS,
            Permission.SECURITY_ROTATE_CERTS,
            Permission.SECURITY_REVOKE_DEVICE,
            Permission.SECURITY_VIEW_AUDIT_LOG,
            Permission.SECURITY_MANAGE_RBAC,
        },
        Role.EMERGENCY_OVERRIDE: {
            Permission.VEHICLE_EMERGENCY_OVERRIDE,
            Permission.VEHICLE_EMERGENCY_STOP,
            Permission.VEHICLE_READ_TELEMETRY,
            Permission.VEHICLE_READ_LOCATION,
        },
        Role.OTA_OPERATOR: {
            Permission.OTA_READ_MANIFEST,
            Permission.OTA_INSTALL,
            Permission.OTA_ROLLBACK,
        },
        Role.AUDITOR: {
            Permission.SECURITY_VIEW_AUDIT_LOG,
            Permission.VEHICLE_READ_TELEMETRY,
        },
    }

    def __init__(
        self,
        role_permissions: Optional[Dict[Role, Set[Permission]]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._role_permissions: Dict[Role, Set[Permission]] = {
            role: set(perms)
            for role, perms in (role_permissions or self.DEFAULT_ROLE_PERMISSIONS).items()
        }
        self._subject_permissions: Dict[str, Set[Permission]] = {}
        # negative grants allow temporary revocation of a permission for
        # a specific subject even if their role grants it.
        self._subject_denials: Dict[str, Set[Permission]] = {}

    # ------------------------------------------------------------------
    # Role-level permissions
    # ------------------------------------------------------------------
    def set_role_permissions(
        self, role: Role, permissions: Iterable[Permission]
    ) -> None:
        with self._lock:
            self._role_permissions[role] = set(permissions)
        logger.info("Replaced permissions for role %s (%d perms)",
                    role.value, len(self._role_permissions[role]))

    def grant_permission_to_role(
        self, role: Role, permission: Permission
    ) -> bool:
        with self._lock:
            perms = self._role_permissions.setdefault(role, set())
            if permission in perms:
                return False
            perms.add(permission)
        logger.info("Granted %s to role %s", permission.value, role.value)
        return True

    def revoke_permission_from_role(
        self, role: Role, permission: Permission
    ) -> bool:
        with self._lock:
            perms = self._role_permissions.get(role)
            if perms is None or permission not in perms:
                return False
            perms.discard(permission)
        logger.info("Revoked %s from role %s", permission.value, role.value)
        return True

    def list_permissions_for_role(self, role: Role) -> Set[Permission]:
        with self._lock:
            return set(self._role_permissions.get(role, set()))

    # ------------------------------------------------------------------
    # Subject-level permissions (direct grants)
    # ------------------------------------------------------------------
    def grant_permission(
        self, subject_id: str, permission: Permission
    ) -> bool:
        with self._lock:
            grants = self._subject_permissions.setdefault(subject_id, set())
            denials = self._subject_denials.get(subject_id, set())
            denials.discard(permission)
            if permission in grants:
                return False
            grants.add(permission)
        logger.info("Granted %s to subject %s", permission.value, subject_id)
        return True

    def revoke_permission(
        self, subject_id: str, permission: Permission
    ) -> bool:
        with self._lock:
            grants = self._subject_permissions.get(subject_id, set())
            denials = self._subject_denials.setdefault(subject_id, set())
            removed = permission in grants
            grants.discard(permission)
            denials.add(permission)
        if removed:
            logger.info("Revoked %s from subject %s", permission.value, subject_id)
        return removed

    def list_permissions_for_subject(self, subject_id: str) -> Set[Permission]:
        with self._lock:
            return set(self._subject_permissions.get(subject_id, set()))

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def check_permission(
        self,
        subject_id: str,
        permission: Permission,
        roles: Optional[Iterable[Role]] = None,
    ) -> bool:
        """Return ``True`` iff ``subject_id`` is allowed ``permission``.

        Decision order:
          1. Explicit subject denial → ``False``.
          2. Explicit subject grant → ``True``.
          3. Any of the subject's roles grants the permission → ``True``.
          4. Otherwise ``False``.
        """
        with self._lock:
            denials = self._subject_denials.get(subject_id, set())
            if permission in denials:
                return False
            grants = self._subject_permissions.get(subject_id, set())
            if permission in grants:
                return True
            if roles:
                for role in roles:
                    if permission in self._role_permissions.get(role, set()):
                        return True
        return False

    def effective_permissions(
        self,
        subject_id: str,
        roles: Optional[Iterable[Role]] = None,
    ) -> Set[Permission]:
        """Compute the union of all permissions a subject currently has."""
        with self._lock:
            result: Set[Permission] = set(self._subject_permissions.get(subject_id, set()))
            if roles:
                for role in roles:
                    result |= self._role_permissions.get(role, set())
            result -= self._subject_denials.get(subject_id, set())
            return result


__all__ = ["PermissionManager"]
