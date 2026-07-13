"""Emergency lockdown.

The :class:`EmergencyLockdown` triggers a full vehicle security lockdown:

  * Disable all external network interfaces (cellular, Wi-Fi, V2X).
  * Disable OTA update acceptance.
  * Force the vehicle into a safe stop (request minimum-risk maneuver).
  * Isolate critical ECUs (only allow safety-critical CAN traffic).
  * Block all diagnostic sessions except authenticated admin sessions.

A lockdown can only be released via :meth:`release_lockdown` with a valid
admin authentication token (PBKDF2 hash of the admin PIN). While locked
down, :meth:`is_locked_down` returns ``True`` and :meth:`get_lockdown_reason``
returns the reason string supplied at trigger time.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .constants import Severity
from .utils import compute_sha256, format_timestamp, safe_compare
from .security_event_logger import SecurityEventLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


class LockdownScope(str, Enum):
    NETWORK = "network"
    OTA = "ota"
    CRITICAL_ECU = "critical_ecu"
    DIAGNOSTICS = "diagnostics"
    SAFE_STOP = "safe_stop"
    FULL = "full"


@dataclass
class LockdownState:
    locked_down: bool = False
    reason: str = ""
    triggered_at: float = 0.0
    triggered_by: str = ""
    scopes: List[LockdownScope] = field(default_factory=list)
    release_attempts: int = 0
    last_release_attempt: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scopes"] = [s.value for s in self.scopes]
        return d


# --------------------------------------------------------------------------- #
# Action callbacks (mockable)
# --------------------------------------------------------------------------- #


ActionCallback = Callable[[str, bool], bool]  # (scope, enable) -> success


def _default_action_callback(scope: str, enable: bool) -> bool:
    """Default callback logs the action and returns success."""
    action = "ENABLE" if enable else "DISABLE"
    logger.info("[default] %s %s", action, scope)
    return True


# --------------------------------------------------------------------------- #
# Admin PIN verification
# --------------------------------------------------------------------------- #


def hash_admin_pin(pin: str, salt: Optional[bytes] = None, iterations: int = 100_000) -> str:
    """Return ``salt_hex:pbkdf2_hex`` for an admin PIN."""
    if salt is None:
        salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
    return f"{salt.hex()}:{derived.hex()}"


def verify_admin_pin(pin: str, stored_hash: str) -> bool:
    try:
        salt_hex, derived_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(derived_hex)
        actual = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 100_000)
        return safe_compare(expected, actual)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Lockdown manager
# --------------------------------------------------------------------------- #


class EmergencyLockdown:
    """Coordinates full-vehicle security lockdown and release."""

    DEFAULT_SCOPES = [
        LockdownScope.NETWORK,
        LockdownScope.OTA,
        LockdownScope.CRITICAL_ECU,
        LockdownScope.DIAGNOSTICS,
        LockdownScope.SAFE_STOP,
    ]

    def __init__(
        self,
        admin_pin_hash: str,
        action_callback: Optional[ActionCallback] = None,
        logger_: Optional[SecurityEventLogger] = None,
    ) -> None:
        self.admin_pin_hash = admin_pin_hash
        self.action_callback = action_callback or _default_action_callback
        self.event_logger = logger_
        self._lock = threading.RLock()
        self._state = LockdownState()
        self._release_callbacks: List[Callable[[], None]] = []

    # ------------------------------------------------------------------ #
    # Trigger
    # ------------------------------------------------------------------ #

    def trigger_lockdown(
        self,
        reason: str,
        triggered_by: str = "system",
        scopes: Optional[List[LockdownScope]] = None,
    ) -> bool:
        """Trigger a full or scoped lockdown."""
        with self._lock:
            if self._state.locked_down:
                logger.warning("lockdown already active; ignoring re-trigger")
                return False
            scopes = scopes or list(self.DEFAULT_SCOPES)
            self._state = LockdownState(
                locked_down=True,
                reason=reason,
                triggered_at=time.time(),
                triggered_by=triggered_by,
                scopes=scopes,
            )

        # Apply each scope action (disable = enable=False)
        success = True
        for scope in scopes:
            try:
                ok = self.action_callback(scope.value, False)
                if not ok:
                    success = False
                    logger.error("lockdown action failed for scope %s", scope.value)
            except Exception as exc:  # noqa: BLE001
                logger.exception("lockdown action raised for %s: %s", scope.value, exc)
                success = False

        if self.event_logger:
            self.event_logger.log_event(
                event_type="lockdown.triggered",
                severity=Severity.CRITICAL.value,
                source="emergency_lockdown",
                message=f"lockdown triggered by {triggered_by}: {reason}",
                details=self._state.to_dict(),
            )
        logger.critical("LOCKDOWN triggered: %s (by=%s, scopes=%s)",
                        reason, triggered_by, [s.value for s in scopes])
        return success

    # ------------------------------------------------------------------ #
    # Release
    # ------------------------------------------------------------------ #

    def release_lockdown(self, admin_pin: str) -> bool:
        """Release the lockdown. Requires the admin PIN."""
        with self._lock:
            self._state.release_attempts += 1
            self._state.last_release_attempt = time.time()
            if not self._state.locked_down:
                logger.info("release_lockdown: not currently locked down")
                return True
            if not verify_admin_pin(admin_pin, self.admin_pin_hash):
                logger.warning("release_lockdown: invalid admin PIN")
                if self.event_logger:
                    self.event_logger.log_event(
                        event_type="lockdown.release_denied",
                        severity=Severity.HIGH.value,
                        source="emergency_lockdown",
                        message="failed lockdown release attempt (bad PIN)",
                    )
                return False

            scopes = list(self._state.scopes)
            self._state.locked_down = False
            self._state.reason = ""
            self._state.scopes = []
            callbacks = list(self._release_callbacks)

        # Re-enable each scope in reverse order
        for scope in reversed(scopes):
            try:
                self.action_callback(scope.value, True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("release action raised for %s: %s", scope.value, exc)

        if self.event_logger:
            self.event_logger.log_event(
                event_type="lockdown.released",
                severity=Severity.HIGH.value,
                source="emergency_lockdown",
                message="lockdown released (admin authenticated)",
            )
        logger.info("lockdown released")
        for cb in callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001
                logger.exception("release callback raised")
        return True

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def is_locked_down(self) -> bool:
        with self._lock:
            return self._state.locked_down

    def get_lockdown_reason(self) -> str:
        with self._lock:
            return self._state.reason

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def get_release_attempts(self) -> int:
        with self._lock:
            return self._state.release_attempts

    def add_release_callback(self, cb: Callable[[], None]) -> None:
        self._release_callbacks.append(cb)

    def set_admin_pin_hash(self, new_hash: str) -> None:
        """Rotate the admin PIN hash (used after a successful PIN change)."""
        with self._lock:
            self.admin_pin_hash = new_hash
