"""
user_authentication.py
======================

Driver / operator authentication with bcrypt-hashed passwords, PIN codes
and a thin biometric wrapper.

The class is responsible for the *user-credential* side of
authentication. Token issuance, session creation, and MFA orchestration
are handled by the higher-level :class:`AuthenticationManager`.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import LockoutPolicy, PasswordPolicy
from .constants import AuthEvent, AuthMethod, AuthStatus
from .utils import generate_salt, hash_password, safe_compare, verify_password

logger = logging.getLogger(__name__)

_PIN_RE = re.compile(r"^\d{4,8}$")


@dataclass
class UserRecord:
    """Stored user-credential record."""

    user_id: str
    username: str
    password_hash: str
    pin_hash: Optional[str] = None
    email: Optional[str] = None
    active: bool = True
    failed_attempts: int = 0
    locked_until: int = 0  # unix ts; 0 = not locked
    last_login_at: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthResult:
    """Outcome of a user-authentication attempt."""

    status: AuthStatus
    user_id: Optional[str] = None
    method: Optional[AuthMethod] = None
    reason: str = ""
    locked_until: int = 0


class UserAuthenticator:
    """Username/password + PIN + biometric authenticator for drivers."""

    def __init__(
        self,
        password_policy: PasswordPolicy,
        lockout_policy: LockoutPolicy,
    ) -> None:
        self.password_policy = password_policy
        self.lockout_policy = lockout_policy
        self._lock = threading.RLock()
        self._users_by_id: Dict[str, UserRecord] = {}
        self._users_by_name: Dict[str, str] = {}  # username -> user_id

    # ------------------------------------------------------------------
    # Registration & lifecycle
    # ------------------------------------------------------------------
    def register_user(
        self,
        username: str,
        password: str,
        *,
        pin: Optional[str] = None,
        email: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserRecord:
        """Register a new user. Raises ``ValueError`` on policy violations."""
        username = (username or "").strip().lower()
        if not username:
            raise ValueError("username must be non-empty")
        if len(username) < 3:
            raise ValueError("username must be at least 3 characters")
        self._validate_password_complexity(password)
        if pin is not None:
            self._validate_pin(pin)

        with self._lock:
            if username in self._users_by_name:
                raise ValueError(f"username {username!r} already registered")
            uid = user_id or f"u_{secrets.token_hex(8)}"
            if uid in self._users_by_id:
                raise ValueError(f"user_id {uid!r} already exists")
            record = UserRecord(
                user_id=uid,
                username=username,
                password_hash=hash_password(
                    password, rounds=self.password_policy.bcrypt_rounds
                ),
                pin_hash=hash_password(pin, rounds=self.password_policy.bcrypt_rounds) if pin else None,
                email=email,
                metadata=dict(metadata or {}),
            )
            self._users_by_id[uid] = record
            self._users_by_name[username] = uid
        logger.info("Registered user %s (id=%s)", username, uid)
        return record

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """Change a user's password after verifying the current one."""
        self._validate_password_complexity(new_password)
        with self._lock:
            record = self._users_by_id.get(user_id)
            if record is None:
                return False
            if not verify_password(current_password, record.password_hash):
                logger.info("change_password: bad current password for %s", user_id)
                return False
            record.password_hash = hash_password(
                new_password, rounds=self.password_policy.bcrypt_rounds
            )
            record.failed_attempts = 0
            record.locked_until = 0
        logger.info("Password changed for user %s", user_id)
        return True

    def reset_password(
        self,
        user_id: str,
        new_password: str,
        *,
        reset_token: Optional[str] = None,
    ) -> bool:
        """Reset a user's password (admin / self-service with reset token).

        The ``reset_token`` is opaque to this class — the caller (e.g. an
        email-reset flow) is responsible for validating it. If
        ``reset_token`` is ``None`` we require admin context (not enforced
        here) — i.e. this method trusts the caller to have authorised the
        reset.
        """
        self._validate_password_complexity(new_password)
        with self._lock:
            record = self._users_by_id.get(user_id)
            if record is None:
                return False
            record.password_hash = hash_password(
                new_password, rounds=self.password_policy.bcrypt_rounds
            )
            record.failed_attempts = 0
            record.locked_until = 0
        logger.info("Password reset for user %s (token=%s)",
                    user_id, "yes" if reset_token else "admin")
        return True

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(
        self,
        username: str,
        password: str,
        *,
        ip_address: Optional[str] = None,
    ) -> AuthResult:
        """Authenticate by username + password."""
        username = (username or "").strip().lower()
        with self._lock:
            uid = self._users_by_name.get(username)
            if uid is None:
                return AuthResult(
                    status=AuthStatus.FAILURE,
                    method=AuthMethod.PASSWORD,
                    reason="unknown user",
                )
            record = self._users_by_id[uid]
            now = int(time.time())
            if record.locked_until > now:
                return AuthResult(
                    status=AuthStatus.LOCKED_OUT,
                    user_id=uid,
                    method=AuthMethod.PASSWORD,
                    reason="account locked",
                    locked_until=record.locked_until,
                )
            if not record.active:
                return AuthResult(
                    status=AuthStatus.FAILURE,
                    user_id=uid,
                    method=AuthMethod.PASSWORD,
                    reason="account disabled",
                )
            if not verify_password(password, record.password_hash):
                return self._record_failure(record, now, ip_address)
            # success
            record.failed_attempts = 0
            record.locked_until = 0
            record.last_login_at = now
        logger.info("User %s authenticated (id=%s, ip=%s)", username, uid, ip_address)
        return AuthResult(
            status=AuthStatus.SUCCESS,
            user_id=uid,
            method=AuthMethod.PASSWORD,
            reason="ok",
        )

    def verify_pin(self, user_id: str, pin: str) -> bool:
        """Verify a short PIN (used by in-cabin kiosk after password auth)."""
        with self._lock:
            record = self._users_by_id.get(user_id)
            if record is None or record.pin_hash is None:
                return False
            return verify_password(pin, record.pin_hash)

    def verify_credentials(self, user_id: str, password: str) -> bool:
        """1:1 password verification by user_id (used by MFA factor verifier)."""
        with self._lock:
            record = self._users_by_id.get(user_id)
            if record is None:
                return False
            return verify_password(password, record.password_hash)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_user(self, user_id: str) -> Optional[UserRecord]:
        with self._lock:
            return self._users_by_id.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        username = (username or "").strip().lower()
        with self._lock:
            uid = self._users_by_name.get(username)
            return self._users_by_id.get(uid) if uid else None

    def deactivate_user(self, user_id: str) -> bool:
        with self._lock:
            record = self._users_by_id.get(user_id)
            if record is None:
                return False
            record.active = False
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _record_failure(
        self,
        record: UserRecord,
        now: int,
        ip_address: Optional[str],
    ) -> AuthResult:
        record.failed_attempts += 1
        if record.failed_attempts >= self.lockout_policy.max_attempts:
            record.locked_until = now + int(self.lockout_policy.lockout_duration.total_seconds())
            logger.warning(
                "User %s locked out until %d (ip=%s)",
                record.user_id, record.locked_until, ip_address,
            )
            return AuthResult(
                status=AuthStatus.LOCKED_OUT,
                user_id=record.user_id,
                method=AuthMethod.PASSWORD,
                reason="max attempts exceeded",
                locked_until=record.locked_until,
            )
        return AuthResult(
            status=AuthStatus.FAILURE,
            user_id=record.user_id,
            method=AuthMethod.PASSWORD,
            reason="bad password",
        )

    def _validate_password_complexity(self, password: str) -> None:
        p = self.password_policy
        if len(password) < p.min_length:
            raise ValueError(f"password must be at least {p.min_length} characters")
        if p.require_digit and not re.search(r"\d", password):
            raise ValueError("password must contain at least one digit")
        if p.require_upper and not re.search(r"[A-Z]", password):
            raise ValueError("password must contain at least one upper-case letter")
        if p.require_symbol and not re.search(r"[^A-Za-z0-9]", password):
            raise ValueError("password must contain at least one symbol")

    @staticmethod
    def _validate_pin(pin: str) -> None:
        if not _PIN_RE.match(pin):
            raise ValueError("PIN must be 4-8 digits")

    # expose salt helper for tests / external integrations
    @staticmethod
    def generate_salt(rounds: int = 12) -> bytes:
        return generate_salt(rounds)


__all__ = ["UserRecord", "AuthResult", "UserAuthenticator"]
