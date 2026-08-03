"""
config.py
=========

Configuration dataclasses and YAML loader for the authentication
sub-system.

The configuration is intentionally explicit and small: it captures the
runtime parameters that operators need to tune (TTLs, bcrypt rounds,
issuer, audit-log path, allowed roles) without exploding into a
Spring-style XML horror show. The dataclasses are frozen so that code
downstream can rely on the fact that configuration will not be mutated
underneath it at runtime.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import (
    DEFAULT_ACCESS_TOKEN_TTL,
    DEFAULT_AUDIENCE,
    DEFAULT_BCRYPT_ROUNDS,
    DEFAULT_ISSUER,
    DEFAULT_REFRESH_TOKEN_TTL,
    DEFAULT_SESSION_TTL,
    LOCKOUT_DURATION,
    MAX_LOGIN_ATTEMPTS,
    Role,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JWTConfig:
    """JSON Web Token configuration."""

    secret_key: str = "change-me-in-production"
    public_key: Optional[str] = None
    private_key: Optional[str] = None
    algorithm: str = "HS256"  # HS256 | RS256 | ES256
    issuer: str = DEFAULT_ISSUER
    audience: str = DEFAULT_AUDIENCE
    access_token_ttl: timedelta = DEFAULT_ACCESS_TOKEN_TTL
    refresh_token_ttl: timedelta = DEFAULT_REFRESH_TOKEN_TTL
    leeway_seconds: int = 5
    blacklist_enabled: bool = True


@dataclass(frozen=True)
class SessionConfig:
    """Session-management configuration."""

    session_ttl: timedelta = DEFAULT_SESSION_TTL
    idle_timeout: timedelta = timedelta(minutes=30)
    max_sessions_per_user: int = 5
    cleanup_interval: timedelta = timedelta(minutes=10)


@dataclass(frozen=True)
class PasswordPolicy:
    """Password hashing / complexity policy."""

    bcrypt_rounds: int = DEFAULT_BCRYPT_ROUNDS
    argon2_time_cost: int = 3
    argon2_memory_cost: int = 65536  # 64 MiB
    argon2_parallelism: int = 2
    min_length: int = 12
    require_digit: bool = True
    require_upper: bool = True
    require_symbol: bool = True


@dataclass(frozen=True)
class LockoutPolicy:
    """Account lockout policy."""

    max_attempts: int = MAX_LOGIN_ATTEMPTS
    lockout_duration: timedelta = LOCKOUT_DURATION
    reset_after: timedelta = timedelta(minutes=30)


@dataclass(frozen=True)
class MFAConfig:
    """Multi-factor authentication configuration."""

    otp_issuer: str = "AVCS"
    totp_step: int = 30
    totp_digits: int = 6
    challenge_ttl: timedelta = timedelta(minutes=5)
    required_for_roles: tuple = (Role.SECURITY_ADMIN, Role.OTA_OPERATOR)


@dataclass(frozen=True)
class AuditConfig:
    """Audit-log configuration."""

    log_path: Path = Path("/var/log/avcs/auth_audit.log")
    hash_algorithm: str = "sha256"
    flush_immediately: bool = True
    max_entry_size: int = 16_384


@dataclass(frozen=True)
class AuthenticationConfig:
    """Top-level authentication configuration."""

    jwt: JWTConfig = field(default_factory=JWTConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    password: PasswordPolicy = field(default_factory=PasswordPolicy)
    lockout: LockoutPolicy = field(default_factory=LockoutPolicy)
    mfa: MFAConfig = field(default_factory=MFAConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    allowed_roles: tuple = (
        Role.DRIVER,
        Role.FLEET_MANAGER,
        Role.SECURITY_ADMIN,
        Role.SERVICE_TECHNICIAN,
        Role.EMERGENCY_OVERRIDE,
        Role.OTA_OPERATOR,
        Role.AUDITOR,
    )
    environment: str = "production"
    debug: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.environment == "production" and self.jwt.secret_key == "change-me-in-production":
            raise ValueError(
                "JWT secret_key must be overridden in production environment"
            )
        if self.jwt.algorithm not in {"HS256", "RS256", "ES256"}:
            raise ValueError(f"Unsupported JWT algorithm: {self.jwt.algorithm}")
        if self.password.bcrypt_rounds < 10:
            logger.warning("bcrypt_rounds < 10 is not recommended (weak KDF)")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _coerce_timedelta(value: Any) -> timedelta:
    """Coerce ``int`` (seconds) / ``str`` (``"30m"``) into a timedelta."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    if isinstance(value, str):
        s = value.strip().lower()
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        if s and s[-1] in units:
            return timedelta(seconds=float(s[:-1]) * units[s[-1]])
        return timedelta(seconds=float(s))
    raise TypeError(f"Cannot coerce {value!r} to timedelta")


def _build_config(raw: Dict[str, Any]) -> AuthenticationConfig:
    """Map a raw YAML dict into a fully-typed :class:`AuthenticationConfig`."""
    jwt_raw = raw.get("jwt", {}) or {}
    sess_raw = raw.get("session", {}) or {}
    pw_raw = raw.get("password", {}) or {}
    lock_raw = raw.get("lockout", {}) or {}
    mfa_raw = raw.get("mfa", {}) or {}
    audit_raw = raw.get("audit", {}) or {}

    jwt_cfg = JWTConfig(
        secret_key=jwt_raw.get("secret_key", JWTConfig.secret_key),
        public_key=jwt_raw.get("public_key"),
        private_key=jwt_raw.get("private_key"),
        algorithm=jwt_raw.get("algorithm", JWTConfig.algorithm),
        issuer=jwt_raw.get("issuer", JWTConfig.issuer),
        audience=jwt_raw.get("audience", JWTConfig.audience),
        access_token_ttl=_coerce_timedelta(
            jwt_raw.get("access_token_ttl", JWTConfig.access_token_ttl)
        ),
        refresh_token_ttl=_coerce_timedelta(
            jwt_raw.get("refresh_token_ttl", JWTConfig.refresh_token_ttl)
        ),
        leeway_seconds=int(jwt_raw.get("leeway_seconds", JWTConfig.leeway_seconds)),
        blacklist_enabled=bool(
            jwt_raw.get("blacklist_enabled", JWTConfig.blacklist_enabled)
        ),
    )

    sess_cfg = SessionConfig(
        session_ttl=_coerce_timedelta(
            sess_raw.get("session_ttl", SessionConfig.session_ttl)
        ),
        idle_timeout=_coerce_timedelta(
            sess_raw.get("idle_timeout", SessionConfig.idle_timeout)
        ),
        max_sessions_per_user=int(
            sess_raw.get("max_sessions_per_user", SessionConfig.max_sessions_per_user)
        ),
        cleanup_interval=_coerce_timedelta(
            sess_raw.get("cleanup_interval", SessionConfig.cleanup_interval)
        ),
    )

    pw_cfg = PasswordPolicy(
        bcrypt_rounds=int(pw_raw.get("bcrypt_rounds", PasswordPolicy.bcrypt_rounds)),
        argon2_time_cost=int(
            pw_raw.get("argon2_time_cost", PasswordPolicy.argon2_time_cost)
        ),
        argon2_memory_cost=int(
            pw_raw.get("argon2_memory_cost", PasswordPolicy.argon2_memory_cost)
        ),
        argon2_parallelism=int(
            pw_raw.get("argon2_parallelism", PasswordPolicy.argon2_parallelism)
        ),
        min_length=int(pw_raw.get("min_length", PasswordPolicy.min_length)),
        require_digit=bool(pw_raw.get("require_digit", PasswordPolicy.require_digit)),
        require_upper=bool(pw_raw.get("require_upper", PasswordPolicy.require_upper)),
        require_symbol=bool(
            pw_raw.get("require_symbol", PasswordPolicy.require_symbol)
        ),
    )

    lock_cfg = LockoutPolicy(
        max_attempts=int(lock_raw.get("max_attempts", LockoutPolicy.max_attempts)),
        lockout_duration=_coerce_timedelta(
            lock_raw.get("lockout_duration", LockoutPolicy.lockout_duration)
        ),
        reset_after=_coerce_timedelta(
            lock_raw.get("reset_after", LockoutPolicy.reset_after)
        ),
    )

    mfa_cfg = MFAConfig(
        otp_issuer=mfa_raw.get("otp_issuer", MFAConfig.otp_issuer),
        totp_step=int(mfa_raw.get("totp_step", MFAConfig.totp_step)),
        totp_digits=int(mfa_raw.get("totp_digits", MFAConfig.totp_digits)),
        challenge_ttl=_coerce_timedelta(
            mfa_raw.get("challenge_ttl", MFAConfig.challenge_ttl)
        ),
        required_for_roles=tuple(mfa_raw.get("required_for_roles", MFAConfig.required_for_roles)),
    )

    audit_cfg = AuditConfig(
        log_path=Path(audit_raw.get("log_path", str(AuditConfig.log_path))),
        hash_algorithm=audit_raw.get("hash_algorithm", AuditConfig.hash_algorithm),
        flush_immediately=bool(
            audit_raw.get("flush_immediately", AuditConfig.flush_immediately)
        ),
        max_entry_size=int(
            audit_raw.get("max_entry_size", AuditConfig.max_entry_size)
        ),
    )

    roles_raw = raw.get("allowed_roles")
    if roles_raw:
        allowed = tuple(Role(r) for r in roles_raw)
    else:
        allowed = AuthenticationConfig.allowed_roles

    return AuthenticationConfig(
        jwt=jwt_cfg,
        session=sess_cfg,
        password=pw_cfg,
        lockout=lock_cfg,
        mfa=mfa_cfg,
        audit=audit_cfg,
        allowed_roles=allowed,
        environment=raw.get("environment", "production"),
        debug=bool(raw.get("debug", False)),
    )


def load_config(path: Optional[str] = None) -> AuthenticationConfig:
    """Load configuration from YAML.

    Resolution order:
      1. The ``path`` argument if provided.
      2. ``$AVCS_AUTH_CONFIG`` environment variable.
      3. ``/etc/avcs/auth.yaml`` if present.
      4. Built-in production-safe defaults.
    """
    candidates: List[str] = []
    if path:
        candidates.append(path)
    env_path = os.environ.get("AVCS_AUTH_CONFIG")
    if env_path:
        candidates.append(env_path)
    candidates.append("/etc/avcs/auth.yaml")

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            logger.info("Loading authentication config from %s", candidate)
            with open(candidate, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            return _build_config(raw)

    logger.warning(
        "No auth config file found; returning built-in defaults. "
        "This is acceptable for tests but unsafe for production."
    )
    return AuthenticationConfig(
        environment="development",
        jwt=JWTConfig(secret_key="dev-secret-not-for-production"),
    )


__all__ = [
    "AuthenticationConfig",
    "JWTConfig",
    "SessionConfig",
    "PasswordPolicy",
    "LockoutPolicy",
    "MFAConfig",
    "AuditConfig",
    "load_config",
]
