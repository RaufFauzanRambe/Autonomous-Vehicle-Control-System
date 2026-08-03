"""
utils.py
========

Cryptographic and identity utility functions used across the
authentication sub-system.

The functions here are deliberately stateless and side-effect free
except where unavoidable (e.g. ``hash_password`` must read the bcrypt
cost factor). They exist so that higher-level modules can stay focused on
business logic rather than re-implementing salt generation, TOTP
computation, or constant-time comparisons in five different ways.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import struct
import time
from typing import Optional, Tuple

import bcrypt

from .constants import DEFAULT_TOTP_DIGITS, DEFAULT_TOTP_STEP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def generate_salt(rounds: int = 12) -> bytes:
    """Return a fresh bcrypt salt at the requested cost factor."""
    if rounds < 4 or rounds > 31:
        raise ValueError("bcrypt rounds must be in [4, 31]")
    return bcrypt.gensalt(rounds=rounds)


def hash_password(password: str, rounds: int = 12) -> str:
    """Hash a password with bcrypt and return the ASCII hash string."""
    if not password:
        raise ValueError("password must be non-empty")
    pw_bytes = password.encode("utf-8")
    salt = generate_salt(rounds)
    digest = bcrypt.hashpw(pw_bytes, salt)
    return digest.decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify ``password`` against a previously computed bcrypt hash."""
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        logger.warning("Malformed password hash supplied to verify_password")
        return False


# ---------------------------------------------------------------------------
# Session / token identifiers
# ---------------------------------------------------------------------------
def generate_session_id(length: int = 32) -> str:
    """Return a cryptographically-secure URL-safe session identifier."""
    return secrets.token_urlsafe(length)


def generate_token_id(length: int = 16) -> str:
    """Return a short opaque token identifier (e.g. for JWT ``jti``)."""
    return secrets.token_hex(length)


def generate_nonce(length: int = 16) -> bytes:
    """Return ``length`` bytes of cryptographic randomness."""
    return secrets.token_bytes(length)


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------
def safe_compare(a: str, b: str) -> bool:
    """Constant-time string comparison; returns ``True`` if equal."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ---------------------------------------------------------------------------
# TOTP (RFC 6238)
# ---------------------------------------------------------------------------
def generate_totp_secret(length: int = 20) -> str:
    """Return a base32-encoded TOTP shared secret."""
    raw = secrets.token_bytes(length)
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def _hotp(secret: str, counter: int, digits: int = DEFAULT_TOTP_DIGITS) -> str:
    """RFC 4226 HOTP implementation."""
    padding = "=" * ((8 - len(secret) % 8) % 8)
    key = base64.b32decode(secret + padding, casefold=True)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{code % (10 ** digits):0{digits}d}"


def compute_totp(
    secret: str,
    timestamp: Optional[int] = None,
    step: int = DEFAULT_TOTP_STEP,
    digits: int = DEFAULT_TOTP_DIGITS,
) -> str:
    """Compute the TOTP value for ``timestamp`` (defaults to now)."""
    if timestamp is None:
        timestamp = int(time.time())
    counter = timestamp // step
    return _hotp(secret, counter, digits)


def verify_totp(
    code: str,
    secret: str,
    timestamp: Optional[int] = None,
    step: int = DEFAULT_TOTP_STEP,
    digits: int = DEFAULT_TOTP_DIGITS,
    window: int = 1,
) -> bool:
    """Verify a TOTP code with ``±window`` step tolerance."""
    if not code or not secret:
        return False
    if timestamp is None:
        timestamp = int(time.time())
    base_counter = timestamp // step
    for offset in range(-window, window + 1):
        candidate = _hotp(secret, base_counter + offset, digits)
        if safe_compare(candidate, code):
            return True
    return False


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def device_fingerprint(components: dict) -> str:
    """Stable device fingerprint over a dict of identifying attributes."""
    blob = repr(sorted(components.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_token(token: str) -> str:
    """One-way SHA-256 of a token for blacklisting / lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def now_ts() -> int:
    """Return the current Unix timestamp."""
    return int(time.time())


def expires_at(issued_at: int, ttl_seconds: int) -> int:
    """Return expiry timestamp given ``issued_at`` and TTL."""
    return issued_at + int(ttl_seconds)


def is_expired(expires_at_ts: int, now: Optional[int] = None) -> bool:
    """Return ``True`` if ``expires_at_ts`` is in the past."""
    return (now or now_ts()) >= expires_at_ts


def split_bearer(header: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a ``Bearer <token>`` / ``Basic <…>`` Authorization header."""
    if not header or " " not in header:
        return None, None
    scheme, _, value = header.partition(" ")
    return scheme.strip().lower(), value.strip() if value else None


__all__ = [
    "generate_salt",
    "hash_password",
    "verify_password",
    "generate_session_id",
    "generate_token_id",
    "generate_nonce",
    "safe_compare",
    "generate_totp_secret",
    "compute_totp",
    "verify_totp",
    "device_fingerprint",
    "hash_token",
    "now_ts",
    "expires_at",
    "is_expired",
    "split_bearer",
]
