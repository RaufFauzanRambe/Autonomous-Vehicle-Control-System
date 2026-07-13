"""Hash and MAC utility functions.

This module exposes a small, well-typed API around the digest
primitives offered by the standard library and the optional
``argon2-cffi`` package.  Every function returns ``bytes`` to make
constant-time comparisons straightforward.

Algorithms supported:
    * SHA-2 family (sha256, sha384, sha512)
    * SHA-3 family (sha3_256, sha3_512)
    * BLAKE2 (blake2b, blake2s)
    * HMAC-SHA256 / HMAC-SHA512
    * PBKDF2-HMAC (RFC 2898)
    * scrypt (RFC 7914)
    * Argon2id (RFC 9106) — requires ``argon2-cffi``
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Iterable, Union

from .constants import (
    DEFAULT_ARGON2_MEMORY_KIB,
    DEFAULT_ARGON2_PARALLELISM,
    DEFAULT_ARGON2_TIME_COST,
    DEFAULT_PBKDF2_HASH,
    DEFAULT_PBKDF2_ITERATIONS,
    DEFAULT_SCRYPT_N,
    DEFAULT_SCRYPT_P,
    DEFAULT_SCRYPT_R,
    HashAlgorithm,
)

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]

# Try to import argon2-cffi; degrade gracefully if unavailable.
try:  # pragma: no cover - exercised in environments without argon2-cffi
    from argon2.low_level import hash_secret_raw, Type as Argon2Type  # type: ignore[import-not-found]

    _ARGON2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ARGON2_AVAILABLE = False
    logger.warning(
        "argon2-cffi not installed; argon2() will raise RuntimeError. "
        "Install with: pip install argon2-cffi"
    )


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

def _hash(algorithm: str, data: BytesLike) -> bytes:
    raw = bytes(data)
    return hashlib.new(algorithm, raw).digest()


# ---------------------------------------------------------------------------
# SHA-2 family
# ---------------------------------------------------------------------------

def sha256(data: BytesLike) -> bytes:
    """Return the SHA-256 digest of ``data`` (32 bytes)."""
    return _hash("sha256", data)


def sha384(data: BytesLike) -> bytes:
    """Return the SHA-384 digest of ``data`` (48 bytes)."""
    return _hash("sha384", data)


def sha512(data: BytesLike) -> bytes:
    """Return the SHA-512 digest of ``data`` (64 bytes)."""
    return _hash("sha512", data)


# ---------------------------------------------------------------------------
# SHA-3 family
# ---------------------------------------------------------------------------

def sha3_256(data: BytesLike) -> bytes:
    """Return the SHA3-256 digest of ``data`` (32 bytes)."""
    return _hash("sha3_256", data)


def sha3_512(data: BytesLike) -> bytes:
    """Return the SHA3-512 digest of ``data`` (64 bytes)."""
    return _hash("sha3_512", data)


# ---------------------------------------------------------------------------
# BLAKE2
# ---------------------------------------------------------------------------

def blake2b(data: BytesLike, digest_size: int = 64, key: BytesLike = b"") -> bytes:
    """Return the BLAKE2b digest of ``data`` (default 64 bytes).

    Args:
        data: Input bytes.
        digest_size: Output size in bytes (1..64).
        key: Optional key for keyed BLAKE2b (MAC mode).
    """
    if not 1 <= digest_size <= 64:
        raise ValueError("blake2b digest_size must be in [1, 64]")
    return hashlib.blake2b(bytes(data), digest_size=digest_size, key=bytes(key)).digest()


def blake2s(data: BytesLike, digest_size: int = 32, key: BytesLike = b"") -> bytes:
    """Return the BLAKE2s digest of ``data`` (default 32 bytes)."""
    if not 1 <= digest_size <= 32:
        raise ValueError("blake2s digest_size must be in [1, 32]")
    return hashlib.blake2s(bytes(data), digest_size=digest_size, key=bytes(key)).digest()


# ---------------------------------------------------------------------------
# HMAC
# ---------------------------------------------------------------------------

def hmac_sha256(key: BytesLike, data: BytesLike) -> bytes:
    """Compute HMAC-SHA256(key, data) and return 32 bytes."""
    return hmac.new(bytes(key), bytes(data), hashlib.sha256).digest()


def hmac_sha512(key: BytesLike, data: BytesLike) -> bytes:
    """Compute HMAC-SHA512(key, data) and return 64 bytes."""
    return hmac.new(bytes(key), bytes(data), hashlib.sha512).digest()


# ---------------------------------------------------------------------------
# Password-based KDFs
# ---------------------------------------------------------------------------

def pbkdf2(
    password: BytesLike,
    salt: BytesLike,
    length: int,
    iterations: int = DEFAULT_PBKDF2_ITERATIONS,
    algorithm: str = DEFAULT_PBKDF2_HASH,
) -> bytes:
    """PBKDF2-HMAC password-based key derivation (RFC 2898)."""
    if length <= 0:
        raise ValueError("length must be positive")
    if iterations < 100_000:
        logger.warning("PBKDF2 iterations < 100000 is insecure: %d", iterations)
    return hashlib.pbkdf2_hmac(
        algorithm, bytes(password), bytes(salt), iterations, dklen=length
    )


def scrypt(
    password: BytesLike,
    salt: BytesLike,
    length: int,
    n: int = DEFAULT_SCRYPT_N,
    r: int = DEFAULT_SCRYPT_R,
    p: int = DEFAULT_SCRYPT_P,
) -> bytes:
    """scrypt password-based key derivation (RFC 7914).

    Defaults follow the 2023 OWASP recommendation (N=2**17, r=8, p=1).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of two >= 2")
    if r <= 0 or p <= 0:
        raise ValueError("r and p must be positive")
    try:
        return hashlib.scrypt(
            bytes(password), salt=bytes(salt), n=n, r=r, p=p, dklen=length
        )
    except ValueError as exc:
        # scrypt has hard memory limits; lower the parameters or raise.
        raise ValueError(f"scrypt parameters too large: {exc}") from exc


def argon2(
    password: BytesLike,
    salt: BytesLike,
    length: int,
    time_cost: int = DEFAULT_ARGON2_TIME_COST,
    memory_cost: int = DEFAULT_ARGON2_MEMORY_KIB,
    parallelism: int = DEFAULT_ARGON2_PARALLELISM,
) -> bytes:
    """Argon2id password-based key derivation (RFC 9106).

    Requires the optional ``argon2-cffi`` package.
    """
    if not _ARGON2_AVAILABLE:
        raise RuntimeError(
            "argon2-cffi is not installed; cannot compute Argon2 digest"
        )
    if length <= 0:
        raise ValueError("length must be positive")
    if len(salt) < 8:
        raise ValueError("salt must be at least 8 bytes for Argon2")
    return hash_secret_raw(
        secret=bytes(password),
        salt=bytes(salt),
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=length,
        type=Argon2Type.ID,
    )


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def hash_stream(
    algorithm: HashAlgorithm,
    stream: Iterable[BytesLike],
    chunk_size: int = 65_536,
) -> bytes:
    """Hash a potentially infinite stream incrementally.

    Args:
        algorithm: One of :class:`HashAlgorithm`.
        stream: An iterable yielding byte chunks.
        chunk_size: Hint for log messages; the actual chunk size is
            determined by the iterable.

    Returns:
        The final digest as ``bytes``.
    """
    mapping = {
        HashAlgorithm.SHA256: "sha256",
        HashAlgorithm.SHA384: "sha384",
        HashAlgorithm.SHA512: "sha512",
        HashAlgorithm.SHA3_256: "sha3_256",
        HashAlgorithm.SHA3_512: "sha3_512",
        HashAlgorithm.BLAKE2B: "blake2b",
        HashAlgorithm.BLAKE2S: "blake2s",
    }
    name = mapping.get(algorithm)
    if name is None:
        raise ValueError(f"Unsupported streaming algorithm: {algorithm}")
    hasher = hashlib.new(name)
    total = 0
    for chunk in stream:
        chunk_bytes = bytes(chunk)
        hasher.update(chunk_bytes)
        total += len(chunk_bytes)
    logger.debug("Streamed %d bytes through %s", total, name)
    return hasher.digest()


__all__ = [
    "sha256",
    "sha384",
    "sha512",
    "sha3_256",
    "sha3_512",
    "blake2b",
    "blake2s",
    "hmac_sha256",
    "hmac_sha512",
    "pbkdf2",
    "scrypt",
    "argon2",
    "hash_stream",
]
