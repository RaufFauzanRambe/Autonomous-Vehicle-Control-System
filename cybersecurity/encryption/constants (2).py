"""Constants and enumerations for the encryption subsystem.

This module centralises all cryptographic constants, enumerations and
default parameters used throughout the ``encryption`` package. Keeping
them in one place guarantees that every consumer uses identical values
and makes security audits significantly easier.

Conforms to:
    * NIST SP 800-38D (GCM)
    * NIST SP 800-56A (ECDH)
    * NIST FIPS 186-5 (ECDSA, Ed25519)
    * RFC 8446 (TLS 1.3)
    * RFC 8017 (PKCS#1 v2.2 / OAEP / PSS)
"""

from __future__ import annotations

import enum
from typing import Final

# ---------------------------------------------------------------------------
# Symmetric cipher parameters
# ---------------------------------------------------------------------------

#: Default AES key size in bytes (AES-256).
DEFAULT_AES_KEY_SIZE: Final[int] = 32  # 256 bits
#: Default AES-GCM initialization vector size in bytes (96 bits per NIST).
DEFAULT_AES_GCM_IV_SIZE: Final[int] = 12  # 96 bits
#: Default AES-GCM authentication tag size in bytes (128 bits).
DEFAULT_AES_GCM_TAG_SIZE: Final[int] = 16  # 128 bits
#: Default AES-CBC initialization vector size in bytes.
DEFAULT_AES_CBC_IV_SIZE: Final[int] = 16  # 128 bits
#: Default AES block size in bytes.
AES_BLOCK_SIZE: Final[int] = 16

# ---------------------------------------------------------------------------
# Asymmetric cipher parameters
# ---------------------------------------------------------------------------

#: Default RSA key size in bits (RSA-3072 ~ 128-bit security).
DEFAULT_RSA_KEY_SIZE: Final[int] = 3072
#: Default RSA public exponent.
DEFAULT_RSA_PUBLIC_EXPONENT: Final[int] = 65537
#: OAEP hash output size used for RSA encryption.
DEFAULT_RSA_OAEP_HASH: Final[str] = "sha256"

# ---------------------------------------------------------------------------
# KDF / Salt / Nonce parameters
# ---------------------------------------------------------------------------

#: Default salt size in bytes.
DEFAULT_SALT_SIZE: Final[int] = 16  # 128 bits
#: Default nonce size in bytes.
DEFAULT_NONCE_SIZE: Final[int] = 16  # 128 bits
#: PBKDF2 default iteration count (OWASP 2023 minimum).
DEFAULT_PBKDF2_ITERATIONS: Final[int] = 600_000
#: Default PBKDF2 hash algorithm.
DEFAULT_PBKDF2_HASH: Final[str] = "sha256"
#: scrypt default N parameter (CPU/memory cost).
DEFAULT_SCRYPT_N: Final[int] = 2 ** 17  # 131072
#: scrypt default block size (r).
DEFAULT_SCRYPT_R: Final[int] = 8
#: scrypt default parallelism (p).
DEFAULT_SCRYPT_P: Final[int] = 1
#: Argon2id default memory cost in KiB.
DEFAULT_ARGON2_MEMORY_KIB: Final[int] = 19_456  # 19 MiB
#: Argon2id default time cost (iterations).
DEFAULT_ARGON2_TIME_COST: Final[int] = 2
#: Argon2id default parallelism.
DEFAULT_ARGON2_PARALLELISM: Final[int] = 1

# ---------------------------------------------------------------------------
# Key rotation / lifecycle
# ---------------------------------------------------------------------------

#: Default symmetric key TTL in seconds (90 days).
DEFAULT_KEY_TTL_SECONDS: Final[int] = 90 * 24 * 3600
#: Default asymmetric key TTL in seconds (365 days).
DEFAULT_ASYM_KEY_TTL_SECONDS: Final[int] = 365 * 24 * 3600
#: Default grace period after rotation during which both versions are valid.
DEFAULT_ROTATION_GRACE_SECONDS: Final[int] = 24 * 3600

# ---------------------------------------------------------------------------
# TLS parameters
# ---------------------------------------------------------------------------

#: Minimum TLS version allowed for V2X communication.
TLS_MIN_VERSION: Final[str] = "TLSv1.3"
#: Default cipher suite preference list for TLS 1.3.
TLS_CIPHER_SUITES: Final[tuple[str, ...]] = (
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class KeyAlgorithm(str, enum.Enum):
    """Enumerates the cryptographic algorithms supported by the package."""

    AES_128_GCM = "AES-128-GCM"
    AES_256_GCM = "AES-256-GCM"
    AES_256_CBC = "AES-256-CBC"
    RSA_2048 = "RSA-2048"
    RSA_3072 = "RSA-3072"
    RSA_4096 = "RSA-4096"
    ECDSA_P256 = "ECDSA-P256"
    ECDSA_P384 = "ECDSA-P384"
    ECDSA_P521 = "ECDSA-P521"
    ED25519 = "Ed25519"
    ED448 = "Ed448"
    X25519 = "X25519"
    X448 = "X448"

    @property
    def is_symmetric(self) -> bool:
        return self.name.startswith("AES")

    @property
    def is_asymmetric(self) -> bool:
        return not self.is_symmetric


class HashAlgorithm(str, enum.Enum):
    """Supported hash / digest algorithms."""

    SHA1 = "SHA1"          # legacy only, do not use for new designs
    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"
    SHA3_256 = "SHA3-256"
    SHA3_512 = "SHA3-512"
    BLAKE2B = "BLAKE2B"
    BLAKE2S = "BLAKE2S"

    @property
    def digest_size(self) -> int:
        sizes = {
            HashAlgorithm.SHA1: 20,
            HashAlgorithm.SHA256: 32,
            HashAlgorithm.SHA384: 48,
            HashAlgorithm.SHA512: 64,
            HashAlgorithm.SHA3_256: 32,
            HashAlgorithm.SHA3_512: 64,
            HashAlgorithm.BLAKE2B: 64,
            HashAlgorithm.BLAKE2S: 32,
        }
        return sizes[self]


class CipherMode(str, enum.Enum):
    """Block cipher modes of operation."""

    ECB = "ECB"  # NOT recommended; included only for legacy testing
    CBC = "CBC"
    CTR = "CTR"
    GCM = "GCM"
    CCM = "CCM"
    XTS = "XTS"


class KeyStatus(str, enum.Enum):
    """Lifecycle status of a managed key."""

    ACTIVE = "ACTIVE"
    ROTATING = "ROTATING"
    RETIRED = "RETIRED"
    COMPROMISED = "COMPROMISED"
    DESTROYED = "DESTROYED"


class KeyUsage(str, enum.Enum):
    """Intended usage of a key, used by the key manager metadata."""

    ENCRYPTION = "ENCRYPTION"
    DECRYPTION = "DECRYPTION"
    SIGNING = "SIGNING"
    VERIFICATION = "VERIFICATION"
    KEY_AGREEMENT = "KEY_AGREEMENT"
    WRAPPING = "WRAPPING"
    TRANSPORT = "TRANSPORT"


class DataClassification(str, enum.Enum):
    """Data classification levels used by ``DataProtectionManager``."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    RESTRICTED = "RESTRICTED"  # e.g. driver biometrics, V2X private keys


__all__ = [
    "DEFAULT_AES_KEY_SIZE",
    "DEFAULT_AES_GCM_IV_SIZE",
    "DEFAULT_AES_GCM_TAG_SIZE",
    "DEFAULT_AES_CBC_IV_SIZE",
    "AES_BLOCK_SIZE",
    "DEFAULT_RSA_KEY_SIZE",
    "DEFAULT_RSA_PUBLIC_EXPONENT",
    "DEFAULT_RSA_OAEP_HASH",
    "DEFAULT_SALT_SIZE",
    "DEFAULT_NONCE_SIZE",
    "DEFAULT_PBKDF2_ITERATIONS",
    "DEFAULT_PBKDF2_HASH",
    "DEFAULT_SCRYPT_N",
    "DEFAULT_SCRYPT_R",
    "DEFAULT_SCRYPT_P",
    "DEFAULT_ARGON2_MEMORY_KIB",
    "DEFAULT_ARGON2_TIME_COST",
    "DEFAULT_ARGON2_PARALLELISM",
    "DEFAULT_KEY_TTL_SECONDS",
    "DEFAULT_ASYM_KEY_TTL_SECONDS",
    "DEFAULT_ROTATION_GRACE_SECONDS",
    "TLS_MIN_VERSION",
    "TLS_CIPHER_SUITES",
    "KeyAlgorithm",
    "HashAlgorithm",
    "CipherMode",
    "KeyStatus",
    "KeyUsage",
    "DataClassification",
]
