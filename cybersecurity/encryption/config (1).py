"""Configuration management for the encryption subsystem.

Defines the :class:`EncryptionConfig` dataclass hierarchy and a YAML
loader (:func:`load_config`).  The configuration is intentionally
declarative so that it can be serialised back to disk or shipped to a
remote KMS without code changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from .constants import (
    DEFAULT_AES_KEY_SIZE,
    DEFAULT_AES_GCM_IV_SIZE,
    DEFAULT_KEY_TTL_SECONDS,
    DEFAULT_ASYM_KEY_TTL_SECONDS,
    DEFAULT_PBKDF2_ITERATIONS,
    DEFAULT_RSA_KEY_SIZE,
    KeyAlgorithm,
    HashAlgorithm,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymmetricConfig:
    """Configuration for symmetric cryptography."""

    default_algorithm: KeyAlgorithm = KeyAlgorithm.AES_256_GCM
    key_size: int = DEFAULT_AES_KEY_SIZE
    iv_size: int = DEFAULT_AES_GCM_IV_SIZE
    tag_size: int = 16


@dataclass(frozen=True)
class AsymmetricConfig:
    """Configuration for asymmetric cryptography."""

    rsa_key_size: int = DEFAULT_RSA_KEY_SIZE
    rsa_public_exponent: int = 65537
    ecdsa_curve: str = "SECP384R1"
    use_ed25519_for_signing: bool = True
    use_x25519_for_kex: bool = True


@dataclass(frozen=True)
class KDFConfig:
    """Configuration for password-based key derivation."""

    pbkdf2_iterations: int = DEFAULT_PBKDF2_ITERATIONS
    pbkdf2_hash: HashAlgorithm = HashAlgorithm.SHA256
    scrypt_n: int = 2 ** 17
    scrypt_r: int = 8
    scrypt_p: int = 1
    argon2_memory_kib: int = 19_456
    argon2_time_cost: int = 2
    argon2_parallelism: int = 1
    salt_size: int = 16


@dataclass(frozen=True)
class RotationConfig:
    """Key rotation policy."""

    symmetric_ttl_seconds: int = DEFAULT_KEY_TTL_SECONDS
    asymmetric_ttl_seconds: int = DEFAULT_ASYM_KEY_TTL_SECONDS
    grace_period_seconds: int = 24 * 3600
    auto_rotate: bool = True


@dataclass(frozen=True)
class TLSConfig:
    """TLS configuration for V2X and fleet communication."""

    min_version: str = "TLSv1.3"
    verify_mode: str = "CERT_REQUIRED"
    check_hostname: bool = True
    prefer_server_ciphers: bool = False
    cipher_suites: tuple[str, ...] = (
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256",
    )


@dataclass(frozen=True)
class StorageConfig:
    """Paths and KMS endpoint configuration."""

    keystore_path: str = "/var/lib/avcs/keys"
    secret_store_path: str = "/var/lib/avcs/secrets"
    cert_store_path: str = "/var/lib/avcs/certs"
    kms_endpoint: Optional[str] = None
    kms_credentials_file: Optional[str] = None
    hsm_slot: Optional[int] = None


@dataclass(frozen=True)
class EncryptionConfig:
    """Top-level configuration object for :class:`~.encryption.EncryptionManager`."""

    symmetric: SymmetricConfig = field(default_factory=SymmetricConfig)
    asymmetric: AsymmetricConfig = field(default_factory=AsymmetricConfig)
    kdf: KDFConfig = field(default_factory=KDFConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    enable_audit_log: bool = True
    audit_log_path: str = "/var/log/avcs/encryption-audit.log"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _coerce_enum(enum_cls, value):
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _build_symmetric(raw: dict[str, Any]) -> SymmetricConfig:
    return SymmetricConfig(
        default_algorithm=_coerce_enum(KeyAlgorithm, raw.get("default_algorithm", KeyAlgorithm.AES_256_GCM)),
        key_size=int(raw.get("key_size", DEFAULT_AES_KEY_SIZE)),
        iv_size=int(raw.get("iv_size", DEFAULT_AES_GCM_IV_SIZE)),
        tag_size=int(raw.get("tag_size", 16)),
    )


def _build_asymmetric(raw: dict[str, Any]) -> AsymmetricConfig:
    return AsymmetricConfig(
        rsa_key_size=int(raw.get("rsa_key_size", DEFAULT_RSA_KEY_SIZE)),
        rsa_public_exponent=int(raw.get("rsa_public_exponent", 65537)),
        ecdsa_curve=str(raw.get("ecdsa_curve", "SECP384R1")),
        use_ed25519_for_signing=bool(raw.get("use_ed25519_for_signing", True)),
        use_x25519_for_kex=bool(raw.get("use_x25519_for_kex", True)),
    )


def _build_kdf(raw: dict[str, Any]) -> KDFConfig:
    return KDFConfig(
        pbkdf2_iterations=int(raw.get("pbkdf2_iterations", DEFAULT_PBKDF2_ITERATIONS)),
        pbkdf2_hash=_coerce_enum(HashAlgorithm, raw.get("pbkdf2_hash", HashAlgorithm.SHA256)),
        scrypt_n=int(raw.get("scrypt_n", 2 ** 17)),
        scrypt_r=int(raw.get("scrypt_r", 8)),
        scrypt_p=int(raw.get("scrypt_p", 1)),
        argon2_memory_kib=int(raw.get("argon2_memory_kib", 19_456)),
        argon2_time_cost=int(raw.get("argon2_time_cost", 2)),
        argon2_parallelism=int(raw.get("argon2_parallelism", 1)),
        salt_size=int(raw.get("salt_size", 16)),
    )


def _build_rotation(raw: dict[str, Any]) -> RotationConfig:
    return RotationConfig(
        symmetric_ttl_seconds=int(raw.get("symmetric_ttl_seconds", DEFAULT_KEY_TTL_SECONDS)),
        asymmetric_ttl_seconds=int(raw.get("asymmetric_ttl_seconds", DEFAULT_ASYM_KEY_TTL_SECONDS)),
        grace_period_seconds=int(raw.get("grace_period_seconds", 24 * 3600)),
        auto_rotate=bool(raw.get("auto_rotate", True)),
    )


def _build_tls(raw: dict[str, Any]) -> TLSConfig:
    return TLSConfig(
        min_version=str(raw.get("min_version", "TLSv1.3")),
        verify_mode=str(raw.get("verify_mode", "CERT_REQUIRED")),
        check_hostname=bool(raw.get("check_hostname", True)),
        prefer_server_ciphers=bool(raw.get("prefer_server_ciphers", False)),
        cipher_suites=tuple(raw.get("cipher_suites", (
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256",
        ))),
    )


def _build_storage(raw: dict[str, Any]) -> StorageConfig:
    return StorageConfig(
        keystore_path=str(raw.get("keystore_path", "/var/lib/avcs/keys")),
        secret_store_path=str(raw.get("secret_store_path", "/var/lib/avcs/secrets")),
        cert_store_path=str(raw.get("cert_store_path", "/var/lib/avcs/certs")),
        kms_endpoint=raw.get("kms_endpoint"),
        kms_credentials_file=raw.get("kms_credentials_file"),
        hsm_slot=raw.get("hsm_slot"),
    )


def load_config(path: str | os.PathLike | None = None) -> EncryptionConfig:
    """Load configuration from a YAML file.

    If ``path`` is ``None`` or the file does not exist, a default
    :class:`EncryptionConfig` is returned.  Environment variables with
    the ``AVCS_ENC_`` prefix override individual YAML keys (e.g.
    ``AVCS_ENC_TLS_MIN_VERSION`` overrides ``tls.min_version``).

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A fully populated :class:`EncryptionConfig`.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        path_obj = Path(path)
        if path_obj.exists():
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML is required to load YAML configuration"
                ) from exc
            with path_obj.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Configuration root must be a mapping")
            raw = loaded
            logger.info("Loaded encryption configuration from %s", path_obj)
        else:
            logger.warning("Config path %s does not exist; using defaults", path_obj)

    # Apply environment variable overrides
    for key, value in os.environ.items():
        if not key.startswith("AVCS_ENC_"):
            continue
        section_name, _, field_name = key[len("AVCS_ENC_"):].lower().partition("__")
        if section_name and field_name:
            raw.setdefault(section_name, {})[field_name] = value

    return EncryptionConfig(
        symmetric=_build_symmetric(raw.get("symmetric", {})),
        asymmetric=_build_asymmetric(raw.get("asymmetric", {})),
        kdf=_build_kdf(raw.get("kdf", {})),
        rotation=_build_rotation(raw.get("rotation", {})),
        tls=_build_tls(raw.get("tls", {})),
        storage=_build_storage(raw.get("storage", {})),
        enable_audit_log=bool(raw.get("enable_audit_log", True)),
        audit_log_path=str(raw.get("audit_log_path", "/var/log/avcs/encryption-audit.log")),
    )


__all__ = [
    "SymmetricConfig",
    "AsymmetricConfig",
    "KDFConfig",
    "RotationConfig",
    "TLSConfig",
    "StorageConfig",
    "EncryptionConfig",
    "load_config",
]
