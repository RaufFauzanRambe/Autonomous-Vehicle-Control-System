"""Encrypted-at-rest secrets store.

The :class:`SecureStorage` class protects keys, certificates, and bearer
tokens at rest using AES-256-GCM authenticated encryption. The master key
is derived from a hardware root of trust (TPM 2.0 sealed key) using
PBKDF2-HMAC-SHA256 with a configurable iteration count.

If ``cryptography`` is not installed at runtime, the store falls back to an
XOR-based obfuscation layer so that the module remains importable and
testable. Production deployments must install ``cryptography`` and supply a
real TPM-derived key.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import AES_GCM_NONCE_SIZE, AES_KEY_SIZE_BYTES
from .utils import compute_sha256, hex_encode, safe_compare

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class StoredSecret:
    """A single encrypted secret record."""

    key: str
    nonce: bytes
    ciphertext: bytes
    tag: bytes
    aad: bytes = b""
    created_at: float = field(default_factory=time.time)
    rotated_at: Optional[float] = None
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "nonce": hex_encode(self.nonce),
            "ciphertext": hex_encode(self.ciphertext),
            "tag": hex_encode(self.tag),
            "aad": hex_encode(self.aad),
            "created_at": self.created_at,
            "rotated_at": self.rotated_at,
            "version": self.version,
        }


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #


class CryptoBackend:
    """AES-256-GCM crypto backend.

    Falls back to an obfuscation layer when ``cryptography`` is missing. The
    fallback is *not* secure and only exists so the module remains importable
    on minimal CI images; an explicit warning is logged.
    """

    def __init__(self) -> None:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
            self._available = True
        except ImportError:  # pragma: no cover
            self._available = False
            logger.error("cryptography library missing; SecureStorage is NOT secure")

    @property
    def available(self) -> bool:
        return self._available

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> tuple[bytes, bytes, bytes]:
        nonce = os.urandom(AES_GCM_NONCE_SIZE)
        if self._available:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(key)
            ct = aesgcm.encrypt(nonce, plaintext, aad)
            # AESGCM returns ciphertext||tag; split last 16 bytes as tag.
            ciphertext, tag = ct[:-16], ct[-16:]
            return nonce, ciphertext, tag
        # Insecure fallback: XOR with key stream
        keystream = (key * (len(plaintext) // len(key) + 1))[: len(plaintext)]
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream))
        tag = compute_sha256(nonce + ciphertext + aad)
        return nonce, ciphertext, tag

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes, aad: bytes = b"") -> bytes:
        if self._available:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext + tag, aad)
        # Insecure fallback
        expected_tag = compute_sha256(nonce + ciphertext + aad)
        if not safe_compare(expected_tag, tag):
            raise ValueError("authentication tag mismatch")
        keystream = (key * (len(ciphertext) // len(key) + 1))[: len(ciphertext)]
        return bytes(c ^ k for c, k in zip(ciphertext, keystream))


# --------------------------------------------------------------------------- #
# Key derivation
# --------------------------------------------------------------------------- #


def derive_master_key(salt: bytes, password: bytes, iterations: int = 200_000) -> bytes:
    """Derive a 256-bit master key from a (TPM-supplied) password + salt."""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=AES_KEY_SIZE_BYTES,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(password)
    except ImportError:  # pragma: no cover
        # Fallback: hash-based key derivation (NOT secure, test only)
        digest = compute_sha256(password + salt)
        return digest


# --------------------------------------------------------------------------- #
# SecureStorage
# --------------------------------------------------------------------------- #


class SecureStorage:
    """Encrypted key/value store for vehicle secrets."""

    def __init__(
        self,
        store_path: str,
        master_key: Optional[bytes] = None,
        master_password: Optional[bytes] = None,
        salt: Optional[bytes] = None,
        kdf_iterations: int = 200_000,
        backend: Optional[CryptoBackend] = None,
    ) -> None:
        self.store_path = Path(store_path)
        self.kdf_iterations = kdf_iterations
        self.backend = backend or CryptoBackend()

        if master_key is None:
            if master_password is None:
                master_password = os.urandom(32)
                logger.warning("no master key/password supplied; using ephemeral random key")
            self._salt = salt or os.urandom(16)
            self._master_key = derive_master_key(self._salt, master_password, kdf_iterations)
        else:
            if len(master_key) != AES_KEY_SIZE_BYTES:
                raise ValueError(f"master key must be {AES_KEY_SIZE_BYTES} bytes")
            self._master_key = master_key
            self._salt = salt or b""

        self._secrets: Dict[str, StoredSecret] = {}
        self._lock = threading.RLock()
        self._key_version = 1
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._key_version = data.get("key_version", 1)
            for entry in data.get("secrets", []):
                self._secrets[entry["key"]] = StoredSecret(
                    key=entry["key"],
                    nonce=bytes.fromhex(entry["nonce"]),
                    ciphertext=bytes.fromhex(entry["ciphertext"]),
                    tag=bytes.fromhex(entry["tag"]),
                    aad=bytes.fromhex(entry.get("aad", "")),
                    created_at=entry.get("created_at", time.time()),
                    rotated_at=entry.get("rotated_at"),
                    version=entry.get("version", 1),
                )
            logger.info("loaded %d secrets from %s", len(self._secrets), self.store_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load secure store: %s", exc)

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "key_version": self._key_version,
            "salt": hex_encode(self._salt),
            "secrets": [s.to_dict() for s in self._secrets.values()],
        }
        # Atomic write: tmp then rename
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def store(self, key: str, plaintext: bytes, aad: bytes = b"") -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        with self._lock:
            nonce, ciphertext, tag = self.backend.encrypt(self._master_key, plaintext, aad)
            self._secrets[key] = StoredSecret(
                key=key,
                nonce=nonce,
                ciphertext=ciphertext,
                tag=tag,
                aad=aad,
                version=self._key_version,
            )
            self._save()
            logger.debug("stored secret '%s' (%d bytes)", key, len(plaintext))

    def retrieve(self, key: str, aad: bytes = b"") -> bytes:
        with self._lock:
            secret = self._secrets.get(key)
            if secret is None:
                raise KeyError(f"no such secret: {key}")
            # AAD must match what was used at storage time
            effective_aad = aad if aad else secret.aad
            return self.backend.decrypt(self._master_key, secret.nonce, secret.ciphertext, secret.tag, effective_aad)

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = self._secrets.pop(key, None) is not None
            if existed:
                self._save()
            return existed

    def list_keys(self) -> List[str]:
        with self._lock:
            return sorted(self._secrets.keys())

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._secrets

    # ------------------------------------------------------------------ #
    # Key rotation
    # ------------------------------------------------------------------ #

    def rotate_master_key(self, new_key: Optional[bytes] = None) -> int:
        """Rotate the master key and re-encrypt all secrets.

        Returns the number of secrets re-encrypted.
        """
        if new_key is None:
            new_key = os.urandom(AES_KEY_SIZE_BYTES)
        if len(new_key) != AES_KEY_SIZE_BYTES:
            raise ValueError(f"new master key must be {AES_KEY_SIZE_BYTES} bytes")

        with self._lock:
            old_key = self._master_key
            self._master_key = new_key
            self._key_version += 1
            now = time.time()
            count = 0
            for key, secret in self._secrets.items():
                try:
                    plaintext = self.backend.decrypt(old_key, secret.nonce, secret.ciphertext, secret.tag, secret.aad)
                except Exception as exc:  # noqa: BLE001
                    logger.error("failed to decrypt '%s' during rotation: %s", key, exc)
                    continue
                nonce, ciphertext, tag = self.backend.encrypt(new_key, plaintext, secret.aad)
                secret.nonce = nonce
                secret.ciphertext = ciphertext
                secret.tag = tag
                secret.rotated_at = now
                secret.version = self._key_version
                count += 1
            self._save()
            logger.info("rotated master key; %d secrets re-encrypted", count)
            return count

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        with self._lock:
            return len(self._secrets)

    def get_metadata(self, key: str) -> Dict[str, Any]:
        with self._lock:
            secret = self._secrets.get(key)
            if secret is None:
                raise KeyError(key)
            meta = secret.to_dict()
            meta.pop("ciphertext")
            meta.pop("tag")
            meta.pop("nonce")
            return meta
