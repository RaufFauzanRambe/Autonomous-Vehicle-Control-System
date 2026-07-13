"""High-level secrets manager using envelope encryption (DEK + KEK).

Each secret stored by :class:`SecretsManager` is encrypted with a
unique Data Encryption Key (DEK).  The DEK itself is wrapped (encrypted)
by a master Key Encryption Key (KEK) and persisted alongside the
ciphertext.  This pattern, popularised by AWS KMS and Google Cloud KMS,
allows the KEK to live in a Hardware Security Module while bulk
encrypted data remains cheap to read and rotate.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .aes_encryption import AESCipher
from .constants import DEFAULT_KEY_TTL_SECONDS
from .utils import bytes_to_base64, base64_to_bytes, safe_random

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


@dataclass
class SecretRecord:
    """Persisted representation of a stored secret."""

    secret_id: str
    name: str
    wrapped_dek_b64: str
    ciphertext_b64: str
    created_at: float = field(default_factory=time.time)
    last_rotated_at: Optional[float] = None
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SecretsManager:
    """Envelope-encrypted secret store backed by a JSON file."""

    def __init__(
        self,
        store_path: Union[str, os.PathLike] = "/var/lib/avcs/secrets/secrets.json",
        kek: Optional[bytes] = None,
    ):
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, SecretRecord] = {}
        self._name_to_id: dict[str, str] = {}
        self._kek = kek or self._load_or_create_kek()
        self._kek_cipher = AESCipher(self._kek)
        self._load_store()

    # ------------------------------------------------------------------
    # KEK
    # ------------------------------------------------------------------

    def _load_or_create_kek(self) -> bytes:
        kek_path = self._store_path.parent / "kek.bin"
        if kek_path.exists():
            kek = kek_path.read_bytes()
            logger.info("Loaded KEK from %s", kek_path)
        else:
            kek = safe_random(32)
            kek_path.write_bytes(kek)
            os.chmod(kek_path, 0o600)
            logger.warning("Generated new KEK at %s — back this file up!", kek_path)
        return kek

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_store(self) -> None:
        if not self._store_path.exists():
            return
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
            for rec_dict in raw.get("secrets", []):
                record = SecretRecord(**rec_dict)
                self._records[record.secret_id] = record
                self._name_to_id[record.name] = record.secret_id
            logger.info("Loaded %d secrets from %s", len(self._records), self._store_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load secrets store: %s", exc)

    def _persist_store(self) -> None:
        payload = {"secrets": [r.to_dict() for r in self._records.values()]}
        tmp = self._store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._store_path)
        os.chmod(self._store_path, 0o600)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store_secret(
        self,
        name: str,
        value: BytesLike,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store ``value`` under ``name`` using a fresh DEK."""
        if name in self._name_to_id:
            raise ValueError(f"Secret {name!r} already exists; use rotate_secret() to update")
        secret_id = f"sec-{uuid.uuid4().hex[:12]}"
        dek = safe_random(32)
        wrapped_dek = self._kek_cipher.encrypt(dek)
        ciphertext = AESCipher(dek).encrypt(value)
        record = SecretRecord(
            secret_id=secret_id,
            name=name,
            wrapped_dek_b64=bytes_to_base64(wrapped_dek),
            ciphertext_b64=bytes_to_base64(ciphertext),
            metadata=metadata or {},
        )
        self._records[secret_id] = record
        self._name_to_id[name] = secret_id
        self._persist_store()
        logger.info("Stored secret %s (id=%s)", name, secret_id)
        return secret_id

    def get_secret(self, name: str) -> bytes:
        """Retrieve and decrypt the secret named ``name``."""
        secret_id = self._name_to_id.get(name)
        if secret_id is None:
            raise KeyError(f"No such secret: {name!r}")
        record = self._records[secret_id]
        dek = self._kek_cipher.decrypt(base64_to_bytes(record.wrapped_dek_b64))
        plaintext = AESCipher(dek).decrypt(base64_to_bytes(record.ciphertext_b64))
        logger.debug("Retrieved secret %s", name)
        return plaintext

    def delete_secret(self, name: str) -> None:
        secret_id = self._name_to_id.pop(name, None)
        if secret_id is None:
            raise KeyError(f"No such secret: {name!r}")
        del self._records[secret_id]
        self._persist_store()
        logger.info("Deleted secret %s", name)

    def list_secrets(self) -> list[dict[str, Any]]:
        """Return metadata for all stored secrets (without revealing values)."""
        return [
            {
                "secret_id": r.secret_id,
                "name": r.name,
                "created_at": r.created_at,
                "last_rotated_at": r.last_rotated_at,
                "version": r.version,
                "metadata": dict(r.metadata),
            }
            for r in self._records.values()
        ]

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate_secret(self, name: str, new_value: Optional[BytesLike] = None) -> str:
        """Rotate the DEK (and optionally the value) for ``name``.

        If ``new_value`` is ``None`` the existing value is re-encrypted
        under a fresh DEK; otherwise the value is replaced.
        """
        secret_id = self._name_to_id.get(name)
        if secret_id is None:
            raise KeyError(f"No such secret: {name!r}")
        record = self._records[secret_id]
        # Decrypt with old DEK
        old_dek = self._kek_cipher.decrypt(base64_to_bytes(record.wrapped_dek_b64))
        old_value = AESCipher(old_dek).decrypt(base64_to_bytes(record.ciphertext_b64))
        value = bytes(new_value) if new_value is not None else old_value
        # Generate fresh DEK and re-encrypt
        new_dek = safe_random(32)
        wrapped = self._kek_cipher.encrypt(new_dek)
        ct = AESCipher(new_dek).encrypt(value)
        record.wrapped_dek_b64 = bytes_to_base64(wrapped)
        record.ciphertext_b64 = bytes_to_base64(ct)
        record.last_rotated_at = time.time()
        record.version += 1
        self._persist_store()
        logger.info("Rotated secret %s (v%d)", name, record.version)
        return record.secret_id

    # ------------------------------------------------------------------
    # Bulk rotate
    # ------------------------------------------------------------------

    def rotate_kek(self, new_kek: bytes) -> None:
        """Replace the KEK and re-wrap every DEK.

        Use this when the KEK itself needs to be rotated (e.g. after
        HSM key ceremony) without touching the underlying secrets.
        """
        if len(new_kek) != 32:
            raise ValueError("KEK must be 32 bytes")
        old_cipher = self._kek_cipher
        new_cipher = AESCipher(new_kek)
        for record in self._records.values():
            dek = old_cipher.decrypt(base64_to_bytes(record.wrapped_dek_b64))
            record.wrapped_dek_b64 = bytes_to_base64(new_cipher.encrypt(dek))
        self._kek = new_kek
        self._kek_cipher = new_cipher
        kek_path = self._store_path.parent / "kek.bin"
        kek_path.write_bytes(new_kek)
        os.chmod(kek_path, 0o600)
        self._persist_store()
        logger.info("Rotated KEK and re-wrapped %d DEKs", len(self._records))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._name_to_id


__all__ = ["SecretsManager", "SecretRecord"]
