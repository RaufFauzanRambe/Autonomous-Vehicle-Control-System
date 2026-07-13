"""Central key registry and lifecycle manager.

The :class:`KeyManager` stores metadata about every cryptographic key
used by the AVCS stack and persists key material to disk.  It is
designed to be backed by an in-process keystore for development and by
a Hardware Security Module (HSM) or cloud KMS for production.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa, x25519

from .constants import (
    DEFAULT_AES_KEY_SIZE,
    DEFAULT_ASYM_KEY_TTL_SECONDS,
    DEFAULT_KEY_TTL_SECONDS,
    DEFAULT_RSA_KEY_SIZE,
    KeyAlgorithm,
    KeyStatus,
    KeyUsage,
)
from .utils import bytes_to_base64, base64_to_bytes

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


@dataclass
class KeyMetadata:
    """Metadata describing a single managed key."""

    key_id: str
    algorithm: KeyAlgorithm
    usage: KeyUsage
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    rotated_from: Optional[str] = None
    version: int = 1
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expires_at == 0.0:
            if self.algorithm.is_symmetric:
                self.expires_at = self.created_at + DEFAULT_KEY_TTL_SECONDS
            else:
                self.expires_at = self.created_at + DEFAULT_ASYM_KEY_TTL_SECONDS

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["algorithm"] = self.algorithm.value
        d["usage"] = self.usage.value
        d["status"] = self.status.value
        return d


@dataclass
class _StoredKey:
    """Internal storage record combining metadata and raw material."""

    metadata: KeyMetadata
    material: bytes  # raw symmetric key OR PEM-encoded asymmetric key
    is_public_only: bool = False


class KeyManager:
    """Registry-based key manager.

    The default backing store is a single JSON file on disk.  In
    production, subclass and override :meth:`_load_store` /
    :meth:`_persist_store` to talk to a KMS or HSM.
    """

    def __init__(self, keystore_path: Union[str, os.PathLike] = "/var/lib/avcs/keys/keystore.json"):
        self._lock = threading.RLock()
        self._store: dict[str, _StoredKey] = {}
        self._keystore_path = Path(keystore_path)
        self._load_store()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_store(self) -> None:
        if not self._keystore_path.exists():
            logger.info("Keystore %s does not exist; starting empty", self._keystore_path)
            return
        try:
            raw = json.loads(self._keystore_path.read_text(encoding="utf-8"))
            for record in raw.get("keys", []):
                meta_dict = record["metadata"]
                meta = KeyMetadata(
                    key_id=meta_dict["key_id"],
                    algorithm=KeyAlgorithm(meta_dict["algorithm"]),
                    usage=KeyUsage(meta_dict["usage"]),
                    status=KeyStatus(meta_dict.get("status", KeyStatus.ACTIVE.value)),
                    created_at=meta_dict.get("created_at", time.time()),
                    expires_at=meta_dict.get("expires_at", 0.0),
                    rotated_from=meta_dict.get("rotated_from"),
                    version=meta_dict.get("version", 1),
                    description=meta_dict.get("description", ""),
                    tags=meta_dict.get("tags", {}),
                )
                stored = _StoredKey(
                    metadata=meta,
                    material=base64_to_bytes(record["material"]),
                    is_public_only=record.get("is_public_only", False),
                )
                self._store[meta.key_id] = stored
            logger.info("Loaded %d keys from %s", len(self._store), self._keystore_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load keystore: %s", exc)

    def _persist_store(self) -> None:
        self._keystore_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"keys": []}
        for stored in self._store.values():
            payload["keys"].append({
                "metadata": stored.metadata.to_dict(),
                "material": bytes_to_base64(stored.material),
                "is_public_only": stored.is_public_only,
            })
        tmp = self._keystore_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._keystore_path)
        os.chmod(self._keystore_path, 0o600)
        logger.debug("Persisted keystore with %d keys", len(self._store))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_key(
        self,
        algorithm: KeyAlgorithm,
        usage: KeyUsage,
        description: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate and store a new key, returning its ``key_id``."""
        key_id = f"key-{uuid.uuid4().hex[:12]}"
        if algorithm.is_symmetric:
            if algorithm in (KeyAlgorithm.AES_256_GCM, KeyAlgorithm.AES_256_CBC):
                material = os.urandom(DEFAULT_AES_KEY_SIZE)
            elif algorithm == KeyAlgorithm.AES_128_GCM:
                material = os.urandom(16)
            else:
                raise ValueError(f"Cannot generate key for {algorithm}")
            stored = _StoredKey(
                metadata=KeyMetadata(
                    key_id=key_id,
                    algorithm=algorithm,
                    usage=usage,
                    description=description,
                    tags=tags or {},
                ),
                material=material,
                is_public_only=False,
            )
        else:
            material, is_public_only = self._generate_asymmetric(algorithm)
            stored = _StoredKey(
                metadata=KeyMetadata(
                    key_id=key_id,
                    algorithm=algorithm,
                    usage=usage,
                    description=description,
                    tags=tags or {},
                ),
                material=material,
                is_public_only=is_public_only,
            )
        with self._lock:
            self._store[key_id] = stored
            self._persist_store()
        logger.info("Created key %s (%s/%s)", key_id, algorithm.value, usage.value)
        return key_id

    @staticmethod
    def _generate_asymmetric(algorithm: KeyAlgorithm) -> tuple[bytes, bool]:
        if algorithm in (KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_3072, KeyAlgorithm.RSA_4096):
            size = {KeyAlgorithm.RSA_2048: 2048, KeyAlgorithm.RSA_3072: 3072, KeyAlgorithm.RSA_4096: 4096}[algorithm]
            priv = rsa.generate_private_key(public_exponent=65537, key_size=size)
            pem = priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            return pem, False
        if algorithm == KeyAlgorithm.ECDSA_P384:
            priv = ec.generate_private_key(ec.SECP384R1())
        elif algorithm == KeyAlgorithm.ECDSA_P256:
            priv = ec.generate_private_key(ec.SECP256R1())
        elif algorithm == KeyAlgorithm.ECDSA_P521:
            priv = ec.generate_private_key(ec.SECP521R1())
        elif algorithm == KeyAlgorithm.ED25519:
            priv = ed25519.Ed25519PrivateKey.generate()
        elif algorithm == KeyAlgorithm.X25519:
            priv = x25519.X25519PrivateKey.generate()
        else:
            raise ValueError(f"Cannot generate key for {algorithm}")
        pem = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return pem, False

    def get_key(self, key_id: str) -> tuple[KeyMetadata, bytes, bool]:
        """Return ``(metadata, material, is_public_only)`` for ``key_id``."""
        with self._lock:
            stored = self._store.get(key_id)
            if stored is None:
                raise KeyError(f"Unknown key_id: {key_id}")
            if stored.metadata.status == KeyStatus.DESTROYED:
                raise PermissionError(f"Key {key_id} has been destroyed")
            return stored.metadata, stored.material, stored.is_public_only

    def delete_key(self, key_id: str, secure_wipe: bool = True) -> None:
        """Remove ``key_id`` from the registry and zero its material."""
        with self._lock:
            stored = self._store.get(key_id)
            if stored is None:
                raise KeyError(f"Unknown key_id: {key_id}")
            if secure_wipe and isinstance(stored.material, bytearray):
                for i in range(len(stored.material)):
                    stored.material[i] = 0
            stored.metadata.status = KeyStatus.DESTROYED
            del self._store[key_id]
            self._persist_store()
            logger.info("Deleted key %s (secure_wipe=%s)", key_id, secure_wipe)

    def list_keys(
        self,
        algorithm: Optional[KeyAlgorithm] = None,
        status: Optional[KeyStatus] = None,
    ) -> list[KeyMetadata]:
        """List key metadata, optionally filtered."""
        with self._lock:
            results = []
            for stored in self._store.values():
                if algorithm is not None and stored.metadata.algorithm != algorithm:
                    continue
                if status is not None and stored.metadata.status != status:
                    continue
                results.append(stored.metadata)
            return sorted(results, key=lambda m: m.created_at)

    def import_key(
        self,
        material: BytesLike,
        algorithm: KeyAlgorithm,
        usage: KeyUsage,
        description: str = "",
        is_public_only: bool = False,
    ) -> str:
        """Import externally-generated key material."""
        key_id = f"key-{uuid.uuid4().hex[:12]}"
        stored = _StoredKey(
            metadata=KeyMetadata(
                key_id=key_id,
                algorithm=algorithm,
                usage=usage,
                description=description,
            ),
            material=bytes(material),
            is_public_only=is_public_only,
        )
        with self._lock:
            self._store[key_id] = stored
            self._persist_store()
        logger.info("Imported key %s (%s)", key_id, algorithm.value)
        return key_id

    def export_key(self, key_id: str, public_only: bool = False) -> bytes:
        """Export raw key material.

        If ``public_only`` is ``True`` and the key is asymmetric, only
        the public component is returned (PEM-encoded).
        """
        meta, material, is_public_only = self.get_key(key_id)
        if public_only and not meta.algorithm.is_symmetric and not is_public_only:
            priv = serialization.load_pem_private_key(material, password=None)
            return priv.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        return material

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def set_status(self, key_id: str, status: KeyStatus) -> None:
        with self._lock:
            stored = self._store.get(key_id)
            if stored is None:
                raise KeyError(f"Unknown key_id: {key_id}")
            stored.metadata.status = status
            self._persist_store()

    def __len__(self) -> int:
        return len(self._store)


__all__ = ["KeyManager", "KeyMetadata"]
