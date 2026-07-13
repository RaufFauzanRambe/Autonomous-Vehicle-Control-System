"""Top-level orchestrator for the AVCS encryption subsystem.

:class:`EncryptionManager` is the single entry point that the rest of
the AVCS stack should use for cryptographic operations.  It wires
together the AES / RSA / ECC primitives, the key manager, the
certificate manager, the digital-signature facade, the secure-channel
implementation, the TLS factory, the data-protection manager and the
secrets manager, exposing a small, opinionated surface area:

    * encrypt_data / decrypt_data   (symmetric by default)
    * sign_data / verify_signature  (Ed25519 by default)
    * hash_data                     (SHA-256 by default)
    * establish_secure_channel      (X25519 + AES-GCM)
    * get_status                    (operational telemetry)

Internal components can be replaced individually for testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union

from .aes_encryption import AESCipher
from .certificate_manager import CertificateManager
from .config import EncryptionConfig, load_config
from .constants import (
    DEFAULT_AES_KEY_SIZE,
    DataClassification,
    HashAlgorithm,
    KeyAlgorithm,
    KeyStatus,
    KeyUsage,
)
from .data_protection import DataProtectionManager
from .digital_signature import DigitalSignature, SignatureAlgorithm
from .ecc_encryption import ECCCipher
from .hash_functions import sha256, sha512, sha3_256, blake2b, hmac_sha256
from .key_manager import KeyManager
from .key_rotation import KeyRotationManager
from .rsa_encryption import RSACipher
from .secrets_manager import SecretsManager
from .secure_channel import SecureChannel
from .tls_manager import TLSManager
from .utils import bytes_to_base64, base64_to_bytes

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


@dataclass
class EncryptionStatus:
    """Snapshot of the encryption subsystem state."""

    initialized_at: float
    default_symmetric_algorithm: str
    default_signature_algorithm: str
    default_hash_algorithm: str
    active_keys: int = 0
    retired_keys: int = 0
    stored_secrets: int = 0
    issued_certificates: int = 0
    pending_rotations: int = 0
    last_error: Optional[str] = None
    components: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "initialized_at_iso": datetime.fromtimestamp(
                self.initialized_at, tz=timezone.utc
            ).isoformat(),
        }


class EncryptionManager:
    """Central facade over every cryptographic capability in AVCS."""

    def __init__(self, config: Optional[EncryptionConfig] = None):
        self._config: EncryptionConfig = config or load_config()
        import time
        self._initialized_at = time.time()

        # Sub-components ---------------------------------------------------
        self._key_manager = KeyManager(self._config.storage.keystore_path)
        self._rotation_manager = KeyRotationManager(
            self._key_manager,
            grace_period_seconds=self._config.rotation.grace_period_seconds,
        )
        self._cert_manager = CertificateManager(self._config.storage.cert_store_path)
        self._secrets_manager = SecretsManager(
            self._config.storage.secret_store_path
        )
        self._data_protection = DataProtectionManager()
        self._tls_manager = TLSManager()

        # Default symmetric key (lazily created)
        self._default_key_id: Optional[str] = None

        # Default signer (Ed25519)
        self._default_signer = DigitalSignature(
            algorithm=SignatureAlgorithm.ED25519
        )

        logger.info("EncryptionManager initialised (symmetric=%s)",
                    self._config.symmetric.default_algorithm.value)

    # ------------------------------------------------------------------
    # Component accessors (for advanced users)
    # ------------------------------------------------------------------

    @property
    def key_manager(self) -> KeyManager:
        return self._key_manager

    @property
    def rotation_manager(self) -> KeyRotationManager:
        return self._rotation_manager

    @property
    def certificate_manager(self) -> CertificateManager:
        return self._cert_manager

    @property
    def secrets_manager(self) -> SecretsManager:
        return self._secrets_manager

    @property
    def data_protection(self) -> DataProtectionManager:
        return self._data_protection

    @property
    def tls_manager(self) -> TLSManager:
        return self._tls_manager

    @property
    def config(self) -> EncryptionConfig:
        return self._config

    # ------------------------------------------------------------------
    # Symmetric encryption
    # ------------------------------------------------------------------

    def _ensure_default_key(self) -> str:
        if self._default_key_id is None:
            self._default_key_id = self._key_manager.create_key(
                algorithm=self._config.symmetric.default_algorithm,
                usage=KeyUsage.ENCRYPTION,
                description="EncryptionManager default symmetric key",
            )
        return self._default_key_id

    def encrypt_data(
        self,
        plaintext: BytesLike,
        key_id: Optional[str] = None,
        associated_data: Optional[BytesLike] = None,
    ) -> dict[str, str]:
        """Encrypt ``plaintext`` with AES-GCM.

        Returns a JSON-serialisable dict containing the key_id, Base64
        ciphertext and Base64 associated data.  Pass the dict verbatim
        to :meth:`decrypt_data` to recover the plaintext.
        """
        key_id = key_id or self._ensure_default_key()
        _, material, _ = self._key_manager.get_key(key_id)
        cipher = AESCipher(material)
        blob = cipher.encrypt(plaintext, associated_data=associated_data)
        return {
            "key_id": key_id,
            "algorithm": self._config.symmetric.default_algorithm.value,
            "ciphertext_b64": bytes_to_base64(blob),
            "associated_data_b64": (
                bytes_to_base64(associated_data) if associated_data else ""
            ),
        }

    def decrypt_data(self, envelope: dict[str, str]) -> bytes:
        """Decrypt an envelope produced by :meth:`encrypt_data`."""
        key_id = envelope["key_id"]
        _, material, _ = self._key_manager.get_key(key_id)
        cipher = AESCipher(material)
        blob = base64_to_bytes(envelope["ciphertext_b64"])
        ad_b64 = envelope.get("associated_data_b64", "")
        ad = base64_to_bytes(ad_b64) if ad_b64 else None
        return cipher.decrypt(blob, associated_data=ad)

    # ------------------------------------------------------------------
    # Digital signatures
    # ------------------------------------------------------------------

    def sign_data(
        self,
        data: BytesLike,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519,
        key_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Sign ``data`` and return a serialisable envelope."""
        if key_id is not None:
            meta, material, is_public_only = self._key_manager.get_key(key_id)
            from cryptography.hazmat.primitives import serialization
            if is_public_only:
                raise PermissionError(f"Key {key_id} is public-only; cannot sign")
            priv = serialization.load_pem_private_key(material, password=None)
            signer = DigitalSignature(algorithm=algorithm, private_key=priv)
        else:
            signer = self._default_signer
        signature = signer.sign(data)
        return {
            "algorithm": algorithm.value,
            "key_id": key_id or "default-ed25519",
            "signature_b64": bytes_to_base64(signature),
        }

    def verify_signature(
        self,
        data: BytesLike,
        envelope: dict[str, str],
        public_key_pem: Optional[bytes] = None,
    ) -> bool:
        """Verify a signature envelope. Returns ``True`` if valid."""
        algorithm = SignatureAlgorithm(envelope["algorithm"])
        if public_key_pem is not None:
            from cryptography.hazmat.primitives import serialization
            pub = serialization.load_pem_public_key(public_key_pem)
            verifier = DigitalSignature(algorithm=algorithm, public_key=pub)
        else:
            verifier = self._default_signer
        signature = base64_to_bytes(envelope["signature_b64"])
        return verifier.verify(signature, data)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def hash_data(
        self,
        data: BytesLike,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> str:
        """Hash ``data`` and return a hex digest."""
        if algorithm == HashAlgorithm.SHA256:
            digest = sha256(data)
        elif algorithm == HashAlgorithm.SHA512:
            digest = sha512(data)
        elif algorithm == HashAlgorithm.SHA3_256:
            digest = sha3_256(data)
        elif algorithm == HashAlgorithm.BLAKE2B:
            digest = blake2b(data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        return digest.hex()

    # ------------------------------------------------------------------
    # Secure channel
    # ------------------------------------------------------------------

    def establish_secure_channel(
        self,
        transport_send,
        transport_recv,
        identity: str = "avcs-component",
        is_server: bool = False,
    ) -> SecureChannel:
        """Create and connect a :class:`SecureChannel`."""
        channel = SecureChannel(
            transport_send=transport_send,
            transport_recv=transport_recv,
            identity=identity,
        )
        if is_server:
            channel.handshake()
        else:
            channel.connect()
        return channel

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> EncryptionStatus:
        """Return a snapshot of the manager's operational state."""
        keys = self._key_manager.list_keys()
        return EncryptionStatus(
            initialized_at=self._initialized_at,
            default_symmetric_algorithm=self._config.symmetric.default_algorithm.value,
            default_signature_algorithm=SignatureAlgorithm.ED25519.value,
            default_hash_algorithm=HashAlgorithm.SHA256.value,
            active_keys=len([k for k in keys if k.status == KeyStatus.ACTIVE]),
            retired_keys=len([k for k in keys if k.status == KeyStatus.RETIRED]),
            stored_secrets=len(self._secrets_manager),
            issued_certificates=len(self._cert_manager.list_issued()),
            pending_rotations=len(self._rotation_manager.pending_rotations()),
            components={
                "key_manager": "ok",
                "rotation_manager": "ok",
                "cert_manager": "ok",
                "secrets_manager": "ok",
                "data_protection": "ok",
                "tls_manager": "ok",
            },
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    def make_aes_cipher(self, key_id: Optional[str] = None) -> AESCipher:
        """Return a ready-to-use :class:`AESCipher` bound to ``key_id``."""
        key_id = key_id or self._ensure_default_key()
        _, material, _ = self._key_manager.get_key(key_id)
        return AESCipher(material)

    def make_rsa_cipher(self, key_id: str) -> RSACipher:
        """Build an :class:`RSACipher` from a managed RSA key."""
        meta, material, is_public_only = self._key_manager.get_key(key_id)
        if meta.algorithm not in (KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_3072, KeyAlgorithm.RSA_4096):
            raise ValueError(f"Key {key_id} is not RSA")
        from cryptography.hazmat.primitives import serialization
        if is_public_only:
            pub = serialization.load_pem_public_key(material)
            return RSACipher(public_key=pub)
        priv = serialization.load_pem_private_key(material, password=None)
        return RSACipher(private_key=priv)

    def make_ecc_cipher(self, key_id: str) -> ECCCipher:
        """Build an :class:`ECCCipher` from a managed EC/Ed25519 key."""
        meta, material, is_public_only = self._key_manager.get_key(key_id)
        from cryptography.hazmat.primitives import serialization
        if is_public_only:
            pub = serialization.load_pem_public_key(material)
            return ECCCipher(public_key=pub, curve_name="SECP384R1")
        priv = serialization.load_pem_private_key(material, password=None)
        return ECCCipher(private_key=priv)


__all__ = ["EncryptionManager", "EncryptionStatus"]
