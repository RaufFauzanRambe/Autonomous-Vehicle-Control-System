"""Elliptic-curve cryptography (ECDSA + ECDH).

Provides :class:`ECCCipher` which wraps:

    * ECDSA on NIST P-384 (default), P-256 and P-521,
    * Ed25519 for fast, deterministic signing,
    * X25519 / ECDH for ephemeral key agreement.

Encryption with an elliptic-curve key is achieved via the
ECIES-inspired pattern of ECDH + AES-GCM (see :meth:`encrypt`).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, x25519
from cryptography.hazmat.primitives.asymmetric.padding import MGF1
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .aes_encryption import AESCipher
from .constants import DEFAULT_AES_GCM_IV_SIZE
from .utils import safe_random

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]

#: Map of curve name -> cryptography curve object.
_CURVE_MAP = {
    "SECP256R1": ec.SECP256R1(),
    "SECP384R1": ec.SECP384R1(),
    "SECP521R1": ec.SECP521R1(),
}

#: Default curve used by :meth:`generate_keypair`.
DEFAULT_CURVE_NAME = "SECP384R1"


class ECCCipher:
    """ECDSA / ECDH helper.

    A single instance can serve as either a signer (needs private key)
    or verifier (needs public key) or both.  Set ``use_ed25519=True``
    to prefer Ed25519 for signing operations.
    """

    def __init__(
        self,
        private_key: Optional[Union[ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey]] = None,
        public_key: Optional[Union[ec.EllipticCurvePublicKey, ed25519.Ed25519PublicKey]] = None,
        curve_name: str = DEFAULT_CURVE_NAME,
        use_ed25519: bool = False,
    ):
        self._curve_name = curve_name
        self._use_ed25519 = use_ed25519
        if private_key is None and public_key is None:
            if use_ed25519:
                private_key = ed25519.Ed25519PrivateKey.generate()
                public_key = private_key.public_key()
            else:
                curve = _CURVE_MAP.get(curve_name)
                if curve is None:
                    raise ValueError(f"Unsupported curve: {curve_name}")
                private_key = ec.generate_private_key(curve)
                public_key = private_key.public_key()
        self._private_key = private_key
        self._public_key = public_key
        if self._public_key is None and private_key is not None:
            self._public_key = private_key.public_key()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_keypair(
        curve_name: str = DEFAULT_CURVE_NAME,
    ) -> tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        """Generate an ECDSA keypair on the requested NIST curve."""
        curve = _CURVE_MAP.get(curve_name)
        if curve is None:
            raise ValueError(f"Unsupported curve: {curve_name}")
        priv = ec.generate_private_key(curve)
        return priv, priv.public_key()

    @staticmethod
    def generate_x25519_keypair() -> tuple[x25519.X25519PrivateKey, x25519.X25519PublicKey]:
        """Generate an X25519 keypair for ECDH."""
        priv = x25519.X25519PrivateKey.generate()
        return priv, priv.public_key()

    @staticmethod
    def generate_ed25519_keypair() -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
        """Generate an Ed25519 keypair."""
        priv = ed25519.Ed25519PrivateKey.generate()
        return priv, priv.public_key()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def public_pem(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def private_pem(self) -> bytes:
        if self._private_key is None:
            raise RuntimeError("No private key available")
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    # ------------------------------------------------------------------
    # ECDH key derivation
    # ------------------------------------------------------------------

    def derive_shared_key(
        self,
        peer_public_key: Union[ec.EllipticCurvePublicKey, x25519.X25519PublicKey, bytes],
        length: int = 32,
        info: Optional[BytesLike] = None,
    ) -> bytes:
        """Derive a symmetric key via ECDH + HKDF-SHA256.

        Args:
            peer_public_key: The peer's public key (object or raw PEM bytes).
            length: Desired output length in bytes.
            info: Optional HKDF ``info`` context string.

        Returns:
            The shared secret expanded to ``length`` bytes.
        """
        if self._private_key is None:
            raise RuntimeError("No private key available for ECDH")

        if isinstance(peer_public_key, (bytes, bytearray)):
            peer_public_key = serialization.load_pem_public_key(bytes(peer_public_key))

        if isinstance(self._private_key, ec.EllipticCurvePrivateKey) and isinstance(
            peer_public_key, ec.EllipticCurvePublicKey
        ):
            shared = self._private_key.exchange(ec.ECDH(), peer_public_key)
        elif isinstance(self._private_key, x25519.X25519PrivateKey) and isinstance(
            peer_public_key, x25519.X25519PublicKey
        ):
            shared = self._private_key.exchange(peer_public_key)
        else:
            raise TypeError(
                "Private/peer public key type mismatch for ECDH"
            )

        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,
            info=bytes(info) if info is not None else b"avcs-ecdh-v1",
        ).derive(shared)
        return derived

    # ------------------------------------------------------------------
    # Sign / Verify
    # ------------------------------------------------------------------

    def sign(self, data: BytesLike) -> bytes:
        """Sign ``data`` with ECDSA-P-384 (or Ed25519 if enabled)."""
        if self._private_key is None:
            raise RuntimeError("No private key available for signing")
        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            return self._private_key.sign(bytes(data))
        # ECDSA
        hash_algo = self._ecdsa_hash()
        return self._private_key.sign(bytes(data), ec.ECDSA(hash_algo))

    def verify(self, signature: BytesLike, data: BytesLike) -> bool:
        """Verify an ECDSA or Ed25519 signature."""
        try:
            if isinstance(self._public_key, ed25519.Ed25519PublicKey):
                self._public_key.verify(bytes(signature), bytes(data))
            else:
                hash_algo = self._ecdsa_hash()
                self._public_key.verify(
                    bytes(signature), bytes(data), ec.ECDSA(hash_algo)
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("EC signature verification failed: %s", exc)
            return False

    def _ecdsa_hash(self) -> hashes.HashAlgorithm:
        if self._curve_name == "SECP256R1":
            return hashes.SHA256()
        if self._curve_name == "SECP384R1":
            return hashes.SHA384()
        return hashes.SHA512()

    # ------------------------------------------------------------------
    # ECIES-style hybrid encryption
    # ------------------------------------------------------------------

    def encrypt(
        self,
        plaintext: BytesLike,
        peer_public_key: Optional[
            Union[ec.EllipticCurvePublicKey, x25519.X25519PublicKey, bytes]
        ] = None,
    ) -> bytes:
        """Encrypt ``plaintext`` using ECIES (ECDH + AES-GCM).

        The encryption flow is:

            1. Generate an ephemeral X25519 keypair.
            2. Derive a shared AES-256 key via ECDH(ephemeral_priv, peer_pub).
            3. Encrypt the plaintext under the derived key with AES-GCM.
            4. Return ``ephemeral_pub || iv || ciphertext || tag``.

        The peer's private key (matching ``peer_public_key``) is needed
        to decrypt.
        """
        if peer_public_key is None:
            raise ValueError("peer_public_key is required for ECIES encryption")
        if isinstance(peer_public_key, (bytes, bytearray)):
            peer_public_key = serialization.load_pem_public_key(bytes(peer_public_key))

        ephemeral_priv = x25519.X25519PrivateKey.generate()
        ephemeral_pub_pem = ephemeral_priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if isinstance(peer_public_key, ec.EllipticCurvePublicKey):
            # Convert EC public key into X25519-compatible form is non-trivial;
            # require an X25519 peer key for ECIES.
            raise TypeError("ECIES requires an X25519 peer public key")
        shared = ephemeral_priv.exchange(peer_public_key)
        key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"avcs-ecies-v1",
        ).derive(shared)
        ct = AESCipher(key=key).encrypt(plaintext)
        return ephemeral_pub_pem + ct

    def decrypt(
        self,
        blob: BytesLike,
        private_key: Optional[x25519.X25519PrivateKey] = None,
    ) -> bytes:
        """Decrypt a blob produced by :meth:`encrypt`.

        Args:
            blob: ``ephemeral_pub || iv || ciphertext || tag``.
            private_key: The X25519 private key matching the peer public
                key used during encryption.  If ``None``, the instance's
                own private key is used (must be X25519).
        """
        raw = bytes(blob)
        if len(raw) < 32 + DEFAULT_AES_GCM_IV_SIZE + 16:
            raise ValueError("Blob too short for ECIES")
        ephemeral_pub_raw = raw[:32]
        ct = raw[32:]
        ephemeral_pub = x25519.X25519PublicKey.from_public_bytes(ephemeral_pub_raw)

        priv = private_key
        if priv is None:
            if not isinstance(self._private_key, x25519.X25519PrivateKey):
                raise TypeError("Instance private key must be X25519 for ECIES")
            priv = self._private_key

        shared = priv.exchange(ephemeral_pub)
        key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"avcs-ecies-v1",
        ).derive(shared)
        return AESCipher(key=key).decrypt(ct)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_private_key(self) -> bool:
        return self._private_key is not None

    @property
    def curve_name(self) -> str:
        return self._curve_name


__all__ = ["ECCCipher", "DEFAULT_CURVE_NAME"]
