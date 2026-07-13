"""RSA asymmetric encryption and signing (RSA-3072).

Implements:

    * Encryption: RSA-OAEP with SHA-256 (RFC 8017).
    * Signing:     RSASSA-PSS with SHA-256 (RFC 8017).

The default key size is 3072 bits which provides ~128-bit classical
security and is the minimum acceptable size for new deployments.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding, rsa
from cryptography.hazmat.primitives.asymmetric.padding import MGF1, OAEP, PSS

from .constants import DEFAULT_RSA_KEY_SIZE, DEFAULT_RSA_PUBLIC_EXPONENT

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


class RSACipher:
    """High-level wrapper around RSA-OAEP / RSA-PSS.

    The class can be constructed with an existing key pair, with only
    the public component (for verification / encryption) or with no key
    at all, in which case a fresh pair is generated on demand.
    """

    def __init__(
        self,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        public_key: Optional[rsa.RSAPublicKey] = None,
    ):
        if private_key is None and public_key is None:
            logger.info("Generating fresh RSA-%d keypair", DEFAULT_RSA_KEY_SIZE)
            private_key = rsa.generate_private_key(
                public_exponent=DEFAULT_RSA_PUBLIC_EXPONENT,
                key_size=DEFAULT_RSA_KEY_SIZE,
            )
        self._private_key = private_key
        self._public_key = public_key or (private_key.public_key() if private_key else None)
        if self._public_key is None:
            raise ValueError("Either private_key or public_key must be provided")

    # ------------------------------------------------------------------
    # Key generation / loading
    # ------------------------------------------------------------------

    @staticmethod
    def generate_keypair(
        key_size: int = DEFAULT_RSA_KEY_SIZE,
        public_exponent: int = DEFAULT_RSA_PUBLIC_EXPONENT,
    ) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate an RSA keypair of the requested size."""
        if key_size < 2048:
            raise ValueError("RSA key size must be at least 2048 bits")
        priv = rsa.generate_private_key(
            public_exponent=public_exponent, key_size=key_size
        )
        return priv, priv.public_key()

    @classmethod
    def load_public_key(cls, data: BytesLike) -> "RSACipher":
        """Build an :class:`RSACipher` from a PEM/DER-encoded public key."""
        pub = serialization.load_pem_public_key(bytes(data))
        if not isinstance(pub, rsa.RSAPublicKey):
            raise TypeError("Loaded key is not an RSA public key")
        return cls(public_key=pub)

    @classmethod
    def load_private_key(
        cls, data: BytesLike, password: Optional[BytesLike] = None
    ) -> "RSACipher":
        """Build an :class:`RSACipher` from a PEM-encoded private key."""
        priv = serialization.load_pem_private_key(bytes(data), password=password)
        if not isinstance(priv, rsa.RSAPrivateKey):
            raise TypeError("Loaded key is not an RSA private key")
        return cls(private_key=priv)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def public_pem(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def private_pem(
        self, password: Optional[BytesLike] = None
    ) -> bytes:
        if self._private_key is None:
            raise RuntimeError("No private key available")
        if password is not None:
            encryption = serialization.BestAvailableEncryption(bytes(password))
        else:
            encryption = serialization.NoEncryption()
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )

    # ------------------------------------------------------------------
    # Padding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _oaep_padding() -> OAEP:
        return OAEP(
            mgf=MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        )

    def _pss_padding(self) -> PSS:
        return PSS(
            mgf=MGF1(hashes.SHA256()),
            salt_length=PSS.MAX_LENGTH,
        )

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: BytesLike) -> bytes:
        """Encrypt ``plaintext`` with RSA-OAEP-SHA256.

        The maximum plaintext size for RSA-3072 is roughly 382 bytes.
        For larger data, use hybrid encryption (RSA-wrap-AES-key).
        """
        pt = bytes(plaintext)
        max_size = (self._public_key.key_size + 7) // 8 - 2 * hashes.SHA256().digest_size - 2
        if len(pt) > max_size:
            raise ValueError(
                f"Plaintext too large for RSA-OAEP: {len(pt)} > {max_size}"
            )
        return self._public_key.encrypt(pt, self._oaep_padding())

    def decrypt(self, ciphertext: BytesLike) -> bytes:
        """Decrypt RSA-OAEP ciphertext."""
        if self._private_key is None:
            raise RuntimeError("No private key available for decryption")
        return self._private_key.decrypt(bytes(ciphertext), self._oaep_padding())

    # ------------------------------------------------------------------
    # Signing / Verification
    # ------------------------------------------------------------------

    def sign(self, data: BytesLike) -> bytes:
        """Sign ``data`` with RSASSA-PSS-SHA256."""
        if self._private_key is None:
            raise RuntimeError("No private key available for signing")
        return self._private_key.sign(bytes(data), self._pss_padding(), hashes.SHA256())

    def verify(self, signature: BytesLike, data: BytesLike) -> bool:
        """Verify a PSS signature. Returns ``True`` if valid."""
        try:
            self._public_key.verify(
                bytes(signature),
                bytes(data),
                self._pss_padding(),
                hashes.SHA256(),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("RSA signature verification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_private_key(self) -> bool:
        return self._private_key is not None

    @property
    def key_size(self) -> int:
        return self._public_key.key_size


__all__ = ["RSACipher"]
