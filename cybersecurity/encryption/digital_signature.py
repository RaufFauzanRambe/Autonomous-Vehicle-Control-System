"""Digital signature abstraction over RSA-PSS, ECDSA and Ed25519.

The :class:`DigitalSignature` class provides a uniform ``sign`` /
``verify`` interface regardless of the underlying algorithm.  Streaming
variants (:meth:`sign_stream`, :meth:`verify_stream`) hash the input
incrementally before signing, so that arbitrarily large payloads (e.g.
firmware images, sensor logs) can be signed without loading them fully
into memory.
"""

from __future__ import annotations

import enum
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.hazmat.primitives.asymmetric.padding import MGF1, PSS

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


class SignatureAlgorithm(str, enum.Enum):
    """Supported signature algorithms."""

    RSA_PSS_SHA256 = "RSA-PSS-SHA256"
    RSA_PSS_SHA384 = "RSA-PSS-SHA384"
    ECDSA_P256 = "ECDSA-P256-SHA256"
    ECDSA_P384 = "ECDSA-P384-SHA384"
    ECDSA_P521 = "ECDSA-P521-SHA512"
    ED25519 = "Ed25519"
    ED448 = "Ed448"


_HASH_MAP = {
    SignatureAlgorithm.RSA_PSS_SHA256: hashes.SHA256(),
    SignatureAlgorithm.RSA_PSS_SHA384: hashes.SHA384(),
    SignatureAlgorithm.ECDSA_P256: hashes.SHA256(),
    SignatureAlgorithm.ECDSA_P384: hashes.SHA384(),
    SignatureAlgorithm.ECDSA_P521: hashes.SHA512(),
}

_STREAM_HASH = {
    SignatureAlgorithm.RSA_PSS_SHA256: "sha256",
    SignatureAlgorithm.RSA_PSS_SHA384: "sha384",
    SignatureAlgorithm.ECDSA_P256: "sha256",
    SignatureAlgorithm.ECDSA_P384: "sha384",
    SignatureAlgorithm.ECDSA_P521: "sha512",
    SignatureAlgorithm.ED25519: "sha512",  # Ed25519 signs the raw message; we pre-hash for streaming
}


class _BackendSigner(ABC):
    """Strategy interface for the concrete signing backends."""

    @abstractmethod
    def sign(self, data: bytes) -> bytes: ...

    @abstractmethod
    def verify(self, signature: bytes, data: bytes) -> bool: ...


class _RSAPSSBackend(_BackendSigner):
    def __init__(self, private_key: Optional[rsa.RSAPrivateKey], public_key: rsa.RSAPublicKey, hash_algo: hashes.HashAlgorithm):
        self._priv = private_key
        self._pub = public_key
        self._hash = hash_algo

    def _padding(self) -> PSS:
        return PSS(mgf=MGF1(self._hash), salt_length=PSS.MAX_LENGTH)

    def sign(self, data: bytes) -> bytes:
        if self._priv is None:
            raise RuntimeError("No private key available")
        return self._priv.sign(data, self._padding(), self._hash)

    def verify(self, signature: bytes, data: bytes) -> bool:
        try:
            self._pub.verify(signature, data, self._padding(), self._hash)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("RSA-PSS verification failed: %s", exc)
            return False


class _ECDSABackend(_BackendSigner):
    def __init__(self, private_key: Optional[ec.EllipticCurvePrivateKey], public_key: ec.EllipticCurvePublicKey, hash_algo: hashes.HashAlgorithm):
        self._priv = private_key
        self._pub = public_key
        self._hash = hash_algo

    def sign(self, data: bytes) -> bytes:
        if self._priv is None:
            raise RuntimeError("No private key available")
        return self._priv.sign(data, ec.ECDSA(self._hash))

    def verify(self, signature: bytes, data: bytes) -> bool:
        try:
            self._pub.verify(signature, data, ec.ECDSA(self._hash))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("ECDSA verification failed: %s", exc)
            return False


class _Ed25519Backend(_BackendSigner):
    def __init__(self, private_key: Optional[ed25519.Ed25519PrivateKey], public_key: ed25519.Ed25519PublicKey):
        self._priv = private_key
        self._pub = public_key

    def sign(self, data: bytes) -> bytes:
        if self._priv is None:
            raise RuntimeError("No private key available")
        return self._priv.sign(data)

    def verify(self, signature: bytes, data: bytes) -> bool:
        try:
            self._pub.verify(signature, data)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ed25519 verification failed: %s", exc)
            return False


class DigitalSignature:
    """High-level, algorithm-agnostic signing facade."""

    def __init__(
        self,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519,
        private_key: Optional[object] = None,
        public_key: Optional[object] = None,
    ):
        self.algorithm = algorithm
        self._backend = self._select_backend(algorithm, private_key, public_key)
        logger.debug("DigitalSignature initialised with %s", algorithm.value)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend(
        algorithm: SignatureAlgorithm,
        private_key: Optional[object],
        public_key: Optional[object],
    ) -> _BackendSigner:
        if algorithm in (SignatureAlgorithm.RSA_PSS_SHA256, SignatureAlgorithm.RSA_PSS_SHA384):
            if private_key is None and public_key is None:
                private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
            pub = public_key or (private_key.public_key() if private_key else None)
            if pub is None:
                raise ValueError("Either private_key or public_key required for RSA-PSS")
            return _RSAPSSBackend(private_key, pub, _HASH_MAP[algorithm])
        if algorithm in (SignatureAlgorithm.ECDSA_P256, SignatureAlgorithm.ECDSA_P384, SignatureAlgorithm.ECDSA_P521):
            curve = {
                SignatureAlgorithm.ECDSA_P256: ec.SECP256R1(),
                SignatureAlgorithm.ECDSA_P384: ec.SECP384R1(),
                SignatureAlgorithm.ECDSA_P521: ec.SECP521R1(),
            }[algorithm]
            if private_key is None and public_key is None:
                private_key = ec.generate_private_key(curve)
            pub = public_key or (private_key.public_key() if private_key else None)
            if pub is None:
                raise ValueError("Either private_key or public_key required for ECDSA")
            return _ECDSABackend(private_key, pub, _HASH_MAP[algorithm])
        if algorithm == SignatureAlgorithm.ED25519:
            if private_key is None and public_key is None:
                private_key = ed25519.Ed25519PrivateKey.generate()
            pub = public_key or (private_key.public_key() if private_key else None)
            if pub is None:
                raise ValueError("Either private_key or public_key required for Ed25519")
            return _Ed25519Backend(private_key, pub)
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # ------------------------------------------------------------------
    # Sign / Verify (in-memory)
    # ------------------------------------------------------------------

    def sign(self, data: BytesLike) -> bytes:
        return self._backend.sign(bytes(data))

    def verify(self, signature: BytesLike, data: BytesLike) -> bool:
        return self._backend.verify(bytes(signature), bytes(data))

    # ------------------------------------------------------------------
    # Streaming variants
    # ------------------------------------------------------------------

    def sign_stream(self, stream: Iterable[BytesLike], chunk_hint: int = 65_536) -> bytes:
        """Hash a stream incrementally and sign the resulting digest.

        For Ed25519 (which is a "one-shot" signer) we hash with SHA-512
        first and then sign the digest.  For RSA-PSS / ECDSA we sign the
        digest with the algorithm's hash to keep signatures reasonably
        compact.

        Returns the signature as ``bytes``.
        """
        digest = self._hash_stream(stream)
        # Sign the digest instead of the raw message
        return self._backend.sign(digest)

    def verify_stream(self, signature: BytesLike, stream: Iterable[BytesLike]) -> bool:
        digest = self._hash_stream(stream)
        return self._backend.verify(bytes(signature), digest)

    def _hash_stream(self, stream: Iterable[BytesLike]) -> bytes:
        name = _STREAM_HASH[self.algorithm]
        hasher = hashlib.new(name)
        total = 0
        for chunk in stream:
            chunk_bytes = bytes(chunk)
            hasher.update(chunk_bytes)
            total += len(chunk_bytes)
        logger.debug("Streamed %d bytes through %s for signing", total, name)
        return hasher.digest()


__all__ = ["DigitalSignature", "SignatureAlgorithm"]
