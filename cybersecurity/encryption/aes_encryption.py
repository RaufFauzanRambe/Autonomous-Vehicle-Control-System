"""AES symmetric encryption primitives.

Provides two flavours of AES:

    * :class:`AESCipher` — AES-256-GCM (authenticated encryption) with a
      random 96-bit IV and a 128-bit authentication tag.  This is the
      recommended primitive for all new code per NIST SP 800-38D.
    * :class:`AESCBC` — AES-256-CBC with PKCS#7 padding and HMAC-SHA256
      (encrypt-then-MAC), provided for backwards compatibility with
      legacy modules only.  Must not be used for new designs.

Both classes serialise ciphertexts as ``iv || ciphertext || tag`` (or
``iv || ciphertext || mac`` for CBC) so callers can treat the output as
a single opaque blob.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Optional, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .constants import (
    AES_BLOCK_SIZE,
    DEFAULT_AES_CBC_IV_SIZE,
    DEFAULT_AES_GCM_IV_SIZE,
    DEFAULT_AES_GCM_TAG_SIZE,
    DEFAULT_AES_KEY_SIZE,
)
from .utils import constant_time_compare, generate_nonce, safe_random

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


# ---------------------------------------------------------------------------
# AES-GCM
# ---------------------------------------------------------------------------

class AESCipher:
    """AES-256-GCM authenticated cipher.

    The same instance can be reused for many operations; the per-message
    IV is generated randomly inside :meth:`encrypt` and prepended to the
    returned ciphertext.
    """

    def __init__(self, key: Optional[BytesLike] = None, key_size: int = DEFAULT_AES_KEY_SIZE):
        if key is None:
            self._key = self.generate_key(key_size)
        else:
            self._key = bytes(key)
            if len(self._key) not in (16, 24, 32):
                raise ValueError(
                    f"AES key must be 16/24/32 bytes; got {len(self._key)}"
                )
        self._aesgcm = AESGCM(self._key)
        logger.debug("AESCipher initialised with %d-byte key", len(self._key))

    # -- key management ----------------------------------------------------

    @staticmethod
    def generate_key(key_size: int = DEFAULT_AES_KEY_SIZE) -> bytes:
        """Generate a random AES key of ``key_size`` bytes."""
        if key_size not in (16, 24, 32):
            raise ValueError("AES key size must be 16, 24 or 32 bytes")
        return safe_random(key_size)

    @property
    def key(self) -> bytes:
        return self._key

    # -- core crypto -------------------------------------------------------

    def encrypt(
        self,
        plaintext: BytesLike,
        associated_data: Optional[BytesLike] = None,
        iv: Optional[BytesLike] = None,
    ) -> bytes:
        """Encrypt ``plaintext`` with AES-GCM.

        Args:
            plaintext: Data to encrypt.
            associated_data: Optional AEAD associated data (authenticated
                but not encrypted, e.g. protocol headers).
            iv: Optional explicit IV (must be 12 bytes).  If ``None``, a
                fresh random IV is generated.  Reusing an IV with the
                same key catastrophically breaks GCM security.

        Returns:
            ``iv || ciphertext || tag`` as ``bytes``.
        """
        pt = bytes(plaintext)
        nonce = bytes(iv) if iv is not None else safe_random(DEFAULT_AES_GCM_IV_SIZE)
        if len(nonce) != DEFAULT_AES_GCM_IV_SIZE:
            raise ValueError(
                f"IV must be {DEFAULT_AES_GCM_IV_SIZE} bytes for GCM; got {len(nonce)}"
            )
        ad = bytes(associated_data) if associated_data is not None else None
        ct_and_tag = self._aesgcm.encrypt(nonce, pt, ad)
        # cryptography returns ciphertext || tag (tag is last 16 bytes)
        return nonce + ct_and_tag

    def decrypt(
        self,
        blob: BytesLike,
        associated_data: Optional[BytesLike] = None,
    ) -> bytes:
        """Decrypt a blob produced by :meth:`encrypt`.

        Raises:
            cryptography.exceptions.InvalidTag: If the tag does not
                verify or the data has been tampered with.
        """
        raw = bytes(blob)
        if len(raw) < DEFAULT_AES_GCM_IV_SIZE + DEFAULT_AES_GCM_TAG_SIZE:
            raise ValueError("Ciphertext too short to contain IV and tag")
        nonce = raw[:DEFAULT_AES_GCM_IV_SIZE]
        ct_and_tag = raw[DEFAULT_AES_GCM_IV_SIZE:]
        ad = bytes(associated_data) if associated_data is not None else None
        return self._aesgcm.decrypt(nonce, ct_and_tag, ad)

    # -- file helpers ------------------------------------------------------

    def encrypt_file(self, src: Union[str, os.PathLike], dst: Union[str, os.PathLike]) -> int:
        """Encrypt the file at ``src`` and write the result to ``dst``.

        Returns the number of plaintext bytes processed.
        """
        src_path, dst_path = Path(src), Path(dst)
        plaintext = src_path.read_bytes()
        blob = self.encrypt(plaintext)
        dst_path.write_bytes(blob)
        logger.info("Encrypted %s -> %s (%d -> %d bytes)",
                    src_path, dst_path, len(plaintext), len(blob))
        return len(plaintext)

    def decrypt_file(self, src: Union[str, os.PathLike], dst: Union[str, os.PathLike]) -> int:
        """Decrypt the file at ``src`` and write the result to ``dst``."""
        src_path, dst_path = Path(src), Path(dst)
        blob = src_path.read_bytes()
        plaintext = self.decrypt(blob)
        dst_path.write_bytes(plaintext)
        logger.info("Decrypted %s -> %s (%d -> %d bytes)",
                    src_path, dst_path, len(blob), len(plaintext))
        return len(plaintext)


# ---------------------------------------------------------------------------
# AES-CBC (legacy)
# ---------------------------------------------------------------------------

class AESCBC:
    """AES-256-CBC with PKCS#7 padding and HMAC-SHA256 (encrypt-then-MAC).

    Included for compatibility with legacy systems only.  New code must
    use :class:`AESCipher` (AES-GCM).
    """

    def __init__(self, key: Optional[BytesLike] = None, mac_key: Optional[BytesLike] = None):
        if key is None:
            self._key = AESCipher.generate_key()
        else:
            self._key = bytes(key)
            if len(self._key) not in (16, 24, 32):
                raise ValueError("AES key must be 16/24/32 bytes")
        if mac_key is None:
            self._mac_key = safe_random(32)
        else:
            self._mac_key = bytes(mac_key)
            if len(self._mac_key) < 32:
                raise ValueError("MAC key must be at least 32 bytes")

    @staticmethod
    def generate_key(key_size: int = DEFAULT_AES_KEY_SIZE) -> bytes:
        return AESCipher.generate_key(key_size)

    def _hmac(self, data: BytesLike) -> bytes:
        import hmac as _hmac
        import hashlib
        return _hmac.new(self._mac_key, bytes(data), hashlib.sha256).digest()

    def encrypt(self, plaintext: BytesLike) -> bytes:
        """Encrypt with AES-CBC + HMAC (encrypt-then-MAC)."""
        iv = safe_random(DEFAULT_AES_CBC_IV_SIZE)
        padder = padding.PKCS7(AES_BLOCK_SIZE * 8).padder()
        padded = padder.update(bytes(plaintext)) + padder.finalize()
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded) + encryptor.finalize()
        mac = self._hmac(iv + ct)
        return iv + ct + mac

    def decrypt(self, blob: BytesLike) -> bytes:
        """Verify HMAC and decrypt AES-CBC ciphertext."""
        raw = bytes(blob)
        if len(raw) < DEFAULT_AES_CBC_IV_SIZE + 32:
            raise ValueError("Ciphertext too short")
        iv = raw[:DEFAULT_AES_CBC_IV_SIZE]
        mac = raw[-32:]
        ct = raw[DEFAULT_AES_CBC_IV_SIZE:-32]
        expected = self._hmac(iv + ct)
        if not constant_time_compare(mac, expected):
            raise InvalidTag("HMAC verification failed")
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ct) + decryptor.finalize()
        unpadder = padding.PKCS7(AES_BLOCK_SIZE * 8).unpadder()
        return unpadder.update(padded) + unpadder.finalize()


__all__ = ["AESCipher", "AESCBC"]
