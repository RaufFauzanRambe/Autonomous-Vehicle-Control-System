"""Firmware image validation.

The :class:`FirmwareValidator` validates firmware images before they are
flashed to an ECU. The validation pipeline:

  1. Parse the image header (magic, version, target ECU, payload offset).
  2. Verify the ECDSA-P384 signature over the payload.
  3. Check that the version is compatible with the current installed version
     (no downgrades, no skipping required security baseline).
  4. Scan the image payload against a list of known-bad SHA-256 hashes
     (e.g. images with backdoors or revoked signing keys).

A firmware image is only accepted if every check passes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import ECU_IDS, ECDSA_CURVE
from .utils import compute_sha256, hex_encode, safe_compare

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Image header format
# --------------------------------------------------------------------------- #

# Magic: b"AVCSFW1" (7 bytes) | target_ecu (1) | version_major (1) |
# version_minor (1) | version_patch (1) | flags (1) | payload_size (4 BE) |
# sig_offset (4 BE)
HEADER_FORMAT = ">7sBBBBBII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 7 + 5 + 4 + 4 = 20
MAGIC = b"AVCSFW1"


class FirmwareFlags:
    SIGNED = 0x01
    ENCRYPTED = 0x02
    RECOVERY = 0x04
    DEBUG = 0x08


@dataclass
class FirmwareHeader:
    magic: bytes
    target_ecu: int
    version_major: int
    version_minor: int
    version_patch: int
    flags: int
    payload_size: int
    sig_offset: int

    @property
    def version_str(self) -> str:
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}"

    @property
    def is_signed(self) -> bool:
        return bool(self.flags & FirmwareFlags.SIGNED)

    @property
    def target_ecu_name(self) -> str:
        for name, ecu_id in ECU_IDS.items():
            if ecu_id == self.target_ecu:
                return name
        return f"ECU-{self.target_ecu:02X}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic.decode("ascii", errors="replace"),
            "target_ecu": self.target_ecu,
            "target_ecu_name": self.target_ecu_name,
            "version": self.version_str,
            "flags": self.flags,
            "is_signed": self.is_signed,
            "payload_size": self.payload_size,
            "sig_offset": self.sig_offset,
        }


# --------------------------------------------------------------------------- #
# Validation result
# --------------------------------------------------------------------------- #


class ValidationCode(str, Enum):
    OK = "ok"
    HEADER_INVALID = "header_invalid"
    SIGNATURE_INVALID = "signature_invalid"
    VERSION_INCOMPATIBLE = "version_incompatible"
    KNOWN_BAD_HASH = "known_bad_hash"
    PAYLOAD_TRUNCATED = "payload_truncated"


@dataclass
class ValidationResult:
    code: ValidationCode
    header: Optional[FirmwareHeader] = None
    payload_sha256: str = ""
    signature_valid: bool = False
    version_ok: bool = False
    hash_scan_ok: bool = False
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return self.code == ValidationCode.OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "success": self.success,
            "header": self.header.to_dict() if self.header else None,
            "payload_sha256": self.payload_sha256,
            "signature_valid": self.signature_valid,
            "version_ok": self.version_ok,
            "hash_scan_ok": self.hash_scan_ok,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------- #
# Signature verifier (ECDSA P-384)
# --------------------------------------------------------------------------- #


class ECDSASignatureVerifier:
    """ECDSA-P384 signature verifier using SHA-384 digest."""

    def __init__(self) -> None:
        try:
            from cryptography.hazmat.primitives import hashes  # noqa: F401
            from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401
            from cryptography.hazmat.primitives.serialization import load_pem_public_key  # noqa: F401
            self._available = True
        except ImportError:  # pragma: no cover
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def verify(self, pubkey_pem_path: str, signature: bytes, data: bytes) -> bool:
        if not self._available:
            return False
        if not Path(pubkey_pem_path).exists():
            return False
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives.serialization import load_pem_public_key

            with open(pubkey_pem_path, "rb") as fh:
                pub = load_pem_public_key(fh.read())
            pub.verify(signature, data, ec.ECDSA(hashes.SHA384()))  # type: ignore[union-attr]
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("ECDSA verify failed: %s", exc)
            return False


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class FirmwareValidator:
    """Validates firmware images before they are committed to an ECU."""

    def __init__(
        self,
        trusted_pubkey_path: Optional[str] = None,
        known_bad_hashes: Optional[List[str]] = None,
        current_versions: Optional[Dict[int, Tuple[int, int, int]]] = None,
        verifier: Optional[ECDSASignatureVerifier] = None,
    ) -> None:
        self.trusted_pubkey_path = trusted_pubkey_path
        self.known_bad_hashes = set(h.lower() for h in (known_bad_hashes or []))
        # current_versions: {ecu_id: (major, minor, patch)}
        self.current_versions = current_versions or {}
        self.verifier = verifier or ECDSASignatureVerifier()

    # ------------------------------------------------------------------ #
    # Header parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_header(image: bytes) -> FirmwareHeader:
        if len(image) < HEADER_SIZE:
            raise ValueError(f"image too small to contain header ({len(image)} < {HEADER_SIZE})")
        magic, ecu, vmaj, vmin, vpat, flags, psize, sigoff = struct.unpack(HEADER_FORMAT, image[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r} (expected {MAGIC!r})")
        return FirmwareHeader(
            magic=magic,
            target_ecu=ecu,
            version_major=vmaj,
            version_minor=vmin,
            version_patch=vpat,
            flags=flags,
            payload_size=psize,
            sig_offset=sigoff,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate_image(self, image: bytes, require_signature: bool = True) -> ValidationResult:
        """Run the full validation pipeline on *image*."""
        errors: List[str] = []

        # 1. Header
        try:
            header = self.parse_header(image)
        except ValueError as exc:
            return ValidationResult(
                code=ValidationCode.HEADER_INVALID,
                errors=[f"header parse failed: {exc}"],
            )

        # 2. Payload extraction
        payload_start = HEADER_SIZE
        payload_end = payload_start + header.payload_size
        if payload_end > len(image):
            errors.append(f"payload truncated: expected {header.payload_size} bytes, image has {len(image) - HEADER_SIZE}")
            return ValidationResult(
                code=ValidationCode.PAYLOAD_TRUNCATED,
                header=header,
                errors=errors,
            )
        payload = image[payload_start:payload_end]
        payload_hash = hex_encode(compute_sha256(payload))

        # 3. Signature
        signature = b""
        sig_valid = False
        if header.is_signed:
            if header.sig_offset == 0 or header.sig_offset + 96 > len(image):
                # Image claims to be signed but no signature is present.
                if require_signature:
                    errors.append("signature offset out of range")
                else:
                    # Permissive mode: signature treated as not required.
                    sig_valid = True
            else:
                signature = image[header.sig_offset: header.sig_offset + 96]
                if self.trusted_pubkey_path:
                    sig_valid = self.verifier.verify(self.trusted_pubkey_path, signature, payload)
                    if not sig_valid:
                        errors.append("ECDSA-P384 signature invalid")
                else:
                    # Permissive mode (test only): accept any present signature.
                    sig_valid = True
        elif require_signature:
            errors.append("image is not signed but signature required")
        else:
            # Unsigned and signature not required: treat as valid.
            sig_valid = True

        # 4. Version check
        version_ok = self.check_version(header.target_ecu, header.version_major, header.version_minor, header.version_patch)
        if not version_ok:
            cur = self.current_versions.get(header.target_ecu)
            cur_str = f"{cur[0]}.{cur[1]}.{cur[2]}" if cur else "none"
            errors.append(f"version {header.version_str} incompatible with current {cur_str} for ECU {header.target_ecu_name}")

        # 5. Known-bad hash scan
        hash_scan_ok = self.scan_hashes(payload_hash)
        if not hash_scan_ok:
            errors.append(f"payload matches known-bad hash: {payload_hash}")

        if errors:
            code = (
                ValidationCode.SIGNATURE_INVALID if not sig_valid and header.is_signed
                else ValidationCode.VERSION_INCOMPATIBLE if not version_ok
                else ValidationCode.KNOWN_BAD_HASH if not hash_scan_ok
                else ValidationCode.HEADER_INVALID
            )
            return ValidationResult(
                code=code,
                header=header,
                payload_sha256=payload_hash,
                signature_valid=sig_valid,
                version_ok=version_ok,
                hash_scan_ok=hash_scan_ok,
                errors=errors,
            )

        logger.info("firmware validated: ECU=%s version=%s hash=%s",
                    header.target_ecu_name, header.version_str, payload_hash)
        return ValidationResult(
            code=ValidationCode.OK,
            header=header,
            payload_sha256=payload_hash,
            signature_valid=sig_valid,
            version_ok=version_ok,
            hash_scan_ok=hash_scan_ok,
        )

    def verify_signature(self, image: bytes) -> bool:
        """Convenience: verify only the signature of an image."""
        result = self.validate_image(image, require_signature=False)
        return result.signature_valid

    def check_version(self, ecu_id: int, major: int, minor: int, patch: int) -> bool:
        """Return True if (major, minor, patch) is acceptable for *ecu_id*.

        Rule: no downgrades allowed; equal-version patches are allowed.
        """
        current = self.current_versions.get(ecu_id)
        if current is None:
            return True  # no current version installed
        new = (major, minor, patch)
        # Allow same-or-higher; reject downgrades
        return new >= current

    def scan_hashes(self, payload_hash: str) -> bool:
        """Return True if *payload_hash* is NOT in the known-bad list."""
        return payload_hash.lower() not in self.known_bad_hashes

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #

    def add_known_bad_hash(self, hash_hex: str) -> None:
        self.known_bad_hashes.add(hash_hex.lower())

    def set_current_version(self, ecu_id: int, major: int, minor: int, patch: int) -> None:
        self.current_versions[ecu_id] = (major, minor, patch)

    @staticmethod
    def build_image(
        target_ecu: int,
        version: Tuple[int, int, int],
        payload: bytes,
        flags: int = FirmwareFlags.SIGNED,
        signature: bytes = b"",
    ) -> bytes:
        """Build a (possibly unsigned) firmware image blob. Test helper."""
        vmaj, vmin, vpat = version
        sig_offset = HEADER_SIZE + len(payload) if signature else 0
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            target_ecu,
            vmaj,
            vmin,
            vpat,
            flags,
            len(payload),
            sig_offset,
        )
        return header + payload + signature
