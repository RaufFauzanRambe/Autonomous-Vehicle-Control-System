"""Secure (measured) boot manager.

Implements the measured-boot flow used on the vehicle compute platform:

  1. Verify the ROM manifest (list of expected measurements).
  2. Verify the bootloader signature with an RSA-3072 key.
  3. Verify the kernel signature.
  4. Verify the initramfs signature.
  5. Extend each measurement into a TPM 2.0 PCR register.
  6. Attest the final PCR state to a remote verifier.

Actual signature verification uses :mod:`cryptography` when a real key is
available; otherwise the manager can run against a software mock that simply
hashes inputs and extends a software PCR bank. The software-mock path makes
the manager fully testable on CI machines without a TPM.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    PCR_APP,
    PCR_BOOTLOADER,
    PCR_INITRAMFS,
    PCR_KERNEL,
    PCR_NUMBER,
    PCR_ROM,
    Severity,
)
from .utils import compute_sha256, hex_encode, safe_compare
from .security_event_logger import SecurityEventLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


class BootStage(str):
    ROM = "rom"
    BOOTLOADER = "bootloader"
    KERNEL = "kernel"
    INITRAMFS = "initramfs"
    APP = "app"


@dataclass
class BootStageResult:
    stage: str
    success: bool
    expected_measurement: str
    actual_measurement: str
    signature_valid: bool
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "expected_measurement": self.expected_measurement,
            "actual_measurement": self.actual_measurement,
            "signature_valid": self.signature_valid,
            "error": self.error,
        }


@dataclass
class ManifestEntry:
    stage: str
    pcr: int
    expected_measurement: str
    signature_path: str
    public_key_path: str


@dataclass
class AttestationReport:
    pcr_values: Dict[int, str]
    stage_results: List[BootStageResult]
    overall_success: bool
    quote: str  # TPM quote blob (hex)
    nonce: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pcr_values": self.pcr_values,
            "stage_results": [r.to_dict() for r in self.stage_results],
            "overall_success": self.overall_success,
            "quote": self.quote,
            "nonce": self.nonce,
        }


# --------------------------------------------------------------------------- #
# Software TPM mock (real one would use tpm2-pytss)
# --------------------------------------------------------------------------- #


class SoftwareTPM:
    """A deterministic software emulation of a TPM 2.0 PCR bank.

    PCR extension follows the standard ``PCRnew = H(PCRold || H(data))``
    formula used by SHA-256 banks on real TPM hardware.
    """

    def __init__(self, pcr_count: int = PCR_NUMBER) -> None:
        self._pcrs: List[bytes] = [b"\x00" * 32 for _ in range(pcr_count)]
        self._event_log: List[Dict[str, Any]] = []
        self._lock_counter = 0

    def extend(self, pcr: int, data: bytes) -> bytes:
        if not 0 <= pcr < len(self._pcrs):
            raise IndexError(f"PCR {pcr} out of range")
        current = self._pcrs[pcr]
        digest = compute_sha256(data)
        new_value = compute_sha256(current + digest)
        self._pcrs[pcr] = new_value
        self._event_log.append({"pcr": pcr, "digest": hex_encode(digest), "new": hex_encode(new_value)})
        return new_value

    def read(self, pcr: int) -> bytes:
        return self._pcrs[pcr]

    def read_all(self) -> Dict[int, str]:
        return {i: hex_encode(v) for i, v in enumerate(self._pcrs)}

    def reset(self, pcr: int) -> None:
        # Only PCR 16-23 are resettable on real hardware; allow it in the mock.
        self._pcrs[pcr] = b"\x00" * 32

    def quote(self, pcrs: List[int], nonce: bytes) -> str:
        # Real TPM quote produces a signed TPMS_ATTEST; here we just hash the
        # selected PCR values + nonce to produce a deterministic token.
        h = hashlib.sha256()
        h.update(nonce)
        for pcr in pcrs:
            h.update(self._pcrs[pcr])
        return hex_encode(h.digest())


# --------------------------------------------------------------------------- #
# Signature verifier abstraction
# --------------------------------------------------------------------------- #


class SignatureVerifier:
    """RSA-PSS / ECDSA verifier.

    The verifier gracefully degrades to a "software measurement only" mode
    if the ``cryptography`` library or the key file is missing. In that mode
    the manager still produces PCR extensions and an attestation report, but
    the ``signature_valid`` flag on each stage result will be ``False``.
    """

    def __init__(self) -> None:
        try:
            from cryptography.hazmat.primitives import hashes  # noqa: F401
            from cryptography.hazmat.primitives.asymmetric import padding, rsa, ec  # noqa: F401
            from cryptography.hazmat.primitives.serialization import load_pem_public_key  # noqa: F401
            self._available = True
        except ImportError:  # pragma: no cover
            self._available = False
            logger.warning("cryptography library not available; signature checks disabled")

    @property
    def available(self) -> bool:
        return self._available

    def verify(self, pubkey_path: str, signature: bytes, data: bytes, alg: str = "rsa-pss") -> bool:
        if not self._available:
            return False
        if not Path(pubkey_path).exists():
            logger.debug("pubkey %s missing", pubkey_path)
            return False
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding, ec
            from cryptography.hazmat.primitives.serialization import load_pem_public_key

            with open(pubkey_path, "rb") as fh:
                pub = load_pem_public_key(fh.read())
            if alg == "rsa-pss":
                pub.verify(  # type: ignore[union-attr]
                    signature, data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256()
                )
            elif alg == "ecdsa-p384":
                pub.verify(signature, data, ec.ECDSA(hashes.SHA384()))  # type: ignore[union-attr]
            else:
                raise ValueError(f"unsupported alg {alg}")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("signature verify failed: %s", exc)
            return False


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class SecureBootManager:
    """Coordinates the measured-boot flow."""

    STAGE_TO_PCR = {
        BootStage.ROM: PCR_ROM,
        BootStage.BOOTLOADER: PCR_BOOTLOADER,
        BootStage.KERNEL: PCR_KERNEL,
        BootStage.INITRAMFS: PCR_INITRAMFS,
        BootStage.APP: PCR_APP,
    }

    def __init__(
        self,
        manifest_path: str,
        tpm: Optional[SoftwareTPM] = None,
        verifier: Optional[SignatureVerifier] = None,
        logger_: Optional[SecurityEventLogger] = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.tpm = tpm or SoftwareTPM()
        self.verifier = verifier or SignatureVerifier()
        self.event_logger = logger_
        self._manifest: Dict[str, ManifestEntry] = {}
        self._stage_results: List[BootStageResult] = []
        self._load_manifest()

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> None:
        path = Path(self.manifest_path)
        if not path.exists():
            logger.warning("manifest %s missing; running in permissive mode", self.manifest_path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("stages", []):
            self._manifest[entry["stage"]] = ManifestEntry(
                stage=entry["stage"],
                pcr=int(entry["pcr"]),
                expected_measurement=entry["expected_measurement"],
                signature_path=entry.get("signature_path", ""),
                public_key_path=entry.get("public_key_path", ""),
            )

    # ------------------------------------------------------------------ #
    # Stage verification
    # ------------------------------------------------------------------ #

    def verify_boot_stage(self, stage: str, image: bytes, signature: Optional[bytes] = None) -> BootStageResult:
        """Verify one boot stage: check measurement, signature, and extend PCR."""
        entry = self._manifest.get(stage)
        pcr = self.STAGE_TO_PCR.get(stage, PCR_APP)
        actual = hex_encode(compute_sha256(image))
        expected = entry.expected_measurement if entry else actual
        measurement_ok = safe_compare(bytes.fromhex(actual), bytes.fromhex(expected)) if entry else True

        sig_ok = False
        if signature is not None and entry is not None and entry.public_key_path:
            alg = "ecdsa-p384" if stage == BootStage.APP else "rsa-pss"
            sig_ok = self.verifier.verify(entry.public_key_path, signature, image, alg)

        success = measurement_ok and (signature is None or sig_ok)
        result = BootStageResult(
            stage=stage,
            success=success,
            expected_measurement=expected,
            actual_measurement=actual,
            signature_valid=sig_ok,
            error="" if success else "measurement mismatch" if not measurement_ok else "signature invalid",
        )
        self._stage_results.append(result)

        # Extend PCR with the actual measurement regardless of pass/fail
        self.extend_pcr(pcr, image)

        if self.event_logger:
            from .constants import EventType
            ev_type = EventType.BOOT_STAGE_OK if success else EventType.BOOT_ATTESTATION_FAIL
            sev = Severity.INFO if success else Severity.CRITICAL
            self.event_logger.log_event(
                event_type=ev_type.value,
                severity=sev.value,
                source=f"secure_boot.{stage}",
                message=f"boot stage {stage} {'passed' if success else 'failed'}",
                details=result.to_dict(),
            )
        logger.info("boot stage %s -> success=%s", stage, success)
        return result

    def extend_pcr(self, pcr: int, data: bytes) -> bytes:
        """Extend a TPM PCR with the SHA-256 digest of *data*."""
        return self.tpm.extend(pcr, data)

    # ------------------------------------------------------------------ #
    # Attestation
    # ------------------------------------------------------------------ #

    def attest_state(self, nonce: Optional[bytes] = None) -> AttestationReport:
        """Produce an attestation report containing all PCR values + quote."""
        if nonce is None:
            nonce = compute_sha256(b"avcs-attest-nonce-v1")
        pcr_values = self.tpm.read_all()
        quote = self.tpm.quote(list(range(PCR_NUMBER)), nonce)
        overall = all(r.success for r in self._stage_results)
        report = AttestationReport(
            pcr_values=pcr_values,
            stage_results=list(self._stage_results),
            overall_success=overall,
            quote=quote,
            nonce=hex_encode(nonce),
        )
        logger.info("attestation produced (overall_success=%s)", overall)
        return report

    def get_pcr_values(self) -> Dict[int, str]:
        return self.tpm.read_all()

    def get_stage_results(self) -> List[BootStageResult]:
        return list(self._stage_results)

    def reset(self) -> None:
        """Reset all PCRs and stage results (used in tests and re-flash)."""
        for i in range(PCR_NUMBER):
            self.tpm.reset(i)
        self._stage_results.clear()
