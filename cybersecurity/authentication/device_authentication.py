"""
device_authentication.py
========================

ECU / IoT device authentication via mutual TLS, X.509 device
certificates, and TPM 2.0 remote attestation quotes.

This module is *not* responsible for the actual TLS handshake — that is
the OS / mTLS library's job. Instead it owns the device registry:
issuing device certificates, validating attestation quotes, revoking
compromised devices, and exposing a single :meth:`authenticate_device`
entry point that combines all three signals.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDSA,
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.exceptions import InvalidSignature
from cryptography.x509.oid import NameOID

from .constants import AuthEvent, AuthMethod, AuthStatus

logger = logging.getLogger(__name__)


@dataclass
class DeviceRecord:
    """A registered ECU / IoT device."""

    device_id: str
    serial: str
    model: str
    public_key_pem: str
    certificate_pem: Optional[str] = None
    tpm_ek_pub_pem: Optional[str] = None  # TPM Endorsement Key
    tpm_ak_pub_pem: Optional[str] = None  # TPM Attestation Key
    enrolled_at: int = field(default_factory=lambda: int(time.time()))
    last_attested_at: int = 0
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TPMQuote:
    """Simplified TPM2_Quote result.

    Real TPM quotes carry a complex TPMS_ATTEST structure; here we capture
    the essential fields needed for verification: a nonce (preventing
    replay), the selected PCRs, and a signature over ``attest_data``.
    """

    pcr_values: Dict[int, str]  # PCR index -> sha256 hex digest
    attest_data: bytes  # signed TPMS_ATTEST structure
    signature: bytes
    signature_scheme: str = "ecdsa-sha256"


@dataclass
class DeviceAuthResult:
    status: AuthStatus
    device_id: Optional[str] = None
    method: Optional[AuthMethod] = None
    reason: str = ""


class DeviceAuthenticator:
    """ECU / IoT device authentication and attestation."""

    # Baseline PCR values that must match a known-good firmware state. The
    # keys are PCR indices; the values are SHA-256 hex digests. In a real
    # deployment this map is provisioned per-model from the firmware
    # signing manifest.
    DEFAULT_PCR_POLICY: Dict[int, Set[str]] = {
        0: set(),  # CRTM / firmware
        7: set(),  # Secure-boot policy
    }

    def __init__(
        self,
        ca_private_key=None,  # EllipticCurvePrivateKey | RSAPrivateKey
        ca_subject: str = "AVCS Device CA",
        pcr_policy: Optional[Dict[int, Set[str]]] = None,
    ) -> None:
        self.ca_private_key = ca_private_key
        self.ca_subject = ca_subject
        self.pcr_policy = pcr_policy or {k: set(v) for k, v in self.DEFAULT_PCR_POLICY.items()}
        self._lock = threading.RLock()
        self._devices: Dict[str, DeviceRecord] = {}
        self._by_serial: Dict[str, str] = {}
        self._revoked_serials: Set[str] = set()
        self._challenge_nonces: Dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_device(
        self,
        serial: str,
        model: str,
        public_key_pem: str,
        *,
        tpm_ek_pub_pem: Optional[str] = None,
        tpm_ak_pub_pem: Optional[str] = None,
        device_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        issue_certificate: bool = True,
    ) -> DeviceRecord:
        """Register a new device and optionally sign an X.509 cert for it."""
        serial = serial.strip()
        if not serial:
            raise ValueError("serial must be non-empty")
        with self._lock:
            if serial in self._by_serial:
                raise ValueError(f"device with serial {serial!r} already registered")
            did = device_id or f"dev_{secrets.token_hex(8)}"
            cert_pem: Optional[str] = None
            if issue_certificate:
                if self.ca_private_key is None:
                    raise RuntimeError("CA private key required to issue certificates")
                public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
                cert_pem = self._issue_device_certificate(did, public_key)
            record = DeviceRecord(
                device_id=did,
                serial=serial,
                model=model,
                public_key_pem=public_key_pem,
                certificate_pem=cert_pem,
                tpm_ek_pub_pem=tpm_ek_pub_pem,
                tpm_ak_pub_pem=tpm_ak_pub_pem,
                metadata=dict(metadata or {}),
            )
            self._devices[did] = record
            self._by_serial[serial] = did
        logger.info("Registered device %s (serial=%s, model=%s)", did, serial, model)
        return record

    def revoke_device(self, device_id: str) -> bool:
        with self._lock:
            record = self._devices.get(device_id)
            if record is None:
                return False
            record.revoked = True
            self._revoked_serials.add(record.serial)
        logger.warning("Revoked device %s (serial=%s)", device_id, record.serial)
        return True

    def get_device(self, device_id: str) -> Optional[DeviceRecord]:
        with self._lock:
            return self._devices.get(device_id)

    def get_device_by_serial(self, serial: str) -> Optional[DeviceRecord]:
        with self._lock:
            did = self._by_serial.get(serial)
            return self._devices.get(did) if did else None

    def list_devices(self) -> Iterable[DeviceRecord]:
        with self._lock:
            return list(self._devices.values())

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def issue_challenge(self, device_id: str) -> Optional[bytes]:
        """Issue a fresh nonce the device must echo back in its quote."""
        nonce = secrets.token_bytes(32)
        with self._lock:
            if device_id not in self._devices:
                return None
            self._challenge_nonces[device_id] = nonce
        return nonce

    def attest_device(
        self,
        device_id: str,
        quote: TPMQuote,
    ) -> DeviceAuthResult:
        """Verify a TPM2_Quote from ``device_id``.

        Verifies:
          1. The AK signature over ``attest_data``.
          2. The nonce matches the one we issued.
          3. The PCR values match policy.
        """
        with self._lock:
            record = self._devices.get(device_id)
            nonce = self._challenge_nonces.pop(device_id, None)
        if record is None:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                method=AuthMethod.TPM_ATTESTATION,
                reason="unknown device",
            )
        if record.revoked:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason="device revoked",
            )
        if nonce is None:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason="no challenge nonce pending",
            )

        # Verify AK signature over the attest data
        if not record.tpm_ak_pub_pem:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason="no AK enrolled",
            )
        try:
            ak_pub = serialization.load_pem_public_key(record.tpm_ak_pub_pem.encode("utf-8"))
        except Exception as exc:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason=f"bad AK: {exc}",
            )
        if not self._verify_attest_signature(ak_pub, quote.attest_data, quote.signature):
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason="attest signature invalid",
            )
        # Nonce must be present in the attest data (we model this as a
        # suffix of the attest bytes — real implementations parse the
        # TPMS_ATTEST structure).
        if nonce not in quote.attest_data and hashlib.sha256(nonce).digest() not in quote.attest_data:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason="nonce mismatch (possible replay)",
            )
        # PCR policy check
        for pcr_idx, allowed in self.pcr_policy.items():
            actual = quote.pcr_values.get(pcr_idx)
            if actual is None:
                return DeviceAuthResult(
                    status=AuthStatus.FAILURE,
                    device_id=device_id,
                    method=AuthMethod.TPM_ATTESTATION,
                    reason=f"PCR{pcr_idx} missing",
                )
            if allowed and actual not in allowed:
                return DeviceAuthResult(
                    status=AuthStatus.FAILURE,
                    device_id=device_id,
                    method=AuthMethod.TPM_ATTESTATION,
                    reason=f"PCR{pcr_idx} value not in policy",
                )
        with self._lock:
            record.last_attested_at = int(time.time())
        return DeviceAuthResult(
            status=AuthStatus.SUCCESS,
            device_id=device_id,
            method=AuthMethod.TPM_ATTESTATION,
            reason="ok",
        )

    def authenticate_device(
        self,
        device_id: str,
        *,
        client_certificate_pem: Optional[str] = None,
        quote: Optional[TPMQuote] = None,
    ) -> DeviceAuthResult:
        """Top-level device authentication combining mTLS + attestation.

        mTLS alone yields a *provisional* SUCCESS — attestation is required
        to upgrade to full trust (i.e. to access safety-critical ECUs).
        """
        with self._lock:
            record = self._devices.get(device_id)
        if record is None:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                method=AuthMethod.MUTUAL_TLS,
                reason="unknown device",
            )
        if record.revoked or record.serial in self._revoked_serials:
            return DeviceAuthResult(
                status=AuthStatus.REVOKED,
                device_id=device_id,
                method=AuthMethod.MUTUAL_TLS,
                reason="device revoked",
            )
        if client_certificate_pem:
            if not self._verify_client_certificate(client_certificate_pem, record):
                return DeviceAuthResult(
                    status=AuthStatus.FAILURE,
                    device_id=device_id,
                    method=AuthMethod.MUTUAL_TLS,
                    reason="client cert invalid",
                )
        else:
            return DeviceAuthResult(
                status=AuthStatus.FAILURE,
                device_id=device_id,
                method=AuthMethod.MUTUAL_TLS,
                reason="client certificate required",
            )
        if quote is not None:
            attestation = self.attest_device(device_id, quote)
            if attestation.status != AuthStatus.SUCCESS:
                return attestation
            return DeviceAuthResult(
                status=AuthStatus.SUCCESS,
                device_id=device_id,
                method=AuthMethod.TPM_ATTESTATION,
                reason="mTLS + attestation OK",
            )
        return DeviceAuthResult(
            status=AuthStatus.SUCCESS,
            device_id=device_id,
            method=AuthMethod.MUTUAL_TLS,
            reason="mTLS OK (attestation pending)",
        )

    # ------------------------------------------------------------------
    # Internal crypto helpers
    # ------------------------------------------------------------------
    def _issue_device_certificate(self, device_id: str, public_key) -> str:
        if self.ca_private_key is None:
            raise RuntimeError("CA key not configured")
        builder = (
            x509.CertificateBuilder()
            .subject_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, device_id)])
            )
            .issuer_name(
                x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, self.ca_subject)])
            )
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(__import__("datetime").datetime.utcnow())
            .not_valid_after(
                __import__("datetime").datetime.utcnow()
                + __import__("datetime").timedelta(days=365)
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True
            )
        )
        cert = builder.sign(self.ca_private_key, hashes.SHA256())
        return cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    def _verify_client_certificate(
        self,
        cert_pem: str,
        record: DeviceRecord,
    ) -> bool:
        try:
            cert = x509.load_pem_x509_certificate(cert_pem.encode("utf-8"))
        except Exception:
            return False
        # Verify the cert's public key matches the registered device key
        if cert.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8") != record.public_key_pem:
            return False
        # In production we would chain to the CA here. We trust the
        # registry's recorded CA in lieu of full path validation.
        now = __import__("datetime").datetime.utcnow()
        try:
            if now < cert.not_valid_before or now >= cert.not_valid_after:
                return False
        except Exception:
            return False
        return True

    def _verify_attest_signature(self, ak_pub, attest_data: bytes, signature: bytes) -> bool:
        try:
            if isinstance(ak_pub, EllipticCurvePublicKey):
                ak_pub.verify(signature, attest_data, ECDSA(hashes.SHA256()))
                return True
            if isinstance(ak_pub, RSAPublicKey):
                ak_pub.verify(signature, attest_data, padding.PKCS1v15(), hashes.SHA256())
                return True
        except InvalidSignature:
            return False
        return False


__all__ = ["DeviceRecord", "TPMQuote", "DeviceAuthResult", "DeviceAuthenticator"]
