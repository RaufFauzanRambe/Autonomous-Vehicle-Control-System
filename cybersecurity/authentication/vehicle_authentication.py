"""
vehicle_authentication.py
=========================

V2V (vehicle-to-vehicle) and V2I (vehicle-to-infrastructure) mutual
authentication modelled on the IEEE 1609.2 SCMS (Security Credential
Management System) certificate framework.

SCMS issues each vehicle a chain of short-lived pseudonym certificates so
that beacons (BSMs — Basic Safety Messages) can be signed without
revealing the vehicle's long-term identity. This module implements a
verifier and a beacon broadcaster that use a simplified 1609.2-style
signed-payload format.

Real production deployments would interface with an HSM-backed signing
module; here we use the ``cryptography`` library so the module is self-
contained and testable.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

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

from .constants import AuthEvent, AuthMethod, AuthStatus

logger = logging.getLogger(__name__)


@dataclass
class VehicleCertificate:
    """A simplified SCMS-style vehicle pseudonym / identity certificate."""

    certificate_id: str
    vin: str
    public_key_pem: str
    issuer: str  # SCMS PCA / RA identifier
    not_before: int
    not_after: int
    serial: str
    revoked: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_expired(self, now: Optional[int] = None) -> bool:
        ts = now or int(time.time())
        return ts < self.not_before or ts >= self.not_after


@dataclass
class Beacon:
    """A signed BSM (Basic Safety Message) ready for broadcast."""

    sender_vin: str
    certificate_id: str
    payload: bytes
    signature: bytes
    timestamp: int


@dataclass
class VehicleAuthResult:
    status: AuthStatus
    remote_vin: Optional[str] = None
    certificate_id: Optional[str] = None
    method: Optional[AuthMethod] = None
    reason: str = ""


class VehicleAuthenticator:
    """V2V / V2I mutual authentication.

    Parameters
    ----------
    local_vin:
        VIN of the host vehicle.
    signing_key:
        ECDSA P-256 (or RSA) private key used to sign outgoing beacons.
    trust_chain:
        Set of issuer identifiers (PCA / RA) whose issued certificates we
        trust.
    """

    def __init__(
        self,
        local_vin: str,
        signing_key,  # EllipticCurvePrivateKey | RSAPrivateKey
        trust_chain: Set[str],
    ) -> None:
        self.local_vin = local_vin
        self.signing_key = signing_key
        self.trust_chain = set(trust_chain)
        self._lock = threading.RLock()
        self._known_certs: Dict[str, VehicleCertificate] = {}
        self._revoked_serials: Set[str] = set()
        self._last_seen: Dict[str, int] = {}  # vin -> last beacon timestamp

    # ------------------------------------------------------------------
    # Certificate store
    # ------------------------------------------------------------------
    def register_vehicle_certificate(self, cert: VehicleCertificate) -> None:
        with self._lock:
            self._known_certs[cert.certificate_id] = cert
        logger.info("Registered vehicle cert %s for VIN %s", cert.certificate_id, cert.vin)

    def revoke_certificate(self, certificate_id: str) -> bool:
        with self._lock:
            cert = self._known_certs.get(certificate_id)
            if cert is None:
                return False
            cert.revoked = True
            self._revoked_serials.add(cert.serial)
        logger.warning("Revoked vehicle certificate %s", certificate_id)
        return True

    def verify_vehicle_certificate(
        self,
        certificate: VehicleCertificate,
        now: Optional[int] = None,
    ) -> bool:
        """Validate a remote vehicle certificate against the trust chain."""
        ts = now or int(time.time())
        if certificate.issuer not in self.trust_chain:
            logger.info("Cert %s: untrusted issuer %s", certificate.certificate_id, certificate.issuer)
            return False
        if certificate.is_expired(ts):
            logger.info("Cert %s: expired", certificate.certificate_id)
            return False
        if certificate.revoked or certificate.serial in self._revoked_serials:
            logger.info("Cert %s: revoked", certificate.certificate_id)
            return False
        return True

    # ------------------------------------------------------------------
    # V2V / V2I mutual auth
    # ------------------------------------------------------------------
    def authenticate_remote_vehicle(
        self,
        beacon: Beacon,
        remote_cert: VehicleCertificate,
    ) -> VehicleAuthResult:
        """Verify a beacon signed by a remote vehicle."""
        if not self.verify_vehicle_certificate(remote_cert):
            return VehicleAuthResult(
                status=AuthStatus.FAILURE,
                remote_vin=remote_cert.vin,
                certificate_id=remote_cert.certificate_id,
                method=AuthMethod.VEHICLE_CERTIFICATE,
                reason="certificate invalid",
            )
        try:
            public_key = self._load_public_key(remote_cert.public_key_pem)
        except Exception as exc:
            logger.info("Failed to parse remote public key: %s", exc)
            return VehicleAuthResult(
                status=AuthStatus.FAILURE,
                remote_vin=remote_cert.vin,
                certificate_id=remote_cert.certificate_id,
                method=AuthMethod.VEHICLE_CERTIFICATE,
                reason="bad public key",
            )
        signed = self._signed_payload(beacon)
        if not self._verify_signature(public_key, signed, beacon.signature):
            return VehicleAuthResult(
                status=AuthStatus.FAILURE,
                remote_vin=remote_cert.vin,
                certificate_id=remote_cert.certificate_id,
                method=AuthMethod.VEHICLE_CERTIFICATE,
                reason="signature invalid",
            )
        with self._lock:
            self._last_seen[remote_cert.vin] = beacon.timestamp
            self._known_certs.setdefault(remote_cert.certificate_id, remote_cert)
        return VehicleAuthResult(
            status=AuthStatus.SUCCESS,
            remote_vin=remote_cert.vin,
            certificate_id=remote_cert.certificate_id,
            method=AuthMethod.VEHICLE_CERTIFICATE,
            reason="ok",
        )

    def authenticate_infrastructure(
        self,
        rsu_certificate: VehicleCertificate,
        challenge_response: bytes,
        expected_challenge: bytes,
    ) -> VehicleAuthResult:
        """V2I mutual auth: verify RSU signed the expected nonce challenge."""
        if not self.verify_vehicle_certificate(rsu_certificate):
            return VehicleAuthResult(
                status=AuthStatus.FAILURE,
                certificate_id=rsu_certificate.certificate_id,
                method=AuthMethod.VEHICLE_CERTIFICATE,
                reason="RSU cert invalid",
            )
        try:
            public_key = self._load_public_key(rsu_certificate.public_key_pem)
        except Exception:
            return VehicleAuthResult(
                status=AuthStatus.FAILURE,
                certificate_id=rsu_certificate.certificate_id,
                method=AuthMethod.VEHICLE_CERTIFICATE,
                reason="bad RSU public key",
            )
        if not self._verify_signature(public_key, expected_challenge, challenge_response):
            return VehicleAuthResult(
                status=AuthStatus.FAILURE,
                certificate_id=rsu_certificate.certificate_id,
                method=AuthMethod.VEHICLE_CERTIFICATE,
                reason="RSU challenge-response invalid",
            )
        return VehicleAuthResult(
            status=AuthStatus.SUCCESS,
            remote_vin=rsu_certificate.vin or "RSU",
            certificate_id=rsu_certificate.certificate_id,
            method=AuthMethod.VEHICLE_CERTIFICATE,
            reason="ok",
        )

    # ------------------------------------------------------------------
    # Beacon broadcast
    # ------------------------------------------------------------------
    def broadcast_beacon(
        self,
        payload: bytes,
        certificate: VehicleCertificate,
    ) -> Beacon:
        """Sign and emit a BSM for broadcast to neighbouring vehicles / RSUs."""
        ts = int(time.time())
        beacon = Beacon(
            sender_vin=self.local_vin,
            certificate_id=certificate.certificate_id,
            payload=payload,
            signature=b"",
            timestamp=ts,
        )
        beacon.signature = self._sign(self._signed_payload(beacon))
        logger.debug("Broadcasting beacon cert=%s ts=%d bytes=%d",
                     certificate.certificate_id, ts, len(payload))
        return beacon

    def issue_challenge(self) -> bytes:
        """Issue a fresh nonce to challenge a remote RSU / vehicle."""
        return secrets.token_bytes(32)

    # ------------------------------------------------------------------
    # Crypto helpers
    # ------------------------------------------------------------------
    def _signed_payload(self, beacon: Beacon) -> bytes:
        return (
            beacon.sender_vin.encode("utf-8")
            + b"|"
            + beacon.certificate_id.encode("utf-8")
            + b"|"
            + str(beacon.timestamp).encode("ascii")
            + b"|"
            + beacon.payload
        )

    def _sign(self, data: bytes) -> bytes:
        if isinstance(self.signing_key, EllipticCurvePrivateKey):
            return self.signing_key.sign(data, ECDSA(hashes.SHA256()))
        if isinstance(self.signing_key, RSAPrivateKey):
            return self.signing_key.sign(
                data, padding.PKCS1v15(), hashes.SHA256()
            )
        raise TypeError("Unsupported signing key type")

    def _verify_signature(self, public_key, data: bytes, signature: bytes) -> bool:
        try:
            if isinstance(public_key, EllipticCurvePublicKey):
                public_key.verify(signature, data, ECDSA(hashes.SHA256()))
            elif isinstance(public_key, RSAPublicKey):
                public_key.verify(
                    signature, data, padding.PKCS1v15(), hashes.SHA256()
                )
            else:
                return False
            return True
        except InvalidSignature:
            return False

    @staticmethod
    def _load_public_key(pem: str):
        loaded = serialization.load_pem_public_key(pem.encode("utf-8"))
        return loaded


__all__ = [
    "VehicleCertificate",
    "Beacon",
    "VehicleAuthResult",
    "VehicleAuthenticator",
]
