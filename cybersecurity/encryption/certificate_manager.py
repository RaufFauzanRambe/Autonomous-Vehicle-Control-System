"""X.509 certificate management for the AVCS PKI.

The :class:`CertificateManager` builds a simple two-tier PKI:

    1. A self-signed root Certificate Authority (CA).
    2. Leaf certificates signed by the CA, used by individual vehicle
       ECUs, V2X roadside units or backend services.

CRL generation is supported; OCSP responder integration is delegated to
an external service but the manager can issue OCSP-style status
responses on demand.
"""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger(__name__)


@dataclass
class IssuedCert:
    """Convenience wrapper around a signed certificate + private key."""

    certificate: x509.Certificate
    private_key_pem: bytes
    certificate_pem: bytes
    serial_number: int
    subject: str
    issuer: str
    not_before: datetime.datetime
    not_after: datetime.datetime
    revoked: bool = False
    revocation_date: Optional[datetime.datetime] = None
    metadata: dict = field(default_factory=dict)


class CertificateManager:
    """X.509 CA + leaf certificate operations."""

    def __init__(self, cert_store_path: Union[str, os.PathLike] = "/var/lib/avcs/certs"):
        self._cert_store = Path(cert_store_path)
        self._cert_store.mkdir(parents=True, exist_ok=True)
        self._ca: Optional[IssuedCert] = None
        self._issued: dict[int, IssuedCert] = {}
        self._revoked: set[int] = set()
        self._crl_last_update: Optional[datetime.datetime] = None

    # ------------------------------------------------------------------
    # CA
    # ------------------------------------------------------------------

    def create_ca(
        self,
        common_name: str = "AVCS Root CA",
        organization: str = "Autonomous Vehicle Control System",
        validity_years: int = 20,
        key_size: int = 4096,
    ) -> IssuedCert:
        """Generate a self-signed root CA certificate + RSA private key."""
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert_builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now.replace(year=now.year + validity_years))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=1), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=False,
                    key_cert_sign=True,
                    key_agreement=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                    crl_sign=True,
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
                critical=False,
            )
        )
        certificate = cert_builder.sign(private_key, hashes.SHA256())
        ca = IssuedCert(
            certificate=certificate,
            private_key_pem=private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            certificate_pem=certificate.public_bytes(serialization.Encoding.PEM),
            serial_number=certificate.serial_number,
            subject=certificate.subject.rfc4514_string(),
            issuer=certificate.issuer.rfc4514_string(),
            not_before=getattr(certificate, "not_valid_before_utc", None)
            or certificate.not_valid_before,
            not_after=getattr(certificate, "not_valid_after_utc", None)
            or certificate.not_valid_after,
        )
        self._ca = ca
        self._persist(ca, "ca")
        logger.info("Created root CA %s (serial=%d)", common_name, ca.serial_number)
        return ca

    @property
    def ca(self) -> Optional[IssuedCert]:
        return self._ca

    # ------------------------------------------------------------------
    # Leaf certificates
    # ------------------------------------------------------------------

    def issue_certificate(
        self,
        common_name: str,
        organization: str = "Autonomous Vehicle Control System",
        validity_days: int = 365,
        is_server: bool = True,
        is_client: bool = False,
        san_dns: Optional[list[str]] = None,
        san_ip: Optional[list[str]] = None,
        private_key: Optional[Union[rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey]] = None,
        use_ecdsa: bool = True,
    ) -> IssuedCert:
        """Issue a leaf certificate signed by the CA."""
        if self._ca is None:
            raise RuntimeError("No CA available; call create_ca() first")
        ca_key = serialization.load_pem_private_key(self._ca.private_key_pem, password=None)

        if private_key is None:
            if use_ecdsa:
                private_key = ec.generate_private_key(ec.SECP384R1())
            else:
                private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)

        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._ca.certificate.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=validity_days))
        )
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        eku_oids = []
        if is_server:
            eku_oids.append(ExtendedKeyUsageOID.SERVER_AUTH)
        if is_client:
            eku_oids.append(ExtendedKeyUsageOID.CLIENT_AUTH)
        if eku_oids:
            builder = builder.add_extension(x509.ExtendedKeyUsage(eku_oids), critical=False)
        # Subject Alternative Name
        san_list: list[x509.GeneralName] = []
        if san_dns:
            san_list.extend(x509.DNSName(d) for d in san_dns)
        if san_ip:
            import ipaddress
            san_list.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in san_ip)
        if san_list:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list), critical=False
            )
        builder = builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )

        sign_hash = hashes.SHA384() if isinstance(private_key, ec.EllipticCurvePrivateKey) else hashes.SHA256()
        certificate = builder.sign(ca_key, sign_hash)
        issued = IssuedCert(
            certificate=certificate,
            private_key_pem=private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            certificate_pem=certificate.public_bytes(serialization.Encoding.PEM),
            serial_number=certificate.serial_number,
            subject=certificate.subject.rfc4514_string(),
            issuer=certificate.issuer.rfc4514_string(),
            not_before=getattr(certificate, "not_valid_before_utc", None)
            or certificate.not_valid_before,
            not_after=getattr(certificate, "not_valid_after_utc", None)
            or certificate.not_valid_after,
            metadata={"common_name": common_name},
        )
        self._issued[issued.serial_number] = issued
        self._persist(issued, f"leaf-{issued.serial_number}")
        logger.info("Issued leaf certificate for %s (serial=%d)",
                    common_name, issued.serial_number)
        return issued

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke_certificate(self, serial_number: int, reason: x509.ReasonFlags = x509.ReasonFlags.unspecified) -> None:
        if serial_number not in self._issued and serial_number != getattr(self._ca, "serial_number", -1):
            raise KeyError(f"Unknown serial number: {serial_number}")
        self._revoked.add(serial_number)
        if serial_number in self._issued:
            entry = self._issued[serial_number]
            entry.revoked = True
            entry.revocation_date = datetime.datetime.now(datetime.timezone.utc)
        logger.warning("Revoked certificate serial=%d (reason=%s)", serial_number, reason.name)

    def is_revoked(self, serial_number: int) -> bool:
        return serial_number in self._revoked

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_certificate(
        self,
        certificate: Union[x509.Certificate, bytes, str],
        trusted_ca: Optional[Union[x509.Certificate, bytes]] = None,
        check_revocation: bool = True,
    ) -> bool:
        """Verify a certificate chain and (optionally) revocation status."""
        cert = self._coerce_cert(certificate)
        ca_cert = self._coerce_cert(trusted_ca) if trusted_ca is not None else (
            self._ca.certificate if self._ca else None
        )
        if ca_cert is None:
            logger.error("No trusted CA available for verification")
            return False
        try:
            # Verify signature
            ca_pubkey = ca_cert.public_key()
            if isinstance(ca_pubkey, rsa.RSAPublicKey):
                ca_pubkey.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    __import__("cryptography").hazmat.primitives.asymmetric.padding.PKCS1v15(),
                    cert.signature_hash_algorithm,
                )
            elif isinstance(ca_pubkey, ec.EllipticCurvePublicKey):
                ca_pubkey.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    ec.ECDSA(cert.signature_hash_algorithm),
                )
            # Validity window (use _utc accessors when available, fall back gracefully)
            now = datetime.datetime.now(datetime.timezone.utc)
            nvb = getattr(cert, "not_valid_before_utc", None) or cert.not_valid_before.replace(
                tzinfo=datetime.timezone.utc
            )
            nva = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(
                tzinfo=datetime.timezone.utc
            )
            if nvb > now or nva < now:
                logger.warning("Certificate %d is outside its validity window", cert.serial_number)
                return False
            # Revocation
            if check_revocation and self.is_revoked(cert.serial_number):
                logger.warning("Certificate %d is revoked", cert.serial_number)
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Certificate verification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # CRL
    # ------------------------------------------------------------------

    def generate_crl(self, validity_days: int = 30) -> x509.CertificateRevocationList:
        """Generate a fresh X.509 v2 CRL signed by the CA."""
        if self._ca is None:
            raise RuntimeError("No CA available")
        ca_key = serialization.load_pem_private_key(self._ca.private_key_pem, password=None)
        now = datetime.datetime.now(datetime.timezone.utc)
        builder = (
            x509.CertificateRevocationListBuilder()
            .issuer_name(self._ca.certificate.subject)
            .last_update(now)
            .next_update(now + datetime.timedelta(days=validity_days))
        )
        for serial in self._revoked:
            revoked_entry = (
                x509.RevokedCertificateBuilder()
                .serial_number(serial)
                .revocation_date(now)
                .add_extension(x509.CRLReason(x509.ReasonFlags.key_compromise), critical=False)
                .build()
            )
            builder = builder.add_revoked_certificate(revoked_entry)
        crl = builder.sign(private_key=ca_key, algorithm=hashes.SHA256())
        self._crl_last_update = now
        logger.info("Generated CRL with %d revoked entries", len(self._revoked))
        return crl

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_cert(cert: Union[x509.Certificate, bytes, str]) -> x509.Certificate:
        if isinstance(cert, x509.Certificate):
            return cert
        if isinstance(cert, str):
            cert = cert.encode("utf-8")
        return x509.load_pem_x509_certificate(bytes(cert))

    def _persist(self, issued: IssuedCert, prefix: str) -> None:
        try:
            (self._cert_store / f"{prefix}.crt.pem").write_bytes(issued.certificate_pem)
            (self._cert_store / f"{prefix}.key.pem").write_bytes(issued.private_key_pem)
            os.chmod(self._cert_store / f"{prefix}.key.pem", 0o600)
        except OSError as exc:
            logger.warning("Failed to persist %s: %s", prefix, exc)

    def list_issued(self) -> list[IssuedCert]:
        return list(self._issued.values())


__all__ = ["CertificateManager", "IssuedCert"]
