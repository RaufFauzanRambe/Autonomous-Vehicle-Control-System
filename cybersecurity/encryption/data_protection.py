"""At-rest and in-transit data protection policies.

The :class:`DataProtectionManager` enforces classification-based
encryption policies across the AVCS data plane.  For example, driver
biometric templates and live location traces are classified
``RESTRICTED`` and must be encrypted at the field level before being
written to the event log or sent over the V2X bus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from .aes_encryption import AESCipher
from .constants import DataClassification
from .utils import bytes_to_base64, base64_to_bytes

logger = logging.getLogger(__name__)

BytesLike = Union[bytes, bytearray, memoryview]


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------

@dataclass
class DataProtectionPolicy:
    """A single rule mapping a data class to an encryption requirement."""

    name: str
    classification: DataClassification
    field_names: list[str] = field(default_factory=list)
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    min_key_size: int = 256
    description: str = ""

    def matches(self, field_name: str, classification: DataClassification) -> bool:
        if classification.value > self.classification.value:
            return False
        if not self.field_names:
            return True
        return any(re.match(p, field_name) for p in self.field_names)


# A simple ordering for DataClassification values
_CLASSIFICATION_ORDER = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.SECRET: 3,
    DataClassification.RESTRICTED: 4,
}
# Patch comparison on enum for convenience
def _cls_lt(self, other: DataClassification) -> bool:
    return _CLASSIFICATION_ORDER[self] < _CLASSIFICATION_ORDER[other]
def _cls_gt(self, other: DataClassification) -> bool:
    return _CLASSIFICATION_ORDER[self] > _CLASSIFICATION_ORDER[other]
DataClassification.__lt__ = _cls_lt  # type: ignore[attr-defined]
DataClassification.__gt__ = _cls_gt  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class DataProtectionManager:
    """Apply classification-aware encryption to AVCS data fields."""

    # Heuristic classification rules: regex -> classification
    DEFAULT_RULES: list[tuple[str, DataClassification]] = [
        (r"(?i).*(biometric|fingerprint|iris|face|retina).*", DataClassification.RESTRICTED),
        (r"(?i).*(location|gps|lat|lon|latitude|longitude).*", DataClassification.RESTRICTED),
        (r"(?i).*(vin|license_plate|plate).*", DataClassification.CONFIDENTIAL),
        (r"(?i).*(password|secret|token|api_key|private_key).*", DataClassification.RESTRICTED),
        (r"(?i).*(email|phone|address).*", DataClassification.CONFIDENTIAL),
        (r"(?i).*(speed|accel|imu|canbus).*", DataClassification.INTERNAL),
    ]

    def __init__(self, master_key: Optional[bytes] = None):
        self._policies: list[DataProtectionPolicy] = []
        self._rules: list[tuple[str, DataClassification]] = list(self.DEFAULT_RULES)
        self._field_keys: dict[str, AESCipher] = {}
        self._master = master_key or AESCipher.generate_key()
        self._master_cipher = AESCipher(self._master)
        logger.debug("DataProtectionManager initialised")

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def add_policy(self, policy: DataProtectionPolicy) -> None:
        self._policies.append(policy)
        logger.info("Added data protection policy '%s' (%s)", policy.name, policy.classification.value)

    def add_classification_rule(self, pattern: str, classification: DataClassification) -> None:
        """Add a regex rule used by :meth:`classify_data`."""
        self._rules.append((pattern, classification))
        logger.debug("Added classification rule: %s -> %s", pattern, classification.value)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_data(self, field_name: str, value: Any = None) -> DataClassification:
        """Heuristically classify a field based on its name (and optional value)."""
        cls = DataClassification.PUBLIC
        for pattern, candidate in self._rules:
            if re.match(pattern, field_name):
                if candidate > cls:
                    cls = candidate
        # Value-based sniffing
        if isinstance(value, str) and any(s in value.lower() for s in ("ssn", "credit card", "passport")):
            cls = DataClassification.RESTRICTED
        return cls

    # ------------------------------------------------------------------
    # Policy application
    # ------------------------------------------------------------------

    def apply_policy(self, field_name: str, value: Any) -> Any:
        """Classify ``field_name`` and encrypt ``value`` if policy requires it."""
        cls = self.classify_data(field_name, value)
        matching = [p for p in self._policies if p.matches(field_name, cls) and p.encrypt_at_rest]
        if not matching:
            return value
        # Choose the most restrictive policy
        policy = max(matching, key=lambda p: _CLASSIFICATION_ORDER[p.classification])
        return self._wrap(field_name, value, policy)

    def _wrap(self, field_name: str, value: Any, policy: DataProtectionPolicy) -> dict:
        cipher = self._cipher_for_field(field_name)
        if isinstance(value, str):
            payload = value.encode("utf-8")
            encoding = "utf-8"
        elif isinstance(value, (bytes, bytearray)):
            payload = bytes(value)
            encoding = "bytes"
        else:
            payload = str(value).encode("utf-8")
            encoding = "json"
        blob = cipher.encrypt(payload)
        return {
            "__encrypted__": True,
            "policy": policy.name,
            "classification": policy.classification.value,
            "encoding": encoding,
            "field": field_name,
            "ciphertext_b64": bytes_to_base64(blob),
        }

    # ------------------------------------------------------------------
    # Field-level encryption
    # ------------------------------------------------------------------

    def encrypt_field(self, field_name: str, value: BytesLike) -> str:
        """Encrypt a single field; returns a Base64 string."""
        cipher = self._cipher_for_field(field_name)
        blob = cipher.encrypt(value)
        return bytes_to_base64(blob)

    def decrypt_field(self, field_name: str, encoded: str) -> bytes:
        """Decrypt a field previously encrypted by :meth:`encrypt_field`."""
        cipher = self._cipher_for_field(field_name)
        return cipher.decrypt(base64_to_bytes(encoded))

    def _cipher_for_field(self, field_name: str) -> AESCipher:
        if field_name not in self._field_keys:
            # Derive a per-field key from the master using HKDF-like mix
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            derived = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"avcs-field-key-" + field_name.encode("utf-8"),
            ).derive(self._master)
            self._field_keys[field_name] = AESCipher(derived)
        return self._field_keys[field_name]

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def protect_record(self, record: dict) -> dict:
        """Return a shallow copy of ``record`` with sensitive fields encrypted."""
        out: dict[str, Any] = {}
        for k, v in record.items():
            out[k] = self.apply_policy(k, v)
        return out

    def unprotect_record(self, record: dict) -> dict:
        """Reverse :meth:`protect_record` for fields encrypted by this manager."""
        out: dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, dict) and v.get("__encrypted__"):
                raw = self.decrypt_field(v["field"], v["ciphertext_b64"])
                if v.get("encoding") == "utf-8":
                    out[k] = raw.decode("utf-8")
                elif v.get("encoding") == "bytes":
                    out[k] = raw
                else:
                    out[k] = raw.decode("utf-8")
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def list_policies(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "classification": p.classification.value,
                "field_patterns": p.field_names,
                "encrypt_at_rest": p.encrypt_at_rest,
                "encrypt_in_transit": p.encrypt_in_transit,
                "min_key_size": p.min_key_size,
            }
            for p in self._policies
        ]


__all__ = ["DataProtectionManager", "DataProtectionPolicy"]
