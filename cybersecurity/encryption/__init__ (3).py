"""AVCS encryption subpackage.

Public API:

    from .encryption import EncryptionManager
    from .aes_encryption import AESCipher, AESCBC
    from .rsa_encryption import RSACipher
    from .ecc_encryption import ECCCipher
    from .key_manager import KeyManager
    from .key_rotation import KeyRotationManager
    from .certificate_manager import CertificateManager
    from .digital_signature import DigitalSignature, SignatureAlgorithm
    from .secure_channel import SecureChannel
    from .tls_manager import TLSManager
    from .data_protection import DataProtectionManager
    from .secrets_manager import SecretsManager
"""

from __future__ import annotations

from .constants import (
    CipherMode,
    DataClassification,
    HashAlgorithm,
    KeyAlgorithm,
    KeyStatus,
    KeyUsage,
)
from .config import EncryptionConfig, load_config
from .utils import (
    bytes_to_base64,
    base64_to_bytes,
    bytes_to_hex,
    hex_to_bytes,
    constant_time_compare,
    derive_key,
    generate_nonce,
    generate_salt,
    safe_random,
)
from .aes_encryption import AESCipher, AESCBC
from .rsa_encryption import RSACipher
from .ecc_encryption import ECCCipher
from .key_manager import KeyManager, KeyMetadata
from .key_rotation import KeyRotationManager, RotationEvent
from .certificate_manager import CertificateManager, IssuedCert
from .digital_signature import DigitalSignature, SignatureAlgorithm
from .secure_channel import SecureChannel, SecureChannelError
from .tls_manager import TLSManager
from .data_protection import DataProtectionManager, DataProtectionPolicy
from .secrets_manager import SecretsManager, SecretRecord
from .encryption import EncryptionManager, EncryptionStatus

__all__ = [
    "CipherMode",
    "DataClassification",
    "HashAlgorithm",
    "KeyAlgorithm",
    "KeyStatus",
    "KeyUsage",
    "EncryptionConfig",
    "load_config",
    "bytes_to_base64",
    "base64_to_bytes",
    "bytes_to_hex",
    "hex_to_bytes",
    "constant_time_compare",
    "derive_key",
    "generate_nonce",
    "generate_salt",
    "safe_random",
    "AESCipher",
    "AESCBC",
    "RSACipher",
    "ECCCipher",
    "KeyManager",
    "KeyMetadata",
    "KeyRotationManager",
    "RotationEvent",
    "CertificateManager",
    "IssuedCert",
    "DigitalSignature",
    "SignatureAlgorithm",
    "SecureChannel",
    "SecureChannelError",
    "TLSManager",
    "DataProtectionManager",
    "DataProtectionPolicy",
    "SecretsManager",
    "SecretRecord",
    "EncryptionManager",
    "EncryptionStatus",
]

__version__ = "1.0.0"
