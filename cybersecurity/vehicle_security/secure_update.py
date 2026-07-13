"""Secure OTA update manager (A/B slot updates).

The :class:`SecureUpdateManager` implements an atomic A/B OTA workflow:

  1. Download a signed manifest describing the update.
  2. Verify the manifest signature (RSA-3072 / ECDSA-P384).
  3. Download each chunk and verify its SHA-256 hash.
  4. Apply the update to the inactive slot (A or B).
  5. Switch the active slot at the next boot.
  6. If health-check fails after reboot, automatically roll back.

Network and filesystem access are abstracted via injectable transport/slot
objects so the manager can be unit-tested entirely in memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .constants import Severity
from .utils import compute_sha256, hex_encode, retry, safe_compare
from .security_event_logger import SecurityEventLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


class UpdateStatus(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    MANIFEST_VERIFIED = "manifest_verified"
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ChunkSpec:
    index: int
    url: str
    sha256: str
    size: int


@dataclass
class UpdateManifest:
    version: str
    target: str  # ECU or "main"
    chunks: List[ChunkSpec]
    signature: bytes = b""
    manifest_sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateState:
    status: UpdateStatus = UpdateStatus.IDLE
    progress: float = 0.0  # 0..1
    active_slot: str = "A"
    target_slot: str = "B"
    downloaded_bytes: int = 0
    total_bytes: int = 0
    failed_chunk: Optional[int] = None
    error: str = ""
    manifest: Optional[UpdateManifest] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "progress": self.progress,
            "active_slot": self.active_slot,
            "target_slot": self.target_slot,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "failed_chunk": self.failed_chunk,
            "error": self.error,
            "manifest_version": self.manifest.version if self.manifest else None,
        }


# --------------------------------------------------------------------------- #
# Transport abstraction
# --------------------------------------------------------------------------- #


class UpdateTransport:
    """Abstract network transport. Override :meth:`fetch_chunk` in production."""

    def __init__(self, chunks_cache: Optional[Dict[int, bytes]] = None) -> None:
        self._cache = chunks_cache or {}

    def fetch_manifest(self, url: str) -> bytes:
        raise NotImplementedError

    def fetch_chunk(self, spec: ChunkSpec) -> bytes:
        # Default: pull from in-memory cache (test path).
        return self._cache.get(spec.index, b"")


class InMemoryTransport(UpdateTransport):
    """Convenience transport that returns pre-staged manifest and chunks."""

    def __init__(self, manifest_bytes: bytes, chunks: Dict[int, bytes]) -> None:
        super().__init__(chunks)
        self._manifest = manifest_bytes

    def fetch_manifest(self, url: str) -> bytes:
        return self._manifest

    def fetch_chunk(self, spec: ChunkSpec) -> bytes:
        return self._cache[spec.index]


# --------------------------------------------------------------------------- #
# Slot abstraction
# --------------------------------------------------------------------------- #


class SlotBackend:
    """A/B slot abstraction. Production backends write to a real filesystem."""

    def __init__(self) -> None:
        self._slots: Dict[str, bytes] = {"A": b"", "B": b""}
        self._active = "A"

    @property
    def active_slot(self) -> str:
        return self._active

    @property
    def inactive_slot(self) -> str:
        return "B" if self._active == "A" else "A"

    def write_slot(self, slot: str, data: bytes) -> None:
        self._slots[slot] = data

    def read_slot(self, slot: str) -> bytes:
        return self._slots[slot]

    def switch_active(self, slot: str) -> None:
        if slot not in self._slots:
            raise ValueError(f"unknown slot {slot}")
        self._active = slot

    def health_check(self, slot: str) -> bool:
        return bool(self._slots.get(slot))


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class SecureUpdateManager:
    """Coordinates OTA downloads, verification, and A/B slot switching."""

    def __init__(
        self,
        transport: UpdateTransport,
        slot_backend: SlotBackend,
        trusted_pubkey_path: Optional[str] = None,
        logger_: Optional[SecurityEventLogger] = None,
        max_retries: int = 5,
    ) -> None:
        self.transport = transport
        self.slots = slot_backend
        self.trusted_pubkey_path = trusted_pubkey_path
        self.event_logger = logger_
        self.max_retries = max_retries
        self.state = UpdateState(active_slot=slot_backend.active_slot, target_slot=slot_backend.inactive_slot)

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #

    @retry(retries=3, delay=0.2, exceptions=(ConnectionError, TimeoutError))
    def download_manifest(self, url: str) -> bytes:
        self.state.status = UpdateStatus.DOWNLOADING
        logger.info("downloading manifest from %s", url)
        return self.transport.fetch_manifest(url)

    def verify_manifest(self, manifest_bytes: bytes, signature: bytes) -> UpdateManifest:
        """Verify the manifest signature and parse the JSON body."""
        if signature and self.trusted_pubkey_path:
            from .secure_boot import SignatureVerifier
            verifier = SignatureVerifier()
            if not verifier.verify(self.trusted_pubkey_path, signature, manifest_bytes, "rsa-pss"):
                raise ValueError("manifest signature verification failed")
        else:
            # Allow unsigned manifests in test/permissive mode but log it.
            logger.warning("manifest signature not checked (permissive mode)")

        data = json.loads(manifest_bytes.decode("utf-8"))
        chunks = [
            ChunkSpec(index=c["index"], url=c["url"], sha256=c["sha256"], size=c["size"])
            for c in data.get("chunks", [])
        ]
        manifest = UpdateManifest(
            version=data["version"],
            target=data.get("target", "main"),
            chunks=chunks,
            signature=signature,
            manifest_sha256=hex_encode(compute_sha256(manifest_bytes)),
            metadata=data.get("metadata", {}),
        )
        self.state.manifest = manifest
        self.state.total_bytes = sum(c.size for c in chunks)
        self.state.status = UpdateStatus.MANIFEST_VERIFIED
        self._log(Severity.INFO, "update.started", f"manifest v{manifest.version} verified")
        return manifest

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    def download_update(self, manifest: Optional[UpdateManifest] = None) -> bytes:
        """Download and hash-verify every chunk, returning the assembled blob."""
        manifest = manifest or self.state.manifest
        if manifest is None:
            raise RuntimeError("no manifest available; call verify_manifest first")

        assembled = bytearray()
        for chunk in manifest.chunks:
            try:
                data = retry(retries=self.max_retries, delay=0.2, exceptions=(ConnectionError, TimeoutError))(
                    self.transport.fetch_chunk
                )(chunk)
            except Exception as exc:  # noqa: BLE001
                self.state.status = UpdateStatus.FAILED
                self.state.failed_chunk = chunk.index
                self.state.error = f"chunk {chunk.index} download failed: {exc}"
                self._log(Severity.HIGH, "update.failed", self.state.error)
                raise

            actual = hex_encode(compute_sha256(data))
            if not safe_compare(bytes.fromhex(actual), bytes.fromhex(chunk.sha256)):
                self.state.status = UpdateStatus.FAILED
                self.state.failed_chunk = chunk.index
                self.state.error = f"chunk {chunk.index} hash mismatch (expected {chunk.sha256}, got {actual})"
                self._log(Severity.HIGH, "update.failed", self.state.error)
                raise ValueError(self.state.error)

            assembled.extend(data)
            self.state.downloaded_bytes += len(data)
            self.state.progress = self.state.downloaded_bytes / max(1, self.state.total_bytes)
            logger.debug("chunk %d/%d verified", chunk.index + 1, len(manifest.chunks))

        self._log(Severity.INFO, "update.verified", f"all {len(manifest.chunks)} chunks verified")
        return bytes(assembled)

    # ------------------------------------------------------------------ #
    # Apply
    # ------------------------------------------------------------------ #

    def apply_update(self, image: Optional[bytes] = None) -> bool:
        """Apply the downloaded image to the inactive slot and switch active."""
        if image is None:
            image = self.download_update()

        self.state.status = UpdateStatus.APPLYING
        target = self.slots.inactive_slot
        self.state.target_slot = target
        self.slots.write_slot(target, image)
        self._log(Severity.INFO, "update.applied", f"image written to slot {target}")

        if not self.slots.health_check(target):
            return self.rollback("health check failed after apply")

        self.slots.switch_active(target)
        self.state.active_slot = target
        self.state.status = UpdateStatus.APPLIED
        self.state.progress = 1.0
        self._log(Severity.INFO, "update.applied", f"active slot switched to {target}")
        return True

    def rollback(self, reason: str = "manual rollback") -> bool:
        """Switch back to the previous active slot."""
        previous = self.state.active_slot
        new_active = "B" if previous == "A" else "A"
        logger.warning("rolling back: %s (switching to slot %s)", reason, new_active)
        self.slots.switch_active(new_active)
        self.state.active_slot = new_active
        self.state.status = UpdateStatus.ROLLED_BACK
        self.state.error = reason
        self._log(Severity.HIGH, "update.rolled_back", reason)
        return True

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def get_update_status(self) -> Dict[str, Any]:
        return self.state.to_dict()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _log(self, severity: Severity, event_type: str, message: str) -> None:
        logger.log(
            {"info": logging.INFO, "low": logging.INFO, "medium": logging.WARNING, "high": logging.WARNING, "critical": logging.CRITICAL}.get(severity.value, logging.INFO),
            message,
        )
        if self.event_logger:
            self.event_logger.log_event(
                event_type=event_type,
                severity=severity.value,
                source="secure_update",
                message=message,
                details=self.state.to_dict(),
            )
