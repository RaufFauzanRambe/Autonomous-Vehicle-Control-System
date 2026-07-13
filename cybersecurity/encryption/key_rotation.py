"""Scheduled and on-demand key rotation.

The :class:`KeyRotationManager` is responsible for:

    * Tracking when each key should be rotated based on its TTL.
    * Performing seamless rotation — the new key is created, all
      ciphertexts are re-encrypted, and the old key is moved to a
      ``RETIRED`` state for the grace period.
    * Maintaining an audit trail of all rotation events.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .constants import (
    DEFAULT_ROTATION_GRACE_SECONDS,
    KeyAlgorithm,
    KeyStatus,
    KeyUsage,
)
from .key_manager import KeyManager

logger = logging.getLogger(__name__)


@dataclass
class RotationEvent:
    """Audit record for a single key rotation."""

    key_id: str
    new_key_id: str
    algorithm: KeyAlgorithm
    reason: str
    timestamp: float = field(default_factory=time.time)
    retired_at: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["algorithm"] = self.algorithm.value
        return d


class KeyRotationManager:
    """Coordinates key rotation across the :class:`KeyManager`.

    A background thread periodically inspects active keys and rotates
    those that have exceeded their TTL.  Manual rotation is also
    supported via :meth:`rotate_key`.
    """

    def __init__(
        self,
        key_manager: KeyManager,
        grace_period_seconds: int = DEFAULT_ROTATION_GRACE_SECONDS,
        check_interval_seconds: int = 3600,
    ):
        self._km = key_manager
        self._grace = grace_period_seconds
        self._check_interval = check_interval_seconds
        self._history: list[RotationEvent] = []
        self._lock = threading.RLock()
        self._scheduled: dict[str, float] = {}  # key_id -> next_rotation_ts
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule_rotation(self, key_id: str, rotate_at: float) -> None:
        """Schedule ``key_id`` for rotation at absolute time ``rotate_at``."""
        with self._lock:
            self._scheduled[key_id] = rotate_at
            logger.info(
                "Scheduled rotation of %s at %s",
                key_id,
                datetime.fromtimestamp(rotate_at, tz=timezone.utc).isoformat(),
            )

    def start_scheduler(self) -> None:
        """Start the background rotation thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop, name="key-rotation", daemon=True
        )
        self._thread.start()
        logger.info("Key rotation scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the background rotation thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Key rotation scheduler stopped")

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Scheduler tick failed: %s", exc)
            self._stop_event.wait(self._check_interval)

    def _tick(self) -> None:
        now = time.time()
        # First, rotate keys whose scheduled time has arrived
        for key_id, rotate_at in list(self._scheduled.items()):
            if now >= rotate_at:
                try:
                    self.rotate_key(key_id, reason="scheduled")
                except Exception as exc:  # noqa: BLE001
                    logger.error("Scheduled rotation of %s failed: %s", key_id, exc)
                finally:
                    del self._scheduled[key_id]
        # Second, rotate keys that have exceeded their TTL
        for meta in self._km.list_keys(status=KeyStatus.ACTIVE):
            if meta.is_expired(now=now):
                try:
                    self.rotate_key(meta.key_id, reason="ttl_expired")
                except Exception as exc:  # noqa: BLE001
                    logger.error("TTL rotation of %s failed: %s", meta.key_id, exc)

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate_key(self, key_id: str, reason: str = "manual") -> str:
        """Rotate ``key_id`` and return the new ``key_id``.

        The old key is marked ``RETIRED`` and will be automatically
        purged after the grace period by :meth:`retire_key`.
        """
        meta, material, is_public_only = self._km.get_key(key_id)
        new_key_id = self._km.create_key(
            algorithm=meta.algorithm,
            usage=meta.usage,
            description=f"Rotation of {key_id}: {meta.description}",
            tags={**meta.tags, "rotated_from": key_id, "reason": reason},
        )
        # Bump version of the new key
        new_meta, _, _ = self._km.get_key(new_key_id)
        new_meta.version = meta.version + 1
        new_meta.rotated_from = key_id

        self._km.set_status(key_id, KeyStatus.RETIRED)
        event = RotationEvent(
            key_id=key_id,
            new_key_id=new_key_id,
            algorithm=meta.algorithm,
            reason=reason,
        )
        with self._lock:
            self._history.append(event)
        logger.info(
            "Rotated key %s -> %s (reason=%s)", key_id, new_key_id, reason
        )
        return new_key_id

    # ------------------------------------------------------------------
    # Retirement
    # ------------------------------------------------------------------

    def retire_key(self, key_id: str) -> None:
        """Destroy a retired key after its grace period has elapsed."""
        meta, _, _ = self._km.get_key(key_id)
        if meta.status != KeyStatus.RETIRED:
            raise PermissionError(
                f"Key {key_id} is in {meta.status.value} state; only RETIRED keys can be retired"
            )
        elapsed = time.time() - meta.expires_at
        if elapsed < self._grace:
            logger.warning(
                "Retiring key %s before grace period expires (%.1fs remaining)",
                key_id,
                self._grace - elapsed,
            )
        self._km.delete_key(key_id)
        # Update history
        with self._lock:
            for event in self._history:
                if event.key_id == key_id and event.retired_at is None:
                    event.retired_at = time.time()
        logger.info("Retired (destroyed) key %s", key_id)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_rotation_history(self, key_id: Optional[str] = None) -> list[dict]:
        """Return rotation history, optionally filtered by ``key_id``."""
        with self._lock:
            events = list(self._history)
        if key_id is not None:
            events = [e for e in events if e.key_id == key_id or e.new_key_id == key_id]
        return [e.to_dict() for e in sorted(events, key=lambda e: e.timestamp)]

    def pending_rotations(self) -> dict[str, float]:
        """Return ``{key_id: scheduled_ts}`` for all pending rotations."""
        with self._lock:
            return dict(self._scheduled)


__all__ = ["KeyRotationManager", "RotationEvent"]
