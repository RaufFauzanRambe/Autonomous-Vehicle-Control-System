"""
biometric_auth.py
=================

Driver biometric authentication (face / fingerprint / voice).

The class is deliberately agnostic about *how* biometric features are
extracted: the caller supplies a pre-computed embedding (a list of
floats) and the manager compares it against enrolled templates using
cosine similarity. This lets the same code path work with on-vehicle
face-recognition ECU, a fingerprint reader, or a cloud voice ID service.

In production the embeddings live in a secure enclave / TPM-protected
store; here we keep them in-memory so the class is testable without
hardware.
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    BIOMETRIC_FACE_THRESHOLD,
    BIOMETRIC_FINGERPRINT_THRESHOLD,
    BIOMETRIC_VOICE_THRESHOLD,
    AuthStatus,
)

logger = logging.getLogger(__name__)


class BiometricModality(enum.Enum):
    """Supported biometric modalities."""

    FACE = "face"
    FINGERPRINT = "fingerprint"
    VOICE = "voice"

    @property
    def default_threshold(self) -> float:
        return {
            BiometricModality.FACE: BIOMETRIC_FACE_THRESHOLD,
            BiometricModality.FINGERPRINT: BIOMETRIC_FINGERPRINT_THRESHOLD,
            BiometricModality.VOICE: BIOMETRIC_VOICE_THRESHOLD,
        }[self]


@dataclass
class BiometricTemplate:
    """A stored biometric template for a subject."""

    template_id: str
    subject_id: str
    modality: BiometricModality
    embedding: Tuple[float, ...]
    enrolled_at: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, candidate: Sequence[float]) -> float:
        """Cosine similarity between this template and ``candidate``."""
        return _cosine_similarity(self.embedding, tuple(candidate))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class BiometricMatchResult:
    """Outcome of an authentication attempt."""

    status: AuthStatus
    subject_id: Optional[str]
    template_id: Optional[str]
    modality: Optional[BiometricModality]
    score: float
    threshold: float
    reason: str = ""


class BiometricAuthenticator:
    """Biometric template store + matcher.

    Parameters
    ----------
    thresholds:
        Optional per-modality match thresholds. Defaults come from
        :class:`constants`.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[BiometricModality, float]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._templates: Dict[str, BiometricTemplate] = {}
        # subject_id -> {modality -> [template_id, ...]}
        self._by_subject: Dict[str, Dict[BiometricModality, List[str]]] = {}
        self._thresholds: Dict[BiometricModality, float] = {
            m: m.default_threshold for m in BiometricModality
        }
        if thresholds:
            self._thresholds.update(thresholds)

    # ------------------------------------------------------------------
    # Enrolment
    # ------------------------------------------------------------------
    def enroll(
        self,
        subject_id: str,
        modality: BiometricModality,
        embedding: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BiometricTemplate:
        """Enroll a new biometric template for ``subject_id``.

        A subject may have multiple templates per modality (e.g. multiple
        fingers); the matcher tries all of them and returns the best.
        """
        if len(embedding) == 0:
            raise ValueError("embedding must be non-empty")
        template = BiometricTemplate(
            template_id=uuid.uuid4().hex,
            subject_id=subject_id,
            modality=modality,
            embedding=tuple(float(x) for x in embedding),
            enrolled_at=int(__import__("time").time()),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._templates[template.template_id] = template
            slot = self._by_subject.setdefault(subject_id, {}).setdefault(modality, [])
            slot.append(template.template_id)
        logger.info(
            "Enrolled %s template %s for subject %s (%d total for modality)",
            modality.value, template.template_id, subject_id, len(slot),
        )
        return template

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(
        self,
        modality: BiometricModality,
        embedding: Sequence[float],
        *,
        restrict_to_subject: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> BiometricMatchResult:
        """Attempt to identify / verify ``embedding`` against enrolled templates.

        If ``restrict_to_subject`` is provided, the matcher performs
        *verification* (1:1) against that subject's templates only.
        Otherwise it performs *identification* (1:N) across all enrolled
        templates of the requested modality.
        """
        thr = threshold if threshold is not None else self._thresholds[modality]
        candidate = tuple(float(x) for x in embedding)
        if not candidate:
            return BiometricMatchResult(
                status=AuthStatus.FAILURE,
                subject_id=None,
                template_id=None,
                modality=modality,
                score=0.0,
                threshold=thr,
                reason="empty embedding",
            )

        with self._lock:
            if restrict_to_subject is not None:
                template_ids = self._by_subject.get(restrict_to_subject, {}).get(modality, [])
                templates = [self._templates[tid] for tid in template_ids if tid in self._templates]
            else:
                templates = [
                    t for t in self._templates.values() if t.modality == modality
                ]

        if not templates:
            return BiometricMatchResult(
                status=AuthStatus.FAILURE,
                subject_id=None,
                template_id=None,
                modality=modality,
                score=0.0,
                threshold=thr,
                reason="no enrolled templates",
            )

        best: Optional[BiometricTemplate] = None
        best_score = -1.0
        for t in templates:
            score = t.similarity(candidate)
            if score > best_score:
                best_score = score
                best = t

        assert best is not None
        if best_score >= thr:
            return BiometricMatchResult(
                status=AuthStatus.SUCCESS,
                subject_id=best.subject_id,
                template_id=best.template_id,
                modality=modality,
                score=best_score,
                threshold=thr,
                reason="match",
            )
        return BiometricMatchResult(
            status=AuthStatus.FAILURE,
            subject_id=best.subject_id,
            template_id=best.template_id,
            modality=modality,
            score=best_score,
            threshold=thr,
            reason="below threshold",
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    def delete_template(self, template_id: str) -> bool:
        with self._lock:
            t = self._templates.pop(template_id, None)
            if t is None:
                return False
            slot = self._by_subject.get(t.subject_id, {}).get(t.modality, [])
            if template_id in slot:
                slot.remove(template_id)
            if not slot:
                self._by_subject.get(t.subject_id, {}).pop(t.modality, None)
            if not self._by_subject.get(t.subject_id):
                self._by_subject.pop(t.subject_id, None)
        logger.info("Deleted biometric template %s", template_id)
        return True

    def delete_all_for_subject(self, subject_id: str) -> int:
        with self._lock:
            tids = [
                tid for mods in self._by_subject.get(subject_id, {}).values()
                for tid in mods
            ]
        for tid in tids:
            self.delete_template(tid)
        return len(tids)

    def list_enrolled(self, subject_id: Optional[str] = None) -> List[BiometricTemplate]:
        with self._lock:
            if subject_id is None:
                return list(self._templates.values())
            tids = [
                tid for mods in self._by_subject.get(subject_id, {}).values()
                for tid in mods
            ]
            return [self._templates[tid] for tid in tids if tid in self._templates]

    def set_threshold(self, modality: BiometricModality, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._thresholds[modality] = threshold
        logger.info("Set %s threshold to %.3f", modality.value, threshold)


__all__ = [
    "BiometricModality",
    "BiometricTemplate",
    "BiometricMatchResult",
    "BiometricAuthenticator",
]
