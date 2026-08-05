#!/usr/bin/env python3
# =============================================================================
# File: python/tracking.py
# Brief: ObjectTracker — SORT / DeepSORT multi-object tracker. Kalman filter
#        state estimation, IoU + optional embedding distance association
#        via the Hungarian algorithm, track lifecycle (confirmed / tentative
#        / deleted), and stable ID assignment.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Multi-object tracking (SORT / DeepSORT) for the Jetson AV stack.

This module consumes per-frame detection lists from
``python/object_detection.py`` and produces stable track IDs with
smoothed bounding boxes. Two association modes are supported:

* **SORT** — IoU-only association. Fast, no extra model required.
* **DeepSORT** — IoU + cosine embedding distance. Requires per-detection
  appearance embeddings (e.g. from a ReID head). Pass the embeddings to
  :meth:`ObjectTracker.update` via the ``embeddings`` argument.

Algorithm references:
* SORT: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
* DeepSORT: Wojke et al., "Simple Online and Realtime Tracking with a
  Deep Association Metric", ICIP 2017.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    linear_sum_assignment = None  # type: ignore


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_STATE_DIM = 7  # [cx, cy, w, h, vx, vy, vw]  (we keep aspect ratio fixed)
_MEAS_DIM = 4   # [cx, cy, w, h]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class Track:
    """A single tracked object."""

    track_id: int
    class_id: int
    class_name: str
    state: np.ndarray  # (7,) Kalman state
    covariance: np.ndarray  # (7, 7) Kalman covariance
    hits: int = 1
    time_since_update: int = 0
    age: int = 0
    confirmed: bool = False
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_confidence: float = 0.0
    embedding: Optional[np.ndarray] = None
    history: List[Tuple[float, float]] = field(default_factory=list)

    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Convert current state to (x1, y1, x2, y2)."""
        cx, cy, w, h = self.state[:4]
        return (int(cx - w * 0.5), int(cy - h * 0.5),
                int(cx + w * 0.5), int(cy + h * 0.5))

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "bbox": list(self.to_bbox()),
            "confidence": float(self.last_confidence),
            "hits": self.hits,
            "age": self.age,
            "confirmed": self.confirmed,
        }


# -----------------------------------------------------------------------------
# KalmanFilter — a lightweight constant-velocity model.
# -----------------------------------------------------------------------------
class KalmanFilter:
    """Constant-velocity Kalman filter for 2D bounding boxes.

    State: ``[cx, cy, w, h, vx, vy, vw]`` (aspect ratio assumed constant).

    This is a minimal reimplementation of the SORT filter so we don't
    depend on ``filterpy`` at runtime.
    """

    def __init__(self, init_bbox: Sequence[float]) -> None:
        # Initialize state at the measured position, zero velocity.
        self.x = np.zeros(_STATE_DIM, dtype=np.float32)
        cx, cy, w, h = self._xyxy_to_xywh(init_bbox)
        self.x[:4] = [cx, cy, w, h]
        self.P = np.eye(_STATE_DIM, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocity

        # Transition matrix (dt = 1 frame).
        self.F = np.eye(_STATE_DIM, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement matrix (observe only [cx, cy, w, h]).
        self.H = np.zeros((_MEAS_DIM, _STATE_DIM), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Process noise.
        self.Q = np.eye(_STATE_DIM, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.Q *= 1.0

        # Measurement noise.
        self.R = np.eye(_MEAS_DIM, dtype=np.float32)
        self.R[2:, 2:] *= 10.0  # more noise on w, h

    @staticmethod
    def _xyxy_to_xywh(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5,
                max(1.0, x2 - x1), max(1.0, y2 - y1))

    def predict(self) -> np.ndarray:
        """Advance the state by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: Sequence[float]) -> np.ndarray:
        """Correct the state with a new measurement (cx, cy, w, h)."""
        z = np.array(measurement, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(_STATE_DIM) - K @ self.H) @ self.P
        return self.x.copy()


# -----------------------------------------------------------------------------
# ObjectTracker
# -----------------------------------------------------------------------------
class ObjectTracker:
    """SORT / DeepSORT multi-object tracker.

    Args:
        max_age: Frames a track is kept without an update before deletion.
        min_hits: Hits required before a track is reported as confirmed.
        iou_threshold: Minimum IoU for a detection to match a track.
        embedding_distance_weight: Weight (0–1) given to the appearance
            embedding distance vs IoU in the cost matrix. Set to 0 for
            pure SORT (IoU only).
        max_distance_embedding: Max cosine distance for a valid match.
        class_agnostic: If True, cross-class matches are allowed.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        embedding_distance_weight: float = 0.0,
        max_distance_embedding: float = 0.7,
        class_agnostic: bool = False,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_distance_weight = embedding_distance_weight
        self.max_distance_embedding = max_distance_embedding
        self.class_agnostic = class_agnostic

        self._tracks: List[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Sequence[Tuple[int, str, float, Tuple[int, int, int, int]]],
        embeddings: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ) -> List[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: List of ``(class_id, class_name, conf, (x1,y1,x2,y2))``.
            embeddings: Optional ``(N, D)`` array of appearance embeddings,
                one per detection. Required when
                ``embedding_distance_weight > 0``.
            frame_idx: Frame index (for logging / debugging).

        Returns:
            List of confirmed :class:`Track` objects after this update.
        """
        # 1. Predict every existing track forward.
        for track in self._tracks:
            track.state = track.state.__class__(KalmanFilter(track.last_bbox).predict()) \
                if False else self._predict(track)
            track.time_since_update += 1
            track.age += 1

        # 2. Build cost matrix between tracks and detections.
        det_bboxes = np.array(
            [d[3] for d in detections] or [[0, 0, 1, 1]],
            dtype=np.float32).reshape(-1, 4)
        det_classes = np.array([d[0] for d in detections] or [0])
        det_confs = np.array([d[2] for d in detections] or [0.0])

        if len(self._tracks) == 0 or len(detections) == 0:
            matches: List[Tuple[int, int]] = []
            unmatched_tracks = list(range(len(self._tracks)))
            unmatched_dets = list(range(len(detections)))
        else:
            cost, iou_matrix = self._build_cost_matrix(
                det_bboxes, det_classes, embeddings)
            matches, unmatched_tracks, unmatched_dets = self._associate(
                cost, iou_matrix)

        # 3. Update matched tracks.
        for t_idx, d_idx in matches:
            track = self._tracks[t_idx]
            meas = KalmanFilter._xyxy_to_xywh(det_bboxes[d_idx])
            track.state = self._update_filter(track, meas)
            track.covariance = np.eye(_STATE_DIM) * 1.0  # placeholder
            track.hits += 1
            track.time_since_update = 0
            track.last_bbox = tuple(int(v) for v in det_bboxes[d_idx])
            track.last_confidence = float(det_confs[d_idx])
            track.class_id = int(det_classes[d_idx])
            if embeddings is not None:
                track.embedding = embeddings[d_idx].copy()
            if track.hits >= self.min_hits:
                track.confirmed = True
            track.history.append(
                (float(track.state[0]), float(track.state[1])))
            if len(track.history) > 100:
                track.history.pop(0)

        # 4. Create new tracks for unmatched detections.
        for d_idx in unmatched_dets:
            kf = KalmanFilter(det_bboxes[d_idx])
            class_id = int(det_classes[d_idx])
            class_name = detections[d_idx][1] if d_idx < len(detections) else str(class_id)
            track = Track(
                track_id=self._next_id,
                class_id=class_id,
                class_name=class_name,
                state=kf.x.copy(),
                covariance=kf.P.copy(),
                hits=1,
                age=0,
                last_bbox=tuple(int(v) for v in det_bboxes[d_idx]),
                last_confidence=float(det_confs[d_idx]),
                embedding=(embeddings[d_idx].copy()
                            if embeddings is not None else None),
            )
            self._tracks.append(track)
            self._next_id += 1

        # 5. Delete stale tracks.
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age]

        # 6. Return confirmed tracks (or all tracks on the first few frames).
        output = []
        for t in self._tracks:
            if (t.confirmed and t.time_since_update == 0) or \
                    (frame_idx < self.min_hits and t.time_since_update == 0):
                output.append(t)
        return output

    # ------------------------------------------------------------------ #
    # Kalman helpers (use a per-track filter stored on the track).
    # ------------------------------------------------------------------ #
    def _predict(self, track: Track) -> np.ndarray:
        """Predict the next state of a track."""
        if not hasattr(track, "_kf") or track._kf is None:
            track._kf = KalmanFilter(track.last_bbox)
        return track._kf.predict()

    def _update_filter(
        self, track: Track, measurement: Sequence[float]
    ) -> np.ndarray:
        """Apply a Kalman update to a track."""
        if not hasattr(track, "_kf") or track._kf is None:
            track._kf = KalmanFilter(track.last_bbox)
        return track._kf.update(measurement)

    # ------------------------------------------------------------------ #
    # Association
    # ------------------------------------------------------------------ #
    def _build_cost_matrix(
        self,
        det_bboxes: np.ndarray,
        det_classes: np.ndarray,
        embeddings: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the cost matrix between existing tracks and detections.

        Cost = (1 - IoU) + lambda * cosine_distance(embeddings)
        """
        n_t = len(self._tracks)
        n_d = len(det_bboxes)
        iou_matrix = np.zeros((n_t, n_d), dtype=np.float32)
        for t_idx, track in enumerate(self._tracks):
            track_box = track.to_bbox()
            for d_idx in range(n_d):
                iou_matrix[t_idx, d_idx] = self._iou(
                    track_box, det_bboxes[d_idx])

        cost = 1.0 - iou_matrix

        if (self.embedding_distance_weight > 0 and embeddings is not None):
            # Cosine distance between track embedding and detection embeddings.
            track_emb = np.array(
                [t.embedding if t.embedding is not None
                  else np.zeros_like(embeddings[0])
                  for t in self._tracks])
            track_emb_norm = track_emb / (
                np.linalg.norm(track_emb, axis=1, keepdims=True) + 1e-8)
            det_emb_norm = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            cos_sim = track_emb_norm @ det_emb_norm.T  # (n_t, n_d)
            cos_dist = 1.0 - cos_sim
            cost = (1.0 - self.embedding_distance_weight) * cost + \
                   self.embedding_distance_weight * cos_dist

        # Forbid cross-class matches (unless class_agnostic).
        if not self.class_agnostic:
            for t_idx, track in enumerate(self._tracks):
                for d_idx in range(n_d):
                    if track.class_id != int(det_classes[d_idx]):
                        cost[t_idx, d_idx] = 1e6
                        iou_matrix[t_idx, d_idx] = 0.0

        return cost, iou_matrix

    def _associate(
        self,
        cost: np.ndarray,
        iou_matrix: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Solve the assignment problem via the Hungarian algorithm."""
        if cost.size == 0:
            return (
                [],
                list(range(cost.shape[0])),
                list(range(cost.shape[1])),
            )
        if not _SCIPY_AVAILABLE:
            # Greedy fallback (slower, less optimal).
            return self._greedy_associate(cost, iou_matrix)

        row_ind, col_ind = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        unmatched_t = set(range(cost.shape[0]))
        unmatched_d = set(range(cost.shape[1]))
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((int(r), int(c)))
                unmatched_t.discard(r)
                unmatched_d.discard(c)
        return matches, list(unmatched_t), list(unmatched_d)

    def _greedy_associate(
        self,
        cost: np.ndarray,
        iou_matrix: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy fallback when scipy is not available."""
        matches: List[Tuple[int, int]] = []
        used_t, used_d = set(), set()
        # Sort by descending IoU.
        flat = iou_matrix.flatten()
        order = np.argsort(-flat)
        for flat_idx in order:
            r, c = divmod(int(flat_idx), iou_matrix.shape[1])
            if r in used_t or c in used_d:
                continue
            if iou_matrix[r, c] < self.iou_threshold:
                break
            matches.append((r, c))
            used_t.add(r)
            used_d.add(c)
        return (
            matches,
            [i for i in range(iou_matrix.shape[0]) if i not in used_t],
            [i for i in range(iou_matrix.shape[1]) if i not in used_d],
        )

    # ------------------------------------------------------------------ #
    # IoU
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """IoU between two xyxy boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.confirmed]

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all tracks (useful when re-localizing)."""
        self._tracks.clear()
        self._next_id = 1


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    tracker = ObjectTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    # Frame 1: 2 detections
    dets = [
        (2, "car", 0.9, (100, 200, 200, 300)),
        (2, "car", 0.85, (500, 200, 600, 300)),
    ]
    tracks = tracker.update(dets, frame_idx=0)
    print(f"Frame 0: {len(tracks)} confirmed tracks (expected 0 — tentative)")
    # Frame 2: same boxes shifted slightly — should confirm.
    dets = [
        (2, "car", 0.88, (102, 202, 202, 302)),
        (2, "car", 0.83, (502, 202, 602, 302)),
    ]
    tracks = tracker.update(dets, frame_idx=1)
    print(f"Frame 1: {len(tracks)} confirmed tracks (expected 2)")
    for t in tracks:
        print(f"  ID={t.track_id} class={t.class_name} bbox={t.to_bbox()}")
