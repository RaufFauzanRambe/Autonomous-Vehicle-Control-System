"""
Object Tracker Module for Autonomous Vehicle Perception

Implements multi-object tracking (MOT) using Kalman filtering
and Hungarian algorithm for data association. Supports both
2D and 3D tracking with track management lifecycle.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment

from .object_detector import DetectedObject, ObjectClass, BoundingBox3D


class TrackState(Enum):
    """Lifecycle states for tracked objects."""
    TENTATIVE = "tentative"    # Newly created, not yet confirmed
    CONFIRMED = "confirmed"    # Consistently detected, reliable track
    COASTING = "coasting"      # Missed detection, predicting only
    LOST = "lost"              # No longer visible, to be deleted


@dataclass
class TrackedObject:
    """Represents a tracked object with state history and predictions."""
    track_id: int
    object_class: ObjectClass
    state: TrackState
    position: np.ndarray          # [x, y, z]
    velocity: np.ndarray          # [vx, vy, vz]
    acceleration: np.ndarray      # [ax, ay, az]
    covariance: np.ndarray        # State covariance matrix (9x9)
    bbox_3d: Optional[BoundingBox3D] = None
    confidence: float = 0.0
    age: int = 0                  # Total frames since creation
    hits: int = 0                 # Total number of positive detections
    hit_streak: int = 0           # Consecutive positive detections
    miss_streak: int = 0          # Consecutive missed detections
    last_detected_timestamp: float = 0.0
    trajectory: List[np.ndarray] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the track is still valid for decision making."""
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    @property
    def predicted_position(self) -> np.ndarray:
        """Get predicted next position based on current state."""
        dt = 0.1  # 100ms prediction step
        return self.position + self.velocity * dt + 0.5 * self.acceleration * dt ** 2


class KalmanTracker:
    """
    Kalman filter-based tracker for a single object.

    Implements a Constant Acceleration (CA) model with 9 state variables:
    [x, y, z, vx, vy, vz, ax, ay, az]
    """

    def __init__(self, initial_position: np.ndarray, initial_velocity: np.ndarray = None):
        """
        Initialize Kalman filter with first detection.

        Args:
            initial_position: Initial 3D position [x, y, z].
            initial_velocity: Initial 3D velocity [vx, vy, vz]. Defaults to zero.
        """
        if initial_velocity is None:
            initial_velocity = np.zeros(3)

        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.zeros(9)
        self.state[:3] = initial_position
        self.state[3:6] = initial_velocity

        # State covariance - high uncertainty initially
        self.covariance = np.eye(9) * 10.0
        self.covariance[3:6, 3:6] = np.eye(3) * 100.0  # Higher velocity uncertainty
        self.covariance[6:9, 6:9] = np.eye(3) * 1000.0  # Higher acceleration uncertainty

        # Process noise
        self.process_noise = np.eye(9) * 0.1
        self.process_noise[3:6, 3:6] = np.eye(3) * 1.0
        self.process_noise[6:9, 6:9] = np.eye(3) * 5.0

        # Measurement noise
        self.measurement_noise = np.eye(3) * 0.5

    def predict(self, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using the motion model.

        Args:
            dt: Time step in seconds.

        Returns:
            Tuple of (predicted_state, predicted_covariance).
        """
        # State transition matrix (constant acceleration model)
        F = np.eye(9)
        F[0, 3] = dt       # x += vx * dt
        F[1, 4] = dt       # y += vy * dt
        F[2, 5] = dt       # z += vz * dt
        F[0, 6] = 0.5 * dt**2  # x += 0.5 * ax * dt^2
        F[1, 7] = 0.5 * dt**2
        F[2, 8] = 0.5 * dt**2
        F[3, 6] = dt       # vx += ax * dt
        F[4, 7] = dt
        F[5, 8] = dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.process_noise

        # Ensure covariance is symmetric
        self.covariance = (self.covariance + self.covariance.T) / 2

        return self.state.copy(), self.covariance.copy()

    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state with a new measurement.

        Args:
            measurement: Measured position [x, y, z].

        Returns:
            Tuple of (updated_state, updated_covariance).
        """
        # Measurement matrix (we only observe position)
        H = np.zeros((3, 9))
        H[0, 0] = 1.0  # observe x
        H[1, 1] = 1.0  # observe y
        H[2, 2] = 1.0  # observe z

        # Innovation (measurement residual)
        y = measurement - H @ self.state

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.measurement_noise

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(9) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.measurement_noise @ K.T

        # Ensure covariance is symmetric
        self.covariance = (self.covariance + self.covariance.T) / 2

        return self.state.copy(), self.covariance.copy()

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self.state[6:9].copy()


class ObjectTracker:
    """
    Multi-Object Tracker using Kalman filtering and Hungarian algorithm.

    Manages the full lifecycle of object tracks including creation,
    confirmation, coasting, and deletion. Uses a distance-based
    cost matrix for data association between detections and existing tracks.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        distance_threshold: float = 5.0,
        max_coasting_frames: int = 10,
    ):
        """
        Initialize the multi-object tracker.

        Args:
            max_age: Maximum age (frames) before deleting a lost track.
            min_hits: Minimum consecutive hits to confirm a tentative track.
            iou_threshold: IoU threshold for matching 3D boxes.
            distance_threshold: Maximum distance for track-detection association.
            max_coasting_frames: Maximum frames to coast before marking as LOST.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.max_coasting_frames = max_coasting_frames

        self._tracks: Dict[int, TrackedObject] = {}
        self._kalman_filters: Dict[int, KalmanTracker] = {}
        self._next_id = 0
        self._frame_count = 0

    def update(self, detections: List[DetectedObject], dt: float = 0.1) -> List[TrackedObject]:
        """
        Update tracks with new detections for the current frame.

        Args:
            detections: List of new detections from the current frame.
            dt: Time step since the last update in seconds.

        Returns:
            List of active TrackedObject instances after update.
        """
        self._frame_count += 1

        # Step 1: Predict all existing tracks forward
        self._predict_all(dt)

        # Step 2: Associate detections with existing tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # Step 3: Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            self._update_track(track_id, det)

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # Step 5: Handle unmatched tracks (coasting/loss)
        for track_id in unmatched_tracks:
            self._handle_miss(track_id)

        # Step 6: Clean up old lost tracks
        self._cleanup()

        return self._get_active_tracks()

    def _predict_all(self, dt: float) -> None:
        """Predict all existing tracks forward by dt seconds."""
        for track_id, kf in self._kalman_filters.items():
            predicted_state, predicted_cov = kf.predict(dt)
            track = self._tracks[track_id]
            track.position = predicted_state[:3]
            track.velocity = predicted_state[3:6]
            track.acceleration = predicted_state[6:9]
            track.covariance = predicted_cov
            track.age += 1

    def _associate(
        self, detections: List[DetectedObject]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with existing tracks using Hungarian algorithm.

        Returns:
            Tuple of (matched_pairs, unmatched_det_indices, unmatched_track_ids).
        """
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(self._tracks.keys())

        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Build cost matrix based on distance
        cost_matrix = np.full((n_tracks, n_dets), float('inf'))
        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            for j, det in enumerate(detections):
                if det.bbox_3d is not None:
                    det_pos = np.array([det.bbox_3d.x, det.bbox_3d.y, det.bbox_3d.z])
                else:
                    continue

                distance = np.linalg.norm(track.position - det_pos)
                if distance < self.distance_threshold:
                    cost_matrix[i, j] = distance

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(n_dets))
        unmatched_tracks = set(track_ids)

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < self.distance_threshold:
                matched.append((track_ids[r], c))
                unmatched_dets.discard(c)
                unmatched_tracks.discard(track_ids[r])

        return matched, list(unmatched_dets), list(unmatched_tracks)

    def _update_track(self, track_id: int, detection: DetectedObject) -> None:
        """Update an existing track with a new detection."""
        track = self._tracks[track_id]
        kf = self._kalman_filters[track_id]

        # Get measurement from detection
        if detection.bbox_3d is not None:
            measurement = np.array([detection.bbox_3d.x, detection.bbox_3d.y, detection.bbox_3d.z])
        else:
            return

        # Kalman filter update
        updated_state, updated_cov = kf.update(measurement)

        # Update track properties
        track.position = updated_state[:3]
        track.velocity = updated_state[3:6]
        track.acceleration = updated_state[6:9]
        track.covariance = updated_cov
        track.confidence = detection.confidence
        track.bbox_3d = detection.bbox_3d
        track.hits += 1
        track.hit_streak += 1
        track.miss_streak = 0
        track.last_detected_timestamp = detection.timestamp
        track.object_class = detection.object_class

        # Store trajectory point
        track.trajectory.append(track.position.copy())
        if len(track.trajectory) > 100:
            track.trajectory.pop(0)

        # Promote tentative tracks to confirmed
        if track.state == TrackState.TENTATIVE and track.hit_streak >= self.min_hits:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.COASTING:
            track.state = TrackState.CONFIRMED

    def _create_track(self, detection: DetectedObject) -> None:
        """Create a new track from an unmatched detection."""
        if detection.bbox_3d is None:
            return

        track_id = self._next_id
        self._next_id += 1

        position = np.array([detection.bbox_3d.x, detection.bbox_3d.y, detection.bbox_3d.z])
        kf = KalmanTracker(position)

        track = TrackedObject(
            track_id=track_id,
            object_class=detection.object_class,
            state=TrackState.TENTATIVE,
            position=position,
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            covariance=kf.covariance,
            bbox_3d=detection.bbox_3d,
            confidence=detection.confidence,
            age=1,
            hits=1,
            hit_streak=1,
            miss_streak=0,
            last_detected_timestamp=detection.timestamp,
            trajectory=[position.copy()],
        )

        self._tracks[track_id] = track
        self._kalman_filters[track_id] = kf

    def _handle_miss(self, track_id: int) -> None:
        """Handle a track that was not matched with any detection."""
        track = self._tracks[track_id]
        track.hit_streak = 0
        track.miss_streak += 1

        if track.state == TrackState.CONFIRMED:
            if track.miss_streak > self.max_coasting_frames:
                track.state = TrackState.LOST
            else:
                track.state = TrackState.COASTING
        elif track.state == TrackState.TENTATIVE:
            if track.miss_streak > 3:
                track.state = TrackState.LOST

    def _cleanup(self) -> None:
        """Remove tracks that have been lost for too long."""
        to_delete = []
        for track_id, track in self._tracks.items():
            if track.state == TrackState.LOST and track.miss_streak > self.max_age:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self._tracks[track_id]
            del self._kalman_filters[track_id]

    def _get_active_tracks(self) -> List[TrackedObject]:
        """Get all currently active tracks."""
        return [
            track for track in self._tracks.values()
            if track.state != TrackState.LOST
        ]

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get a specific track by its ID."""
        return self._tracks.get(track_id)

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get only confirmed tracks for decision making."""
        return [
            track for track in self._tracks.values()
            if track.state == TrackState.CONFIRMED
        ]

    def reset(self) -> None:
        """Reset the tracker, clearing all tracks."""
        self._tracks.clear()
        self._kalman_filters.clear()
        self._next_id = 0
        self._frame_count = 0
