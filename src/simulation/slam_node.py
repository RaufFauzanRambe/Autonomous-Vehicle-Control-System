"""
SLAM Node Module for Autonomous Vehicle Localization

Implements Simultaneous Localization and Mapping using LiDAR
point cloud registration for real-time vehicle pose estimation
and incremental map building.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .ekf_localizer import EKFLocalizer, VehicleState
from .map_manager import MapManager, LaneSegment


@dataclass
class SLAMConfig:
    """SLAM system configuration."""
    # ICP parameters
    icp_max_iterations: int = 50
    icp_tolerance: float = 1e-6
    icp_max_correspondence_distance: float = 2.0

    # Map update parameters
    map_update_interval: float = 0.5       # seconds between map updates
    keyframe_distance: float = 1.0         # meters between keyframes
    keyframe_angle: float = np.radians(10) # radians between keyframes

    # Loop closure parameters
    loop_closure_enabled: bool = True
    loop_closure_radius: float = 15.0      # search radius for loop closure
    loop_closure_min_translation: float = 50.0  # min distance before checking

    # General parameters
    publish_rate: float = 20.0             # Hz
    local_map_radius: float = 50.0         # meters


@dataclass
class KeyFrame:
    """Represents a keyframe in the SLAM map."""
    frame_id: int
    pose: np.ndarray          # 4x4 transformation matrix
    timestamp: float
    point_cloud: np.ndarray   # Associated point cloud (N, 4)
    features: Optional[Dict] = None


class SLAMNode:
    """
    LiDAR SLAM node for autonomous vehicle localization.

    Uses Iterative Closest Point (ICP) for scan matching
    and maintains a pose graph for loop closure detection.
    Provides real-time vehicle pose estimates and incrementally
    builds a consistent map of the environment.
    """

    def __init__(
        self,
        config: Optional[SLAMConfig] = None,
        initial_state: Optional[VehicleState] = None,
    ):
        """
        Initialize the SLAM node.

        Args:
            config: SLAM configuration. Uses defaults if None.
            initial_state: Initial vehicle state estimate.
        """
        self.config = config or SLAMConfig()
        self._localizer = EKFLocalizer(initial_state=initial_state)
        self._map_manager = MapManager()
        self._keyframes: List[KeyFrame] = []
        self._pose_graph: List[Tuple[int, int, np.ndarray]] = []  # (i, j, relative_pose)
        self._current_frame_id = 0
        self._last_keyframe_pose: Optional[np.ndarray] = None
        self._local_map: Optional[np.ndarray] = None
        self._is_running = False

    @property
    def current_pose(self) -> VehicleState:
        """Get the current vehicle pose estimate."""
        return self._localizer.state

    def start(self) -> None:
        """Start the SLAM node."""
        self._is_running = True

    def stop(self) -> None:
        """Stop the SLAM node."""
        self._is_running = False

    def update(
        self,
        point_cloud: np.ndarray,
        timestamp: float = 0.0,
        imu_accel: Optional[np.ndarray] = None,
    ) -> VehicleState:
        """
        Process a new LiDAR scan.

        Performs scan matching against the local map, updates
        the EKF state, and decides whether to add a keyframe.

        Args:
            point_cloud: Current LiDAR scan (N, 4) [x, y, z, intensity].
            timestamp: Scan timestamp.
            imu_accel: IMU acceleration for prediction step.

        Returns:
            Updated vehicle state estimate.
        """
        if not self._is_running:
            raise RuntimeError("SLAM node is not running. Call start() first.")

        # EKF predict step
        dt = 0.05  # 50ms default scan period
        self._localizer.predict(dt, imu_accel)

        # Scan matching
        if self._local_map is not None and len(self._local_map) > 0:
            # ICP scan matching against local map
            relative_transform = self._icp_match(point_cloud, self._local_map)

            if relative_transform is not None:
                # Convert ICP result to pose measurement
                pose_measurement = self._transform_to_pose(relative_transform)
                self._localizer.update_lidar(pose_measurement)

        # Keyframe decision
        if self._should_add_keyframe():
            self._add_keyframe(point_cloud, timestamp)
            self._update_local_map()

        return self._localizer.state

    def _icp_match(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Iterative Closest Point (ICP) scan matching.

        Aligns the source point cloud to the target by iteratively
        finding closest points and minimizing the alignment error.

        Args:
            source: Source point cloud (N, 3).
            target: Target point cloud (M, 3).
            initial_guess: Initial 4x4 transformation guess.

        Returns:
            4x4 transformation matrix, or None if matching failed.
        """
        if initial_guess is None:
            transform = np.eye(4)
        else:
            transform = initial_guess.copy()

        src_xyz = source[:, :3]
        tgt_xyz = target[:, :3]

        for iteration in range(self.config.icp_max_iterations):
            # Transform source points
            ones = np.ones((len(src_xyz), 1))
            src_h = np.hstack([src_xyz, ones])
            transformed = (transform @ src_h.T).T[:, :3]

            # Find nearest neighbors
            distances, indices = self._find_nearest_neighbors(transformed, tgt_xyz)

            # Filter by max correspondence distance
            valid = distances < self.config.icp_max_correspondence_distance
            if np.sum(valid) < 10:
                return None

            matched_src = src_xyz[valid]
            matched_tgt = tgt_xyz[indices[valid]]

            # Compute optimal transform using SVD
            delta_transform = self._compute_rigid_transform(matched_src, matched_tgt)

            # Update transform
            transform = delta_transform @ transform

            # Check convergence
            mean_error = np.mean(distances[valid])
            if mean_error < self.config.icp_tolerance:
                break

        return transform

    @staticmethod
    def _find_nearest_neighbors(
        source: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbors in target for each source point.

        Uses brute-force search. In production, use a KD-tree
        for efficient nearest neighbor queries.
        """
        # Brute force (use scipy.spatial.cKDTree in production)
        distances = np.full(len(source), float('inf'))
        indices = np.full(len(source), -1, dtype=np.int32)

        for i, s in enumerate(source):
            dists = np.linalg.norm(target - s, axis=1)
            idx = np.argmin(dists)
            distances[i] = dists[idx]
            indices[i] = idx

        return distances, indices

    @staticmethod
    def _compute_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute rigid body transform (R, t) from source to target using SVD.

        Args:
            source: (N, 3) source points.
            target: (N, 3) target points.

        Returns:
            4x4 transformation matrix.
        """
        centroid_src = source.mean(axis=0)
        centroid_tgt = target.mean(axis=0)

        src_centered = source - centroid_src
        tgt_centered = target - centroid_tgt

        # Cross-covariance matrix
        H = src_centered.T @ tgt_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Rotation
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = centroid_tgt - R @ centroid_src

        # Build 4x4 transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _should_add_keyframe(self) -> bool:
        """Determine if the current pose warrants adding a new keyframe."""
        current_pose = self._localizer.state

        if self._last_keyframe_pose is None:
            return True

        # Check translation distance
        last_pos = self._last_keyframe_pose[:3, 3]
        curr_pos = current_pose.position
        translation = np.linalg.norm(curr_pos - last_pos)

        # Check rotation difference
        last_yaw = np.arctan2(self._last_keyframe_pose[1, 0], self._last_keyframe_pose[0, 0])
        yaw_diff = abs(current_pose.yaw - last_yaw)
        yaw_diff = min(yaw_diff, 2 * np.pi - yaw_diff)

        return (
            translation > self.config.keyframe_distance or
            yaw_diff > self.config.keyframe_angle
        )

    def _add_keyframe(self, point_cloud: np.ndarray, timestamp: float) -> None:
        """Add a keyframe at the current pose."""
        state = self._localizer.state

        # Build 4x4 pose matrix
        pose = np.eye(4)
        R = EKFLocalizer._rotation_matrix(state.orientation)
        pose[:3, :3] = R
        pose[:3, 3] = state.position

        keyframe = KeyFrame(
            frame_id=self._current_frame_id,
            pose=pose,
            timestamp=timestamp,
            point_cloud=point_cloud.copy(),
        )

        self._keyframes.append(keyframe)
        self._current_frame_id += 1
        self._last_keyframe_pose = pose

    def _update_local_map(self) -> None:
        """Rebuild the local map from recent keyframes."""
        local_points = []
        state = self._localizer.state

        for kf in self._keyframes:
            # Check if keyframe is within local map radius
            kf_pos = kf.pose[:3, 3]
            dist = np.linalg.norm(kf_pos - state.position)
            if dist <= self.config.local_map_radius:
                # Transform keyframe points to world frame
                ones = np.ones((len(kf.point_cloud), 1))
                pts_h = np.hstack([kf.point_cloud[:, :3], ones])
                world_pts = (kf.pose @ pts_h.T).T[:, :3]
                local_points.append(world_pts)

        if local_points:
            self._local_map = np.vstack(local_points)
        else:
            self._local_map = None

    @staticmethod
    def _transform_to_pose(transform: np.ndarray) -> np.ndarray:
        """Convert 4x4 transform to [x, y, z, roll, pitch, yaw]."""
        x, y, z = transform[:3, 3]
        R = transform[:3, :3]

        # Extract Euler angles (ZYX convention)
        pitch = np.arcsin(-R[2, 0])
        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0

        return np.array([x, y, z, roll, pitch, yaw])

    def check_loop_closure(self) -> Optional[Tuple[int, int, np.ndarray]]:
        """
        Check for loop closure opportunities.

        Compares current pose against historical keyframes
        to detect revisited locations for map optimization.

        Returns:
            Tuple of (current_keyframe_id, matched_keyframe_id, relative_transform)
            or None if no loop closure detected.
        """
        if not self.config.loop_closure_enabled or len(self._keyframes) < 10:
            return None

        current = self._keyframes[-1]
        current_pos = current.pose[:3, 3]

        # Skip recent keyframes
        for kf in self._keyframes[:-10]:
            kf_pos = kf.pose[:3, 3]
            dist = np.linalg.norm(current_pos - kf_pos)

            if dist < self.config.loop_closure_radius:
                # Verify with ICP
                if self._local_map is not None:
                    relative = self._icp_match(
                        current.point_cloud[:, :3],
                        kf.point_cloud[:, :3],
                    )
                    if relative is not None:
                        return (current.frame_id, kf.frame_id, relative)

        return None
