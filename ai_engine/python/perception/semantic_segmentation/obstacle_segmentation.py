"""
Obstacle Segmentation Module for Autonomous Vehicle Perception.

Implements obstacle detection and segmentation with static/dynamic
classification and distance estimation. Supports both instance-level
and semantic-level obstacle segmentation for safe navigation.

Architecture Overview:
    ┌──────────────────┐
    │   Input Image     │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Segmentation    │──── Obstacle Semantic Mask
    │  Network         │
    │  (DeepLabV3+)    │──── Instance Embeddings
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Motion          │──── Static/Dynamic Classification
    │  Analysis        │
    │  (Optical Flow)  │──── Velocity Estimation
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Distance        │──── Per-Obstacle Distance
    │  Estimation      │
    │  (Monocular/Stereo)│
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Tracking        │──── Obstacle ID + Trajectory
    │  (Kalman Filter) │
    └──────────────────┘

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .segmentation_model import (
    BaseSegmentationModel,
    ConvBlock,
    Decoder,
    DecoderConfig,
    Encoder,
    EncoderConfig,
    ModelConfig,
    BackboneType,
    TensorLike,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Obstacle-Specific Enums and Constants
# ---------------------------------------------------------------------------

class ObstacleClass(Enum):
    """Obstacle semantic class definitions for autonomous driving."""

    BACKGROUND = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    TRAFFIC_CONE = 4
    BARRIER = 5
    ANIMAL = 6
    CONSTRUCTION = 7
    DEBRIS = 8
    POLE = 9
    UNKNOWN_OBSTACLE = 10


class ObstacleMotionState(Enum):
    """Motion state classification for obstacles."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    STOPPED = "stopped"        # Was dynamic, now stopped
    UNKNOWN = "unknown"


class ObstacleThreatLevel(Enum):
    """Threat level assessment for obstacles."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Obstacle class hierarchy for grouped processing
OBSTACLE_CLASS_GROUPS = {
    "movable": {ObstacleClass.VEHICLE, ObstacleClass.PEDESTRIAN, ObstacleClass.CYCLIST, ObstacleClass.ANIMAL},
    "immovable": {ObstacleClass.TRAFFIC_CONE, ObstacleClass.BARRIER, ObstacleClass.CONSTRUCTION, ObstacleClass.POLE},
    "temporary": {ObstacleClass.DEBRIS, ObstacleClass.UNKNOWN_OBSTACLE},
}

# Typical obstacle dimensions in meters (length, width, height)
OBSTACLE_DIMENSIONS = {
    ObstacleClass.VEHICLE: (4.5, 1.8, 1.5),
    ObstacleClass.PEDESTRIAN: (0.5, 0.5, 1.7),
    ObstacleClass.CYCLIST: (1.8, 0.6, 1.7),
    ObstacleClass.TRAFFIC_CONE: (0.3, 0.3, 0.75),
    ObstacleClass.BARRIER: (1.0, 0.3, 0.9),
    ObstacleClass.ANIMAL: (1.0, 0.5, 0.5),
    ObstacleClass.CONSTRUCTION: (2.0, 1.0, 1.5),
    ObstacleClass.DEBRIS: (0.5, 0.5, 0.3),
    ObstacleClass.POLE: (0.2, 0.2, 3.0),
}

# Focal lengths for common cameras (pixels) for monocular depth estimation
CAMERA_FOCAL_LENGTHS = {
    "kitti": 721.5377,
    "cityscapes": 2262.0,
    "waymo": 2640.0,
    "nuscenes": 1266.0,
}

# Camera height above ground (meters)
DEFAULT_CAMERA_HEIGHT = 1.65


# ---------------------------------------------------------------------------
# Obstacle Segmentation Configuration
# ---------------------------------------------------------------------------

@dataclass
class ObstacleSegmentationConfig:
    """Configuration for obstacle segmentation model.

    Attributes:
        num_classes: Number of obstacle classes.
        input_size: Input image size (H, W).
        encoder_backbone: Backbone architecture.
        use_instance_seg: Whether to perform instance segmentation.
        embedding_dim: Instance embedding dimension.
        use_motion_analysis: Whether to classify static/dynamic.
        motion_threshold: Optical flow magnitude threshold for motion detection.
        use_distance_estimation: Whether to estimate obstacle distance.
        camera_height: Camera height above ground (meters).
        focal_length: Camera focal length in pixels.
        min_obstacle_area: Minimum obstacle area in pixels.
        max_obstacle_distance: Maximum detection distance (meters).
        threat_assessment: Whether to compute threat levels.
        tracking_enabled: Whether to enable obstacle tracking.
    """

    num_classes: int = len(ObstacleClass)
    input_size: Tuple[int, int] = (512, 1024)
    encoder_backbone: BackboneType = BackboneType.RESNET101
    use_instance_seg: bool = True
    embedding_dim: int = 8
    use_motion_analysis: bool = True
    motion_threshold: float = 2.0
    use_distance_estimation: bool = True
    camera_height: float = DEFAULT_CAMERA_HEIGHT
    focal_length: float = 721.5
    min_obstacle_area: int = 100
    max_obstacle_distance: float = 100.0
    threat_assessment: bool = True
    tracking_enabled: bool = True


# ---------------------------------------------------------------------------
# Obstacle Instance Representation
# ---------------------------------------------------------------------------

@dataclass
class ObstacleInstance:
    """Represents a single detected obstacle instance.

    Attributes:
        instance_id: Unique instance identifier.
        class_id: Semantic class of the obstacle.
        class_name: Human-readable class name.
        mask: Binary segmentation mask.
        bbox: Bounding box (x_min, y_min, x_max, y_max).
        confidence: Detection confidence score.
        area: Area in pixels.
        centroid: Centroid (x, y) in image coordinates.
        motion_state: Static/dynamic classification.
        velocity: Estimated velocity (vx, vy) in m/s.
        distance: Estimated distance in meters.
        threat_level: Assessed threat level.
        track_id: Tracking ID (None if not tracked).
        age: Number of frames this obstacle has been tracked.
    """

    instance_id: int = 0
    class_id: int = 0
    class_name: str = "unknown"
    mask: Optional[np.ndarray] = None
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    area: int = 0
    centroid: Tuple[float, float] = (0.0, 0.0)
    motion_state: ObstacleMotionState = ObstacleMotionState.UNKNOWN
    velocity: Tuple[float, float] = (0.0, 0.0)
    distance: float = float("inf")
    threat_level: ObstacleThreatLevel = ObstacleThreatLevel.NONE
    track_id: Optional[int] = None
    age: int = 0

    @property
    def is_dynamic(self) -> bool:
        """Check if obstacle is in motion."""
        return self.motion_state in (ObstacleMotionState.DYNAMIC, ObstacleMotionState.STOPPED)

    @property
    def is_movable_type(self) -> bool:
        """Check if obstacle belongs to a movable class."""
        return ObstacleClass(self.class_id) in OBSTACLE_CLASS_GROUPS["movable"]

    @property
    def width(self) -> int:
        """Get bounding box width."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Get bounding box height."""
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """Get bounding box aspect ratio."""
        w = self.width
        h = self.height
        return w / max(h, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "area": self.area,
            "centroid": list(self.centroid),
            "motion_state": self.motion_state.value,
            "velocity": list(self.velocity),
            "distance": round(self.distance, 2),
            "threat_level": self.threat_level.value,
            "track_id": self.track_id,
            "age": self.age,
        }


# ---------------------------------------------------------------------------
# Static/Dynamic Classification
# ---------------------------------------------------------------------------

class MotionClassifier:
    """Classifies obstacles as static or dynamic using motion cues.

    Uses optical flow, scene flow, and geometric constraints to
    determine whether an obstacle is in motion.

    Methods:
        - Optical flow magnitude analysis
        - Ego-motion compensation
        - Temporal consistency checking
    """

    def __init__(
        self,
        flow_threshold: float = 2.0,
        ego_motion_threshold: float = 0.5,
        temporal_window: int = 5,
        min_consistent_frames: int = 3,
    ) -> None:
        """Initialize motion classifier.

        Args:
            flow_threshold: Optical flow magnitude threshold (pixels/frame).
            ego_motion_threshold: Threshold for ego-motion compensated flow.
            temporal_window: Number of frames for temporal consistency.
            min_consistent_frames: Minimum frames with consistent motion.
        """
        self.flow_threshold = flow_threshold
        self.ego_motion_threshold = ego_motion_threshold
        self.temporal_window = temporal_window
        self.min_consistent_frames = min_consistent_frames
        self._history: Dict[int, List[ObstacleMotionState]] = {}

    def classify_from_flow(
        self,
        obstacle_mask: np.ndarray,
        optical_flow: np.ndarray,
        ego_motion: Optional[Tuple[float, float, float]] = None,
    ) -> ObstacleMotionState:
        """Classify obstacle motion state from optical flow.

        Args:
            obstacle_mask: Binary mask of the obstacle.
            optical_flow: Optical flow field of shape (H, W, 2).
            ego_motion: Optional ego vehicle motion (vx, vy, omega) for compensation.

        Returns:
            Classified motion state.
        """
        # Extract flow within obstacle region
        flow_masked = optical_flow[obstacle_mask > 0]

        if len(flow_masked) == 0:
            return ObstacleMotionState.UNKNOWN

        # Compute flow magnitude statistics
        magnitudes = np.sqrt(flow_masked[:, 0] ** 2 + flow_masked[:, 1] ** 2)
        median_mag = np.median(magnitudes)
        mean_mag = np.mean(magnitudes)

        # Ego-motion compensation
        if ego_motion is not None:
            vx_ego, vy_ego, omega_ego = ego_motion
            # Compute expected flow at obstacle location due to ego motion
            ys, xs = np.where(obstacle_mask > 0)
            if len(xs) > 0:
                cx = np.mean(xs)
                cy = np.mean(ys)
                # Simplified ego-motion flow model
                expected_flow_x = -vx_ego + omega_ego * cy
                expected_flow_y = -vy_ego - omega_ego * cx
                # Compensate
                compensated_mag = np.sqrt(
                    (flow_masked[:, 0] - expected_flow_x) ** 2 +
                    (flow_masked[:, 1] - expected_flow_y) ** 2
                )
                median_mag = np.median(compensated_mag)

        # Classification
        if median_mag < self.ego_motion_threshold:
            return ObstacleMotionState.STATIC
        elif median_mag > self.flow_threshold:
            return ObstacleMotionState.DYNAMIC
        else:
            return ObstacleMotionState.UNKNOWN

    def update_temporal(
        self, instance_id: int, motion_state: ObstacleMotionState
    ) -> ObstacleMotionState:
        """Update temporal motion state history and get smoothed result.

        Args:
            instance_id: Obstacle instance ID.
            motion_state: Current frame motion state.

        Returns:
            Temporally smoothed motion state.
        """
        if instance_id not in self._history:
            self._history[instance_id] = []

        self._history[instance_id].append(motion_state)

        # Keep only recent history
        if len(self._history[instance_id]) > self.temporal_window:
            self._history[instance_id] = self._history[instance_id][-self.temporal_window:]

        # Count motion states in window
        history = self._history[instance_id]
        dynamic_count = sum(1 for s in history if s == ObstacleMotionState.DYNAMIC)
        static_count = sum(1 for s in history if s == ObstacleMotionState.STATIC)

        if dynamic_count >= self.min_consistent_frames:
            return ObstacleMotionState.DYNAMIC
        elif static_count >= self.min_consistent_frames:
            return ObstacleMotionState.STATIC
        elif dynamic_count > 0 and static_count > 0:
            return ObstacleMotionState.STOPPED
        else:
            return motion_state

    def estimate_velocity(
        self,
        obstacle_mask: np.ndarray,
        optical_flow: np.ndarray,
        distance: float = 10.0,
        fps: float = 30.0,
        focal_length: float = 721.5,
    ) -> Tuple[float, float]:
        """Estimate obstacle velocity from optical flow and distance.

        Args:
            obstacle_mask: Binary obstacle mask.
            optical_flow: Optical flow field (H, W, 2).
            distance: Estimated distance to obstacle (meters).
            fps: Camera frame rate.
            focal_length: Camera focal length (pixels).

        Returns:
            Estimated velocity (vx, vy) in m/s.
        """
        flow_masked = optical_flow[obstacle_mask > 0]

        if len(flow_masked) == 0 or distance <= 0:
            return (0.0, 0.0)

        # Average flow within obstacle region
        mean_flow_x = np.mean(flow_masked[:, 0])
        mean_flow_y = np.mean(flow_masked[:, 1])

        # Convert pixel flow to real-world velocity
        # v = flow_pixels * distance / focal_length * fps
        scale = distance / focal_length * fps
        vx = -mean_flow_x * scale  # Negative: right motion = positive vx
        vy = mean_flow_y * scale

        return (float(vx), float(vy))

    def reset_history(self) -> None:
        """Reset temporal motion state history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Distance Estimation
# ---------------------------------------------------------------------------

class DistanceEstimator:
    """Estimates distance to obstacles using monocular depth cues.

    Supports multiple estimation strategies:
        - Bottom-boundary-based: Uses contact point with ground plane
        - Size-based: Uses known object dimensions
        - Depth-network-based: Uses monocular depth estimation output
        - Stereo-based: Uses stereo disparity (if available)

    Ground plane model:
        Distance = (f * H_camera) / (y_bottom - y_horizon)

    where:
        f = focal length (pixels)
        H_camera = camera height (meters)
        y_bottom = bottom y-coordinate of obstacle (pixels)
        y_horizon = horizon y-coordinate (pixels)
    """

    def __init__(
        self,
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        focal_length: float = 721.5,
        image_height: int = 512,
        horizon_ratio: float = 0.35,
        use_ground_plane: bool = True,
    ) -> None:
        """Initialize distance estimator.

        Args:
            camera_height: Camera height above ground (meters).
            focal_length: Camera focal length (pixels).
            image_height: Image height in pixels.
            horizon_ratio: Horizon position as ratio of image height.
            use_ground_plane: Whether to use ground plane model.
        """
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.image_height = image_height
        self.horizon_y = int(image_height * horizon_ratio)
        self.use_ground_plane = use_ground_plane

    def estimate_from_bottom_boundary(
        self, bbox: Tuple[int, int, int, int]
    ) -> float:
        """Estimate distance using obstacle bottom boundary and ground plane.

        This is the most reliable monocular distance cue for obstacles
        in contact with the ground plane.

        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max).

        Returns:
            Estimated distance in meters.
        """
        _, _, _, y_bottom = bbox

        # Distance = f * H / (y_bottom - y_horizon)
        pixel_offset = y_bottom - self.horizon_y
        if pixel_offset <= 0:
            return float("inf")

        distance = (self.focal_length * self.camera_height) / pixel_offset
        return max(0.0, distance)

    def estimate_from_size(
        self,
        bbox: Tuple[int, int, int, int],
        known_height: float,
    ) -> float:
        """Estimate distance using known object size.

        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max).
            known_height: Known real-world height of the object (meters).

        Returns:
            Estimated distance in meters.
        """
        _, y_min, _, y_max = bbox
        pixel_height = y_max - y_min

        if pixel_height <= 0:
            return float("inf")

        distance = (known_height * self.focal_length) / pixel_height
        return max(0.0, distance)

    def estimate_from_depth_map(
        self,
        obstacle_mask: np.ndarray,
        depth_map: np.ndarray,
    ) -> float:
        """Estimate distance using a pre-computed depth map.

        Args:
            obstacle_mask: Binary obstacle mask.
            depth_map: Depth map with per-pixel distance estimates.

        Returns:
            Median distance to the obstacle in meters.
        """
        depth_values = depth_map[obstacle_mask > 0]

        if len(depth_values) == 0:
            return float("inf")

        # Use median for robustness
        valid_depths = depth_values[depth_values > 0]
        if len(valid_depths) == 0:
            return float("inf")

        return float(np.median(valid_depths))

    def estimate_distance(
        self,
        obstacle: ObstacleInstance,
        depth_map: Optional[np.ndarray] = None,
        strategy: str = "auto",
    ) -> float:
        """Estimate distance using best available strategy.

        Args:
            obstacle: Detected obstacle instance.
            depth_map: Optional pre-computed depth map.
            strategy: Estimation strategy ('auto', 'ground', 'size', 'depth').

        Returns:
            Estimated distance in meters.
        """
        if strategy == "auto":
            if depth_map is not None and obstacle.mask is not None:
                dist = self.estimate_from_depth_map(obstacle.mask, depth_map)
            elif self.use_ground_plane:
                dist = self.estimate_from_bottom_boundary(obstacle.bbox)
            else:
                known_height = OBSTACLE_DIMENSIONS.get(
                    ObstacleClass(obstacle.class_id), (1.0, 1.0, 1.0)
                )[2]
                dist = self.estimate_from_size(obstacle.bbox, known_height)
        elif strategy == "ground":
            dist = self.estimate_from_bottom_boundary(obstacle.bbox)
        elif strategy == "size":
            known_height = OBSTACLE_DIMENSIONS.get(
                ObstacleClass(obstacle.class_id), (1.0, 1.0, 1.0)
            )[2]
            dist = self.estimate_from_size(obstacle.bbox, known_height)
        elif strategy == "depth":
            if depth_map is not None and obstacle.mask is not None:
                dist = self.estimate_from_depth_map(obstacle.mask, depth_map)
            else:
                dist = float("inf")
        else:
            dist = self.estimate_from_bottom_boundary(obstacle.bbox)

        return dist


# ---------------------------------------------------------------------------
# Threat Assessment
# ---------------------------------------------------------------------------

class ThreatAssessor:
    """Assesses threat level of detected obstacles for path planning.

    Considers multiple factors:
        - Distance to ego vehicle
        - Relative velocity (closing rate)
        - Obstacle class and size
        - Position relative to planned path
        - Time-to-collision (TTC)

    Threat levels:
        NONE (0): No threat, obstacle is far away and not in path
        LOW (1): Minor concern, monitoring required
        MEDIUM (2): Moderate threat, may require speed adjustment
        HIGH (3): Significant threat, evasive action may be needed
        CRITICAL (4): Immediate danger, emergency braking required
    """

    # Distance thresholds for threat levels (meters)
    DISTANCE_THRESHOLDS = {
        ObstacleThreatLevel.NONE: 50.0,
        ObstacleThreatLevel.LOW: 30.0,
        ObstacleThreatLevel.MEDIUM: 15.0,
        ObstacleThreatLevel.HIGH: 8.0,
        ObstacleThreatLevel.CRITICAL: 3.0,
    }

    # TTC thresholds (seconds)
    TTC_THRESHOLDS = {
        ObstacleThreatLevel.NONE: 10.0,
        ObstacleThreatLevel.LOW: 5.0,
        ObstacleThreatLevel.MEDIUM: 3.0,
        ObstacleThreatLevel.HIGH: 1.5,
        ObstacleThreatLevel.CRITICAL: 0.5,
    }

    # Class-based threat multipliers
    CLASS_THREAT_MULTIPLIER = {
        ObstacleClass.PEDESTRIAN: 1.5,
        ObstacleClass.CYCLIST: 1.3,
        ObstacleClass.VEHICLE: 1.2,
        ObstacleClass.ANIMAL: 1.1,
        ObstacleClass.TRAFFIC_CONE: 0.8,
        ObstacleClass.BARRIER: 1.0,
        ObstacleClass.CONSTRUCTION: 0.9,
        ObstacleClass.DEBRIS: 0.7,
        ObstacleClass.POLE: 0.6,
    }

    def __init__(
        self,
        ego_speed: float = 0.0,
        path_width: float = 3.5,
        ego_position_x: float = 0.5,
    ) -> None:
        """Initialize threat assessor.

        Args:
            ego_speed: Current ego vehicle speed (m/s).
            path_width: Width of planned path (meters).
            ego_position_x: Normalized x-position of ego vehicle in image.
        """
        self.ego_speed = ego_speed
        self.path_width = path_width
        self.ego_position_x = ego_position_x

    def compute_ttc(
        self, distance: float, closing_rate: float
    ) -> float:
        """Compute Time-To-Collision.

        Args:
            distance: Distance to obstacle (meters).
            closing_rate: Closing rate (m/s), positive = approaching.

        Returns:
            Time-to-collision in seconds (inf if not approaching).
        """
        if closing_rate <= 0:
            return float("inf")
        return distance / closing_rate

    def is_in_path(
        self, obstacle: ObstacleInstance, image_width: int = 1024
    ) -> bool:
        """Check if obstacle is within the planned path.

        Args:
            obstacle: Obstacle instance.
            image_width: Image width for coordinate normalization.

        Returns:
            True if obstacle is in the planned path.
        """
        # Normalize obstacle centroid to [0, 1]
        cx_normalized = obstacle.centroid[0] / image_width

        # Check if within path bounds
        half_path = self.path_width / (2 * image_width * 0.01)  # Rough conversion
        in_path = abs(cx_normalized - self.ego_position_x) < 0.15  # 15% margin

        return in_path

    def assess(
        self,
        obstacle: ObstacleInstance,
        image_width: int = 1024,
    ) -> ObstacleThreatLevel:
        """Assess threat level of an obstacle.

        Args:
            obstacle: Obstacle instance with distance and velocity.
            image_width: Image width for path calculations.

        Returns:
            Assessed threat level.
        """
        distance = obstacle.distance

        # Distance-based threat
        distance_threat = ObstacleThreatLevel.NONE
        for level in reversed(list(ObstacleThreatLevel)):
            if distance <= self.DISTANCE_THRESHOLDS[level]:
                distance_threat = level
                break

        # TTC-based threat
        velocity_mag = math.sqrt(obstacle.velocity[0] ** 2 + obstacle.velocity[1] ** 2)
        closing_rate = velocity_mag  # Simplified: assume all motion is closing
        if obstacle.is_dynamic and closing_rate > 0:
            ttc = self.compute_ttc(distance, closing_rate)
            ttc_threat = ObstacleThreatLevel.NONE
            for level in reversed(list(ObstacleThreatLevel)):
                if ttc <= self.TTC_THRESHOLDS[level]:
                    ttc_threat = level
                    break
        else:
            ttc_threat = ObstacleThreatLevel.NONE

        # Path-based threat adjustment
        in_path = self.is_in_path(obstacle, image_width)
        if not in_path:
            # Reduce threat if not in path
            distance_threat = ObstacleThreatLevel(max(0, distance_threat.value - 1))

        # Class-based multiplier
        try:
            class_mult = self.CLASS_THREAT_MULTIPLIER.get(
                ObstacleClass(obstacle.class_id), 1.0
            )
        except ValueError:
            class_mult = 1.0

        # Combine threats (take maximum, boosted by class multiplier)
        combined_level = max(distance_threat.value, ttc_threat.value)
        if class_mult > 1.2 and combined_level > 0:
            combined_level = min(4, combined_level + 1)

        return ObstacleThreatLevel(combined_level)


# ---------------------------------------------------------------------------
# Obstacle Tracker (Simplified Kalman Filter)
# ---------------------------------------------------------------------------

class ObstacleTracker:
    """Tracks obstacles across frames using simplified Kalman filtering.

    Maintains track IDs and predicts obstacle positions for
    temporal consistency and motion estimation.

    State vector: [x, y, vx, vy] (centroid position and velocity)
    """

    def __init__(
        self,
        max_lost_frames: int = 5,
        iou_threshold: float = 0.3,
        distance_threshold: float = 50.0,
    ) -> None:
        """Initialize obstacle tracker.

        Args:
            max_lost_frames: Maximum frames before track is deleted.
            iou_threshold: IoU threshold for track matching.
            distance_threshold: Maximum centroid distance for matching.
        """
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def _compute_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1)

    def update(
        self, obstacles: List[ObstacleInstance]
    ) -> List[ObstacleInstance]:
        """Update tracks with new obstacle detections.

        Args:
            obstacles: List of newly detected obstacles.

        Returns:
            List of obstacles with assigned track IDs.
        """
        matched_obstacles: List[ObstacleInstance] = []
        unmatched_track_ids = set(self._tracks.keys())
        matched_track_ids: set = set()

        for obstacle in obstacles:
            best_track_id = None
            best_score = -1.0

            for track_id in unmatched_track_ids:
                track = self._tracks[track_id]
                # IoU matching
                iou = self._compute_iou(obstacle.bbox, track["bbox"])
                # Distance matching
                centroid_dist = math.sqrt(
                    (obstacle.centroid[0] - track["centroid"][0]) ** 2 +
                    (obstacle.centroid[1] - track["centroid"][1]) ** 2
                )
                dist_score = max(0, 1 - centroid_dist / self.distance_threshold)

                # Combined score
                score = 0.5 * iou + 0.5 * dist_score
                if score > best_score and score > self.iou_threshold:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                track = self._tracks[best_track_id]
                # Simple exponential moving average for position
                alpha = 0.3
                track["centroid"] = (
                    alpha * obstacle.centroid[0] + (1 - alpha) * track["centroid"][0],
                    alpha * obstacle.centroid[1] + (1 - alpha) * track["centroid"][1],
                )
                track["bbox"] = obstacle.bbox
                track["lost_frames"] = 0
                track["age"] += 1

                obstacle.track_id = best_track_id
                obstacle.age = track["age"]
                matched_track_ids.add(best_track_id)
            else:
                # Create new track
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = {
                    "centroid": obstacle.centroid,
                    "bbox": obstacle.bbox,
                    "class_id": obstacle.class_id,
                    "lost_frames": 0,
                    "age": 1,
                }
                obstacle.track_id = track_id
                obstacle.age = 1

            matched_obstacles.append(obstacle)

        # Update unmatched tracks
        for track_id in unmatched_track_ids - matched_track_ids:
            self._tracks[track_id]["lost_frames"] += 1

        # Remove lost tracks
        lost_ids = [
            tid for tid, track in self._tracks.items()
            if track["lost_frames"] > self.max_lost_frames
        ]
        for tid in lost_ids:
            del self._tracks[tid]

        return matched_obstacles

    def get_active_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Get all currently active tracks."""
        return dict(self._tracks)

    def reset(self) -> None:
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 1


# ---------------------------------------------------------------------------
# Obstacle Segmentation Model
# ---------------------------------------------------------------------------

class ObstacleSegmentationModel(BaseSegmentationModel):
    """Obstacle detection and segmentation model.

    Combines semantic segmentation, instance segmentation, motion
    classification, and distance estimation for comprehensive obstacle
    perception in autonomous driving.

    Example:
        >>> config = ObstacleSegmentationConfig()
        >>> model = ObstacleSegmentationModel(config)
        >>> model.build_model()
        >>> image = np.random.randn(1, 3, 512, 1024).astype(np.float32)
        >>> obstacles = model.detect_obstacles(image)
        >>> for obs in obstacles:
        ...     print(f"Obstacle: {obs.class_name}, dist={obs.distance:.1f}m, "
        ...           f"motion={obs.motion_state.value}, threat={obs.threat_level.name}")
    """

    def __init__(self, config: Union[ObstacleSegmentationConfig, ModelConfig]) -> None:
        if isinstance(config, ObstacleSegmentationConfig):
            model_config = ModelConfig(
                num_classes=config.num_classes,
                input_size=config.input_size,
                encoder=EncoderConfig(backbone=config.encoder_backbone),
            )
            self.obstacle_config = config
        else:
            model_config = config
            self.obstacle_config = ObstacleSegmentationConfig(
                num_classes=config.num_classes,
                input_size=config.input_size,
            )

        super().__init__(model_config)
        self._motion_classifier = MotionClassifier(
            flow_threshold=self.obstacle_config.motion_threshold
        )
        self._distance_estimator = DistanceEstimator(
            camera_height=self.obstacle_config.camera_height,
            focal_length=self.obstacle_config.focal_length,
            image_height=self.obstacle_config.input_size[0],
        )
        self._threat_assessor = ThreatAssessor()
        self._tracker = ObstacleTracker() if self.obstacle_config.tracking_enabled else None

    def build_model(self) -> None:
        """Build the obstacle segmentation model."""
        self._encoder = Encoder(self.config.encoder)
        self._decoder = Decoder(
            self.config.decoder,
            self._encoder.get_feature_channels(),
            self.config.num_classes,
        )
        self._is_built = True
        logger.info(
            f"ObstacleSegmentationModel built: "
            f"{self.config.encoder.backbone.value} backbone, "
            f"{self.obstacle_config.num_classes} classes, "
            f"instance_seg={self.obstacle_config.use_instance_seg}, "
            f"motion={self.obstacle_config.use_motion_analysis}, "
            f"distance={self.obstacle_config.use_distance_estimation}"
        )

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass through obstacle segmentation model.

        Args:
            x: Input image tensor (N, 3, H, W).

        Returns:
            Obstacle segmentation logits.
        """
        if not self._is_built:
            self.build_model()

        if isinstance(x, np.ndarray):
            n = x.shape[0] if x.ndim == 4 else 1
            h, w = self.config.output_size
            output = np.random.randn(n, self.config.num_classes, h, w).astype(np.float32) * 0.01

            if self.obstacle_config.use_instance_seg:
                self._instance_embeddings = np.random.randn(
                    n, self.obstacle_config.embedding_dim, h, w
                ).astype(np.float32) * 0.1
            return output
        return np.zeros((1, self.config.num_classes, *self.config.output_size), dtype=np.float32)

    def get_loss(
        self,
        predictions: TensorLike,
        targets: TensorLike,
        weights: Optional[TensorLike] = None,
    ) -> float:
        """Compute obstacle segmentation loss.

        Uses focal loss to handle class imbalance (background dominates).

        Args:
            predictions: Model output logits.
            targets: Ground truth labels.
            weights: Optional per-class weights.

        Returns:
            Loss value.
        """
        if not isinstance(predictions, np.ndarray):
            return 0.0

        n, c, h, w = predictions.shape
        logits_flat = predictions.reshape(n, c, -1)
        targets_flat = targets.reshape(n, -1) if targets.ndim == 3 else targets.reshape(n, -1)

        # Focal loss computation
        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        shifted = logits_flat - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        probs = np.exp(log_probs)

        batch_idx = np.arange(n)[:, np.newaxis]
        pixel_idx = np.arange(h * w)[np.newaxis, :]
        target_probs = probs[batch_idx, targets_flat, pixel_idx]

        # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
        gamma = 2.0
        focal_weight = (1 - target_probs) ** gamma
        focal_loss = -focal_weight * log_probs[batch_idx, targets_flat, pixel_idx]

        if weights is not None:
            sample_weights = weights[targets_flat]
            focal_loss = focal_loss * sample_weights

        return float(np.mean(focal_loss))

    def detect_obstacles(
        self,
        image: TensorLike,
        optical_flow: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        ego_motion: Optional[Tuple[float, float, float]] = None,
    ) -> List[ObstacleInstance]:
        """Full obstacle detection pipeline.

        Args:
            image: Input image tensor.
            optical_flow: Optional optical flow for motion analysis.
            depth_map: Optional depth map for distance estimation.
            ego_motion: Optional ego vehicle motion (vx, vy, omega).

        Returns:
            List of detected obstacle instances.
        """
        # Forward pass
        logits = self.predict(image)
        seg_mask = self.get_segmentation_mask(logits)

        # Extract obstacle instances
        obstacles = self._extract_instances(seg_mask, logits)

        # Motion classification
        if self.obstacle_config.use_motion_analysis and optical_flow is not None:
            for obs in obstacles:
                if obs.mask is not None:
                    motion = self._motion_classifier.classify_from_flow(
                        obs.mask, optical_flow, ego_motion
                    )
                    obs.motion_state = self._motion_classifier.update_temporal(
                        obs.instance_id, motion
                    )
                    obs.velocity = self._motion_classifier.estimate_velocity(
                        obs.mask, optical_flow, obs.distance
                    )

        # Distance estimation
        if self.obstacle_config.use_distance_estimation:
            for obs in obstacles:
                obs.distance = self._distance_estimator.estimate_distance(
                    obs, depth_map
                )

        # Threat assessment
        if self.obstacle_config.threat_assessment:
            h, w = self.obstacle_config.input_size
            for obs in obstacles:
                obs.threat_level = self._threat_assessor.assess(obs, w)

        # Tracking
        if self._tracker is not None:
            obstacles = self._tracker.update(obstacles)

        return obstacles

    def _extract_instances(
        self, seg_mask: np.ndarray, logits: np.ndarray
    ) -> List[ObstacleInstance]:
        """Extract obstacle instances from segmentation mask.

        Args:
            seg_mask: Semantic segmentation mask.
            logits: Model output logits for confidence estimation.

        Returns:
            List of obstacle instances.
        """
        obstacles: List[ObstacleInstance] = []

        if seg_mask.ndim == 3:
            seg_mask = seg_mask[0]

        # Get obstacle class IDs (non-background)
        obstacle_classes = set(np.unique(seg_mask)) - {ObstacleClass.BACKGROUND.value}

        instance_id = 0
        for class_id in obstacle_classes:
            try:
                class_enum = ObstacleClass(class_id)
            except ValueError:
                continue

            class_mask = (seg_mask == class_id).astype(np.uint8)

            # Connected component analysis (simplified)
            components = self._find_connected_components(class_mask)

            for component_mask in components:
                area = np.sum(component_mask)
                if area < self.obstacle_config.min_obstacle_area:
                    continue

                instance_id += 1
                bbox = self._compute_bbox(component_mask)
                centroid = self._compute_centroid(component_mask)

                # Compute confidence from logits
                confidence = self._compute_confidence(logits, component_mask, class_id)

                obs = ObstacleInstance(
                    instance_id=instance_id,
                    class_id=class_id,
                    class_name=class_enum.name,
                    mask=component_mask,
                    bbox=bbox,
                    confidence=confidence,
                    area=int(area),
                    centroid=centroid,
                )
                obstacles.append(obs)

        return obstacles

    def _find_connected_components(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find connected components using flood fill.

        Args:
            mask: Binary mask.

        Returns:
            List of component masks.
        """
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components = []

        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0 and not visited[i, j]:
                    component = np.zeros_like(mask)
                    queue = [(i, j)]
                    visited[i, j] = True
                    while queue:
                        cy, cx = queue.pop(0)
                        component[cy, cx] = 1
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] > 0:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                    if np.sum(component) >= self.obstacle_config.min_obstacle_area:
                        components.append(component)

        return components

    @staticmethod
    def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute bounding box from mask."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return (0, 0, 0, 0)
        return (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))

    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid from mask."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return (0.0, 0.0)
        return (float(np.mean(xs)), float(np.mean(ys)))

    @staticmethod
    def _compute_confidence(
        logits: np.ndarray, mask: np.ndarray, class_id: int
    ) -> float:
        """Compute detection confidence from logits."""
        if not isinstance(logits, np.ndarray):
            return 0.5

        if logits.ndim == 4:
            logits = logits[0]

        if class_id < logits.shape[0]:
            class_logits = logits[class_id]
            masked_logits = class_logits[mask > 0]
            if len(masked_logits) > 0:
                # Sigmoid for confidence
                confidence = 1.0 / (1.0 + np.exp(-masked_logits))
                return float(np.mean(confidence))

        return 0.5

    def get_obstacle_summary(
        self, obstacles: List[ObstacleInstance]
    ) -> Dict[str, Any]:
        """Generate a summary of detected obstacles.

        Args:
            obstacles: List of detected obstacle instances.

        Returns:
            Summary dictionary.
        """
        summary = {
            "total_obstacles": len(obstacles),
            "dynamic_count": sum(1 for o in obstacles if o.is_dynamic),
            "static_count": sum(1 for o in obstacles if o.motion_state == ObstacleMotionState.STATIC),
            "class_counts": {},
            "closest_obstacle": None,
            "highest_threat": ObstacleThreatLevel.NONE.name,
            "obstacles_in_path": 0,
        }

        # Count by class
        for obs in obstacles:
            name = obs.class_name
            summary["class_counts"][name] = summary["class_counts"].get(name, 0) + 1

        # Find closest obstacle
        if obstacles:
            closest = min(obstacles, key=lambda o: o.distance)
            summary["closest_obstacle"] = closest.to_dict()

        # Highest threat
        if obstacles:
            max_threat = max(obstacles, key=lambda o: o.threat_level.value)
            summary["highest_threat"] = max_threat.threat_level.name

        return summary
