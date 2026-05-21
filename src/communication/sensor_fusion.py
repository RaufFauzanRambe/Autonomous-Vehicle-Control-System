"""
Sensor Fusion Node for Autonomous Vehicle Perception

Implements multi-sensor fusion using an Extended Kalman Filter (EKF)
to combine data from cameras, LiDAR, radar, and IMU sensors into
a unified environmental model.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from .object_tracker import TrackedObject, TrackState


class SensorType(Enum):
    """Supported sensor types for fusion."""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"


@dataclass
class SensorReading:
    """Represents a single sensor measurement."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    covariance: np.ndarray
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusedObject:
    """Represents a fused object from multiple sensor sources."""
    object_id: int
    position: np.ndarray       # [x, y, z] in world frame
    velocity: np.ndarray       # [vx, vy, vz]
    covariance: np.ndarray     # Fused position covariance (3x3)
    size: np.ndarray           # [length, width, height]
    classification: Dict[str, float]  # class -> probability
    source_sensors: List[str]
    confidence: float
    timestamp: float


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear sensor fusion.

    Supports fusing measurements from multiple sensors with
    different coordinate frames and measurement models.
    """

    def __init__(self, state_dim: int = 6):
        """
        Initialize EKF.

        Args:
            state_dim: Dimension of the state vector (default: [x,y,z,vx,vy,vz]).
        """
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim) * 100.0
        self.process_noise = np.eye(state_dim) * 0.1

    def predict(self, dt: float) -> None:
        """
        Predict step using constant velocity model.

        Args:
            dt: Time step in seconds.
        """
        F = np.eye(self.state_dim)
        # Position += velocity * dt
        for i in range(min(3, self.state_dim // 2)):
            F[i, i + 3] = dt

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(
        self,
        measurement: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """
        Update step with a new measurement.

        Args:
            measurement: Measurement vector.
            H: Measurement Jacobian matrix.
            R: Measurement noise covariance.
        """
        # Innovation
        y = measurement - H @ self.state

        # Innovation covariance
        S = H @ self.covariance @ H.T + R

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T


class SensorFusionNode:
    """
    Multi-sensor fusion node for autonomous vehicle perception.

    Combines detections from multiple sensors (camera, LiDAR, radar)
    into a single, unified perception output with improved accuracy
    and reliability compared to any single sensor alone.
    """

    def __init__(
        self,
        fusion_mode: str = "late_fusion",
        time_synchronization_threshold: float = 0.05,
        max_sensor_latency: float = 0.1,
        publish_rate: float = 10.0,
    ):
        """
        Initialize the sensor fusion node.

        Args:
            fusion_mode: Fusion strategy ('early', 'late', or 'hybrid').
            time_synchronization_threshold: Max time difference for sync (seconds).
            max_sensor_latency: Maximum acceptable sensor latency (seconds).
            publish_rate: Output publish rate in Hz.
        """
        self.fusion_mode = fusion_mode
        self.time_sync_threshold = time_synchronization_threshold
        self.max_sensor_latency = max_sensor_latency
        self.publish_rate = publish_rate

        self._sensor_buffers: Dict[str, List[SensorReading]] = {}
        self._fused_objects: List[FusedObject] = []
        self._ekf_filters: Dict[int, ExtendedKalmanFilter] = {}
        self._lock = threading.Lock()
        self._running = False
        self._latest_timestamp = 0.0

    def start(self) -> None:
        """Start the sensor fusion node."""
        self._running = True
        self._fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self._fusion_thread.start()

    def stop(self) -> None:
        """Stop the sensor fusion node."""
        self._running = False
        if hasattr(self, '_fusion_thread'):
            self._fusion_thread.join(timeout=2.0)

    def add_sensor_reading(self, reading: SensorReading) -> None:
        """
        Add a new sensor reading to the fusion buffer.

        Args:
            reading: The sensor reading to add.
        """
        with self._lock:
            if reading.sensor_id not in self._sensor_buffers:
                self._sensor_buffers[reading.sensor_id] = []

            self._sensor_buffers[reading.sensor_id].append(reading)

            # Keep buffer bounded (last 10 readings per sensor)
            if len(self._sensor_buffers[reading.sensor_id]) > 10:
                self._sensor_buffers[reading.sensor_id].pop(0)

    def get_fused_objects(self) -> List[FusedObject]:
        """Get the latest fused objects."""
        with self._lock:
            return list(self._fused_objects)

    def _fusion_loop(self) -> None:
        """Main fusion loop running at the configured publish rate."""
        period = 1.0 / self.publish_rate
        while self._running:
            start_time = time.time()
            self._fuse()
            elapsed = time.time() - start_time
            sleep_time = max(0, period - elapsed)
            time.sleep(sleep_time)

    def _fuse(self) -> None:
        """Perform one fusion step."""
        with self._lock:
            # Synchronize sensor readings by timestamp
            synchronized = self._synchronize_readings()
            if not synchronized:
                return

            # Apply fusion based on mode
            if self.fusion_mode == "late_fusion":
                self._late_fusion(synchronized)
            elif self.fusion_mode == "early_fusion":
                self._early_fusion(synchronized)
            else:
                self._hybrid_fusion(synchronized)

    def _synchronize_readings(self) -> Dict[str, SensorReading]:
        """
        Synchronize readings from different sensors to a common timestamp.

        Returns:
            Dictionary mapping sensor_id to the best matching reading.
        """
        synchronized = {}
        if not self._sensor_buffers:
            return synchronized

        # Find the most recent common timestamp
        latest_times = []
        for sensor_id, buffer in self._sensor_buffers.items():
            if buffer:
                latest_times.append(buffer[-1].timestamp)

        if not latest_times:
            return synchronized

        target_time = max(latest_times)

        for sensor_id, buffer in self._sensor_buffers.items():
            if not buffer:
                continue

            # Find reading closest to target time
            best_reading = None
            min_diff = float('inf')
            for reading in buffer:
                diff = abs(reading.timestamp - target_time)
                if diff < min_diff and diff < self.time_sync_threshold:
                    min_diff = diff
                    best_reading = reading

            if best_reading is not None:
                synchronized[sensor_id] = best_reading

        return synchronized

    def _late_fusion(self, synchronized: Dict[str, SensorReading]) -> None:
        """
        Late fusion: fuse detection-level outputs from each sensor.

        Each sensor runs its own detection pipeline, and we fuse
        the resulting tracked objects at the object level.
        """
        fused_objects = []

        # Collect all tracked objects from sensors
        all_tracks: List[Tuple[str, TrackedObject]] = []
        for sensor_id, reading in synchronized.items():
            # Extract tracks from sensor reading (placeholder)
            # In practice, this would deserialize tracked objects from the reading
            pass

        # Cluster tracks by spatial proximity
        clusters = self._cluster_tracks(all_tracks)

        # Fuse each cluster into a single FusedObject
        for cluster in clusters:
            if len(cluster) < 1:
                continue

            fused = self._fuse_cluster(cluster)
            if fused is not None:
                fused_objects.append(fused)

        self._fused_objects = fused_objects

    def _early_fusion(self, synchronized: Dict[str, SensorReading]) -> None:
        """
        Early fusion: combine raw sensor data before detection.

        Raw point clouds and images are combined into a unified
        representation before running detection.
        """
        # Placeholder for early fusion implementation
        pass

    def _hybrid_fusion(self, synchronized: Dict[str, SensorReading]) -> None:
        """
        Hybrid fusion: combine early and late fusion strategies.

        Uses early fusion for LiDAR + camera data, and late fusion
        for radar data which has complementary information.
        """
        # Placeholder for hybrid fusion implementation
        pass

    def _cluster_tracks(
        self, tracks: List[Tuple[str, TrackedObject]], distance_threshold: float = 2.0
    ) -> List[List[Tuple[str, TrackedObject]]]:
        """
        Cluster tracks from different sensors that likely correspond
        to the same physical object.

        Args:
            tracks: List of (sensor_id, TrackedObject) tuples.
            distance_threshold: Maximum distance for clustering.

        Returns:
            List of track clusters.
        """
        if not tracks:
            return []

        clusters = []
        assigned = set()

        for i, (sensor_i, track_i) in enumerate(tracks):
            if i in assigned:
                continue

            cluster = [(sensor_i, track_i)]
            assigned.add(i)

            for j, (sensor_j, track_j) in enumerate(tracks):
                if j in assigned or sensor_j == sensor_i:
                    continue

                distance = np.linalg.norm(track_i.position - track_j.position)
                if distance < distance_threshold:
                    cluster.append((sensor_j, track_j))
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _fuse_cluster(self, cluster: List[Tuple[str, TrackedObject]]) -> Optional[FusedObject]:
        """
        Fuse a cluster of tracks into a single FusedObject.

        Uses weighted averaging based on sensor confidence and
        Kalman filter covariance for optimal fusion.
        """
        if not cluster:
            return None

        # Weighted average position
        total_weight = 0.0
        weighted_position = np.zeros(3)
        weighted_velocity = np.zeros(3)

        for sensor_id, track in cluster:
            weight = track.confidence / (np.trace(track.covariance[:3, :3]) + 1e-6)
            weighted_position += weight * track.position
            weighted_velocity += weight * track.velocity
            total_weight += weight

        if total_weight == 0:
            return None

        fused_position = weighted_position / total_weight
        fused_velocity = weighted_velocity / total_weight

        # Fused covariance (covariance intersection)
        fused_cov = np.zeros((3, 3))
        for sensor_id, track in cluster:
            fused_cov += np.linalg.inv(track.covariance[:3, :3])
        fused_cov = np.linalg.inv(fused_cov) if np.any(fused_cov != 0) else np.eye(3)

        # Merge classification probabilities
        classification = {}
        for sensor_id, track in cluster:
            class_name = track.object_class.value
            classification[class_name] = classification.get(class_name, 0) + track.confidence

        # Normalize
        total = sum(classification.values())
        if total > 0:
            classification = {k: v / total for k, v in classification.items()}

        return FusedObject(
            object_id=cluster[0][1].track_id,
            position=fused_position,
            velocity=fused_velocity,
            covariance=fused_cov,
            size=np.array([4.5, 1.8, 1.5]),  # Default vehicle size
            classification=classification,
            source_sensors=[sid for sid, _ in cluster],
            confidence=np.mean([t.confidence for _, t in cluster]),
            timestamp=max(t.last_detected_timestamp for _, t in cluster),
        )
