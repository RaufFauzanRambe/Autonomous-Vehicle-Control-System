"""
LiDAR Processor Module for Autonomous Vehicle Perception

Handles LiDAR point cloud preprocessing, ground plane estimation,
clustering, and feature extraction for object detection pipelines.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PointCloud:
    """Represents a LiDAR point cloud with metadata."""
    points: np.ndarray       # (N, 4) array: [x, y, z, intensity]
    timestamp: float
    frame_id: str = "lidar"
    sensor_pose: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3]

    @property
    def intensities(self) -> np.ndarray:
        return self.points[:, 3]


@dataclass
class Cluster:
    """Represents a point cloud cluster (potential object)."""
    points: np.ndarray           # (M, 4) subset of points
    centroid: np.ndarray         # (3,) cluster center
    bounding_box: Dict           # 3D bounding box parameters
    num_points: int
    mean_intensity: float


class LidarProcessor:
    """
    LiDAR point cloud processor for autonomous driving.

    Provides a complete pipeline from raw point cloud input to
    segmented clusters ready for object classification.
    """

    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.2),
        point_cloud_range: Tuple[float, ...] = (0, -40, -3, 70.4, 40, 1),
        min_cluster_points: int = 10,
        max_cluster_points: int = 5000,
        cluster_tolerance: float = 0.5,
        ground_threshold: float = 0.2,
        remove_radius: float = 1.5,
    ):
        """
        Initialize the LiDAR processor.

        Args:
            voxel_size: Voxel grid dimensions (x, y, z) in meters.
            point_cloud_range: ROI bounds (x_min, y_min, z_min, x_max, y_max, z_max).
            min_cluster_points: Minimum points for a valid cluster.
            max_cluster_points: Maximum points per cluster.
            cluster_tolerance: DBSCAN epsilon for clustering.
            ground_threshold: Height threshold for ground plane removal.
            remove_radius: Radius around ego vehicle to remove points.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.min_cluster_points = min_cluster_points
        self.max_cluster_points = max_cluster_points
        self.cluster_tolerance = cluster_tolerance
        self.ground_threshold = ground_threshold
        self.remove_radius = remove_radius

    def process(self, point_cloud: PointCloud) -> List[Cluster]:
        """
        Full processing pipeline for a LiDAR point cloud.

        Args:
            point_cloud: Input point cloud.

        Returns:
            List of extracted clusters.
        """
        # Step 1: Remove invalid points
        cleaned = self._remove_invalid_points(point_cloud.points)

        # Step 2: Crop to ROI
        cropped = self._crop_roi(cleaned)

        # Step 3: Remove ego vehicle points
        ego_removed = self._remove_ego_points(cropped)

        # Step 4: Remove ground plane
        ground_removed = self._remove_ground_plane(ego_removed)

        # Step 5: Voxelize for downsampling
        downsampled = self._voxelize(ground_removed)

        # Step 6: Cluster remaining points
        clusters = self._cluster(downsampled)

        return clusters

    def _remove_invalid_points(self, points: np.ndarray) -> np.ndarray:
        """Remove NaN and infinite points."""
        valid_mask = np.all(np.isfinite(points), axis=1)
        return points[valid_mask]

    def _crop_roi(self, points: np.ndarray) -> np.ndarray:
        """Crop points to the region of interest."""
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return points[mask]

    def _remove_ego_points(self, points: np.ndarray) -> np.ndarray:
        """Remove points within the ego vehicle's footprint."""
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        mask = distances > self.remove_radius
        return points[mask]

    def _remove_ground_plane(self, points: np.ndarray) -> np.ndarray:
        """
        Remove ground plane points using RANSAC-like estimation.

        Estimates a plane model and removes points within the
        ground threshold of the estimated plane.
        """
        if len(points) < 100:
            return points

        # Simple height-based ground removal
        # In production, use RANSAC or plane fitting
        ground_mask = points[:, 2] > (np.percentile(points[:, 2], 5) + self.ground_threshold)
        return points[ground_mask]

    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud using voxel grid filtering.

        Each voxel is represented by the centroid of all points within it.
        """
        if len(points) == 0:
            return points

        # Compute voxel indices
        x_min = self.point_cloud_range[0]
        y_min = self.point_cloud_range[1]
        z_min = self.point_cloud_range[2]

        voxel_indices = np.floor(
            (points[:, :3] - np.array([x_min, y_min, z_min])) /
            np.array(self.voxel_size)
        ).astype(np.int32)

        # Group by voxel and compute centroids
        unique_voxels, inverse = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        downsampled = np.zeros((len(unique_voxels), 4))
        for i in range(len(unique_voxels)):
            voxel_points = points[inverse == i]
            downsampled[i, :3] = voxel_points[:, :3].mean(axis=0)
            downsampled[i, 3] = voxel_points[:, 3].mean()

        return downsampled

    def _cluster(self, points: np.ndarray) -> List[Cluster]:
        """
        Cluster points using DBSCAN algorithm.

        Groups nearby points into clusters that likely represent
        individual objects in the scene.
        """
        if len(points) < self.min_cluster_points:
            return []

        # DBSCAN-like clustering using distance-based neighbors
        labels = self._dbscan(points)

        clusters = []
        unique_labels = set(labels) - {-1}  # Remove noise label

        for label in unique_labels:
            cluster_points = points[labels == label]

            if not (self.min_cluster_points <= len(cluster_points) <= self.max_cluster_points):
                continue

            centroid = cluster_points[:, :3].mean(axis=0)
            bbox = self._compute_bounding_box(cluster_points)

            cluster = Cluster(
                points=cluster_points,
                centroid=centroid,
                bounding_box=bbox,
                num_points=len(cluster_points),
                mean_intensity=cluster_points[:, 3].mean(),
            )
            clusters.append(cluster)

        return clusters

    def _dbscan(self, points: np.ndarray) -> np.ndarray:
        """
        Simplified DBSCAN clustering implementation.

        Args:
            points: (N, 4) point cloud array.

        Returns:
            (N,) array of cluster labels (-1 for noise).
        """
        n = len(points)
        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0

        # Build neighbor lists
        for i in range(n):
            if labels[i] != -1:
                continue

            # Find neighbors
            distances = np.linalg.norm(points[i, :3] - points[:, :3], axis=1)
            neighbors = np.where(distances < self.cluster_tolerance)[0]

            if len(neighbors) < self.min_cluster_points:
                continue

            # Expand cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)
            j = 0

            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] == -1:
                    labels[q] = cluster_id
                elif labels[q] != -1:
                    j += 1
                    continue

                labels[q] = cluster_id
                q_distances = np.linalg.norm(points[q, :3] - points[:, :3], axis=1)
                q_neighbors = np.where(q_distances < self.cluster_tolerance)[0]

                if len(q_neighbors) >= self.min_cluster_points:
                    seed_set.extend(q_neighbors.tolist())

                j += 1

            cluster_id += 1

        return labels

    def _compute_bounding_box(self, points: np.ndarray) -> Dict:
        """Compute a 3D bounding box around a cluster of points."""
        xyz = points[:, :3]
        min_bounds = xyz.min(axis=0)
        max_bounds = xyz.max(axis=0)
        center = (min_bounds + max_bounds) / 2
        dimensions = max_bounds - min_bounds

        return {
            'center': center.tolist(),
            'dimensions': dimensions.tolist(),
            'min_bounds': min_bounds.tolist(),
            'max_bounds': max_bounds.tolist(),
            'volume': float(np.prod(dimensions)),
        }
