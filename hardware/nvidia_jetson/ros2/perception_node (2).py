#!/usr/bin/env python3
# =============================================================================
# File: ros2/perception_node.py
# Brief: PerceptionNode — fuses 2D camera detections (from AINode) with 3D
#        LiDAR point clouds to produce a unified 3D object list and an
#        obstacle / drivable-area occupancy grid. Publishes to topics
#        consumed by PlanningNode.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Sensor-fusion ROS2 node for the Jetson AV stack.

Topic interface
---------------
  Subscriptions:
    /ai/detections         [vision_msgs/Detection2DArray]  — from AINode
    /lidar/points          [sensor_msgs/PointCloud2]        — best-effort
    /camera/camera_info    [sensor_msgs/CameraInfo]         — for projection
    /tf                    [tf2_msgs/TFMessage]              — lidar->camera

  Publishers:
    /perception/objects    [vision_msgs/Detection3DArray]   — fused objects
    /perception/obstacles  [nav_msgs/OccupancyGrid]         — 2D obstacle map
    /perception/drivable   [sensor_msgs/Image]              — drivable mask
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Header
from vision_msgs.msg import (
    Detection2DArray, Detection3DArray, Detection3D,
    ObjectHypothesisWithPose, BoundingBox3D,
)
try:
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from geometry_msgs.msg import Pose, Point, Quaternion
    _NAV_MSGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NAV_MSGS_AVAILABLE = False
    OccupancyGrid = None  # type: ignore

try:
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    _PC2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PC2_AVAILABLE = False


# -----------------------------------------------------------------------------
# Tiny 3D track record (one per fused object).
# -----------------------------------------------------------------------------
class FusedObject:
    """A single fused 3D object (camera + LiDAR)."""

    __slots__ = ("track_id", "class_id", "class_name", "confidence",
                 "position", "dimensions", "velocity", "last_seen",
                 "hits")

    def __init__(self, track_id: int, class_id: int, class_name: str) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.confidence: float = 0.0
        self.position: np.ndarray = np.zeros(3, dtype=np.float32)
        self.dimensions: np.ndarray = np.zeros(3, dtype=np.float32)
        self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        self.last_seen: float = 0.0
        self.hits: int = 0


# -----------------------------------------------------------------------------
# PerceptionNode
# -----------------------------------------------------------------------------
class PerceptionNode(Node):
    """Fuse camera detections and LiDAR point clouds into 3D objects."""

    def __init__(self) -> None:
        super().__init__("perception_node")

        # --- Parameters ---------------------------------------------------
        self.declare_parameter("det_topic", "/ai/detections")
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("objects_topic", "/perception/objects")
        self.declare_parameter("obstacles_topic", "/perception/obstacles")
        self.declare_parameter("drivable_topic", "/perception/drivable")
        self.declare_parameter("grid_resolution", 0.2)   # m / cell
        self.declare_parameter("grid_size_x", 100)        # cells
        self.declare_parameter("grid_size_y", 100)
        self.declare_parameter("max_object_age_s", 1.0)
        self.declare_parameter("publish_rate_hz", 20.0)

        det_topic = self.get_parameter("det_topic").value
        lidar_topic = self.get_parameter("lidar_topic").value
        caminfo_topic = self.get_parameter("camera_info_topic").value
        objects_topic = self.get_parameter("objects_topic").value
        obstacles_topic = self.get_parameter("obstacles_topic").value
        drivable_topic = self.get_parameter("drivable_topic").value
        self.grid_resolution = float(self.get_parameter("grid_resolution").value)
        self.grid_size_x = int(self.get_parameter("grid_size_x").value)
        self.grid_size_y = int(self.get_parameter("grid_size_y").value)
        self.max_object_age = float(self.get_parameter("max_object_age_s").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)

        # --- QoS ----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers --------------------------------------------------
        self.sub_det = self.create_subscription(
            Detection2DArray, det_topic, self._on_detections, sensor_qos)
        self.sub_pc = self.create_subscription(
            PointCloud2, lidar_topic, self._on_pointcloud, sensor_qos)
        self.sub_caminfo = self.create_subscription(
            CameraInfo, caminfo_topic, self._on_camera_info, reliable_qos)

        # --- Publishers ---------------------------------------------------
        self.pub_objects = self.create_publisher(
            Detection3DArray, objects_topic, reliable_qos)
        if _NAV_MSGS_AVAILABLE:
            self.pub_obstacles = self.create_publisher(
                OccupancyGrid, obstacles_topic, reliable_qos)
        else:
            self.pub_obstacles = None
            self.get_logger().warning(
                "nav_msgs not available — skipping /perception/obstacles.")
        self.pub_drivable = self.create_publisher(
            Image, drivable_topic, sensor_qos)

        # --- Internal state ----------------------------------------------
        self.camera_info: Optional[CameraInfo] = None
        self.latest_detections: List[Dict] = []
        self.latest_pc: Optional[np.ndarray] = None  # (N, 3)
        self.fused_objects: Dict[int, FusedObject] = {}
        self._next_id = 1

        # Periodic publish timer.
        self._publish_timer = self.create_timer(
            1.0 / max(rate_hz, 1.0), self._publish)

        self.get_logger().info(
            f"PerceptionNode ready — sub={det_topic} + {lidar_topic}, "
            f"grid={self.grid_size_x}x{self.grid_size_y} "
            f"res={self.grid_resolution}m")

    # ------------------------------------------------------------------ #
    # Subscribers
    # ------------------------------------------------------------------ #
    def _on_camera_info(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics for 2D→3D projection."""
        self.camera_info = msg
        self.get_logger().info(
            f"CameraInfo: {msg.width}x{msg.height}, fx={msg.k[0]:.1f}",
            throttle_duration_sec=10.0)

    def _on_detections(self, msg: Detection2DArray) -> None:
        """Cache the latest 2D detections."""
        self.latest_detections = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            hyp = det.results[0] if det.results else None
            class_id = int(hyp.id) if hyp and hyp.id.isdigit() else -1
            score = float(hyp.score) if hyp else 0.0
            self.latest_detections.append({
                "class_id": class_id,
                "score": score,
                "bbox": (cx - w * 0.5, cy - h * 0.5,
                         cx + w * 0.5, cy + h * 0.5),
            })

    def _on_pointcloud(self, msg: PointCloud2) -> None:
        """Cache the latest LiDAR point cloud as an (N, 3) array."""
        if not _PC2_AVAILABLE:
            return
        try:
            pts = read_points_numpy(msg, field_names=("x", "y", "z"),
                                     skip_nans=True)
            # Filter out ground points (z < -0.3 in lidar frame) and
            # points beyond 50 m.
            mask = (pts[:, 2] > -0.3) & (np.linalg.norm(pts, axis=1) < 50.0)
            self.latest_pc = pts[mask].astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"PointCloud decode failed: {exc}",
                                       throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ #
    # Fusion + publish loop
    # ------------------------------------------------------------------ #
    def _publish(self) -> None:
        """Run one fusion + publish cycle."""
        now = time.time()
        # Age out stale objects.
        stale = [oid for oid, obj in self.fused_objects.items()
                  if now - obj.last_seen > self.max_object_age]
        for oid in stale:
            del self.fused_objects[oid]

        # Fuse: associate detections with LiDAR clusters.
        if self.latest_pc is not None and len(self.latest_detections) > 0:
            self._fuse()

        # Publish 3D object list.
        self._publish_objects(now)
        # Publish occupancy grid.
        if self.pub_obstacles is not None:
            self._publish_obstacles(now)

    def _fuse(self) -> None:
        """Associate 2D detections with LiDAR points to produce 3D objects.

        For simplicity, this implementation projects each LiDAR point into
        the camera image (using cached intrinsics) and assigns it to a
        detection whose 2D bbox contains the projected pixel. The centroid
        of the matched LiDAR points becomes the 3D object position.
        """
        if self.camera_info is None or self.latest_pc is None:
            return
        K = np.array(self.camera_info.k, dtype=np.float32).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Project LiDAR points into the image plane (assumes LiDAR is in
        # camera optical frame; in production, apply a TF transform here).
        pts = self.latest_pc
        valid = pts[:, 2] > 0.5  # in front of camera
        pts = pts[valid]
        if len(pts) == 0:
            return
        u = pts[:, 0] * fx / pts[:, 2] + cx
        v = pts[:, 1] * fy / pts[:, 2] + cy

        # For each detection, find the points whose projection falls in the bbox.
        for det in self.latest_detections:
            x1, y1, x2, y2 = det["bbox"]
            in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            if not np.any(in_box):
                continue
            cluster = pts[in_box]
            position = cluster.mean(axis=0)
            dims = cluster.max(axis=0) - cluster.min(axis=0)

            # Match to existing object by class + nearest neighbor.
            obj = self._find_match(det["class_id"], position)
            if obj is None:
                obj = FusedObject(self._next_id, det["class_id"], "")
                self._next_id += 1
                self.fused_objects[obj.track_id] = obj
                obj.position = position
            else:
                # Smooth position with EWMA.
                alpha = 0.4
                new_pos = alpha * position + (1 - alpha) * obj.position
                # Estimate velocity (m/s) — assumes ~constant publish rate.
                dt = max(1e-3, time.time() - obj.last_seen)
                obj.velocity = (new_pos - obj.position) / dt
                obj.position = new_pos
            obj.dimensions = dims
            obj.confidence = det["score"]
            obj.last_seen = time.time()
            obj.hits += 1

    def _find_match(
        self, class_id: int, position: np.ndarray, max_dist: float = 2.0
    ) -> Optional[FusedObject]:
        """Find the closest existing object of the same class within max_dist."""
        best: Optional[FusedObject] = None
        best_dist = max_dist
        for obj in self.fused_objects.values():
            if obj.class_id != class_id:
                continue
            d = float(np.linalg.norm(obj.position - position))
            if d < best_dist:
                best, best_dist = obj, d
        return best

    # ------------------------------------------------------------------ #
    # Publishers
    # ------------------------------------------------------------------ #
    def _publish_objects(self, now: float) -> None:
        """Publish the fused 3D object list."""
        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        for obj in self.fused_objects.values():
            det = Detection3D()
            det.header = msg.header
            bbox = BoundingBox3D()
            bbox.center.position.x = float(obj.position[0])
            bbox.center.position.y = float(obj.position[1])
            bbox.center.position.z = float(obj.position[2])
            # Identity orientation (orientation refinement left to the
            # tracking layer).
            bbox.center.orientation.w = 1.0
            bbox.size.x = float(max(obj.dimensions[0], 0.1))
            bbox.size.y = float(max(obj.dimensions[1], 0.1))
            bbox.size.z = float(max(obj.dimensions[2], 0.1))
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.score = obj.confidence
            hyp.id = str(obj.class_id)
            det.results.append(hyp)
            msg.detections.append(det)
        self.pub_objects.publish(msg)

    def _publish_obstacles(self, now: float) -> None:
        """Publish a 2D bird's-eye-view occupancy grid built from LiDAR."""
        if not _NAV_MSGS_AVAILABLE:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.info = MapMetaData()
        msg.info.resolution = self.grid_resolution
        msg.info.width = self.grid_size_x
        msg.info.height = self.grid_size_y
        # Origin — center the grid on the ego vehicle.
        msg.info.origin = Pose()
        msg.info.origin.position.x = -self.grid_size_x * self.grid_resolution * 0.5
        msg.info.origin.position.y = -self.grid_size_y * self.grid_resolution * 0.5
        msg.info.origin.orientation.w = 1.0

        grid = np.zeros(
            (self.grid_size_y, self.grid_size_x), dtype=np.int8)
        if self.latest_pc is not None and len(self.latest_pc) > 0:
            pts = self.latest_pc
            gx = ((pts[:, 0] - msg.info.origin.position.x)
                   / self.grid_resolution).astype(np.int32)
            gy = ((pts[:, 1] - msg.info.origin.position.y)
                   / self.grid_resolution).astype(np.int32)
            inside = (gx >= 0) & (gx < self.grid_size_x) & \
                     (gy >= 0) & (gy < self.grid_size_y)
            for x, y in zip(gx[inside], gy[inside]):
                grid[y, x] = 100  # occupied
        msg.data = grid.flatten().tolist()
        self.pub_obstacles.publish(msg)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(args=None) -> None:  # pragma: no cover
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
