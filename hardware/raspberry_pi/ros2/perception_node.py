"""
File:        ros2/perception_node.py
Brief:       PerceptionNode — ROS 2 lifecycle node that subscribes to
             camera images and lidar scans, runs a simple CV-based
             obstacle detector, and publishes Detection2DArray +
             occupancy obstacle map.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    CvBridge = None  # type: ignore
    _CV_BRIDGE_AVAILABLE = False

try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    _OPENCV_AVAILABLE = False


# ----------------------------------------------------------------------
# Internal data classes
# ----------------------------------------------------------------------
@dataclass(slots=True)
class DetectedObject:
    """A detected obstacle in image space."""

    class_id: str
    confidence: float
    cx: float      # center x in pixels
    cy: float      # center y in pixels
    width: float   # bbox width in pixels
    height: float  # bbox height in pixels
    distance_m: Optional[float] = None


# ----------------------------------------------------------------------
# PerceptionNode
# ----------------------------------------------------------------------
class PerceptionNode(LifecycleNode):
    """ROS 2 lifecycle node that runs perception.

    Lifecycle states:
        UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED

    Subscriptions:
        /camera/image_raw  (sensor_msgs/Image)
        /lidar/scan        (sensor_msgs/LaserScan)

    Publishers:
        /perception/obstacles      (vision_msgs/Detection2DArray)
        /perception/obstacle_map   (nav_msgs/OccupancyGrid)
    """

    QOS_SENSOR = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=DurabilityPolicy.VOLATILE,
    )
    QOS_RELIABLE = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    def __init__(self, node_name: str = "perception") -> None:
        super().__init__(node_name)
        self._declare_parameters()
        self._bridge = CvBridge() if _CV_BRIDGE_AVAILABLE else None
        self._cb_group = ReentrantCallbackGroup()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_scan: Optional[LaserScan] = None
        self._image_lock = threading.RLock()
        self._scan_lock = threading.RLock()
        self._last_publish = 0.0
        self._detection_count = 0
        self._sub_image = None
        self._sub_scan = None
        self._pub_obstacles = None
        self._pub_map = None
        self._timer = None
        logger.info("PerceptionNode created (lifecycle)")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("lidar_topic",  "/lidar/scan")
        self.declare_parameter("detection_confidence", 0.5)
        self.declare_parameter("max_obstacle_distance_m", 8.0)
        self.declare_parameter("publish_rate_hz", 10)
        self.declare_parameter("grid_resolution_m", 0.10)
        self.declare_parameter("grid_width_m", 20.0)
        self.declare_parameter("grid_height_m", 20.0)

    # ------------------------------------------------------------------
    # Lifecycle: configure
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        logger.info("Configuring perception node")
        self._sub_image = self.create_subscription(
            Image,
            self.get_parameter("camera_topic").value,
            self._on_image,
            self.QOS_SENSOR,
            callback_group=self._cb_group,
        )
        self._sub_scan = self.create_subscription(
            LaserScan,
            self.get_parameter("lidar_topic").value,
            self._on_scan,
            self.QOS_SENSOR,
            callback_group=self._cb_group,
        )
        self._pub_obstacles = self.create_lifecycle_publisher(
            Detection2DArray, "obstacles", self.QOS_RELIABLE,
        )
        self._pub_map = self.create_lifecycle_publisher(
            OccupancyGrid, "obstacle_map", self.QOS_RELIABLE,
        )
        logger.info("Perception configured — subscriptions and publishers ready")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Lifecycle: activate
    # ------------------------------------------------------------------
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Activating perception node")
        rate_hz = int(self.get_parameter("publish_rate_hz").value)
        period = 1.0 / max(1, rate_hz)
        self._timer = self.create_timer(period, self._on_timer,
                                        callback_group=self._cb_group)
        return super().on_activate(state)

    # ------------------------------------------------------------------
    # Lifecycle: deactivate
    # ------------------------------------------------------------------
    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Deactivating perception node")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        return super().on_deactivate(state)

    # ------------------------------------------------------------------
    # Lifecycle: cleanup
    # ------------------------------------------------------------------
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        logger.info("Cleaning up perception node")
        self._destroy_subscriptions()
        self._destroy_publishers()
        self._latest_image = None
        self._latest_scan = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        logger.info("Shutting down perception node")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        if self._bridge is None:
            return
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._image_lock:
                self._latest_image = cv_img
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image conversion failed: {}", exc)

    def _on_scan(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._latest_scan = msg

    # ------------------------------------------------------------------
    # Periodic publish
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        now = time.monotonic()
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if now - self._last_publish < 1.0 / rate_hz:
            return
        self._last_publish = now

        # Detect obstacles from the latest frame
        with self._image_lock:
            img = self._latest_image
        detections = self._detect_obstacles_cv(img) if img is not None else []

        # Augment with lidar distance where possible
        with self._scan_lock:
            scan = self._latest_scan
        if scan is not None:
            for det in detections:
                det.distance_m = self._range_for_pixel(scan, det.cx, det.width)

        self._publish_detections(detections)
        if scan is not None:
            self._publish_obstacle_map(scan)

    # ------------------------------------------------------------------
    # CV-based obstacle detection (very simple — replace with YOLO/etc.)
    # ------------------------------------------------------------------
    def _detect_obstacles_cv(self, img: np.ndarray) -> List[DetectedObject]:
        """Detect obstacles using a simple color+contour filter.

        This is a placeholder; replace with a proper detector
        (YOLO via TensorRT, MobileNet-SSD, etc.) in production.
        """
        if not _OPENCV_AVAILABLE:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        confidence_threshold = float(
            self.get_parameter("detection_confidence").value
        )
        max_dist = float(self.get_parameter("max_obstacle_distance_m").value)
        detections: List[DetectedObject] = []
        h, w = img.shape[:2]
        min_area = (w * h) * 0.005   # at least 0.5% of frame
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Heuristic confidence: relative area
            confidence = min(1.0, area / (w * h))
            if confidence < confidence_threshold:
                continue
            detections.append(DetectedObject(
                class_id="obstacle",
                confidence=confidence,
                cx=x + bw / 2.0,
                cy=y + bh / 2.0,
                width=bw,
                height=bh,
                distance_m=None,
            ))
            # Cap to max_dist
            if max_dist <= 0:
                break
        self._detection_count += len(detections)
        return detections

    def _range_for_pixel(self, scan: LaserScan, cx: float,
                         bbox_width_px: float) -> Optional[float]:
        """Return the lidar range corresponding to the center of a detection.

        This is a placeholder that maps the pixel column to an azimuth.
        """
        # Assume 90° FOV camera centered on the lidar
        fov_rad = math.radians(90.0)
        normalized = (cx / float(self._image_width())) - 0.5   # -0.5..0.5
        angle = normalized * fov_rad
        # Map to scan index
        if scan.angle_max - scan.angle_min == 0:
            return None
        idx = int(((angle - scan.angle_min)
                   / (scan.angle_max - scan.angle_min))
                  * len(scan.ranges))
        if 0 <= idx < len(scan.ranges):
            r = scan.ranges[idx]
            if scan.range_min <= r <= scan.range_max:
                return float(r)
        return None

    def _image_width(self) -> int:
        with self._image_lock:
            if self._latest_image is None:
                return 640
            return int(self._latest_image.shape[1])

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_detections(self, detections: List[DetectedObject]) -> None:
        if self._pub_obstacles is None or not self._pub_obstacles.is_activated():
            return
        msg = Detection2DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        for det in detections:
            d = Detection2D()
            d.header = msg.header
            d.results.append(ObjectHypothesisWithPose())
            d.results[0].score = det.confidence
            d.results[0].id = det.class_id
            d.bbox.center.position.x = det.cx
            d.bbox.center.position.y = det.cy
            d.bbox.size_x = det.width
            d.bbox.size_y = det.height
            msg.detections.append(d)
        self._pub_obstacles.publish(msg)

    def _publish_obstacle_map(self, scan: LaserScan) -> None:
        if self._pub_map is None or not self._pub_map.is_activated():
            return
        resolution = float(self.get_parameter("grid_resolution_m").value)
        width_m = float(self.get_parameter("grid_width_m").value)
        height_m = float(self.get_parameter("grid_height_m").value)
        width = int(width_m / resolution)
        height = int(height_m / resolution)
        grid = np.full((height, width), -1, dtype=np.int8)  # -1 = unknown

        cx = width // 2
        cy = height // 2
        max_dist = float(self.get_parameter("max_obstacle_distance_m").value)
        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max) or r > max_dist:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            x = int(cx + r * math.cos(angle) / resolution)
            y = int(cy + r * math.sin(angle) / resolution)
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = 100   # occupied

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_link"
        msg.info.resolution = resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position = Point(x=-width_m / 2.0,
                                         y=-height_m / 2.0,
                                         z=0.0)
        msg.info.origin.orientation = Quaternion(w=1.0)
        msg.data = grid.flatten().tolist()
        self._pub_map.publish(msg)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    def _destroy_subscriptions(self) -> None:
        for sub in (self._sub_image, self._sub_scan):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_image = None
        self._sub_scan = None

    def _destroy_publishers(self) -> None:
        for pub in (self._pub_obstacles, self._pub_map):
            if pub is not None:
                self.destroy_publisher(pub)
        self._pub_obstacles = None
        self._pub_map = None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "detection_count": self._detection_count,
            "has_image": self._latest_image is not None,
            "has_scan": self._latest_scan is not None,
        }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        # Bring it to ACTIVE
        node.configure()
        node.activate()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("Perception node crashed: {}", exc)
    finally:
        node.deactivate()
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
