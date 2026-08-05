#!/usr/bin/env python3
# =============================================================================
# File: ros2/ai_node.py
# Brief: AINode — ROS2 (rclpy) node that subscribes to an image topic, runs
#        TensorRT inference on each frame, and publishes detection results.
#        Designed for sensor-data QoS (best-effort, large queue) so that
#        slow inference does not back-pressure the camera driver.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""AI inference ROS2 node for the Jetson autonomous vehicle stack.

Topic interface
---------------
  Subscriptions:
    /camera/image_raw   [sensor_msgs/Image]    — best-effort, large queue.

  Publishers:
    /ai/detections      [vision_msgs/Detection2DArray]
    /ai/image_annotated [sensor_msgs/Image]    — optional visualization
    /ai/fps             [std_msgs/Float32]      — rolling FPS estimate

Parameters
----------
  engine_path        Path to the TensorRT .engine file.
  input_topic        Image topic to subscribe to.
  detection_topic    Topic to publish detection results on.
  publish_annotated  If True, also publish a visualized image.
  confidence         Detection confidence threshold.
  device_id          CUDA device index (always 0 on Jetson).
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

# Add the python/ directory to sys.path so we can import the AI modules
# shipped alongside this node when running outside of a colcon workspace.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import Float32, Header  # noqa: E402

# vision_msgs may not be available on minimal ROS2 installs; degrade gracefully.
try:
    from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose  # noqa: E402
    _VISION_MSGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VISION_MSGS_AVAILABLE = False
    Detection2DArray = None  # type: ignore

from inference import InferenceEngine  # noqa: E402
from object_detection import ObjectDetector  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _image_to_numpy(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a numpy array (H x W x C)."""
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "mono8":
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return arr.reshape(msg.height, msg.width)
    if msg.encoding in ("rgba8", "bgra8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return arr.reshape(msg.height, msg.width, 4)[:, :, :3]
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _numpy_to_image(arr: np.ndarray, encoding: str = "rgb8") -> Image:
    """Convert a numpy array to a sensor_msgs/Image."""
    msg = Image()
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.step = arr.shape[1] * (3 if arr.ndim == 3 else 1)
    msg.data = arr.tobytes()
    return msg


# -----------------------------------------------------------------------------
# AINode
# -----------------------------------------------------------------------------
class AINode(Node):
    """ROS2 node that runs TensorRT inference on incoming images."""

    def __init__(self) -> None:
        super().__init__("ai_node")

        # --- Parameters ---------------------------------------------------
        self.declare_parameter("engine_path", "/models/yolov5s.engine")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("detection_topic", "/ai/detections")
        self.declare_parameter("annotated_topic", "/ai/image_annotated")
        self.declare_parameter("fps_topic", "/ai/fps")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("confidence", 0.45)
        self.declare_parameter("device_id", 0)
        self.declare_parameter("class_names", "/models/coco.names")
        self.declare_parameter("yolo_version", "v8")
        self.declare_parameter("max_queue_size", 4)

        engine_path = self.get_parameter("engine_path").value
        input_topic = self.get_parameter("input_topic").value
        det_topic = self.get_parameter("detection_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        fps_topic = self.get_parameter("fps_topic").value
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.device_id = int(self.get_parameter("device_id").value)
        class_names = self.get_parameter("class_names").value
        yolo_version = self.get_parameter("yolo_version").value
        max_queue = int(self.get_parameter("max_queue_size").value)

        # --- Load the engine ---------------------------------------------
        # Use CUDA context isolation so we don't conflict with other nodes
        # that might also touch the GPU.
        self.get_logger().info(f"Loading TensorRT engine: {engine_path}")
        try:
            self.engine = InferenceEngine(
                engine_path,
                max_batch_size=1,
                device_id=self.device_id,
            )
        except Exception as exc:
            self.get_logger().fatal(f"Failed to load engine: {exc}")
            raise

        # --- Detector post-processor -------------------------------------
        try:
            self.detector = ObjectDetector(
                class_names=class_names,
                input_shape=(640, 640),
                conf_threshold=self.confidence,
                yolo_version=yolo_version,
            )
        except Exception as exc:
            self.get_logger().fatal(f"Failed to init ObjectDetector: {exc}")
            raise

        # --- QoS for image stream ----------------------------------------
        # Best-effort + large queue is the standard pattern for sensor data.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=max_queue,
        )
        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Pub / Sub ---------------------------------------------------
        self.sub_image = self.create_subscription(
            Image, input_topic, self._on_image, image_qos)

        if _VISION_MSGS_AVAILABLE:
            self.pub_detections = self.create_publisher(
                Detection2DArray, det_topic, det_qos)
        else:
            self.pub_detections = None
            self.get_logger().warning(
                "vision_msgs not available — skipping /ai/detections publisher.")

        if self.publish_annotated:
            self.pub_annotated = self.create_publisher(
                Image, annotated_topic, image_qos)
        else:
            self.pub_annotated = None

        self.pub_fps = self.create_publisher(Float32, fps_topic, 10)

        # --- FPS tracking ------------------------------------------------
        self._frame_times: deque = deque(maxlen=30)
        self._fps_timer = self.create_timer(1.0, self._publish_fps)

        self.get_logger().info(
            f"AINode ready — sub={input_topic}, pub={det_topic}, "
            f"engine={self.engine.engine_hash}, "
            f"inputs={self.engine.input_names}, "
            f"outputs={self.engine.output_names}")

    # ------------------------------------------------------------------ #
    # Image callback
    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image) -> None:
        """Run inference on one image message."""
        try:
            image = _image_to_numpy(msg)
        except ValueError as exc:
            self.get_logger().warning(f"Skipping image: {exc}")
            return

        # Preprocess (letterbox + normalize + NCHW).
        try:
            tensor, scale, pad = self.detector.preprocess(
                image, input_shape=(640, 640), bgr_to_rgb=(msg.encoding == "bgr8"))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"Preprocess failed: {exc}")
            return

        # Inference.
        try:
            input_name = self.engine.input_names[0]
            self.engine.infer_async({input_name: tensor})
            result = self.engine.sync()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"Inference failed: {exc}")
            return

        # Pick the first output tensor (YOLOv5/v8 single-output case).
        output_name = self.engine.output_names[0]
        raw_output = result.outputs[output_name]

        # Post-process.
        det_result = self.detector.detect(
            raw_output, image_shape=image.shape[:2],
            scale=scale, pad=pad)

        # Publish detections.
        if self.pub_detections is not None:
            det_msg = self._to_detection_msg(det_result, msg.header)
            self.pub_detections.publish(det_msg)

        # Publish annotated image (optional).
        if self.pub_annotated is not None:
            try:
                annotated = self.detector.draw(image, det_result)
                self.pub_annotated.publish(
                    _numpy_to_image(annotated, encoding="rgb8"))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Draw failed: {exc}")

        # FPS bookkeeping.
        self._frame_times.append(time.perf_counter())

    def _to_detection_msg(self, det_result, header: Header):
        """Convert a DetectionResult into a vision_msgs Detection2DArray."""
        if not _VISION_MSGS_AVAILABLE:
            return None
        msg = Detection2DArray()
        msg.header = header
        for d in det_result.detections:
            det = Detection2D()
            det.header = header
            x1, y1, x2, y2 = d.bbox
            det.bbox.center.position.x = float((x1 + x2) * 0.5)
            det.bbox.center.position.y = float((y1 + y2) * 0.5)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.score = d.confidence
            hyp.id = str(d.class_id)
            det.results.append(hyp)
            msg.detections.append(det)
        return msg

    # ------------------------------------------------------------------ #
    # FPS publishing
    # ------------------------------------------------------------------ #
    def _publish_fps(self) -> None:
        """Compute rolling FPS and publish it on /ai/fps."""
        if len(self._frame_times) < 2:
            return
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return
        fps = (len(self._frame_times) - 1) / span
        msg = Float32()
        msg.data = float(fps)
        self.pub_fps.publish(msg)
        self.get_logger().info(f"FPS: {fps:.1f}", throttle_duration_sec=5.0)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(args=None) -> None:  # pragma: no cover
    rclpy.init(args=args)
    node = AINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
