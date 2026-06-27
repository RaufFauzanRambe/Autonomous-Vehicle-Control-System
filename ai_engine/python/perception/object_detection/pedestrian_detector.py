"""
Pedestrian Detector
===================

Specialized pedestrian detection module for autonomous driving, built on
top of a YOLOv8 base detector with additional capabilities:

- **Pose estimation support**: Optionally attach a lightweight pose
  estimator (e.g. ViTPose-S) to obtain 17-keypoint skeletons for
  detected pedestrians, enabling gesture / intent recognition.
- **Distance estimation**: Monocular distance estimation using camera
  intrinsics and assumed pedestrian height (1.7 m average).
- **Tracking ID assignment**: Simple IoU-based tracker that assigns
  persistent track IDs across frames for trajectory analysis.

Usage::

    from object_detection.pedestrian_detector import PedestrianDetector

    config = {
        "model_path": "yolov8s-ped.pt",
        "device": "cuda:0",
        "conf_threshold": 0.3,
        "use_pose": True,
        "use_tracking": True,
        "camera_height": 1.5,
        "focal_length": 800.0,
    }
    detector = PedestrianDetector(config)
    results = detector.detect_image(image)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .detection_utils import (
    BBox,
    Detection,
    DetectionResult,
    compute_iou_matrix,
    nms,
    scale_boxes,
    clip_boxes,
)
from .object_detector import ObjectDetector, register_detector
from .preprocessing import LetterBoxPreprocessor

logger = logging.getLogger(__name__)


# ===================================================================
# Constants
# ===================================================================

# Average adult pedestrian height in metres (used for monocular distance)
DEFAULT_PEDESTRIAN_HEIGHT_M = 1.70
DEFAULT_CHILD_HEIGHT_M = 1.20

# COCO 17-keypoint layout for pose estimation
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connectivity for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# Pedestrian class IDs in the detection model
PEDESTRIAN_CLASS_ID = 0  # person in COCO


# ===================================================================
# Pose data structure
# ===================================================================


@dataclass
class Keypoint:
    """Single keypoint with position and confidence."""

    x: float
    y: float
    confidence: float = 0.0
    name: str = ""

    @property
    def is_valid(self) -> bool:
        return self.confidence > 0.3

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.confidence)


@dataclass
class Pose:
    """17-keypoint skeleton for a single pedestrian."""

    keypoints: List[Keypoint] = field(default_factory=list)

    def __post_init__(self):
        # Ensure we always have 17 keypoints
        while len(self.keypoints) < 17:
            self.keypoints.append(Keypoint(0, 0, 0.0, COCO_KEYPOINT_NAMES[len(self.keypoints)]))

    @classmethod
    def from_numpy(cls, kpts: np.ndarray) -> "Pose":
        """Create Pose from ``(17, 3)`` array ``[x, y, conf]``."""
        keypoints = []
        for i in range(min(17, kpts.shape[0])):
            keypoints.append(Keypoint(
                x=float(kpts[i, 0]),
                y=float(kpts[i, 1]),
                confidence=float(kpts[i, 2]),
                name=COCO_KEYPOINT_NAMES[i] if i < len(COCO_KEYPOINT_NAMES) else f"kpt_{i}",
            ))
        return cls(keypoints=keypoints)

    def to_numpy(self) -> np.ndarray:
        arr = np.zeros((17, 3), dtype=np.float32)
        for i, kp in enumerate(self.keypoints[:17]):
            arr[i] = [kp.x, kp.y, kp.confidence]
        return arr

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    @property
    def num_valid(self) -> int:
        return sum(1 for kp in self.keypoints if kp.is_valid)

    @property
    def head_center(self) -> Optional[Tuple[float, float]]:
        nose = self.get_keypoint("nose")
        if nose and nose.is_valid:
            return (nose.x, nose.y)
        return None

    @property
    def body_center(self) -> Optional[Tuple[float, float]]:
        ls = self.get_keypoint("left_shoulder")
        rs = self.get_keypoint("right_shoulder")
        if ls and rs and ls.is_valid and rs.is_valid:
            return ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        return None


# ===================================================================
# Simple IoU Tracker
# ===================================================================


class PedestrianTracker:
    """Simple IoU-based tracker for assigning persistent IDs to pedestrians.

    The tracker maintains a list of active tracks.  Each frame, detections
    are matched to existing tracks by maximising IoU.  Unmatched detections
    start new tracks; unmatched tracks are aged out after *max_age* frames.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a match.
    max_age : int
        Number of frames a track can survive without a match.
    min_hits : int
        Minimum consecutive matches before a track is considered confirmed.
    """

    _next_id: int = 1

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        # Active tracks: list of dicts
        self.tracks: List[Dict[str, Any]] = []
        self.frame_count: int = 0

    @dataclass
    class Track:
        track_id: int
        bbox: np.ndarray    # (4,) xyxy
        age: int = 0
        hits: int = 1
        last_seen: int = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Match detections to tracks and return track IDs.

        Parameters
        ----------
        detections : np.ndarray
            ``(N, 4)`` xyxy boxes for current frame.

        Returns
        -------
        np.ndarray
            ``(N,)`` integer track IDs (0 = unconfirmed / new track).
        """
        self.frame_count += 1
        num_det = len(detections)

        if num_det == 0:
            # Age all tracks
            for t in self.tracks:
                t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return np.array([], dtype=np.int64)

        # No existing tracks – create new ones
        if len(self.tracks) == 0:
            track_ids: List[int] = []
            for i in range(num_det):
                tid = PedestrianTracker._next_id
                PedestrianTracker._next_id += 1
                self.tracks.append({
                    "track_id": tid,
                    "bbox": detections[i],
                    "age": 0,
                    "hits": 1,
                    "last_seen": self.frame_count,
                })
                track_ids.append(tid if self.min_hits <= 1 else 0)
            return np.array(track_ids, dtype=np.int64)

        # Compute IoU matrix
        track_boxes = np.array([t["bbox"] for t in self.tracks], dtype=np.float32)
        iou_matrix = compute_iou_matrix(track_boxes, detections)  # (T, D)

        # Greedy matching
        assigned_tracks: set = set()
        assigned_dets: set = set()
        matches: List[Tuple[int, int]] = []

        # Sort by IoU descending
        flat_indices = np.argsort(-iou_matrix.ravel())
        for flat_idx in flat_indices:
            t_idx = int(flat_idx // num_det)
            d_idx = int(flat_idx % num_det)
            if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                break
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                continue
            matches.append((t_idx, d_idx))
            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        # Update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx]["bbox"] = detections[d_idx]
            self.tracks[t_idx]["age"] = 0
            self.tracks[t_idx]["hits"] += 1
            self.tracks[t_idx]["last_seen"] = self.frame_count

        # Age unmatched tracks
        for t_idx in range(len(self.tracks)):
            if t_idx not in assigned_tracks:
                self.tracks[t_idx]["age"] += 1

        # Create new tracks for unmatched detections
        for d_idx in range(num_det):
            if d_idx not in assigned_dets:
                tid = PedestrianTracker._next_id
                PedestrianTracker._next_id += 1
                self.tracks.append({
                    "track_id": tid,
                    "bbox": detections[d_idx],
                    "age": 0,
                    "hits": 1,
                    "last_seen": self.frame_count,
                })

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        # Build output track IDs
        result_ids = np.zeros(num_det, dtype=np.int64)

        # Map detections to track IDs via matches
        for t_idx, d_idx in matches:
            track = self.tracks  # might have been modified
            # Find the track with matching last_seen and bbox
            for t in self.tracks:
                if t["last_seen"] == self.frame_count and np.array_equal(t["bbox"], detections[d_idx]):
                    result_ids[d_idx] = t["track_id"] if t["hits"] >= self.min_hits else 0
                    break

        # New unmatched detections get 0 (unconfirmed)
        for d_idx in range(num_det):
            if d_idx not in assigned_dets:
                result_ids[d_idx] = 0

        return result_ids

    def reset(self) -> None:
        """Reset all tracks."""
        self.tracks.clear()
        self.frame_count = 0
        PedestrianTracker._next_id = 1


# ===================================================================
# PedestrianDetector
# ===================================================================


@register_detector("pedestrian")
class PedestrianDetector(ObjectDetector):
    """Pedestrian detector with pose estimation, distance, and tracking.

    Config keys (in addition to base keys):

        - ``use_pose`` (bool): Attach pose estimator (default False).
        - ``use_tracking`` (bool): Enable IoU tracker (default True).
        - ``camera_height`` (float): Camera height in metres (default 1.5).
        - ``focal_length`` (float): Focal length in pixels (default 800).
        - ``pedestrian_height_m`` (float): Assumed pedestrian height (1.7 m).
        - ``tracker_iou_threshold`` (float): IoU threshold for tracker.
        - ``tracker_max_age`` (int): Max frames before track expires.
        - ``min_box_height`` (float): Minimum bbox height in px.
        - ``max_box_height_ratio`` (float): Max bbox height / image height.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Override class names for pedestrian-only model
        config.setdefault("class_names", ["pedestrian"])
        config.setdefault("conf_threshold", 0.3)
        config.setdefault("iou_threshold", 0.4)

        super().__init__(config)

        # Pose
        self.use_pose: bool = config.get("use_pose", False)
        self._pose_model: Any = None

        # Tracking
        self.use_tracking: bool = config.get("use_tracking", True)
        self._tracker = PedestrianTracker(
            iou_threshold=config.get("tracker_iou_threshold", 0.3),
            max_age=config.get("tracker_max_age", 30),
            min_hits=config.get("tracker_min_hits", 3),
        )

        # Camera / distance
        self.camera_height: float = config.get("camera_height", 1.5)
        self.focal_length: float = config.get("focal_length", 800.0)
        self.pedestrian_height_m: float = config.get("pedestrian_height_m", DEFAULT_PEDESTRIAN_HEIGHT_M)

        # Box size filters
        self.min_box_height: float = config.get("min_box_height", 30.0)
        self.max_box_height_ratio: float = config.get("max_box_height_ratio", 0.9)

        # Preprocessor
        self._preprocessor = LetterBoxPreprocessor(
            input_size=self.input_size,
            normalize=True,
            mean=(0.0, 0.0, 0.0),
            std=(255.0, 255.0, 255.0),
        )

        # Model
        self._yolo_model: Any = None
        self._onnx_session: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_path.suffix.lower() == ".pt":
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(model_path))
        elif model_path.suffix.lower() == ".onnx":
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if "cuda" in self.device.lower():
                providers.insert(0, "CUDAExecutionProvider")
            self._onnx_session = ort.InferenceSession(
                str(model_path), providers=providers,
            )
        else:
            raise ValueError(f"Unsupported format: {model_path.suffix}")

        self._is_loaded = True

        # Optionally load pose model
        if self.use_pose:
            pose_path = config.get("pose_model_path") if (config := self.config).get("pose_model_path") else None
            if pose_path:
                self._load_pose_model(pose_path)
            else:
                logger.warning("use_pose=True but no pose_model_path provided; pose disabled")
                self.use_pose = False

    def _load_pose_model(self, path: str) -> None:
        """Load a lightweight pose estimation model."""
        try:
            from ultralytics import YOLO
            self._pose_model = YOLO(path)
            logger.info("Loaded pose model from %s", path)
        except Exception as e:
            logger.warning("Failed to load pose model: %s; disabling pose", e)
            self.use_pose = False

    # ------------------------------------------------------------------
    # Preprocess / detect / postprocess
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._preprocessor(image)

    def detect(self, tensor: np.ndarray) -> np.ndarray:
        if self._yolo_model is not None:
            img = (tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            results = self._yolo_model.predict(
                img, device=self.device, verbose=False,
                conf=self.conf_threshold, iou=self.iou_threshold,
                classes=[0],  # person only
            )
            if results and hasattr(results[0], "boxes") and results[0].boxes.data.numel() > 0:
                import torch
                data = results[0].boxes.data.cpu().numpy()
                n = data.shape[0]
                raw = np.zeros((1, 5, n), dtype=np.float32)  # 4 box + 1 conf
                raw[0, 0, :] = data[:, 0]
                raw[0, 1, :] = data[:, 1]
                raw[0, 2, :] = data[:, 2]
                raw[0, 3, :] = data[:, 3]
                raw[0, 4, :] = data[:, 4]
                return raw
            return np.zeros((1, 5, 0), dtype=np.float32)

        elif self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            return self._onnx_session.run(None, {input_name: tensor})[0]

        raise RuntimeError("No model loaded")

    def postprocess(
        self,
        raw_output: np.ndarray,
        meta: Dict[str, Any],
    ) -> DetectionResult:
        orig_shape = meta.get("orig_shape", self.input_shape) if hasattr(self, 'input_shape') else meta.get("orig_shape", self.input_size)

        # Decode – output is (1, 5, N) with [x1,y1,x2,y2,conf] or Ultralytics-style
        if raw_output.ndim == 3:
            raw_output = raw_output[0]

        if raw_output.shape[0] == 5:
            # Direct x1y1x2y2 + conf format
            boxes = raw_output[:4, :].T  # (N, 4)
            scores = raw_output[4, :]
            class_ids = np.zeros(len(scores), dtype=np.int64)  # all pedestrian
        else:
            # Standard YOLOv8 format (4 + C, N)
            box_preds = raw_output[:4, :].T
            cls_preds = raw_output[4:, :].T
            cls_scores = 1.0 / (1.0 + np.exp(-cls_preds))
            scores = cls_scores.max(axis=1)
            class_ids = cls_scores.argmax(axis=1)
            boxes = box_preds
            # Convert xcycwh to xyxy if needed
            if boxes[:, 2].mean() > 0 and (boxes[:, 0] - boxes[:, 2] / 2).mean() < 0:
                from .detection_utils import xywh_to_xyxy
                boxes = xywh_to_xyxy(boxes)

        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes.size == 0:
            return DetectionResult(model_name="pedestrian", device=self.device)

        # Rescale to original image space
        boxes = scale_boxes(boxes, self.input_size, (orig_shape[0], orig_shape[1]))
        boxes = clip_boxes(boxes, orig_shape)

        # NMS
        keep = nms(boxes, scores, iou_threshold=self.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Size filtering
        img_h = orig_shape[0]
        valid_mask = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            box_h = boxes[i, 3] - boxes[i, 1]
            if box_h < self.min_box_height or box_h > img_h * self.max_box_height_ratio:
                valid_mask[i] = False
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]

        # Tracking
        track_ids: Optional[np.ndarray] = None
        if self.use_tracking:
            track_ids = self._tracker.update(boxes)

        # Run pose estimation if enabled
        poses: List[Optional[Pose]] = []
        if self.use_pose and self._pose_model is not None:
            orig_image = meta.get("orig_image")
            if orig_image is not None:
                poses = self._estimate_poses(orig_image, boxes)
            else:
                poses = [None] * len(boxes)

        # Build detections
        detections: List[Detection] = []
        for i in range(len(boxes)):
            bbox = BBox(*boxes[i].tolist())
            distance = self._estimate_distance(bbox)
            velocity = None  # would need temporal data

            attrs: Dict[str, Any] = {}
            if track_ids is not None and i < len(track_ids):
                attrs["track_id"] = int(track_ids[i])
            if distance is not None:
                attrs["distance_m"] = round(distance, 2)
            if self.use_pose and i < len(poses) and poses[i] is not None:
                attrs["pose"] = poses[i]  # type: ignore[assignment]

            det = Detection(
                bbox=bbox,
                class_id=0,  # pedestrian
                class_name="pedestrian",
                confidence=float(scores[i]),
                track_id=int(track_ids[i]) if track_ids is not None and i < len(track_ids) and track_ids[i] > 0 else None,
                distance=distance,
                velocity=velocity,
                attributes=attrs,
            )
            detections.append(det)

        return DetectionResult(
            detections=detections,
            model_name="pedestrian",
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Distance estimation
    # ------------------------------------------------------------------

    def _estimate_distance(self, bbox: BBox) -> Optional[float]:
        """Monocular distance estimation using assumed pedestrian height.

        Uses the pinhole camera model:
            distance = (real_height × focal_length) / pixel_height
        """
        pixel_height = bbox.height
        if pixel_height < 1:
            return None
        distance = (self.pedestrian_height_m * self.focal_length) / pixel_height

        # Sanity check: limit to 0–100 m
        return min(max(distance, 0.5), 100.0)

    def _estimate_velocity(
        self,
        prev_bbox: Optional[BBox],
        curr_bbox: BBox,
        dt: float,
        distance: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        """Estimate pedestrian velocity from two consecutive bboxes.

        Returns (vx, vy) in m/s or None if insufficient data.
        """
        if prev_bbox is None or distance is None or dt <= 0:
            return None

        # Pixel displacement
        dx_px = curr_bbox.center[0] - prev_bbox.center[0]
        dy_px = curr_bbox.center[1] - prev_bbox.center[1]

        # Approximate metres per pixel at the estimated distance
        m_per_px = self.pedestrian_height_m / max(curr_bbox.height, 1.0)

        vx = dx_px * m_per_px / dt
        vy = dy_px * m_per_px / dt

        # Sanity: pedestrians rarely move faster than 10 m/s
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed > 10.0:
            return None

        return (vx, vy)

    # ------------------------------------------------------------------
    # Pose estimation
    # ------------------------------------------------------------------

    def _estimate_poses(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Optional[Pose]]:
        """Run pose estimation on cropped pedestrian regions."""
        if self._pose_model is None:
            return [None] * len(boxes)

        poses: List[Optional[Pose]] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            # Add margin around the person
            h, w = image.shape[:2]
            margin = int(max(x2 - x1, y2 - y1) * 0.15)
            cx1 = max(0, x1 - margin)
            cy1 = max(0, y1 - margin)
            cx2 = min(w, x2 + margin)
            cy2 = min(h, y2 + margin)

            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                poses.append(None)
                continue

            try:
                results = self._pose_model.predict(crop, verbose=False)
                if results and hasattr(results[0], "keypoints"):
                    kpts = results[0].keypoints
                    if kpts is not None and kpts.data.numel() > 0:
                        import torch
                        kpt_data = kpts.data[0].cpu().numpy()  # (17, 3)
                        # Shift keypoints back to full image coordinates
                        kpt_data[:, 0] += cx1
                        kpt_data[:, 1] += cy1
                        poses.append(Pose.from_numpy(kpt_data))
                    else:
                        poses.append(None)
                else:
                    poses.append(None)
            except Exception as e:
                logger.debug("Pose estimation failed for detection %d: %s", i, e)
                poses.append(None)

        return poses

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def detect_pedestrians(
        self,
        image: np.ndarray,
        min_distance: float = 0.0,
        max_distance: float = 100.0,
    ) -> DetectionResult:
        """Detect pedestrians with distance filtering.

        Parameters
        ----------
        image : np.ndarray
            BGR image.
        min_distance, max_distance : float
            Distance range in metres.

        Returns
        -------
        DetectionResult
        """
        result = self.detect_image(image)
        result.detections = [
            d for d in result.detections
            if d.distance is not None and min_distance <= d.distance <= max_distance
        ]
        return result

    def get_pedestrian_count(self, result: DetectionResult) -> int:
        return len(result.detections)

    def reset_tracker(self) -> None:
        """Reset the pedestrian tracker state."""
        self._tracker.reset()
