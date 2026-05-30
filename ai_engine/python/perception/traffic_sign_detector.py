"""
Traffic Sign Detector
=====================

Specialized detector for traffic signs in autonomous driving scenarios.
Built on top of the YOLOv8 architecture with traffic-sign-specific
enhancements including:

- **8 traffic sign categories**: speed limit, stop, yield, no entry,
  warning, mandatory, prohibitory, and other.
- **ROI filtering**: Only detect signs within a predefined region of
  interest (upper portion of the image) to reduce false positives.
- **Size-based filtering**: Traffic signs must fall within a realistic
  pixel-size range given the camera geometry.
- **Shape-aware NMS**: Exploits the fact that signs are typically
  isolated and rarely heavily occluded.

Model zoo (trained on GTSDB + TT100K merged dataset):

    - ``traffic_sign_yolov8n.pt``   – nano,   3.2 MB,  142 FPS
    - ``traffic_sign_yolov8s.pt``   – small,  11.2 MB,  98 FPS
    - ``traffic_sign_yolov8m.pt``   – medium, 25.9 MB,  62 FPS

Usage::

    from object_detection.traffic_sign_detector import TrafficSignDetector

    config = {
        "model_path": "traffic_sign_yolov8s.pt",
        "device": "cuda:0",
        "conf_threshold": 0.3,
        "roi_top_ratio": 0.0,
        "roi_bottom_ratio": 0.7,
    }
    detector = TrafficSignDetector(config)
    results = detector.detect_image(image)
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .detection_utils import (
    BBox,
    Detection,
    DetectionResult,
    class_aware_nms,
    clip_boxes,
    filter_small_boxes,
    nms,
    scale_boxes,
    xywh_to_xyxy,
)
from .object_detector import ObjectDetector, register_detector
from .postprocessing import filter_detections_by_class, refine_boxes
from .preprocessing import LetterBoxPreprocessor

logger = logging.getLogger(__name__)


# ===================================================================
# Traffic Sign Categories
# ===================================================================


class TrafficSignCategory(IntEnum):
    """8-category traffic sign taxonomy used by the detector.

    The taxonomy groups signs by functional meaning, balancing granularity
    with detection reliability.  Fine-grained sub-classification is left
    to a downstream classifier.
    """

    SPEED_LIMIT = 0       # Speed limit signs (20, 30, 40, 50, 60, 70, 80, 100, 120)
    STOP = 1              # Stop signs (octagonal)
    YIELD = 2             # Yield / give-way signs (inverted triangle)
    NO_ENTRY = 3          # No entry signs (red circle, white horizontal bar)
    WARNING = 4           # General warning / danger signs (triangles)
    MANDATORY = 5         # Mandatory / blue-circle signs (go straight, turn, etc.)
    PROHIBITORY = 6       # Other prohibitory (no overtaking, no parking, etc.)
    OTHER = 7             # Supplementary / information / other


# Human-readable names
SIGN_CATEGORY_NAMES: Dict[int, str] = {
    TrafficSignCategory.SPEED_LIMIT: "speed_limit",
    TrafficSignCategory.STOP: "stop",
    TrafficSignCategory.YIELD: "yield",
    TrafficSignCategory.NO_ENTRY: "no_entry",
    TrafficSignCategory.WARNING: "warning",
    TrafficSignCategory.MANDATORY: "mandatory",
    TrafficSignCategory.PROHIBITORY: "prohibitory",
    TrafficSignCategory.OTHER: "other",
}

# Typical sign colors in BGR for visualization
SIGN_CATEGORY_COLORS: Dict[int, Tuple[int, int, int]] = {
    TrafficSignCategory.SPEED_LIMIT: (0, 0, 255),       # red
    TrafficSignCategory.STOP: (0, 0, 200),              # dark red
    TrafficSignCategory.YIELD: (0, 255, 255),           # yellow
    TrafficSignCategory.NO_ENTRY: (0, 0, 180),          # maroon
    TrafficSignCategory.WARNING: (0, 200, 255),         # orange
    TrafficSignCategory.MANDATORY: (255, 100, 0),       # blue
    TrafficSignCategory.PROHIBITORY: (0, 100, 200),     # dark blue-red
    TrafficSignCategory.OTHER: (200, 200, 200),         # grey
}

# Realistic pixel-size bounds for signs given typical automotive cameras
# (width range in pixels at 640×480)
SIGN_SIZE_BOUNDS: Dict[int, Tuple[float, float]] = {
    TrafficSignCategory.SPEED_LIMIT: (12, 300),
    TrafficSignCategory.STOP: (15, 350),
    TrafficSignCategory.YIELD: (12, 280),
    TrafficSignCategory.NO_ENTRY: (12, 280),
    TrafficSignCategory.WARNING: (12, 280),
    TrafficSignCategory.MANDATORY: (10, 250),
    TrafficSignCategory.PROHIBITORY: (10, 250),
    TrafficSignCategory.OTHER: (8, 300),
}


# ===================================================================
# ROI helpers
# ===================================================================


def compute_roi_mask(
    image_shape: Tuple[int, int],
    top_ratio: float = 0.0,
    bottom_ratio: float = 0.7,
    side_margin_ratio: float = 0.05,
) -> np.ndarray:
    """Build a binary mask defining the region of interest for sign search.

    Signs are typically found in the upper 70 % of the image and away
    from the extreme edges.

    Parameters
    ----------
    image_shape : tuple
        ``(H, W)``.
    top_ratio : float
        Fraction from top that is always included (0.0 = top row).
    bottom_ratio : float
        Fraction of image height that defines the bottom boundary (0.7).
    side_margin_ratio : float
        Fraction of width to exclude on each side.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 mask (255 inside ROI, 0 outside).
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    x1 = int(w * side_margin_ratio)
    x2 = w - int(w * side_margin_ratio)

    mask[y1:y2, x1:x2] = 255
    return mask


def bbox_in_roi(
    bbox: BBox,
    image_shape: Tuple[int, int],
    top_ratio: float = 0.0,
    bottom_ratio: float = 0.7,
    min_overlap: float = 0.5,
) -> bool:
    """Check whether a bounding box overlaps sufficiently with the ROI.

    Parameters
    ----------
    min_overlap : float
        Minimum IoU between the box and the ROI region.
    """
    h, w = image_shape
    roi = BBox(
        x1=w * 0.05, y1=h * top_ratio,
        x2=w * 0.95, y2=h * bottom_ratio,
    )
    return bbox.iou(roi) >= min_overlap or roi.iou(bbox) >= min_overlap


# ===================================================================
# TrafficSignDetector
# ===================================================================


@register_detector("traffic_sign")
class TrafficSignDetector(ObjectDetector):
    """YOLOv8-based traffic sign detector with ROI and size filtering.

    Config keys (in addition to base ``ObjectDetector`` keys):

        - ``roi_top_ratio`` (float): Top of ROI as fraction of height.
        - ``roi_bottom_ratio`` (float): Bottom of ROI as fraction of height.
        - ``min_sign_area`` (int): Minimum sign area in px².
        - ``max_sign_area`` (int): Maximum sign area in px².
        - ``use_roi_filter`` (bool): Enable ROI-based pre-filtering.
        - ``use_size_filter`` (bool): Enable size-based filtering.
        - ``nms_per_category`` (bool): Run NMS per sign category.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Override class names with traffic sign categories
        config.setdefault("class_names", [
            SIGN_CATEGORY_NAMES[i] for i in range(8)
        ])
        config.setdefault("conf_threshold", 0.3)
        config.setdefault("iou_threshold", 0.4)

        super().__init__(config)

        # ROI parameters
        self.roi_top_ratio: float = config.get("roi_top_ratio", 0.0)
        self.roi_bottom_ratio: float = config.get("roi_bottom_ratio", 0.7)
        self.use_roi_filter: bool = config.get("use_roi_filter", True)

        # Size filtering
        self.min_sign_area: int = config.get("min_sign_area", 100)
        self.max_sign_area: int = config.get("max_sign_area", 90000)
        self.use_size_filter: bool = config.get("use_size_filter", True)

        # NMS
        self.nms_per_category: bool = config.get("nms_per_category", True)

        # Preprocessor
        self._preprocessor = LetterBoxPreprocessor(
            input_size=self.input_size,
            normalize=True,
            mean=(0.0, 0.0, 0.0),
            std=(255.0, 255.0, 255.0),
        )

        # Internal YOLOv8 session
        self._yolo_model: Any = None
        self._onnx_session: Any = None
        self._num_classes = 8

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        suffix = model_path.suffix.lower()
        if suffix == ".pt":
            self._load_ultralytics(model_path)
        elif suffix == ".onnx":
            self._load_onnx(model_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        self._is_loaded = True

    def _load_ultralytics(self, model_path: Path) -> None:
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(model_path))
            logger.info("Loaded Ultralytics traffic-sign model from %s", model_path)
        except ImportError:
            raise ImportError("ultralytics package required for .pt models")

    def _load_onnx(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required for .onnx models")

        providers = ["CPUExecutionProvider"]
        if "cuda" in self.device.lower():
            providers.insert(0, "CUDAExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._onnx_session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers,
        )
        logger.info("Loaded ONNX traffic-sign model from %s", model_path)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._preprocessor(image)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, tensor: np.ndarray) -> np.ndarray:
        if self._yolo_model is not None:
            return self._infer_ultralytics_raw(tensor)
        elif self._onnx_session is not None:
            return self._infer_onnx(tensor)
        else:
            raise RuntimeError("No model loaded")

    def _infer_ultralytics_raw(self, tensor: np.ndarray) -> np.ndarray:
        img = (tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        results = self._yolo_model.predict(
            img, device=self.device, verbose=False,
            conf=self.conf_threshold, iou=self.iou_threshold,
        )
        if results and hasattr(results[0], "boxes") and results[0].boxes.data.numel() > 0:
            import torch
            data = results[0].boxes.data.cpu().numpy()
            n = data.shape[0]
            c = self._num_classes
            raw = np.zeros((1, 4 + c, n), dtype=np.float32)
            raw[0, 0, :] = (data[:, 0] + data[:, 2]) / 2
            raw[0, 1, :] = (data[:, 1] + data[:, 3]) / 2
            raw[0, 2, :] = data[:, 2] - data[:, 0]
            raw[0, 3, :] = data[:, 3] - data[:, 1]
            for i in range(n):
                cls_id = int(data[i, 5])
                if cls_id < c:
                    raw[0, 4 + cls_id, i] = data[i, 4]
            return raw
        return np.zeros((1, 4 + self._num_classes, 0), dtype=np.float32)

    def _infer_onnx(self, tensor: np.ndarray) -> np.ndarray:
        input_name = self._onnx_session.get_inputs()[0].name
        return self._onnx_session.run(None, {input_name: tensor})[0]

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------

    def postprocess(
        self,
        raw_output: np.ndarray,
        meta: Dict[str, Any],
    ) -> DetectionResult:
        orig_shape = meta.get("orig_shape", self.input_size)

        # Decode raw output
        if raw_output.ndim == 3:
            raw_output = raw_output[0]
        box_preds = raw_output[:4, :].T
        cls_preds = raw_output[4:, :].T

        cls_scores = 1.0 / (1.0 + np.exp(-cls_preds))
        scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)

        mask = scores >= self.conf_threshold
        boxes_xyxy = xywh_to_xyxy(box_preds[mask])
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes_xyxy.size == 0:
            return DetectionResult(model_name="traffic_sign", device=self.device)

        # Rescale
        boxes_xyxy = scale_boxes(boxes_xyxy, self.input_size, (orig_shape[0], orig_shape[1]))
        boxes_xyxy = clip_boxes(boxes_xyxy, orig_shape)

        # NMS
        if self.nms_per_category:
            keep = class_aware_nms(
                boxes_xyxy, scores, class_ids,
                iou_threshold=self.iou_threshold,
                max_detections=self.max_detections,
            )
        else:
            keep = nms(boxes_xyxy, scores, iou_threshold=self.iou_threshold)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Build detections with sign-specific filtering
        detections: List[Detection] = []
        for i in range(len(boxes_xyxy)):
            bbox = BBox(*boxes_xyxy[i].tolist())
            cat_id = int(class_ids[i])

            # ROI filter
            if self.use_roi_filter and not bbox_in_roi(
                bbox, orig_shape,
                top_ratio=self.roi_top_ratio,
                bottom_ratio=self.roi_bottom_ratio,
            ):
                continue

            # Size filter
            if self.use_size_filter:
                area = bbox.area
                if area < self.min_sign_area or area > self.max_sign_area:
                    continue
                # Category-specific size bounds
                bounds = SIGN_SIZE_BOUNDS.get(cat_id)
                if bounds:
                    w = bbox.width
                    if w < bounds[0] or w > bounds[1]:
                        continue

            det = Detection(
                bbox=bbox,
                class_id=cat_id,
                class_name=SIGN_CATEGORY_NAMES.get(cat_id, f"sign_{cat_id}"),
                confidence=float(scores[i]),
                attributes={
                    "category": SIGN_CATEGORY_NAMES.get(cat_id, "unknown"),
                    "color_bgr": SIGN_CATEGORY_COLORS.get(cat_id, (128, 128, 128)),
                },
            )
            detections.append(det)

        return DetectionResult(
            detections=detections,
            model_name="traffic_sign",
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Traffic-sign-specific API
    # ------------------------------------------------------------------

    def detect_signs(
        self,
        image: np.ndarray,
        categories: Optional[List[TrafficSignCategory]] = None,
    ) -> DetectionResult:
        """Detect traffic signs, optionally filtered by category.

        Parameters
        ----------
        image : np.ndarray
            BGR image.
        categories : list, optional
            Only return signs from these categories.

        Returns
        -------
        DetectionResult
        """
        result = self.detect_image(image)
        if categories is not None:
            cat_ids = {int(c) for c in categories}
            result.detections = [
                d for d in result.detections if d.class_id in cat_ids
            ]
        return result

    def get_sign_color(self, class_id: int) -> Tuple[int, int, int]:
        """Return the representative BGR color for a sign category."""
        return SIGN_CATEGORY_COLORS.get(class_id, (128, 128, 128))

    def estimate_sign_distance(
        self,
        bbox: BBox,
        real_height_m: float = 0.6,
        focal_length_px: float = 800.0,
    ) -> float:
        """Estimate distance to a sign using pinhole camera model.

        Parameters
        ----------
        bbox : BBox
            Detected bounding box.
        real_height_m : float
            Real-world sign height in metres (default 0.6 m).
        focal_length_px : float
            Camera focal length in pixels.

        Returns
        -------
        float
            Estimated distance in metres.
        """
        pixel_height = max(bbox.height, 1.0)
        return (real_height_m * focal_length_px) / pixel_height
