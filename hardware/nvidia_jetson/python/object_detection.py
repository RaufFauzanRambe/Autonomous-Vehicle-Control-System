#!/usr/bin/env python3
# =============================================================================
# File: python/object_detection.py
# Brief: ObjectDetector — YOLOv5 / YOLOv8 post-processing with NMS, label
#        mapping, and OpenCV bbox drawing. Designed to consume the raw
#        output tensor from python/inference.py::InferenceEngine.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Object detection post-processing for the Jetson autonomous vehicle stack.

This module is detector-agnostic for the YOLO family (YOLOv5, YOLOv6,
YOLOv7, YOLOv8, YOLO-NAS) and supports both the *single-tensor* output
layout (YOLOv5/v7: `[B, A*(5+C), H, W]`) and the *multi-tensor* layout
(YOLOv8: `[B, 4+C, N]` with NMS done in Python).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class Detection:
    """A single object detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    mask: Optional[np.ndarray] = None  # optional segmentation mask

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def to_dict(self) -> Dict[str, Union[int, float, str, List[int]]]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": list(self.bbox),
        }


@dataclass
class DetectionResult:
    """Per-image detection result."""

    detections: List[Detection] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)  # (H, W)

    @property
    def num_objects(self) -> int:
        return len(self.detections)

    def filter_by_class(self, class_ids: Iterable[int]) -> "DetectionResult":
        """Return a new result containing only the given class IDs."""
        ids = set(class_ids)
        return DetectionResult(
            detections=[d for d in self.detections if d.class_id in ids],
            image_shape=self.image_shape)

    def to_list(self) -> List[Dict[str, Union[int, float, str, List[int]]]]:
        return [d.to_dict() for d in self.detections]


# -----------------------------------------------------------------------------
# ObjectDetector
# -----------------------------------------------------------------------------
class ObjectDetector:
    """Post-processes YOLOv5/v8 output tensors into detection lists.

    Args:
        class_names: List of class name strings (index == class_id) or path
            to a JSON / text file containing one class name per line.
        input_shape: The (H, W) the network was trained on (e.g. (640, 640)).
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS suppression.
        max_detections: Maximum number of detections to return per image.
        agnostic_nms: If True, NMS is class-agnostic (faster, slight recall loss).
        yolo_version: One of {"v5", "v7", "v8", "v6", "nas"}.
        filter_classes: Optional iterable of class IDs to keep (others discarded).
    """

    def __init__(
        self,
        class_names: Union[List[str], str, Path],
        input_shape: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        max_detections: int = 300,
        agnostic_nms: bool = False,
        yolo_version: str = "v8",
        filter_classes: Optional[Iterable[int]] = None,
    ) -> None:
        self.class_names = self._load_class_names(class_names)
        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.agnostic_nms = agnostic_nms
        self.yolo_version = yolo_version.lower()
        self.filter_classes = set(filter_classes) if filter_classes else None

        if self.yolo_version not in {"v5", "v6", "v7", "v8", "nas"}:
            raise ValueError(
                f"Unsupported yolo_version: {yolo_version}")

        # Per-class colors (stable across runs).
        rng = np.random.default_rng(seed=42)
        self._colors = rng.integers(0, 255, size=(len(self.class_names), 3),
                                     dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(
        self,
        raw_output: np.ndarray,
        image_shape: Tuple[int, int],
        scale: Optional[float] = None,
        pad: Optional[Tuple[int, int]] = None,
    ) -> DetectionResult:
        """Post-process raw network output into a :class:`DetectionResult`.

        Args:
            raw_output: The raw output tensor from TensorRT (numpy array).
                For YOLOv5/v7 the shape is ``(B, N, 5+C)`` where N is the
                number of anchor predictions; for YOLOv8 the shape is
                ``(B, 4+C, N)`` and is transposed internally.
            image_shape: Original image shape (H, W) before resizing.
            scale: Optional resize scale factor (used for letterbox padding).
            pad: Optional (pad_x, pad_y) offsets from letterbox preprocessing.

        Returns:
            :class:`DetectionResult` with detections in original-image coords.
        """
        # Ensure batch dim is squeezed (we handle one image at a time).
        if raw_output.ndim == 3:
            raw_output = raw_output[0]
        if raw_output.ndim != 2:
            raise ValueError(
                f"Expected 2D output, got shape {raw_output.shape}")

        # YOLOv8 uses a transposed layout: (4+C, N) -> (N, 4+C).
        if self.yolo_version in {"v8", "nas"} and raw_output.shape[0] < raw_output.shape[1]:
            raw_output = raw_output.T  # (N, 4+C)

        # Convert (cx, cy, w, h, conf_per_class...) → list of candidate boxes.
        boxes, scores, class_ids = self._decode(raw_output)
        if boxes.size == 0:
            return DetectionResult(detections=[], image_shape=image_shape)

        # NMS (per-class or class-agnostic).
        keep = self._nms(
            boxes, scores, class_ids,
            iou_threshold=self.iou_threshold,
            agnostic=self.agnostic_nms)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Sort by score, keep top-K.
        order = np.argsort(-scores)[: self.max_detections]
        boxes, scores, class_ids = boxes[order], scores[order], class_ids[order]

        # Rescale to original image coords (undo letterbox).
        if scale is not None and pad is not None:
            boxes[:, 0::2] = (boxes[:, 0::2] - pad[0]) / scale
            boxes[:, 1::2] = (boxes[:, 1::2] - pad[1]) / scale
        boxes = self._clip_boxes(boxes, image_shape)

        # Apply class filter.
        if self.filter_classes is not None:
            mask = np.array(
                [cid in self.filter_classes for cid in class_ids],
                dtype=bool)
            boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        detections = [
            Detection(
                class_id=int(cid),
                class_name=self.class_names[int(cid)]
                    if int(cid) < len(self.class_names) else str(cid),
                confidence=float(s),
                bbox=(int(x1), int(y1), int(x2), int(y2)))
            for x1, y1, x2, y2, cid, s in zip(
                boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
                class_ids, scores)
        ]
        return DetectionResult(detections=detections, image_shape=image_shape)

    # ------------------------------------------------------------------ #
    # Decoders
    # ------------------------------------------------------------------ #
    def _decode(
        self, raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw network output into boxes, scores, class IDs.

        Returns:
            boxes:    ``(N, 4)`` in xyxy format (input-resolution coords).
            scores:   ``(N,)``  max class probability × objectness.
            class_ids:``(N,)``  argmax class.
        """
        if self.yolo_version in {"v8", "nas"}:
            return self._decode_v8(raw)
        return self._decode_v5(raw)

    def _decode_v5(
        self, raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """YOLOv5 / v7 layout: (N, 5+C) — cx, cy, w, h, obj, c1, c2, ..."""
        nc = raw.shape[1] - 5
        xywh = raw[:, :4]
        obj = raw[:, 4]
        cls = raw[:, 5:]
        scores = obj[:, None] * cls  # (N, C)
        if scores.size == 0:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )
        class_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(scores)), class_ids]

        # Confidence threshold.
        keep = max_scores > self.conf_threshold
        xywh, max_scores, class_ids = xywh[keep], max_scores[keep], class_ids[keep]

        # xywh -> xyxy
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
        return (
            boxes.astype(np.float32),
            max_scores.astype(np.float32),
            class_ids.astype(np.int32),
        )

    def _decode_v8(
        self, raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """YOLOv8 / NAS layout: (N, 4+C) — cx, cy, w, h, c1, c2, ... (no obj)."""
        nc = raw.shape[1] - 4
        xywh = raw[:, :4]
        cls = raw[:, 4:]
        if cls.size == 0:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )
        class_ids = np.argmax(cls, axis=1)
        max_scores = cls[np.arange(len(cls)), class_ids]

        keep = max_scores > self.conf_threshold
        xywh, max_scores, class_ids = (
            xywh[keep], max_scores[keep], class_ids[keep])

        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5
        return (
            boxes.astype(np.float32),
            max_scores.astype(np.float32),
            class_ids.astype(np.int32),
        )

    # ------------------------------------------------------------------ #
    # NMS (pure-numpy, no torchvision dependency for portability)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float = 0.5,
        agnostic: bool = False,
    ) -> np.ndarray:
        """Greedy non-maximum suppression returning the indices to keep."""
        if boxes.size == 0:
            return np.array([], dtype=np.int32)
        order = np.argsort(-scores)
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            ious = ObjectDetector._iou(boxes[i], boxes[order[1:]])
            if agnostic:
                mask = ious < iou_threshold
            else:
                same_class = class_ids[i] == class_ids[order[1:]]
                mask = (ious < iou_threshold) | ~same_class
            order = order[1:][mask]
        return np.array(keep, dtype=np.int32)

    @staticmethod
    def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
        """IoU between one xyxy box and an array of xyxy boxes."""
        x1 = np.maximum(box[0], others[:, 0])
        y1 = np.maximum(box[1], others[:, 1])
        x2 = np.minimum(box[2], others[:, 2])
        y2 = np.minimum(box[3], others[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        union = area_box + area_others - inter
        return np.where(union > 0, inter / union, 0.0)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clip_boxes(
        boxes: np.ndarray, shape: Tuple[int, int]
    ) -> np.ndarray:
        """Clip boxes to image bounds ``(H, W)``."""
        if boxes.size == 0:
            return boxes
        h, w = shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _load_class_names(
        spec: Union[List[str], str, Path]
    ) -> List[str]:
        """Load class names from a list, a JSON array, or a text file."""
        if isinstance(spec, list):
            return [str(s) for s in spec]
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Class names file not found: {path}")
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return [str(s) for s in json.load(f)]
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # ------------------------------------------------------------------ #
    # Preprocessing helpers (CPU, used as fallback when CUDA kernels
    # from cuda/image_processing.cu are unavailable).
    # ------------------------------------------------------------------ #
    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize an image to fit a target shape with letterbox padding.

        Returns:
            (padded_image, scale, (pad_x, pad_y))
        """
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for letterbox().")
        h, w = image.shape[:2]
        nh, nw = new_shape
        scale = min(nw / w, nh / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h),
                              interpolation=cv2.INTER_LINEAR)
        pad_x = (nw - new_w) // 2
        pad_y = (nh - new_h) // 2
        out = np.full((nh, nw, *image.shape[2:]), color,
                       dtype=image.dtype)
        out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return out, scale, (pad_x, pad_y)

    @staticmethod
    def preprocess(
        image: np.ndarray,
        input_shape: Tuple[int, int] = (640, 640),
        dtype: np.dtype = np.float32,
        bgr_to_rgb: bool = True,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Full YOLO preprocessing: letterbox + normalize + NCHW."""
        if bgr_to_rgb and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, scale, pad = ObjectDetector.letterbox(image, input_shape)
        # Scale to [0, 1] (YOLOv5/v8 default).
        tensor = padded.astype(dtype, copy=False) / 255.0
        # HWC -> CHW.
        tensor = tensor.transpose(2, 0, 1) if tensor.ndim == 3 else tensor
        # Add batch dim.
        tensor = tensor[None, ...]
        return np.ascontiguousarray(tensor), scale, pad

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult,
        thickness: int = 2,
        font_scale: float = 0.5,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Draw detection boxes on an image (returns a copy)."""
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for draw().")
        out = image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = self._colors[det.class_id % len(self._colors)].tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            if draw_labels:
                label = f"{det.class_name} {det.confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y_label = max(y1, th + 4)
                cv2.rectangle(
                    out, (x1, y_label - th - 4),
                    (x1 + tw + 4, y_label + baseline), color, -1)
                cv2.putText(out, label, (x1 + 2, y_label - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), 1, cv2.LINE_AA)
        return out


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="ObjectDetector smoke test.")
    parser.add_argument("--classes", default="coco.names",
                        help="Class names file")
    parser.add_argument("--output", default="dummy_output.npy",
                        help="Path to a numpy file with raw YOLO output")
    args = parser.parse_args()

    det = ObjectDetector(
        class_names=args.classes,
        input_shape=(640, 640),
        conf_threshold=0.45,
        yolo_version="v8",
    )
    if os.path.exists(args.output):
        raw = np.load(args.output)
        res = det.detect(raw, image_shape=(720, 1280))
        print(f"Got {res.num_objects} detections.")
        for d in res.detections[:10]:
            print(f"  {d.class_name} ({d.confidence:.2f}) @ {d.bbox}")
    else:
        # Synthetic test: 3 dummy predictions.
        raw = np.array([
            [320, 320, 100, 100, 0.1, 0.9, 0.0],
            [321, 321, 100, 100, 0.1, 0.85, 0.0],
            [800, 400, 200, 150, 0.05, 0.0, 0.95],
        ], dtype=np.float32)
        res = det.detect(raw, image_shape=(720, 1280))
        print(f"Got {res.num_objects} detections from synthetic data.")
