"""
YOLOv8 Object Detector
=======================

Concrete implementation of :class:`ObjectDetector` using the YOLOv8
architecture.  Supports inference via:

1. **Ultralytics Python API** – the easiest path; loads ``.pt`` weights
   directly and handles preprocessing internally.
2. **ONNX Runtime** – for TensorRT-optimised or exported ONNX models.
3. **OpenVINO** – for Intel-edge deployment (optional).

The detector applies the full YOLOv8 post-processing pipeline:
    - Decode raw grid predictions → xyxy boxes + class scores
    - Confidence thresholding
    - Class-aware Non-Maximum Suppression
    - Optional ROI filtering and class whitelisting

Typical usage::

    from object_detection.yolov8_detector import YOLOv8Detector

    config = {
        "model_path": "yolov8n.pt",
        "device": "cuda:0",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "input_size": [640, 640],
    }
    detector = YOLOv8Detector(config)
    result = detector.detect_image(image)
"""

from __future__ import annotations

import logging
import time
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
    nms,
    scale_boxes,
    xywh_to_xyxy,
)
from .object_detector import ObjectDetector, register_detector
from .postprocessing import filter_detections_by_class, refine_boxes
from .preprocessing import LetterBoxPreprocessor

logger = logging.getLogger(__name__)


# ===================================================================
# YOLOv8 Decoder  –  converts raw model output to boxes + scores
# ===================================================================


class YOLOv8Decoder:
    """Decode YOLOv8 raw output tensor into candidate detections.

    YOLOv8 output shape: ``(1, 4 + num_classes, num_anchors)`` where
    the first 4 rows are ``[xc, yc, w, h]`` (centre format) and the
    remaining rows are per-class logits.

    Parameters
    ----------
    num_classes : int
        Number of detection classes.
    conf_threshold : float
        Minimum class-confidence to keep a detection.
    input_size : tuple
        ``(H, W)`` of the network input.
    strides : tuple
        Feature-map strides for the 3 detection heads.
    """

    def __init__(
        self,
        num_classes: int = 80,
        conf_threshold: float = 0.25,
        input_size: Tuple[int, int] = (640, 640),
        strides: Tuple[int, ...] = (8, 16, 32),
    ) -> None:
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.strides = strides

    def decode(
        self,
        raw: np.ndarray,
        max_detections: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw YOLOv8 inference output.

        Parameters
        ----------
        raw : np.ndarray
            Shape ``(1, 4+C, N)`` or ``(4+C, N)``.
        max_detections : int
            Hard cap on returned detections.

        Returns
        -------
        boxes : np.ndarray  ``(K, 4)`` xyxy in input-pixel space
        scores : np.ndarray ``(K,)``
        class_ids : np.ndarray ``(K,)``
        """
        if raw.ndim == 3:
            raw = raw[0]  # remove batch dim

        # Split into box and class components
        box_preds = raw[:4, :].T       # (N, 4)  xcycwh
        cls_preds = raw[4:, :].T       # (N, C)

        # Class confidence via sigmoid
        cls_scores = 1.0 / (1.0 + np.exp(-cls_preds))  # sigmoid
        max_scores = cls_scores.max(axis=1)
        max_class_ids = cls_scores.argmax(axis=1)

        # Confidence filter
        mask = max_scores >= self.conf_threshold
        box_preds = box_preds[mask]
        max_scores = max_scores[mask]
        max_class_ids = max_class_ids[mask]

        if box_preds.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # Convert xcycwh → xyxy (already in input-pixel coords for YOLOv8)
        boxes_xyxy = xywh_to_xyxy(box_preds)

        # Cap detections
        if len(boxes_xyxy) > max_detections:
            topk = np.argsort(max_scores)[::-1][:max_detections]
            boxes_xyxy = boxes_xyxy[topk]
            max_scores = max_scores[topk]
            max_class_ids = max_class_ids[topk]

        return boxes_xyxy.astype(np.float32), max_scores.astype(np.float32), max_class_ids.astype(np.int64)


# ===================================================================
# YOLOv8Detector
# ===================================================================


@register_detector("yolov8")
class YOLOv8Detector(ObjectDetector):
    """YOLOv8-based object detector with Ultralytics or ONNX backend.

    Parameters
    ----------
    config : dict
        Must include ``model_path``.  Optional keys override defaults:
        ``device``, ``conf_threshold``, ``iou_threshold``, ``input_size``,
        ``max_detections``, ``class_names``, ``backend``.
    """

    # Default COCO class names (80 classes) – trimmed for brevity but
    # includes the subset relevant for autonomous driving.
    _COCO_NAMES: List[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.backend: str = config.get("backend", "ultralytics")
        self.half: bool = config.get("half", False) and self.device != "cpu"
        self.max_detections: int = config.get("max_detections", 300)
        self.class_whitelist: Optional[List[int]] = config.get("class_whitelist", None)

        # Set default class names if not provided
        if not self.class_names:
            self.class_names = self._COCO_NAMES

        # Preprocessor
        self._preprocessor = LetterBoxPreprocessor(
            input_size=self.input_size,
            normalize=True,
            mean=(0.0, 0.0, 0.0),
            std=(255.0, 255.0, 255.0),
        )

        # Decoder
        self._decoder = YOLOv8Decoder(
            num_classes=len(self.class_names),
            conf_threshold=self.conf_threshold,
            input_size=self.input_size,
        )

        # ONNX session placeholder
        self._onnx_session: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model from *model_path*.

        Supported formats:
            - ``.pt``  → Ultralytics API
            - ``.onnx`` → ONNX Runtime
            - ``.engine`` / ``.trt`` → TensorRT via ONNX Runtime (if available)
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        suffix = model_path.suffix.lower()
        if suffix == ".pt":
            self._load_ultralytics(model_path)
        elif suffix == ".onnx":
            self._load_onnx(model_path)
        elif suffix in (".engine", ".trt"):
            self._load_tensorrt(model_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        self._is_loaded = True

    def _load_ultralytics(self, model_path: Path) -> None:
        """Load via the Ultralytics Python package."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            # Warm-up with a forward pass
            dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy, device=self.device, verbose=False,
                half=self.half, conf=self.conf_threshold,
            )
            logger.info("Loaded Ultralytics YOLOv8 model from %s", model_path)
        except ImportError:
            logger.warning("ultralytics not available; falling back to ONNX")
            onnx_path = model_path.with_suffix(".onnx")
            if onnx_path.exists():
                self._load_onnx(onnx_path)
            else:
                raise

    def _load_onnx(self, model_path: Path) -> None:
        """Load via ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference")

        providers: List[str] = []
        if "cuda" in self.device.lower():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self._onnx_session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers,
        )
        self.model = self._onnx_session
        logger.info("Loaded ONNX model from %s (providers=%s)", model_path, providers)

    def _load_tensorrt(self, model_path: Path) -> None:
        """Load TensorRT engine via ONNX Runtime with TRT EP."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime with TensorRT EP required")

        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_path": str(model_path),
                "trt_fp16_enable": self.half,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._onnx_session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers,
        )
        self.model = self._onnx_session
        logger.info("Loaded TensorRT engine from %s", model_path)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Letterbox-resize, BGR→RGB, normalize to ``[0, 1]`` float32."""
        return self._preprocessor(image)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, tensor: np.ndarray) -> np.ndarray:
        """Run raw model inference.

        Returns shape ``(1, 4+C, N)`` for standard YOLOv8 output.
        """
        if self.backend == "ultralytics" and hasattr(self.model, "predict"):
            return self._infer_ultralytics(tensor)
        elif self._onnx_session is not None:
            return self._infer_onnx(tensor)
        else:
            raise RuntimeError("No inference backend available")

    def _infer_ultralytics(self, tensor: np.ndarray) -> np.ndarray:
        """Inference through the Ultralytics predict API."""
        # tensor is (1, C, H, W); convert back to HWC uint8 for YOLO wrapper
        img = (tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        results = self.model.predict(
            img, device=self.device, verbose=False,
            half=self.half, conf=self.conf_threshold,
        )
        # Return as standard YOLOv8 raw output shape
        if results and hasattr(results[0], "boxes"):
            r = results[0].boxes
            if r.data.numel() > 0:
                import torch
                data = r.data.cpu().numpy()  # (N, 6)  x1,y1,x2,y2,conf,cls
                n = data.shape[0]
                c = len(self.class_names)
                raw = np.zeros((1, 4 + c, n), dtype=np.float32)
                raw[0, 0, :] = (data[:, 0] + data[:, 2]) / 2  # xc
                raw[0, 1, :] = (data[:, 1] + data[:, 3]) / 2  # yc
                raw[0, 2, :] = data[:, 2] - data[:, 0]         # w
                raw[0, 3, :] = data[:, 3] - data[:, 1]         # h
                for i in range(n):
                    cls_id = int(data[i, 5])
                    raw[0, 4 + cls_id, i] = data[i, 4]
                return raw
        return np.zeros((1, 4 + len(self.class_names), 0), dtype=np.float32)

    def _infer_onnx(self, tensor: np.ndarray) -> np.ndarray:
        """Inference through ONNX Runtime."""
        input_name = self._onnx_session.get_inputs()[0].name
        feed = {input_name: tensor}
        outputs = self._onnx_session.run(None, feed)
        return outputs[0]  # (1, 4+C, N) or (1, N, 4+C)

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------

    def postprocess(
        self,
        raw_output: np.ndarray,
        meta: Dict[str, Any],
    ) -> DetectionResult:
        """Decode raw output, apply NMS, rescale to original image coords."""
        boxes, scores, class_ids = self._decoder.decode(
            raw_output, max_detections=self.max_detections * 2,
        )

        if boxes.size == 0:
            return DetectionResult(model_name="yolov8", device=self.device)

        # Rescale boxes from letterbox space → original image space
        orig_shape = meta.get("orig_shape", self.input_size)
        boxes = scale_boxes(
            boxes,
            from_shape=self.input_size,
            to_shape=(orig_shape[0], orig_shape[1]),
        )
        boxes = clip_boxes(boxes, orig_shape)

        # Class-aware NMS
        keep = class_aware_nms(
            boxes, scores, class_ids,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Optional class whitelist
        if self.class_whitelist is not None:
            whitelist = set(self.class_whitelist)
            mask = np.array([int(cid) in whitelist for cid in class_ids])
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

        # Optional box refinement (local consensus averaging)
        boxes = refine_boxes(boxes, scores)

        # Build Detection objects
        detections: List[Detection] = []
        for i in range(len(boxes)):
            det = Detection(
                bbox=BBox(*boxes[i].tolist()),
                class_id=int(class_ids[i]),
                class_name=self.get_class_name(int(class_ids[i])),
                confidence=float(scores[i]),
            )
            detections.append(det)

        return DetectionResult(
            detections=detections,
            model_name="yolov8",
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Convenience: detect using Ultralytics directly (fast path)
    # ------------------------------------------------------------------

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """Full pipeline, with fast-path optimisation for Ultralytics backend."""
        # Fast path: let Ultralytics handle the entire pipeline
        if (
            self.backend == "ultralytics"
            and hasattr(self.model, "predict")
            and self._is_loaded
        ):
            return self._detect_ultralytics_fast(image)

        # Template-method path (ONNX / OpenVINO)
        return super().detect_image(image)

    def _detect_ultralytics_fast(self, image: np.ndarray) -> DetectionResult:
        """Use Ultralytics' internal preprocessing + NMS for speed."""
        t0 = time.perf_counter()
        results = self.model.predict(
            image, device=self.device, verbose=False,
            half=self.half, conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        t1 = time.perf_counter()

        detections: List[Detection] = []
        if results and hasattr(results[0], "boxes"):
            r = results[0].boxes
            if r.data.numel() > 0:
                import torch
                data = r.data.cpu().numpy()
                for i in range(data.shape[0]):
                    x1, y1, x2, y2, conf, cls = data[i]
                    det = Detection(
                        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                        class_id=int(cls),
                        class_name=self.get_class_name(int(cls)),
                        confidence=float(conf),
                    )
                    detections.append(det)

        return DetectionResult(
            detections=detections,
            inference_time_ms=(t1 - t0) * 1000,
            image_shape=image.shape[:2],
            model_name="yolov8",
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Batch detection – leverages Ultralytics batch support when available."""
        if self.backend == "ultralytics" and hasattr(self.model, "predict"):
            return self._detect_batch_ultralytics(images)
        return [self.detect_image(img) for img in images]

    def _detect_batch_ultralytics(self, images: List[np.ndarray]) -> List[DetectionResult]:
        t0 = time.perf_counter()
        results = self.model.predict(
            images, device=self.device, verbose=False,
            half=self.half, conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        output: List[DetectionResult] = []
        for idx, r in enumerate(results):
            detections: List[Detection] = []
            if hasattr(r, "boxes") and r.boxes.data.numel() > 0:
                import torch
                data = r.boxes.data.cpu().numpy()
                for i in range(data.shape[0]):
                    x1, y1, x2, y2, conf, cls = data[i]
                    det = Detection(
                        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                        class_id=int(cls),
                        class_name=self.get_class_name(int(cls)),
                        confidence=float(conf),
                    )
                    detections.append(det)
            output.append(DetectionResult(
                detections=detections,
                inference_time_ms=total_ms / len(images),
                image_shape=images[idx].shape[:2],
                model_name="yolov8",
                device=self.device,
            ))
        return output

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def model_info(self) -> Dict[str, Any]:
        """Return model metadata (parameters, FLOPs, etc.)."""
        info: Dict[str, Any] = {
            "backend": self.backend,
            "device": self.device,
            "half": self.half,
            "input_size": self.input_size,
            "num_classes": len(self.class_names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }
        if self.backend == "ultralytics" and hasattr(self.model, "model"):
            try:
                m = self.model.model
                params = sum(p.numel() for p in m.parameters())
                info["parameters"] = params
                info["flops"] = getattr(m, "flops", None)
            except Exception:
                pass
        return info
