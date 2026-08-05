#!/usr/bin/env python3
# =============================================================================
# File: python/segmentation.py
# Brief: SegmentationModel — post-processing for semantic segmentation
#        networks (FCN, DeepLabV3+, U-Net, ENet) on TensorRT. Handles
#        argmax, color mapping, FPS measurement, and drivable-area extraction.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Semantic segmentation post-processing for the Jetson AV stack.

Consumes raw TensorRT output of shape ``(B, C, H, W)`` (logits) and
produces:

* Class-index map ``(H, W)`` uint8,
* Colorized visualization (RGB) using a fixed palette,
* Drivable-area mask,
* Per-class pixel fractions,
* Real-time FPS estimate.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore


# -----------------------------------------------------------------------------
# Default Cityscapes-style 19-class palette (RGB).
# -----------------------------------------------------------------------------
DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (128, 64, 128),   # 0 road
    (244, 35, 232),   # 1 sidewalk
    (70, 70, 70),     # 2 building
    (102, 102, 156),  # 3 wall
    (190, 153, 153),  # 4 fence
    (153, 153, 153),  # 5 pole
    (250, 170, 30),   # 6 traffic light
    (220, 220, 0),    # 7 traffic sign
    (107, 142, 35),   # 8 vegetation
    (152, 251, 152),  # 9 terrain
    (70, 130, 180),   # 10 sky
    (220, 20, 60),    # 11 person
    (255, 0, 0),      # 12 rider
    (0, 0, 142),      # 13 car
    (0, 0, 70),       # 14 truck
    (0, 60, 100),     # 15 bus
    (0, 80, 100),     # 16 train
    (0, 0, 230),      # 17 motorcycle
    (119, 11, 32),    # 18 bicycle
]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class SegmentationResult:
    """Per-frame segmentation output."""

    class_map: np.ndarray            # (H, W) uint8
    color_map: Optional[np.ndarray]  # (H, W, 3) uint8 RGB or None
    drivable_mask: Optional[np.ndarray]  # (H, W) bool or None
    class_fractions: Dict[int, float] = field(default_factory=dict)
    fps: float = 0.0
    latency_ms: float = 0.0
    input_shape: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_fractions": self.class_fractions,
            "fps": float(self.fps),
            "latency_ms": float(self.latency_ms),
            "input_shape": list(self.input_shape),
        }


# -----------------------------------------------------------------------------
# SegmentationModel
# -----------------------------------------------------------------------------
class SegmentationModel:
    """Post-processes semantic segmentation output tensors.

    Args:
        class_names: Optional list of class name strings.
        palette: Optional ``[(r, g, b), ...]`` palette. Defaults to the
            Cityscapes 19-class palette above.
        drivable_class_ids: Class IDs that count as "drivable" (road,
            parking, etc.). Defaults to ``[0]``.
        apply_softmax: If True, apply softmax before argmax (needed when
            the network outputs raw logits; not needed for already-softmaxed
            outputs).
        fps_window: Number of recent frames used to compute the rolling FPS.
        ignore_class_id: Optional class ID to exclude from colorization
            (e.g. void/unlabeled).
    """

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        palette: Optional[Sequence[Tuple[int, int, int]]] = None,
        drivable_class_ids: Sequence[int] = (0,),
        apply_softmax: bool = True,
        fps_window: int = 30,
        ignore_class_id: Optional[int] = None,
    ) -> None:
        self.class_names = list(class_names) if class_names else []
        self.palette = list(palette) if palette else list(DEFAULT_PALETTE)
        # Extend palette with grey if there are more classes than colors.
        while len(self.palette) < max(64, len(self.class_names)):
            self.palette.append((128, 128, 128))
        self.drivable_class_ids = set(drivable_class_ids)
        self.apply_softmax = apply_softmax
        self.fps_window = fps_window
        self.ignore_class_id = ignore_class_id

        # FPS tracking.
        self._frame_times: Deque[float] = deque(maxlen=fps_window)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def postprocess(
        self,
        logits: np.ndarray,
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> SegmentationResult:
        """Post-process raw network logits into a SegmentationResult.

        Args:
            logits: Network output. Shape ``(B, C, H, W)`` for FCN/DeepLab
                or ``(H, W, C)`` for already-NHWC tensors. The batch
                dimension is squeezed if it is 1.
            original_shape: ``(H, W)`` of the original image. If provided,
                the class map is resized to match.
        """
        t0 = time.perf_counter()
        if logits.ndim == 4:
            if logits.shape[0] != 1:
                raise ValueError(
                    f"Batched post-processing not supported (got B={logits.shape[0]}).")
            logits = logits[0]  # (C, H, W)
        if logits.ndim == 3 and logits.shape[-1] < logits.shape[0]:
            # Already (H, W, C).
            pass
        elif logits.ndim == 3 and logits.shape[0] < logits.shape[-1]:
            logits = np.transpose(logits, (1, 2, 0))  # CHW -> HWC
        elif logits.ndim != 3:
            raise ValueError(
                f"Unsupported logits shape: {logits.shape}")

        # Softmax (numerically stable).
        if self.apply_softmax:
            probs = self._softmax(logits, axis=-1)
            class_map = np.argmax(probs, axis=-1).astype(np.uint8)
        else:
            class_map = np.argmax(logits, axis=-1).astype(np.uint8)

        if self.ignore_class_id is not None:
            class_map[class_map == self.ignore_class_id] = 0

        if original_shape is not None and original_shape != class_map.shape[:2]:
            interp = (
                cv2.INTER_NEAREST if _CV2_AVAILABLE else None)
            if interp is not None:
                class_map = cv2.resize(
                    class_map, (original_shape[1], original_shape[0]),
                    interpolation=interp)

        color_map = self._colorize(class_map)
        drivable_mask = self._compute_drivable_mask(class_map)
        fractions = self._compute_fractions(class_map)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_times.append(t0)
        fps = self._estimate_fps()

        return SegmentationResult(
            class_map=class_map,
            color_map=color_map,
            drivable_mask=drivable_mask,
            class_fractions=fractions,
            fps=fps,
            latency_ms=latency_ms,
            input_shape=tuple(class_map.shape[:2]),
        )

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _colorize(self, class_map: np.ndarray) -> Optional[np.ndarray]:
        """Map class IDs to an RGB image using the palette."""
        if not _CV2_AVAILABLE:
            return None
        h, w = class_map.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        palette = np.array(self.palette, dtype=np.uint8)
        # Clip indices to palette size.
        idx = np.clip(class_map, 0, len(palette) - 1)
        out[:] = palette[idx]
        return out

    def _compute_drivable_mask(
        self, class_map: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return a boolean mask of drivable pixels."""
        if not self.drivable_class_ids:
            return None
        mask = np.zeros_like(class_map, dtype=bool)
        for cid in self.drivable_class_ids:
            mask |= (class_map == cid)
        return mask

    def _compute_fractions(
        self, class_map: np.ndarray
    ) -> Dict[int, float]:
        """Return per-class pixel fraction (top-K only)."""
        total = class_map.size
        if total == 0:
            return {}
        counts = np.bincount(class_map.ravel(), minlength=len(self.palette))
        fractions = {int(cid): float(c) / total
                     for cid, c in enumerate(counts) if c > 0}
        return dict(sorted(fractions.items(),
                            key=lambda kv: -kv[1])[:10])

    def _estimate_fps(self) -> float:
        """Rolling FPS estimate based on recent frame timestamps."""
        if len(self._frame_times) < 2:
            return 0.0
        # Time span between oldest and newest.
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    # ------------------------------------------------------------------ #
    # Overlay / drawing
    # ------------------------------------------------------------------ #
    def overlay(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend ``image`` with the colorized segmentation map."""
        if not _CV2_AVAILABLE or result.color_map is None:
            return image.copy()
        if image.shape[:2] != result.color_map.shape[:2]:
            result.color_map = cv2.resize(
                result.color_map, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # color_map is RGB; image is BGR by convention.
        color_bgr = cv2.cvtColor(result.color_map, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(image, 1 - alpha, color_bgr, alpha, 0)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess(
        image: np.ndarray,
        input_shape: Tuple[int, int] = (512, 1024),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        dtype: np.dtype = np.float32,
        bgr_to_rgb: bool = True,
    ) -> np.ndarray:
        """Standard ImageNet preprocessing: resize, normalize, NCHW."""
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for preprocess().")
        if bgr_to_rgb and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (input_shape[1], input_shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(dtype, copy=False) / 255.0
        arr = (arr - np.array(mean, dtype=dtype)) / np.array(std, dtype=dtype)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return np.ascontiguousarray(arr[None, ...])  # add batch dim


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Synthetic test: 3-class logits, 64x128.
    h, w, c = 64, 128, 3
    logits = np.random.randn(c, h, w).astype(np.float32)
    # Make class 0 (road) dominant in the bottom half.
    logits[0, h // 2:, :] += 2.0
    model = SegmentationModel(
        class_names=["road", "sidewalk", "building"],
        drivable_class_ids=[0],
    )
    res = model.postprocess(logits)
    print(f"Latency: {res.latency_ms:.2f} ms")
    print(f"Class fractions: {res.class_fractions}")
    print(f"Drivable pixels: {int(res.drivable_mask.sum())} / {h*w}")
