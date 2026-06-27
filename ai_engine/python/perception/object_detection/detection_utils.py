"""
Detection Utility Functions
===========================

Low-level utility functions used throughout the object-detection pipeline.

Functions provided:
    - IoU computation (pairwise, batched)
    - Non-Maximum Suppression (standard greedy NMS)
    - Bounding-box format conversions (xyxy ↔ xywh ↔ xcycwh)
    - Anchor generation (FPN-style, YOLO-style)
    - Misc. geometry helpers (point-in-box, box intersection, etc.)

All functions are pure NumPy and have zero dependency on deep-learning
frameworks, making them safe for TensorRT / ONNX Runtime inference paths.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np


# ===================================================================
# IoU Computation
# ===================================================================


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union for two xyxy boxes.

    Parameters
    ----------
    box_a, box_b : np.ndarray
        Shape ``(4,)`` – ``[x1, y1, x2, y2]``.

    Returns
    -------
    float
        IoU value in ``[0, 1]``.
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of xyxy boxes.

    Parameters
    ----------
    boxes_a : np.ndarray
        Shape ``(N, 4)``.
    boxes_b : np.ndarray
        Shape ``(M, 4)``.

    Returns
    -------
    np.ndarray
        Shape ``(N, M)`` IoU matrix.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])   # (N, M)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def compute_giou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute Generalized IoU (GIoU) for matched box pairs.

    Parameters
    ----------
    boxes_a, boxes_b : np.ndarray
        Shape ``(N, 4)`` xyxy.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` GIoU values (range ``[-1, 1]``).
    """
    ix1 = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    iy1 = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    ix2 = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    iy2 = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter

    # Enclosing box
    cx1 = np.minimum(boxes_a[:, 0], boxes_b[:, 0])
    cy1 = np.minimum(boxes_a[:, 1], boxes_b[:, 1])
    cx2 = np.maximum(boxes_a[:, 2], boxes_b[:, 2])
    cy2 = np.maximum(boxes_a[:, 3], boxes_b[:, 3])
    enclosing = (cx2 - cx1) * (cy2 - cy1)

    iou = np.where(union > 0, inter / union, 0.0)
    giou = iou - (enclosing - union) / np.where(enclosing > 0, enclosing, 1.0)
    return giou.astype(np.float32)


def compute_diou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute Distance IoU (DIoU) for matched box pairs.

    Penalises distance between box centres in addition to overlap.
    """
    ix1 = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    iy1 = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    ix2 = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    iy2 = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0)

    # Centre distance
    cx_a = (boxes_a[:, 0] + boxes_a[:, 2]) / 2
    cy_a = (boxes_a[:, 1] + boxes_a[:, 3]) / 2
    cx_b = (boxes_b[:, 0] + boxes_b[:, 2]) / 2
    cy_b = (boxes_b[:, 1] + boxes_b[:, 3]) / 2
    centre_dist_sq = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2

    # Diagonal of enclosing box
    cx1 = np.minimum(boxes_a[:, 0], boxes_b[:, 0])
    cy1 = np.minimum(boxes_a[:, 1], boxes_b[:, 1])
    cx2 = np.maximum(boxes_a[:, 2], boxes_b[:, 2])
    cy2 = np.maximum(boxes_a[:, 3], boxes_b[:, 3])
    diag_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    diou = iou - centre_dist_sq / np.where(diag_sq > 0, diag_sq, 1.0)
    return diou.astype(np.float32)


# ===================================================================
# Non-Maximum Suppression
# ===================================================================


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> np.ndarray:
    """Standard greedy Non-Maximum Suppression on xyxy boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Shape ``(N, 4)`` – ``[x1, y1, x2, y2]``.
    scores : np.ndarray
        Shape ``(N,)`` detection confidence scores.
    iou_threshold : float
        IoU threshold for suppression.
    max_detections : int
        Maximum number of detections to return.

    Returns
    -------
    np.ndarray
        Indices of kept detections, shape ``(K,)``.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0 and len(keep) < max_detections:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.where(areas[i] + areas[rest] - inter > 0,
                               areas[i] + areas[rest] - inter, 1.0)

        mask = iou <= iou_threshold
        order = rest[mask]

    return np.array(keep, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> np.ndarray:
    """Apply NMS independently per class.

    Parameters
    ----------
    boxes : np.ndarray
        ``(N, 4)`` xyxy.
    scores : np.ndarray
        ``(N,)``.
    class_ids : np.ndarray
        ``(N,)`` integer class labels.

    Returns
    -------
    np.ndarray
        Kept indices.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    keep: List[int] = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_indices = np.where(cls_mask)[0]
        cls_keep = nms(
            boxes[cls_mask], scores[cls_mask],
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        keep.extend(cls_indices[cls_keep].tolist())

    # Re-sort by score and cap
    keep = sorted(keep, key=lambda idx: scores[idx], reverse=True)
    return np.array(keep[:max_detections], dtype=np.int64)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
    max_detections: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS: decay scores of overlapping boxes instead of hard removal.

    Parameters
    ----------
    method : str
        ``"gaussian"`` – ``score *= exp(-iou² / sigma)``
        ``"linear"``   – ``score *= 1 - iou`` when iou > threshold

    Returns
    -------
    keep_indices : np.ndarray
    new_scores : np.ndarray
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    sc = scores.copy().astype(np.float64)

    areas = (x2 - x1) * (y2 - y1)
    n = len(boxes)
    order = np.arange(n)
    keep: List[int] = []

    for _ in range(n):
        max_idx = np.argmax(sc[order])
        i = order[max_idx]
        keep.append(int(i))

        if len(keep) >= max_detections or order.size <= 1:
            break

        rest = np.delete(order, max_idx)

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.where(areas[i] + areas[rest] - inter > 0,
                               areas[i] + areas[rest] - inter, 1.0)

        if method == "gaussian":
            sc[rest] *= np.exp(-(iou ** 2) / sigma)
        elif method == "linear":
            sc[rest] *= np.where(iou > iou_threshold, 1.0 - iou, 1.0)
        else:
            raise ValueError(f"Unknown soft-NMS method: {method}")

        mask = sc[rest] >= score_threshold
        order = rest[mask]

    keep = np.array(keep, dtype=np.int64)
    return keep, sc[keep].astype(np.float32)


# ===================================================================
# Box Format Conversions
# ===================================================================


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """``[x1, y1, x2, y2]`` → ``[x1, y1, w, h]``."""
    if boxes.size == 0:
        return boxes.copy()
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``[x1, y1, w, h]`` → ``[x1, y1, x2, y2]``."""
    if boxes.size == 0:
        return boxes.copy()
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def xyxy_to_xcycwh(boxes: np.ndarray) -> np.ndarray:
    """``[x1, y1, x2, y2]`` → ``[cx, cy, w, h]``."""
    if boxes.size == 0:
        return boxes.copy()
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def xcycwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``[cx, cy, w, h]`` → ``[x1, y1, x2, y2]``."""
    if boxes.size == 0:
        return boxes.copy()
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def scale_boxes(
    boxes: np.ndarray,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
) -> np.ndarray:
    """Scale xyxy boxes from one image size to another.

    Parameters
    ----------
    boxes : np.ndarray
        ``(N, 4)`` in source pixel coords.
    from_shape : tuple
        ``(H_src, W_src)``.
    to_shape : tuple
        ``(H_dst, W_dst)``.

    Returns
    -------
    np.ndarray
        Scaled boxes.
    """
    if boxes.size == 0:
        return boxes.copy()
    gain = min(to_shape[0] / from_shape[0], to_shape[1] / from_shape[1])
    pad_x = (to_shape[1] - from_shape[1] * gain) / 2
    pad_y = (to_shape[0] - from_shape[0] * gain) / 2

    out = boxes.copy().astype(np.float64)
    out[:, [0, 2]] = out[:, [0, 2]] * gain + pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * gain + pad_y
    return out.astype(np.float32)


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clip xyxy boxes to image boundaries.

    Parameters
    ----------
    boxes : np.ndarray
        ``(N, 4)``.
    shape : tuple
        ``(H, W)``.

    Returns
    -------
    np.ndarray
        Clipped boxes.
    """
    if boxes.size == 0:
        return boxes.copy()
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, shape[1])
    out[:, 1] = np.clip(out[:, 1], 0, shape[0])
    out[:, 2] = np.clip(out[:, 2], 0, shape[1])
    out[:, 3] = np.clip(out[:, 3], 0, shape[0])
    return out


# ===================================================================
# Anchor Generation
# ===================================================================


def generate_anchors(
    feature_map_sizes: List[Tuple[int, int]],
    strides: List[int],
    anchor_sizes: List[List[int]],
    anchor_ratios: List[float] = (0.5, 1.0, 2.0),
) -> np.ndarray:
    """Generate multi-scale anchor boxes for single-stage detectors.

    Parameters
    ----------
    feature_map_sizes : list of (H, W)
        Spatial size of each FPN level.
    strides : list of int
        Stride of each FPN level relative to input.
    anchor_sizes : list of list of int
        Base anchor size (in pixels) per level.
    anchor_ratios : list of float
        Aspect ratios to enumerate per base size.

    Returns
    -------
    np.ndarray
        Shape ``(TotalAnchors, 4)`` in ``xcycwh`` format.
    """
    anchors: List[np.ndarray] = []

    for fm_size, stride, sizes in zip(feature_map_sizes, strides, anchor_sizes):
        fm_h, fm_w = fm_size
        for y in range(fm_h):
            for x in range(fm_w):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                for base_size in sizes:
                    for ratio in anchor_ratios:
                        h = base_size / math.sqrt(ratio)
                        w = base_size * math.sqrt(ratio)
                        anchors.append([cx, cy, w, h])

    return np.array(anchors, dtype=np.float32)


def generate_yolo_anchors(
    input_size: int = 640,
    strides: Tuple[int, ...] = (8, 16, 32),
    anchors_per_level: int = 3,
) -> List[np.ndarray]:
    """Generate YOLO-style anchor tensors for each detection head.

    Returns
    -------
    list of np.ndarray
        Each element has shape ``(A, 2)`` where A = anchors_per_level.
        Columns are ``(w, h)`` in pixel space.
    """
    # Default COCO-trained YOLOv8 anchors (scaled to 640)
    default_anchors = {
        640: [
            [[10, 13], [16, 30], [33, 23]],         # P3/8
            [[30, 61], [62, 45], [59, 119]],         # P4/16
            [[116, 90], [156, 198], [373, 326]],     # P5/32
        ],
        1280: [
            [[19, 27], [44, 27], [38, 64]],
            [[74, 53], [60, 134], [137, 105]],
            [[137, 180], [198, 259], [327, 438]],
        ],
    }

    anchor_set = default_anchors.get(input_size, default_anchors[640])
    result: List[np.ndarray] = []
    for level_anchors in anchor_set:
        result.append(np.array(level_anchors, dtype=np.float32))
    return result


# ===================================================================
# Geometry Helpers
# ===================================================================


def point_in_box(px: float, py: float, box: np.ndarray) -> bool:
    """Check whether a 2-D point lies inside an xyxy box."""
    return bool(box[0] <= px <= box[2] and box[1] <= py <= box[3])


def box_intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area of two xyxy boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def compute_box_aspect_ratios(boxes: np.ndarray) -> np.ndarray:
    """Compute width/height ratio for each box."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.where(h > 0, w / h, 0.0)


def merge_boxes(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Weighted box merge – average box coordinates weighted by score.

    Useful for test-time ensemble / WBF.
    """
    if boxes.size == 0:
        return boxes
    total = scores.sum()
    if total == 0:
        return boxes.mean(axis=0, keepdims=True)
    weights = scores / total
    merged = (boxes * weights[:, None]).sum(axis=0)
    return merged.reshape(1, -1)


def filter_small_boxes(
    boxes: np.ndarray,
    min_width: float = 10.0,
    min_height: float = 10.0,
) -> np.ndarray:
    """Return indices of boxes larger than the given minimum dimensions."""
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_width) & (h >= min_height)
    return np.where(mask)[0]
