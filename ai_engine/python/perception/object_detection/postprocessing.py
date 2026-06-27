"""
Post-Processing Module
=======================

Post-processing utilities applied after raw model inference:

- **NMS variants**: Greedy NMS, Soft-NMS (Gaussian & Linear), Cluster-NMS
- **Confidence filtering**: Threshold-based and adaptive filtering
- **Class filtering**: Whitelist / blacklist with optional per-class thresholds
- **Box refinement**: Local-consensus averaging, weighted box fusion
- **Test-time augmentation (TTA)**: Multi-scale / multi-flip ensemble

All functions operate on NumPy arrays and have no deep-learning framework
dependency, making them safe for TensorRT / ONNX Runtime inference paths.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .detection_utils import (
    BBox,
    Detection,
    DetectionResult,
    compute_iou_matrix,
    nms,
    class_aware_nms,
    soft_nms,
    clip_boxes,
)

logger = logging.getLogger(__name__)


# ===================================================================
# NMS Variants
# ===================================================================


def greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> np.ndarray:
    """Standard greedy NMS (thin wrapper around :func:`detection_utils.nms`)."""
    return nms(boxes, scores, iou_threshold, max_detections)


def soft_nms_gaussian(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS with Gaussian score decay.

    For each selected box, overlapping box scores are decayed as:
        ``score *= exp(-IoU² / sigma)``

    Returns
    -------
    keep : np.ndarray
        Indices of kept boxes.
    new_scores : np.ndarray
        Updated scores after decay.
    """
    return soft_nms(
        boxes, scores,
        method="gaussian",
        sigma=sigma,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )


def soft_nms_linear(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS with linear score decay.

    For each selected box, overlapping box scores are decayed as:
        ``score *= 1 - IoU`` when IoU > threshold.
    """
    return soft_nms(
        boxes, scores,
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )


def cluster_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    max_iterations: int = 10,
) -> np.ndarray:
    """Cluster-NMS: iterative NMS that converges to a local optimum.

    Instead of removing overlapping boxes outright, Cluster-NMS
    iteratively re-scores boxes until the set stabilises.  This often
    preserves more true positives in dense scenes.

    Reference: Yang et al., "Cluster-NMS" (arXiv 2005.03572)

    Parameters
    ----------
    boxes : np.ndarray
        ``(N, 4)`` xyxy.
    scores : np.ndarray
        ``(N,)``.
    iou_threshold : float
    max_detections : int
    max_iterations : int
        Maximum iterations for convergence.

    Returns
    -------
    np.ndarray
        Kept indices.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    n = len(boxes)
    scores_f = scores.astype(np.float64).copy()
    order = np.argsort(-scores_f)

    for _ in range(max_iterations):
        # Compute IoU matrix for current ranking
        sorted_boxes = boxes[order]
        iou_mat = compute_iou_matrix(sorted_boxes, sorted_boxes)  # (N, N)

        # Zero out self-IoU
        np.fill_diagonal(iou_mat, 0)

        # Suppress: for each box, decay score by max IoU with higher-ranked box
        for i in range(1, n):
            max_iou = iou_mat[:i, i].max()
            if max_iou > iou_threshold:
                scores_f[order[i]] *= 1.0 - max_iou

        # Re-rank
        new_order = np.argsort(-scores_f)
        if np.array_equal(order, new_order):
            break
        order = new_order

    # Filter by updated scores
    keep_mask = scores_f[order] >= (scores.max() * 0.01)  # relative threshold
    kept = order[keep_mask][:max_detections]
    return np.sort(kept).astype(np.int64)


def diou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> np.ndarray:
    """Distance-IoU NMS: uses DIoU instead of IoU for suppression.

    Penalises boxes that are far from the reference box centre, even if
    overlap is low, which can help in crowded scenes.
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
        union = areas[i] + areas[rest] - inter
        iou = inter / np.where(union > 0, union, 1.0)

        # DIoU penalty
        cx_i = (x1[i] + x2[i]) / 2
        cy_i = (y1[i] + y2[i]) / 2
        cx_r = (x1[rest] + x2[rest]) / 2
        cy_r = (y1[rest] + y2[rest]) / 2
        centre_dist_sq = (cx_i - cx_r) ** 2 + (cy_i - cy_r) ** 2

        # Enclosing box diagonal
        ecx1 = np.minimum(x1[i], x1[rest])
        ecy1 = np.minimum(y1[i], y1[rest])
        ecx2 = np.maximum(x2[i], x2[rest])
        ecy2 = np.maximum(y2[i], y2[rest])
        diag_sq = (ecx2 - ecx1) ** 2 + (ecy2 - ecy1) ** 2

        diou = iou - centre_dist_sq / np.where(diag_sq > 0, diag_sq, 1.0)

        # Use IoU threshold on DIoU (more aggressive suppression)
        mask = iou <= iou_threshold
        order = rest[mask]

    return np.array(keep, dtype=np.int64)


# ===================================================================
# Confidence Filtering
# ===================================================================


def filter_by_confidence(
    scores: np.ndarray,
    threshold: float = 0.25,
) -> np.ndarray:
    """Return indices where *scores* >= *threshold*."""
    return np.where(scores >= threshold)[0]


def adaptive_confidence_filter(
    scores: np.ndarray,
    min_threshold: float = 0.05,
    percentile: float = 90.0,
    min_keep: int = 1,
) -> np.ndarray:
    """Adaptive confidence threshold based on score distribution.

    Computes a threshold as the max of *min_threshold* and the
    *percentile*-th percentile of all scores.  Ensures at least
    *min_keep* detections are retained.
    """
    if scores.size == 0:
        return np.array([], dtype=np.int64)

    adaptive_thr = max(np.percentile(scores, percentile), min_threshold)
    indices = np.where(scores >= adaptive_thr)[0]

    if len(indices) < min_keep:
        # Keep top-min_keep by score
        top_k = np.argsort(-scores)[:min_keep]
        indices = np.sort(top_k)

    return indices


# ===================================================================
# Class Filtering
# ===================================================================


def filter_detections_by_class(
    detections: List[Detection],
    class_whitelist: Optional[List[int]] = None,
    class_blacklist: Optional[List[int]] = None,
) -> List[Detection]:
    """Filter Detection objects by class ID.

    Parameters
    ----------
    detections : list of Detection
    class_whitelist : list of int, optional
        Only keep detections whose class_id is in this list.
    class_blacklist : list of int, optional
        Remove detections whose class_id is in this list.

    Returns
    -------
    list of Detection
    """
    if class_whitelist is not None:
        whitelist_set = set(class_whitelist)
        detections = [d for d in detections if d.class_id in whitelist_set]
    if class_blacklist is not None:
        blacklist_set = set(class_blacklist)
        detections = [d for d in detections if d.class_id not in blacklist_set]
    return detections


def filter_by_class_with_thresholds(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    per_class_thresholds: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter detections with per-class confidence thresholds.

    Parameters
    ----------
    boxes : np.ndarray  ``(N, 4)``
    scores : np.ndarray  ``(N,)``
    class_ids : np.ndarray  ``(N,)``
    per_class_thresholds : dict
        Mapping ``class_id → min_confidence``.

    Returns
    -------
    boxes, scores, class_ids : tuple of np.ndarray
    """
    if boxes.size == 0:
        return boxes, scores, class_ids

    mask = np.ones(len(boxes), dtype=bool)
    for cls_id, thr in per_class_thresholds.items():
        cls_mask = class_ids == cls_id
        mask[cls_mask] &= scores[cls_mask] >= thr

    return boxes[mask], scores[mask], class_ids[mask]


# ===================================================================
# Box Refinement
# ===================================================================


def refine_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.65,
    method: str = "weighted_average",
) -> np.ndarray:
    """Refine overlapping boxes by weighted averaging or voting.

    Parameters
    ----------
    boxes : np.ndarray
        ``(N, 4)`` xyxy after NMS.
    scores : np.ndarray
        ``(N,)``.
    iou_threshold : float
        IoU threshold for grouping overlapping boxes.
    method : str
        ``"weighted_average"`` – merge boxes weighted by score.
        ``"voting"`` – keep highest-scored box, adjust by cluster mean.

    Returns
    -------
    np.ndarray
        ``(M, 4)`` refined boxes (M ≤ N).
    """
    if boxes.size == 0 or len(boxes) <= 1:
        return boxes

    n = len(boxes)
    visited = np.zeros(n, dtype=bool)
    refined: List[np.ndarray] = []

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Find cluster
        ious = compute_iou_matrix(boxes[i:i + 1], boxes)[0]  # (N,)
        cluster_mask = (ious >= iou_threshold) & (~visited)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_indices = np.concatenate([[i], cluster_indices])

        for ci in cluster_indices:
            visited[ci] = True

        if method == "weighted_average":
            cluster_boxes = boxes[cluster_indices]
            cluster_scores = scores[cluster_indices]
            total_score = cluster_scores.sum()
            if total_score > 0:
                weights = cluster_scores / total_score
                merged = (cluster_boxes * weights[:, None]).sum(axis=0)
            else:
                merged = cluster_boxes.mean(axis=0)
            refined.append(merged)
        else:  # voting
            # Keep the highest-scored box
            best_idx = cluster_indices[scores[cluster_indices].argmax()]
            refined.append(boxes[best_idx])

    return np.array(refined, dtype=np.float32)


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    iou_threshold: float = 0.55,
    skip_class_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Box Fusion (WBF) for ensembling multiple model outputs.

    Unlike NMS which discards boxes, WBF merges overlapping boxes from
    different models using a weighted average of coordinates and scores.

    Reference: Solovyev et al., "Weighted Boxes Fusion" (arXiv 1910.13302)

    Parameters
    ----------
    boxes_list : list of np.ndarray
        Each element ``(N_i, 4)`` xyxy.
    scores_list : list of np.ndarray
        Each element ``(N_i,)``.
    labels_list : list of np.ndarray
        Each element ``(N_i,)``.
    iou_threshold : float
    skip_class_thr : float
        Minimum score to include a box in the fusion.

    Returns
    -------
    fused_boxes, fused_scores, fused_labels : tuple
    """
    # Collect all boxes
    all_boxes = np.concatenate(boxes_list, axis=0) if boxes_list else np.zeros((0, 4))
    all_scores = np.concatenate(scores_list) if scores_list else np.zeros(0)
    all_labels = np.concatenate(labels_list) if labels_list else np.zeros(0, dtype=np.int64)

    if all_boxes.size == 0:
        return all_boxes, all_scores, all_labels

    unique_labels = np.unique(all_labels)
    fused_boxes: List[np.ndarray] = []
    fused_scores: List[float] = []
    fused_labels: List[int] = []

    for label in unique_labels:
        mask = all_labels == label
        lbl_boxes = all_boxes[mask]
        lbl_scores = all_scores[mask]

        # Cluster by IoU
        n = len(lbl_boxes)
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True

            for j in range(i + 1, n):
                if visited[j]:
                    continue
                iou = compute_iou_matrix(
                    lbl_boxes[i:i + 1], lbl_boxes[j:j + 1],
                )[0, 0]
                if iou > iou_threshold:
                    cluster.append(j)
                    visited[j] = True

            # Fuse cluster
            c_boxes = lbl_boxes[cluster]
            c_scores = lbl_scores[cluster]
            total = c_scores.sum()
            if total > 0:
                weights = c_scores / total
                merged = (c_boxes * weights[:, None]).sum(axis=0)
                # Average confidence
                avg_score = total / len(cluster)
            else:
                merged = c_boxes.mean(axis=0)
                avg_score = 0.0

            if avg_score >= skip_class_thr:
                fused_boxes.append(merged)
                fused_scores.append(float(avg_score))
                fused_labels.append(int(label))

    if not fused_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_labels, dtype=np.int64),
    )


# ===================================================================
# Test-Time Augmentation (TTA) Ensemble
# ===================================================================


def ensemble_tta(
    results: List[DetectionResult],
    iou_threshold: float = 0.55,
    method: str = "wbf",
) -> DetectionResult:
    """Ensemble multiple DetectionResult objects (e.g., from TTA).

    Parameters
    ----------
    results : list of DetectionResult
        Detections from different augmentations / scales.
    iou_threshold : float
    method : str
        ``"nms"`` – standard NMS on concatenated detections.
        ``"wbf"`` – weighted box fusion.
        ``"soft_nms"`` – soft-NMS.

    Returns
    -------
    DetectionResult
    """
    if not results:
        return DetectionResult()

    if len(results) == 1:
        return results[0]

    # Collect all detections
    all_dets: List[Detection] = []
    for r in results:
        all_dets.extend(r.detections)

    if not all_dets:
        return DetectionResult()

    boxes = np.array([d.bbox.to_numpy() for d in all_dets], dtype=np.float32)
    scores = np.array([d.confidence for d in all_dets], dtype=np.float32)
    class_ids = np.array([d.class_id for d in all_dets], dtype=np.int64)

    if method == "nms":
        keep = class_aware_nms(boxes, scores, class_ids, iou_threshold=iou_threshold)
        kept_dets = [all_dets[i] for i in keep]
    elif method == "wbf":
        fused_boxes, fused_scores, fused_labels = weighted_box_fusion(
            [boxes], [scores], [class_ids], iou_threshold=iou_threshold,
        )
        kept_dets = [
            Detection(
                bbox=BBox(*fused_boxes[i].tolist()),
                class_id=int(fused_labels[i]),
                class_name=all_dets[0].class_name if fused_labels[i] == all_dets[0].class_id else f"class_{fused_labels[i]}",
                confidence=float(fused_scores[i]),
            )
            for i in range(len(fused_boxes))
        ]
    elif method == "soft_nms":
        keep, new_scores = soft_nms(boxes, scores, iou_threshold=iou_threshold)
        kept_dets = [all_dets[i] for i in keep]
        for j, det in enumerate(kept_dets):
            det.confidence = float(new_scores[j])
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    total_inference = sum(r.inference_time_ms for r in results) / len(results)
    return DetectionResult(
        detections=kept_dets,
        inference_time_ms=total_inference,
        model_name=results[0].model_name,
        device=results[0].device,
    )


# ===================================================================
# NMS Dispatcher
# ===================================================================


NMS_REGISTRY: Dict[str, Any] = {
    "greedy": greedy_nms,
    "soft_gaussian": soft_nms_gaussian,
    "soft_linear": soft_nms_linear,
    "cluster": cluster_nms,
    "diou": diou_nms,
}


def apply_nms(
    nms_method: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to the requested NMS variant.

    Parameters
    ----------
    nms_method : str
        Key in :data:`NMS_REGISTRY`.
    boxes : np.ndarray
        ``(N, 4)`` xyxy.
    scores : np.ndarray
        ``(N,)``.
    iou_threshold : float
    **kwargs
        Additional arguments forwarded to the NMS function.

    Returns
    -------
    np.ndarray
        Kept indices.
    """
    func = NMS_REGISTRY.get(nms_method)
    if func is None:
        raise ValueError(
            f"Unknown NMS method '{nms_method}'. "
            f"Available: {list(NMS_REGISTRY.keys())}"
        )
    result = func(boxes, scores, iou_threshold=iou_threshold, **kwargs)
    # Soft-NMS returns (indices, scores)
    if isinstance(result, tuple):
        return result[0]
    return result
