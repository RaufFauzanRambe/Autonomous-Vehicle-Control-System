"""
Post-Processing Module for Semantic Segmentation.

Provides post-processing operations to refine segmentation model outputs
including CRF refinement, morphological operations, connected component
analysis, and small region removal.

Post-Processing Pipeline:
    Model Output ──▶ Argmax Mask ──▶ CRF Refinement ──▶ Morphological Ops
                                       │                    │
                                       ▼                    ▼
                                  Refined Mask ──▶ Connected Components
                                                        │
                                                        ▼
                                                  Small Region Removal
                                                        │
                                                        ▼
                                                    Final Mask

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PostprocessConfig:
    """Configuration for post-processing pipeline.

    Attributes:
        use_crf: Whether to apply CRF refinement.
        crf_iterations: Number of CRF mean-field iterations.
        crf_bilateral_weight: Weight for bilateral kernel.
        crf_spatial_sigma: Spatial sigma for bilateral kernel.
        crf_color_sigma: Color sigma for bilateral kernel.
        use_morphology: Whether to apply morphological operations.
        morph_operation: Type of morphological operation.
        morph_kernel_size: Kernel size for morphological operations.
        morph_iterations: Number of morphological iterations.
        fill_holes: Whether to fill holes in masks.
        remove_small_regions: Whether to remove small connected components.
        min_region_area: Minimum area for small region removal.
        smooth_boundaries: Whether to smooth mask boundaries.
        boundary_smooth_iterations: Number of boundary smoothing iterations.
        use_confidence_filter: Whether to filter by confidence.
        confidence_threshold: Minimum confidence for pixel assignment.
        merge_nearby_instances: Whether to merge nearby instances.
        merge_distance_threshold: Distance threshold for merging.
        apply_per_class: Whether to apply operations per class.
    """

    use_crf: bool = False
    crf_iterations: int = 5
    crf_bilateral_weight: float = 5.0
    crf_spatial_sigma: float = 3.0
    crf_color_sigma: float = 10.0
    use_morphology: bool = True
    morph_operation: str = "close_open"
    morph_kernel_size: int = 5
    morph_iterations: int = 1
    fill_holes: bool = True
    remove_small_regions: bool = True
    min_region_area: int = 256
    smooth_boundaries: bool = False
    boundary_smooth_iterations: int = 1
    use_confidence_filter: bool = False
    confidence_threshold: float = 0.5
    merge_nearby_instances: bool = False
    merge_distance_threshold: float = 20.0
    apply_per_class: bool = True


# ---------------------------------------------------------------------------
# Connected Component Analysis
# ---------------------------------------------------------------------------

class ConnectedComponentAnalyzer:
    """Connected component analysis for segmentation masks.

    Provides efficient connected component labeling using union-find
    with path compression and union by rank.

    Supports:
        - 4-connectivity (default)
        - 8-connectivity
        - Component statistics (area, centroid, bounding box)
        - Component filtering by area
    """

    def __init__(self, connectivity: int = 4) -> None:
        """Initialize connected component analyzer.

        Args:
            connectivity: Connectivity type (4 or 8).
        """
        if connectivity not in (4, 8):
            raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")
        self.connectivity = connectivity

        if connectivity == 4:
            self._offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            self._offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]

    def label(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Label connected components in a binary mask.

        Args:
            mask: Binary mask (H, W) with values 0 or 1.

        Returns:
            Tuple of (labeled mask, number of components).
        """
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        parent: Dict[int, int] = {}
        rank: Dict[int, int] = {}
        next_label = 1

        def find(x: int) -> int:
            """Find root with path compression."""
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            """Union by rank."""
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank.get(rx, 0) < rank.get(ry, 0):
                rx, ry = ry, rx
            parent[ry] = rx
            if rank.get(rx, 0) == rank.get(ry, 0):
                rank[rx] = rank.get(rx, 0) + 1

        # First pass: assign provisional labels
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0:
                    continue

                neighbors = []
                for dy, dx in self._offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                        neighbors.append(labels[ny, nx])

                if not neighbors:
                    labels[y, x] = next_label
                    parent[next_label] = next_label
                    rank[next_label] = 0
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    labels[y, x] = min_label
                    for n in neighbors:
                        union(min_label, n)

        # Second pass: replace with root labels
        label_map: Dict[int, int] = {}
        new_label = 0
        for y in range(h):
            for x in range(w):
                if labels[y, x] > 0:
                    root = find(labels[y, x])
                    if root not in label_map:
                        new_label += 1
                        label_map[root] = new_label
                    labels[y, x] = label_map[root]

        return labels, new_label

    def get_component_stats(
        self, labeled_mask: np.ndarray, num_components: int
    ) -> List[Dict[str, Any]]:
        """Compute statistics for each connected component.

        Args:
            labeled_mask: Labeled mask from label().
            num_components: Number of components.

        Returns:
            List of component statistics dictionaries.
        """
        stats = []

        for comp_id in range(1, num_components + 1):
            component = labeled_mask == comp_id
            area = int(np.sum(component))

            if area == 0:
                continue

            ys, xs = np.where(component)

            stat = {
                "id": comp_id,
                "area": area,
                "centroid": (float(np.mean(xs)), float(np.mean(ys))),
                "bbox": (int(np.min(xs)), int(np.min(ys)),
                         int(np.max(xs)), int(np.max(ys))),
                "width": int(np.max(xs) - np.min(xs)) + 1,
                "height": int(np.max(ys) - np.min(ys)) + 1,
            }
            stats.append(stat)

        return stats

    def filter_by_area(
        self,
        labeled_mask: np.ndarray,
        min_area: int,
        max_area: Optional[int] = None,
    ) -> np.ndarray:
        """Remove components outside the specified area range.

        Args:
            labeled_mask: Labeled mask.
            min_area: Minimum component area.
            max_area: Maximum component area (None = no limit).

        Returns:
            Filtered labeled mask.
        """
        num_components = int(np.max(labeled_mask))
        result = np.zeros_like(labeled_mask)

        for comp_id in range(1, num_components + 1):
            component = labeled_mask == comp_id
            area = np.sum(component)

            if area >= min_area and (max_area is None or area <= max_area):
                result[component] = comp_id

        return result

    def extract_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Extract the largest connected component.

        Args:
            mask: Binary mask.

        Returns:
            Mask containing only the largest component.
        """
        labeled, num = self.label(mask)
        if num == 0:
            return np.zeros_like(mask)

        # Find largest component
        best_id = 0
        best_area = 0
        for comp_id in range(1, num + 1):
            area = np.sum(labeled == comp_id)
            if area > best_area:
                best_area = area
                best_id = comp_id

        return (labeled == best_id).astype(mask.dtype)


# ---------------------------------------------------------------------------
# Hole Filling
# ---------------------------------------------------------------------------

class HoleFiller:
    """Fills holes in segmentation masks.

    A hole is a background region completely surrounded by foreground.
    Uses flood fill from borders to identify exterior background,
    then inverts to find and fill holes.
    """

    def __init__(self, connectivity: int = 4) -> None:
        """Initialize hole filler.

        Args:
            connectivity: Flood fill connectivity (4 or 8).
        """
        self.connectivity = connectivity
        if connectivity == 4:
            self._offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            self._offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in a binary mask.

        Args:
            mask: Binary mask (H, W).

        Returns:
            Mask with holes filled.
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        h, w = mask.shape
        result = mask.copy()

        # Create background mask
        background = (mask == 0).astype(np.uint8)

        # Flood fill from all border pixels to find exterior background
        exterior = np.zeros_like(background)
        queue: List[Tuple[int, int]] = []

        # Seed from borders
        for y in range(h):
            if background[y, 0]:
                exterior[y, 0] = 1
                queue.append((y, 0))
            if background[y, w - 1]:
                exterior[y, w - 1] = 1
                queue.append((y, w - 1))
        for x in range(w):
            if background[0, x]:
                exterior[0, x] = 1
                queue.append((0, x))
            if background[h - 1, x]:
                exterior[h - 1, x] = 1
                queue.append((h - 1, x))

        # BFS flood fill
        while queue:
            cy, cx = queue.pop(0)
            for dy, dx in self._offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not exterior[ny, nx] and background[ny, nx]:
                    exterior[ny, nx] = 1
                    queue.append((ny, nx))

        # Holes = background - exterior
        holes = background & ~exterior
        result[holes > 0] = 1

        return result

    def fill_holes_per_class(
        self, mask: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Fill holes for each class independently.

        Args:
            mask: Segmentation mask (H, W) with class indices.
            num_classes: Number of classes.

        Returns:
            Mask with holes filled per class.
        """
        result = mask.copy()

        for cls in range(1, num_classes):  # Skip background
            class_mask = (mask == cls).astype(np.uint8)
            if np.sum(class_mask) > 0:
                filled = self.fill_holes(class_mask)
                result[filled > 0] = cls

        return result


# ---------------------------------------------------------------------------
# Boundary Smoothing
# ---------------------------------------------------------------------------

class BoundarySmoother:
    """Smooths segmentation mask boundaries.

    Applies iterative majority voting at boundaries to produce
    smoother class transitions, reducing jagged edges from
    pixel-level predictions.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        iterations: int = 1,
        threshold: float = 0.5,
    ) -> None:
        """Initialize boundary smoother.

        Args:
            kernel_size: Kernel size for local voting.
            iterations: Number of smoothing iterations.
            threshold: Fraction of neighbors needed to change a pixel's class.
        """
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.threshold = threshold
        self._half = kernel_size // 2

    def smooth(self, mask: np.ndarray, num_classes: int) -> np.ndarray:
        """Apply boundary smoothing.

        Args:
            mask: Segmentation mask (H, W).
            num_classes: Number of classes.

        Returns:
            Smoothed mask.
        """
        result = mask.copy()
        h, w = result.shape

        for _ in range(self.iterations):
            # Find boundary pixels
            boundary = np.zeros((h, w), dtype=bool)
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    center = result[y, x]
                    if (result[y - 1, x] != center or result[y + 1, x] != center or
                        result[y, x - 1] != center or result[y, x + 1] != center):
                        boundary[y, x] = True

            # Apply majority voting at boundaries
            for y in range(self._half, h - self._half):
                for x in range(self._half, w - self._half):
                    if not boundary[y, x]:
                        continue

                    # Count class votes in local neighborhood
                    region = result[
                        y - self._half:y + self._half + 1,
                        x - self._half:x + self._half + 1,
                    ]
                    votes = np.zeros(num_classes, dtype=np.int32)
                    for cls in range(num_classes):
                        votes[cls] = np.sum(region == cls)

                    # Majority class
                    majority_cls = np.argmax(votes)
                    if votes[majority_cls] > self.threshold * self.kernel_size ** 2:
                        result[y, x] = majority_cls

        return result


# ---------------------------------------------------------------------------
# Instance Merging
# ---------------------------------------------------------------------------

class InstanceMerger:
    """Merges nearby instance segments of the same class.

    Useful for combining fragmented detections of the same object
    that were split due to occlusion or segmentation errors.
    """

    def __init__(
        self,
        distance_threshold: float = 20.0,
        iou_threshold: float = 0.0,
        same_class_only: bool = True,
    ) -> None:
        """Initialize instance merger.

        Args:
            distance_threshold: Maximum centroid distance for merging.
            iou_threshold: Minimum IoU for merging.
            same_class_only: Whether to only merge same-class instances.
        """
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.same_class_only = same_class_only

    def compute_distance(
        self, bbox1: Tuple[int, ...], bbox2: Tuple[int, ...]
    ) -> float:
        """Compute minimum distance between two bounding boxes.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2).
            bbox2: Second bounding box.

        Returns:
            Minimum distance between boxes.
        """
        dx = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
        dy = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def compute_iou(
        self, bbox1: Tuple[int, ...], bbox2: Tuple[int, ...]
    ) -> float:
        """Compute IoU between two bounding boxes.

        Args:
            bbox1: First bounding box.
            bbox2: Second bounding box.

        Returns:
            IoU value.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1)

    def merge_instances(
        self,
        instance_mask: np.ndarray,
        class_mask: np.ndarray,
    ) -> np.ndarray:
        """Merge nearby instances of the same class.

        Args:
            instance_mask: Instance label mask (H, W).
            class_mask: Semantic class mask (H, W).

        Returns:
            Merged instance mask.
        """
        num_instances = int(np.max(instance_mask))
        if num_instances <= 1:
            return instance_mask

        # Compute instance statistics
        instance_info: Dict[int, Dict] = {}
        for inst_id in range(1, num_instances + 1):
            mask = instance_mask == inst_id
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            instance_info[inst_id] = {
                "class": int(class_mask[ys[0], xs[0]]),
                "centroid": (float(np.mean(xs)), float(np.mean(ys))),
                "bbox": (int(np.min(xs)), int(np.min(ys)),
                         int(np.max(xs)), int(np.max(ys))),
                "area": int(np.sum(mask)),
            }

        # Find merge pairs
        merge_groups: Dict[int, int] = {}  # inst_id -> group_id
        group_counter = 0

        inst_ids = list(instance_info.keys())
        for i in range(len(inst_ids)):
            for j in range(i + 1, len(inst_ids)):
                id_i, id_j = inst_ids[i], inst_ids[j]
                info_i, info_j = instance_info[id_i], instance_info[id_j]

                # Check if same class
                if self.same_class_only and info_i["class"] != info_j["class"]:
                    continue

                # Check distance
                dist = self.compute_distance(info_i["bbox"], info_j["bbox"])
                if dist > self.distance_threshold:
                    continue

                # Check IoU
                iou = self.compute_iou(info_i["bbox"], info_j["bbox"])
                if iou < self.iou_threshold:
                    continue

                # Merge groups
                if id_i in merge_groups and id_j in merge_groups:
                    # Union groups
                    group_i = merge_groups[id_i]
                    group_j = merge_groups[id_j]
                    for k, v in merge_groups.items():
                        if v == group_j:
                            merge_groups[k] = group_i
                elif id_i in merge_groups:
                    merge_groups[id_j] = merge_groups[id_i]
                elif id_j in merge_groups:
                    merge_groups[id_i] = merge_groups[id_j]
                else:
                    group_counter += 1
                    merge_groups[id_i] = group_counter
                    merge_groups[id_j] = group_counter

        # Apply merges
        result = instance_mask.copy()
        merged_groups: Dict[int, int] = {}  # group_id -> new_inst_id

        for inst_id, group_id in merge_groups.items():
            if group_id not in merged_groups:
                merged_groups[group_id] = min(
                    k for k, v in merge_groups.items() if v == group_id
                )
            target_id = merged_groups[group_id]
            result[instance_mask == inst_id] = target_id

        return result


# ---------------------------------------------------------------------------
# Main Post-Processing Pipeline
# ---------------------------------------------------------------------------

class PostprocessingPipeline:
    """Complete post-processing pipeline for segmentation outputs.

    Orchestrates all post-processing steps in configurable order:
        1. Confidence filtering
        2. CRF refinement
        3. Morphological operations
        4. Hole filling
        5. Small region removal
        6. Boundary smoothing
        7. Instance merging

    Example:
        >>> config = PostprocessConfig(use_morphology=True, fill_holes=True)
        >>> pipeline = PostprocessingPipeline(config)
        >>> mask = pipeline.process(raw_mask, image, num_classes=19)
    """

    def __init__(self, config: PostprocessConfig) -> None:
        """Initialize post-processing pipeline.

        Args:
            config: Post-processing configuration.
        """
        self.config = config
        self._cc_analyzer = ConnectedComponentAnalyzer()
        self._hole_filler = HoleFiller()
        self._boundary_smoother = BoundarySmoother()
        self._instance_merger = InstanceMerger(
            distance_threshold=config.merge_distance_threshold
        ) if config.merge_nearby_instances else None

    def process(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        num_classes: int = 19,
    ) -> np.ndarray:
        """Apply full post-processing pipeline.

        Args:
            mask: Input segmentation mask (H, W).
            image: Optional original image for CRF.
            probabilities: Optional class probabilities for confidence filtering.
            num_classes: Number of classes.

        Returns:
            Post-processed segmentation mask.
        """
        result = mask.copy()

        # 1. Confidence filtering
        if self.config.use_confidence_filter and probabilities is not None:
            max_probs = np.max(probabilities, axis=0)
            low_conf = max_probs < self.config.confidence_threshold
            result[low_conf] = 0

        # 2. CRF refinement
        if self.config.use_crf and image is not None:
            from .mask_generator import CRFRefiner, MaskGeneratorConfig
            crf_config = MaskGeneratorConfig(
                crf_iterations=self.config.crf_iterations,
                crf_bilateral_weight=self.config.crf_bilateral_weight,
                crf_spatial_sigma=self.config.crf_spatial_sigma,
                crf_color_sigma=self.config.crf_color_sigma,
            )
            refiner = CRFRefiner(crf_config)
            if probabilities is not None:
                refined_probs = refiner.refine(probabilities, image)
                result = np.argmax(refined_probs, axis=0).astype(np.uint8)
            else:
                result = refiner.refine_mask(result, image, num_classes)

        # 3. Morphological operations
        if self.config.use_morphology:
            result = self._apply_morphology(result, num_classes)

        # 4. Hole filling
        if self.config.fill_holes:
            result = self._hole_filler.fill_holes_per_class(result, num_classes)

        # 5. Small region removal
        if self.config.remove_small_regions:
            result = self._remove_small_regions(result, num_classes)

        # 6. Boundary smoothing
        if self.config.smooth_boundaries:
            result = self._boundary_smoother.smooth(result, num_classes)

        return result

    def _apply_morphology(
        self, mask: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Apply morphological operations per class.

        Args:
            mask: Segmentation mask.
            num_classes: Number of classes.

        Returns:
            Morphologically processed mask.
        """
        from .mask_generator import MorphologicalProcessor

        morph = MorphologicalProcessor(
            kernel_size=self.config.morph_kernel_size,
            close_iterations=self.config.morph_iterations,
            open_iterations=self.config.morph_iterations,
            fill_holes=False,  # Handled separately
        )

        result = mask.copy()

        if self.config.apply_per_class:
            for cls in range(1, num_classes):  # Skip background
                class_mask = (mask == cls).astype(np.uint8)
                if np.sum(class_mask) == 0:
                    continue

                if self.config.morph_operation == "close":
                    processed = morph.close(class_mask)
                elif self.config.morph_operation == "open":
                    processed = morph.open(class_mask)
                elif self.config.morph_operation == "close_open":
                    processed = morph.close(class_mask)
                    processed = morph.open(processed)
                elif self.config.morph_operation == "open_close":
                    processed = morph.open(class_mask)
                    processed = morph.close(processed)
                else:
                    processed = class_mask

                result[processed > 0] = cls
        else:
            # Apply globally to foreground
            fg_mask = (mask > 0).astype(np.uint8)
            processed = morph.close(fg_mask)
            processed = morph.open(processed)
            result[processed == 0] = 0

        return result

    def _remove_small_regions(
        self, mask: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Remove small connected components.

        Args:
            mask: Segmentation mask.
            num_classes: Number of classes.

        Returns:
            Cleaned mask.
        """
        result = mask.copy()

        if self.config.apply_per_class:
            for cls in range(1, num_classes):
                class_mask = (mask == cls).astype(np.uint8)
                if np.sum(class_mask) == 0:
                    continue

                labeled, num_components = self._cc_analyzer.label(class_mask)
                labeled = self._cc_analyzer.filter_by_area(
                    labeled, self.config.min_region_area
                )

                # Reconstruct class mask
                new_class_mask = (labeled > 0).astype(np.uint8)
                result[(mask == cls) & (new_class_mask == 0)] = 0
        else:
            fg_mask = (mask > 0).astype(np.uint8)
            labeled, _ = self._cc_analyzer.label(fg_mask)
            labeled = self._cc_analyzer.filter_by_area(
                labeled, self.config.min_region_area
            )
            result[labeled == 0] = 0

        return result
