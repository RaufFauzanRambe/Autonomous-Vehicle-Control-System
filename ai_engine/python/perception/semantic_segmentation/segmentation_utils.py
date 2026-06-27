"""
Segmentation Utility Functions for Autonomous Vehicle Perception.

Provides common utility functions for semantic segmentation including
class mapping, color palettes, polygon extraction from masks,
mask IoU computation, and coordinate transformations.

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cityscapes Class Definitions
# ---------------------------------------------------------------------------

CITYSCAPES_CLASSES = [
    {"id": 0, "name": "road", "category": "flat", "train_id": 0, "evaluate": True},
    {"id": 1, "name": "sidewalk", "category": "flat", "train_id": 1, "evaluate": True},
    {"id": 2, "name": "building", "category": "construction", "train_id": 2, "evaluate": True},
    {"id": 3, "name": "wall", "category": "construction", "train_id": 3, "evaluate": True},
    {"id": 4, "name": "fence", "category": "construction", "train_id": 4, "evaluate": True},
    {"id": 5, "name": "pole", "category": "object", "train_id": 5, "evaluate": True},
    {"id": 6, "name": "traffic_light", "category": "object", "train_id": 6, "evaluate": True},
    {"id": 7, "name": "traffic_sign", "category": "object", "train_id": 7, "evaluate": True},
    {"id": 8, "name": "vegetation", "category": "nature", "train_id": 8, "evaluate": True},
    {"id": 9, "name": "terrain", "category": "nature", "train_id": 9, "evaluate": True},
    {"id": 10, "name": "sky", "category": "sky", "train_id": 10, "evaluate": True},
    {"id": 11, "name": "person", "category": "human", "train_id": 11, "evaluate": True},
    {"id": 12, "name": "rider", "category": "human", "train_id": 12, "evaluate": True},
    {"id": 13, "name": "car", "category": "vehicle", "train_id": 13, "evaluate": True},
    {"id": 14, "name": "truck", "category": "vehicle", "train_id": 14, "evaluate": True},
    {"id": 15, "name": "bus", "category": "vehicle", "train_id": 15, "evaluate": True},
    {"id": 16, "name": "train", "category": "vehicle", "train_id": 16, "evaluate": True},
    {"id": 17, "name": "motorcycle", "category": "vehicle", "train_id": 17, "evaluate": True},
    {"id": 18, "name": "bicycle", "category": "vehicle", "train_id": 18, "evaluate": True},
]

# KITTI classes for road/obstacle segmentation
KITTI_CLASSES = [
    {"id": 0, "name": "road", "train_id": 0},
    {"id": 1, "name": "sidewalk", "train_id": 1},
    {"id": 2, "name": "building", "train_id": 2},
    {"id": 3, "name": "wall", "train_id": 3},
    {"id": 4, "name": "fence", "train_id": 4},
    {"id": 5, "name": "pole", "train_id": 5},
    {"id": 6, "name": "traffic_light", "train_id": 6},
    {"id": 7, "name": "traffic_sign", "train_id": 7},
    {"id": 8, "name": "vegetation", "train_id": 8},
    {"id": 9, "name": "terrain", "train_id": 9},
    {"id": 10, "name": "sky", "train_id": 10},
    {"id": 11, "name": "person", "train_id": 11},
    {"id": 12, "name": "rider", "train_id": 12},
    {"id": 13, "name": "car", "train_id": 13},
    {"id": 14, "name": "truck", "train_id": 14},
    {"id": 15, "name": "bus", "train_id": 15},
    {"id": 16, "name": "train", "train_id": 16},
    {"id": 17, "name": "motorcycle", "train_id": 17},
    {"id": 18, "name": "bicycle", "train_id": 18},
]


# ---------------------------------------------------------------------------
# Color Palettes
# ---------------------------------------------------------------------------

def get_cityscapes_palette() -> np.ndarray:
    """Get Cityscapes standard color palette.

    Returns:
        Array of shape (19, 3) with RGB colors for each class.
    """
    return np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic_light
        [220, 220, 0],    # traffic_sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)


def get_kitti_palette() -> np.ndarray:
    """Get KITTI dataset color palette.

    Returns:
        Array of shape (19, 3) with RGB colors.
    """
    return np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic_light
        [220, 220, 0],    # traffic_sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)


def get_road_palette() -> np.ndarray:
    """Get simplified road segmentation palette.

    Returns:
        Array of shape (5, 3) for road-specific classes.
    """
    return np.array([
        [0, 0, 0],        # background
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [190, 153, 153],  # parking
        [153, 153, 153],  # rail_track
    ], dtype=np.uint8)


def get_lane_palette() -> np.ndarray:
    """Get lane segmentation palette.

    Returns:
        Array of shape (10, 3) for lane classes.
    """
    return np.array([
        [0, 0, 0],        # background
        [255, 255, 255],  # solid_white
        [255, 255, 0],    # solid_yellow
        [200, 200, 200],  # dashed_white
        [200, 200, 0],    # dashed_yellow
        [255, 128, 0],    # double_solid
        [180, 180, 180],  # double_dashed
        [255, 0, 255],    # crosswalk
        [255, 0, 0],      # stop_line
        [128, 128, 128],  # road_edge
    ], dtype=np.uint8)


def generate_distinct_palette(num_classes: int, seed: int = 42) -> np.ndarray:
    """Generate a visually distinct color palette.

    Uses HSV color space with evenly distributed hues for maximum
    visual distinction between classes.

    Args:
        num_classes: Number of classes.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (num_classes, 3) with RGB colors.
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    rng = np.random.RandomState(seed)

    for i in range(num_classes):
        hue = (i * 360 / num_classes) % 360
        saturation = 0.7 + 0.3 * rng.random()
        value = 0.6 + 0.4 * rng.random()

        # HSV to RGB conversion
        c = value * saturation
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = value - c

        if hue < 60:
            r, g, b = c, x, 0
        elif hue < 120:
            r, g, b = x, c, 0
        elif hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        elif hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        palette[i] = [
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255),
        ]

    return palette


def colorize_mask(
    mask: np.ndarray,
    palette: Optional[np.ndarray] = None,
    num_classes: int = 19,
) -> np.ndarray:
    """Colorize a segmentation mask using a color palette.

    Args:
        mask: Segmentation mask with class indices (H, W).
        palette: Color palette of shape (num_classes, 3). Uses Cityscapes if None.
        num_classes: Number of classes (used if palette is None).

    Returns:
        Colorized mask of shape (H, W, 3) as uint8.
    """
    if palette is None:
        palette = get_cityscapes_palette()

    h, w = mask.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(min(len(palette), num_classes)):
        colorized[mask == cls_id] = palette[cls_id]

    return colorized


# ---------------------------------------------------------------------------
# Class Mapping
# ---------------------------------------------------------------------------

class ClassMapper:
    """Maps between different dataset class schemes.

    Handles conversion between Cityscapes, KITTI, and custom
    class definitions for cross-dataset compatibility.
    """

    def __init__(
        self,
        source_classes: List[Dict[str, Any]],
        target_classes: List[Dict[str, Any]],
    ) -> None:
        """Initialize class mapper.

        Args:
            source_classes: Source dataset class definitions.
            target_classes: Target dataset class definitions.
        """
        self.source_classes = source_classes
        self.target_classes = target_classes
        self._mapping = self._build_mapping()

    def _build_mapping(self) -> Dict[int, int]:
        """Build mapping from source to target class IDs.

        Returns:
            Dictionary mapping source_id -> target_id.
        """
        mapping = {}
        source_by_name = {c["name"]: c["id"] for c in self.source_classes}
        target_by_name = {c["name"]: c["id"] for c in self.target_classes}

        for name, src_id in source_by_name.items():
            if name in target_by_name:
                mapping[src_id] = target_by_name[name]
            else:
                mapping[src_id] = 0  # Map unknown to background

        return mapping

    def map_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply class mapping to a segmentation mask.

        Args:
            mask: Source segmentation mask.

        Returns:
            Mapped segmentation mask.
        """
        result = np.zeros_like(mask)
        for src_id, tgt_id in self._mapping.items():
            result[mask == src_id] = tgt_id
        return result

    @staticmethod
    def cityscapes_to_road() -> "ClassMapper":
        """Create mapper from Cityscapes to road segmentation classes."""
        road_classes = [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "road"},
            {"id": 2, "name": "sidewalk"},
            {"id": 3, "name": "parking"},
            {"id": 4, "name": "rail_track"},
        ]
        return ClassMapper(CITYSCAPES_CLASSES, road_classes)

    @staticmethod
    def cityscapes_to_obstacle() -> "ClassMapper":
        """Create mapper from Cityscapes to obstacle classes."""
        obstacle_classes = [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "car"},
            {"id": 2, "name": "person"},
            {"id": 3, "name": "rider"},
            {"id": 4, "name": "truck"},
            {"id": 5, "name": "bus"},
        ]
        return ClassMapper(CITYSCAPES_CLASSES, obstacle_classes)


# ---------------------------------------------------------------------------
# Polygon Extraction
# ---------------------------------------------------------------------------

class PolygonExtractor:
    """Extracts polygon boundaries from segmentation masks.

    Converts binary or multi-class masks to polygon representations
    suitable for vector maps and geometric analysis.

    Uses marching squares algorithm for contour extraction and
    Douglas-Peucker algorithm for polygon simplification.
    """

    def __init__(
        self,
        simplify_epsilon: float = 2.0,
        min_polygon_area: float = 100.0,
        min_polygon_vertices: int = 3,
    ) -> None:
        """Initialize polygon extractor.

        Args:
            simplify_epsilon: Douglas-Peucker simplification tolerance (pixels).
            min_polygon_area: Minimum polygon area to keep.
            min_polygon_vertices: Minimum vertices for a valid polygon.
        """
        self.simplify_epsilon = simplify_epsilon
        self.min_polygon_area = min_polygon_area
        self.min_polygon_vertices = min_polygon_vertices

    def extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract contour points from a binary mask using marching squares.

        Args:
            mask: Binary mask (H, W).

        Returns:
            List of contour arrays, each of shape (N, 2).
        """
        h, w = mask.shape
        contours: List[List[Tuple[int, int]]] = []
        visited_edges = set()

        # Scan for boundary pixels
        for y in range(h - 1):
            for x in range(w - 1):
                # 2x2 block classification
                v00 = mask[y, x]
                v01 = mask[y, x + 1]
                v10 = mask[y + 1, x]
                v11 = mask[y + 1, x + 1]

                # Check if this is a boundary cell
                if v00 != v01 or v00 != v10 or v00 != v11:
                    # Find boundary edges
                    if v00 != v01 and (y, x, 'h') not in visited_edges:
                        visited_edges.add((y, x, 'h'))
                    if v00 != v10 and (y, x, 'v') not in visited_edges:
                        visited_edges.add((y, x, 'v'))

        # Trace contours from boundary edges
        # Simplified: extract all boundary pixel coordinates
        boundary_points: List[Tuple[int, int]] = []

        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    # Check if boundary pixel
                    is_boundary = False
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w or mask[ny, nx] == 0:
                            is_boundary = True
                            break
                    if is_boundary:
                        boundary_points.append((x, y))

        if boundary_points:
            contours.append(np.array(boundary_points, dtype=np.int32))

        return contours

    def simplify_polygon(
        self, points: np.ndarray, epsilon: Optional[float] = None
    ) -> np.ndarray:
        """Simplify polygon using Douglas-Peucker algorithm.

        Args:
            points: Polygon points of shape (N, 2).
            epsilon: Simplification tolerance (uses default if None).

        Returns:
            Simplified polygon points.
        """
        if epsilon is None:
            epsilon = self.simplify_epsilon

        if len(points) <= 2:
            return points

        # Find the point with maximum distance from line start-end
        start = points[0]
        end = points[-1]

        # Compute distances from each point to line start-end
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-8:
            # All points are at same location
            return np.array([start, end])

        line_unit = line_vec / line_len
        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(points) - 1):
            point_vec = points[i] - start
            # Distance from point to line
            proj = np.dot(point_vec, line_unit)
            if proj < 0:
                dist = np.linalg.norm(point_vec)
            elif proj > line_len:
                dist = np.linalg.norm(points[i] - end)
            else:
                dist = np.linalg.norm(point_vec - proj * line_unit)

            if dist > max_dist:
                max_dist = dist
                max_idx = i

        # Recursive simplification
        if max_dist > epsilon:
            left = self.simplify_polygon(points[:max_idx + 1], epsilon)
            right = self.simplify_polygon(points[max_idx:], epsilon)
            return np.vstack([left[:-1], right])
        else:
            return np.array([start, end])

    def compute_polygon_area(self, points: np.ndarray) -> float:
        """Compute polygon area using the shoelace formula.

        Args:
            points: Polygon vertices of shape (N, 2).

        Returns:
            Absolute polygon area.
        """
        if len(points) < 3:
            return 0.0

        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    def extract_polygons(
        self, mask: np.ndarray, class_id: int = 1
    ) -> List[Dict[str, Any]]:
        """Extract polygons from a segmentation mask for a specific class.

        Args:
            mask: Segmentation mask (H, W) with class indices.
            class_id: Target class to extract polygons for.

        Returns:
            List of polygon dictionaries with vertices and metadata.
        """
        class_mask = (mask == class_id).astype(np.uint8)
        contours = self.extract_contours(class_mask)

        polygons = []
        for contour in contours:
            # Simplify contour
            simplified = self.simplify_polygon(contour.astype(np.float64))

            # Check validity
            area = self.compute_polygon_area(simplified)
            if area < self.min_polygon_area:
                continue
            if len(simplified) < self.min_polygon_vertices:
                continue

            # Compute bounding box
            x_min = int(np.min(simplified[:, 0]))
            y_min = int(np.min(simplified[:, 1]))
            x_max = int(np.max(simplified[:, 0]))
            y_max = int(np.max(simplified[:, 1]))

            polygon = {
                "class_id": class_id,
                "vertices": simplified.tolist(),
                "num_vertices": len(simplified),
                "area": area,
                "bbox": [x_min, y_min, x_max, y_max],
                "centroid": [
                    float(np.mean(simplified[:, 0])),
                    float(np.mean(simplified[:, 1])),
                ],
            }
            polygons.append(polygon)

        return polygons

    def extract_all_polygons(
        self, mask: np.ndarray, num_classes: int = 19
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Extract polygons for all classes in the mask.

        Args:
            mask: Segmentation mask (H, W).
            num_classes: Number of classes.

        Returns:
            Dictionary mapping class_id to list of polygons.
        """
        all_polygons: Dict[int, List[Dict[str, Any]]] = {}

        for cls_id in range(num_classes):
            if np.any(mask == cls_id):
                polygons = self.extract_polygons(mask, cls_id)
                if polygons:
                    all_polygons[cls_id] = polygons

        return all_polygons


# ---------------------------------------------------------------------------
# Mask IoU and Overlap
# ---------------------------------------------------------------------------

def compute_mask_iou(
    mask1: np.ndarray, mask2: np.ndarray
) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.

    Returns:
        IoU value in range [0, 1].
    """
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))

    if union == 0:
        return 1.0 if np.sum(mask1) == 0 and np.sum(mask2) == 0 else 0.0

    return float(intersection / union)


def compute_mask_dice(
    mask1: np.ndarray, mask2: np.ndarray
) -> float:
    """Compute Dice coefficient between two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.

    Returns:
        Dice coefficient in range [0, 1].
    """
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    total = np.sum(mask1 > 0) + np.sum(mask2 > 0)

    if total == 0:
        return 1.0

    return float(2 * intersection / total)


def compute_per_class_iou(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute IoU for each class.

    Args:
        pred: Predicted segmentation mask (H, W).
        target: Ground truth mask (H, W).
        num_classes: Number of classes.

    Returns:
        Array of IoU values per class.
    """
    ious = np.zeros(num_classes, dtype=np.float64)

    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        iou = compute_mask_iou(pred_mask.astype(np.uint8), target_mask.astype(np.uint8))
        ious[cls] = iou

    return ious


def compute_confusion_matrix(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute confusion matrix from segmentation masks.

    Args:
        pred: Predicted mask (H, W) with class indices.
        target: Ground truth mask (H, W) with class indices.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
        Rows = ground truth, Columns = predictions.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Flatten masks
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Filter valid class indices
    valid = (target_flat >= 0) & (target_flat < num_classes) & \
            (pred_flat >= 0) & (pred_flat < num_classes)
    pred_valid = pred_flat[valid]
    target_valid = target_flat[valid]

    # Accumulate
    for t, p in zip(target_valid, pred_valid):
        confusion[int(t), int(p)] += 1

    return confusion


def compute_boundary_mask(
    mask: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """Extract boundary pixels from a segmentation mask.

    Args:
        mask: Segmentation mask (H, W).
        kernel_size: Kernel size for boundary detection.

    Returns:
        Binary boundary mask (H, W).
    """
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=np.uint8)
    pad = kernel_size // 2

    for y in range(h):
        for x in range(w):
            center_class = mask[y, x]
            # Check neighbors
            is_boundary = False
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] != center_class:
                            is_boundary = True
                            break
                if is_boundary:
                    break
            boundary[y, x] = 1 if is_boundary else 0

    return boundary


def compute_boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Compute Boundary F1 score.

    Measures the quality of segmentation boundaries by comparing
    predicted and ground truth boundary pixels with a tolerance.

    Args:
        pred: Predicted segmentation mask (H, W).
        target: Ground truth mask (H, W).
        tolerance: Maximum distance for boundary matching (pixels).

    Returns:
        Boundary F1 score in [0, 1].
    """
    pred_boundary = compute_boundary_mask(pred)
    target_boundary = compute_boundary_mask(target)

    if np.sum(target_boundary) == 0 and np.sum(pred_boundary) == 0:
        return 1.0
    if np.sum(target_boundary) == 0 or np.sum(pred_boundary) == 0:
        return 0.0

    # Dilate target boundary by tolerance for matching
    h, w = pred_boundary.shape
    dilated_target = target_boundary.copy()

    for _ in range(tolerance):
        new_dilated = dilated_target.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if dilated_target[y, x] > 0:
                    new_dilated[y - 1:y + 2, x - 1:x + 2] = 1
        dilated_target = new_dilated

    # Precision: fraction of predicted boundary near target boundary
    precision = np.sum(pred_boundary & dilated_target) / max(np.sum(pred_boundary), 1)

    # Dilate predicted boundary for recall
    dilated_pred = pred_boundary.copy()
    for _ in range(tolerance):
        new_dilated = dilated_pred.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if dilated_pred[y, x] > 0:
                    new_dilated[y - 1:y + 2, x - 1:x + 2] = 1
        dilated_pred = new_dilated

    # Recall: fraction of target boundary near predicted boundary
    recall = np.sum(target_boundary & dilated_pred) / max(np.sum(target_boundary), 1)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


# ---------------------------------------------------------------------------
# Coordinate Transformations
# ---------------------------------------------------------------------------

def image_to_bev(
    mask: np.ndarray,
    camera_height: float = 1.65,
    focal_length: float = 721.5,
    bev_resolution: float = 0.05,
    bev_range: Tuple[float, float, float, float] = (0.0, 5.0, -10.0, 10.0),
) -> np.ndarray:
    """Transform segmentation mask from image space to Bird's Eye View.

    Uses inverse perspective mapping (IPM) to project image pixels
    onto a ground plane BEV representation.

    Args:
        mask: Segmentation mask (H, W).
        camera_height: Camera height above ground (meters).
        focal_length: Camera focal length (pixels).
        bev_resolution: BEV grid resolution (meters/pixel).
        bev_range: (x_min, x_max, z_min, z_max) in meters.

    Returns:
        BEV segmentation mask.
    """
    h, w = mask.shape
    x_min, x_max, z_min, z_max = bev_range

    bev_w = int((x_max - x_min) / bev_resolution)
    bev_h = int((z_max - z_min) / bev_resolution)
    bev_mask = np.zeros((bev_h, bev_w), dtype=mask.dtype)

    # Center of image
    cx = w / 2.0
    cy = h / 2.0

    # Inverse perspective mapping
    for bz in range(bev_h):
        z = z_min + bz * bev_resolution  # Forward distance
        if z <= 0:
            continue

        for bx in range(bev_w):
            x = x_min + bx * bev_resolution  # Lateral distance

            # Project BEV point to image
            u = int(focal_length * x / z + cx)
            v = int(focal_length * camera_height / z + cy)

            if 0 <= u < w and 0 <= v < h:
                bev_mask[bz, bx] = mask[v, u]

    return bev_mask


def remap_labels(
    mask: np.ndarray,
    label_mapping: Dict[int, int],
    ignore_label: int = 255,
) -> np.ndarray:
    """Remap class labels in a segmentation mask.

    Args:
        mask: Input segmentation mask (H, W).
        label_mapping: Dictionary mapping source -> target labels.
        ignore_label: Label to ignore during remapping.

    Returns:
        Remapped segmentation mask.
    """
    result = np.full_like(mask, ignore_label)

    for src_label, tgt_label in label_mapping.items():
        result[mask == src_label] = tgt_label

    return result


def merge_masks(
    masks: List[np.ndarray],
    priority: Optional[List[int]] = None,
) -> np.ndarray:
    """Merge multiple segmentation masks with priority ordering.

    Args:
        masks: List of segmentation masks (H, W).
        priority: Priority order for overlapping regions (higher index = higher priority).

    Returns:
        Merged segmentation mask.
    """
    if not masks:
        return np.array([], dtype=np.uint8)

    h, w = masks[0].shape
    result = np.zeros((h, w), dtype=np.uint8)

    if priority is None:
        priority = list(range(len(masks)))

    for idx in priority:
        if idx < len(masks):
            nonzero = masks[idx] > 0
            result[nonzero] = masks[idx][nonzero]

    return result


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
    method: str = "nearest",
) -> np.ndarray:
    """Resize a segmentation mask to target size.

    Uses nearest-neighbor interpolation to preserve class labels.

    Args:
        mask: Input mask (H, W).
        target_size: Target size (H', W').
        method: Interpolation method ('nearest' only supported).

    Returns:
        Resized mask.
    """
    h, w = mask.shape
    th, tw = target_size

    result = np.zeros(target_size, dtype=mask.dtype)

    for y in range(th):
        src_y = min(int(y * h / th), h - 1)
        for x in range(tw):
            src_x = min(int(x * w / tw), w - 1)
            result[y, x] = mask[src_y, src_x]

    return result
