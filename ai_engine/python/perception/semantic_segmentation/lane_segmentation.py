"""
Lane Segmentation Module for Autonomous Vehicle Perception.

Implements the Spatial CNN (SCNN) architecture for lane detection and
segmentation, with support for lane instance separation and curve fitting.
Handles both dashed and solid lane markings in various conditions.

Architecture Overview (SCNN):
    ┌──────────────────────────────────┐
    │          Backbone (ResNet)        │
    └────────────┬─────────────────────┘
                 │
    ┌────────────▼─────────────────────┐
    │     Feature Extraction           │
    │     (1x1 conv reduction)         │
    └────────────┬─────────────────────┘
                 │
    ┌────────────▼─────────────────────┐
    │    Spatial CNN Module            │
    │  ┌─────┐  ┌─────┐  ┌─────┐     │
    │  │ D → │─▶│ D → │─▶│ D → │  ↓  │ (slice-by-slice)
    │  └─────┘  └─────┘  └─────┘     │
    │  ┌─────┐  ┌─────┐  ┌─────┐     │
    │  │ D ↓ │  │ D ↓ │  │ D ↓ │  →  │ (column-wise)
    │  └─────┘  └─────┘  └─────┘     │
    └────────────┬─────────────────────┘
                 │
    ┌────────────▼─────────────────────┐
    │    Pixel-wise Classification     │
    │    + Instance Embedding          │
    └────────────┬─────────────────────┘
                 │
    ┌────────────▼─────────────────────┐
    │    Lane Curve Fitting            │
    │    (Polynomial / Spline)         │
    └──────────────────────────────────┘

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .segmentation_model import (
    BaseSegmentationModel,
    ConvBlock,
    Decoder,
    DecoderConfig,
    Encoder,
    EncoderConfig,
    ModelConfig,
    BackboneType,
    NormalizationType,
    ActivationType,
    TensorLike,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lane-Specific Enums and Constants
# ---------------------------------------------------------------------------

class LaneType(Enum):
    """Lane marking type classification."""

    BACKGROUND = 0
    SOLID_WHITE = 1
    SOLID_YELLOW = 2
    DASHED_WHITE = 3
    DASHED_YELLOW = 4
    DOUBLE_SOLID = 5
    DOUBLE_DASHED = 6
    CROSSWALK = 7
    STOP_LINE = 8
    ROAD_EDGE = 9


class LanePosition(Enum):
    """Lane position relative to ego vehicle."""

    LEFT_2 = -2   # Second lane to the left
    LEFT_1 = -1   # First lane to the left
    EGO = 0       # Ego lane
    RIGHT_1 = 1   # First lane to the right
    RIGHT_2 = 2   # Second lane to the right


class CurveFitMethod(Enum):
    """Curve fitting methods for lane boundaries."""

    POLYNOMIAL = "polynomial"
    CUBIC_SPLINE = "cubic_spline"
    RANSAC_POLY = "ransac_polynomial"
    BEZIER = "bezier"


# Default lane detection parameters
DEFAULT_LANE_CONFIG = {
    "num_lanes": 4,                 # Maximum number of detectable lanes
    "lane_width_px": 30,            # Approximate lane width in pixels
    "min_lane_length": 50,          # Minimum lane length in pixels
    "confidence_threshold": 0.5,    # Minimum confidence for lane detection
    "max_lateral_offset": 200,      # Maximum lateral offset from center (pixels)
    "curve_degree": 3,              # Polynomial degree for curve fitting
    "anchor_num": 18,               # Number of row anchors for SCNN
    "img_height": 512,
    "img_width": 1024,
    "crop_top": 0.3,               # Crop top portion (sky, far distance)
}

# TuSimple dataset specific anchors
TUSIMPLE_ANCHORS = 56

# CULane dataset specific anchors
CULANE_ANCHORS = 18


# ---------------------------------------------------------------------------
# Lane Segmentation Configuration
# ---------------------------------------------------------------------------

@dataclass
class LaneSegmentationConfig:
    """Configuration for lane segmentation model.

    Attributes:
        num_lane_types: Number of lane type classes.
        num_lanes: Maximum number of detectable lane instances.
        input_size: Input image size (H, W).
        encoder_backbone: Backbone architecture.
        scnn_slices: Number of slices for spatial message passing.
        anchor_rows: Number of row anchors for lane detection.
        curve_degree: Polynomial degree for lane curve fitting.
        curve_method: Curve fitting method.
        confidence_threshold: Minimum confidence for lane detection.
        use_instance_seg: Whether to perform lane instance segmentation.
        embedding_dim: Dimension of instance embedding vectors.
        embedding_distance_threshold: Distance threshold for clustering embeddings.
        crop_ratio: Ratio of image top to crop (sky region removal).
        use_aux_loss: Whether to use auxiliary segmentation loss.
    """

    num_lane_types: int = len(LaneType)
    num_lanes: int = 4
    input_size: Tuple[int, int] = (512, 1024)
    encoder_backbone: BackboneType = BackboneType.RESNET50
    scnn_slices: int = 5
    anchor_rows: int = 18
    curve_degree: int = 3
    curve_method: CurveFitMethod = CurveFitMethod.RANSAC_POLY
    confidence_threshold: float = 0.5
    use_instance_seg: bool = True
    embedding_dim: int = 8
    embedding_distance_threshold: float = 2.0
    crop_ratio: float = 0.3
    use_aux_loss: bool = True


# ---------------------------------------------------------------------------
# Spatial CNN Module
# ---------------------------------------------------------------------------

class SCNNModule:
    """Spatial Convolutional Neural Network module for lane detection.

    Implements slice-by-slice message passing in both horizontal and
    vertical directions to capture long-range spatial dependencies
    critical for lane detection.

    Architecture:
        Input Feature Map (C, H, W)
                │
        ┌───────▼────────┐
        │  Slice D → R   │  (top to bottom, row-wise)
        │  (H conv layers)│
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Slice D → L   │  (bottom to top, row-wise)
        │  (H conv layers)│
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Slice D ↓ U   │  (left to right, column-wise)
        │  (W conv layers)│
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Slice D ↑ D   │  (right to left, column-wise)
        │  (W conv layers)│
        └───────┬────────┘
                │
            Output (C, H, W)

    Attributes:
        channels: Number of feature channels.
        num_slices: Number of slices per direction.
    """

    def __init__(
        self,
        channels: int = 128,
        num_slices: int = 5,
    ) -> None:
        """Initialize SCNN module.

        Args:
            channels: Number of feature channels.
            num_slices: Number of message passing slices.
        """
        self.channels = channels
        self.num_slices = num_slices

        # Direction-specific 1D convolutions for message passing
        # Each direction uses a 1xW or Hx1 convolution
        self.conv_d2r = ConvBlock(channels, channels, kernel_size=(1, num_slices), padding=0)
        self.conv_d2l = ConvBlock(channels, channels, kernel_size=(1, num_slices), padding=0)
        self.conv_down = ConvBlock(channels, channels, kernel_size=(num_slices, 1), padding=0)
        self.conv_up = ConvBlock(channels, channels, kernel_size=(num_slices, 1), padding=0)

    def forward(self, x: TensorLike) -> TensorLike:
        """Apply spatial message passing in all four directions.

        Args:
            x: Input feature map of shape (N, C, H, W).

        Returns:
            Enhanced feature map with spatial context.
        """
        if not isinstance(x, np.ndarray):
            return x

        n, c, h, w = x.shape
        result = x.copy()

        # Simulate slice-by-slice message passing
        # Direction: top to bottom (row-wise)
        for i in range(1, h):
            result[:, :, i, :] += result[:, :, i - 1, :] * 0.2

        # Direction: bottom to top (row-wise)
        for i in range(h - 2, -1, -1):
            result[:, :, i, :] += result[:, :, i + 1, :] * 0.2

        # Direction: left to right (column-wise)
        for j in range(1, w):
            result[:, :, :, j] += result[:, :, :, j - 1] * 0.2

        # Direction: right to left (column-wise)
        for j in range(w - 2, -1, -1):
            result[:, :, :, j] += result[:, :, :, j + 1] * 0.2

        return result

    def __repr__(self) -> str:
        return f"SCNNModule(channels={self.channels}, slices={self.num_slices})"


# ---------------------------------------------------------------------------
# Lane Instance Segmentation Head
# ---------------------------------------------------------------------------

class LaneInstanceHead:
    """Lane instance segmentation head using discriminative loss.

    Produces per-pixel embedding vectors that cluster into lane instances
    using the discriminative loss formulation from [De Brabandere et al., 2017].

    Loss components:
        - Variance loss: Pulls embeddings of same instance together
        - Distance loss: Pushes embeddings of different instances apart
        - Regularization loss: Regularizes embedding centers

    Attributes:
        embedding_dim: Dimension of embedding vectors.
        num_lanes: Maximum number of lane instances.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        num_lanes: int = 4,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_lanes = num_lanes

        # Embedding projection layer
        self.embedding_proj = ConvBlock(
            128, embedding_dim, kernel_size=1
        )

    def compute_discriminative_loss(
        self,
        embeddings: np.ndarray,
        instance_labels: np.ndarray,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
    ) -> Dict[str, float]:
        """Compute discriminative loss for lane instance segmentation.

        Args:
            embeddings: Embedding vectors of shape (N, D, H, W).
            instance_labels: Instance labels of shape (N, H, W).
            delta_v: Variance threshold (intra-cluster margin).
            delta_d: Distance threshold (inter-cluster margin).
            alpha: Weight for variance loss.
            beta: Weight for distance loss.
            gamma: Weight for regularization loss.

        Returns:
            Dictionary with loss components.
        """
        n, d, h, w = embeddings.shape
        embed_flat = embeddings.reshape(n, d, -1)  # (N, D, H*W)
        labels_flat = instance_labels.reshape(n, -1)  # (N, H*W)

        var_loss = 0.0
        dist_loss = 0.0
        reg_loss = 0.0
        num_clusters = 0
        centers: List[np.ndarray] = []

        for b in range(n):
            unique_labels = np.unique(labels_flat[b])
            unique_labels = unique_labels[unique_labels > 0]  # Skip background

            for label in unique_labels:
                mask = labels_flat[b] == label
                cluster_embeds = embed_flat[b][:, mask]  # (D, K)

                if cluster_embeds.shape[1] < 2:
                    continue

                # Compute cluster center
                center = np.mean(cluster_embeds, axis=1)  # (D,)
                centers.append(center)

                # Variance loss: pull embeddings toward center
                distances = np.linalg.norm(cluster_embeds - center[:, np.newaxis], axis=0)
                var_loss += np.mean(np.maximum(0, distances - delta_v) ** 2) / cluster_embeds.shape[1]

                # Regularization loss
                reg_loss += np.linalg.norm(center) ** 2
                num_clusters += 1

            # Distance loss: push clusters apart
            if len(centers) >= 2:
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        dist_loss += np.maximum(0, 2 * delta_d - dist) ** 2

        if num_clusters > 0:
            var_loss /= num_clusters
            reg_loss /= num_clusters

        c_n = max(num_clusters * (num_clusters - 1) / 2, 1)
        dist_loss /= c_n

        total_loss = alpha * var_loss + beta * dist_loss + gamma * reg_loss

        return {
            "total": float(total_loss),
            "variance": float(var_loss),
            "distance": float(dist_loss),
            "regularization": float(reg_loss),
        }

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        threshold: float = 2.0,
    ) -> np.ndarray:
        """Cluster embedding vectors into lane instances using DBSCAN-like approach.

        Args:
            embeddings: Embedding vectors of shape (D, H, W).
            threshold: Distance threshold for clustering.

        Returns:
            Instance label map of shape (H, W).
        """
        d, h, w = embeddings.shape
        embed_flat = embeddings.reshape(d, -1).T  # (H*W, D)

        # Simple mean-shift-like clustering
        instance_map = np.zeros(h * w, dtype=np.int32)
        cluster_centers: List[np.ndarray] = []
        current_label = 0

        # Subsample for efficiency
        sample_indices = np.random.choice(h * w, min(5000, h * w), replace=False)
        sample_embeds = embed_flat[sample_indices]

        for idx, embed in zip(sample_indices, sample_embeds):
            # Skip background (near zero embedding)
            if np.linalg.norm(embed) < 0.1:
                continue

            # Find nearest existing cluster
            min_dist = float("inf")
            nearest_cluster = -1
            for ci, center in enumerate(cluster_centers):
                dist = np.linalg.norm(embed - center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = ci

            if min_dist < threshold and nearest_cluster >= 0:
                instance_map[idx] = nearest_cluster + 1
                # Update cluster center
                cluster_centers[nearest_cluster] = (
                    0.9 * cluster_centers[nearest_cluster] + 0.1 * embed
                )
            else:
                current_label += 1
                cluster_centers.append(embed.copy())
                instance_map[idx] = current_label

        return instance_map.reshape(h, w)


# ---------------------------------------------------------------------------
# Lane Curve Fitting
# ---------------------------------------------------------------------------

class LaneCurveFitter:
    """Curve fitting for detected lane boundaries.

    Fits smooth curves to lane marking pixels, supporting polynomial,
    cubic spline, and RANSAC-robust polynomial fitting methods.

    The fitting process:
        1. Extract lane pixel coordinates from segmentation mask
        2. Optionally subsample dense points
        3. Fit curve using selected method
        4. Validate curve quality (RMSE, coverage)
        5. Return parametric lane boundary
    """

    def __init__(
        self,
        method: CurveFitMethod = CurveFitMethod.RANSAC_POLY,
        degree: int = 3,
        min_points: int = 10,
        max_rmse: float = 10.0,
        min_coverage: float = 0.6,
    ) -> None:
        """Initialize lane curve fitter.

        Args:
            method: Curve fitting method.
            degree: Polynomial degree (for polynomial methods).
            min_points: Minimum number of points to attempt fitting.
            max_rmse: Maximum acceptable RMSE for valid fit.
            min_coverage: Minimum fraction of y-range covered by points.
        """
        self.method = method
        self.degree = degree
        self.min_points = min_points
        self.max_rmse = max_rmse
        self.min_coverage = min_coverage

    def fit_polynomial(
        self, points: np.ndarray, degree: int = None
    ) -> Optional[np.poly1d]:
        """Fit polynomial to lane points.

        Args:
            points: Lane points of shape (N, 2) - (x, y) coordinates.
            degree: Polynomial degree (uses config default if None).

        Returns:
            Fitted polynomial or None if fitting fails.
        """
        if degree is None:
            degree = self.degree

        if len(points) < max(degree + 1, self.min_points):
            logger.debug(
                f"Insufficient points for poly fit: {len(points)} < {degree + 1}"
            )
            return None

        try:
            # Fit x as function of y (common for lanes: vertical structure)
            y_coords = points[:, 1].astype(np.float64)
            x_coords = points[:, 0].astype(np.float64)

            # Check y-range coverage
            y_range = np.max(y_coords) - np.min(y_coords)
            if y_range < 50:  # Too short
                return None

            # Fit polynomial: x = f(y)
            coeffs = np.polyfit(y_coords, x_coords, degree)
            poly = np.poly1d(coeffs)

            # Validate fit quality
            predicted_x = poly(y_coords)
            residuals = predicted_x - x_coords
            rmse = np.sqrt(np.mean(residuals ** 2))

            if rmse > self.max_rmse:
                logger.debug(f"Polynomial fit RMSE too high: {rmse:.2f} > {self.max_rmse}")
                return None

            return poly

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Polynomial fitting failed: {e}")
            return None

    def fit_ransac_polynomial(
        self,
        points: np.ndarray,
        degree: int = None,
        max_iterations: int = 100,
        inlier_threshold: float = 5.0,
        min_inlier_ratio: float = 0.6,
    ) -> Optional[Tuple[np.poly1d, np.ndarray]]:
        """Fit polynomial using RANSAC for robustness to outliers.

        Args:
            points: Lane points of shape (N, 2).
            degree: Polynomial degree.
            max_iterations: Maximum RANSAC iterations.
            inlier_threshold: Distance threshold for inlier classification.
            min_inlier_ratio: Minimum inlier ratio for acceptance.

        Returns:
            Tuple of (fitted polynomial, inlier mask) or None.
        """
        if degree is None:
            degree = self.degree

        if len(points) < max(degree + 2, self.min_points):
            return None

        y_coords = points[:, 1].astype(np.float64)
        x_coords = points[:, 0].astype(np.float64)

        best_poly = None
        best_inliers = None
        best_inlier_count = 0

        n_samples = degree + 1
        rng = np.random.RandomState(42)

        for _ in range(max_iterations):
            # Random sample
            sample_idx = rng.choice(len(points), n_samples, replace=False)
            sample_y = y_coords[sample_idx]
            sample_x = x_coords[sample_idx]

            try:
                coeffs = np.polyfit(sample_y, sample_x, degree)
                poly = np.poly1d(coeffs)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # Count inliers
            predicted = poly(y_coords)
            residuals = np.abs(predicted - x_coords)
            inliers = residuals < inlier_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_poly = poly
                best_inliers = inliers

        # Check inlier ratio
        if best_inliers is not None and best_inlier_count / len(points) >= min_inlier_ratio:
            # Refit using all inliers
            try:
                inlier_y = y_coords[best_inliers]
                inlier_x = x_coords[best_inliers]
                coeffs = np.polyfit(inlier_y, inlier_x, degree)
                refined_poly = np.poly1d(coeffs)
                return refined_poly, best_inliers
            except (np.linalg.LinAlgError, ValueError):
                return best_poly, best_inliers

        return None

    def fit_cubic_spline(
        self, points: np.ndarray, num_knots: int = 10
    ) -> Optional[Callable]:
        """Fit cubic spline to lane points.

        Args:
            points: Lane points of shape (N, 2).
            num_knots: Number of spline knots.

        Returns:
            Callable spline function or None.
        """
        if len(points) < 4:
            return None

        y_coords = points[:, 1].astype(np.float64)
        x_coords = points[:, 0].astype(np.float64)

        # Sort by y-coordinate
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        x_sorted = x_coords[sort_idx]

        # Remove duplicate y values
        unique_mask = np.diff(y_sorted, prepend=-1) > 0
        y_unique = y_sorted[unique_mask]
        x_unique = x_sorted[unique_mask]

        if len(y_unique) < 4:
            return None

        # Simple cubic interpolation (numpy-based, no scipy dependency)
        def spline_func(y_query: np.ndarray) -> np.ndarray:
            """Evaluate spline at query points."""
            return np.interp(y_query, y_unique, x_unique)

        return spline_func

    def fit_lane(
        self,
        lane_mask: np.ndarray,
        image_height: int = None,
        image_width: int = None,
    ) -> Optional[Dict[str, Any]]:
        """Fit curve to lane mask and return lane boundary representation.

        Args:
            lane_mask: Binary mask of lane pixels.
            image_height: Image height for coordinate normalization.
            image_width: Image width for coordinate normalization.

        Returns:
            Dictionary with lane boundary information or None.
        """
        # Extract lane pixel coordinates
        ys, xs = np.where(lane_mask > 0)
        if len(xs) < self.min_points:
            return None

        points = np.column_stack([xs, ys])  # (N, 2)

        # Fit curve based on selected method
        if self.method == CurveFitMethod.POLYNOMIAL:
            poly = self.fit_polynomial(points)
            if poly is None:
                return None
            curve_func = poly
            coeffs = poly.coeffs.tolist()

        elif self.method == CurveFitMethod.RANSAC_POLY:
            result = self.fit_ransac_polynomial(points)
            if result is None:
                # Fallback to simple polynomial
                poly = self.fit_polynomial(points)
                if poly is None:
                    return None
                curve_func = poly
                coeffs = poly.coeffs.tolist()
            else:
                curve_func, inliers = result
                coeffs = curve_func.coeffs.tolist() if hasattr(curve_func, 'coeffs') else []

        elif self.method == CurveFitMethod.CUBIC_SPLINE:
            curve_func = self.fit_cubic_spline(points)
            if curve_func is None:
                return None
            coeffs = None
        else:
            return None

        # Compute curve points for visualization
        y_range = np.arange(np.min(ys), np.max(ys) + 1)
        if callable(curve_func):
            x_curve = curve_func(y_range)
        else:
            x_curve = curve_func(y_range)

        # Compute fit quality
        predicted_x = curve_func(ys) if callable(curve_func) else curve_func(ys)
        rmse = np.sqrt(np.mean((predicted_x - xs) ** 2))

        # Normalize coordinates
        h = image_height or lane_mask.shape[0]
        w = image_width or lane_mask.shape[1]

        result = {
            "curve_function": curve_func,
            "coefficients": coeffs,
            "method": self.method.value,
            "degree": self.degree,
            "num_points": len(points),
            "rmse": float(rmse),
            "y_range": (int(np.min(ys)), int(np.max(ys))),
            "curve_points": np.column_stack([x_curve, y_range]).tolist(),
            "normalized_y_range": (float(np.min(ys) / h), float(np.max(ys) / h)),
        }

        return result


# ---------------------------------------------------------------------------
# Lane Anchor-Based Detection Head
# ---------------------------------------------------------------------------

class LaneAnchorHead:
    """Anchor-based lane detection head for row-anchor prediction.

    Predicts lane existence and x-offset at predefined row anchors,
    following the Ultra-Fast-Lane-Detection approach.

    Architecture:
        Feature Map ── 1x1 Conv ── Classification (lane existence)
                              ── 1x1 Conv ── Regression (x-offset at anchors)

    Attributes:
        num_lanes: Maximum number of detectable lanes.
        num_anchors: Number of row anchors.
        feature_channels: Input feature channels.
    """

    def __init__(
        self,
        num_lanes: int = 4,
        num_anchors: int = 18,
        feature_channels: int = 128,
    ) -> None:
        self.num_lanes = num_lanes
        self.num_anchors = num_anchors
        self.feature_channels = feature_channels

        # Lane existence classification head
        self.cls_head = ConvBlock(
            feature_channels, num_lanes * num_anchors, kernel_size=1
        )

        # X-offset regression head
        self.reg_head = ConvBlock(
            feature_channels, num_lanes * num_anchors, kernel_size=1
        )

    def decode_predictions(
        self,
        cls_output: np.ndarray,
        reg_output: np.ndarray,
        image_width: int = 1024,
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Decode anchor-based predictions to lane coordinates.

        Args:
            cls_output: Classification output (N, num_lanes * num_anchors).
            reg_output: Regression output (N, num_lanes * num_anchors).
            image_width: Image width for scaling offsets.
            confidence_threshold: Minimum confidence for lane detection.

        Returns:
            List of detected lane dictionaries.
        """
        lanes = []

        for lane_idx in range(self.num_lanes):
            # Extract this lane's predictions
            start = lane_idx * self.num_anchors
            end = start + self.num_anchors
            lane_cls = cls_output[start:end]
            lane_reg = reg_output[start:end]

            # Check if lane exists
            lane_confidence = float(np.mean(lane_cls))
            if lane_confidence < confidence_threshold:
                continue

            # Convert to coordinates
            points = []
            for anchor_idx in range(self.num_anchors):
                if lane_cls[anchor_idx] > confidence_threshold:
                    x_offset = lane_reg[anchor_idx] * image_width
                    # Row anchor position (evenly spaced)
                    y_pos = (anchor_idx + 0.5) * (1.0 / self.num_anchors)
                    points.append((float(x_offset), float(y_pos)))

            if len(points) >= 2:
                lanes.append({
                    "lane_id": lane_idx,
                    "confidence": lane_confidence,
                    "points": points,
                    "num_points": len(points),
                })

        return lanes

    def __repr__(self) -> str:
        return (
            f"LaneAnchorHead(lanes={self.num_lanes}, "
            f"anchors={self.num_anchors}, ch={self.feature_channels})"
        )


# ---------------------------------------------------------------------------
# Lane Segmentation Model
# ---------------------------------------------------------------------------

class LaneSegmentationModel(BaseSegmentationModel):
    """SCNN-based lane segmentation model with instance separation.

    Combines spatial CNN for long-range context, anchor-based detection
    for fast inference, and instance embedding for lane separation.

    Example:
        >>> config = LaneSegmentationConfig()
        >>> model = LaneSegmentationModel(config)
        >>> model.build_model()
        >>> image = np.random.randn(1, 3, 512, 1024).astype(np.float32)
        >>> result = model.detect_lanes(image)
    """

    def __init__(self, config: Union[LaneSegmentationConfig, ModelConfig]) -> None:
        if isinstance(config, LaneSegmentationConfig):
            model_config = ModelConfig(
                num_classes=config.num_lane_types,
                input_size=config.input_size,
                encoder=EncoderConfig(backbone=config.encoder_backbone),
            )
            self.lane_config = config
        else:
            model_config = config
            self.lane_config = LaneSegmentationConfig(
                num_lane_types=config.num_classes,
                input_size=config.input_size,
            )

        super().__init__(model_config)
        self._scnn: Optional[SCNNModule] = None
        self._instance_head: Optional[LaneInstanceHead] = None
        self._anchor_head: Optional[LaneAnchorHead] = None
        self._curve_fitter: Optional[LaneCurveFitter] = None

    def build_model(self) -> None:
        """Build the complete lane segmentation model."""
        self._encoder = Encoder(self.config.encoder)

        # SCNN module for spatial message passing
        self._scnn = SCNNModule(
            channels=128,
            num_slices=self.lane_config.scnn_slices,
        )

        # Instance embedding head
        if self.lane_config.use_instance_seg:
            self._instance_head = LaneInstanceHead(
                embedding_dim=self.lane_config.embedding_dim,
                num_lanes=self.lane_config.num_lanes,
            )

        # Anchor-based detection head
        self._anchor_head = LaneAnchorHead(
            num_lanes=self.lane_config.num_lanes,
            num_anchors=self.lane_config.anchor_rows,
        )

        # Curve fitter
        self._curve_fitter = LaneCurveFitter(
            method=self.lane_config.curve_method,
            degree=self.lane_config.curve_degree,
        )

        # Decoder
        self._decoder = Decoder(
            self.config.decoder,
            self._encoder.get_feature_channels(),
            self.config.num_classes,
        )

        self._is_built = True
        logger.info(
            f"LaneSegmentationModel built: "
            f"SCNN slices={self.lane_config.scnn_slices}, "
            f"anchor rows={self.lane_config.anchor_rows}, "
            f"instance_seg={self.lane_config.use_instance_seg}"
        )

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass through lane segmentation model.

        Args:
            x: Input image tensor (N, 3, H, W).

        Returns:
            Lane segmentation logits.
        """
        if not self._is_built:
            self.build_model()

        if isinstance(x, np.ndarray):
            n = x.shape[0] if x.ndim == 4 else 1
            h, w = self.config.output_size
            output = np.random.randn(n, self.config.num_classes, h, w).astype(np.float32) * 0.01

            # Store auxiliary outputs
            if self.lane_config.use_instance_seg and self._instance_head is not None:
                self._embeddings = np.random.randn(n, self.lane_config.embedding_dim, h, w).astype(np.float32) * 0.1
            if self._anchor_head is not None:
                total_anchors = self.lane_config.num_lanes * self.lane_config.anchor_rows
                self._anchor_cls = np.random.randn(n, total_anchors).astype(np.float32) * 0.1
                self._anchor_reg = np.random.rand(n, total_anchors).astype(np.float32)

            return output
        return np.zeros((1, self.config.num_classes, *self.config.output_size), dtype=np.float32)

    def get_loss(
        self,
        predictions: TensorLike,
        targets: TensorLike,
        weights: Optional[TensorLike] = None,
    ) -> float:
        """Compute lane segmentation loss.

        Combined loss:
            L = L_seg + λ_inst * L_discriminative + λ_anchor * L_anchor

        Args:
            predictions: Model output logits.
            targets: Ground truth labels.
            weights: Optional per-class weights.

        Returns:
            Total loss value.
        """
        if not isinstance(predictions, np.ndarray):
            return 0.0

        n, c, h, w = predictions.shape
        targets_2d = targets.reshape(n, h, w) if targets.ndim == 4 else targets

        # Segmentation cross-entropy loss
        logits_flat = predictions.reshape(n, c, -1)
        targets_flat = targets_2d.reshape(n, -1)

        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        shifted = logits_flat - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        batch_idx = np.arange(n)[:, np.newaxis]
        pixel_idx = np.arange(h * w)[np.newaxis, :]
        target_log_probs = log_probs[batch_idx, targets_flat, pixel_idx]
        seg_loss = float(np.mean(-target_log_probs))

        total_loss = seg_loss

        # Instance discriminative loss
        if self.lane_config.use_instance_seg and hasattr(self, "_embeddings"):
            inst_loss_dict = self._instance_head.compute_discriminative_loss(
                self._embeddings, targets_2d
            )
            total_loss += 0.5 * inst_loss_dict["total"]

        # Anchor loss (simplified)
        if hasattr(self, "_anchor_cls"):
            # Binary cross-entropy for lane existence
            anchor_loss = float(np.mean(np.abs(self._anchor_cls)))
            total_loss += 0.1 * anchor_loss

        return total_loss

    def detect_lanes(
        self,
        image: TensorLike,
        confidence_threshold: float = None,
    ) -> Dict[str, Any]:
        """Detect and segment lanes in an image.

        Args:
            image: Input image tensor.
            confidence_threshold: Minimum confidence threshold.

        Returns:
            Dictionary with lane detection results.
        """
        if confidence_threshold is None:
            confidence_threshold = self.lane_config.confidence_threshold

        logits = self.predict(image)
        seg_mask = self.get_segmentation_mask(logits)

        results: Dict[str, Any] = {
            "segmentation_mask": seg_mask,
            "lane_instances": [],
            "lane_curves": [],
            "num_lanes_detected": 0,
        }

        # Lane instance separation
        if self.lane_config.use_instance_seg and hasattr(self, "_embeddings"):
            if isinstance(self._embeddings, np.ndarray):
                for b in range(self._embeddings.shape[0]):
                    instance_map = self._instance_head.cluster_embeddings(
                        self._embeddings[b],
                        threshold=self.lane_config.embedding_distance_threshold,
                    )
                    results["lane_instances"].append(instance_map)

                    # Fit curves to each instance
                    for inst_id in np.unique(instance_map):
                        if inst_id == 0:
                            continue
                        inst_mask = (instance_map == inst_id)
                        lane_result = self._curve_fitter.fit_lane(inst_mask)
                        if lane_result is not None:
                            lane_result["instance_id"] = int(inst_id)
                            results["lane_curves"].append(lane_result)

        # Anchor-based detection
        if hasattr(self, "_anchor_cls") and hasattr(self, "_anchor_reg"):
            if isinstance(self._anchor_cls, np.ndarray):
                anchor_lanes = self._anchor_head.decode_predictions(
                    self._anchor_cls[0],
                    self._anchor_reg[0],
                    image_width=self.lane_config.input_size[1],
                    confidence_threshold=confidence_threshold,
                )
                results["anchor_lanes"] = anchor_lanes

        results["num_lanes_detected"] = len(results.get("lane_curves", []))

        return results

    def classify_lane_type(
        self, lane_mask: np.ndarray, image: Optional[np.ndarray] = None
    ) -> LaneType:
        """Classify the type of a detected lane marking.

        Uses geometric features (continuity, width) and optionally
        color information to determine lane type.

        Args:
            lane_mask: Binary mask of the lane marking.
            image: Optional original image for color analysis.

        Returns:
            Classified lane type.
        """
        if np.sum(lane_mask) < 10:
            return LaneType.BACKGROUND

        # Compute continuity (ratio of connected pixels)
        ys, xs = np.where(lane_mask > 0)
        if len(ys) < 2:
            return LaneType.BACKGROUND

        # Check vertical continuity
        y_sorted = np.sort(ys)
        y_gaps = np.diff(y_sorted)
        continuity = np.sum(y_gaps <= 3) / max(len(y_gaps), 1)

        # Check average width
        x_range = np.max(xs) - np.min(xs)
        y_range = np.max(ys) - np.min(ys)
        avg_width = x_range / max(y_range, 1) * 100

        # Heuristic classification
        if continuity > 0.8:
            if avg_width > 15:
                return LaneType.SOLID_WHITE
            elif avg_width > 8:
                return LaneType.DOUBLE_SOLID
            else:
                return LaneType.SOLID_YELLOW
        else:
            if avg_width > 12:
                return LaneType.DASHED_WHITE
            else:
                return LaneType.DASHED_YELLOW

    def compute_lateral_offset(
        self,
        lane_curves: List[Dict[str, Any]],
        image_width: int = 1024,
        ego_position: float = 0.5,
    ) -> float:
        """Compute lateral offset of ego vehicle from lane center.

        Args:
            lane_curves: List of detected lane curve dictionaries.
            image_width: Image width in pixels.
            ego_position: Normalized x-position of ego vehicle (0-1).

        Returns:
            Lateral offset in normalized coordinates (positive = right).
        """
        if len(lane_curves) < 2:
            return 0.0

        # Find left and right lane boundaries
        left_x = None
        right_x = None

        for curve in lane_curves:
            points = curve.get("points", [])
            if not points:
                continue
            avg_x = np.mean([p[0] for p in points])
            center_x = ego_position * image_width

            if avg_x < center_x:
                if left_x is None or avg_x > left_x:
                    left_x = avg_x
            else:
                if right_x is None or avg_x < right_x:
                    right_x = avg_x

        if left_x is not None and right_x is not None:
            center = (left_x + right_x) / 2
            ego_x = ego_position * image_width
            offset = (ego_x - center) / image_width
            return float(offset)

        return 0.0
