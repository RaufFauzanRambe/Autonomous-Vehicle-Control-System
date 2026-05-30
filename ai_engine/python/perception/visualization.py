"""
Visualization Module for Semantic Segmentation in Autonomous Driving.

Provides comprehensive visualization utilities including:
    - Segmentation mask overlay with alpha blending on original images
    - Color-coded class visualization using Cityscapes palette
    - Side-by-side comparison of prediction vs ground truth
    - Error maps highlighting misclassified regions
    - Confusion matrix heatmap visualization
    - Per-class IoU bar charts
    - Boundary quality visualization
    - Uncertainty map visualization
    - Multi-image grid layouts

Usage:
    from semantic_segmentation.visualization import SegmentationVisualizer

    visualizer = SegmentationVisualizer(num_classes=19)
    overlay = visualizer.overlay_mask(image, pred_mask, alpha=0.5)
    comparison = visualizer.create_comparison(image, pred_mask, gt_mask)
    error_map = visualizer.create_error_map(pred_mask, gt_mask)

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .segmentation_utils import (
    CITYSCAPES_CLASSES,
    colorize_mask,
    generate_distinct_palette,
    get_cityscapes_palette,
    get_kitti_palette,
    get_road_palette,
    get_lane_palette,
    compute_boundary_mask,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color Palette Management
# ---------------------------------------------------------------------------

class PaletteManager:
    """Manages color palettes for segmentation visualization.

    Provides centralized access to standard and custom palettes,
    with support for palette generation and modification.

    Attributes:
        num_classes: Number of segmentation classes.
        palette: Current active color palette.
    """

    def __init__(
        self,
        num_classes: int = 19,
        palette_type: str = "cityscapes",
        custom_palette: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize palette manager.

        Args:
            num_classes: Number of segmentation classes.
            palette_type: Palette type ('cityscapes', 'kitti', 'road', 'lane', 'custom').
            custom_palette: Custom palette array of shape (num_classes, 3).
        """
        self.num_classes = num_classes
        self.palette_type = palette_type

        if custom_palette is not None:
            self.palette = custom_palette
        elif palette_type == "cityscapes":
            self.palette = get_cityscapes_palette()
        elif palette_type == "kitti":
            self.palette = get_kitti_palette()
        elif palette_type == "road":
            self.palette = get_road_palette()
        elif palette_type == "lane":
            self.palette = get_lane_palette()
        else:
            self.palette = generate_distinct_palette(num_classes)

        # Ensure palette has enough entries
        if len(self.palette) < num_classes:
            extra = generate_distinct_palette(
                num_classes - len(self.palette),
                seed=len(self.palette),
            )
            self.palette = np.vstack([self.palette, extra])

    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get RGB color for a specific class.

        Args:
            class_id: Class index.

        Returns:
            RGB color tuple.
        """
        if class_id < len(self.palette):
            return tuple(int(c) for c in self.palette[class_id])
        return (0, 0, 0)

    def get_class_name(self, class_id: int) -> str:
        """Get human-readable name for a class.

        Args:
            class_id: Class index.

        Returns:
            Class name string.
        """
        if class_id < len(CITYSCAPES_CLASSES):
            return CITYSCAPES_CLASSES[class_id]["name"]
        return f"class_{class_id}"


# ---------------------------------------------------------------------------
# Main Visualizer
# ---------------------------------------------------------------------------

class SegmentationVisualizer:
    """Comprehensive visualization toolkit for semantic segmentation.

    Provides methods for creating publication-quality visualizations
    of segmentation results, including overlays, comparisons, error
    maps, and statistical charts.

    Example:
        >>> visualizer = SegmentationVisualizer(num_classes=19)
        >>> overlay = visualizer.overlay_mask(image, pred_mask, alpha=0.5)
        >>> comparison = visualizer.create_comparison(image, pred_mask, gt_mask)
        >>> error_map = visualizer.create_error_map(pred_mask, gt_mask)
        >>> confusion_vis = visualizer.visualize_confusion_matrix(conf_matrix)
    """

    # Default Cityscapes class names
    CITYSCAPES_NAMES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle",
    ]

    def __init__(
        self,
        num_classes: int = 19,
        palette_type: str = "cityscapes",
        class_names: Optional[List[str]] = None,
        custom_palette: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize visualizer.

        Args:
            num_classes: Number of segmentation classes.
            palette_type: Color palette type.
            class_names: Human-readable class names.
            custom_palette: Custom color palette.
        """
        self.num_classes = num_classes
        self.palette_manager = PaletteManager(
            num_classes=num_classes,
            palette_type=palette_type,
            custom_palette=custom_palette,
        )
        self.class_names = class_names or self.CITYSCAPES_NAMES[:num_classes]
        self.palette = self.palette_manager.palette

    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        ignore_class: int = -1,
    ) -> np.ndarray:
        """Overlay segmentation mask on original image with alpha blending.

        Creates a semi-transparent overlay of the colorized segmentation
        mask on top of the original image, allowing visual assessment
        of segmentation quality.

        Args:
            image: Original image (H, W, 3) as uint8 [0, 255].
            mask: Segmentation mask (H, W) with class indices.
            alpha: Blending factor (0 = image only, 1 = mask only).
            ignore_class: Class ID to skip in overlay (-1 = none).

        Returns:
            Blended image (H, W, 3) as uint8.
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[0] in (3, 1):
            image = np.transpose(image, (1, 2, 0))

        # Ensure image is uint8 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        h, w = mask.shape
        if image.shape[:2] != (h, w):
            # Resize image to match mask
            resized = np.zeros((h, w, 3), dtype=np.uint8)
            oh, ow = image.shape[:2]
            for y in range(h):
                src_y = min(int(y * oh / h), oh - 1)
                for x in range(w):
                    src_x = min(int(x * ow / w), ow - 1)
                    resized[y, x] = image[src_y, src_x]
            image = resized

        # Colorize mask
        colorized = colorize_mask(mask, self.palette, self.num_classes)

        # Alpha blending
        blended = image.copy().astype(np.float32)
        mask_region = np.ones((h, w), dtype=bool)

        if ignore_class >= 0:
            mask_region = mask != ignore_class

        alpha_3d = np.zeros((h, w, 1), dtype=np.float32)
        alpha_3d[mask_region] = alpha

        blended = blended * (1 - alpha_3d) + colorized.astype(np.float32) * alpha_3d
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended

    def create_comparison(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create side-by-side comparison of prediction and ground truth.

        Layout: [Original | GT Overlay | Pred Overlay]

        Args:
            image: Original image (H, W, 3) as uint8.
            pred_mask: Predicted segmentation mask (H, W).
            gt_mask: Ground truth mask (H, W).
            alpha: Overlay blending factor.

        Returns:
            Comparison image (H, W*3, 3) as uint8.
        """
        # Ensure consistent dimensions
        if image.ndim == 3 and image.shape[0] in (3, 1):
            image = np.transpose(image, (1, 2, 0))

        # Normalize image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Create individual panels
        original = image.copy()
        gt_overlay = self.overlay_mask(image, gt_mask, alpha=alpha)
        pred_overlay = self.overlay_mask(image, pred_mask, alpha=alpha)

        # Add labels
        original = self._add_text_label(original, "Original")
        gt_overlay = self._add_text_label(gt_overlay, "Ground Truth")
        pred_overlay = self._add_text_label(pred_overlay, "Prediction")

        # Concatenate horizontally
        if original.shape[:2] != gt_overlay.shape[:2]:
            # Resize to match
            target_h, target_w = original.shape[:2]
            gt_overlay = self._resize_image(gt_overlay, (target_h, target_w))
            pred_overlay = self._resize_image(pred_overlay, (target_h, target_w))

        comparison = np.concatenate([original, gt_overlay, pred_overlay], axis=1)

        return comparison

    def create_error_map(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        error_color: Tuple[int, int, int] = (255, 0, 0),
        correct_color: Tuple[int, int, int] = (0, 255, 0),
        ignore_label: int = 255,
    ) -> np.ndarray:
        """Create error map highlighting misclassified regions.

        Colors:
            - Green: Correctly classified pixels
            - Red: Misclassified pixels
            - Black: Ignored pixels

        Args:
            pred_mask: Predicted segmentation mask (H, W).
            gt_mask: Ground truth mask (H, W).
            error_color: RGB color for errors.
            correct_color: RGB color for correct predictions.
            ignore_label: Label to ignore.

        Returns:
            Error map (H, W, 3) as uint8.
        """
        h, w = pred_mask.shape
        error_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Valid pixels
        valid = gt_mask != ignore_label

        # Correct predictions
        correct = (pred_mask == gt_mask) & valid
        error_map[correct] = correct_color

        # Misclassified pixels
        errors = (pred_mask != gt_mask) & valid
        error_map[errors] = error_color

        # Ignored pixels remain black
        return error_map

    def create_boundary_comparison(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Visualize boundary quality by comparing predicted and GT boundaries.

        Draws predicted boundaries in one color and GT boundaries in another
        on top of the original image or a blank canvas.

        Args:
            pred_mask: Predicted segmentation mask (H, W).
            gt_mask: Ground truth mask (H, W).
            image: Optional background image (H, W, 3).

        Returns:
            Boundary comparison image (H, W, 3) as uint8.
        """
        h, w = pred_mask.shape

        if image is not None:
            canvas = image.copy()
            if canvas.max() <= 1.0:
                canvas = (canvas * 255).astype(np.uint8)
        else:
            canvas = np.full((h, w, 3), 240, dtype=np.uint8)

        # Compute boundaries
        pred_boundary = compute_boundary_mask(pred_mask)
        gt_boundary = compute_boundary_mask(gt_mask)

        # Draw GT boundaries (green)
        gt_pixels = gt_boundary > 0
        canvas[gt_pixels] = [0, 200, 0]

        # Draw predicted boundaries (red), drawn after so they overlay
        pred_pixels = pred_boundary > 0
        canvas[pred_pixels] = [200, 0, 0]

        # Overlapping boundaries (yellow)
        overlap = (pred_boundary > 0) & (gt_boundary > 0)
        canvas[overlap] = [200, 200, 0]

        return canvas

    def visualize_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        normalize: bool = True,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        title: str = "Confusion Matrix",
    ) -> np.ndarray:
        """Visualize confusion matrix as a heatmap image.

        Args:
            confusion_matrix: Confusion matrix of shape (N, N).
            normalize: Whether to normalize by row (ground truth).
            class_names: Class name labels.
            figsize: Figure size (width, height) in inches.
            title: Plot title.

        Returns:
            Confusion matrix visualization as (H, W, 3) uint8 array.
        """
        matrix = confusion_matrix.copy().astype(np.float64)

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums = np.clip(row_sums, 1, None)
            matrix = matrix / row_sums

        n = matrix.shape[0]
        names = class_names or self.class_names[:n]

        # Create heatmap visualization
        cell_size = 40  # pixels per cell
        label_height = 80
        label_width = 120
        canvas_h = n * cell_size + label_height
        canvas_w = n * cell_size + label_width

        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        # Draw cells
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                # Map value to color (0=white, 1=dark blue)
                intensity = 1.0 - value
                r = int(intensity * 255)
                g = int(intensity * 255)
                b = int(255 - intensity * 100)

                y_start = label_height + i * cell_size
                x_start = label_width + j * cell_size
                canvas[y_start:y_start + cell_size, x_start:x_start + cell_size] = [r, g, b]

                # Draw value text
                text_color = (255, 255, 255) if value > 0.5 else (0, 0, 0)
                # Simple text rendering: just mark the cell with the value range
                if value > 0.01:
                    mid_y = y_start + cell_size // 2
                    mid_x = x_start + cell_size // 2
                    # Draw a small indicator
                    indicator_size = max(2, int(value * cell_size * 0.3))
                    y1 = max(mid_y - indicator_size, y_start + 1)
                    y2 = min(mid_y + indicator_size, y_start + cell_size - 1)
                    x1 = max(mid_x - indicator_size, x_start + 1)
                    x2 = min(mid_x + indicator_size, x_start + cell_size - 1)
                    canvas[y1:y2, x1:x2] = text_color

        # Draw grid lines
        for i in range(n + 1):
            y = label_height + i * cell_size
            if y < canvas_h:
                canvas[y, label_width:canvas_w] = [128, 128, 128]
            x = label_width + i * cell_size
            if x < canvas_w:
                canvas[label_height:canvas_h, x] = [128, 128, 128]

        # Draw diagonal highlight
        for i in range(n):
            y_start = label_height + i * cell_size
            x_start = label_width + i * cell_size
            # Slightly brighter diagonal
            canvas[y_start:y_start + cell_size, x_start:x_start + cell_size] = \
                np.clip(
                    canvas[y_start:y_start + cell_size, x_start:x_start + cell_size].astype(np.int16) + 30,
                    0, 255
                ).astype(np.uint8)

        return canvas

    def create_per_class_iou_bar_chart(
        self,
        iou_values: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Per-Class IoU",
        bar_width: int = 30,
        chart_height: int = 400,
    ) -> np.ndarray:
        """Create a bar chart visualization of per-class IoU.

        Args:
            iou_values: Array of IoU values per class.
            class_names: Class name labels.
            title: Chart title.
            bar_width: Width of each bar in pixels.
            chart_height: Chart height in pixels.

        Returns:
            Bar chart image (H, W, 3) as uint8.
        """
        n = len(iou_values)
        names = class_names or self.class_names[:n]

        margin_left = 100
        margin_right = 40
        margin_top = 60
        margin_bottom = 80

        chart_w = margin_left + n * bar_width + margin_right
        chart_h = margin_top + chart_height + margin_bottom

        canvas = np.full((chart_h, chart_w, 3), 255, dtype=np.uint8)

        # Draw axes
        # Y-axis
        canvas[margin_top:margin_top + chart_height, margin_left] = [0, 0, 0]
        # X-axis
        canvas[margin_top + chart_height, margin_left:margin_left + n * bar_width] = [0, 0, 0]

        # Draw grid lines and Y-axis labels
        for val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y = margin_top + int((1.0 - val) * chart_height)
            canvas[y, margin_left:margin_left + n * bar_width] = [200, 200, 200]
            # Y-axis label
            label = f"{val:.1f}"
            x_start = max(0, margin_left - len(label) * 8 - 5)
            canvas[y, x_start:margin_left - 2] = [0, 0, 0]

        # Draw bars
        for i in range(n):
            iou = float(iou_values[i])
            bar_height = max(1, int(iou * chart_height))

            x_start = margin_left + i * bar_width + 2
            x_end = x_start + bar_width - 4
            y_start = margin_top + chart_height - bar_height
            y_end = margin_top + chart_height

            # Color based on IoU value
            if iou >= 0.7:
                color = [46, 139, 87]  # Green
            elif iou >= 0.5:
                color = [255, 165, 0]  # Orange
            elif iou >= 0.3:
                color = [255, 100, 50]  # Dark orange
            else:
                color = [220, 20, 60]  # Red

            if x_end > x_start and y_end > y_start:
                canvas[y_start:y_end, x_start:x_end] = color

            # X-axis class name (abbreviated)
            name = names[i] if i < len(names) else f"{i}"
            short_name = name[:6] if len(name) > 6 else name
            # Simple text area for label
            label_y = margin_top + chart_height + 5
            label_x = x_start
            canvas[label_y:label_y + 3, label_x:min(label_x + bar_width - 4, chart_w)] = [0, 0, 0]

        # Title area
        canvas[5:8, margin_left:margin_left + min(200, n * bar_width)] = [0, 0, 0]

        return canvas

    def create_multi_overlay(
        self,
        image: np.ndarray,
        masks: Dict[str, np.ndarray],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create overlay with multiple named masks in a grid.

        Args:
            image: Original image (H, W, 3).
            masks: Dictionary mapping mask names to mask arrays.
            alpha: Blending factor.

        Returns:
            Grid of overlaid images (H*rows, W*cols, 3) as uint8.
        """
        n_masks = len(masks)
        if n_masks == 0:
            return image

        # Calculate grid dimensions
        cols = min(3, n_masks + 1)  # +1 for original
        rows = math.ceil((n_masks + 1) / cols)

        # Ensure image is uint8
        if image.ndim == 3 and image.shape[0] in (3, 1):
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        grid = np.full((h * rows, w * cols, 3), 255, dtype=np.uint8)

        # Place original image
        grid[:h, :w] = image
        grid[:h, :w] = self._add_text_label(grid[:h, :w], "Original")

        # Place overlays
        panels = [("Original", image)]
        for name, mask in masks.items():
            overlay = self.overlay_mask(image, mask, alpha=alpha)
            overlay = self._add_text_label(overlay, name)
            panels.append((name, overlay))

        for idx, (name, panel) in enumerate(panels):
            row = idx // cols
            col = idx % cols
            y_start = row * h
            x_start = col * w
            y_end = min(y_start + h, grid.shape[0])
            x_end = min(x_start + w, grid.shape[1])
            grid[y_start:y_end, x_start:x_end] = panel[:y_end - y_start, :x_end - x_start]

        return grid

    def create_uncertainty_map(
        self,
        probabilities: np.ndarray,
        method: str = "entropy",
        colormap: str = "jet",
    ) -> np.ndarray:
        """Create uncertainty visualization from class probabilities.

        Args:
            probabilities: Class probabilities (C, H, W) or (H, W).
            method: Uncertainty method ('entropy', 'margin', 'max_prob').
            colormap: Colormap name ('jet', 'hot', 'viridis').

        Returns:
            Uncertainty map (H, W, 3) as uint8.
        """
        if probabilities.ndim == 3 and probabilities.shape[0] > 1:
            # Multi-class probabilities (C, H, W)
            if method == "entropy":
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=0)
                max_entropy = np.log(probabilities.shape[0])
                uncertainty = entropy / max_entropy
            elif method == "margin":
                sorted_probs = np.sort(probabilities, axis=0)
                margin = sorted_probs[-1] - sorted_probs[-2]
                uncertainty = 1.0 - margin
            else:  # max_prob
                uncertainty = 1.0 - np.max(probabilities, axis=0)
        else:
            # Single-channel probability
            if probabilities.ndim == 3:
                probs = probabilities[0]
            else:
                probs = probabilities
            uncertainty = 1.0 - probs

        # Normalize to [0, 1]
        uncertainty = np.clip(uncertainty, 0, 1)

        # Apply colormap
        vis = self._apply_colormap(uncertainty, colormap)
        return vis

    def create_legend(
        self,
        class_ids: Optional[List[int]] = None,
        swatch_size: int = 20,
        max_cols: int = 3,
    ) -> np.ndarray:
        """Create a color legend for segmentation classes.

        Args:
            class_ids: List of class IDs to include (None = all).
            swatch_size: Size of color swatch in pixels.
            max_cols: Maximum number of columns.

        Returns:
            Legend image (H, W, 3) as uint8.
        """
        if class_ids is None:
            class_ids = list(range(self.num_classes))

        n = len(class_ids)
        cols = min(max_cols, n)
        rows = math.ceil(n / cols)

        entry_height = swatch_size + 8
        text_width = 120
        entry_width = swatch_size + text_width + 10

        canvas_h = rows * entry_height + 10
        canvas_w = cols * entry_width + 10
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        for idx, cls_id in enumerate(class_ids):
            row = idx // cols
            col = idx % cols

            x = 5 + col * entry_width
            y = 5 + row * entry_height

            # Draw color swatch
            color = self.palette_manager.get_class_color(cls_id)
            canvas[y:y + swatch_size, x:x + swatch_size] = color

            # Draw border around swatch
            canvas[y, x:x + swatch_size] = [0, 0, 0]
            canvas[y + swatch_size - 1, x:x + swatch_size] = [0, 0, 0]
            canvas[y:y + swatch_size, x] = [0, 0, 0]
            canvas[y:y + swatch_size, x + swatch_size - 1] = [0, 0, 0]

            # Text area (simple background for readability)
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
            text_x = x + swatch_size + 5
            text_y = y + 2
            # Simple text representation: small dark rectangle
            canvas[text_y:text_y + swatch_size - 4,
                   text_x:text_x + min(text_width, entry_width - swatch_size - 15)] = [240, 240, 240]

        return canvas

    def _add_text_label(
        self,
        image: np.ndarray,
        text: str,
        position: str = "top_left",
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Add a text label to an image.

        Simple text rendering using a colored rectangle as background.

        Args:
            image: Input image (H, W, 3).
            text: Label text.
            position: Position ('top_left', 'top_right', 'bottom_left').
            bg_color: Background color.
            text_color: Text color (used as indicator).

        Returns:
            Image with label overlay.
        """
        result = image.copy()
        h, w = result.shape[:2]

        # Calculate label dimensions
        label_h = 24
        label_w = min(len(text) * 9 + 12, w)

        if position == "top_left":
            y1, y2 = 4, 4 + label_h
            x1, x2 = 4, 4 + label_w
        elif position == "top_right":
            y1, y2 = 4, 4 + label_h
            x1, x2 = w - label_w - 4, w - 4
        else:
            y1, y2 = h - label_h - 4, h - 4
            x1, x2 = 4, 4 + label_w

        # Ensure bounds
        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)

        # Draw semi-transparent background
        if y2 > y1 and x2 > x1:
            bg = np.array(bg_color, dtype=np.uint8)
            result[y1:y2, x1:x2] = (result[y1:y2, x1:x2].astype(np.float32) * 0.4 +
                                     bg.astype(np.float32) * 0.6).astype(np.uint8)
            # Draw border
            result[y1, x1:x2] = bg
            result[y2 - 1, x1:x2] = bg
            result[y1:y2, x1] = bg
            result[y1:y2, x2 - 1] = bg

            # Draw text indicator line
            text_y = y1 + label_h // 2
            text_x_start = x1 + 6
            text_x_end = min(x2 - 6, text_x_start + len(text) * 7)
            if text_x_end > text_x_start:
                result[text_y:text_y + 2, text_x_start:text_x_end] = text_color

        return result

    def _apply_colormap(
        self,
        values: np.ndarray,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Apply colormap to a 2D array of values in [0, 1].

        Args:
            values: 2D array of normalized values.
            colormap: Colormap name.

        Returns:
            Colorized image (H, W, 3) as uint8.
        """
        h, w = values.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)

        if colormap == "jet":
            # Jet colormap approximation
            for y in range(h):
                for x in range(w):
                    v = float(np.clip(values[y, x], 0, 1))
                    if v < 0.25:
                        r, g, b = 0, int(v * 4 * 255), 255
                    elif v < 0.5:
                        r, g, b = 0, 255, int((0.5 - v) * 4 * 255)
                    elif v < 0.75:
                        r, g, b = int((v - 0.5) * 4 * 255), 255, 0
                    else:
                        r, g, b = 255, int((1.0 - v) * 4 * 255), 0
                    result[y, x] = [r, g, b]
        elif colormap == "hot":
            for y in range(h):
                for x in range(w):
                    v = float(np.clip(values[y, x], 0, 1))
                    r = int(min(v * 3, 1.0) * 255)
                    g = int(max(0, min((v - 0.33) * 3, 1.0)) * 255)
                    b = int(max(0, min((v - 0.66) * 3, 1.0)) * 255)
                    result[y, x] = [r, g, b]
        elif colormap == "viridis":
            # Simplified viridis approximation
            for y in range(h):
                for x in range(w):
                    v = float(np.clip(values[y, x], 0, 1))
                    r = int((0.267 + v * 0.329 + v * v * (-1.424) + v * v * v * 1.828) * 255)
                    g = int((0.004 + v * 1.260 + v * v * (-0.956) + v * v * v * 0.692) * 255)
                    b = int((0.329 + v * (-0.703) + v * v * (1.327) + v * v * v * (-0.953)) * 255)
                    result[y, x] = [max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))]
        else:
            # Grayscale fallback
            gray = (values * 255).astype(np.uint8)
            result[:, :, 0] = gray
            result[:, :, 1] = gray
            result[:, :, 2] = gray

        return result

    @staticmethod
    def _resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize image to target size using nearest-neighbor.

        Args:
            image: Input image (H, W, C) or (H, W).
            target_size: Target size (H', W').

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]
        th, tw = target_size
        result = np.zeros((th, tw, *image.shape[2:]), dtype=image.dtype)

        for y in range(th):
            src_y = min(int(y * h / th), h - 1)
            for x in range(tw):
                src_x = min(int(x * w / tw), w - 1)
                result[y, x] = image[src_y, src_x]

        return result

    def save_visualization(
        self,
        image: np.ndarray,
        output_path: str,
    ) -> None:
        """Save visualization image to file.

        Args:
            image: Image to save (H, W, 3) as uint8.
            output_path: Output file path (.png, .jpg).
        """
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(image).save(output_path)
            logger.debug(f"Visualization saved: {output_path}")
        except ImportError:
            # Fallback: save as numpy array
            np_path = output_path.rsplit(".", 1)[0] + ".npz"
            np.savez(np_path, image=image)
            logger.debug(f"Visualization saved (numpy): {np_path}")

    def create_evaluation_summary(
        self,
        metrics: Any,
        sample_images: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> np.ndarray:
        """Create a comprehensive evaluation summary image.

        Combines key metrics, per-class IoU chart, and sample predictions
        into a single summary visualization.

        Args:
            metrics: SegmentationMetrics object.
            sample_images: Optional list of sample dicts with 'image', 'pred', 'gt'.

        Returns:
            Summary image as uint8 array.
        """
        # Create per-class IoU chart
        if hasattr(metrics, "per_class") and metrics.per_class:
            iou_values = np.array([m.iou for m in metrics.per_class])
            iou_chart = self.create_per_class_iou_bar_chart(iou_values)
        else:
            iou_chart = np.full((400, 600, 3), 240, dtype=np.uint8)

        # Add sample predictions if available
        if sample_images:
            sample_panels = []
            for sample in sample_images[:4]:
                comparison = self.create_comparison(
                    image=sample.get("image", np.zeros((256, 512, 3), dtype=np.uint8)),
                    pred_mask=sample.get("pred", np.zeros((256, 512), dtype=np.uint8)),
                    gt_mask=sample.get("gt", np.zeros((256, 512), dtype=np.uint8)),
                )
                sample_panels.append(comparison)

            if sample_panels:
                # Stack vertically
                samples_vis = np.concatenate(sample_panels, axis=0)
                # Combine with IoU chart
                total_h = iou_chart.shape[0] + samples_vis.shape[0]
                total_w = max(iou_chart.shape[1], samples_vis.shape[1])
                summary = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
                summary[:iou_chart.shape[0], :iou_chart.shape[1]] = iou_chart
                summary[iou_chart.shape[0]:iou_chart.shape[0] + samples_vis.shape[0],
                        :samples_vis.shape[1]] = samples_vis
                return summary

        return iou_chart
