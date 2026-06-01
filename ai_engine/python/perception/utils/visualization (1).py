"""
Visualization Module
====================

Drawing utilities for rendering detection results on images and video frames.

Features:
    - Color-coded bounding boxes per class
    - Confidence score labels
    - Class name labels
    - Track ID overlays
    - Distance / velocity annotations
    - Keypoint and skeleton drawing (for pose estimation)
    - 3D bounding box projection overlay
    - ROI region overlay
    - Save annotated images to disk

Usage::

    from object_detection.visualization import draw_detections, save_annotated

    annotated = draw_detections(image, result)
    save_annotated(annotated, "output.jpg")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .detection_utils import BBox, Detection, DetectionResult
from .pedestrian_detector import Pose, SKELETON_CONNECTIONS, COCO_KEYPOINT_NAMES

logger = logging.getLogger(__name__)


# ===================================================================
# Default Colour Palette
# ===================================================================

# 20 distinct BGR colours (COCO-style)
DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),      # 0  green
    (0, 0, 255),      # 1  red
    (255, 0, 0),      # 2  blue
    (0, 255, 255),    # 3  yellow
    (255, 255, 0),    # 4  cyan
    (255, 0, 255),    # 5  magenta
    (0, 165, 255),    # 6  orange
    (128, 0, 128),    # 7  purple
    (0, 128, 255),    # 8  dark orange
    (255, 128, 0),    # 9  light blue
    (128, 255, 0),    # 10 lime
    (0, 255, 128),    # 11 spring green
    (255, 128, 128),  # 12 light red
    (128, 128, 255),  # 13 light blue
    (255, 255, 128),  # 14 light yellow
    (128, 255, 255),  # 15 light cyan
    (200, 200, 200),  # 16 grey
    (100, 100, 100),  # 17 dark grey
    (64, 200, 64),    # 18 olive
    (200, 64, 64),    # 19 maroon
]

# Autonomous-vehicle-specific colours
AV_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 255, 0),
    "truck": (0, 165, 255),
    "bus": (0, 200, 255),
    "motorcycle": (255, 255, 0),
    "bicycle": (255, 200, 0),
    "pedestrian": (0, 0, 255),
    "cyclist": (0, 128, 255),
    "traffic_sign": (0, 255, 255),
    "traffic_light": (255, 255, 0),
    "speed_limit": (0, 0, 255),
    "stop": (0, 0, 200),
    "yield": (0, 255, 255),
    "warning": (0, 200, 255),
    "mandatory": (255, 100, 0),
}


def get_class_color(
    class_id: int,
    class_name: str = "",
    palette: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[int, int, int]:
    """Get a BGR colour for a class, using AV-specific overrides first."""
    if class_name in AV_CLASS_COLORS:
        return AV_CLASS_COLORS[class_name]
    pal = palette or DEFAULT_PALETTE
    return pal[class_id % len(pal)]


# ===================================================================
# Drawing Functions
# ===================================================================


def draw_bbox(
    image: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    style: str = "rect",
) -> np.ndarray:
    """Draw a bounding box on the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (modified in place).
    bbox : BBox
    color : tuple
        BGR colour.
    thickness : int
    style : str
        ``"rect"`` – standard rectangle.
        ``"corner"`` – corner-only lines (looks cleaner).
        ``"dashed"`` – dashed rectangle.
    """
    x1, y1, x2, y2 = bbox.to_int().to_numpy().astype(int)

    if style == "rect":
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    elif style == "corner":
        corner_len = max(10, int(min(x2 - x1, y2 - y1) * 0.2))
        # Top-left
        cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # Top-right
        cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # Bottom-left
        cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # Bottom-right
        cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    elif style == "dashed":
        dash_len = 10
        for side in [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1)),  # left
        ]:
            pt1, pt2 = side
            _draw_dashed_line(image, pt1, pt2, color, thickness, dash_len)

    return image


def _draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
) -> None:
    """Draw a dashed line between two points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = max(int((dx ** 2 + dy ** 2) ** 0.5), 1)
    dashes = dist // (dash_len * 2)

    for i in range(dashes + 1):
        s = i / max(dashes, 1)
        e = min((i + 0.5) / max(dashes, 1), 1.0)
        start = (int(pt1[0] + dx * s), int(pt1[1] + dy * s))
        end = (int(pt1[0] + dx * e), int(pt1[1] + dy * e))
        cv2.line(image, start, end, color, thickness)


def draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.5,
    thickness: int = 1,
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 4,
) -> np.ndarray:
    """Draw a text label with optional background rectangle.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    text : str
    position : tuple
        ``(x, y)`` – top-left corner of the label.
    color : tuple
        Text colour (BGR).
    font_scale : float
    thickness : int
    bg_color : tuple, optional
        Background fill colour.  If None, a semi-transparent dark box is used.
    padding : int
        Padding around the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Ensure label stays within image bounds
    h_img, w_img = image.shape[:2]
    if y - th - padding < 0:
        y = th + padding
    if x + tw + padding > w_img:
        x = w_img - tw - padding

    # Background
    if bg_color is not None:
        cv2.rectangle(
            image,
            (x - padding, y - th - padding),
            (x + tw + padding, y + baseline + padding),
            bg_color, -1,
        )
    else:
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - th - padding),
            (x + tw + padding, y + baseline + padding),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def draw_confidence_bar(
    image: np.ndarray,
    bbox: BBox,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    bar_width: int = 4,
) -> np.ndarray:
    """Draw a vertical confidence bar on the left side of a bbox."""
    x1, y1, x2, y2 = bbox.to_int().to_numpy().astype(int)
    bar_x = max(0, x1 - bar_width - 2)
    bar_top = y1
    bar_bottom = y2
    bar_height = bar_bottom - bar_top

    # Background (dark)
    cv2.rectangle(image, (bar_x, bar_top), (bar_x + bar_width, bar_bottom), (40, 40, 40), -1)

    # Filled portion
    fill_height = int(bar_height * confidence)
    cv2.rectangle(
        image,
        (bar_x, bar_bottom - fill_height),
        (bar_x + bar_width, bar_bottom),
        color, -1,
    )
    return image


# ===================================================================
# High-Level Detection Drawing
# ===================================================================


def draw_detection(
    image: np.ndarray,
    detection: Detection,
    show_confidence: bool = True,
    show_class: bool = True,
    show_track_id: bool = True,
    show_distance: bool = True,
    show_velocity: bool = True,
    box_style: str = "rect",
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw a single Detection on the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    detection : Detection
    show_confidence, show_class, show_track_id, show_distance, show_velocity : bool
        Toggle individual annotations.
    box_style : str
        ``"rect"``, ``"corner"``, or ``"dashed"``.
    line_thickness : int
    font_scale : float
    """
    color = get_class_color(detection.class_id, detection.class_name)
    bbox = detection.bbox

    # Bounding box
    draw_bbox(image, bbox, color=color, thickness=line_thickness, style=box_style)

    # Label text
    parts: List[str] = []
    if show_class:
        parts.append(detection.class_name)
    if show_confidence:
        parts.append(f"{detection.confidence:.2f}")
    if show_track_id and detection.track_id is not None:
        parts.append(f"ID:{detection.track_id}")
    if show_distance and detection.distance is not None:
        parts.append(f"{detection.distance:.1f}m")
    if show_velocity and detection.velocity is not None:
        vx, vy = detection.velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5 * 3.6  # km/h
        parts.append(f"{speed:.0f}km/h")

    label_text = " ".join(parts)
    x1, y1 = int(bbox.x1), int(bbox.y1)
    draw_label(
        image, label_text,
        position=(x1, y1 - 4),
        color=(255, 255, 255),
        font_scale=font_scale,
        thickness=1,
    )

    # Confidence bar
    if show_confidence:
        draw_confidence_bar(image, bbox, detection.confidence, color=color)

    return image


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    show_confidence: bool = True,
    show_class: bool = True,
    show_track_id: bool = True,
    show_distance: bool = True,
    show_velocity: bool = True,
    box_style: str = "rect",
    line_thickness: int = 2,
    font_scale: float = 0.5,
    max_detections: int = 100,
) -> np.ndarray:
    """Draw all detections from a DetectionResult on the image.

    Also draws a summary info line at the top (FPS, count, etc.).

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    result : DetectionResult
    max_detections : int
        Cap on number of detections drawn (performance guard).

    Returns
    -------
    np.ndarray
        Annotated image (copy).
    """
    annotated = image.copy()

    dets = result.detections[:max_detections]
    for det in dets:
        draw_detection(
            annotated, det,
            show_confidence=show_confidence,
            show_class=show_class,
            show_track_id=show_track_id,
            show_distance=show_distance,
            show_velocity=show_velocity,
            box_style=box_style,
            line_thickness=line_thickness,
            font_scale=font_scale,
        )

    # Info overlay
    info_parts: List[str] = []
    info_parts.append(f"Detections: {len(result.detections)}")
    if result.fps > 0:
        info_parts.append(f"FPS: {result.fps:.1f}")
    if result.inference_time_ms > 0:
        info_parts.append(f"Latency: {result.total_time_ms:.1f}ms")
    if result.model_name:
        info_parts.append(f"Model: {result.model_name}")

    if info_parts:
        info_text = " | ".join(info_parts)
        draw_label(
            annotated, info_text,
            position=(10, 24),
            color=(0, 255, 0),
            font_scale=0.6,
            thickness=1,
            bg_color=(0, 0, 0),
        )

    return annotated


# ===================================================================
# Pose / Skeleton Drawing
# ===================================================================


def draw_pose(
    image: np.ndarray,
    pose: Pose,
    keypoint_radius: int = 4,
    keypoint_color: Tuple[int, int, int] = (0, 255, 255),
    skeleton_color: Tuple[int, int, int] = (200, 200, 200),
    skeleton_thickness: int = 2,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Draw keypoints and skeleton on the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    pose : Pose
    keypoint_radius : int
    keypoint_color : tuple
        BGR colour for keypoint circles.
    skeleton_color : tuple
        BGR colour for skeleton lines.
    skeleton_thickness : int
    min_confidence : float
        Minimum keypoint confidence to draw.
    """
    kpts = pose.keypoints

    # Draw skeleton
    for i, j in SKELETON_CONNECTIONS:
        kp_a = kpts[i] if i < len(kpts) else None
        kp_b = kpts[j] if j < len(kpts) else None
        if (
            kp_a and kp_b
            and kp_a.is_valid and kp_b.is_valid
            and kp_a.confidence >= min_confidence
            and kp_b.confidence >= min_confidence
        ):
            cv2.line(
                image,
                (int(kp_a.x), int(kp_a.y)),
                (int(kp_b.x), int(kp_b.y)),
                skeleton_color, skeleton_thickness, cv2.LINE_AA,
            )

    # Draw keypoints
    for kp in kpts:
        if kp.is_valid and kp.confidence >= min_confidence:
            cv2.circle(
                image,
                (int(kp.x), int(kp.y)),
                keypoint_radius, keypoint_color, -1, cv2.LINE_AA,
            )

    return image


# ===================================================================
# ROI Overlay
# ===================================================================


def draw_roi(
    image: np.ndarray,
    top_ratio: float = 0.0,
    bottom_ratio: float = 0.7,
    side_margin_ratio: float = 0.05,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the Region of Interest boundary on the image."""
    h, w = image.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    x1 = int(w * side_margin_ratio)
    x2 = w - int(w * side_margin_ratio)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    # Dim area outside ROI
    overlay = image.copy()
    # Top
    overlay[:y1, :] = (overlay[:y1, :] * 0.5).astype(np.uint8)
    # Bottom
    overlay[y2:, :] = (overlay[y2:, :] * 0.5).astype(np.uint8)
    # Left
    overlay[y1:y2, :x1] = (overlay[y1:y2, :x1] * 0.5).astype(np.uint8)
    # Right
    overlay[y1:y2, x2:] = (overlay[y1:y2, x2:] * 0.5).astype(np.uint8)

    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image


# ===================================================================
# 3D Box Projection
# ===================================================================


def draw_3d_bbox(
    image: np.ndarray,
    bbox_3d_dict: Dict[str, float],
    focal_length: float = 800.0,
    cx: float = 320.0,
    cy: float = 240.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw an approximate 3-D bounding box projected onto the image.

    Parameters
    ----------
    image : np.ndarray
    bbox_3d_dict : dict
        Output of ``BBox3D.to_dict()`` – must contain x, y, z, l, w, h, yaw.
    """
    from .vehicle_detector import BBox3D
    bb3d = BBox3D(
        x=bbox_3d_dict["x"], y=bbox_3d_dict["y"], z=bbox_3d_dict["z"],
        length=bbox_3d_dict["length"], width=bbox_3d_dict["width"],
        height=bbox_3d_dict["height"], yaw=bbox_3d_dict.get("yaw", 0.0),
    )

    # 8 corners of the 3D box in camera frame
    l, w, h = bb3d.length / 2, bb3d.width / 2, bb3d.height / 2
    corners_3d = np.array([
        [-l, -h, w], [l, -h, w], [l, h, w], [-l, h, w],     # front face
        [-l, -h, -w], [l, -h, -w], [l, h, -w], [-l, h, -w], # rear face
    ])

    # Rotate by yaw
    cos_y, sin_y = np.cos(bb3d.yaw), np.sin(bb3d.yaw)
    rot = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ])
    corners_3d = (rot @ corners_3d.T).T + np.array([bb3d.x, bb3d.y, bb3d.z])

    # Project to 2D
    corners_2d = []
    for c in corners_3d:
        if c[2] <= 0.1:
            corners_2d.append(None)
            continue
        u = focal_length * c[0] / c[2] + cx
        v = focal_length * c[1] / c[2] + cy
        corners_2d.append((int(u), int(v)))

    # Draw edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front
        (4, 5), (5, 6), (6, 7), (7, 4),  # back
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting
    ]
    for i, j in edges:
        if corners_2d[i] is not None and corners_2d[j] is not None:
            cv2.line(image, corners_2d[i], corners_2d[j], color, thickness, cv2.LINE_AA)

    return image


# ===================================================================
# Save Helpers
# ===================================================================


def save_annotated(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """Save an annotated image to disk.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    path : str or Path
        Output file path (extension determines format).
    quality : int
        JPEG quality (1–100), ignored for PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(str(path), image)

    logger.info("Saved annotated image to %s", path)


def create_comparison_grid(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    grid_cols: int = 2,
    padding: int = 5,
    label_height: int = 30,
) -> np.ndarray:
    """Arrange multiple images into a comparison grid.

    Parameters
    ----------
    images : list of np.ndarray
        BGR images (should have similar dimensions).
    labels : list of str, optional
        Title for each image.
    grid_cols : int
        Number of columns.
    padding : int
        Pixel padding between images.
    label_height : int
        Height reserved for text label above each image.

    Returns
    -------
    np.ndarray
        Grid image.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Determine cell size from the first image
    cell_h, cell_w = images[0].shape[:2]
    n = len(images)
    rows = (n + grid_cols - 1) // grid_cols

    canvas_h = rows * (cell_h + label_height + padding) + padding
    canvas_w = grid_cols * (cell_w + padding) + padding
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for idx, img in enumerate(images):
        r, c = divmod(idx, grid_cols)
        y_off = r * (cell_h + label_height + padding) + padding + label_height
        x_off = c * (cell_w + padding) + padding

        # Resize if needed
        if img.shape[:2] != (cell_h, cell_w):
            img = cv2.resize(img, (cell_w, cell_h))

        canvas[y_off:y_off + cell_h, x_off:x_off + cell_w] = img

        # Label
        if labels and idx < len(labels):
            cv2.putText(
                canvas, labels[idx],
                (x_off + 5, y_off - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return canvas
