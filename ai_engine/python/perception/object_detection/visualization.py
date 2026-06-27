"""
Visualization utilities for the Autonomous Vehicle Control System.

Provides functions for plotting trajectories, sensor data overlays,
debug annotations, and simple animation helpers using matplotlib.

Usage:
    from utils.visualization import plot_trajectory, plot_sensor_overlay, animate_trajectory

    plot_trajectory(waypoints, title="Planned Path")
    plot_sensor_overlay(lidar_points, camera_image_rgb, detections)
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.animation as mplanim
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
except ImportError:
    raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Colour palette – consistent, colourblind-friendly
# ---------------------------------------------------------------------------

class Palette:
    """Consistent colour palette for AV visualizations."""
    EGO_VEHICLE = "#2196F3"
    OTHER_VEHICLE = "#FF5722"
    PEDESTRIAN = "#FFC107"
    CYCLIST = "#4CAF50"
    TRAFFIC_LIGHT_RED = "#F44336"
    TRAFFIC_LIGHT_GREEN = "#4CAF50"
    TRAFFIC_LIGHT_YELLOW = "#FFEB3B"
    LANE_BOUNDARY = "#FFFFFF"
    PATH_PLANNED = "#00BCD4"
    PATH_ACTUAL = "#9C27B0"
    LIDAR_POINT = "#00E676"
    OBSTACLE = "#FF9800"
    SAFETY_ZONE = "#F4433644"
    ROAD = "#616161"
    BACKGROUND = "#212121"

    @classmethod
    def detection_colour(cls, label: str) -> str:
        mapping = {
            "car": cls.OTHER_VEHICLE,
            "truck": cls.OTHER_VEHICLE,
            "bus": cls.OTHER_VEHICLE,
            "person": cls.PEDESTRIAN,
            "pedestrian": cls.PEDESTRIAN,
            "bicycle": cls.CYCLIST,
            "motorcycle": cls.CYCLIST,
        }
        return mapping.get(label.lower(), cls.OBSTACLE)


# ---------------------------------------------------------------------------
# Trajectory plotting
# ---------------------------------------------------------------------------

def plot_trajectory(
    waypoints: np.ndarray,
    title: str = "Trajectory",
    show_heading: bool = True,
    show_velocity: bool = False,
    velocities: Optional[np.ndarray] = None,
    color: str = Palette.PATH_PLANNED,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot a 2D trajectory from an array of waypoints.

    Args:
        waypoints: (N, 2) or (N, 3) array [x, y, (theta)].
        title: Plot title.
        show_heading: Draw heading arrows at intervals.
        show_velocity: Colour the path by velocity magnitude.
        velocities: (N,) velocity values – required if *show_velocity* is True.
        color: Line colour (ignored when *show_velocity* is True).
        ax: Existing axes to draw on.
        figsize: Figure size for new figure.
        save_path: If provided, save figure to this path.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xs, ys = waypoints[:, 0], waypoints[:, 1]

    if show_velocity and velocities is not None:
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(velocities.min(), velocities.max())
        lc = LineCollection(segments, cmap="plasma", norm=norm)
        lc.set_array(velocities[:-1])
        lc.set_linewidth(2.5)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("Velocity (m/s)")
    else:
        ax.plot(xs, ys, "-", color=color, linewidth=2, label="Path")
        ax.plot(xs[0], ys[0], "o", color="green", markersize=10, label="Start", zorder=5)
        ax.plot(xs[-1], ys[-1], "s", color="red", markersize=10, label="End", zorder=5)

    if show_heading and waypoints.shape[1] >= 3:
        step = max(1, len(waypoints) // 12)
        for i in range(0, len(waypoints), step):
            theta = waypoints[i, 2]
            dx, dy = math.cos(theta) * 1.5, math.sin(theta) * 1.5
            ax.annotate(
                "", xy=(xs[i] + dx, ys[i] + dy), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Sensor data overlay
# ---------------------------------------------------------------------------

def plot_lidar_points(
    points: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    point_size: float = 0.5,
    max_range: float = 80.0,
    title: str = "LiDAR Point Cloud (Top-Down)",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Plot a top-down view of LiDAR point cloud data.

    Args:
        points: (N, 2) or (N, 3) array of [x, y, (z)] in vehicle frame.
        intensities: (N,) reflectance values for colour mapping.
        ax: Existing axes.
        point_size: Scatter marker size.
        max_range: Axis limit in metres.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        The Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=Palette.BACKGROUND)
        ax.set_facecolor(Palette.BACKGROUND)
    else:
        fig = ax.figure

    xs, ys = points[:, 0], points[:, 1]
    c = intensities if intensities is not None else Palette.LIDAR_POINT
    ax.scatter(xs, ys, s=point_size, c=c, cmap="viridis" if intensities is not None else None, alpha=0.7)

    # Draw ego vehicle rectangle
    ego = mpatches.FancyBboxPatch(
        (-1.0, -0.9), 2.0, 1.8,
        boxstyle="round,pad=0.1",
        facecolor=Palette.EGO_VEHICLE, edgecolor="white", linewidth=1.5, zorder=10,
    )
    ax.add_patch(ego)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.15, color="white")

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_sensor_overlay(
    lidar_points: Optional[np.ndarray] = None,
    detections: Optional[List[Dict[str, Any]]] = None,
    waypoints: Optional[np.ndarray] = None,
    title: str = "Sensor Fusion Overlay",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Draw a combined sensor overlay with detections and path.

    Args:
        lidar_points: (N, 2) point cloud.
        detections: List of dicts with keys ``label``, ``x``, ``y``,
            ``length``, ``width``, ``heading``.
        waypoints: (M, 2) planned path.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        The Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=Palette.BACKGROUND)
    ax.set_facecolor(Palette.BACKGROUND)

    if lidar_points is not None:
        ax.scatter(
            lidar_points[:, 0], lidar_points[:, 1],
            s=0.3, c=Palette.LIDAR_POINT, alpha=0.5,
        )

    if detections is not None:
        for det in detections:
            colour = Palette.detection_colour(det.get("label", "unknown"))
            cx, cy = det["x"], det["y"]
            length = det.get("length", 4.5)
            width = det.get("width", 2.0)
            heading = det.get("heading", 0.0)

            # Rotated bounding box
            corners = _box_corners(cx, cy, length, width, heading)
            poly = plt.Polygon(corners, fill=False, edgecolor=colour, linewidth=2, zorder=8)
            ax.add_patch(poly)
            ax.text(cx, cy + width, det.get("label", ""), color=colour,
                    fontsize=8, ha="center", va="bottom", zorder=9)

    if waypoints is not None:
        ax.plot(waypoints[:, 0], waypoints[:, 1], "-", color=Palette.PATH_PLANNED,
                linewidth=2, label="Planned Path", zorder=7)

    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.15, color="white")
    ax.legend(loc="upper right", facecolor="#333333", edgecolor="white", labelcolor="white")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def _box_corners(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    """Compute the 4 corners of a rotated bounding box."""
    hl, hw = length / 2, width / 2
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (rot @ local.T).T + np.array([cx, cy])


# ---------------------------------------------------------------------------
# Debug overlay for camera images
# ---------------------------------------------------------------------------

def draw_detections_on_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box detections on a BGR image using OpenCV.

    Args:
        image: (H, W, 3) BGR uint8 image.
        detections: List of dicts with ``label``, ``bbox`` (x1, y1, x2, y2),
            ``confidence``.
        color_map: Mapping from label to (B, G, R) colour tuple.
        line_thickness: Box border thickness.

    Returns:
        Annotated image copy.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image annotation")

    img = image.copy()
    default_map = {
        "car": (0, 0, 255),
        "truck": (0, 100, 255),
        "person": (0, 255, 255),
        "bicycle": (0, 255, 0),
    }
    cmap = color_map or default_map

    for det in detections:
        label = det.get("label", "unknown")
        colour = cmap.get(label, (255, 165, 0))
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, line_thickness)
        conf = det.get("confidence", 0.0)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_trajectory(
    trajectory_frames: List[np.ndarray],
    ego_poses: Optional[List[np.ndarray]] = None,
    detections_per_frame: Optional[List[List[Dict[str, Any]]]] = None,
    interval: int = 50,
    max_range: float = 80.0,
    title: str = "Trajectory Animation",
    save_path: Optional[Union[str, Path]] = None,
) -> mplanim.FuncAnimation:
    """Create a trajectory animation using matplotlib.

    Args:
        trajectory_frames: List of (N_i, 2) arrays – one per time step.
        ego_poses: Optional list of [x, y, theta] for the ego vehicle per frame.
        detections_per_frame: Optional per-frame detection lists (same format as
            :func:`plot_sensor_overlay`).
        interval: Milliseconds between frames.
        max_range: Axis range.
        title: Animation title.
        save_path: If provided, save as GIF/MP4.

    Returns:
        A :class:`~matplotlib.animation.FuncAnimation` object.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=Palette.BACKGROUND)
    ax.set_facecolor(Palette.BACKGROUND)

    def _update(frame_idx: int):
        ax.clear()
        ax.set_facecolor(Palette.BACKGROUND)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect("equal")
        ax.set_title(f"{title}  t={frame_idx}", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.15, color="white")

        pts = trajectory_frames[frame_idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=Palette.LIDAR_POINT, alpha=0.5)

        if ego_poses is not None:
            pose = ego_poses[frame_idx]
            ego = mpatches.FancyBboxPatch(
                (pose[0] - 1.0, pose[1] - 0.9), 2.0, 1.8,
                boxstyle="round,pad=0.1",
                facecolor=Palette.EGO_VEHICLE, edgecolor="white", linewidth=1.5, zorder=10,
            )
            ax.add_patch(ego)

        if detections_per_frame is not None:
            dets = detections_per_frame[frame_idx]
            for det in dets:
                colour = Palette.detection_colour(det.get("label", "unknown"))
                cx, cy = det["x"], det["y"]
                length = det.get("length", 4.5)
                width = det.get("width", 2.0)
                heading = det.get("heading", 0.0)
                corners = _box_corners(cx, cy, length, width, heading)
                poly = plt.Polygon(corners, fill=False, edgecolor=colour, linewidth=2, zorder=8)
                ax.add_patch(poly)

    anim = mplanim.FuncAnimation(fig, _update, frames=len(trajectory_frames), interval=interval, blit=False)

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == ".gif":
            anim.save(str(save_path), writer="pillow", fps=1000 // interval)
        else:
            anim.save(str(save_path), writer="ffmpeg", fps=1000 // interval)

    return anim


# ---------------------------------------------------------------------------
# Utility: fig to numpy
# ---------------------------------------------------------------------------

def figure_to_array(fig: Figure) -> np.ndarray:
    """Render a matplotlib Figure to an (H, W, 3) uint8 RGB numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    if cv2 is not None:
        arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)
