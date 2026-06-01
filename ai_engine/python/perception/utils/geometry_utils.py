"""
Geometry utilities for the Autonomous Vehicle Control System.

Provides line intersection, polygon operations, convex hull, rotation
matrices, coordinate frame transforms, bounding-box helpers, and
distance metrics used by perception and planning modules.

Usage:
    from utils.geometry_utils import line_intersection, polygon_area, convex_hull, rotation_matrix_2d

    pt = line_intersection(line1, line2)
    area = polygon_area(vertices)
    hull = convex_hull(points)
    R = rotation_matrix_2d(math.pi / 4)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Point / vector helpers
# ---------------------------------------------------------------------------

def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.asarray(p1)[:2] - np.asarray(p2)[:2]))


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.asarray(p1)[:3] - np.asarray(p2)[:3]))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint of two points."""
    return (np.asarray(p1) + np.asarray(p2)) / 2.0


def point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    """Perpendicular distance from *point* to the infinite line through
    *line_start* and *line_end*.
    """
    p = np.asarray(point, dtype=float)[:2]
    a = np.asarray(line_start, dtype=float)[:2]
    b = np.asarray(line_end, dtype=float)[:2]
    ab = b - a
    ap = p - a
    cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
    denom = np.linalg.norm(ab)
    if denom < 1e-12:
        return float(np.linalg.norm(ap))
    return float(cross / denom)


def point_to_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float:
    """Shortest distance from *point* to the **line segment**
    seg_start→seg_end.
    """
    p = np.asarray(point, dtype=float)[:2]
    a = np.asarray(seg_start, dtype=float)[:2]
    b = np.asarray(seg_end, dtype=float)[:2]
    ab = b - a
    ap = p - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 < 1e-12:
        return float(np.linalg.norm(ap))
    t = np.dot(ap, ab) / ab_len2
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def closest_point_on_segment(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> np.ndarray:
    """Return the closest point on segment to *point*."""
    p = np.asarray(point, dtype=float)[:2]
    a = np.asarray(seg_start, dtype=float)[:2]
    b = np.asarray(seg_end, dtype=float)[:2]
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 < 1e-12:
        return a.copy()
    t = np.dot(p - a, ab) / ab_len2
    t = max(0.0, min(1.0, t))
    return a + t * ab


# ---------------------------------------------------------------------------
# Line intersection
# ---------------------------------------------------------------------------

def line_intersection(
    line1: Tuple[np.ndarray, np.ndarray],
    line2: Tuple[np.ndarray, np.ndarray],
) -> Optional[np.ndarray]:
    """Find the intersection point of two infinite 2D lines.

    Each line is defined by two points: ``(p1, p2)``.

    Returns:
        Intersection point as (2,) array, or ``None`` if lines are parallel.
    """
    p1, p2 = np.asarray(line1[0], dtype=float)[:2], np.asarray(line1[1], dtype=float)[:2]
    p3, p4 = np.asarray(line2[0], dtype=float)[:2], np.asarray(line2[1], dtype=float)[:2]

    d1 = p2 - p1
    d2 = p4 - p3

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None  # Parallel or collinear

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def segment_intersection(
    seg1: Tuple[np.ndarray, np.ndarray],
    seg2: Tuple[np.ndarray, np.ndarray],
) -> Optional[np.ndarray]:
    """Find the intersection of two **finite** 2D line segments.

    Returns:
        Intersection point, or ``None`` if segments don't cross.
    """
    p1, p2 = np.asarray(seg1[0], dtype=float)[:2], np.asarray(seg1[1], dtype=float)[:2]
    p3, p4 = np.asarray(seg2[0], dtype=float)[:2], np.asarray(seg2[1], dtype=float)[:2]

    d1 = p2 - p1
    d2 = p4 - p3

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None

    dp = p3 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    u = (dp[0] * d1[1] - dp[1] * d1[0]) / cross

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t * d1
    return None


# ---------------------------------------------------------------------------
# Polygon operations
# ---------------------------------------------------------------------------

def polygon_area(vertices: np.ndarray) -> float:
    """Compute the area of a simple (non-self-intersecting) polygon using
    the shoelace formula.

    Args:
        vertices: (N, 2) array of polygon vertices in order (CW or CCW).

    Returns:
        Unsigned area.
    """
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    if n < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Compute the centroid of a simple polygon.

    Args:
        vertices: (N, 2) ordered vertices.

    Returns:
        (2,) centroid [cx, cy].
    """
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    if n == 0:
        return np.zeros(2)
    if n < 3:
        return v.mean(axis=0)

    x, y = v[:, 0], v[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-12:
        return v.mean(axis=0)

    cx = ((x + xn) * cross).sum() / (6.0 * a)
    cy = ((y + yn) * cross).sum() / (6.0 * a)
    return np.array([cx, cy])


def point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Ray-casting algorithm to test if a point lies inside a polygon.

    Args:
        point: (2,) test point.
        vertices: (N, 2) polygon vertices.

    Returns:
        True if *point* is inside (or on the boundary of) the polygon.
    """
    px, py = float(point[0]), float(point[1])
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = v[i]
        xj, yj = v[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Estimate the intersection area of two convex polygons using the
    Sutherland-Hodgman clipping algorithm.

    For non-convex polygons the result may be inaccurate.
    """
    clipped = _sutherland_hodgman_clip(poly1, poly2)
    if len(clipped) < 3:
        return 0.0
    return polygon_area(np.array(clipped))


def _sutherland_hodgman_clip(
    subject: np.ndarray,
    clip: np.ndarray,
) -> List[np.ndarray]:
    """Sutherland-Hodgman polygon clipping – clips *subject* by *clip*.
    Both must be convex, ordered vertex lists.
    """
    def _inside(p: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
               (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0

    def _intersection(p1: np.ndarray, p2: np.ndarray,
                      edge_start: np.ndarray, edge_end: np.ndarray) -> np.ndarray:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = edge_start
        x4, y4 = edge_end
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1  # Fallback
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    output = [np.asarray(v, dtype=float) for v in subject]
    clip_verts = [np.asarray(v, dtype=float) for v in clip]
    n_clip = len(clip_verts)

    for i in range(n_clip):
        if len(output) == 0:
            return []
        edge_start = clip_verts[i]
        edge_end = clip_verts[(i + 1) % n_clip]
        input_list = output
        output = []
        for j in range(len(input_list)):
            current = input_list[j]
            prev = input_list[j - 1]
            if _inside(current, edge_start, edge_end):
                if not _inside(prev, edge_start, edge_end):
                    output.append(_intersection(prev, current, edge_start, edge_end))
                output.append(current)
            elif _inside(prev, edge_start, edge_end):
                output.append(_intersection(prev, current, edge_start, edge_end))

    return output


def iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
    """Intersection-over-Union of two axis-aligned 2D bounding boxes.

    Args:
        box1, box2: (4,) arrays as [x1, y1, x2, y2] (top-left, bottom-right).

    Returns:
        IoU value in [0, 1].
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area < 1e-12:
        return 0.0
    return float(inter_area / union_area)


def rotated_iou_2d(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """IoU of two rotated (oriented) bounding boxes.

    Each box dict has keys: ``cx``, ``cy``, ``length``, ``width``, ``heading``.
    Uses polygon clipping internally.
    """
    poly1 = _oriented_box_corners(box1["cx"], box1["cy"], box1["length"], box1["width"], box1["heading"])
    poly2 = _oriented_box_corners(box2["cx"], box2["cy"], box2["length"], box2["width"], box2["heading"])
    inter = polygon_intersection_area(poly1, poly2)
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    union = area1 + area2 - inter
    if union < 1e-12:
        return 0.0
    return inter / union


def _oriented_box_corners(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    """Compute the 4 corners of a rotated bounding box."""
    hl, hw = length / 2, width / 2
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (R @ local.T).T + np.array([cx, cy])


# ---------------------------------------------------------------------------
# Convex hull – Andrew's monotone chain
# ---------------------------------------------------------------------------

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2D points using Andrew's
    monotone chain algorithm.

    Args:
        points: (N, 2) input points.

    Returns:
        (M, 2) hull vertices in CCW order, without duplicate first point.
    """
    pts = np.asarray(points, dtype=float)
    # Sort by x, then y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    n = len(pts)

    if n < 3:
        return pts

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def is_convex(vertices: np.ndarray) -> bool:
    """Check whether a polygon defined by ordered vertices is convex."""
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    if n < 3:
        return False
    sign = None
    for i in range(n):
        o, a, b = v[i], v[(i + 1) % n], v[(i + 2) % n]
        cross = (a[0] - o[0]) * (b[1] - a[1]) - (a[1] - o[1]) * (b[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        current_sign = cross > 0
        if sign is None:
            sign = current_sign
        elif current_sign != sign:
            return False
    return True


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

def rotation_matrix_2d(angle: float) -> np.ndarray:
    """2×2 rotation matrix for angle *angle* (radians, CCW positive)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotation_matrix_3d(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    order: str = "ZYX",
) -> np.ndarray:
    """3×3 rotation matrix from Euler angles.

    Default order is ZYX (yaw-pitch-roll), which is the aerospace convention.

    Args:
        roll: Rotation about X (radians).
        pitch: Rotation about Y (radians).
        yaw: Rotation about Z (radians).
        order: Application order string (e.g., ``"ZYX"``).

    Returns:
        (3, 3) rotation matrix.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    matrices = {"X": Rx, "Y": Ry, "Z": Rz}
    result = np.eye(3)
    for axis in reversed(order.upper()):
        result = result @ matrices.get(axis, np.eye(3))
    return result


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) from a 3×3 rotation matrix (ZYX convention).

    Returns:
        (roll, pitch, yaw) in radians.
    """
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)

    if abs(sy) < 1.0 - 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return roll, pitch, yaw


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3×3 rotation matrix."""
    w, x, y, z = q
    norm_q = math.sqrt(w * w + x * x + y * y + z * z)
    if norm_q < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm_q, x / norm_q, y / norm_q, z / norm_q

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def transform_point(
    point: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Apply rigid-body transform: rotation then translation.

    Args:
        point: (2,) or (3,) point.
        translation: Same dimension as point.
        rotation: 2×2 or 3×3 rotation matrix.

    Returns:
        Transformed point.
    """
    return rotation @ point + translation


def inverse_transform(
    point: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Apply inverse rigid-body transform.

    Given a point in the target frame, return the point in the source frame.
    """
    R_inv = rotation.T  # Rotation matrices are orthogonal
    return R_inv @ (point - translation)


def homogeneous_transform(
    point: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply a 4×4 homogeneous transformation matrix to a 3D point.

    Args:
        point: (3,) point.
        matrix: (4, 4) homogeneous transform.

    Returns:
        (3,) transformed point.
    """
    p_h = np.append(point, 1.0)
    result = matrix @ p_h
    return result[:3] / result[3]


def build_homogeneous_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Build a 4×4 homogeneous transformation matrix from R and t.

    Args:
        rotation: (3, 3) rotation matrix.
        translation: (3,) translation vector.

    Returns:
        (4, 4) homogeneous transform.
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T
