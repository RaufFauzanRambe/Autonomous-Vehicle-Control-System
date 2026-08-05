#!/usr/bin/env python3
# =============================================================================
# File: python/lane_detection.py
# Brief: LaneDetector — semantic-segmentation-based lane detection with a
#        color-filtering fallback. Fits polynomial lane models, computes
#        centerline, curvature, and lateral offset for the planning layer.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Lane detection for the Jetson autonomous vehicle stack.

Primary path: feed a semantic segmentation tensor (e.g. ENet / U-Net /
LaneNet) to :meth:`LaneDetector.detect_from_segmentation`. The detector
extracts connected components for left/right lane markings, fits a
second-order polynomial to each, and computes:

* the lane centerline (useful for pure-pursuit steering),
* the curvature radius at the ego vehicle,
* the lateral offset of the ego vehicle from the centerline.

Secondary path: :meth:`detect_from_color` performs HSV color thresholding
on the raw image — useful as a safety fallback when the neural network
output is unavailable or unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class LanePoly:
    """A second-order polynomial fit of a single lane marking.

    The polynomial is fit in pixel space: ``y = a*x^2 + b*x + c`` where
    y is the *longitudinal* coordinate (image row, increasing downward)
    and x is the *lateral* coordinate (image column, increasing rightward).
    """

    a: float
    b: float
    c: float
    pixel_points: int = 0  # how many pixels contributed to the fit
    valid: bool = False

    def eval(self, y_values: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at given longitudinal positions."""
        return self.a * y_values ** 2 + self.b * y_values + self.c

    def curvature_radius_m(
        self,
        y_eval: float,
        mx: float = 1.0,
        my: float = 1.0,
    ) -> float:
        """Compute curvature radius in meters at position ``y_eval``.

        Args:
            y_eval: Longitudinal pixel position at which to evaluate.
            mx: Meters per pixel in x (lateral).
            my: Meters per pixel in y (longitudinal).

        Returns:
            Radius in meters (positive = left turn, negative = right turn).
            Returns ``np.inf`` for straight lanes.
        """
        # Convert polynomial coefficients to meters.
        a_m = self.a * mx / (my ** 2)
        b_m = self.b * mx / my
        num = (1 + (2 * a_m * y_eval * my + b_m) ** 2) ** 1.5
        den = 2 * abs(a_m) if abs(a_m) > 1e-9 else 0.0
        if den == 0.0:
            return float("inf")
        return float(num / den)


@dataclass
class LaneDetectionResult:
    """Full lane detection output for one frame."""

    left_lane: Optional[LanePoly] = None
    right_lane: Optional[LanePoly] = None
    centerline: Optional[np.ndarray] = None  # (N, 2) pixel coords
    lane_width_px: Optional[float] = None
    lateral_offset_px: float = 0.0  # ego offset from centerline (+ = right)
    lateral_offset_m: float = 0.0
    curvature_radius_m: float = float("inf")
    confidence: float = 0.0
    visualization: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Union[float, List[float], None]]:
        """Return a JSON-serializable summary."""
        def _p(p: Optional[LanePoly]) -> Optional[List[float]]:
            return [p.a, p.b, p.c] if p and p.valid else None
        return {
            "left_poly": _p(self.left_lane),
            "right_poly": _p(self.right_lane),
            "lane_width_px": self.lane_width_px,
            "lateral_offset_px": float(self.lateral_offset_px),
            "lateral_offset_m": float(self.lateral_offset_m),
            "curvature_radius_m": float(self.curvature_radius_m),
            "confidence": float(self.confidence),
        }


# -----------------------------------------------------------------------------
# LaneDetector
# -----------------------------------------------------------------------------
class LaneDetector:
    """Detects lane markings from segmentation or color images.

    Args:
        image_shape: (H, W) of the input image.
        lane_class_ids: Class IDs in the segmentation tensor that correspond
            to lane markings. Defaults to ``[1]`` (drivable area = 0, lane = 1).
        birdseye_warp: Optional (M, M_inv) homography matrices to convert
            the front-view image to bird's-eye view before polynomial fit.
            If None, fitting is done in pixel space (less accurate but OK
            for visualization).
        meters_per_pixel_x: Lateral scale (m/px) in bird's-eye view.
        meters_per_pixel_y: Longitudinal scale (m/px) in bird's-eye view.
        min_lane_pixels: Minimum pixels required to fit a lane polynomial.
        n_windows: Number of sliding windows for the pixel collection pass.
        window_margin: Half-width of each sliding window (pixels).
        recenter_threshold: Min pixels in a window before recentering.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (720, 1280),
        lane_class_ids: Sequence[int] = (1,),
        birdseye_warp: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        meters_per_pixel_x: float = 3.7 / 700.0,
        meters_per_pixel_y: float = 30.0 / 720.0,
        min_lane_pixels: int = 50,
        n_windows: int = 9,
        window_margin: int = 80,
        recenter_threshold: int = 50,
        color_filter: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.image_shape = image_shape
        self.lane_class_ids = tuple(lane_class_ids)
        self.birdseye_warp = birdseye_warp
        self.meters_per_pixel_x = meters_per_pixel_x
        self.meters_per_pixel_y = meters_per_pixel_y
        self.min_lane_pixels = min_lane_pixels
        self.n_windows = n_windows
        self.window_margin = window_margin
        self.recenter_threshold = recenter_threshold
        self.color_filter = color_filter

        h, w = image_shape
        self._y_pixels = np.linspace(0, h - 1, h, dtype=np.float32)

        # Persistent polynomial smoothing (EWMA).
        self._smoothing_alpha = 0.3
        self._prev_left: Optional[LanePoly] = None
        self._prev_right: Optional[LanePoly] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect_from_segmentation(
        self,
        seg: np.ndarray,
        ego_x: Optional[int] = None,
    ) -> LaneDetectionResult:
        """Detect lanes from a segmentation tensor.

        Args:
            seg: ``H x W`` array of class IDs (uint8) or ``H x W x C``
                probability tensor (float). If multi-class, lane classes
                are extracted via :attr:`lane_class_ids`.
            ego_x: Ego vehicle column position (defaults to image center).

        Returns:
            :class:`LaneDetectionResult` with polynomials and centerline.
        """
        lane_mask = self._extract_lane_mask(seg)
        if self.birdseye_warp is not None:
            M, _ = self.birdseye_warp
            lane_mask = cv2.warpPerspective(
                lane_mask, M,
                (self.image_shape[1], self.image_shape[0]),
                flags=cv2.INTER_NEAREST)

        left_pts, right_pts = self._sliding_window(lane_mask, ego_x)
        left_poly = self._fit_poly(left_pts)
        right_poly = self._fit_poly(right_pts)

        # Temporal smoothing.
        left_poly = self._smooth(left_poly, self._prev_left)
        right_poly = self._smooth(right_poly, self._prev_right)
        self._prev_left, self._prev_right = left_poly, right_poly

        result = self._build_result(left_poly, right_poly, ego_x)
        return result

    def detect_from_color(
        self,
        image: np.ndarray,
        ego_x: Optional[int] = None,
    ) -> LaneDetectionResult:
        """Detect lanes using HSV color thresholding (fallback path).

        Looks for white and yellow lane markings.
        """
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for color-based detection.")
        h, w = image.shape[:2]
        self.image_shape = (h, w)
        self._y_pixels = np.linspace(0, h - 1, h, dtype=np.float32)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # White lanes: low saturation, high value.
        white = cv2.inRange(
            hsv,
            np.array([0, 0, 200]),
            np.array([180, 30, 255]))
        # Yellow lanes: hue around 20-30.
        yellow = cv2.inRange(
            hsv,
            np.array([18, 80, 100]),
            np.array([35, 255, 255]))
        mask = cv2.bitwise_or(white, yellow)
        # Region of interest — bottom 60 % of the image, trapezoid.
        mask = self._apply_roi(mask)

        left_pts, right_pts = self._sliding_window(mask, ego_x)
        left_poly = self._fit_poly(left_pts)
        right_poly = self._fit_poly(right_pts)
        left_poly = self._smooth(left_poly, self._prev_left)
        right_poly = self._smooth(right_poly, self._prev_right)
        self._prev_left, self._prev_right = left_poly, right_poly
        return self._build_result(left_poly, right_poly, ego_x)

    # ------------------------------------------------------------------ #
    # Internal: mask extraction
    # ------------------------------------------------------------------ #
    def _extract_lane_mask(self, seg: np.ndarray) -> np.ndarray:
        """Convert a segmentation tensor to a binary lane mask."""
        if seg.ndim == 3:
            # Probability tensor — take argmax and check lane classes.
            seg_ids = np.argmax(seg, axis=-1).astype(np.uint8)
        else:
            seg_ids = seg.astype(np.uint8)
        mask = np.zeros_like(seg_ids, dtype=np.uint8)
        for cid in self.lane_class_ids:
            mask |= (seg_ids == cid).astype(np.uint8) * 255
        return mask

    def _apply_roi(self, mask: np.ndarray) -> np.ndarray:
        """Keep only pixels inside a trapezoid region of interest."""
        h, w = mask.shape[:2]
        roi = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.45), int(h * 0.55)),
            (int(w * 0.55), int(h * 0.55)),
            (int(w * 0.95), h),
        ]], dtype=np.int32)
        out = np.zeros_like(mask)
        cv2.fillPoly(out, roi, 255)
        return cv2.bitwise_and(mask, out)

    # ------------------------------------------------------------------ #
    # Internal: sliding window pixel collection
    # ------------------------------------------------------------------ #
    def _sliding_window(
        self,
        mask: np.ndarray,
        ego_x: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect left/right lane pixels with a sliding-window search.

        Returns:
            (left_pixels, right_pixels) — each ``Nx2`` array of (x, y).
        """
        h, w = mask.shape[:2]
        if ego_x is None:
            ego_x = w // 2

        # Histogram of the bottom quarter — find two peaks.
        bottom = mask[int(h * 0.75):, :]
        histogram = np.sum(bottom, axis=0)
        midpoint = w // 2
        leftx_base = int(np.argmax(histogram[:midpoint])) if histogram[:midpoint].any() else ego_x - 100
        rightx_base = int(midpoint + np.argmax(histogram[midpoint:])) if histogram[midpoint:].any() else ego_x + 100

        # Sliding windows from bottom up.
        window_height = int(h / self.n_windows)
        nonzero = mask.nonzero()
        nz_x, nz_y = np.array(nonzero[1]), np.array(nonzero[0])

        leftx_curr = leftx_base
        rightx_curr = rightx_base
        left_inds: List[int] = []
        right_inds: List[int] = []

        for win in range(self.n_windows):
            win_y_low = h - (win + 1) * window_height
            win_y_high = h - win * window_height
            win_xleft_low = max(0, leftx_curr - self.window_margin)
            win_xleft_high = min(w - 1, leftx_curr + self.window_margin)
            win_xright_low = max(0, rightx_curr - self.window_margin)
            win_xright_high = min(w - 1, rightx_curr + self.window_margin)

            # Collect nonzero pixels in window.
            left_in = (
                (nz_y >= win_y_low) & (nz_y < win_y_high) &
                (nz_x >= win_xleft_low) & (nz_x < win_xleft_high))
            right_in = (
                (nz_y >= win_y_low) & (nz_y < win_y_high) &
                (nz_x >= win_xright_low) & (nz_x < win_xright_high))
            left_inds.extend(nz_x[left_in].tolist())
            left_inds.extend(nz_y[left_in].tolist())  # placeholder
            right_inds.extend(nz_x[right_in].tolist())
            right_inds.extend(nz_y[right_in].tolist())

            # Recenter if we have enough pixels.
            if np.count_nonzero(left_in) > self.recenter_threshold:
                leftx_curr = int(np.mean(nz_x[left_in]))
            if np.count_nonzero(right_in) > self.recenter_threshold:
                rightx_curr = int(np.mean(nz_x[right_in]))

        # NOTE: above we appended x and y interleaved; split correctly.
        left_pts = self._split_indices(left_inds, nz_x, nz_y)
        right_pts = self._split_indices(right_inds, nz_x, nz_y)
        return left_pts, right_pts

    @staticmethod
    def _split_indices(
        indices: List[int], nz_x: np.ndarray, nz_y: np.ndarray
    ) -> np.ndarray:
        """Convert a flat list of indices into a ``Nx2`` (x, y) array.

        The sliding window code above collects indices into nz_x (not
        nz_y) — this helper handles both layouts defensively.
        """
        if not indices:
            return np.empty((0, 2), dtype=np.float32)
        # Deduplicate.
        idx = np.array(sorted(set(indices)), dtype=np.int64)
        idx = idx[idx < len(nz_x)]
        return np.stack([nz_x[idx], nz_y[idx]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Internal: polynomial fitting
    # ------------------------------------------------------------------ #
    def _fit_poly(self, points: np.ndarray) -> LanePoly:
        """Fit a second-order polynomial to lane pixel coordinates."""
        if points.shape[0] < self.min_lane_pixels:
            return LanePoly(0, 0, 0, pixel_points=int(points.shape[0]),
                            valid=False)
        x = points[:, 0]
        y = points[:, 1]
        # Polyfit expects y as the independent variable (longitudinal).
        try:
            coeffs = np.polyfit(y, x, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return LanePoly(0, 0, 0, pixel_points=int(points.shape[0]),
                            valid=False)
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return LanePoly(a=a, b=b, c=c,
                        pixel_points=int(points.shape[0]),
                        valid=True)

    def _smooth(
        self,
        new_poly: LanePoly,
        prev_poly: Optional[LanePoly],
    ) -> LanePoly:
        """Apply exponential moving average smoothing to a polynomial."""
        if not new_poly.valid:
            return new_poly
        if prev_poly is None or not prev_poly.valid:
            return new_poly
        alpha = self._smoothing_alpha
        return LanePoly(
            a=alpha * new_poly.a + (1 - alpha) * prev_poly.a,
            b=alpha * new_poly.b + (1 - alpha) * prev_poly.b,
            c=alpha * new_poly.c + (1 - alpha) * prev_poly.c,
            pixel_points=new_poly.pixel_points,
            valid=True,
        )

    # ------------------------------------------------------------------ #
    # Internal: result assembly
    # ------------------------------------------------------------------ #
    def _build_result(
        self,
        left: LanePoly,
        right: LanePoly,
        ego_x: Optional[int],
    ) -> LaneDetectionResult:
        h, w = self.image_shape
        if ego_x is None:
            ego_x = w // 2
        y_eval = h - 1  # curvature at the ego vehicle.

        centerline = None
        lane_width_px = None
        lateral_offset_px = 0.0
        curvature_m = float("inf")
        confidence = 0.0

        if left.valid and right.valid:
            # Centerline is the midpoint of left/right at each y.
            xs_l = left.eval(self._y_pixels)
            xs_r = right.eval(self._y_pixels)
            centerline = np.stack([
                (xs_l + xs_r) * 0.5, self._y_pixels], axis=1)
            lane_width_px = float(np.mean(xs_r - xs_l))
            center_x_at_ego = float(centerline[-1, 0])
            lateral_offset_px = float(ego_x - center_x_at_ego)
            lateral_offset_m = lateral_offset_px * self.meters_per_pixel_x
            # Average curvature of the two polynomials.
            rl = left.curvature_radius_m(
                y_eval, self.meters_per_pixel_x, self.meters_per_pixel_y)
            rr = right.curvature_radius_m(
                y_eval, self.meters_per_pixel_x, self.meters_per_pixel_y)
            curvature_m = float(np.mean([rl, rr]))
            confidence = min(1.0,
                (left.pixel_points + right.pixel_points) /
                (2 * self.min_lane_pixels * 4))
        elif left.valid:
            # Assume ego is 1.8 m (or ~ lane_width/2) right of the left lane.
            lane_width_px = 350.0
            centerline = np.stack([
                left.eval(self._y_pixels) + lane_width_px * 0.5,
                self._y_pixels], axis=1)
            center_x_at_ego = float(centerline[-1, 0])
            lateral_offset_px = float(ego_x - center_x_at_ego)
            lateral_offset_m = lateral_offset_px * self.meters_per_pixel_x
            curvature_m = left.curvature_radius_m(
                y_eval, self.meters_per_pixel_x, self.meters_per_pixel_y)
            confidence = min(0.6, left.pixel_points /
                              (self.min_lane_pixels * 4))
        elif right.valid:
            lane_width_px = 350.0
            centerline = np.stack([
                right.eval(self._y_pixels) - lane_width_px * 0.5,
                self._y_pixels], axis=1)
            center_x_at_ego = float(centerline[-1, 0])
            lateral_offset_px = float(ego_x - center_x_at_ego)
            lateral_offset_m = lateral_offset_px * self.meters_per_pixel_x
            curvature_m = right.curvature_radius_m(
                y_eval, self.meters_per_pixel_x, self.meters_per_pixel_y)
            confidence = min(0.6, right.pixel_points /
                              (self.min_lane_pixels * 4))

        return LaneDetectionResult(
            left_lane=left,
            right_lane=right,
            centerline=centerline,
            lane_width_px=lane_width_px,
            lateral_offset_px=lateral_offset_px,
            lateral_offset_m=lateral_offset_m,
            curvature_radius_m=curvature_m,
            confidence=float(confidence),
        )

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def draw(
        self,
        image: np.ndarray,
        result: LaneDetectionResult,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Draw lane polygon + centerline on the image (returns a copy)."""
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for draw().")
        out = image.copy()
        if result.centerline is None:
            return out
        h, w = out.shape[:2]
        ys = np.linspace(0, h - 1, h, dtype=np.float32)
        overlay = out.copy()

        if result.left_lane and result.left_lane.valid:
            xl = result.left_lane.eval(ys)
            pts = np.stack([xl, ys], axis=1).astype(np.int32)
            cv2.polylines(overlay, [pts], False, (0, 255, 0), 8)
        if result.right_lane and result.right_lane.valid:
            xr = result.right_lane.eval(ys)
            pts = np.stack([xr, ys], axis=1).astype(np.int32)
            cv2.polylines(overlay, [pts], False, (0, 0, 255), 8)

        if (result.left_lane and result.left_lane.valid and
                result.right_lane and result.right_lane.valid):
            xl = result.left_lane.eval(ys)
            xr = result.right_lane.eval(ys)
            poly = np.concatenate([
                np.stack([xl, ys], axis=1),
                np.stack([xr[::-1], ys[::-1]], axis=1),
            ]).astype(np.int32)
            cv2.fillPoly(overlay, [poly], (0, 255, 255))

        # Centerline in blue.
        cl = result.centerline.astype(np.int32)
        cv2.polylines(overlay, [cl], False, (255, 0, 0), 3)
        return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Synthetic test — a 720x1280 mask with two diagonal lane stripes.
    h, w = 720, 1280
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (300, h), (550, 0), 255, 30)
    cv2.line(mask, (980, h), (730, 0), 255, 30)
    det = LaneDetector(image_shape=(h, w))
    # Pretend the mask is a 1-class segmentation output.
    seg = (mask > 0).astype(np.uint8)
    res = det.detect_from_segmentation(seg)
    print(f"Lateral offset (px): {res.lateral_offset_px:.1f}")
    print(f"Curvature (m): {res.curvature_radius_m:.1f}")
    print(f"Confidence: {res.confidence:.2f}")
