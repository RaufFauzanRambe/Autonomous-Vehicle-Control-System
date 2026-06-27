"""
Lane Curvature Estimation Module
=================================

Estimates lane curvature using polynomial fitting on bird's eye view
images, computes radius of curvature, heading angle, and lateral
offset from lane center.

Classes:
    CurvatureResult - Dataclass for curvature estimation results
    CurvatureEstimator - Main curvature estimation interface

Typical usage:
    >>> estimator = CurvatureEstimator(config)
    >>> result = estimator.estimate(bev_binary, left_base, right_base)
    >>> print(f"Radius: {result.radius_of_curvature:.1f}m")
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CurvatureResult:
    """Result of lane curvature estimation.

    Attributes:
        left_fit: Polynomial coefficients for left lane [A, B, C] (y = Ax² + Bx + C).
        right_fit: Polynomial coefficients for right lane [A, B, C].
        left_fit_world: Polynomial coefficients in world coordinates.
        right_fit_world: Polynomial coefficients in world coordinates.
        left_curve_radius: Radius of curvature for left lane in meters.
        right_curve_radius: Radius of curvature for right lane in meters.
        radius_of_curvature: Combined (average) radius of curvature in meters.
        center_offset: Lateral offset from lane center in meters (positive = right).
        heading_angle: Heading angle relative to lane center in degrees.
        left_lane_pts: Pixel coordinates of fitted left lane.
        right_lane_pts: Pixel coordinates of fitted right lane.
        confidence: Detection confidence [0, 1].
        is_valid: Whether the estimation is considered valid.
    """
    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None
    left_fit_world: Optional[np.ndarray] = None
    right_fit_world: Optional[np.ndarray] = None
    left_curve_radius: float = 0.0
    right_curve_radius: float = 0.0
    radius_of_curvature: float = 0.0
    center_offset: float = 0.0
    heading_angle: float = 0.0
    left_lane_pts: Optional[np.ndarray] = None
    right_lane_pts: Optional[np.ndarray] = None
    confidence: float = 0.0
    is_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curvature result."""
        return {
            "left_curve_radius": round(self.left_curve_radius, 2),
            "right_curve_radius": round(self.right_curve_radius, 2),
            "radius_of_curvature": round(self.radius_of_curvature, 2),
            "center_offset": round(self.center_offset, 4),
            "heading_angle": round(self.heading_angle, 2),
            "confidence": round(self.confidence, 3),
            "is_valid": self.is_valid,
        }


class CurvatureEstimator:
    """Lane curvature estimator using sliding window and polynomial fitting.

    Implements a sliding window histogram approach to detect lane pixels
    in bird's eye view images, then fits second-degree polynomials to
    compute curvature radius, lateral offset, and heading angle.

    Attributes:
        config: Configuration dictionary for curvature estimation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the curvature estimator.

        Args:
            config: Configuration dictionary with lane_curvature settings.
        """
        self.config = config
        self._sw_cfg = config.get("sliding_window", {})
        self._poly_cfg = config.get("polynomial_degree", 2)
        self._min_samples = config.get("min_sample_points", 10)

        # Physical conversion factors
        self.ym_per_pix = config.get("ym_per_pix", 0.04167)
        self.xm_per_pix = config.get("xm_per_pix", 0.00578)

        # Valid curvature range
        self.min_radius = config.get("min_radius", 100.0)
        self.max_radius = config.get("max_radius", 5000.0)

        # Temporal smoothing
        self._alpha = config.get("temporal_smoothing_alpha", 0.3)
        self._prev_left_fit: Optional[np.ndarray] = None
        self._prev_right_fit: Optional[np.ndarray] = None

        # Heading estimation
        self._heading_cfg = config.get("heading", {})
        self._lookahead = self._heading_cfg.get("lookahead_distance", 10.0)
        self._max_heading = self._heading_cfg.get("max_heading_angle", 45.0)

        logger.info("CurvatureEstimator initialized")

    def estimate(
        self,
        bev_binary: np.ndarray,
        left_base: Optional[int] = None,
        right_base: Optional[int] = None,
    ) -> CurvatureResult:
        """Estimate lane curvature from a bird's eye view binary image.

        Uses a sliding window approach to find lane pixels, then fits
        polynomials to compute curvature and offset.

        Args:
            bev_binary: Binary (thresholded) bird's eye view image.
            left_base: X-position of left lane base (auto-detected if None).
            right_base: X-position of right lane base (auto-detected if None).

        Returns:
            CurvatureResult with all estimated parameters.
        """
        result = CurvatureResult()

        if bev_binary is None or bev_binary.size == 0:
            logger.warning("Empty BEV image for curvature estimation")
            return result

        # Auto-detect lane bases if not provided
        if left_base is None or right_base is None:
            left_base, right_base = self._detect_lane_bases(bev_binary)

        if left_base is None or right_base is None:
            logger.warning("Could not detect lane bases")
            return result

        # Find lane pixels using sliding windows
        left_x, left_y, right_x, right_y, out_img = self._sliding_window_search(
            bev_binary, left_base, right_base
        )

        # Fit polynomials in pixel space
        left_fit = self._fit_polynomial(left_y, left_x)
        right_fit = self._fit_polynomial(right_y, right_x)

        if left_fit is None or right_fit is None:
            logger.warning("Polynomial fitting failed")
            return result

        # Apply temporal smoothing
        left_fit = self._smooth_fit(left_fit, self._prev_left_fit)
        right_fit = self._smooth_fit(right_fit, self._prev_right_fit)
        self._prev_left_fit = left_fit.copy()
        self._prev_right_fit = right_fit.copy()

        # Store pixel-space fits
        result.left_fit = left_fit
        result.right_fit = right_fit

        # Generate fitted lane points for visualization
        plot_y = np.linspace(0, bev_binary.shape[0] - 1, bev_binary.shape[0])
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        result.left_lane_pts = np.column_stack([left_fit_x, plot_y])
        result.right_lane_pts = np.column_stack([right_fit_x, plot_y])

        # Fit polynomials in world coordinates
        left_fit_world = self._fit_polynomial_world(left_y, left_x)
        right_fit_world = self._fit_polynomial_world(right_y, right_x)

        result.left_fit_world = left_fit_world
        result.right_fit_world = right_fit_world

        # Compute curvature radii
        result.left_curve_radius = self._compute_curvature_radius(left_fit_world)
        result.right_curve_radius = self._compute_curvature_radius(right_fit_world)

        # Average curvature
        if result.left_curve_radius > 0 and result.right_curve_radius > 0:
            result.radius_of_curvature = (
                (result.left_curve_radius + result.right_curve_radius) / 2.0
            )
        elif result.left_curve_radius > 0:
            result.radius_of_curvature = result.left_curve_radius
        elif result.right_curve_radius > 0:
            result.radius_of_curvature = result.right_curve_radius

        # Clamp to valid range
        result.radius_of_curvature = np.clip(
            result.radius_of_curvature, self.min_radius, self.max_radius
        )

        # Compute lateral offset
        result.center_offset = self._compute_center_offset(
            left_fit, right_fit, bev_binary.shape
        )

        # Compute heading angle
        result.heading_angle = self._compute_heading_angle(
            left_fit, right_fit, bev_binary.shape
        )

        # Compute confidence
        result.confidence = self._compute_confidence(
            left_x, right_x, left_fit, right_fit, bev_binary.shape
        )

        # Validate result
        result.is_valid = (
            result.confidence > 0.3
            and abs(result.center_offset) < 3.0
            and abs(result.heading_angle) < self._max_heading
        )

        logger.debug(
            f"Curvature: R={result.radius_of_curvature:.1f}m, "
            f"offset={result.center_offset:.3f}m, "
            f"heading={result.heading_angle:.2f}°, "
            f"conf={result.confidence:.2f}"
        )
        return result

    def estimate_from_prior(
        self,
        bev_binary: np.ndarray,
        prev_left_fit: np.ndarray,
        prev_right_fit: np.ndarray,
    ) -> CurvatureResult:
        """Estimate curvature using a previous fit as a starting point.

        More efficient than full sliding window search when a good
        prior estimate is available.

        Args:
            bev_binary: Binary BEV image.
            prev_left_fit: Previous left lane polynomial coefficients.
            prev_right_fit: Previous right lane polynomial coefficients.

        Returns:
            CurvatureResult with updated estimates.
        """
        result = CurvatureResult()
        margin = self._sw_cfg.get("margin", 80)

        # Identify lane pixels within margin of previous fit
        nonzero = bev_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (nonzerox > (prev_left_fit[0] * nonzeroy ** 2 +
                         prev_left_fit[1] * nonzeroy + prev_left_fit[2] - margin)) &
            (nonzerox < (prev_left_fit[0] * nonzeroy ** 2 +
                         prev_left_fit[1] * nonzeroy + prev_left_fit[2] + margin))
        )
        right_lane_inds = (
            (nonzerox > (prev_right_fit[0] * nonzeroy ** 2 +
                         prev_right_fit[1] * nonzeroy + prev_right_fit[2] - margin)) &
            (nonzerox < (prev_right_fit[0] * nonzeroy ** 2 +
                         prev_right_fit[1] * nonzeroy + prev_right_fit[2] + margin))
        )

        left_x = nonzerox[left_lane_inds]
        left_y = nonzeroy[left_lane_inds]
        right_x = nonzerox[right_lane_inds]
        right_y = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fit = self._fit_polynomial(left_y, left_x)
        right_fit = self._fit_polynomial(right_y, right_x)

        if left_fit is None or right_fit is None:
            # Fall back to previous fit
            result.left_fit = prev_left_fit
            result.right_fit = prev_right_fit
            result.confidence = 0.2
            result.is_valid = False
            return result

        # Apply smoothing
        left_fit = self._smooth_fit(left_fit, prev_left_fit)
        right_fit = self._smooth_fit(right_fit, prev_right_fit)

        result.left_fit = left_fit
        result.right_fit = right_fit

        # Generate points
        plot_y = np.linspace(0, bev_binary.shape[0] - 1, bev_binary.shape[0])
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
        result.left_lane_pts = np.column_stack([left_fit_x, plot_y])
        result.right_lane_pts = np.column_stack([right_fit_x, plot_y])

        # World-space fits
        left_fit_world = self._fit_polynomial_world(left_y, left_x)
        right_fit_world = self._fit_polynomial_world(right_y, right_x)
        result.left_fit_world = left_fit_world
        result.right_fit_world = right_fit_world

        result.left_curve_radius = self._compute_curvature_radius(left_fit_world)
        result.right_curve_radius = self._compute_curvature_radius(right_fit_world)
        result.radius_of_curvature = (
            (result.left_curve_radius + result.right_curve_radius) / 2.0
        )
        result.radius_of_curvature = np.clip(
            result.radius_of_curvature, self.min_radius, self.max_radius
        )
        result.center_offset = self._compute_center_offset(
            left_fit, right_fit, bev_binary.shape
        )
        result.heading_angle = self._compute_heading_angle(
            left_fit, right_fit, bev_binary.shape
        )
        result.confidence = self._compute_confidence(
            left_x, right_x, left_fit, right_fit, bev_binary.shape
        )
        result.is_valid = result.confidence > 0.3

        return result

    def _detect_lane_bases(
        self, bev_binary: np.ndarray
    ) -> Tuple[Optional[int], Optional[int]]:
        """Detect the base x-positions of left and right lanes.

        Uses a histogram of the bottom portion of the BEV image
        to find lane starting positions.

        Args:
            bev_binary: Binary BEV image.

        Returns:
            Tuple of (left_base_x, right_base_x) or (None, None).
        """
        histogram = np.sum(
            bev_binary[bev_binary.shape[0] // 2:, :], axis=0
        )
        midpoint = len(histogram) // 2

        left_base = int(np.argmax(histogram[:midpoint]))
        right_base = midpoint + int(np.argmax(histogram[midpoint:]))

        # Validate peaks
        if histogram[left_base] < 10:
            left_base = None
        if histogram[midpoint:] is not None and histogram[right_base] < 10:
            right_base = None

        return left_base, right_base

    def _sliding_window_search(
        self,
        binary: np.ndarray,
        left_base: int,
        right_base: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Search for lane pixels using sliding windows.

        Args:
            binary: Binary BEV image.
            left_base: Left lane base x-position.
            right_base: Right lane base x-position.

        Returns:
            Tuple of (left_x, left_y, right_x, right_y, output_image).
        """
        n_windows = self._sw_cfg.get("n_windows", 9)
        window_width = self._sw_cfg.get("window_width", 100)
        min_pixels = self._sw_cfg.get("min_pixels", 50)

        h, w = binary.shape
        window_height = h // n_windows

        # Identify nonzero pixels
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_x_pts: List[int] = []
        left_y_pts: List[int] = []
        right_x_pts: List[int] = []
        right_y_pts: List[int] = []

        left_current = left_base
        right_current = right_base

        out_img = np.dstack((binary, binary, binary)) * 255

        for window in range(n_windows):
            # Window boundaries
            y_low = h - (window + 1) * window_height
            y_high = h - window * window_height
            x_left_low = left_current - window_width // 2
            x_left_high = left_current + window_width // 2
            x_right_low = right_current - window_width // 2
            x_right_high = right_current + window_width // 2

            # Draw windows on output image
            cv2.rectangle(
                out_img, (x_left_low, y_low), (x_left_high, y_high), (0, 255, 0), 2
            )
            cv2.rectangle(
                out_img, (x_right_low, y_low), (x_right_high, y_high), (0, 255, 0), 2
            )

            # Identify nonzero pixels within windows
            good_left_inds = (
                (nonzeroy >= y_low) & (nonzeroy < y_high) &
                (nonzerox >= x_left_low) & (nonzerox < x_left_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= y_low) & (nonzeroy < y_high) &
                (nonzerox >= x_right_low) & (nonzerox < x_right_high)
            ).nonzero()[0]

            # Append pixel indices
            left_x_pts.extend(nonzerox[good_left_inds].tolist())
            left_y_pts.extend(nonzeroy[good_left_inds].tolist())
            right_x_pts.extend(nonzerox[good_right_inds].tolist())
            right_y_pts.extend(nonzeroy[good_right_inds].tolist())

            # Recenter next window if enough pixels found
            if len(good_left_inds) > min_pixels:
                left_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > min_pixels:
                right_current = int(np.mean(nonzerox[good_right_inds]))

        return (
            np.array(left_x_pts), np.array(left_y_pts),
            np.array(right_x_pts), np.array(right_y_pts),
            out_img,
        )

    def _fit_polynomial(
        self, y: np.ndarray, x: np.ndarray
    ) -> Optional[np.ndarray]:
        """Fit a polynomial to lane pixel coordinates.

        Args:
            y: Y-coordinates of lane pixels.
            x: X-coordinates of lane pixels.

        Returns:
            Polynomial coefficients [A, B, C] or None if fitting fails.
        """
        if len(y) < self._min_samples:
            logger.debug(
                f"Insufficient points for fitting: {len(y)} < {self._min_samples}"
            )
            return None

        try:
            coeffs = np.polyfit(y, x, self._poly_cfg)
            return coeffs
        except (np.RankWarning, np.linalg.LinAlgError) as e:
            logger.warning(f"Polynomial fitting failed: {e}")
            return None

    def _fit_polynomial_world(
        self, y_px: np.ndarray, x_px: np.ndarray
    ) -> Optional[np.ndarray]:
        """Fit a polynomial in world coordinates.

        Converts pixel coordinates to meters before fitting.

        Args:
            y_px: Y-coordinates in pixels.
            x_px: X-coordinates in pixels.

        Returns:
            Polynomial coefficients in world coordinates [A_w, B_w, C_w].
        """
        if len(y_px) < self._min_samples:
            return None

        # Convert to world coordinates
        y_world = y_px.astype(np.float64) * self.ym_per_pix
        x_world = x_px.astype(np.float64) * self.xm_per_pix

        try:
            return np.polyfit(y_world, x_world, self._poly_cfg)
        except (np.RankWarning, np.linalg.LinAlgError):
            return None

    def _compute_curvature_radius(
        self, fit_world: Optional[np.ndarray]
    ) -> float:
        """Compute radius of curvature from world-space polynomial.

        For y = Ax² + Bx + C, the radius at point y is:
        R = ((1 + (2Ay + B)²)^(3/2)) / |2A|

        Args:
            fit_world: Polynomial coefficients in world coordinates.

        Returns:
            Radius of curvature in meters.
        """
        if fit_world is None:
            return 0.0

        A = fit_world[0]
        B = fit_world[1]

        # Evaluate at the bottom of the image (closest to vehicle)
        y_eval = 30.0 * self.ym_per_pix  # Max y in world coords

        numerator = (1.0 + (2.0 * A * y_eval + B) ** 2) ** 1.5
        denominator = abs(2.0 * A)

        if denominator < 1e-10:
            return float('inf')

        radius = numerator / denominator
        return float(np.clip(radius, self.min_radius, self.max_radius))

    def _compute_center_offset(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        image_shape: Tuple[int, ...],
    ) -> float:
        """Compute lateral offset from lane center.

        Args:
            left_fit: Left lane polynomial coefficients.
            right_fit: Right lane polynomial coefficients.
            image_shape: Shape of the BEV image.

        Returns:
            Offset in meters (positive = right of center).
        """
        h = image_shape[0]
        # Evaluate at the bottom of the image
        y_eval = h - 1

        left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        lane_center_px = (left_x + right_x) / 2.0
        image_center_px = image_shape[1] / 2.0

        offset_px = image_center_px - lane_center_px
        offset_m = offset_px * self.xm_per_pix

        return float(offset_m)

    def _compute_heading_angle(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        image_shape: Tuple[int, ...],
    ) -> float:
        """Compute heading angle relative to lane direction.

        Args:
            left_fit: Left lane polynomial coefficients.
            right_fit: Right lane polynomial coefficients.
            image_shape: Shape of the BEV image.

        Returns:
            Heading angle in degrees.
        """
        h = image_shape[0]
        y_eval = h - 1

        # Evaluate the derivative at the bottom
        avg_fit = (left_fit + right_fit) / 2.0
        # dy/dx for polynomial: x = Ay² + By + C -> dx/dy = 2Ay + B
        # In BEV: y is forward, x is lateral
        dx_dy = 2.0 * avg_fit[0] * y_eval + avg_fit[1]

        # Heading angle: arctan of the lateral rate
        angle_rad = np.arctan(dx_dy)
        angle_deg = np.degrees(angle_rad)

        return float(np.clip(angle_deg, -self._max_heading, self._max_heading))

    def _smooth_fit(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply exponential moving average smoothing to polynomial fits.

        Args:
            current: Current polynomial fit.
            previous: Previous polynomial fit (None for first frame).

        Returns:
            Smoothed polynomial coefficients.
        """
        if previous is None:
            return current

        alpha = self._alpha
        return alpha * current + (1.0 - alpha) * previous

    def _compute_confidence(
        self,
        left_x: np.ndarray,
        right_x: np.ndarray,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        image_shape: Tuple[int, ...],
    ) -> float:
        """Compute confidence score for the curvature estimation.

        Based on the number of detected lane pixels and the
        reasonableness of the fitted polynomials.

        Args:
            left_x: Left lane x-pixel coordinates.
            right_x: Right lane x-pixel coordinates.
            left_fit: Left lane polynomial fit.
            right_fit: Right lane polynomial fit.
            image_shape: Shape of the BEV image.

        Returns:
            Confidence score [0, 1].
        """
        total_pixels = image_shape[0] * image_shape[1]

        # Pixel count score
        left_coverage = len(left_x) / max(total_pixels * 0.01, 1)
        right_coverage = len(right_x) / max(total_pixels * 0.01, 1)
        pixel_score = min(1.0, (left_coverage + right_coverage) / 2.0)

        # Lane width reasonableness
        h = image_shape[0]
        y_eval = h - 1
        left_x_eval = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_x_eval = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        lane_width_px = right_x_eval - left_x_eval
        expected_width = 3.7 / self.xm_per_pix  # Standard lane width in pixels
        width_ratio = lane_width_px / expected_width if expected_width > 0 else 0

        if 0.5 < width_ratio < 2.0:
            width_score = 1.0 - abs(1.0 - width_ratio) * 0.5
        else:
            width_score = 0.0

        # Curvature reasonableness
        curv_score = 1.0
        left_A = abs(left_fit[0])
        right_A = abs(right_fit[0])
        if left_A > 0.001 or right_A > 0.001:
            # Very high curvature - potentially unstable
            curv_score = 0.5

        confidence = 0.4 * pixel_score + 0.4 * width_score + 0.2 * curv_score
        return float(np.clip(confidence, 0.0, 1.0))

    def reset(self) -> None:
        """Reset temporal smoothing state."""
        self._prev_left_fit = None
        self._prev_right_fit = None
        logger.debug("CurvatureEstimator state reset")
