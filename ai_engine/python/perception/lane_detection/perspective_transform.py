"""
Perspective Transform Module
============================

Implements bird's eye view (BEV) transformation and inverse perspective
mapping (IPM) for lane detection, enabling curvature measurement in
real-world coordinates.

Classes:
    PerspectiveTransformer - BEV transform and IPM operations

Typical usage:
    >>> transformer = PerspectiveTransformer(config)
    >>> bev_image = transformer.to_bev(image)
    >>> inv_image = transformer.from_bev(bev_image)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PerspectiveTransformer:
    """Bird's eye view transformation for lane detection.

    Converts perspective-distorted front-camera views to top-down
    (bird's eye) views where lane lines appear parallel, enabling
    accurate curvature and offset measurements in world coordinates.

    Attributes:
        config: Configuration dictionary for transform parameters.
        M: Forward perspective transformation matrix (3x3).
        M_inv: Inverse perspective transformation matrix (3x3).
        src_points: Source points in image coordinates.
        dst_points: Destination points in BEV coordinates.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the perspective transformer.

        Computes the forward and inverse homography matrices from
        the configured source and destination point pairs.

        Args:
            config: Configuration dictionary with perspective_transform
                section containing src_points, dst_points, and bev params.
        """
        self.config = config

        # Load source and destination points
        pts_config = config.get("perspective_transform", config)
        self.src_points = np.array(
            pts_config.get("src_points", [
                [220, 720], [560, 475], [720, 475], [1060, 720]
            ]),
            dtype=np.float32,
        )
        self.dst_points = np.array(
            pts_config.get("dst_points", [
                [320, 720], [320, 0], [960, 0], [960, 720]
            ]),
            dtype=np.float32,
        )

        # BEV physical dimensions
        bev_config = pts_config.get("bev", {})
        self.bev_width_m = bev_config.get("width", 3.7)
        self.bev_length_m = bev_config.get("length", 30.0)
        self.pixels_per_meter_x = bev_config.get("pixels_per_meter_x", 173)
        self.pixels_per_meter_y = bev_config.get("pixels_per_meter_y", 24)

        # Compute homography matrices
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # Validate matrices
        if self.M is None or self.M_inv is None:
            raise ValueError("Failed to compute perspective transform matrices")

        logger.info(
            f"PerspectiveTransformer initialized: "
            f"{self.src_points.shape[0]} point pairs, "
            f"BEV={self.bev_width_m}m x {self.bev_length_m}m"
        )

    def to_bev(
        self,
        image: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Transform an image from front-camera view to bird's eye view.

        Args:
            image: Input image in front-camera perspective.
            output_size: Optional (width, height) for output BEV image.
                Defaults to the size determined by destination points.
            interpolation: OpenCV interpolation method.

        Returns:
            Bird's eye view image.
        """
        if output_size is None:
            # Determine size from destination points
            max_x = int(self.dst_points[:, 0].max())
            max_y = int(self.dst_points[:, 1].max())
            output_size = (max_x, max_y)

        bev = cv2.warpPerspective(
            image, self.M, output_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        logger.debug(
            f"BEV transform: {image.shape} -> {bev.shape}"
        )
        return bev

    def from_bev(
        self,
        bev_image: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Transform an image from bird's eye view back to front-camera view.

        Args:
            bev_image: Input bird's eye view image.
            output_size: Optional (width, height) for output image.
            interpolation: OpenCV interpolation method.

        Returns:
            Image in front-camera perspective.
        """
        if output_size is None:
            max_x = int(self.src_points[:, 0].max())
            max_y = int(self.src_points[:, 1].max())
            output_size = (max_x, max_y)

        original = cv2.warpPerspective(
            bev_image, self.M_inv, output_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        logger.debug(
            f"Inverse BEV transform: {bev_image.shape} -> {original.shape}"
        )
        return original

    def transform_points(
        self, points: np.ndarray, inverse: bool = False
    ) -> np.ndarray:
        """Transform a set of 2D points between perspectives.

        Args:
            points: Input points as Nx2 array (or Nx1x2).
            inverse: If True, use inverse transform (BEV -> original).

        Returns:
            Transformed points as Nx2 array.
        """
        # Ensure proper shape for cv2.perspectiveTransform
        if points.ndim == 2:
            pts = points.reshape(-1, 1, 2).astype(np.float32)
        else:
            pts = points.astype(np.float32)

        matrix = self.M_inv if inverse else self.M
        transformed = cv2.perspectiveTransform(pts, matrix)

        if transformed.ndim == 3:
            return transformed.reshape(-1, 2)

        return transformed

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert BEV pixel coordinates to world coordinates (meters).

        Assumes the BEV image origin is at the top-center of the
        ground plane visible from the camera, with y-axis pointing
        forward (away from vehicle) and x-axis pointing right.

        Args:
            px: Pixel x-coordinate in BEV image.
            py: Pixel y-coordinate in BEV image.

        Returns:
            Tuple of (lateral_m, longitudinal_m) in world coordinates.
            lateral_m: Positive = right of center.
            longitudinal_m: Positive = forward from vehicle.
        """
        # Get BEV center
        bev_center_x = (self.dst_points[0][0] + self.dst_points[3][0]) / 2
        bev_center_y = self.dst_points[0][1]  # Bottom of BEV = vehicle position

        # Convert to world coordinates
        lateral_m = (px - bev_center_x) / self.pixels_per_meter_x
        longitudinal_m = (bev_center_y - py) / self.pixels_per_meter_y

        return lateral_m, longitudinal_m

    def world_to_pixel(self, lateral_m: float,
                       longitudinal_m: float) -> Tuple[float, float]:
        """Convert world coordinates to BEV pixel coordinates.

        Args:
            lateral_m: Lateral offset in meters (positive = right).
            longitudinal_m: Forward distance in meters.

        Returns:
            Tuple of (px, py) in BEV pixel coordinates.
        """
        bev_center_x = (self.dst_points[0][0] + self.dst_points[3][0]) / 2
        bev_center_y = self.dst_points[0][1]

        px = bev_center_x + lateral_m * self.pixels_per_meter_x
        py = bev_center_y - longitudinal_m * self.pixels_per_meter_y

        return px, py

    def get_bev_histogram(
        self,
        bev_image: np.ndarray,
        y_start: int = 0,
        y_end: Optional[int] = None,
    ) -> np.ndarray:
        """Compute a horizontal histogram of the BEV image for lane detection.

        Used to find the base positions of lane lines by identifying
        peaks in the histogram along the x-axis.

        Args:
            bev_image: Bird's eye view image (preferably edge-detected).
            y_start: Starting y row for the histogram region.
            y_end: Ending y row (defaults to bottom quarter).

        Returns:
            1D histogram array with counts per x-column.
        """
        if len(bev_image.shape) == 3:
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bev_image

        h = gray.shape[0]
        if y_end is None:
            y_end = h // 2  # Bottom half

        y_start = max(0, y_start)
        y_end = min(h, y_end)

        # Sum along columns in the specified row range
        histogram = np.sum(gray[y_start:y_end, :], axis=0).astype(np.int64)

        return histogram

    def find_histogram_peaks(
        self,
        histogram: np.ndarray,
        min_peak_height: int = 50,
        min_distance: int = 100,
    ) -> np.ndarray:
        """Find peaks in a histogram corresponding to lane line positions.

        Args:
            histogram: 1D histogram array.
            min_peak_height: Minimum height for a peak to be considered.
            min_distance: Minimum distance between peaks in pixels.

        Returns:
            Array of peak x-positions.
        """
        midpoint = len(histogram) // 2

        # Find the strongest peak in the left half
        left_peak = None
        left_max = np.max(histogram[:midpoint])
        if left_max > min_peak_height:
            left_peak = np.argmax(histogram[:midpoint])

        # Find the strongest peak in the right half
        right_peak = None
        right_max = np.max(histogram[midpoint:])
        if right_max > min_peak_height:
            right_peak = midpoint + np.argmax(histogram[midpoint:])

        peaks = []
        if left_peak is not None:
            peaks.append(left_peak)
        if right_peak is not None:
            peaks.append(right_peak)

        if not peaks:
            logger.debug("No histogram peaks found")
            return np.array([])

        return np.array(peaks, dtype=np.int32)

    def update_src_points(self, src_points: np.ndarray) -> None:
        """Dynamically update the source points and recompute homography.

        Useful for adapting to different camera positions or
        recalibrating the transform online.

        Args:
            src_points: New source points (4x2 array).
        """
        if src_points.shape != (4, 2):
            raise ValueError("Source points must be a 4x2 array")

        self.src_points = src_points.astype(np.float32)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        logger.info("Perspective transform updated with new source points")

    def compute_calibration_from_road(
        self,
        lane_width_px: float,
        vanishing_point: Tuple[float, float],
        image_height: int,
    ) -> None:
        """Compute perspective transform from road geometry.

        Automatically determines source points based on lane width
        and vanishing point, then updates the transform.

        Args:
            lane_width_px: Approximate lane width in pixels at bottom.
            vanishing_point: Vanishing point coordinates (x, y).
            image_height: Image height in pixels.
        """
        vp_x, vp_y = vanishing_point

        # Compute source points based on vanishing point geometry
        # Bottom of image: spread out by lane width
        bottom_y = image_height - 1
        left_bottom = vp_x - lane_width_px / 2
        right_bottom = vp_x + lane_width_px / 2

        # At vanishing point height: close together
        margin = lane_width_px * 0.05
        left_top = vp_x - margin
        right_top = vp_x + margin
        top_y = vp_y + 20  # Slightly below vanishing point

        self.src_points = np.array([
            [left_bottom, bottom_y],
            [left_top, top_y],
            [right_top, top_y],
            [right_bottom, bottom_y],
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        logger.info(
            f"Perspective transform calibrated from road: "
            f"lane_width={lane_width_px:.0f}px, "
            f"vanishing_point=({vp_x:.0f}, {vp_y:.0f})"
        )

    def validate_transform(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Validate the perspective transform by round-tripping a test image.

        Applies forward and inverse transforms and measures the
        reconstruction quality.

        Args:
            test_image: Test image to validate with.

        Returns:
            Dictionary with validation metrics.
        """
        bev = self.to_bev(test_image)
        reconstructed = self.from_bev(bev, output_size=(test_image.shape[1], test_image.shape[0]))

        # Compute MSE in the ROI region
        mask = (reconstructed > 0).astype(np.float32)
        if mask.sum() > 0:
            diff = np.abs(test_image.astype(np.float32) - reconstructed.astype(np.float32))
            mse = np.sum(diff * mask) / (mask.sum() * 3.0)
        else:
            mse = float('inf')

        return {
            "mse": float(mse),
            "psnr": float(10 * np.log10(255**2 / max(mse, 1e-10))),
            "bev_shape": bev.shape,
            "reconstructed_shape": reconstructed.shape,
            "valid": mse < 50.0,
        }
