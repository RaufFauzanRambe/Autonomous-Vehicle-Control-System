"""Lane-line extraction from an edge map with a trapezoidal ROI and Hough segments.

Extension point: replace the per-side least-squares line fit with a
sliding-window polynomial fit or a RANSAC-based model, and add temporal
filtering of the lane parameters across frames.
"""

from __future__ import annotations

import cv2
import numpy as np

_MIN_ABS_SLOPE = 0.3  # [-] reject near-horizontal segments below this |slope|


class LaneDetector:
    """Detect left/right lane lines in a BGR frame using Hough segments.

    Pipeline: optional external edge map -> trapezoidal ROI mask ->
    probabilistic Hough transform -> slope-based left/right assignment ->
    per-side least-squares line fit in image coordinates.

    Extension point: swap the Hough + polyfit stage for a deep lane
    model (e.g. LaneNet) and fit higher-order polynomials in bird's-eye
    view coordinates.
    """

    def __init__(self, config: dict) -> None:
        """Configure the detector.

        Args:
            config: ``lane_detection`` section of the module YAML with keys
                ``hough_threshold`` [-] (minimum Hough votes),
                ``min_line_length_px`` [px], ``max_line_gap_px`` [px] and
                ``roi_top_ratio`` [-] (fraction of image height where the
                trapezoidal ROI starts, 0.4-0.7 typical).
        """
        self._hough_threshold = int(config["hough_threshold"])
        self._min_line_length_px = int(config["min_line_length_px"])
        self._max_line_gap_px = int(config["max_line_gap_px"])
        self._roi_top_ratio = float(config["roi_top_ratio"])

    def detect(self, image: np.ndarray, edges: np.ndarray | None = None) -> dict:
        """Estimate averaged left and right lane lines for one frame.

        Segments whose slope magnitude is below 0.3 (near-horizontal
        clutter) and exactly vertical segments (undefined slope sign) are
        rejected. Left segments have ``slope < 0`` and right segments have
        ``slope > 0`` in image coordinates (y grows downward). Each side
        is averaged by fitting ``y = slope * x + intercept`` to all
        segment endpoints with ``numpy.polyfit``.

        Args:
            image: BGR frame, ``uint8`` array ``(height, width, 3)``; used
                to compute edges when ``edges`` is not supplied and for
                the frame geometry.
            edges: Optional precomputed Canny edge map, ``uint8`` array
                ``(height, width)`` with values in {0, 255}. When ``None``,
                edges are computed internally with generic Canny
                thresholds (100, 200).

        Returns:
            Dictionary with keys:

            * ``"left_line"``: ``(slope, intercept)`` [1, px] of the left
              lane in image coordinates, or ``None`` if no segment passed.
            * ``"right_line"``: ``(slope, intercept)`` [1, px] of the
              right lane, or ``None``.
            * ``"n_segments"``: number of Hough segments kept after the
              slope gate [-].
        """
        height, width = image.shape[:2]
        if edges is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

        # Trapezoidal ROI: full width at the bottom, narrowed at the top.
        y_top = int(self._roi_top_ratio * height)
        roi_polygon = np.array(
            [
                [0, height - 1],
                [int(0.45 * width), y_top],
                [int(0.55 * width), y_top],
                [width - 1, height - 1],
            ],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_polygon], 255)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

        hough = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self._hough_threshold,
            minLineLength=self._min_line_length_px,
            maxLineGap=self._max_line_gap_px,
        )

        left_x: list[float] = []
        left_y: list[float] = []
        right_x: list[float] = []
        right_y: list[float] = []
        n_segments = 0

        if hough is not None:
            for x1, y1, x2, y2 in hough[:, 0]:
                dx = float(x2 - x1)
                if abs(dx) < 1e-6:
                    # Exactly vertical segment: no usable slope sign.
                    continue
                slope = float(y2 - y1) / dx
                if abs(slope) < _MIN_ABS_SLOPE:
                    continue
                n_segments += 1
                if slope < 0.0:
                    left_x.extend([float(x1), float(x2)])
                    left_y.extend([float(y1), float(y2)])
                else:
                    right_x.extend([float(x1), float(x2)])
                    right_y.extend([float(y1), float(y2)])

        left_line = self._fit_line(left_x, left_y)
        right_line = self._fit_line(right_x, right_y)
        return {"left_line": left_line, "right_line": right_line, "n_segments": n_segments}

    @staticmethod
    def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
        """Least-squares fit ``y = slope * x + intercept`` to endpoints.

        Args:
            xs: x coordinates [px] of all segment endpoints of one side.
            ys: y coordinates [px] of the same endpoints.

        Returns:
            ``(slope, intercept)`` with slope [1] and intercept [px], or
            ``None`` when fewer than two points are available.
        """
        if len(xs) < 2:
            return None
        slope, intercept = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
        return (float(slope), float(intercept))
