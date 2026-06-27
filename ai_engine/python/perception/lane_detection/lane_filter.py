"""
Lane Filter Module
==================

Filters lane line candidates based on geometric constraints including
angle, length, parallelism, and region of interest.

Classes:
    LaneFilter - Lane candidate filtering and Hough line detection

Typical usage:
    >>> lane_filter = LaneFilter(config)
    >>> filtered_lines = lane_filter.filter(edge_image, raw_lines)
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LaneSide(Enum):
    """Enum for lane position relative to vehicle center."""
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class FilteredLine:
    """Represents a filtered lane line candidate.

    Attributes:
        start: Start point (x, y).
        end: End point (x, y).
        angle: Line angle in degrees from horizontal.
        length: Line length in pixels.
        side: Left/right classification.
        confidence: Detection confidence [0, 1].
    """

    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        angle: float,
        length: float,
        side: LaneSide = LaneSide.UNKNOWN,
        confidence: float = 1.0,
    ) -> None:
        self.start = start
        self.end = end
        self.angle = angle
        self.length = length
        self.side = side
        self.confidence = confidence

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Get the midpoint of the line."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )

    @property
    def slope(self) -> float:
        """Get the slope of the line."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        if abs(dx) < 1e-6:
            return float('inf')
        return dy / dx

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]."""
        return np.array([
            self.start[0], self.start[1],
            self.end[0], self.end[1]
        ])

    def __repr__(self) -> str:
        return (
            f"FilteredLine(start={self.start}, end={self.end}, "
            f"angle={self.angle:.1f}°, length={self.length:.1f}, "
            f"side={self.side.value}, conf={self.confidence:.2f})"
        )


class LaneFilter:
    """Lane candidate filter with geometric and spatial constraints.

    Applies angle, length, parallelism, and ROI filters to raw Hough
    lines to extract likely lane boundary candidates.

    Attributes:
        config: Configuration dictionary for filtering parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the lane filter.

        Args:
            config: Configuration dictionary with filter settings.
                Expected sections: min_angle, max_angle, min_length,
                max_line_gap, hough, parallel, midpoint_x_ratio,
                merge_distance, merge_angle_diff.
        """
        self.config = config
        self._hough_cfg = config.get("hough", {})
        self._parallel_cfg = config.get("parallel", {})

        logger.info("LaneFilter initialized")

    def detect_hough_lines(
        self, edge_image: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """Detect lines using probabilistic Hough transform.

        Args:
            edge_image: Binary edge image.

        Returns:
            List of (x1, y1, x2, y2) line segments.
        """
        rho = self._hough_cfg.get("rho", 1)
        theta = self._hough_cfg.get("theta", np.pi / 180)
        threshold = self._hough_cfg.get("threshold", 40)
        min_length = self._hough_cfg.get("min_line_length", 30)
        max_gap = self._hough_cfg.get("max_line_gap", 15)

        lines = cv2.HoughLinesP(
            edge_image, rho, theta, threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )

        if lines is None:
            logger.debug("No Hough lines detected")
            return []

        raw_lines = [
            (line[0][0], line[0][1], line[0][2], line[0][3])
            for line in lines
        ]

        logger.debug(f"Detected {len(raw_lines)} raw Hough lines")
        return raw_lines

    def filter(
        self,
        edge_image: np.ndarray,
        raw_lines: Optional[List[Tuple[float, float, float, float]]] = None,
        image_width: int = 1280,
    ) -> Tuple[List[FilteredLine], List[FilteredLine]]:
        """Apply the full filtering pipeline to lane line candidates.

        Pipeline:
        1. Detect Hough lines (if not provided)
        2. Compute line properties (angle, length)
        3. Apply angle filter
        4. Apply length filter
        5. Classify left/right
        6. Merge nearby lines
        7. Apply parallel constraint

        Args:
            edge_image: Binary edge image for Hough detection.
            raw_lines: Optional pre-detected Hough lines.
            image_width: Image width for left/right midpoint calculation.

        Returns:
            Tuple of (left_lane_lines, right_lane_lines).
        """
        # Step 1: Detect lines if not provided
        if raw_lines is None:
            raw_lines = self.detect_hough_lines(edge_image)

        if not raw_lines:
            return [], []

        # Step 2: Compute line properties and create FilteredLine objects
        filtered: List[FilteredLine] = []
        for x1, y1, x2, y2 in raw_lines:
            angle = self._compute_angle(x1, y1, x2, y2)
            length = self._compute_length(x1, y1, x2, y2)
            filtered.append(
                FilteredLine(
                    start=(x1, y1), end=(x2, y2),
                    angle=angle, length=length
                )
            )

        # Step 3: Angle filter
        min_angle = self.config.get("min_angle", 20.0)
        max_angle = self.config.get("max_angle", 85.0)
        filtered = self._angle_filter(filtered, min_angle, max_angle)
        logger.debug(f"After angle filter: {len(filtered)} lines")

        # Step 4: Length filter
        min_length = self.config.get("min_length", 30)
        filtered = self._length_filter(filtered, min_length)
        logger.debug(f"After length filter: {len(filtered)} lines")

        # Step 5: Classify left/right
        midpoint_x = image_width * self.config.get("midpoint_x_ratio", 0.5)
        filtered = self._classify_sides(filtered, midpoint_x)

        # Step 6: Merge nearby lines
        merge_dist = self.config.get("merge_distance", 20)
        merge_angle = self.config.get("merge_angle_diff", 10)
        left_lines = [l for l in filtered if l.side == LaneSide.LEFT]
        right_lines = [l for l in filtered if l.side == LaneSide.RIGHT]

        left_lines = self._merge_lines(left_lines, merge_dist, merge_angle)
        right_lines = self._merge_lines(right_lines, merge_dist, merge_angle)

        # Step 7: Parallel constraint
        if left_lines and right_lines:
            left_lines, right_lines = self._parallel_constraint(
                left_lines, right_lines
            )

        logger.debug(
            f"Filter result: {len(left_lines)} left, "
            f"{len(right_lines)} right lines"
        )
        return left_lines, right_lines

    def _compute_angle(self, x1: float, y1: float,
                       x2: float, y2: float) -> float:
        """Compute the angle of a line segment from horizontal.

        Args:
            x1, y1, x2, y2: Line endpoint coordinates.

        Returns:
            Angle in degrees [0, 90].
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def _compute_length(self, x1: float, y1: float,
                        x2: float, y2: float) -> float:
        """Compute the length of a line segment.

        Args:
            x1, y1, x2, y2: Line endpoint coordinates.

        Returns:
            Length in pixels.
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _angle_filter(
        self, lines: List[FilteredLine],
        min_angle: float, max_angle: float
    ) -> List[FilteredLine]:
        """Filter lines by angle from horizontal.

        Lane lines are expected to have angles roughly between 20° and 85°
        from horizontal (nearly vertical to moderately angled).

        Args:
            lines: Input line list.
            min_angle: Minimum acceptable angle in degrees.
            max_angle: Maximum acceptable angle in degrees.

        Returns:
            Filtered line list.
        """
        return [
            line for line in lines
            if min_angle <= line.angle <= max_angle
        ]

    def _length_filter(
        self, lines: List[FilteredLine], min_length: float
    ) -> List[FilteredLine]:
        """Filter lines by minimum length.

        Args:
            lines: Input line list.
            min_length: Minimum acceptable length in pixels.

        Returns:
            Filtered line list.
        """
        return [line for line in lines if line.length >= min_length]

    def _classify_sides(
        self, lines: List[FilteredLine], midpoint_x: float
    ) -> List[FilteredLine]:
        """Classify lines as left or right lane boundaries.

        Uses the midpoint x-coordinate of each line relative to the
        image midpoint to determine side assignment. Also considers
        the slope direction for more robust classification.

        Args:
            lines: Input line list.
            midpoint_x: X-coordinate dividing left and right halves.

        Returns:
            Lines with side classification updated.
        """
        for line in lines:
            mid_x = line.midpoint[0]
            slope = line.slope

            # Primary: use midpoint position
            if mid_x < midpoint_x:
                # Left lane lines typically have negative slope
                # (going from bottom-right to top-left)
                if slope < 0:
                    line.side = LaneSide.LEFT
                    line.confidence = 0.9
                elif slope > 0:
                    # Unusual but possible (e.g., merging lane)
                    line.side = LaneSide.LEFT
                    line.confidence = 0.5
                else:
                    line.side = LaneSide.LEFT
                    line.confidence = 0.6
            else:
                # Right lane lines typically have positive slope
                # (going from bottom-left to top-right)
                if slope > 0:
                    line.side = LaneSide.RIGHT
                    line.confidence = 0.9
                elif slope < 0:
                    line.side = LaneSide.RIGHT
                    line.confidence = 0.5
                else:
                    line.side = LaneSide.RIGHT
                    line.confidence = 0.6

        return lines

    def _merge_lines(
        self, lines: List[FilteredLine],
        merge_distance: float, merge_angle_diff: float
    ) -> List[FilteredLine]:
        """Merge nearby lines with similar angles.

        Groups lines that are close together and have similar orientations,
        then replaces each group with a single averaged line.

        Args:
            lines: Input line list (same side).
            merge_distance: Maximum distance between line midpoints to merge.
            merge_angle_diff: Maximum angle difference to merge (degrees).

        Returns:
            Merged line list.
        """
        if len(lines) <= 1:
            return lines

        merged: List[FilteredLine] = []
        used = [False] * len(lines)

        for i, line_a in enumerate(lines):
            if used[i]:
                continue

            group = [line_a]
            used[i] = True

            for j, line_b in enumerate(lines):
                if used[j]:
                    continue

                # Check distance between midpoints
                dist = np.sqrt(
                    (line_a.midpoint[0] - line_b.midpoint[0]) ** 2 +
                    (line_a.midpoint[1] - line_b.midpoint[1]) ** 2
                )

                # Check angle difference
                angle_diff = abs(line_a.angle - line_b.angle)

                if dist < merge_distance and angle_diff < merge_angle_diff:
                    group.append(line_b)
                    used[j] = True

            # Average the group
            merged.append(self._average_lines(group))

        logger.debug(f"Merged {len(lines)} -> {len(merged)} lines")
        return merged

    def _average_lines(self, lines: List[FilteredLine]) -> FilteredLine:
        """Average a group of lines into a single representative line.

        Uses weighted averaging based on line length.

        Args:
            lines: Lines to average.

        Returns:
            Averaged FilteredLine.
        """
        if len(lines) == 1:
            return lines[0]

        total_weight = sum(l.length for l in lines)

        if total_weight < 1e-6:
            return lines[0]

        avg_x1 = sum(l.start[0] * l.length for l in lines) / total_weight
        avg_y1 = sum(l.start[1] * l.length for l in lines) / total_weight
        avg_x2 = sum(l.end[0] * l.length for l in lines) / total_weight
        avg_y2 = sum(l.end[1] * l.length for l in lines) / total_weight
        avg_angle = sum(l.angle * l.length for l in lines) / total_weight
        avg_length = sum(l.length for l in lines) / len(lines)
        avg_confidence = sum(l.confidence for l in lines) / len(lines)

        return FilteredLine(
            start=(avg_x1, avg_y1),
            end=(avg_x2, avg_y2),
            angle=avg_angle,
            length=avg_length,
            side=lines[0].side,
            confidence=avg_confidence,
        )

    def _parallel_constraint(
        self,
        left_lines: List[FilteredLine],
        right_lines: List[FilteredLine],
    ) -> Tuple[List[FilteredLine], List[FilteredLine]]:
        """Apply parallel constraint between left and right lanes.

        Lane boundaries should be roughly parallel. This filter removes
        line pairs that don't satisfy the parallel constraint and
        pairs that are too close or too far apart.

        Args:
            left_lines: Left lane candidates.
            right_lines: Right lane candidates.

        Returns:
            Filtered (left_lines, right_lines).
        """
        max_angle_diff = self._parallel_cfg.get("max_angle_diff", 10.0)
        max_distance = self._parallel_cfg.get("max_distance", 600)
        min_distance = self._parallel_cfg.get("min_distance", 100)

        valid_left: List[FilteredLine] = []
        valid_right: List[FilteredLine] = []

        for left in left_lines:
            for right in right_lines:
                # Check angle similarity
                angle_diff = abs(left.angle - right.angle)
                if angle_diff > max_angle_diff:
                    continue

                # Check distance between midpoints
                dist = np.sqrt(
                    (left.midpoint[0] - right.midpoint[0]) ** 2 +
                    (left.midpoint[1] - right.midpoint[1]) ** 2
                )
                if dist < min_distance or dist > max_distance:
                    continue

                # Valid pair
                valid_left.append(left)
                valid_right.append(right)
                break  # Found a valid match for this left line

        return valid_left, valid_right

    def roi_filter(
        self,
        lines: List[FilteredLine],
        roi_vertices: np.ndarray,
    ) -> List[FilteredLine]:
        """Filter lines that fall within the region of interest.

        A line is considered inside the ROI if its midpoint is
        inside the ROI polygon.

        Args:
            lines: Input line list.
            roi_vertices: ROI polygon vertices (Nx2 array).

        Returns:
            Filtered line list.
        """
        filtered: List[FilteredLine] = []
        for line in lines:
            mid = line.midpoint
            dist = cv2.pointPolygonTest(
                roi_vertices.astype(np.float32),
                (float(mid[0]), float(mid[1])),
                False,
            )
            if dist >= 0:
                filtered.append(line)

        logger.debug(
            f"ROI filter: {len(lines)} -> {len(filtered)} lines"
        )
        return filtered
