"""
Lane Classification Module
==========================

Classifies detected lane lines by type (solid, dashed, double, merging)
and color (white, yellow) for semantic understanding of road markings.

Classes:
    LaneType - Enum for lane marking types
    LaneColor - Enum for lane marking colors
    LaneClassification - Dataclass for classification results
    LaneClassifier - Main classification interface

Typical usage:
    >>> classifier = LaneClassifier(config)
    >>> result = classifier.classify(line_image, line_pixels)
    >>> print(result.lane_type, result.lane_color)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LaneType(Enum):
    """Lane marking type classification."""
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    DOUBLE_DASHED = "double_dashed"
    SOLID_DASHED = "solid_dashed"      # Left solid, right dashed
    DASHED_SOLID = "dashed_solid"      # Left dashed, right solid
    MERGING = "merging"
    DIVERGING = "diverging"
    UNKNOWN = "unknown"


class LaneColor(Enum):
    """Lane marking color classification."""
    WHITE = "white"
    YELLOW = "yellow"
    BLUE = "blue"          # Special markings (e.g., handicapped parking)
    RED = "red"            # Bus lanes, no-stopping zones
    UNKNOWN = "unknown"


@dataclass
class LaneClassification:
    """Result of lane classification.

    Attributes:
        lane_type: Detected lane marking type.
        lane_color: Detected lane marking color.
        type_confidence: Confidence in type classification [0, 1].
        color_confidence: Confidence in color classification [0, 1].
        dash_ratio: Ratio of dash to total length (0 = solid, 1 = all dashes).
        dash_count: Number of detected dashes.
        avg_dash_length: Average dash length in pixels.
        avg_gap_length: Average gap length in pixels.
    """
    lane_type: LaneType = LaneType.UNKNOWN
    lane_color: LaneColor = LaneColor.UNKNOWN
    type_confidence: float = 0.0
    color_confidence: float = 0.0
    dash_ratio: float = 0.0
    dash_count: int = 0
    avg_dash_length: float = 0.0
    avg_gap_length: float = 0.0

    @property
    def overall_confidence(self) -> float:
        """Combined confidence score."""
        return (self.type_confidence + self.color_confidence) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize classification result."""
        return {
            "lane_type": self.lane_type.value,
            "lane_color": self.lane_color.value,
            "type_confidence": round(self.type_confidence, 3),
            "color_confidence": round(self.color_confidence, 3),
            "dash_ratio": round(self.dash_ratio, 3),
            "dash_count": self.dash_count,
            "avg_dash_length": round(self.avg_dash_length, 1),
            "avg_gap_length": round(self.avg_gap_length, 1),
        }


class LaneClassifier:
    """Lane marking type and color classifier.

    Analyzes detected lane line segments to determine their type
    (solid, dashed, double, etc.) and color (white, yellow).
    Uses both spatial analysis of line continuity and color sampling
    around line pixels.

    Attributes:
        config: Configuration dictionary for classification parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the lane classifier.

        Args:
            config: Configuration dictionary with lane_classifier settings.
        """
        self.config = config
        self._dash_cfg = config.get("dash_analysis", {})
        self._double_cfg = config.get("double_lane", {})
        self._merge_cfg = config.get("merging", {})
        self._color_cfg = config.get("color_classification", {})

        # Precompute color range arrays for HSV classification
        white_range = self._color_cfg.get("white_hsv", [[0, 0, 200], [180, 30, 255]])
        yellow_range = self._color_cfg.get("yellow_hsv", [[15, 80, 100], [35, 255, 255]])

        self._white_lower = np.array(white_range[0], dtype=np.uint8)
        self._white_upper = np.array(white_range[1], dtype=np.uint8)
        self._yellow_lower = np.array(yellow_range[0], dtype=np.uint8)
        self._yellow_upper = np.array(yellow_range[1], dtype=np.uint8)

        logger.info("LaneClassifier initialized")

    def classify(
        self,
        image: np.ndarray,
        line_pixels: np.ndarray,
        line_mask: Optional[np.ndarray] = None,
    ) -> LaneClassification:
        """Classify a lane line segment.

        Performs type classification (solid/dashed/double/merging) and
        color classification (white/yellow) on a detected lane line.

        Args:
            image: Original BGR image for color sampling.
            line_pixels: Nx2 array of (x, y) pixel coordinates along the line.
            line_mask: Optional binary mask of the line for spatial analysis.

        Returns:
            LaneClassification with type, color, and confidence scores.
        """
        result = LaneClassification()

        if len(line_pixels) < 5:
            logger.warning("Too few line pixels for classification")
            return result

        # Type classification
        type_result = self._classify_type(line_pixels, line_mask, image)
        result.lane_type = type_result[0]
        result.type_confidence = type_result[1]

        # Dash analysis
        dash_info = self._analyze_dash_pattern(line_pixels, line_mask)
        result.dash_ratio = dash_info["dash_ratio"]
        result.dash_count = dash_info["dash_count"]
        result.avg_dash_length = dash_info["avg_dash_length"]
        result.avg_gap_length = dash_info["avg_gap_length"]

        # Refine type based on dash analysis
        if result.lane_type == LaneType.UNKNOWN and dash_info["dash_count"] > 0:
            if dash_info["dash_ratio"] > self._dash_cfg.get("continuity_threshold", 0.6):
                result.lane_type = LaneType.SOLID
            else:
                result.lane_type = LaneType.DASHED

        # Color classification
        color_result = self._classify_color(image, line_pixels)
        result.lane_color = color_result[0]
        result.color_confidence = color_result[1]

        # Double lane detection
        is_double = self._detect_double_lane(line_pixels, line_mask, image)
        if is_double and result.lane_type in (LaneType.SOLID, LaneType.DASHED):
            if result.lane_type == LaneType.SOLID:
                result.lane_type = LaneType.DOUBLE_SOLID
            else:
                result.lane_type = LaneType.DOUBLE_DASHED

        logger.debug(
            f"Classification: type={result.lane_type.value}, "
            f"color={result.lane_color.value}, "
            f"confidence={result.overall_confidence:.2f}"
        )
        return result

    def _classify_type(
        self,
        line_pixels: np.ndarray,
        line_mask: Optional[np.ndarray],
        image: np.ndarray,
    ) -> Tuple[LaneType, float]:
        """Classify the lane marking type.

        Uses spatial continuity analysis to determine if the line is
        solid, dashed, or has other patterns.

        Args:
            line_pixels: Nx2 array of line pixel coordinates.
            line_mask: Optional binary mask of the line.
            image: Original image for context.

        Returns:
            Tuple of (LaneType, confidence).
        """
        # Analyze continuity along the y-axis (assuming roughly vertical lanes)
        y_coords = line_pixels[:, 1]
        if len(y_coords) < 2:
            return LaneType.UNKNOWN, 0.0

        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        y_range = y_max - y_min

        if y_range < 10:
            return LaneType.UNKNOWN, 0.0

        # Compute continuity: fraction of y-rows that have line pixels
        unique_y = np.unique(y_coords.astype(int))
        continuity = len(unique_y) / y_range

        threshold = self._dash_cfg.get("continuity_threshold", 0.6)
        min_dash = self._dash_cfg.get("min_dash_length", 15)
        max_dash = self._dash_cfg.get("max_dash_length", 80)

        if continuity >= threshold:
            # High continuity -> solid line
            return LaneType.SOLID, min(1.0, continuity)
        else:
            # Low continuity -> dashed line
            confidence = 1.0 - continuity
            return LaneType.DASHED, min(1.0, confidence)

    def _analyze_dash_pattern(
        self,
        line_pixels: np.ndarray,
        line_mask: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Analyze the dash pattern of a lane line.

        Detects individual dashes and gaps by analyzing the continuity
        of line pixels along the line direction.

        Args:
            line_pixels: Nx2 array of line pixel coordinates.
            line_mask: Optional binary mask of the line.

        Returns:
            Dictionary with dash analysis results.
        """
        y_coords = line_pixels[:, 1]
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        y_range = y_max - y_min

        if y_range < 10:
            return {
                "dash_ratio": 1.0,
                "dash_count": 0,
                "avg_dash_length": 0.0,
                "avg_gap_length": 0.0,
            }

        # Build a 1D occupancy signal along y-axis
        unique_y = np.unique(y_coords.astype(int))
        occupancy = np.zeros(y_range + 1, dtype=bool)
        for y in unique_y:
            occupancy[y - y_min] = True

        # Find dash and gap segments using run-length encoding
        dashes: List[int] = []
        gaps: List[int] = []
        current_len = 1
        current_type = occupancy[0]

        for i in range(1, len(occupancy)):
            if occupancy[i] == current_type:
                current_len += 1
            else:
                if current_type:
                    dashes.append(current_len)
                else:
                    gaps.append(current_len)
                current_len = 1
                current_type = occupancy[i]

        # Don't forget the last segment
        if current_type:
            dashes.append(current_len)
        else:
            gaps.append(current_len)

        # Filter dashes/gaps by configured size ranges
        min_dash = self._dash_cfg.get("min_dash_length", 15)
        max_dash = self._dash_cfg.get("max_dash_length", 80)
        min_gap = self._dash_cfg.get("min_gap_length", 15)
        max_gap = self._dash_cfg.get("max_gap_length", 120)

        valid_dashes = [d for d in dashes if min_dash <= d <= max_dash]
        valid_gaps = [g for g in gaps if min_gap <= g <= max_gap]

        total_dash = sum(dashes)
        dash_ratio = total_dash / max(y_range, 1)

        return {
            "dash_ratio": dash_ratio,
            "dash_count": len(valid_dashes),
            "avg_dash_length": np.mean(valid_dashes) if valid_dashes else 0.0,
            "avg_gap_length": np.mean(valid_gaps) if valid_gaps else 0.0,
        }

    def _classify_color(
        self,
        image: np.ndarray,
        line_pixels: np.ndarray,
    ) -> Tuple[LaneColor, float]:
        """Classify the lane marking color.

        Samples colors around the line pixels in HSV space and
        determines the dominant lane color.

        Args:
            image: Original BGR image.
            line_pixels: Nx2 array of line pixel coordinates.

        Returns:
            Tuple of (LaneColor, confidence).
        """
        if len(image.shape) == 2:
            return LaneColor.UNKNOWN, 0.0

        sample_region = self._color_cfg.get("color_sample_region", 5)
        confidence_threshold = self._color_cfg.get("confidence_threshold", 0.6)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Sample colors around line pixels
        white_count = 0
        yellow_count = 0
        total_samples = 0

        # Subsample for efficiency
        step = max(1, len(line_pixels) // 100)
        sampled_pixels = line_pixels[::step]

        for px, py in sampled_pixels:
            x, y = int(px), int(py)

            # Sample a small region around the pixel
            x_start = max(0, x - sample_region)
            x_end = min(w, x + sample_region + 1)
            y_start = max(0, y - sample_region)
            y_end = min(h, y + sample_region + 1)

            region = hsv[y_start:y_end, x_start:x_end]
            if region.size == 0:
                continue

            # Check white pixels
            white_mask = cv2.inRange(region, self._white_lower, self._white_upper)
            white_count += np.count_nonzero(white_mask)

            # Check yellow pixels
            yellow_mask = cv2.inRange(region, self._yellow_lower, self._yellow_upper)
            yellow_count += np.count_nonzero(yellow_mask)

            total_samples += region.shape[0] * region.shape[1]

        if total_samples == 0:
            return LaneColor.UNKNOWN, 0.0

        white_ratio = white_count / total_samples
        yellow_ratio = yellow_count / total_samples

        if white_ratio > yellow_ratio and white_ratio > confidence_threshold * 0.1:
            confidence = min(1.0, white_ratio * 5)
            return LaneColor.WHITE, confidence
        elif yellow_ratio > white_ratio and yellow_ratio > confidence_threshold * 0.1:
            confidence = min(1.0, yellow_ratio * 5)
            return LaneColor.YELLOW, confidence

        return LaneColor.UNKNOWN, 0.0

    def _detect_double_lane(
        self,
        line_pixels: np.ndarray,
        line_mask: Optional[np.ndarray],
        image: np.ndarray,
    ) -> bool:
        """Detect if the lane marking is a double line.

        Checks for the presence of two parallel lines within a
        configured separation range.

        Args:
            line_pixels: Nx2 array of line pixel coordinates.
            line_mask: Optional binary mask of the line.
            image: Original image for edge detection.

        Returns:
            True if a double line is detected.
        """
        if len(image.shape) == 2:
            return False

        min_sep = self._double_cfg.get("min_separation", 5)
        max_sep = self._double_cfg.get("max_separation", 25)
        similarity = self._double_cfg.get("intensity_similarity", 0.7)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the average direction of the line
        if len(line_pixels) < 2:
            return False

        # Compute perpendicular direction
        dx = line_pixels[-1][0] - line_pixels[0][0]
        dy = line_pixels[-1][1] - line_pixels[0][1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            return False

        # Perpendicular direction
        perp_x = -dy / length
        perp_y = dx / length

        # Sample pixels along the perpendicular direction at various offsets
        mid_idx = len(line_pixels) // 2
        mid_x, mid_y = line_pixels[mid_idx]

        intensity_profile = []
        for offset in range(-max_sep, max_sep + 1):
            sx = int(mid_x + perp_x * offset)
            sy = int(mid_y + perp_y * offset)
            if 0 <= sx < gray.shape[1] and 0 <= sy < gray.shape[0]:
                intensity_profile.append(gray[sy, sx])
            else:
                intensity_profile.append(0)

        # Look for two peaks in the intensity profile
        if len(intensity_profile) < 2 * min_sep:
            return False

        # Simple peak detection
        profile = np.array(intensity_profile, dtype=np.float32)
        threshold = np.mean(profile) + np.std(profile)

        peaks = []
        for i in range(1, len(profile) - 1):
            if profile[i] > threshold and profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
                peaks.append(i)

        if len(peaks) >= 2:
            # Check separation between peaks
            separations = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
            for sep in separations:
                if min_sep <= sep <= max_sep:
                    return True

        return False

    def classify_batch(
        self,
        image: np.ndarray,
        lane_segments: List[np.ndarray],
    ) -> List[LaneClassification]:
        """Classify multiple lane segments at once.

        Args:
            image: Original BGR image.
            lane_segments: List of Nx2 pixel arrays for each segment.

        Returns:
            List of LaneClassification results.
        """
        results = []
        for pixels in lane_segments:
            result = self.classify(image, pixels)
            results.append(result)

        logger.debug(f"Batch classified {len(results)} lane segments")
        return results
