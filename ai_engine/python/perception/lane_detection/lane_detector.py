"""
Main Lane Detector Module
=========================

Orchestrates the complete lane detection pipeline combining all
sub-modules: preprocessing, edge detection, perspective transform,
filtering, classification, curvature estimation, tracking, and
departure warning.

Classes:
    DetectionResult - Dataclass for complete detection output
    LaneDetector - Main lane detection pipeline

Typical usage:
    >>> detector = LaneDetector(config_path="config.yaml")
    >>> result = detector.detect(frame)
    >>> result = detector.detect_video(video_path)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from lane_detection.calibration import CameraCalibrator, CalibrationParams
from lane_detection.preprocessing import ImagePreprocessor
from lane_detection.edge_detection import EdgeDetector
from lane_detection.perspective_transform import PerspectiveTransformer
from lane_detection.lane_filter import LaneFilter, FilteredLine, LaneSide
from lane_detection.lane_classifier import (
    LaneClassifier, LaneClassification, LaneType, LaneColor
)
from lane_detection.lane_curvature import CurvatureEstimator, CurvatureResult
from lane_detection.lane_tracker import LaneTracker, TrackedLane
from lane_detection.lane_departure_warning import (
    LaneDepartureWarning, DepartureEvent, WarningLevel
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Complete result from lane detection pipeline.

    Attributes:
        left_lane: Left lane polynomial fit [A, B, C] or None.
        right_lane: Right lane polynomial fit [A, B, C] or None.
        left_lane_pts: Left lane pixel coordinates (Nx2).
        right_lane_pts: Right lane pixel coordinates (Nx2).
        curvature: Curvature estimation result.
        left_classification: Left lane classification.
        right_classification: Right lane classification.
        tracked_lanes: List of tracked lanes.
        departure_event: Lane departure warning event.
        left_filtered: Filtered left lane lines.
        right_filtered: Filtered right lane lines.
        edge_image: Edge detection output.
        bev_image: Bird's eye view image.
        processing_time_ms: Processing time in milliseconds.
        frame_index: Frame index in video sequence.
        is_valid: Whether detection is valid.
    """
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    left_lane_pts: Optional[np.ndarray] = None
    right_lane_pts: Optional[np.ndarray] = None
    curvature: Optional[CurvatureResult] = None
    left_classification: Optional[LaneClassification] = None
    right_classification: Optional[LaneClassification] = None
    tracked_lanes: List[TrackedLane] = field(default_factory=list)
    departure_event: Optional[DepartureEvent] = None
    left_filtered: List[FilteredLine] = field(default_factory=list)
    right_filtered: List[FilteredLine] = field(default_factory=list)
    edge_image: Optional[np.ndarray] = None
    bev_image: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    frame_index: int = 0
    is_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection result."""
        result = {
            "has_left_lane": self.left_lane is not None,
            "has_right_lane": self.right_lane is not None,
            "curvature": self.curvature.to_dict() if self.curvature else None,
            "left_classification": (
                self.left_classification.to_dict()
                if self.left_classification else None
            ),
            "right_classification": (
                self.right_classification.to_dict()
                if self.right_classification else None
            ),
            "departure_level": (
                self.departure_event.level.value
                if self.departure_event else "none"
            ),
            "tracked_count": len(self.tracked_lanes),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "frame_index": self.frame_index,
            "is_valid": self.is_valid,
        }
        if self.curvature:
            result["radius_of_curvature"] = round(
                self.curvature.radius_of_curvature, 2
            )
            result["center_offset"] = round(self.curvature.center_offset, 4)
        return result


class LaneDetector:
    """Main lane detection pipeline orchestrator.

    Combines all lane detection sub-modules into a coherent pipeline
    that processes individual frames or video streams.

    Pipeline Order:
        1. Preprocessing (undistort, color convert, equalize, ROI)
        2. Edge Detection (Canny/Sobel/combined)
        3. Perspective Transform (bird's eye view)
        4. Lane Filtering (Hough lines + geometric constraints)
        5. Curvature Estimation (sliding window + polynomial fit)
        6. Lane Classification (type + color)
        7. Lane Tracking (Kalman filter + ID management)
        8. Departure Warning (offset + TLC)

    Attributes:
        config: Full configuration dictionary.
        preprocessor: Image preprocessor instance.
        edge_detector: Edge detector instance.
        perspective: Perspective transformer instance.
        lane_filter: Lane filter instance.
        classifier: Lane classifier instance.
        curvature: Curvature estimator instance.
        tracker: Lane tracker instance.
        ldw: Lane departure warning instance.
        calibrator: Camera calibrator instance.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the lane detection pipeline.

        Args:
            config_path: Path to YAML configuration file.
            config: Configuration dictionary (alternative to config_path).

        Raises:
            ValueError: If neither config_path nor config is provided.
        """
        # Load configuration
        if config_path is not None:
            self.config = self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")

        # Initialize calibration
        self.calibrator = CameraCalibrator()
        try:
            self.calibrator.calibrate_from_config(self.config)
        except Exception as e:
            logger.warning(f"Calibration from config failed: {e}")

        calibration_params = None
        if self.calibrator.params.calibrated:
            calibration_params = self.calibrator.params.to_dict()

        # Initialize pipeline components
        self.preprocessor = ImagePreprocessor(
            self.config.get("preprocessing", {}),
            calibration_params=calibration_params,
        )
        self.edge_detector = EdgeDetector(
            self.config.get("edge_detection", {})
        )
        self.perspective = PerspectiveTransformer(
            self.config.get("perspective_transform", {})
        )
        self.lane_filter = LaneFilter(
            self.config.get("lane_filter", {})
        )
        self.classifier = LaneClassifier(
            self.config.get("lane_classifier", {})
        )
        self.curvature = CurvatureEstimator(
            self.config.get("lane_curvature", {})
        )
        self.tracker = LaneTracker(
            self.config.get("lane_tracker", {})
        )
        self.ldw = LaneDepartureWarning(
            self.config.get("lane_departure_warning", {})
        )

        # State
        self._frame_index = 0
        self._prev_curvature_result: Optional[CurvatureResult] = None

        logger.info("LaneDetector pipeline initialized")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run the full lane detection pipeline on a single frame.

        Args:
            frame: Input BGR image from front camera.

        Returns:
            DetectionResult with all pipeline outputs.
        """
        start_time = time.time()
        result = DetectionResult(frame_index=self._frame_index)

        try:
            # Stage 1: Preprocessing
            processed = self.preprocessor.process(frame)

            # Stage 2: Edge Detection
            edges = self.edge_detector.detect(processed)
            result.edge_image = edges

            # Stage 3: Perspective Transform
            bev_edges = self.perspective.to_bev(edges)
            result.bev_image = bev_edges

            # Stage 4: Lane Filtering (on original edge image)
            h, w = frame.shape[:2]
            left_lines, right_lines = self.lane_filter.filter(
                edges, image_width=w
            )
            result.left_filtered = left_lines
            result.right_filtered = right_lines

            # Stage 5: Curvature Estimation (on BEV edges)
            curvature_result = self.curvature.estimate(bev_edges)

            # Try using prior fit if available and current fit fails
            if not curvature_result.is_valid and self._prev_curvature_result is not None:
                prev = self._prev_curvature_result
                if prev.left_fit is not None and prev.right_fit is not None:
                    curvature_result = self.curvature.estimate_from_prior(
                        bev_edges, prev.left_fit, prev.right_fit
                    )

            result.curvature = curvature_result
            result.left_lane = curvature_result.left_fit
            result.right_lane = curvature_result.right_fit
            result.left_lane_pts = curvature_result.left_lane_pts
            result.right_lane_pts = curvature_result.right_lane_pts

            if curvature_result.is_valid:
                self._prev_curvature_result = curvature_result

            # Stage 6: Lane Classification
            if result.left_lane_pts is not None and len(result.left_lane_pts) > 5:
                result.left_classification = self.classifier.classify(
                    frame, result.left_lane_pts.astype(np.int32)
                )
            if result.right_lane_pts is not None and len(result.right_lane_pts) > 5:
                result.right_classification = self.classifier.classify(
                    frame, result.right_lane_pts.astype(np.int32)
                )

            # Stage 7: Lane Tracking
            left_conf = (
                result.left_classification.type_confidence
                if result.left_classification else 0.5
            )
            right_conf = (
                result.right_classification.type_confidence
                if result.right_classification else 0.5
            )

            result.tracked_lanes = self.tracker.update(
                left_fit=curvature_result.left_fit,
                right_fit=curvature_result.right_fit,
                left_confidence=curvature_result.confidence if curvature_result.left_fit else 0.0,
                right_confidence=curvature_result.confidence if curvature_result.right_fit else 0.0,
                frame_idx=self._frame_index,
                image_shape=frame.shape,
            )

            # Stage 8: Lane Departure Warning
            if curvature_result.is_valid:
                speed = self.config.get("camera", {}).get("fps", 30) * 0.3  # Rough estimate
                result.departure_event = self.ldw.check_departure(
                    center_offset=curvature_result.center_offset,
                    lateral_velocity=0.0,
                    speed=speed,
                    curvature=curvature_result.radius_of_curvature,
                    confidence=curvature_result.confidence,
                )

                # Trigger warning if needed
                if result.departure_event.level != WarningLevel.NONE:
                    self.ldw.trigger_warning(result.departure_event)

            # Determine overall validity
            result.is_valid = (
                curvature_result.is_valid
                or (result.left_lane is not None and result.right_lane is not None)
            )

        except Exception as e:
            logger.error(f"Detection pipeline error: {e}", exc_info=True)
            result.is_valid = False

        # Record timing
        result.processing_time_ms = (time.time() - start_time) * 1000.0
        self._frame_index += 1

        logger.debug(
            f"Frame {result.frame_index}: "
            f"valid={result.is_valid}, "
            f"R={curvature_result.radius_of_curvature:.0f}m, "
            f"t={result.processing_time_ms:.1f}ms"
        )
        return result

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
        callback: Optional[Any] = None,
    ) -> List[DetectionResult]:
        """Process a video file through the lane detection pipeline.

        Args:
            video_path: Path to the input video file.
            output_path: Optional path for annotated output video.
            display: Whether to display frames in a window.
            max_frames: Maximum number of frames to process.
            callback: Optional callback function(DetectionResult, frame)
                called for each frame.

        Returns:
            List of DetectionResult for each processed frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Processing video: {video_path}, "
            f"{width}x{height} @ {fps}fps, "
            f"{total_frames} frames"
        )

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results: List[DetectionResult] = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                result = self.detect(frame)
                results.append(result)

                if callback:
                    callback(result, frame)

                if writer and result.is_valid:
                    annotated = self._annotate_frame(frame, result)
                    writer.write(annotated)

                if display:
                    annotated = self._annotate_frame(frame, result)
                    cv2.imshow("Lane Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        logger.info(f"Processed {frame_count} frames from {video_path}")
        return results

    def _annotate_frame(
        self, frame: np.ndarray, result: DetectionResult
    ) -> np.ndarray:
        """Draw basic lane annotations on a frame.

        Args:
            frame: Original BGR frame.
            result: Detection result to visualize.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        # Draw filtered lines
        for line in result.left_filtered:
            pt1 = (int(line.start[0]), int(line.start[1]))
            pt2 = (int(line.end[0]), int(line.end[1]))
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        for line in result.right_filtered:
            pt1 = (int(line.start[0]), int(line.start[1]))
            pt2 = (int(line.end[0]), int(line.end[1]))
            cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)

        # Draw polynomial fits
        if result.left_lane_pts is not None:
            pts = result.left_lane_pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], False, (0, 255, 0), 3)

        if result.right_lane_pts is not None:
            pts = result.right_lane_pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], False, (0, 255, 255), 3)

        # Draw departure warning
        if result.departure_event and result.departure_event.level != WarningLevel.NONE:
            if result.departure_event.level == WarningLevel.CRITICAL:
                cv2.putText(
                    annotated, "LANE DEPARTURE WARNING!",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3
                )
            elif result.departure_event.level == WarningLevel.WARNING:
                cv2.putText(
                    annotated, "Approaching Lane Boundary",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 165, 255), 2
                )

        # Draw curvature info
        if result.curvature and result.curvature.is_valid:
            cv2.putText(
                annotated,
                f"R = {result.curvature.radius_of_curvature:.0f}m",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )
            cv2.putText(
                annotated,
                f"Offset = {result.curvature.center_offset:.2f}m",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

        return annotated

    def reset(self) -> None:
        """Reset all pipeline state for a new sequence."""
        self._frame_index = 0
        self._prev_curvature_result = None
        self.curvature.reset()
        self.tracker.reset()
        self.ldw.reset()
        self.preprocessor.reset()
        logger.info("LaneDetector pipeline reset")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration.

        Returns:
            Dictionary with pipeline component configurations.
        """
        return {
            "preprocessing": list(self.preprocessor.config.keys()),
            "edge_method": self.edge_detector.method,
            "perspective_src": self.perspective.src_points.tolist(),
            "curvature_ym_per_pix": self.curvature.ym_per_pix,
            "curvature_xm_per_pix": self.curvature.xm_per_pix,
            "tracking_lanes": len(self.tracker.tracked_lanes),
            "ldw_active": self.ldw.active,
            "frame_index": self._frame_index,
        }
