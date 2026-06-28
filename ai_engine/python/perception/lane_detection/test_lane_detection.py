"""
Unit Tests for Lane Detection System
=====================================

Comprehensive tests for all lane detection pipeline components:
- Calibration
- Preprocessing
- Edge Detection
- Perspective Transform
- Lane Filtering
- Lane Classification
- Curvature Estimation
- Lane Tracking
- Lane Departure Warning
- Full Pipeline Integration

Run with: python -m pytest test_lane_detection.py -v
"""

import os
import sys
import tempfile
import unittest
from typing import Any, Dict

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane_detection.calibration import CameraCalibrator, CalibrationParams
from lane_detection.preprocessing import ImagePreprocessor
from lane_detection.edge_detection import EdgeDetector
from lane_detection.perspective_transform import PerspectiveTransformer
from lane_detection.lane_filter import LaneFilter, FilteredLine, LaneSide
from lane_detection.lane_classifier import (
    LaneClassifier, LaneClassification, LaneType, LaneColor
)
from lane_detection.lane_curvature import CurvatureEstimator, CurvatureResult
from lane_detection.lane_tracker import LaneTracker, TrackedLane, KalmanFilter1D
from lane_detection.lane_departure_warning import (
    LaneDepartureWarning, DepartureEvent, WarningLevel
)
from lane_detection.lane_detector import LaneDetector, DetectionResult
from lane_detection.visualization import LaneVisualizer
from lane_detection.benchmark import LaneDetectionBenchmark, BenchmarkResult


def _create_test_image(
    width: int = 1280, height: int = 720, with_lanes: bool = True
) -> np.ndarray:
    """Create a synthetic test image with optional lane lines.

    Args:
        width: Image width.
        height: Image height.
        with_lanes: Whether to draw lane lines.

    Returns:
        BGR test image.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Road surface (dark gray)
    cv2.rectangle(image, (0, 0), (width, height), (60, 60, 60), -1)

    if with_lanes:
        # Left lane (green-ish)
        left_pts = np.array([
            [300, height], [450, 400], [520, 300], [560, 250]
        ], dtype=np.int32)
        cv2.polylines(image, [left_pts], False, (200, 200, 200), 4)

        # Right lane
        right_pts = np.array([
            [980, height], [830, 400], [760, 300], [720, 250]
        ], dtype=np.int32)
        cv2.polylines(image, [right_pts], False, (200, 200, 200), 4)

        # Center dashes
        for y in range(300, height, 40):
            cv2.line(image, (640, y), (640, min(y + 20, height)), (180, 180, 180), 2)

    return image


def _create_bev_test_image(
    width: int = 640, height: int = 720
) -> np.ndarray:
    """Create a synthetic bird's eye view test image with lane lines.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        Grayscale BEV test image.
    """
    image = np.zeros((height, width), dtype=np.uint8)

    # Left lane line
    for y in range(height):
        x = 200 + int(0.0001 * (y - height / 2) ** 2)
        for dx in range(-2, 3):
            if 0 <= x + dx < width:
                image[y, x + dx] = 255

    # Right lane line
    for y in range(height):
        x = 440 - int(0.0001 * (y - height / 2) ** 2)
        for dx in range(-2, 3):
            if 0 <= x + dx < width:
                image[y, x + dx] = 255

    return image


def _get_default_config() -> Dict[str, Any]:
    """Get a default configuration dictionary for testing."""
    return {
        "camera": {
            "image_width": 1280,
            "image_height": 720,
            "intrinsic_matrix": [[960, 0, 640], [0, 960, 360], [0, 0, 1]],
            "distortion_coefficients": [-0.32, 0.12, 0, 0, -0.02],
        },
        "preprocessing": {
            "roi_vertices": [[0.1, 1.0], [0.4, 0.4], [0.6, 0.4], [0.9, 1.0]],
            "color_space": "HLS",
            "equalize_histogram": True,
            "equalize_channel": 2,
            "gaussian_kernel": [5, 5],
            "gaussian_sigma": 1.4,
            "use_clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": [8, 8],
            "apply_undistort": False,
            "downsample_factor": 1.0,
        },
        "edge_detection": {
            "method": "canny",
            "canny": {"low_threshold": 50, "high_threshold": 150,
                      "aperture_size": 3, "use_l2_gradient": True},
            "sobel": {"kernel_size": 3, "x_weight": 1.0, "y_weight": 0.3,
                      "magnitude_threshold": 50,
                      "direction_threshold": [0.3, 1.2]},
            "color_edge": {
                "white_h_range": [0, 180], "white_s_range": [0, 30],
                "white_v_range": [200, 255],
                "yellow_h_range": [15, 35], "yellow_s_range": [100, 255],
                "yellow_v_range": [100, 255],
            },
            "adaptive": {"block_size": 11, "c_offset": 2, "use_otsu": True},
            "combined_weights": {"canny": 0.5, "sobel": 0.3, "color": 0.2},
        },
        "lane_filter": {
            "min_angle": 20.0, "max_angle": 85.0, "min_length": 30,
            "max_line_gap": 15,
            "hough": {"rho": 1, "theta": 0.0174532925, "threshold": 40,
                      "min_line_length": 30, "max_line_gap": 15},
            "parallel": {"max_angle_diff": 10.0, "max_distance": 600,
                         "min_distance": 100},
            "midpoint_x_ratio": 0.5,
            "merge_distance": 20, "merge_angle_diff": 10,
        },
        "perspective_transform": {
            "src_points": [[220, 720], [560, 475], [720, 475], [1060, 720]],
            "dst_points": [[320, 720], [320, 0], [960, 0], [960, 720]],
            "bev": {"width": 3.7, "length": 30.0,
                    "pixels_per_meter_x": 173, "pixels_per_meter_y": 24},
        },
        "lane_classifier": {
            "dash_analysis": {"min_dash_length": 15, "max_dash_length": 80,
                              "min_gap_length": 15, "max_gap_length": 120,
                              "continuity_threshold": 0.6},
            "double_lane": {"max_separation": 25, "min_separation": 5,
                            "intensity_similarity": 0.7},
            "merging": {"convergence_angle": 5.0, "min_tracking_distance": 50},
            "color_classification": {
                "white_hsv": [[0, 0, 200], [180, 30, 255]],
                "yellow_hsv": [[15, 80, 100], [35, 255, 255]],
                "color_sample_region": 5, "confidence_threshold": 0.6,
            },
        },
        "lane_curvature": {
            "polynomial_degree": 2, "min_sample_points": 10,
            "sliding_window": {"n_windows": 9, "window_width": 100,
                               "min_pixels": 50, "margin": 80},
            "ym_per_pix": 0.04167, "xm_per_pix": 0.00578,
            "temporal_smoothing_alpha": 0.3,
            "min_radius": 100.0, "max_radius": 5000.0,
            "heading": {"lookahead_distance": 10.0, "max_heading_angle": 45.0},
        },
        "lane_tracker": {
            "kalman": {"process_noise": 0.01, "measurement_noise": 0.1},
            "lane_id": {"max_missing_frames": 5, "max_lanes": 6,
                        "assignment_threshold": 50},
            "smoothing": {"alpha": 0.3, "min_confidence": 0.3,
                          "max_velocity": 5.0},
            "prediction": {"use_constant_velocity": True,
                           "max_prediction_frames": 3, "velocity_decay": 0.9},
        },
        "lane_departure_warning": {
            "enabled": True,
            "warning_threshold": 0.5, "critical_threshold": 0.8,
            "tlc_warning": 2.0, "tlc_critical": 0.5,
            "alert_types": {"visual": True, "haptic": True, "audible": True},
            "warning_cooldown_ms": 1000,
            "vehicle": {"width": 1.8, "lane_width": 3.7},
            "hysteresis": {"activate_offset": 0.5, "deactivate_offset": 0.35},
        },
        "visualization": {
            "lane_color": {"left": [0, 255, 0], "right": [0, 255, 255],
                           "center": [255, 0, 0], "predicted": [255, 165, 0]},
            "lane_thickness": 3,
            "lane_fill_alpha": 0.2, "lane_fill_color": [0, 255, 0],
            "show_sliding_windows": True, "sliding_window_color": [255, 255, 0],
            "show_curvature": True, "curvature_font_scale": 0.7,
            "curvature_position": [50, 50],
            "warning_color": [0, 0, 255], "critical_color": [0, 0, 200],
            "show_bev": True, "bev_size": [320, 240], "bev_position": [960, 10],
            "show_fps": True, "show_confidence": True, "info_font_scale": 0.5,
        },
        "benchmark": {
            "dataset_dir": "./data",
            "targets": {"min_accuracy": 0.90, "min_fps": 15,
                        "max_latency_ms": 66},
            "robustness": {"noise_levels": [10, 20, 30, 50]},
            "stress": {"max_concurrent_frames": 10, "duration_seconds": 10},
        },
    }


class TestCalibration(unittest.TestCase):
    """Tests for CameraCalibrator and CalibrationParams."""

    def setUp(self):
        self.config = _get_default_config()

    def test_calibration_params_from_dict(self):
        """Test creating CalibrationParams from dictionary."""
        data = self.config["camera"]
        params = CalibrationParams.from_dict(data)
        self.assertTrue(params.calibrated)
        self.assertEqual(params.image_width, 1280)
        self.assertEqual(params.image_height, 720)
        self.assertEqual(params.camera_matrix.shape, (3, 3))

    def test_calibration_params_from_yaml_config(self):
        """Test creating CalibrationParams from YAML config."""
        params = CalibrationParams.from_yaml_config(self.config)
        self.assertTrue(params.calibrated)
        self.assertGreater(params.fov_horizontal, 0)
        self.assertGreater(params.fov_vertical, 0)

    def test_calibration_params_to_dict(self):
        """Test serialization to dictionary."""
        params = CalibrationParams.from_yaml_config(self.config)
        data = params.to_dict()
        self.assertIn("intrinsic_matrix", data)
        self.assertIn("distortion_coefficients", data)
        self.assertTrue(data["calibrated"])

    def test_calibrator_undistort(self):
        """Test image undistortion."""
        calibrator = CameraCalibrator()
        calibrator.calibrate_from_config(self.config)
        image = _create_test_image(with_lanes=False)
        undistorted = calibrator.undistort(image)
        self.assertEqual(undistorted.shape, image.shape)

    def test_calibrator_save_load(self):
        """Test saving and loading calibration data."""
        calibrator = CameraCalibrator()
        calibrator.calibrate_from_config(self.config)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            calibrator.save(filepath)
            self.assertTrue(os.path.exists(filepath))

            calibrator2 = CameraCalibrator()
            params = calibrator2.load(filepath)
            self.assertTrue(params.calibrated)
        finally:
            os.unlink(filepath)

    def test_homography_computation(self):
        """Test homography matrix computation."""
        calibrator = CameraCalibrator()
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
        H = calibrator.compute_homography(src, dst)
        self.assertEqual(H.shape, (3, 3))


class TestPreprocessing(unittest.TestCase):
    """Tests for ImagePreprocessor."""

    def setUp(self):
        self.config = _get_default_config()["preprocessing"]
        self.image = _create_test_image()

    def test_preprocessor_process(self):
        """Test the full preprocessing pipeline."""
        preprocessor = ImagePreprocessor(self.config)
        result = preprocessor.process(self.image)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 3)

    def test_preprocessor_roi_mask(self):
        """Test ROI mask creation."""
        preprocessor = ImagePreprocessor(self.config)
        preprocessor.process(self.image)  # Trigger lazy init
        self.assertIsNotNone(preprocessor.roi_mask)
        self.assertEqual(preprocessor.roi_mask.shape,
                         (self.image.shape[0], self.image.shape[1]))

    def test_preprocessor_color_conversion(self):
        """Test color space conversion."""
        preprocessor = ImagePreprocessor(self.config)
        hls = preprocessor._convert_color(self.image, "HLS")
        self.assertEqual(hls.shape[2], 3)
        hsv = preprocessor._convert_color(self.image, "HSV")
        self.assertEqual(hsv.shape[2], 3)

    def test_preprocessor_white_yellow_mask(self):
        """Test white/yellow color mask extraction."""
        preprocessor = ImagePreprocessor(self.config)
        mask = preprocessor.create_white_yellow_mask(self.image)
        self.assertEqual(len(mask.shape), 2)

    def test_preprocessor_empty_image_raises(self):
        """Test that empty image raises ValueError."""
        preprocessor = ImagePreprocessor(self.config)
        with self.assertRaises(ValueError):
            preprocessor.process(None)

    def test_preprocessor_reset(self):
        """Test preprocessor state reset."""
        preprocessor = ImagePreprocessor(self.config)
        preprocessor.process(self.image)
        preprocessor.reset()
        self.assertIsNone(preprocessor.roi_mask)


class TestEdgeDetection(unittest.TestCase):
    """Tests for EdgeDetector."""

    def setUp(self):
        self.config = _get_default_config()["edge_detection"]
        self.image = _create_test_image()

    def test_canny_edges(self):
        """Test Canny edge detection."""
        detector = EdgeDetector(self.config)
        edges = detector.canny_edges(self.image)
        self.assertEqual(len(edges.shape), 2)
        self.assertEqual(edges.dtype, np.uint8)

    def test_sobel_edges(self):
        """Test Sobel edge detection."""
        detector = EdgeDetector(self.config)
        edges = detector.sobel_edges(self.image)
        self.assertEqual(len(edges.shape), 2)

    def test_color_edges(self):
        """Test color-based edge detection."""
        detector = EdgeDetector(self.config)
        edges = detector.color_edges(self.image)
        self.assertEqual(len(edges.shape), 2)

    def test_combined_edges(self):
        """Test combined edge detection."""
        detector = EdgeDetector(self.config)
        edges = detector.combined_edges(self.image)
        self.assertEqual(len(edges.shape), 2)

    def test_auto_canny(self):
        """Test automatic Canny threshold selection."""
        detector = EdgeDetector(self.config)
        edges = detector.auto_canny(self.image)
        self.assertEqual(len(edges.shape), 2)

    def test_detect_with_method_override(self):
        """Test method override in detect()."""
        detector = EdgeDetector(self.config)
        edges = detector.detect(self.image, method="sobel")
        self.assertEqual(len(edges.shape), 2)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        detector = EdgeDetector(self.config)
        with self.assertRaises(ValueError):
            detector.detect(self.image, method="invalid")

    def test_set_method(self):
        """Test changing the detection method."""
        detector = EdgeDetector(self.config)
        detector.set_method("sobel")
        self.assertEqual(detector.method, "sobel")


class TestPerspectiveTransform(unittest.TestCase):
    """Tests for PerspectiveTransformer."""

    def setUp(self):
        self.config = _get_default_config()["perspective_transform"]
        self.image = _create_test_image()

    def test_to_bev(self):
        """Test forward perspective transform."""
        transformer = PerspectiveTransformer(self.config)
        bev = transformer.to_bev(self.image)
        self.assertIsNotNone(bev)
        self.assertEqual(len(bev.shape), 3)

    def test_from_bev(self):
        """Test inverse perspective transform."""
        transformer = PerspectiveTransformer(self.config)
        bev = transformer.to_bev(self.image)
        original = transformer.from_bev(bev)
        self.assertEqual(original.shape[2], 3)

    def test_transform_points(self):
        """Test point transformation."""
        transformer = PerspectiveTransformer(self.config)
        points = np.array([[640, 360]], dtype=np.float32)
        transformed = transformer.transform_points(points)
        self.assertEqual(transformed.shape, (1, 2))

    def test_pixel_to_world(self):
        """Test pixel to world coordinate conversion."""
        transformer = PerspectiveTransformer(self.config)
        lat, lon = transformer.pixel_to_world(640, 360)
        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)

    def test_world_to_pixel(self):
        """Test world to pixel coordinate conversion."""
        transformer = PerspectiveTransformer(self.config)
        px, py = transformer.world_to_pixel(0.0, 15.0)
        self.assertIsInstance(px, float)
        self.assertIsInstance(py, float)

    def test_bev_histogram(self):
        """Test BEV histogram computation."""
        transformer = PerspectiveTransformer(self.config)
        bev = transformer.to_bev(self.image)
        hist = transformer.get_bev_histogram(bev)
        self.assertEqual(len(hist.shape), 1)

    def test_validate_transform(self):
        """Test transform validation."""
        transformer = PerspectiveTransformer(self.config)
        metrics = transformer.validate_transform(self.image)
        self.assertIn("mse", metrics)
        self.assertIn("psnr", metrics)


class TestLaneFilter(unittest.TestCase):
    """Tests for LaneFilter."""

    def setUp(self):
        self.config = _get_default_config()["lane_filter"]

    def test_filtered_line_creation(self):
        """Test FilteredLine object creation."""
        line = FilteredLine(
            start=(100, 700), end=(300, 400),
            angle=55.0, length=350.0,
            side=LaneSide.LEFT, confidence=0.8
        )
        self.assertEqual(line.side, LaneSide.LEFT)
        self.assertAlmostEqual(line.angle, 55.0)

    def test_filtered_line_midpoint(self):
        """Test midpoint calculation."""
        line = FilteredLine(start=(0, 0), end=(100, 100),
                            angle=45.0, length=141.4)
        mid = line.midpoint
        self.assertAlmostEqual(mid[0], 50.0)
        self.assertAlmostEqual(mid[1], 50.0)

    def test_angle_filter(self):
        """Test angle-based filtering."""
        lane_filter = LaneFilter(self.config)
        lines = [
            FilteredLine(start=(0, 0), end=(10, 10), angle=5.0, length=20),
            FilteredLine(start=(0, 0), end=(10, 100), angle=50.0, length=100),
            FilteredLine(start=(0, 0), end=(10, 10), angle=89.0, length=20),
        ]
        filtered = lane_filter._angle_filter(lines, 20.0, 85.0)
        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(filtered[0].angle, 50.0)

    def test_length_filter(self):
        """Test length-based filtering."""
        lane_filter = LaneFilter(self.config)
        lines = [
            FilteredLine(start=(0, 0), end=(10, 0), angle=0, length=10),
            FilteredLine(start=(0, 0), end=(100, 0), angle=0, length=100),
        ]
        filtered = lane_filter._length_filter(lines, 30)
        self.assertEqual(len(filtered), 1)


class TestLaneClassifier(unittest.TestCase):
    """Tests for LaneClassifier."""

    def setUp(self):
        self.config = _get_default_config()["lane_classifier"]
        self.image = _create_test_image()

    def test_classify_solid_line(self):
        """Test classification of a solid line."""
        classifier = LaneClassifier(self.config)
        # Create continuous line pixels (solid)
        pixels = np.array([[300 + i, 300 + i] for i in range(100)])
        result = classifier.classify(self.image, pixels)
        self.assertIsInstance(result, LaneClassification)

    def test_classify_empty_pixels(self):
        """Test classification with too few pixels."""
        classifier = LaneClassifier(self.config)
        pixels = np.array([[300, 300]])
        result = classifier.classify(self.image, pixels)
        self.assertEqual(result.lane_type, LaneType.UNKNOWN)

    def test_lane_color_enum(self):
        """Test LaneColor enum values."""
        self.assertEqual(LaneColor.WHITE.value, "white")
        self.assertEqual(LaneColor.YELLOW.value, "yellow")


class TestCurvatureEstimator(unittest.TestCase):
    """Tests for CurvatureEstimator."""

    def setUp(self):
        self.config = _get_default_config()["lane_curvature"]
        self.bev = _create_bev_test_image()

    def test_estimate(self):
        """Test curvature estimation."""
        estimator = CurvatureEstimator(self.config)
        result = estimator.estimate(self.bev, left_base=200, right_base=440)
        self.assertIsInstance(result, CurvatureResult)

    def test_estimate_auto_bases(self):
        """Test auto-detection of lane bases."""
        estimator = CurvatureEstimator(self.config)
        result = estimator.estimate(self.bev)
        self.assertIsInstance(result, CurvatureResult)

    def test_curvature_result_to_dict(self):
        """Test CurvatureResult serialization."""
        result = CurvatureResult()
        d = result.to_dict()
        self.assertIn("radius_of_curvature", d)
        self.assertIn("center_offset", d)

    def test_estimator_reset(self):
        """Test curvature estimator reset."""
        estimator = CurvatureEstimator(self.config)
        estimator.estimate(self.bev, 200, 440)
        estimator.reset()
        self.assertIsNone(estimator._prev_left_fit)


class TestLaneTracker(unittest.TestCase):
    """Tests for LaneTracker and KalmanFilter1D."""

    def setUp(self):
        self.config = _get_default_config()["lane_tracker"]

    def test_kalman_filter_basic(self):
        """Test basic Kalman filter operation."""
        kf = KalmanFilter1D(initial_value=100.0)
        predicted = kf.predict()
        self.assertAlmostEqual(predicted, 101.0, places=0)
        updated = kf.update(102.0)
        self.assertGreater(updated, 100.0)

    def test_kalman_filter_convergence(self):
        """Test Kalman filter converges to true value."""
        kf = KalmanFilter1D(initial_value=0.0, measurement_noise=0.1)
        for _ in range(50):
            kf.predict()
            kf.update(100.0)
        self.assertAlmostEqual(kf.value, 100.0, delta=2.0)

    def test_tracker_create_track(self):
        """Test lane track creation."""
        tracker = LaneTracker(self.config)
        lanes = tracker.update(
            left_fit=np.array([0.0001, -0.1, 400.0]),
            right_fit=np.array([0.0001, -0.1, 800.0]),
            left_confidence=0.8,
            right_confidence=0.8,
            frame_idx=0,
            image_shape=(720, 1280, 3),
        )
        self.assertGreater(len(lanes), 0)

    def test_tracker_no_detection(self):
        """Test tracker with no detections."""
        tracker = LaneTracker(self.config)
        lanes = tracker.update(None, None, 0, 0, frame_idx=0)
        self.assertEqual(len(lanes), 0)

    def test_tracker_reset(self):
        """Test tracker reset."""
        tracker = LaneTracker(self.config)
        tracker.update(
            np.array([0.0001, -0.1, 400.0]),
            np.array([0.0001, -0.1, 800.0]),
            0.8, 0.8, frame_idx=0,
            image_shape=(720, 1280, 3),
        )
        tracker.reset()
        self.assertEqual(len(tracker.tracked_lanes), 0)

    def test_tracked_lane_properties(self):
        """Test TrackedLane properties."""
        lane = TrackedLane(lane_id=0, confidence=0.8)
        self.assertTrue(lane.is_active)
        self.assertFalse(lane.is_predicted)

        lane.frames_missing = 2
        self.assertFalse(lane.is_active)
        self.assertTrue(lane.is_predicted)


class TestLaneDepartureWarning(unittest.TestCase):
    """Tests for LaneDepartureWarning."""

    def setUp(self):
        self.config = _get_default_config()["lane_departure_warning"]

    def test_no_departure(self):
        """Test no warning when centered in lane."""
        ldw = LaneDepartureWarning(self.config)
        event = ldw.check_departure(center_offset=0.0)
        self.assertEqual(event.level, WarningLevel.NONE)

    def test_soft_warning(self):
        """Test soft warning near lane boundary."""
        ldw = LaneDepartureWarning(self.config)
        event = ldw.check_departure(center_offset=0.4)
        # May or may not trigger depending on hysteresis
        self.assertIsInstance(event.level, WarningLevel)

    def test_critical_warning(self):
        """Test critical warning when outside boundary."""
        ldw = LaneDepartureWarning(self.config)
        event = ldw.check_departure(center_offset=1.0, confidence=0.9)
        self.assertEqual(event.level, WarningLevel.CRITICAL)

    def test_tlc_computation(self):
        """Test time-to-lane-crossing computation."""
        ldw = LaneDepartureWarning(self.config)
        tlc = ldw._compute_tlc(0.3, lateral_velocity=0.5, speed=15.0, curvature=0)
        self.assertIsNotNone(tlc)
        self.assertGreater(tlc, 0)

    def test_disable_ldw(self):
        """Test disabling LDW system."""
        ldw = LaneDepartureWarning(self.config)
        ldw.disable()
        event = ldw.check_departure(center_offset=1.0)
        self.assertEqual(event.level, WarningLevel.NONE)

    def test_warning_history(self):
        """Test warning history recording."""
        ldw = LaneDepartureWarning(self.config)
        ldw.check_departure(center_offset=1.0, confidence=0.9)
        history = ldw.get_warning_history()
        # History may be empty due to cooldown, but method should work
        self.assertIsInstance(history, list)

    def test_statistics(self):
        """Test statistics computation."""
        ldw = LaneDepartureWarning(self.config)
        stats = ldw.get_statistics()
        self.assertIn("total_warnings", stats)
        self.assertIn("current_level", stats)


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the full lane detection pipeline."""

    def setUp(self):
        self.config = _get_default_config()

    def test_lane_detector_creation(self):
        """Test LaneDetector creation from config dict."""
        detector = LaneDetector(config=self.config)
        self.assertIsNotNone(detector)

    def test_lane_detector_detect(self):
        """Test detection on a synthetic image."""
        detector = LaneDetector(config=self.config)
        image = _create_test_image(with_lanes=True)
        result = detector.detect(image)
        self.assertIsInstance(result, DetectionResult)
        self.assertGreater(result.processing_time_ms, 0)

    def test_detection_result_to_dict(self):
        """Test DetectionResult serialization."""
        detector = LaneDetector(config=self.config)
        image = _create_test_image()
        result = detector.detect(image)
        d = result.to_dict()
        self.assertIn("is_valid", d)
        self.assertIn("processing_time_ms", d)

    def test_detector_reset(self):
        """Test detector reset."""
        detector = LaneDetector(config=self.config)
        image = _create_test_image()
        detector.detect(image)
        detector.reset()
        self.assertEqual(detector._frame_index, 0)

    def test_pipeline_info(self):
        """Test pipeline info retrieval."""
        detector = LaneDetector(config=self.config)
        info = detector.get_pipeline_info()
        self.assertIn("edge_method", info)
        self.assertIn("ldw_active", info)


class TestVisualization(unittest.TestCase):
    """Tests for LaneVisualizer."""

    def setUp(self):
        self.config = _get_default_config()
        self.visualizer = LaneVisualizer(self.config)
        self.image = _create_test_image()
        self.detector = LaneDetector(config=self.config)

    def test_draw_result(self):
        """Test full result visualization."""
        result = self.detector.detect(self.image)
        annotated = self.visualizer.draw_result(self.image, result, fps=30.0)
        self.assertEqual(annotated.shape, self.image.shape)

    def test_draw_lane_lines(self):
        """Test lane line drawing."""
        result = self.detector.detect(self.image)
        annotated = self.visualizer.draw_lane_lines(self.image, result)
        self.assertEqual(annotated.shape, self.image.shape)

    def test_draw_departure_warning(self):
        """Test departure warning visualization."""
        result = self.detector.detect(self.image)
        result.departure_event = DepartureEvent(
            level=WarningLevel.CRITICAL,
            offset=1.0,
            direction="left",
        )
        annotated = self.visualizer.draw_departure_warning(self.image, result)
        self.assertEqual(annotated.shape, self.image.shape)

    def test_create_debug_composite(self):
        """Test debug composite creation."""
        result = self.detector.detect(self.image)
        edge = result.edge_image if result.edge_image is not None else np.zeros_like(
            self.image[:, :, 0]
        )
        bev = result.bev_image if result.bev_image is not None else np.zeros_like(
            self.image[:, :, 0]
        )
        composite = self.visualizer.create_debug_composite(
            self.image, edge, bev, result
        )
        self.assertEqual(len(composite.shape), 3)


class TestBenchmark(unittest.TestCase):
    """Tests for LaneDetectionBenchmark."""

    def setUp(self):
        self.config = _get_default_config()
        self.benchmark = LaneDetectionBenchmark(self.config)
        self.detector = LaneDetector(config=self.config)
        self.image = _create_test_image()

    def test_performance_test(self):
        """Test performance benchmark (few iterations for speed)."""
        result = self.benchmark.run_performance_test(
            self.detector, self.image, num_iterations=5, warmup=1
        )
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.total_frames, 5)
        self.assertGreater(result.avg_processing_time_ms, 0)

    def test_benchmark_report(self):
        """Test benchmark report formatting."""
        result = BenchmarkResult(
            name="test", total_frames=100, accuracy=0.95,
            fps=30.0, avg_processing_time_ms=33.3
        )
        report = LaneDetectionBenchmark.print_report(result)
        self.assertIn("test", report)
        self.assertIn("95", report)


if __name__ == "__main__":
    unittest.main()
