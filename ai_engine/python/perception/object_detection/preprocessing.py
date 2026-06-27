"""
Image Preprocessing Module
=========================

Handles image preprocessing for lane detection including undistortion,
ROI cropping, color space conversion, histogram equalization, and
contrast enhancement.

Classes:
    ImagePreprocessor - Main preprocessing pipeline

Typical usage:
    >>> preprocessor = ImagePreprocessor(config)
    >>> processed = preprocessor.process(frame)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing pipeline for lane detection.

    Applies a sequence of preprocessing steps to raw camera frames
    to prepare them for lane detection algorithms. The pipeline
    includes undistortion, color conversion, contrast enhancement,
    ROI cropping, and noise reduction.

    Attributes:
        config: Configuration dictionary for preprocessing parameters.
        roi_mask: Precomputed mask for the region of interest.
        undistort_map_x: Precomputed undistortion map for x-axis.
        undistort_map_y: Precomputed undistortion map for y-axis.
    """

    # Supported color space conversions
    COLOR_CONVERSIONS: Dict[str, int] = {
        "HLS": cv2.COLOR_BGR2HLS,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "YUV": cv2.COLOR_BGR2YUV,
        "RGB": cv2.COLOR_BGR2RGB,
        "GRAY": cv2.COLOR_BGR2GRAY,
    }

    def __init__(self, config: Dict[str, Any],
                 calibration_params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the image preprocessor.

        Args:
            config: Configuration dictionary with preprocessing settings.
                Expected keys: roi_vertices, color_space, equalize_histogram,
                equalize_channel, gaussian_kernel, gaussian_sigma,
                use_clahe, clahe_clip_limit, clahe_grid_size,
                apply_undistort, downsample_factor.
            calibration_params: Optional camera calibration parameters
                containing intrinsic matrix and distortion coefficients.
        """
        self.config = config
        self.calibration_params = calibration_params
        self.roi_mask: Optional[np.ndarray] = None
        self.undistort_map_x: Optional[np.ndarray] = None
        self.undistort_map_y: Optional[np.ndarray] = None
        self._image_shape: Optional[Tuple[int, int]] = None

        self._init_roi_mask()
        self._init_undistortion_maps()
        self._init_clahe()

        logger.info("ImagePreprocessor initialized successfully")

    def _init_roi_mask(self) -> None:
        """Precompute the region of interest mask from config vertices."""
        vertices = self.config.get("roi_vertices", [
            [0.1, 1.0], [0.4, 0.4], [0.6, 0.4], [0.9, 1.0]
        ])
        # We'll create the mask lazily when we know the image dimensions
        self._roi_vertices_normalized = vertices
        logger.debug(f"ROI vertices (normalized): {vertices}")

    def _create_roi_mask(self, height: int, width: int) -> np.ndarray:
        """Create the ROI mask for the given image dimensions.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Binary mask of shape (height, width) with ROI set to 255.
        """
        vertices = np.array([
            [int(v[0] * width), int(v[1] * height)]
            for v in self._roi_vertices_normalized
        ], dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def _init_undistortion_maps(self) -> None:
        """Initialize undistortion remap matrices if calibration is available."""
        if not self.config.get("apply_undistort", True):
            logger.info("Undistortion disabled in config")
            return

        if self.calibration_params is None:
            logger.warning("No calibration params provided; skipping undistortion init")
            return

        camera_matrix = np.array(
            self.calibration_params.get("intrinsic_matrix", []), dtype=np.float64
        )
        dist_coeffs = np.array(
            self.calibration_params.get("distortion_coefficients", []), dtype=np.float64
        )

        if camera_matrix.size == 0 or dist_coeffs.size == 0:
            logger.warning("Empty calibration matrices; skipping undistortion init")
            return

        # Maps will be computed on first use when image size is known
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        logger.info("Undistortion parameters loaded")

    def _compute_undistortion_maps(self, height: int, width: int) -> None:
        """Compute undistortion remap matrices for the given image size.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
        """
        if not hasattr(self, '_camera_matrix'):
            return

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix, self._dist_coeffs,
            (width, height), 1, (width, height)
        )
        self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(
            self._camera_matrix, self._dist_coeffs, None,
            new_camera_matrix, (width, height), cv2.CV_32FC1
        )
        self._new_camera_matrix = new_camera_matrix
        logger.debug(f"Undistortion maps computed for {width}x{height}")

    def _init_clahe(self) -> None:
        """Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if self.config.get("use_clahe", True):
            clip_limit = self.config.get("clahe_clip_limit", 2.0)
            grid_size = tuple(self.config.get("clahe_grid_size", [8, 8]))
            self._clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=grid_size
            )
            logger.debug(f"CLAHE initialized: clip={clip_limit}, grid={grid_size}")
        else:
            self._clahe = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to a frame.

        The pipeline order:
        1. Downsampling (if configured)
        2. Undistortion (if calibrated)
        3. Color space conversion
        4. Histogram equalization / CLAHE
        5. Gaussian blur
        6. ROI masking

        Args:
            frame: Input BGR image (HxWxC).

        Returns:
            Preprocessed image ready for lane detection.

        Raises:
            ValueError: If frame is empty or has wrong number of channels.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR image, got shape {frame.shape}"
            )

        result = frame.copy()
        h, w = result.shape[:2]

        # Step 1: Downsample
        result = self._apply_downsample(result)

        # Update dimensions after potential downsampling
        h, w = result.shape[:2]

        # Lazy initialization of size-dependent resources
        if self._image_shape != (h, w):
            self._image_shape = (h, w)
            self.roi_mask = self._create_roi_mask(h, w)
            if self.config.get("apply_undistort", True) and self.calibration_params:
                self._compute_undistortion_maps(h, w)

        # Step 2: Undistort
        if self.config.get("apply_undistort", True) and self.undistort_map_x is not None:
            result = self._apply_undistortion(result)

        # Step 3: Color conversion
        color_space = self.config.get("color_space", "HLS")
        converted = self._convert_color(result, color_space)

        # Step 4: Histogram equalization / CLAHE
        if self.config.get("equalize_histogram", True) or self.config.get("use_clahe", True):
            converted = self._apply_equalization(converted)

        # Step 5: Gaussian blur
        kernel = tuple(self.config.get("gaussian_kernel", [5, 5]))
        sigma = self.config.get("gaussian_sigma", 1.4)
        converted = cv2.GaussianBlur(converted, kernel, sigma)

        # Step 6: ROI masking
        if self.roi_mask is not None:
            if len(converted.shape) == 2:
                converted = cv2.bitwise_and(converted, self.roi_mask)
            else:
                mask_3ch = cv2.merge([self.roi_mask] * converted.shape[2])
                converted = cv2.bitwise_and(converted, mask_3ch)

        logger.debug(
            f"Preprocessing complete: {frame.shape} -> {converted.shape}"
        )
        return converted

    def _apply_downsample(self, image: np.ndarray) -> np.ndarray:
        """Downsample the image by the configured factor.

        Args:
            image: Input image.

        Returns:
            Downsampled image or original if factor is 1.0.
        """
        factor = self.config.get("downsample_factor", 1.0)
        if factor >= 1.0 or factor <= 0.0:
            return image

        new_width = int(image.shape[1] * factor)
        new_height = int(image.shape[0] * factor)
        return cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    def _apply_undistortion(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion using precomputed remap matrices.

        Args:
            image: Distorted input image.

        Returns:
            Undistorted image.
        """
        if self.undistort_map_x is None or self.undistort_map_y is None:
            return image

        return cv2.remap(
            image, self.undistort_map_x, self.undistort_map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    def _convert_color(self, image: np.ndarray,
                       color_space: str) -> np.ndarray:
        """Convert image to the specified color space.

        Args:
            image: BGR input image.
            color_space: Target color space name (HLS, HSV, LAB, YUV, RGB, GRAY).

        Returns:
            Converted image in the target color space.
        """
        conversion = self.COLOR_CONVERSIONS.get(color_space.upper())
        if conversion is None:
            logger.warning(
                f"Unknown color space '{color_space}'; using BGR"
            )
            return image

        return cv2.cvtColor(image, conversion)

    def _apply_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization or CLAHE to improve contrast.

        For multi-channel images, only the specified channel is equalized.
        For single-channel images, the whole image is equalized.

        Args:
            image: Input image (single or multi-channel).

        Returns:
            Contrast-enhanced image.
        """
        if len(image.shape) == 2:
            # Single channel - apply directly
            if self._clahe is not None:
                return self._clahe.apply(image)
            return cv2.equalizeHist(image)

        # Multi-channel - equalize specified channel
        equalize_channel = self.config.get("equalize_channel", 2)
        channels = list(cv2.split(image))

        if 0 <= equalize_channel < len(channels):
            if self._clahe is not None:
                channels[equalize_channel] = self._clahe.apply(
                    channels[equalize_channel]
                )
            else:
                channels[equalize_channel] = cv2.equalizeHist(
                    channels[equalize_channel]
                )

        return cv2.merge(channels)

    def extract_channel(self, image: np.ndarray,
                        channel_index: int) -> np.ndarray:
        """Extract a specific channel from a multi-channel image.

        Args:
            image: Multi-channel input image.
            channel_index: Zero-based index of the channel to extract.

        Returns:
            Single-channel image.

        Raises:
            ValueError: If channel_index is out of range.
        """
        if len(image.shape) == 2:
            return image

        if channel_index < 0 or channel_index >= image.shape[2]:
            raise ValueError(
                f"Channel index {channel_index} out of range "
                f"for image with {image.shape[2]} channels"
            )

        return image[:, :, channel_index]

    def create_white_yellow_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a binary mask isolating white and yellow regions.

        Used to enhance lane markings in various lighting conditions.
        Operates in HSV color space for better color segmentation.

        Args:
            image: BGR input image.

        Returns:
            Binary mask where white/yellow pixels are 255, others 0.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # White mask - low saturation, high value
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        # Yellow mask - specific hue range
        yellow_lower = np.array([15, 80, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Combine masks
        combined = cv2.bitwise_or(white_mask, yellow_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        return combined

    def adaptive_threshold(self, gray: np.ndarray,
                           method: str = "otsu") -> np.ndarray:
        """Apply adaptive thresholding for robust lane edge extraction.

        Args:
            gray: Grayscale input image.
            method: Thresholding method - "otsu", "adaptive_mean",
                or "adaptive_gaussian".

        Returns:
            Binary thresholded image.
        """
        if method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "adaptive_mean":
            block_size = self.config.get("adaptive", {}).get("block_size", 11)
            c_offset = self.config.get("adaptive", {}).get("c_offset", 2)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, block_size, c_offset
            )
        elif method == "adaptive_gaussian":
            block_size = self.config.get("adaptive", {}).get("block_size", 11)
            c_offset = self.config.get("adaptive", {}).get("c_offset", 2)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_offset
            )
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        return binary

    def get_roi_vertices_pixels(self, height: int,
                                width: int) -> np.ndarray:
        """Get ROI vertices in pixel coordinates.

        Args:
            height: Image height.
            width: Image width.

        Returns:
            Array of ROI vertices as (x, y) pixel coordinates.
        """
        return np.array([
            [int(v[0] * width), int(v[1] * height)]
            for v in self._roi_vertices_normalized
        ], dtype=np.int32)

    def reset(self) -> None:
        """Reset cached state, forcing recomputation on next frame."""
        self._image_shape = None
        self.roi_mask = None
        self.undistort_map_x = None
        self.undistort_map_y = None
        logger.debug("ImagePreprocessor state reset")
