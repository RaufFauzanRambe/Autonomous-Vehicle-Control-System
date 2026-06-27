"""
Edge Detection Module
====================

Implements multiple edge detection strategies for lane boundary extraction,
including Canny, Sobel, color-based, adaptive, and combined approaches.

Classes:
    EdgeDetector - Main edge detection interface

Typical usage:
    >>> detector = EdgeDetector(config)
    >>> edges = detector.detect(preprocessed_image)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Multi-strategy edge detector optimized for lane boundaries.

    Provides several edge detection methods and the ability to combine
    them for robust lane edge extraction under varying conditions.

    Attributes:
        config: Configuration dictionary for edge detection parameters.
        method: Active detection method name.
    """

    VALID_METHODS = {"canny", "sobel", "color", "adaptive", "combined"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the edge detector.

        Args:
            config: Configuration dictionary with edge detection settings.
                Expected sections: canny, sobel, color_edge, adaptive,
                combined_weights.
        """
        self.config = config
        self.method = config.get("method", "canny")

        if self.method not in self.VALID_METHODS:
            logger.warning(
                f"Unknown method '{self.method}'; falling back to 'canny'"
            )
            self.method = "canny"

        # Precompute color range arrays
        color_cfg = config.get("color_edge", {})
        self._white_lower = np.array([
            color_cfg.get("white_h_range", [0, 180])[0],
            color_cfg.get("white_s_range", [0, 30])[0],
            color_cfg.get("white_v_range", [200, 255])[0],
        ])
        self._white_upper = np.array([
            color_cfg.get("white_h_range", [0, 180])[1],
            color_cfg.get("white_s_range", [0, 30])[1],
            color_cfg.get("white_v_range", [200, 255])[1],
        ])
        self._yellow_lower = np.array([
            color_cfg.get("yellow_h_range", [15, 35])[0],
            color_cfg.get("yellow_s_range", [100, 255])[0],
            color_cfg.get("yellow_v_range", [100, 255])[0],
        ])
        self._yellow_upper = np.array([
            color_cfg.get("yellow_h_range", [15, 35])[1],
            color_cfg.get("yellow_s_range", [100, 255])[1],
            color_cfg.get("yellow_v_range", [100, 255])[1],
        ])

        logger.info(f"EdgeDetector initialized with method='{self.method}'")

    def detect(self, image: np.ndarray,
               method: Optional[str] = None) -> np.ndarray:
        """Detect edges in the input image.

        Args:
            image: Input image (BGR, grayscale, or other multi-channel).
            method: Override the configured detection method.

        Returns:
            Binary edge image (uint8, 0 or 255).

        Raises:
            ValueError: If an unknown method is specified.
        """
        method = method or self.method

        if method == "canny":
            return self.canny_edges(image)
        elif method == "sobel":
            return self.sobel_edges(image)
        elif method == "color":
            return self.color_edges(image)
        elif method == "adaptive":
            return self.adaptive_edges(image)
        elif method == "combined":
            return self.combined_edges(image)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")

    def canny_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection.

        Automatically converts to grayscale if needed, then applies
        Canny with configured thresholds.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary Canny edge image.
        """
        gray = self._ensure_grayscale(image)

        canny_cfg = self.config.get("canny", {})
        low = canny_cfg.get("low_threshold", 50)
        high = canny_cfg.get("high_threshold", 150)
        aperture = canny_cfg.get("aperture_size", 3)
        use_l2 = canny_cfg.get("use_l2_gradient", True)

        edges = cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=use_l2)

        logger.debug(
            f"Canny edges: low={low}, high={high}, "
            f"non-zero pixels={np.count_nonzero(edges)}"
        )
        return edges

    def sobel_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel-based edge detection with directional filtering.

        Computes gradient magnitude and direction, then applies
        thresholds to isolate lane-like edges.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary Sobel edge image.
        """
        gray = self._ensure_grayscale(image)

        sobel_cfg = self.config.get("sobel", {})
        ksize = sobel_cfg.get("kernel_size", 3)
        x_weight = sobel_cfg.get("x_weight", 1.0)
        y_weight = sobel_cfg.get("y_weight", 0.3)
        mag_threshold = sobel_cfg.get("magnitude_threshold", 50)
        dir_range = sobel_cfg.get("direction_threshold", [0.3, 1.2])

        # Compute gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        # Weighted combination
        weighted = np.abs(sobel_x) * x_weight + np.abs(sobel_y) * y_weight

        # Gradient magnitude
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Gradient direction
        direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

        # Apply thresholds
        mag_binary = (magnitude > mag_threshold).astype(np.uint8) * 255
        dir_binary = (
            (direction >= dir_range[0]) & (direction <= dir_range[1])
        ).astype(np.uint8) * 255

        # Combine magnitude and direction
        combined = cv2.bitwise_and(mag_binary, dir_binary)

        logger.debug(
            f"Sobel edges: ksize={ksize}, mag_thresh={mag_threshold}, "
            f"non-zero pixels={np.count_nonzero(combined)}"
        )
        return combined

    def color_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect lane edges using color segmentation.

        Identifies white and yellow lane markings in HSV space,
        then produces a binary edge mask.

        Args:
            image: Input BGR image.

        Returns:
            Binary color-based edge image.
        """
        if len(image.shape) == 2:
            # Grayscale - fall back to thresholding
            _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            return binary

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # White lane mask
        white_mask = cv2.inRange(hsv, self._white_lower, self._white_upper)

        # Yellow lane mask
        yellow_mask = cv2.inRange(hsv, self._yellow_lower, self._yellow_upper)

        # Combine
        combined = cv2.bitwise_or(white_mask, yellow_mask)

        # Morphological operations to clean up noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Remove small noise
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)

        # Close gaps in lane markings
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)

        # Extract edges from the color mask
        edges = cv2.Canny(combined, 50, 150)

        # Dilate slightly to thicken edges
        edges = cv2.dilate(edges, kernel_small, iterations=1)

        logger.debug(
            f"Color edges: white_px={np.count_nonzero(white_mask)}, "
            f"yellow_px={np.count_nonzero(yellow_mask)}, "
            f"edge_px={np.count_nonzero(edges)}"
        )
        return edges

    def adaptive_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for edge detection.

        Uses Otsu's method or adaptive Gaussian thresholding for robust
        edge extraction under varying lighting conditions.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Binary adaptively-thresholded edge image.
        """
        gray = self._ensure_grayscale(image)

        adaptive_cfg = self.config.get("adaptive", {})
        use_otsu = adaptive_cfg.get("use_otsu", True)
        block_size = adaptive_cfg.get("block_size", 11)
        c_offset = adaptive_cfg.get("c_offset", 2)

        if use_otsu:
            # Otsu's method for global threshold
            otsu_threshold, _ = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Use Otsu threshold to set Canny parameters adaptively
            low = int(otsu_threshold * 0.5)
            high = int(otsu_threshold)
            edges = cv2.Canny(gray, low, high)
        else:
            # Adaptive Gaussian thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_offset
            )
            # Edge detection on the binary image
            edges = cv2.Canny(binary, 50, 150)

        logger.debug(
            f"Adaptive edges: otsu={use_otsu}, "
            f"non-zero pixels={np.count_nonzero(edges)}"
        )
        return edges

    def combined_edges(self, image: np.ndarray) -> np.ndarray:
        """Combine multiple edge detection methods for robustness.

        Uses configurable weights to blend Canny, Sobel, and color-based
        edge maps into a single edge image.

        Args:
            image: Input image (BGR).

        Returns:
            Combined binary edge image.
        """
        weights_cfg = self.config.get("combined_weights", {
            "canny": 0.5, "sobel": 0.3, "color": 0.2
        })

        w_canny = weights_cfg.get("canny", 0.5)
        w_sobel = weights_cfg.get("sobel", 0.3)
        w_color = weights_cfg.get("color", 0.2)

        # Get individual edge maps
        canny_map = self.canny_edges(image).astype(np.float32) / 255.0
        sobel_map = self.sobel_edges(image).astype(np.float32) / 255.0
        color_map = self.color_edges(image).astype(np.float32) / 255.0

        # Weighted combination
        combined = (
            w_canny * canny_map +
            w_sobel * sobel_map +
            w_color * color_map
        )

        # Normalize and threshold
        if combined.max() > 0:
            combined = (combined / combined.max() * 255).astype(np.uint8)
        else:
            combined = combined.astype(np.uint8)

        # Final threshold to get binary edge map
        _, binary = cv2.threshold(combined, 80, 255, cv2.THRESH_BINARY)

        logger.debug(
            f"Combined edges: weights=({w_canny}, {w_sobel}, {w_color}), "
            f"non-zero pixels={np.count_nonzero(binary)}"
        )
        return binary

    def auto_canny(self, image: np.ndarray,
                   sigma: float = 0.33) -> np.ndarray:
        """Compute Canny edges with automatically determined thresholds.

        Uses the median of the image to compute high and low thresholds,
        following the automatic Canny method by Rosebrock (2014).

        Args:
            image: Input image (BGR or grayscale).
            sigma: Threshold scaling factor (default 0.33).

        Returns:
            Binary Canny edge image.
        """
        gray = self._ensure_grayscale(image)
        median = np.median(gray)

        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))

        edges = cv2.Canny(gray, low, high)

        logger.debug(
            f"Auto Canny: median={median:.1f}, "
            f"low={low}, high={high}"
        )
        return edges

    def detect_lane_edges_multi_scale(
        self, image: np.ndarray, scales: list = [1.0, 0.5, 0.25]
    ) -> np.ndarray:
        """Detect lane edges at multiple scales for robustness.

        Processes the image at different resolutions and combines
        the results to capture both fine and coarse lane structures.

        Args:
            image: Input image (BGR).
            scales: List of scale factors to process at.

        Returns:
            Combined multi-scale binary edge image.
        """
        h, w = image.shape[:2]
        accumulated = np.zeros((h, w), dtype=np.float32)

        for scale in scales:
            if scale == 1.0:
                scaled = image
                target_h, target_w = h, w
            else:
                target_w = int(w * scale)
                target_h = int(h * scale)
                scaled = cv2.resize(image, (target_w, target_h),
                                    interpolation=cv2.INTER_AREA)

            # Detect edges at this scale
            edges = self.canny_edges(scaled)

            # Resize back to original size
            if scale != 1.0:
                edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
                _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

            accumulated += edges.astype(np.float32)

        # Normalize and threshold
        if accumulated.max() > 0:
            accumulated = (accumulated / accumulated.max() * 255).astype(np.uint8)
        else:
            accumulated = accumulated.astype(np.uint8)

        _, result = cv2.threshold(accumulated, 64, 255, cv2.THRESH_BINARY)

        logger.debug(
            f"Multi-scale edges: scales={scales}, "
            f"non-zero pixels={np.count_nonzero(result)}"
        )
        return result

    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if it isn't already.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Grayscale image (single channel).
        """
        if len(image.shape) == 2:
            return image

        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 1:
            return image[:, :, 0]

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def set_method(self, method: str) -> None:
        """Change the active edge detection method.

        Args:
            method: Detection method name.

        Raises:
            ValueError: If method is not recognized.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Valid: {self.VALID_METHODS}"
            )
        self.method = method
        logger.info(f"Edge detection method changed to '{method}'")
