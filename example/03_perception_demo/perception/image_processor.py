"""Image preprocessing and synthetic road-scene generation.

This module holds :class:`ImageProcessor`, which provides

* classical preprocessing (grayscale conversion, Gaussian smoothing,
  Canny edge extraction) used by the lane detector, and
* a deterministic synthetic road frame generator used by the demo
  because ``sample_data/`` ships no camera images (its ``.gitkeep``
  marks where real frames would be placed).

Extension point: replace the fixed pipeline in ``preprocess`` with a
configurable graph (e.g. undistortion from a calibrated camera model,
CLAHE contrast normalization, learned feature extractors) and replace
``synthetic_road_image`` with a camera driver or dataset reader.
"""

from __future__ import annotations

import cv2
import numpy as np

_SKY_BGR = (205, 205, 205)
_ROAD_BGR = (140, 140, 140)
_LINE_WHITE_BGR = (255, 255, 255)
_VEHICLE_RED_BGR = (50, 50, 150)
_VEHICLE_BLUE_BGR = (180, 110, 70)
_SIGN_RED_BGR = (0, 0, 220)


class ImageProcessor:
    """Grayscale conversion, Gaussian smoothing and Canny edge extraction.

    The processor is stateless between calls; all parameters come from
    the ``preprocessing`` section of ``config/perception_config.yaml``.

    Extension point: add histogram normalization (CLAHE) and lens
    undistortion before smoothing for real camera frames.
    """

    def __init__(self, config: dict) -> None:
        """Configure the preprocessing chain.

        Args:
            config: ``preprocessing`` section of the module YAML with keys
                ``gaussian_kernel_px`` [px] (odd kernel size),
                ``canny_low`` [-] and ``canny_high`` [-] (Canny hysteresis
                thresholds on 0-255 gradient magnitude).
        """
        self._kernel_px = int(config["gaussian_kernel_px"])
        if self._kernel_px % 2 == 0:
            raise ValueError(f"gaussian_kernel_px must be odd, got {self._kernel_px}")
        self._canny_low = float(config["canny_low"])
        self._canny_high = float(config["canny_high"])

    def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run the gray -> blur -> Canny chain on one BGR frame.

        Args:
            image: Input frame, ``uint8`` array of shape ``(height, width, 3)``
                in BGR channel order.

        Returns:
            Dictionary with three arrays:

            * ``"gray"``: ``uint8`` array ``(height, width)`` in [0, 255].
            * ``"blurred"``: Gaussian-smoothed grayscale, ``uint8``
              ``(height, width)``.
            * ``"edges"``: binary Canny edge map, ``uint8`` ``(height, width)``
              with values in {0, 255}.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self._kernel_px, self._kernel_px), 0)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)
        return {"gray": gray, "blurred": blurred, "edges": edges}

    @staticmethod
    def synthetic_road_image(width: int, height: int, seed: int = 42) -> tuple[np.ndarray, dict]:
        """Render a deterministic synthetic road scene (no real camera needed).

        The scene contains a light-gray sky, a dark-gray road trapezoid
        that narrows toward the upper middle of the frame, solid white
        side lane lines, a dashed white center line, two filled parked
        vehicles and a red roadside circle that serves as a stop-sign
        stand-in. Drawing is fully deterministic (plain OpenCV/numpy
        primitives, no randomness); ``seed`` is kept for interface
        stability so callers can pass it uniformly.

        The road surface is intentionally brighter (gray 140) than the
        vehicles and the sign (gray 66-106) so that Otsu thresholding in
        :class:`~perception.object_detection.ObjectDetector` separates
        the dark objects from the ground plane.

        Args:
            width: Frame width [px], typical 640-1920.
            height: Frame height [px], typical 480-1080.
            seed: Random seed, unused because the renderer is
                deterministic; kept for API compatibility.

        Returns:
            Tuple ``(image_bgr, meta)``:

            * ``image_bgr``: ``uint8`` array ``(height, width, 3)``, BGR order.
            * ``meta``: Ground-truth annotations with keys
              ``"sign_bbox_px"``: ``[x1, y1, x2, y2]`` [px] bounding box of
              the sign stand-in, and ``"object_bboxes_px"``: list of two
              ``[x1, y1, x2, y2]`` [px] vehicle bounding boxes.

        Extension point: replace this renderer with a camera driver, a
        rosbag/KITTI reader, or a photorealistic simulator bridge, and
        derive ``meta`` from dataset ground truth annotations.
        """
        del seed  # deterministic rendering; parameter kept for interface stability
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        if min(width, height) < 120:
            raise ValueError(f"frame {width}x{height} is too small for the scene proportions")

        image = np.full((height, width, 3), _SKY_BGR, dtype=np.uint8)

        # Road trapezoid: full width at the bottom, narrowing toward the
        # upper middle (vanishing strip at 55 percent of frame height).
        y_road_top = int(0.55 * height)
        road_polygon = np.array(
            [
                [int(0.42 * width), y_road_top],
                [int(0.58 * width), y_road_top],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [road_polygon], _ROAD_BGR)

        # Solid white side lane lines, slightly inside the road edges.
        line_thickness = max(2, height // 100)
        cv2.line(
            image,
            (int(0.44 * width), int(0.57 * height)),
            (int(0.06 * width), height - 1),
            _LINE_WHITE_BGR,
            line_thickness,
        )
        cv2.line(
            image,
            (int(0.56 * width), int(0.57 * height)),
            (int(0.94 * width), height - 1),
            _LINE_WHITE_BGR,
            line_thickness,
        )

        # Dashed white center line, exactly vertical (symmetric camera).
        dash_length = max(8, height // 20)
        gap_length = dash_length
        center_thickness = max(2, height // 160)
        x_center = width // 2
        y_cursor = int(0.58 * height)
        while y_cursor < height - 1:
            y_end = min(y_cursor + dash_length, height - 1)
            cv2.line(image, (x_center, y_cursor), (x_center, y_end), _LINE_WHITE_BGR, center_thickness)
            y_cursor += dash_length + gap_length

        # Two filled rectangles as parked vehicles on the roadside.
        left_vehicle = (int(0.05 * width), int(0.60 * height), int(0.20 * width), int(0.76 * height))
        right_vehicle = (int(0.80 * width), int(0.60 * height), int(0.95 * width), int(0.76 * height))
        cv2.rectangle(
            image,
            (left_vehicle[0], left_vehicle[1]),
            (left_vehicle[2], left_vehicle[3]),
            _VEHICLE_RED_BGR,
            thickness=-1,
        )
        cv2.rectangle(
            image,
            (right_vehicle[0], right_vehicle[1]),
            (right_vehicle[2], right_vehicle[3]),
            _VEHICLE_BLUE_BGR,
            thickness=-1,
        )

        # Red filled circle on the roadside as a stop-sign stand-in.
        radius = max(12, height // 12)
        sign_center = (int(0.30 * width), int(0.45 * height))
        cv2.circle(image, sign_center, radius, _SIGN_RED_BGR, thickness=-1)

        meta = {
            "sign_bbox_px": [
                int(sign_center[0] - radius),
                int(sign_center[1] - radius),
                int(sign_center[0] + radius),
                int(sign_center[1] + radius),
            ],
            "object_bboxes_px": [list(left_vehicle), list(right_vehicle)],
        }
        return image, meta
