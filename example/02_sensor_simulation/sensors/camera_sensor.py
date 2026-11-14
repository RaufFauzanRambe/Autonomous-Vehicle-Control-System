"""Pinhole-camera sensor model producing bounding-box detections.

Simulates a front-facing monocular camera. One virtual lead vehicle is kept
at a fixed ego-relative offset (constant-gap stand-in for a real tracking
scenario) and projected onto the image plane with a pinhole model. Every
reading dict returned by :meth:`CameraSensor.get_reading` has EXACTLY the
four envelope keys ``sensor_type``, ``timestamp``, ``frame_id`` and
``data``, matching ``shared/schemas/sensor_data.schema.json``.

Extension point: render synthetic images (cv2) of the lead vehicle into
``sample_data/camera/`` and derive detections from a lightweight detector
instead of projecting a single analytic bbox.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent       # .../02_sensor_simulation/sensors
ROOT = MODULE_DIR.parents[1]                        # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# VehicleState lives in a digit-leading namespace package and therefore
# cannot be reached with a regular import statement; see project README.
VehicleState = importlib.import_module(
    "01_basic_simulation.vehicle.vehicle_state"
).VehicleState


class CameraSensor:
    """Monocular camera sensor with analytic pinhole projection.

    The virtual lead vehicle sits at a constant ego-relative offset
    (45 m ahead, 3 m to the left), i.e. a constant-gap stand-in rather than
    a physically tracked object.

    Args:
        config: FLAT ``camera`` section of ``config/sensor_config.yaml`` with
            keys ``image_width_px`` [px], ``image_height_px`` [px],
            ``fov_deg`` [deg] (horizontal), ``max_detection_range_m`` [m],
            ``detection_noise_std_px`` [px] and the optional
            ``frame_id`` (default ``"camera_front"``).
        seed: Seed for the internal ``np.random.default_rng``; readings are
            deterministic for a fixed seed.
    """

    #: Longitudinal (forward) offset of the virtual lead vehicle [m].
    LEAD_LONGITUDINAL_OFFSET_M: float = 45.0
    #: Lateral offset of the virtual lead vehicle, positive = left [m].
    LEAD_LATERAL_OFFSET_M: float = 3.0
    #: Half extent of the projected bounding box on each side [px].
    BBOX_HALF_SIZE_PX: float = 20.0
    #: Fixed detection confidence of the analytic stand-in detector [-].
    DETECTION_CONFIDENCE: float = 0.92

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        """Store camera parameters and initialize the reading buffer.

        Args:
            config: Flat camera configuration section (see class docstring).
            seed: Random seed for detection jitter.
        """
        self.image_width_px: int = int(config.get("image_width_px", 1280))
        self.image_height_px: int = int(config.get("image_height_px", 720))
        self.fov_deg: float = float(config.get("fov_deg", 90.0))
        self.max_detection_range_m: float = float(config.get("max_detection_range_m", 80.0))
        self.detection_noise_std_px: float = float(config.get("detection_noise_std_px", 3.0))
        self.frame_id: str = str(config.get("frame_id", "camera_front"))
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self._timestamp: float = 0.0
        self._data: Dict[str, Any] = self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, Any]:
        """Zero/empty camera payload used before the first ``update``.

        Returns:
            Empty detection list plus the (unknown-at-start) zero image size.
        """
        return {"detections": [], "image_width_px": 0, "image_height_px": 0}

    def update(self, state: VehicleState, dt: float) -> None:
        """Project the virtual lead vehicle and refresh the buffered reading.

        The lead vehicle is placed at a fixed ego-relative offset, rotated
        by the ego yaw into the world frame. Its lateral/longitudinal
        ego-frame coordinates are projected with the pinhole model
        ``focal_px = (image_width_px / 2) / tan(fov_rad / 2)``; the bbox
        center is jittered with zero-mean Gaussian noise of standard
        deviation ``detection_noise_std_px``. Beyond
        ``max_detection_range_m`` the detection list is empty. ``dt`` is
        accepted for interface uniformity but unused (the stand-in gap is
        constant).

        Args:
            state: Current ego vehicle state (``x``, ``y`` [m], ``yaw``
                [rad] are consumed).
            dt: Time step [s] (unused; kept for the common sensor API).
        """
        del dt  # constant-gap stand-in: no temporal dynamics to integrate

        # World-frame position of the constant-gap lead vehicle.
        lead_x_m = state.x + math.cos(state.yaw) * self.LEAD_LONGITUDINAL_OFFSET_M - math.sin(
            state.yaw
        ) * self.LEAD_LATERAL_OFFSET_M
        lead_y_m = state.y + math.sin(state.yaw) * self.LEAD_LONGITUDINAL_OFFSET_M + math.cos(
            state.yaw
        ) * self.LEAD_LATERAL_OFFSET_M

        # Back-rotation into the ego frame (x forward, y left) for projection.
        dx_m = lead_x_m - state.x
        dy_m = lead_y_m - state.y
        longitudinal_m = math.cos(state.yaw) * dx_m + math.sin(state.yaw) * dy_m
        lateral_m = -math.sin(state.yaw) * dx_m + math.cos(state.yaw) * dy_m

        ego_range_m = math.hypot(longitudinal_m, lateral_m)
        detections: list[Dict[str, Any]] = []
        if ego_range_m <= self.max_detection_range_m:
            fov_rad = math.radians(self.fov_deg)
            focal_px = (self.image_width_px / 2.0) / math.tan(fov_rad / 2.0)
            u_center = self.image_width_px / 2.0 + focal_px * (lateral_m / longitudinal_m)
            v_center = self.image_height_px / 2.0
            u_jitter, v_jitter = self.rng.normal(0.0, self.detection_noise_std_px, size=2)
            u_center += float(u_jitter)
            v_center += float(v_jitter)
            half = self.BBOX_HALF_SIZE_PX
            detections.append(
                {
                    "label": "vehicle",
                    "bbox_px": [
                        float(u_center - half),
                        float(v_center - half),
                        float(u_center + half),
                        float(v_center + half),
                    ],
                    "confidence": self.DETECTION_CONFIDENCE,
                }
            )

        self._timestamp = float(state.timestamp)
        self._data = {
            "detections": detections,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
        }

    def get_reading(self) -> Dict[str, Any]:
        """Return the last computed camera reading.

        Returns:
            Envelope dict with EXACTLY the keys ``sensor_type`` (``"camera"``),
            ``timestamp`` [s], ``frame_id`` and ``data``. ``data`` contains
            ``detections`` (list of ``{label, bbox_px, confidence}``),
            ``image_width_px`` [px] and ``image_height_px`` [px]. Before the
            first :meth:`update` call the payload is zero/empty.
        """
        return {
            "sensor_type": "camera",
            "timestamp": self._timestamp,
            "frame_id": self.frame_id,
            "data": self._data,
        }
