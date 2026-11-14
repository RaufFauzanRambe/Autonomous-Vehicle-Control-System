"""Synthetic scanning lidar sensor producing top-down point clouds.

Generates a standalone synthetic point cloud over the front semicircle of
the ego vehicle. Every reading dict returned by :meth:`LidarSensor.get_reading`
has EXACTLY the four envelope keys ``sensor_type``, ``timestamp``,
``frame_id`` and ``data``, matching ``shared/schemas/sensor_data.schema.json``.

The cloud is deliberately standalone (uniform ranges): it exercises
consumers of point-cloud payloads without needing a world model.

Extension point: ray-cast the beam set (``channels`` beams spread over the
front semicircle, transformed by the ego yaw) against the road environment
of ``01_basic_simulation`` so returns come from real road geometry.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

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


class LidarSensor:
    """Synthetic spinning lidar over the ego front semicircle.

    Beams are spread with ``np.linspace(-pi/2, pi/2, points_per_scan)``
    relative to the ego yaw (front semicircle; sensor frame: x forward,
    y left). Ranges are drawn uniformly between 10 m and ``max_range_m``
    and perturbed with zero-mean Gaussian noise of standard deviation
    ``range_noise_std_m``; results are clamped to the physical interval
    ``[min_range_m, max_range_m]``. Points are stored in the sensor frame,
    so for the straight-line demo (ego yaw constant) the cloud is invariant
    to the heading value.

    Args:
        config: FLAT ``lidar`` section of ``config/sensor_config.yaml`` with
            keys ``channels`` [-], ``points_per_scan`` [-],
            ``min_range_m`` [m], ``max_range_m`` [m],
            ``range_noise_std_m`` [m] and the optional ``frame_id``
            (default ``"lidar_top"``).
        seed: Seed for the internal ``np.random.default_rng``; scans are
            deterministic for a fixed seed.
    """

    #: Minimum uniform-draw range of the synthetic return distribution [m].
    MIN_UNIFORM_RANGE_M: float = 10.0

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        """Store lidar parameters and initialize the reading buffer.

        Args:
            config: Flat lidar configuration section (see class docstring).
            seed: Random seed for range noise.
        """
        self.channels: int = int(config.get("channels", 16))
        self.points_per_scan: int = int(config.get("points_per_scan", 120))
        self.min_range_m: float = float(config.get("min_range_m", 1.0))
        self.max_range_m: float = float(config.get("max_range_m", 100.0))
        self.range_noise_std_m: float = float(config.get("range_noise_std_m", 0.02))
        self.frame_id: str = str(config.get("frame_id", "lidar_top"))
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self._timestamp: float = 0.0
        self._data: Dict[str, Any] = self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, Any]:
        """Zero/empty lidar payload used before the first ``update``.

        Returns:
            Zero point count and an empty point list.
        """
        return {"n_points": 0, "points_xy_m": []}

    def update(self, state: VehicleState, dt: float) -> None:
        """Generate one synthetic scan and refresh the buffered reading.

        Bearings span ``[-pi/2, pi/2]`` around the ego front direction
        (sensor frame x axis). Each range is
        ``uniform(10, max_range_m) + normal(0, range_noise_std_m)``,
        clamped to ``[min_range_m, max_range_m]``. Points are computed as
        ``[range * cos(bearing), range * sin(bearing)]`` in the sensor
        frame and rounded to 3 decimals. ``dt`` is accepted for interface
        uniformity but unused (each scan is independent).

        Args:
            state: Current ego vehicle state (``yaw`` [rad] defines the
                front semicircle orientation in the world frame).
            dt: Time step [s] (unused; kept for the common sensor API).
        """
        del dt  # standalone synthetic cloud: scans are independent samples

        bearings = np.linspace(-math.pi / 2.0, math.pi / 2.0, self.points_per_scan)
        ranges = self.rng.uniform(
            self.MIN_UNIFORM_RANGE_M, self.max_range_m, size=self.points_per_scan
        ) + self.rng.normal(0.0, self.range_noise_std_m, size=self.points_per_scan)
        ranges = np.clip(ranges, self.min_range_m, self.max_range_m)

        points_xy: List[List[float]] = [
            [round(float(r * math.cos(b)), 3), round(float(r * math.sin(b)), 3)]
            for r, b in zip(ranges, bearings)
        ]

        self._timestamp = float(state.timestamp)
        self._data = {"n_points": len(points_xy), "points_xy_m": points_xy}

    def get_reading(self) -> Dict[str, Any]:
        """Return the last computed lidar reading.

        Returns:
            Envelope dict with EXACTLY the keys ``sensor_type`` (``"lidar"``),
            ``timestamp`` [s], ``frame_id`` and ``data``. ``data`` contains
            ``n_points`` [-] and ``points_xy_m`` ([m], sensor frame, rounded
            to 3 decimals). Before the first :meth:`update` call the payload
            is zero/empty.
        """
        return {
            "sensor_type": "lidar",
            "timestamp": self._timestamp,
            "frame_id": self.frame_id,
            "data": self._data,
        }
