"""Inertial measurement unit model with bias and Gaussian noise.

Reports longitudinal/lateral acceleration and yaw rate. The yaw rate is
formed by finite differencing the ego yaw between consecutive updates.
Every reading dict returned by :meth:`IMUSensor.get_reading` has EXACTLY
the four envelope keys ``sensor_type``, ``timestamp``, ``frame_id`` and
``data``, matching ``shared/schemas/sensor_data.schema.json``.

Extension point: run the state derivatives through a first-order specific
force model (gravity projection on a non-flat road) and estimate/remove
the bias with an Allan-variance-based calibration pipeline.
"""

from __future__ import annotations

import importlib
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


class IMUSensor:
    """Accelerometer + gyroscope stand-in with constant bias and white noise.

    On each ``update`` the longitudinal acceleration is
    ``state.acceleration + bias_ax_mpss + normal(0, accel_noise_std_mpss)``
    [m/s^2]; the lateral acceleration is pure white noise [m/s^2]. The yaw
    rate is the finite difference ``(state.yaw - prev_yaw) / dt`` [rad/s]
    with the previous yaw stored internally (0.0 baseline on the very first
    update) plus ``normal(0, gyro_noise_std_radps)`` noise.

    Args:
        config: FLAT ``imu`` section of ``config/sensor_config.yaml`` with
            keys ``accel_noise_std_mpss`` [m/s^2],
            ``gyro_noise_std_radps`` [rad/s], ``bias_ax_mpss`` [m/s^2] and
            ``frame_id`` (default ``"imu_body"``).
        seed: Seed for the internal ``np.random.default_rng``; readings are
            deterministic for a fixed seed.
    """

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        """Store IMU parameters and initialize the reading buffer.

        Args:
            config: Flat IMU configuration section (see class docstring).
            seed: Random seed for accelerometer and gyroscope noise.
        """
        self.accel_noise_std_mpss: float = float(config.get("accel_noise_std_mpss", 0.05))
        self.gyro_noise_std_radps: float = float(config.get("gyro_noise_std_radps", 0.005))
        self.bias_ax_mpss: float = float(config.get("bias_ax_mpss", 0.1))
        self.frame_id: str = str(config.get("frame_id", "imu_body"))
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self._prev_yaw: float | None = None
        self._timestamp: float = 0.0
        self._data: Dict[str, Any] = self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, Any]:
        """Zero/empty IMU payload used before the first ``update``.

        Returns:
            Zero accelerations and zero yaw rate.
        """
        return {"ax_mpss": 0.0, "ay_mpss": 0.0, "yaw_rate_radps": 0.0}

    def update(self, state: VehicleState, dt: float) -> None:
        """Measure accelerations and yaw rate for the current ego state.

        The yaw rate finite difference uses the yaw of the previous
        ``update`` call (first call uses a 0.0 baseline, i.e. reports only
        noise). A non-positive ``dt`` is treated as "no interval elapsed"
        and also yields the 0.0 baseline for the yaw rate.

        Args:
            state: Current ego vehicle state (``acceleration`` [m/s^2],
                ``yaw`` [rad] and ``timestamp`` [s] are consumed).
            dt: Time step [s] used for the yaw finite difference.
        """
        ax_mpss = (
            state.acceleration
            + self.bias_ax_mpss
            + float(self.rng.normal(0.0, self.accel_noise_std_mpss))
        )
        ay_mpss = float(self.rng.normal(0.0, self.accel_noise_std_mpss))

        if self._prev_yaw is None or dt <= 0.0:
            yaw_rate_radps = 0.0
        else:
            yaw_rate_radps = (state.yaw - self._prev_yaw) / dt
        yaw_rate_radps += float(self.rng.normal(0.0, self.gyro_noise_std_radps))
        self._prev_yaw = state.yaw

        self._timestamp = float(state.timestamp)
        self._data = {
            "ax_mpss": float(ax_mpss),
            "ay_mpss": ay_mpss,
            "yaw_rate_radps": float(yaw_rate_radps),
        }

    def get_reading(self) -> Dict[str, Any]:
        """Return the last computed IMU reading.

        Returns:
            Envelope dict with EXACTLY the keys ``sensor_type`` (``"imu"``),
            ``timestamp`` [s], ``frame_id`` and ``data``. ``data`` contains
            ``ax_mpss`` [m/s^2], ``ay_mpss`` [m/s^2] and
            ``yaw_rate_radps`` [rad/s], all in the body frame. Before the
            first :meth:`update` call the payload is zero/empty.
        """
        return {
            "sensor_type": "imu",
            "timestamp": self._timestamp,
            "frame_id": self.frame_id,
            "data": self._data,
        }
