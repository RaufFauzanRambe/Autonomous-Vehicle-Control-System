"""GNSS position sensor model with Gaussian position noise.

Reports the ego position corrupted by zero-mean Gaussian noise. Every
reading dict returned by :meth:`GPSSensor.get_reading` has EXACTLY the
four envelope keys ``sensor_type``, ``timestamp``, ``frame_id`` and
``data``, matching ``shared/schemas/sensor_data.schema.json``.

Extension point: add correlated (colored) noise via an Ornstein-Uhlenbeck
process, satellite-drift biases and occasional multipath outliers.
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


class GPSSensor:
    """Independent two-axis Gaussian position noise sensor.

    Args:
        config: FLAT ``gps`` section of ``config/sensor_config.yaml`` with
            keys ``position_noise_std_m`` [m] and ``frame_id``
            (default ``"gps_antenna"``).
        seed: Seed for the internal ``np.random.default_rng``; readings are
            deterministic for a fixed seed.
    """

    #: Fixed fix-quality value reported by the idealized receiver [-].
    FIX_QUALITY: float = 0.9

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        """Store GPS parameters and initialize the reading buffer.

        Args:
            config: Flat GPS configuration section (see class docstring).
            seed: Random seed for position noise.
        """
        self.position_noise_std_m: float = float(config.get("position_noise_std_m", 1.5))
        self.frame_id: str = str(config.get("frame_id", "gps_antenna"))
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self._timestamp: float = 0.0
        self._data: Dict[str, Any] = self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, Any]:
        """Zero/empty GPS payload used before the first ``update``.

        Returns:
            Zero position and zero fix quality.
        """
        return {"x_m": 0.0, "y_m": 0.0, "fix_quality": 0.0}

    def update(self, state: VehicleState, dt: float) -> None:
        """Measure the ego position with additive Gaussian noise.

        ``x_measured = state.x + normal(0, position_noise_std_m)`` and
        likewise for ``y``; the noise draws for the two axes are
        independent. ``dt`` is accepted for interface uniformity but
        unused (no temporal correlation is modeled).

        Args:
            state: Current ego vehicle state (``x``, ``y`` [m] and
                ``timestamp`` [s] are consumed).
            dt: Time step [s] (unused; kept for the common sensor API).
        """
        del dt  # white noise: no temporal correlation to integrate

        measured_x_m = state.x + float(self.rng.normal(0.0, self.position_noise_std_m))
        measured_y_m = state.y + float(self.rng.normal(0.0, self.position_noise_std_m))

        self._timestamp = float(state.timestamp)
        self._data = {
            "x_m": measured_x_m,
            "y_m": measured_y_m,
            "fix_quality": self.FIX_QUALITY,
        }

    def get_reading(self) -> Dict[str, Any]:
        """Return the last computed GPS reading.

        Returns:
            Envelope dict with EXACTLY the keys ``sensor_type`` (``"gps"``),
            ``timestamp`` [s], ``frame_id`` and ``data``. ``data`` contains
            ``x_m`` [m], ``y_m`` [m] (noise-corrupted world-frame position)
            and ``fix_quality`` [-]. Before the first :meth:`update` call
            the payload is zero/empty.
        """
        return {
            "sensor_type": "gps",
            "timestamp": self._timestamp,
            "frame_id": self.frame_id,
            "data": self._data,
        }
