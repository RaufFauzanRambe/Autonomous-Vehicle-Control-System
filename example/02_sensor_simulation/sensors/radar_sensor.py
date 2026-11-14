"""Radar sensor model tracking static virtual targets in range/azimuth.

Two static targets are placed at fixed WORLD-frame coordinates. For each
target the sensor reports range, azimuth relative to the ego heading and
the analytic range rate of a static target. Every reading dict returned by
:meth:`RadarSensor.get_reading` has EXACTLY the four envelope keys
``sensor_type``, ``timestamp``, ``frame_id`` and ``data``, matching
``shared/schemas/sensor_data.schema.json``.

Extension point: add per-target RCS/amplitude, azimuth noise, Doppler
ambiguity handling and multi-path ghosts, and filter targets by the
configured field of view instead of range alone.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


class RadarSensor:
    """Forward radar observing two static virtual targets.

    Static targets live at fixed WORLD-frame coordinates
    ``TARGETS = ((40.0, -2.0), (70.0, 3.0))`` [m] and are reported with the
    stable identifiers ``"A"`` and ``"B"``. For a static target the range
    rate is ``-ego_speed * cos(azimuth)`` where ``azimuth`` is measured
    relative to the ego heading; zero-mean Gaussian noise of standard
    deviation ``range_noise_std_m`` is added to the range rate. Range and
    azimuth themselves are computed exactly in this stand-in. Targets are
    reported only while within ``max_range_m``; the configured field of
    view is stored as metadata but not used for filtering.

    Args:
        config: FLAT ``radar`` section of ``config/sensor_config.yaml`` with
            keys ``max_range_m`` [m], ``fov_deg`` [deg],
            ``range_noise_std_m`` [m] and the optional ``frame_id``
            (default ``"radar_front"``).
        seed: Seed for the internal ``np.random.default_rng``; readings are
            deterministic for a fixed seed.
    """

    #: Static virtual targets in the WORLD frame ((x, y) tuples) [m].
    TARGETS: Tuple[Tuple[float, float], ...] = ((40.0, -2.0), (70.0, 3.0))
    #: Stable target identifiers aligned with :attr:`TARGETS` [-].
    TARGET_IDS: Tuple[str, ...] = ("A", "B")

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        """Store radar parameters and initialize the reading buffer.

        Args:
            config: Flat radar configuration section (see class docstring).
            seed: Random seed for range-rate noise.
        """
        self.max_range_m: float = float(config.get("max_range_m", 200.0))
        self.fov_deg: float = float(config.get("fov_deg", 70.0))
        self.range_noise_std_m: float = float(config.get("range_noise_std_m", 0.1))
        self.frame_id: str = str(config.get("frame_id", "radar_front"))
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self._timestamp: float = 0.0
        self._data: Dict[str, Any] = self._empty_data()

    @staticmethod
    def _empty_data() -> Dict[str, Any]:
        """Zero/empty radar payload used before the first ``update``.

        Returns:
            Empty target list.
        """
        return {"targets": []}

    def update(self, state: VehicleState, dt: float) -> None:
        """Recompute target geometry from the current ego state.

        For every static target: ``range_m = hypot(dx, dy)`` [m],
        ``azimuth_rad = atan2(dy, dx) - ego_yaw`` wrapped to ``[-pi, pi]``,
        ``range_rate_mps = -ego_speed * cos(azimuth)`` plus Gaussian noise
        of standard deviation ``range_noise_std_m`` [m/s]. Only targets
        within ``max_range_m`` are reported. ``dt`` is accepted for
        interface uniformity but unused (targets are static).

        Args:
            state: Current ego vehicle state (``x``, ``y`` [m], ``yaw``
                [rad], ``speed`` [m/s] are consumed).
            dt: Time step [s] (unused; kept for the common sensor API).
        """
        del dt  # static targets: geometry depends only on the current state

        targets: List[Dict[str, Any]] = []
        for target_id, (target_x, target_y) in zip(self.TARGET_IDS, self.TARGETS):
            dx_m = target_x - state.x
            dy_m = target_y - state.y
            range_m = math.hypot(dx_m, dy_m)
            if range_m > self.max_range_m:
                continue
            azimuth_rad = math.atan2(dy_m, dx_m) - state.yaw
            # Wrap to [-pi, pi] so the azimuth is a well-defined relative angle.
            azimuth_rad = math.atan2(math.sin(azimuth_rad), math.cos(azimuth_rad))
            range_rate_mps = -state.speed * math.cos(azimuth_rad) + float(
                self.rng.normal(0.0, self.range_noise_std_m)
            )
            targets.append(
                {
                    "target_id": target_id,
                    "range_m": float(range_m),
                    "azimuth_rad": float(azimuth_rad),
                    "range_rate_mps": float(range_rate_mps),
                }
            )

        self._timestamp = float(state.timestamp)
        self._data = {"targets": targets}

    def get_reading(self) -> Dict[str, Any]:
        """Return the last computed radar reading.

        Returns:
            Envelope dict with EXACTLY the keys ``sensor_type`` (``"radar"``),
            ``timestamp`` [s], ``frame_id`` and ``data``. ``data`` contains
            ``targets``, a list of ``{target_id, range_m [m], azimuth_rad
            [rad], range_rate_mps [m/s]}`` entries. Before the first
            :meth:`update` call the payload is zero/empty.
        """
        return {
            "sensor_type": "radar",
            "timestamp": self._timestamp,
            "frame_id": self.frame_id,
            "data": self._data,
        }
