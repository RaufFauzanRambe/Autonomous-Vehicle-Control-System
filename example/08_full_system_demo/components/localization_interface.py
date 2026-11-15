"""Noisy localization interface for the end-to-end system demo.

The real pipeline would fuse GNSS, odometry and inertial measurements
with the ``SensorFusion`` component of module ``04_localization_demo``.
For this self-contained demo the interface perturbs the ground-truth
pose with zero-mean Gaussian noise, emulating a raw GNSS fix.

Extension point: plug module 04 SensorFusion here, feeding the truth
pose through the fusion filter instead of direct additive noise.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent            # .../08_full_system_demo/components
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, MODULE_DIR.parent, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vehicle.vehicle_state import VehicleState  # noqa: E402  (path setup above)


class LocalizationInterface:
    """GNSS-style localization returning a noisy copy of the true pose.

    Args:
        config: Localization configuration dictionary with key
            ``gps_noise_std_m`` (position noise standard deviation [m],
            typical 0.1-5.0).

    Attributes:
        gps_noise_std_m: Standard deviation of the zero-mean Gaussian
            noise added to x and y independently [m].
        last_estimate: Most recent pose estimate, ``None`` before the
            first call to :meth:`get_pose`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store localization configuration.

        Args:
            config: Mapping with ``gps_noise_std_m`` [m].

        Raises:
            KeyError: If the required configuration key is missing.
        """
        self.gps_noise_std_m = float(config["gps_noise_std_m"])
        self.last_estimate: Optional[VehicleState] = None

    def get_pose(
        self,
        truth: VehicleState,
        rng: np.random.Generator,
    ) -> VehicleState:
        """Return the true pose with independent x/y Gaussian noise.

        Args:
            truth: Ground-truth ego state (x/y [m], yaw [rad], speed
                [m/s]).
            rng: Random generator used to draw the position noise; the
                two draws (x then y) are independent zero-mean Gaussian
                samples with standard deviation ``gps_noise_std_m`` [m].

        Returns:
            A new :class:`VehicleState` identical to ``truth`` except
            for the perturbed x/y values [m]. The estimate is also
            stored in :attr:`last_estimate`.

        Extension point: replace the additive noise with module 04
        SensorFusion (GNSS + wheel odometry + IMU) so the estimate
        covariance and yaw error become part of the experiment.
        """
        estimate = replace(
            truth,
            x=float(truth.x + float(rng.normal(0.0, self.gps_noise_std_m))),
            y=float(truth.y + float(rng.normal(0.0, self.gps_noise_std_m))),
        )
        self.last_estimate = estimate
        return estimate
