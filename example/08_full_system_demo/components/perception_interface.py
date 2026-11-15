"""Synthetic perception interface for the end-to-end system demo.

In the full project pipeline this component would wrap the sensor
simulation of module ``02_sensor_simulation`` and feed its raw readings
through the perception stack of module ``03_perception_demo``. For this
self-contained demo it instead injects a configurable number of static
vehicles at fixed ego-relative offsets, so the downstream planning and
control stages always receive a realistic-looking detection list.

Detections are expressed in the ego frame: ``distance_m`` is the
Euclidean magnitude of the ego-relative offset and ``bearing_rad`` is
``atan2(lateral, longitudinal)`` of that offset, i.e. the angle from the
ego forward axis (positive to the left).

Extension point: replace the synthetic actor injection with real sensor
readings from module 02 passed through module 03, returning the same
detection dictionary schema.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent            # .../08_full_system_demo/components
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, MODULE_DIR.parent, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vehicle.vehicle_state import VehicleState  # noqa: E402  (path setup above)


class PerceptionInterface:
    """Stub perception stack producing synthetic vehicle detections.

    Args:
        config: Perception configuration dictionary with keys
            ``max_range_m`` (detection range [m], typical 50-200) and
            ``n_static_actors`` (number of injected synthetic vehicles
            [-], typical 0-4).

    Attributes:
        max_range_m: Detections beyond this range are discarded [m].
        n_static_actors: Number of synthetic static vehicles injected
            per tick [-].
    """

    #: Fixed ego-relative offsets (longitudinal_m ahead, lateral_m left)
    #: [m], [m]; the list cycles when more actors are requested.
    _STATIC_OFFSETS_M: Tuple[Tuple[float, float], ...] = ((30.0, 2.0), (60.0, -2.0))

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store perception configuration.

        Args:
            config: Mapping with ``max_range_m`` [m] and
                ``n_static_actors`` [-].

        Raises:
            KeyError: If a required configuration key is missing.
        """
        self.max_range_m = float(config["max_range_m"])
        self.n_static_actors = int(config["n_static_actors"])

    def sense(
        self,
        truth: VehicleState,
        rng: np.random.Generator,
    ) -> List[Dict[str, Any]]:
        """Produce one frame of synthetic vehicle detections.

        Each tick injects ``n_static_actors`` static vehicles at the
        fixed ego-relative offsets (30 m ahead with +2 m lateral offset
        and 60 m ahead with -2 m lateral offset; the offset list cycles
        when more actors are requested). Only detections within
        ``max_range_m`` are reported.

        Args:
            truth: Ego ground-truth state (x/y [m], yaw [rad], speed
                [m/s]). Unused by this stub because the offsets are
                ego-relative; kept for signature compatibility with the
                real sensor pipeline.
            rng: Random generator reserved for future sensor noise
                models; the synthetic offsets are deterministic, so this
                stub consumes no randomness.

        Returns:
            List of detection dictionaries, each with ``label`` [-],
            ``distance_m`` [m], ``bearing_rad`` [rad] and ``speed_mps``
            [m/s].

        Extension point: sample the static actors into world coordinates
        via ``truth``, add range/bearing noise from ``rng``, and forward
        module 02 sensor readings through module 03 instead.
        """
        detections: List[Dict[str, Any]] = []
        for index in range(self.n_static_actors):
            longitudinal_m, lateral_m = self._STATIC_OFFSETS_M[
                index % len(self._STATIC_OFFSETS_M)
            ]
            distance_m = math.hypot(longitudinal_m, lateral_m)
            if distance_m > self.max_range_m:
                continue
            detections.append(
                {
                    "label": "vehicle",
                    "distance_m": float(distance_m),
                    "bearing_rad": float(math.atan2(lateral_m, longitudinal_m)),
                    "speed_mps": 0.0,
                }
            )
        return detections
