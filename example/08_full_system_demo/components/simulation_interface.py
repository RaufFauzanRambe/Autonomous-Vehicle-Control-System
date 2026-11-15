"""Simulation harness for the end-to-end system demo.

Wraps the kinematic bicycle model of module ``01_basic_simulation``:
``SimulationInterface`` owns the tick rate, the simulated clock, the
cumulative travelled distance, the goal check and the ground-truth
vehicle state that the perception, localization and planning interfaces
observe each tick.

The vehicle parameter dictionary is defined inline on purpose: the demo
must stay runnable without a parameter file owned by another module, and
the values below are exactly the module 01 defaults (wheelbase 2.7 m,
max steering 0.6 rad, max speed 33 m/s, max accel 3 m/s^2, max decel
8 m/s^2, drag coefficient 0.02, mass 1500 kg), documented here so the
coupling stays explicit.

Extension point: replace the inline default parameters with a
configuration-owned vehicle parameter set, and add multi-vehicle world
state or weather-dependent model changes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MODULE_DIR = Path(__file__).resolve().parent            # .../08_full_system_demo/components
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, MODULE_DIR.parent, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vehicle.vehicle_model import VehicleModel  # noqa: E402  (path setup above)
from vehicle.vehicle_state import VehicleState  # noqa: E402  (path setup above)


class SimulationInterface:
    """Tick-based simulation harness around the module 01 vehicle model.

    Args:
        config: System configuration dictionary; the ``system`` section
            must provide ``tick_rate_hz`` (pipeline tick rate [Hz],
            typical 5-50) and ``duration_s`` (maximum simulated time per
            scenario [s], typical 10-60).
        scenario: Scenario dictionary providing ``start`` (mapping with
            ``x`` [m], ``y`` [m], ``yaw`` [rad], ``speed`` [m/s]) and
            ``goal`` (mapping with ``x`` [m], ``y`` [m]).

    Attributes:
        dt_s: Fixed integration step ``1.0 / tick_rate_hz`` [s].
        duration_s: Maximum simulated time per scenario [s].
        goal_xy: Goal position as ``(x, y)`` [m].
    """

    #: Module 01 default vehicle parameters, inlined and documented so
    #: this demo has no file dependency on other module folders.
    _DEFAULT_VEHICLE_PARAMS: Dict[str, float] = {
        "wheelbase_m": 2.7,        # [m] kinematic bicycle wheelbase
        "max_steering_rad": 0.6,   # [rad] steering angle clamp
        "max_speed_mps": 33.0,     # [m/s] speed clamp
        "max_accel_mpss": 3.0,     # [m/s^2] throttle = +1 acceleration
        "max_decel_mpss": 8.0,     # [m/s^2] throttle = -1 deceleration
        "drag_coefficient": 0.02,  # [-] speed-dependent drag factor
        "mass_kg": 1500.0,         # [kg] vehicle mass
    }

    _GOAL_TOLERANCE_M: float = 5.0  # [m] euclidean distance within which the goal counts as reached

    def __init__(self, config: Dict[str, Any], scenario: Dict[str, Any]) -> None:
        """Store timing, goal and vehicle model configuration.

        Args:
            config: System configuration dictionary (see class docstring).
            scenario: Scenario dictionary (see class docstring).

        Raises:
            KeyError: If required configuration or scenario keys are
                missing.
        """
        system = config["system"]
        self.dt_s = 1.0 / float(system["tick_rate_hz"])
        self.duration_s = float(system["duration_s"])
        self.goal_xy: Tuple[float, float] = (
            float(scenario["goal"]["x"]),
            float(scenario["goal"]["y"]),
        )
        self._start_state = scenario["start"]
        self._model = VehicleModel(dict(self._DEFAULT_VEHICLE_PARAMS))
        self._elapsed_s: float = 0.0
        self._distance_m: float = 0.0
        self._truth: Optional[VehicleState] = None

    @property
    def truth_state(self) -> VehicleState:
        """Ground-truth vehicle state of the current tick.

        The state is lazily initialized from the scenario ``start``
        mapping (x/y [m], yaw [rad], speed [m/s]) with ``timestamp=0.0``,
        zero acceleration and zero steering angle.

        Returns:
            The current ground-truth :class:`VehicleState`.
        """
        if self._truth is None:
            start = self._start_state
            self._truth = VehicleState(
                timestamp=0.0,
                x=float(start["x"]),
                y=float(start["y"]),
                yaw=float(start["yaw"]),
                speed=float(start["speed"]),
                acceleration=0.0,
                steering_angle=0.0,
            )
        return self._truth

    @property
    def time_s(self) -> float:
        """Simulated time elapsed since the first step [s]."""
        return self._elapsed_s

    @property
    def distance_m(self) -> float:
        """Cumulative travelled path length since the first step [m]."""
        return self._distance_m

    def step(self, truth: VehicleState, controls: Dict[str, float]) -> VehicleState:
        """Advance the simulation by one tick.

        Args:
            truth: Current ground-truth state (as returned by
                :attr:`truth_state`).
            controls: Control dictionary with ``throttle`` [-] in
                [-1, 1] and ``steering`` [rad] in [-0.6, 0.6].

        Returns:
            The new ground-truth :class:`VehicleState` after one
            kinematic bicycle Euler step of ``dt_s`` [s]; the internal
            clock, cumulative distance and stored truth state are
            advanced accordingly. Travelled distance per tick is the
            Euclidean displacement between the old and new positions.
        """
        new_state = self._model.step(
            truth,
            float(controls["throttle"]),
            float(controls["steering"]),
            self.dt_s,
        )
        self._distance_m += math.hypot(
            new_state.x - truth.x, new_state.y - truth.y
        )
        self._elapsed_s += self.dt_s
        self._truth = new_state
        return new_state

    def is_complete(self) -> bool:
        """Check whether the scenario run is finished.

        The run finishes when the goal is reached (euclidean distance of
        the ground-truth position to the goal below 5.0 m) or when the
        configured duration has elapsed.

        Returns:
            True when the goal is reached or the duration is used up,
            False otherwise.
        """
        truth = self.truth_state
        goal_distance_m = math.hypot(
            truth.x - self.goal_xy[0], truth.y - self.goal_xy[1]
        )
        return (
            goal_distance_m < self._GOAL_TOLERANCE_M
            or self._elapsed_s >= self.duration_s
        )
