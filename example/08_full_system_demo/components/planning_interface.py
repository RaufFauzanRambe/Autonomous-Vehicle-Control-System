"""Goal-based planning interface for the end-to-end system demo.

The real pipeline would replace this component with the A* global path
planner of module ``05_path_planning_demo`` and the tracking controller
of module ``06_motion_control_simulation``. For this self-contained demo
the planner steers straight toward a fixed goal point: it computes the
heading toward the goal from the (noisy) localization estimate, applies
a slow-down zone near the goal, and exposes a pure-pursuit-style
lookahead point.

Extension point: replace with the module 05 A* path plus the module 06
tracking controller; the ``compute_target`` return schema is designed to
stay compatible with that upgrade.
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

from vehicle.vehicle_state import VehicleState  # noqa: E402  (path setup above)


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle onto the canonical interval [-pi, pi].

    Args:
        angle_rad: Unwrapped angle [rad].

    Returns:
        Equivalent angle wrapped to [-pi, pi] [rad].
    """
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class PlanningInterface:
    """Straight-line goal planner producing speed and heading targets.

    Args:
        config: Planning configuration dictionary with keys
            ``target_speed_mps`` (fallback cruise target [m/s], typical
            8-30) and ``steering_gain`` (heading-error to steering
            factor [-], typical 0.5-1.5; consumed by the surrogate
            controller in ``main.py``).

    Attributes:
        target_speed_mps: Fallback cruise target speed [m/s].
        steering_gain: Heading-error to steering factor [-].
    """

    _GOAL_SLOWDOWN_DISTANCE_M: float = 10.0  # [m] distance inside which the target speed is capped
    _GOAL_SLOWDOWN_SPEED_MPS: float = 2.0    # [m/s] capped target speed in the slow-down zone
    _LOOKAHEAD_DISTANCE_M: float = 5.0       # [m] pure-pursuit-style lookahead distance

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store planning configuration.

        Args:
            config: Mapping with ``target_speed_mps`` [m/s] and
                ``steering_gain`` [-].

        Raises:
            KeyError: If a required configuration key is missing.
        """
        self.target_speed_mps = float(config["target_speed_mps"])
        self.steering_gain = float(config["steering_gain"])
        self._goal_xy: Optional[Tuple[float, float]] = None

    def set_goal(self, x: float, y: float) -> None:
        """Set the goal point the planner steers toward.

        Args:
            x: Goal x position [m].
            y: Goal y position [m].
        """
        self._goal_xy = (float(x), float(y))

    def compute_target(self, estimate: VehicleState) -> Dict[str, Any]:
        """Compute the speed/heading target toward the goal.

        Args:
            estimate: Current (noisy) ego pose estimate from the
                localization interface (x/y [m], yaw [rad], speed [m/s]).

        Returns:
            Target dictionary with ``target_speed_mps`` [m/s] (the
            configured cruise target, capped to ``min(target, 2.0)``
            [m/s] when the goal lies within 10 m), ``target_heading_rad``
            [rad] (atan2 heading toward the goal, wrapped to [-pi, pi])
            and ``lookahead_point_xy`` ([x, y] in [m]: the estimated
            position advanced 5 m along the target heading).

        Raises:
            RuntimeError: If :meth:`set_goal` has not been called yet.

        Extension point: return the first waypoint of a module 05 A*
        path (re-planned at a fixed rate) and let the module 06 tracking
        controller produce the speed/heading pair instead.
        """
        if self._goal_xy is None:
            raise RuntimeError("set_goal() must be called before compute_target().")
        goal_x, goal_y = self._goal_xy
        dx_m = goal_x - estimate.x
        dy_m = goal_y - estimate.y
        goal_distance_m = math.hypot(dx_m, dy_m)

        target_heading_rad = wrap_angle(math.atan2(dy_m, dx_m))
        target_speed_mps = self.target_speed_mps
        if goal_distance_m < self._GOAL_SLOWDOWN_DISTANCE_M:
            target_speed_mps = min(
                self.target_speed_mps, self._GOAL_SLOWDOWN_SPEED_MPS
            )

        lookahead_point_xy = [
            float(estimate.x + self._LOOKAHEAD_DISTANCE_M * math.cos(target_heading_rad)),
            float(estimate.y + self._LOOKAHEAD_DISTANCE_M * math.sin(target_heading_rad)),
        ]
        return {
            "target_speed_mps": float(target_speed_mps),
            "target_heading_rad": float(target_heading_rad),
            "lookahead_point_xy": lookahead_point_xy,
        }
