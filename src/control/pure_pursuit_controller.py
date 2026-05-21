"""
Pure Pursuit Lateral Controller for Autonomous Vehicle Control System.

Implements the classic Pure Pursuit algorithm for path following with
adaptive lookahead distance, curvature-based steering, and path
interpolation for smooth tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PurePursuitParams:
    """Tunable parameters for the Pure Pursuit controller.

    Attributes:
        k_ld: Lookahead distance coefficient: ``L_d = k_ld * v + min_ld``.
        min_ld: Minimum lookahead distance [m].
        max_ld: Maximum lookahead distance [m].
        max_steer: Maximum steering angle [rad].
        wheelbase: Vehicle wheelbase [m].
        interpolation_resolution: Maximum spacing between interpolated
            path points [m].  Finer resolution improves tracking accuracy
            at the cost of memory.
    """

    k_ld: float = 0.8
    min_ld: float = 3.0
    max_ld: float = 20.0
    max_steer: float = 0.6
    wheelbase: float = 2.9
    interpolation_resolution: float = 0.5


class PurePursuitController:
    """Pure Pursuit lateral controller for path following.

    The Pure Pursuit algorithm tracks a path by steering toward a
    lookahead point on the reference trajectory.  The lookahead distance
    adapts with vehicle speed for stable behaviour at all velocities.

    The steering command is derived from the circular arc that connects
    the rear axle to the lookahead point:

        δ = arctan(2 · L · sin(α) / L_d)

    where *L* is the wheelbase, *α* is the angle between the vehicle
    heading and the lookahead point, and *L_d* is the lookahead distance.

    Features:
    - Adaptive lookahead distance scaled by velocity.
    - Linear interpolation of path for sub-point accuracy.
    - Curvature computation for downstream speed planning.

    Example::

        ctrl = PurePursuitController(params=PurePursuitParams(k_ld=1.0))
        steer = ctrl.compute(
            vehicle_x=5.0, vehicle_y=2.0, vehicle_yaw=0.0,
            vehicle_speed=10.0,
            path_x=px, path_y=py,
        )
    """

    def __init__(self, params: Optional[PurePursuitParams] = None) -> None:
        """Initialise the Pure Pursuit controller.

        Args:
            params: Tunable parameters. Uses defaults if *None*.
        """
        self.params = params if params is not None else PurePursuitParams()

        # Cached interpolated path
        self._interp_x: Optional[NDArray[np.float64]] = None
        self._interp_y: Optional[NDArray[np.float64]] = None
        self._path_interp_dirty: bool = True

        # Diagnostic state
        self._last_lookahead_dist: float = 0.0
        self._last_alpha: float = 0.0
        self._last_curvature: float = 0.0
        self._last_target_idx: int = 0

    # ------------------------------------------------------------------
    # Path interpolation
    # ------------------------------------------------------------------

    def interpolate_path(
        self,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate the reference path for smoother tracking.

        Uses linear interpolation to ensure the path points are spaced
        no more than ``interpolation_resolution`` metres apart.

        Args:
            path_x: Original path x-coordinates (1-D).
            path_y: Original path y-coordinates (1-D).

        Returns:
            Tuple ``(interp_x, interp_y)`` of interpolated coordinates.
        """
        if len(path_x) < 2:
            return path_x.copy(), path_y.copy()

        res = self.params.interpolation_resolution
        interp_x_list: list[float] = [float(path_x[0])]
        interp_y_list: list[float] = [float(path_y[0])]

        for i in range(1, len(path_x)):
            dx = path_x[i] - path_x[i - 1]
            dy = path_y[i] - path_y[i - 1]
            seg_len = math.sqrt(dx * dx + dy * dy)

            if seg_len < 1e-9:
                continue

            n_sub = max(1, int(math.ceil(seg_len / res)))
            for j in range(1, n_sub + 1):
                t = j / n_sub
                interp_x_list.append(float(path_x[i - 1] + t * dx))
                interp_y_list.append(float(path_y[i - 1] + t * dy))

        interp_x = np.array(interp_x_list, dtype=np.float64)
        interp_y = np.array(interp_y_list, dtype=np.float64)
        return interp_x, interp_y

    def set_path(
        self,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
    ) -> None:
        """Set (or update) the reference path and trigger interpolation.

        Args:
            path_x: Path x-coordinates.
            path_y: Path y-coordinates.
        """
        self._interp_x, self._interp_y = self.interpolate_path(path_x, path_y)
        self._path_interp_dirty = False
        self._last_target_idx = 0

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _adaptive_lookahead(self, speed: float) -> float:
        """Compute speed-adaptive lookahead distance.

        L_d = clip(k_ld · v + min_ld, min_ld, max_ld)

        Args:
            speed: Current longitudinal speed [m/s].

        Returns:
            Lookahead distance [m].
        """
        ld = self.params.k_ld * speed + self.params.min_ld
        return float(np.clip(ld, self.params.min_ld, self.params.max_ld))

    def _find_lookahead_point(
        self,
        rx: float,
        ry: float,
        ryaw: float,
        lookahead_dist: float,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
        start_idx: int = 0,
    ) -> Tuple[float, float, int]:
        """Find the lookahead target point on the path.

        Iterates forward from *start_idx* until the first point whose
        arc-length distance from the rear axle exceeds *lookahead_dist*.

        If no such point exists (end of path), returns the last point.

        Args:
            rx: Rear-axle x.
            ry: Rear-axle y.
            ryaw: Vehicle heading.
            lookahead_dist: Desired lookahead distance.
            path_x: Path x-coordinates.
            path_y: Path y-coordinates.
            start_idx: Start searching from this index.

        Returns:
            Tuple ``(target_x, target_y, target_index)``.
        """
        # Find nearest point first
        dx = path_x[start_idx:] - rx
        dy = path_y[start_idx:] - ry
        dists = np.sqrt(dx * dx + dy * dy)
        nearest_local = int(np.argmin(dists))
        nearest_idx = start_idx + nearest_local

        # Search forward from nearest for the lookahead point
        target_idx = nearest_idx
        for i in range(nearest_idx, len(path_x)):
            dx_t = path_x[i] - rx
            dy_t = path_y[i] - ry
            dist_t = math.sqrt(dx_t * dx_t + dy_t * dy_t)
            if dist_t >= lookahead_dist:
                target_idx = i
                break
        else:
            # End of path – use last point
            target_idx = len(path_x) - 1

        # Linear interpolation for smoother target
        if target_idx > 0:
            dx_prev = path_x[target_idx - 1] - rx
            dy_prev = path_y[target_idx - 1] - ry
            dist_prev = math.sqrt(dx_prev * dx_prev + dy_prev * dy_prev)
            dx_curr = path_x[target_idx] - rx
            dy_curr = path_y[target_idx] - ry
            dist_curr = math.sqrt(dx_curr * dx_curr + dy_curr * dy_curr)

            if dist_curr - dist_prev > 1e-9:
                t = (lookahead_dist - dist_prev) / (dist_curr - dist_prev)
                t = float(np.clip(t, 0.0, 1.0))
                target_x = path_x[target_idx - 1] + t * (path_x[target_idx] - path_x[target_idx - 1])
                target_y = path_y[target_idx - 1] + t * (path_y[target_idx] - path_y[target_idx - 1])
            else:
                target_x = float(path_x[target_idx])
                target_y = float(path_y[target_idx])
        else:
            target_x = float(path_x[target_idx])
            target_y = float(path_y[target_idx])

        return target_x, target_y, target_idx

    def _compute_alpha(
        self,
        rx: float,
        ry: float,
        ryaw: float,
        target_x: float,
        target_y: float,
    ) -> float:
        """Compute the angle α between vehicle heading and target point.

        Args:
            rx: Rear-axle x.
            ry: Rear-axle y.
            ryaw: Vehicle heading [rad].
            target_x: Lookahead point x.
            target_y: Lookahead point y.

        Returns:
            Angle α [rad].
        """
        dx = target_x - rx
        dy = target_y - ry
        angle_to_target = math.atan2(dy, dx)
        alpha = self._normalise_angle(angle_to_target - ryaw)
        return alpha

    @staticmethod
    def _compute_curvature(wheelbase: float, alpha: float, lookahead: float) -> float:
        """Compute the path curvature implied by the steering command.

        κ = 2 · sin(α) / L_d

        Args:
            wheelbase: Vehicle wheelbase [m].
            alpha: Angle to lookahead point [rad].
            lookahead: Lookahead distance [m].

        Returns:
            Curvature κ [1/m].
        """
        if lookahead < 1e-6:
            return 0.0
        return 2.0 * math.sin(alpha) / lookahead

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        vehicle_speed: float,
        path_x: NDArray[np.float64],
        path_y: NDArray[np.float64],
    ) -> float:
        """Compute the Pure Pursuit steering command.

        If :meth:`set_path` was called previously, the interpolated path
        is used; otherwise interpolation is performed on-the-fly.

        Args:
            vehicle_x: Rear-axle x position [m].
            vehicle_y: Rear-axle y position [m].
            vehicle_yaw: Vehicle heading [rad].
            vehicle_speed: Longitudinal speed [m/s].
            path_x: Reference path x-coordinates (1-D).
            path_y: Reference path y-coordinates (1-D).

        Returns:
            Steering angle [rad], clipped to ``[-max_steer, max_steer]``.
        """
        if len(path_x) == 0 or len(path_y) == 0:
            raise ValueError("Path arrays must not be empty")

        # Use interpolated path if available
        if self._interp_x is not None and not self._path_interp_dirty:
            px = self._interp_x
            py = self._interp_y
        else:
            px, py = self.interpolate_path(path_x, path_y)

        # 1. Adaptive lookahead distance
        ld = self._adaptive_lookahead(vehicle_speed)

        # 2. Rear axle position (assume vehicle_x, vehicle_y are at rear axle)
        rx, ry = vehicle_x, vehicle_y

        # 3. Find lookahead point
        start_idx = min(self._last_target_idx, len(px) - 1)
        target_x, target_y, target_idx = self._find_lookahead_point(
            rx, ry, vehicle_yaw, ld, px, py, start_idx
        )

        # 4. Compute angle to target
        alpha = self._compute_alpha(rx, ry, vehicle_yaw, target_x, target_y)

        # 5. Compute steering: δ = arctan(2 · L · sin(α) / L_d)
        L = self.params.wheelbase
        if ld > 1e-6:
            steer = math.atan2(2.0 * L * math.sin(alpha), ld)
        else:
            steer = 0.0

        # 6. Clip to steering limits
        steer = float(np.clip(steer, -self.params.max_steer, self.params.max_steer))

        # 7. Curvature for downstream speed planner
        kappa = self._compute_curvature(L, alpha, ld)

        # Save diagnostics
        self._last_lookahead_dist = ld
        self._last_alpha = alpha
        self._last_curvature = kappa
        self._last_target_idx = target_idx

        return steer

    # ------------------------------------------------------------------
    # Angle utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_angle(angle: float) -> float:
        """Normalise angle to ``[-π, π]``."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def lookahead_distance(self) -> float:
        """Lookahead distance from the last computation [m]."""
        return self._last_lookahead_dist

    @property
    def alpha(self) -> float:
        """Angle to the lookahead target from the last computation [rad]."""
        return self._last_alpha

    @property
    def curvature(self) -> float:
        """Implied path curvature from the last computation [1/m]."""
        return self._last_curvature

    @property
    def target_index(self) -> int:
        """Index of the lookahead target point on the path."""
        return self._last_target_idx

    def reset(self) -> None:
        """Reset internal state and cached path."""
        self._interp_x = None
        self._interp_y = None
        self._path_interp_dirty = True
        self._last_lookahead_dist = 0.0
        self._last_alpha = 0.0
        self._last_curvature = 0.0
        self._last_target_idx = 0

    def __repr__(self) -> str:
        return (
            f"PurePursuitController(k_ld={self.params.k_ld}, "
            f"wheelbase={self.params.wheelbase}, "
            f"lookahead=[{self.params.min_ld:.1f}, {self.params.max_ld:.1f}])"
        )
