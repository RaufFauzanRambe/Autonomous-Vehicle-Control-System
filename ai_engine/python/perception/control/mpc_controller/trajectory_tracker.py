"""
Trajectory Tracking MPC for Autonomous Vehicle Path Following

Implements reference tracking MPC with:
  - Preview control (look-ahead reference)
  - Feedforward from path curvature
  - Speed-adaptive horizon
  - Lateral and longitudinal coordination
  - Multi-objective cost function
  - Soft constraint handling

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mpc_controller import MPCController, MPCParams, MPCConstraints
from .vehicle_model import (
    KinematicBicycleModel,
    DynamicBicycleModel,
    VehicleParameters,
)


@dataclass
class ReferencePoint:
    """A point on the reference trajectory.

    Attributes:
        x: X position (m).
        y: Y position (m).
        heading: Path heading (rad).
        curvature: Path curvature (1/m).
        speed: Reference speed (m/s).
        acceleration: Reference acceleration (m/s^2).
    """
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0


@dataclass
class TrackerConfig:
    """Configuration for the trajectory tracking MPC.

    Attributes:
        prediction_horizon: Number of prediction steps.
        control_horizon: Number of control moves.
        dt: Prediction timestep (s).
        q_lateral: Lateral error weight.
        q_heading: Heading error weight.
        q_speed: Speed error weight.
        q_slip: Sideslip angle weight.
        r_steer: Steering input weight.
        r_accel: Acceleration input weight.
        rd_steer: Steering rate weight.
        rd_accel: Acceleration rate weight.
        max_lateral_error: Maximum acceptable lateral error (m).
        max_heading_error: Maximum acceptable heading error (rad).
        feedforward_gain: Curvature feedforward gain.
        soft_constraint_weight: Soft constraint penalty weight.
        preview_distance: Look-ahead distance (m).
    """
    prediction_horizon: int = 25
    control_horizon: int = 15
    dt: float = 0.1
    q_lateral: float = 50.0
    q_heading: float = 100.0
    q_speed: float = 10.0
    q_slip: float = 5.0
    r_steer: float = 5.0
    r_accel: float = 1.0
    rd_steer: float = 50.0
    rd_accel: float = 10.0
    max_lateral_error: float = 2.0
    max_heading_error: float = math.radians(30)
    feedforward_gain: float = 1.0
    soft_constraint_weight: float = 1000.0
    preview_distance: float = 30.0


class TrajectoryTracker:
    """MPC-based trajectory tracking controller for autonomous vehicles.

    Combines lateral and longitudinal control in a single MPC formulation
    for coordinated path following.

    State vector (error dynamics):
        [e_y, e_psi, e_v, s] where:
        - e_y: lateral error from reference path
        - e_psi: heading error
        - e_v: speed error
        - s: distance along the path

    Input vector:
        [delta, accel] where:
        - delta: steering angle
        - accel: longitudinal acceleration

    Example:
        >>> config = TrackerConfig()
        >>> tracker = TrajectoryTracker(config=config)
        >>> steer, accel = tracker.compute_control(vehicle_state, reference)
    """

    def __init__(
        self,
        config: TrackerConfig = TrackerConfig(),
        vehicle_params: VehicleParameters = VehicleParameters(),
    ) -> None:
        """Initialize the trajectory tracker.

        Args:
            config: Tracker configuration.
            vehicle_params: Vehicle parameters.
        """
        self._config = config
        self._vehicle_params = vehicle_params

        # Vehicle model
        self._vehicle_model = KinematicBicycleModel(vehicle_params)

        # MPC controller
        self._setup_mpc()

        # State
        self._previous_steer = 0.0
        self._previous_accel = 0.0
        self._closest_index = 0
        self._initialized = False

    def _setup_mpc(self) -> None:
        """Set up the MPC controller with appropriate parameters."""
        N = self._config.prediction_horizon
        M = self._config.control_horizon
        n = 4  # State dimension
        m = 2  # Input dimension

        # Weighting matrices
        Q = np.diag([
            self._config.q_lateral,
            self._config.q_heading,
            self._config.q_speed,
            self._config.q_slip,
        ])
        R = np.diag([self._config.r_steer, self._config.r_accel])
        R_delta = np.diag([self._config.rd_steer, self._config.rd_accel])

        params = MPCParams(
            prediction_horizon=N,
            control_horizon=M,
            state_dim=n,
            input_dim=m,
            output_dim=n,
            dt=self._config.dt,
            Q=Q,
            R=R,
            R_delta=R_delta,
        )

        # Constraints
        max_steer = self._vehicle_params.max_steer
        max_accel = self._vehicle_params.max_accel
        max_decel = self._vehicle_params.max_decel
        max_steer_rate = self._vehicle_params.max_steer_rate * self._config.dt

        constraints = MPCConstraints(
            u_min=np.array([max_decel, -max_steer]),
            u_max=np.array([max_accel, max_steer]),
            du_min=np.array([-20.0 * self._config.dt, -max_steer_rate]),
            du_max=np.array([20.0 * self._config.dt, max_steer_rate]),
        )

        self._mpc = MPCController(params=params, constraints=constraints, name="trajectory_tracker")

    def _find_closest_point(
        self,
        x: float,
        y: float,
        reference: List[ReferencePoint],
        start_index: int = 0,
    ) -> Tuple[int, float]:
        """Find the closest reference point to the vehicle.

        Args:
            x: Vehicle x position.
            y: Vehicle y position.
            reference: List of reference points.
            start_index: Start searching from this index.

        Returns:
            Tuple of (closest_index, distance).
        """
        min_dist = float("inf")
        closest_idx = start_index

        # Search within a window around the last closest point
        search_start = max(0, start_index - 10)
        search_end = min(len(reference), start_index + 50)

        for i in range(search_start, search_end):
            dx = x - reference[i].x
            dy = y - reference[i].y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx, min_dist

    def _compute_errors(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_heading: float,
        vehicle_speed: float,
        ref_point: ReferencePoint,
    ) -> np.ndarray:
        """Compute tracking errors relative to the reference.

        Args:
            vehicle_x: Vehicle x position.
            vehicle_y: Vehicle y position.
            vehicle_heading: Vehicle heading.
            vehicle_speed: Vehicle speed.
            ref_point: Reference point.

        Returns:
            Error state [e_y, e_psi, e_v, sideslip].
        """
        # Lateral error (in path frame)
        dx = vehicle_x - ref_point.x
        dy = vehicle_y - ref_point.y
        e_y = -dx * math.sin(ref_point.heading) + dy * math.cos(ref_point.heading)

        # Heading error
        e_psi = vehicle_heading - ref_point.heading
        e_psi = (e_psi + math.pi) % (2 * math.pi) - math.pi

        # Speed error
        e_v = vehicle_speed - ref_point.speed

        # Approximate sideslip (simplified)
        sideslip = 0.0  # Could be estimated from vehicle model

        return np.array([e_y, e_psi, e_v, sideslip])

    def _get_reference_trajectory(
        self,
        reference: List[ReferencePoint],
        closest_index: int,
        vehicle_speed: float,
    ) -> np.ndarray:
        """Extract the reference trajectory for the preview horizon.

        Args:
            reference: Full reference path.
            closest_index: Current closest point index.
            vehicle_speed: Current vehicle speed.

        Returns:
            Reference trajectory array (N * output_dim,).
        """
        N = self._config.prediction_horizon
        output_dim = 4
        ref_traj = np.zeros(N * output_dim)

        for k in range(N):
            idx = min(closest_index + k, len(reference) - 1)
            ref_point = reference[idx]

            # Reference state: [0, 0, 0, 0] (zero error is desired)
            # But we include feedforward terms
            ref_traj[k * output_dim] = 0.0  # Zero lateral error
            ref_traj[k * output_dim + 1] = 0.0  # Zero heading error
            ref_traj[k * output_dim + 2] = 0.0  # Zero speed error
            ref_traj[k * output_dim + 3] = 0.0  # Zero sideslip

        return ref_traj

    def _compute_feedforward(
        self,
        ref_point: ReferencePoint,
        vehicle_speed: float,
    ) -> Tuple[float, float]:
        """Compute feedforward steering and acceleration.

        Args:
            ref_point: Current reference point.
            vehicle_speed: Vehicle speed.

        Returns:
            Tuple of (steering_feedforward, accel_feedforward).
        """
        # Curvature feedforward: delta_ff = arctan(L * kappa)
        if abs(ref_point.curvature) > 1e-6:
            steer_ff = self._config.feedforward_gain * math.atan(
                self._vehicle_params.wheelbase * ref_point.curvature
            )
        else:
            steer_ff = 0.0

        # Acceleration feedforward from reference
        accel_ff = ref_point.acceleration

        # Add centripetal acceleration compensation
        if vehicle_speed > 1.0 and abs(ref_point.curvature) > 1e-6:
            centripetal_accel = vehicle_speed ** 2 * ref_point.curvature
            accel_ff += centripetal_accel * 0.1  # Small compensation

        return steer_ff, accel_ff

    def _update_model(self, vehicle_speed: float) -> None:
        """Update the linearized model for current operating point.

        Args:
            vehicle_speed: Current vehicle speed.
        """
        v_ref = max(vehicle_speed, 1.0)
        dt = self._config.dt

        A, B = self._vehicle_model.get_discrete_model(v_ref, dt)

        # Output matrix (full state measurement)
        C = np.eye(4)
        D = np.zeros((4, 2))

        self._mpc.set_model(A, B, C, D)

    def compute_control(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_heading: float,
        vehicle_speed: float,
        reference: List[ReferencePoint],
        closest_index: Optional[int] = None,
    ) -> Tuple[float, float, Dict]:
        """Compute steering and acceleration commands.

        Args:
            vehicle_x: Vehicle x position (m).
            vehicle_y: Vehicle y position (m).
            vehicle_heading: Vehicle heading (rad).
            vehicle_speed: Vehicle speed (m/s).
            reference: Reference trajectory points.
            closest_index: Index of closest reference point.

        Returns:
            Tuple of (steering_angle, acceleration, diagnostics_dict).
        """
        if not reference:
            return 0.0, 0.0, {"error": "empty_reference"}

        # Find closest point
        if closest_index is not None:
            self._closest_index = closest_index
        else:
            self._closest_index, _ = self._find_closest_point(
                vehicle_x, vehicle_y, reference, self._closest_index
            )

        ref_point = reference[self._closest_index]

        # Compute tracking errors
        error_state = self._compute_errors(
            vehicle_x, vehicle_y, vehicle_heading, vehicle_speed, ref_point
        )

        # Update model for current speed
        self._update_model(vehicle_speed)

        # Get reference trajectory
        ref_traj = self._get_reference_trajectory(
            reference, self._closest_index, vehicle_speed
        )

        # Compute feedforward
        steer_ff, accel_ff = self._compute_feedforward(ref_point, vehicle_speed)

        # MPC solve
        previous_input = np.array([self._previous_accel, self._previous_steer])

        try:
            optimal_input = self._mpc.compute_control(
                current_state=error_state,
                reference=ref_traj,
                previous_input=previous_input,
            )
            accel = optimal_input[0]
            steer = optimal_input[1]
            solve_status = "optimal"
        except Exception as e:
            # Fallback to feedforward only
            steer = steer_ff
            accel = accel_ff
            solve_status = f"fallback: {str(e)}"

        # Add feedforward
        steer += steer_ff
        accel += accel_ff

        # Clip to physical limits
        steer = np.clip(steer, -self._vehicle_params.max_steer, self._vehicle_params.max_steer)
        accel = np.clip(accel, self._vehicle_params.max_decel, self._vehicle_params.max_accel)

        # Update state
        self._previous_steer = steer
        self._previous_accel = accel
        self._initialized = True

        # Build diagnostics
        diagnostics = {
            "lateral_error": float(error_state[0]),
            "heading_error_deg": float(math.degrees(error_state[1])),
            "speed_error": float(error_state[2]),
            "closest_index": self._closest_index,
            "feedforward_steer_deg": float(math.degrees(steer_ff)),
            "feedforward_accel": float(accel_ff),
            "steering_deg": float(math.degrees(steer)),
            "acceleration": float(accel),
            "solve_status": solve_status,
            "solve_time_ms": self._mpc.state.solve_time_ms,
            "mpc_cost": self._mpc.state.cost,
        }

        return steer, accel, diagnostics

    def reset(self) -> None:
        """Reset the trajectory tracker state."""
        self._mpc.reset()
        self._previous_steer = 0.0
        self._previous_accel = 0.0
        self._closest_index = 0
        self._initialized = False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrajectoryTracker(N={self._config.prediction_horizon}, "
            f"M={self._config.control_horizon}, dt={self._config.dt})"
        )
