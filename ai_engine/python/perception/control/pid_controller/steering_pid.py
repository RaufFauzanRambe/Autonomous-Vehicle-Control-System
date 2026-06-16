"""
Steering PID Controller for Autonomous Vehicle Lateral Control

Implements steering control with:
  - Stanley method integration for heading error
  - Pure Pursuit integration for path tracking
  - Feedforward steering based on curvature
  - Speed-adaptive gain scheduling
  - Understeer compensation
  - Lateral acceleration limiting

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .pid_controller import (
    AntiWindupMode,
    DerivativeMode,
    PIDController,
    PIDGains,
    PIDLimits,
    PIDFilterParams,
)
from .controller_utils import RateLimiter, LowPassFilter, Saturate, InterpolatingTable


class SteeringMode(Enum):
    """Steering controller operating mode."""
    STANLEY = "stanley"           # Stanley lateral control
    PURE_PURSUIT = "pure_pursuit"  # Pure Pursuit path tracking
    PID_ONLY = "pid_only"          # Pure PID on lateral error
    COMBINED = "combined"          # Combined Stanley + Pure Pursuit + PID


@dataclass
class VehicleGeometry:
    """Vehicle geometric parameters.

    Attributes:
        wheelbase: Distance between front and rear axles in m.
        track_width: Distance between left and right wheels in m.
        front_overhang: Distance from front axle to front bumper in m.
        rear_overhang: Distance from rear axle to rear bumper in m.
        max_steer_angle: Maximum steering angle in radians.
        max_steer_rate: Maximum steering rate in rad/s.
        steering_ratio: Ratio between steering wheel and tire angle.
        cg_to_front: Distance from CG to front axle in m.
        cg_to_rear: Distance from CG to rear axle in m.
    """
    wheelbase: float = 2.85
    track_width: float = 1.6
    front_overhang: float = 0.95
    rear_overhang: float = 1.0
    max_steer_angle: float = math.radians(35)
    max_steer_rate: float = math.radians(120)
    steering_ratio: float = 16.0
    cg_to_front: float = 1.42
    cg_to_rear: float = 1.43


@dataclass
class PathPoint:
    """A point on the reference path.

    Attributes:
        x: X position in m.
        y: Y position in m.
        heading: Path heading angle in radians.
        curvature: Path curvature in 1/m.
        speed: Reference speed at this point in m/s.
    """
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    speed: float = 0.0


@dataclass
class VehicleState:
    """Current vehicle state for steering control.

    Attributes:
        x: X position in m.
        y: Y position in m.
        heading: Vehicle heading in radians.
        speed: Vehicle speed in m/s.
        lateral_velocity: Lateral velocity in m/s.
        yaw_rate: Yaw rate in rad/s.
        steering_angle: Current steering angle in radians.
    """
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    steering_angle: float = 0.0


@dataclass
class SteeringControllerConfig:
    """Configuration for the steering PID controller.

    Attributes:
        kp: Proportional gain for lateral error.
        ki: Integral gain for lateral error.
        kd: Derivative gain for lateral error.
        stanley_k: Stanley method gain.
        pure_pursuit_ld_coeff: Pure Pursuit look-ahead distance coefficient.
        pure_pursuit_min_ld: Minimum look-ahead distance in m.
        max_lateral_accel: Maximum allowed lateral acceleration in m/s^2.
        curvature_feedforward_gain: Feedforward gain for curvature compensation.
        understeer_gain: Understeer compensation gain.
        speed_gain_schedule: Speed-dependent gain schedule table.
        dt: Controller timestep in seconds.
    """
    kp: float = 0.8
    ki: float = 0.01
    kd: float = 0.15
    stanley_k: float = 2.5
    pure_pursuit_ld_coeff: float = 1.5
    pure_pursuit_min_ld: float = 3.0
    max_lateral_accel: float = 4.0
    curvature_feedforward_gain: float = 1.0
    understeer_gain: float = 0.1
    speed_gain_schedule: Optional[dict] = None
    dt: float = 0.01


class SteeringPIDController:
    """Steering PID controller for autonomous vehicle lateral control.

    This controller integrates PID control with Stanley and Pure Pursuit
    methods for robust path tracking. It supports:
    - Stanley method for heading and lateral error correction
    - Pure Pursuit for smooth path following
    - Curvature feedforward steering
    - Speed-adaptive gain scheduling
    - Lateral acceleration limiting for safety
    - Understeer gradient compensation

    Example:
        >>> config = SteeringControllerConfig()
        >>> geometry = VehicleGeometry()
        >>> ctrl = SteeringPIDController(config=config, geometry=geometry)
        >>> steer = ctrl.update(
        ...     vehicle_state=VehicleState(x=0, y=0, heading=0, speed=15.0),
        ...     path_points=[PathPoint(x=10, y=0, heading=0, curvature=0)],
        ...     closest_index=0,
        ... )
    """

    def __init__(
        self,
        config: SteeringControllerConfig = SteeringControllerConfig(),
        geometry: VehicleGeometry = VehicleGeometry(),
    ) -> None:
        """Initialize the steering PID controller.

        Args:
            config: Steering controller configuration.
            geometry: Vehicle geometric parameters.
        """
        self._config = config
        self._geometry = geometry
        self._mode = SteeringMode.COMBINED

        # Internal PID controller for lateral error
        self._pid = PIDController(
            gains=PIDGains(kp=config.kp, ki=config.ki, kd=config.kd),
            limits=PIDLimits(
                output_min=-math.radians(10),
                output_max=math.radians(10),
                integral_min=-math.radians(5),
                integral_max=math.radians(5),
                rate_limit=geometry.max_steer_rate,
            ),
            filter_params=PIDFilterParams(derivative_filter_coeff=0.2),
            dt=config.dt,
            derivative_mode=DerivativeMode.ON_MEASUREMENT,
            anti_windup_mode=AntiWindupMode.BACK_CALCULATION,
            name="steering_pid",
        )

        # Steering rate limiter
        self._steer_rate_limiter = RateLimiter(
            rate_limit=geometry.max_steer_rate,
            initial_value=0.0,
        )

        # Low-pass filter for steering output
        self._steer_filter = LowPassFilter(
            cutoff_freq=8.0,
            dt=config.dt,
            initial_value=0.0,
        )

        # Speed-dependent gain scheduling
        speed_schedule = config.speed_gain_schedule or {
            "speeds": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            "kp_factors": [1.5, 1.3, 1.1, 1.0, 0.85, 0.7, 0.6, 0.5],
            "kd_factors": [2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5],
        }
        self._kp_schedule = InterpolatingTable(
            x_values=speed_schedule["speeds"],
            y_values=speed_schedule["kp_factors"],
        )
        self._kd_schedule = InterpolatingTable(
            x_values=speed_schedule["speeds"],
            y_values=speed_schedule["kd_factors"],
        )

        # State tracking
        self._prev_lateral_error = 0.0
        self._prev_heading_error = 0.0
        self._integral_lateral_error = 0.0
        self._current_steering_angle = 0.0

    @property
    def mode(self) -> SteeringMode:
        """Return current steering mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: SteeringMode) -> None:
        """Set the steering mode."""
        self._mode = mode

    @property
    def pid(self) -> PIDController:
        """Return the internal PID controller."""
        return self._pid

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Set the PID gains directly.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
        """
        self._pid.set_gains(PIDGains(kp=kp, ki=ki, kd=kd))

    def _compute_lateral_error(
        self,
        vehicle_state: VehicleState,
        closest_point: PathPoint,
    ) -> float:
        """Compute signed lateral error from the path.

        Positive = vehicle is to the left of the path.

        Args:
            vehicle_state: Current vehicle state.
            closest_point: Closest point on the path.

        Returns:
            Signed lateral error in meters.
        """
        dx = vehicle_state.x - closest_point.x
        dy = vehicle_state.y - closest_point.y
        # Transform to path frame
        lateral_error = -dx * math.sin(closest_point.heading) + dy * math.cos(closest_point.heading)
        return lateral_error

    def _compute_heading_error(
        self,
        vehicle_state: VehicleState,
        closest_point: PathPoint,
    ) -> float:
        """Compute heading error (yaw error).

        Args:
            vehicle_state: Current vehicle state.
            closest_point: Closest point on the path.

        Returns:
            Heading error in radians, normalized to [-pi, pi].
        """
        heading_error = vehicle_state.heading - closest_point.heading
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        return heading_error

    def _stanley_steering(
        self,
        heading_error: float,
        lateral_error: float,
        speed: float,
    ) -> float:
        """Compute Stanley method steering angle.

        The Stanley method computes:
            delta = heading_error + arctan(k * e_fa / v)

        Args:
            heading_error: Heading error in radians.
            lateral_error: Lateral error in meters.
            speed: Vehicle speed in m/s.

        Returns:
            Stanley steering angle in radians.
        """
        # Use front axle lateral error (approximate from CG)
        e_fa = lateral_error + self._geometry.cg_to_front * math.sin(heading_error)

        # Stanley cross-track error correction
        speed_safe = max(speed, 1.0)  # Avoid division by zero
        cross_track_correction = math.atan2(
            self._config.stanley_k * e_fa,
            speed_safe,
        )

        steering = heading_error + cross_track_correction
        return steering

    def _pure_pursuit_steering(
        self,
        vehicle_state: VehicleState,
        path_points: list[PathPoint],
        closest_index: int,
    ) -> float:
        """Compute Pure Pursuit steering angle.

        Args:
            vehicle_state: Current vehicle state.
            path_points: List of path points.
            closest_index: Index of closest path point.

        Returns:
            Pure Pursuit steering angle in radians.
        """
        speed = max(vehicle_state.speed, 1.0)
        look_ahead_dist = (
            self._config.pure_pursuit_ld_coeff * speed
            + self._config.pure_pursuit_min_ld
        )

        # Find the look-ahead point
        target_point = self._find_lookahead_point(
            vehicle_state, path_points, closest_index, look_ahead_dist
        )

        if target_point is None:
            return 0.0

        # Transform target to vehicle frame
        dx = target_point.x - vehicle_state.x
        dy = target_point.y - vehicle_state.y

        # Rotate to vehicle heading frame
        local_x = dx * math.cos(vehicle_state.heading) + dy * math.sin(vehicle_state.heading)
        local_y = -dx * math.sin(vehicle_state.heading) + dy * math.cos(vehicle_state.heading)

        # Pure Pursuit formula
        ld_sq = local_x ** 2 + local_y ** 2
        if ld_sq < 0.01:
            return 0.0

        steering = math.atan2(
            2.0 * self._geometry.wheelbase * local_y,
            ld_sq,
        )

        return steering

    def _find_lookahead_point(
        self,
        vehicle_state: VehicleState,
        path_points: list[PathPoint],
        closest_index: int,
        look_ahead_dist: float,
    ) -> Optional[PathPoint]:
        """Find the look-ahead point on the path.

        Args:
            vehicle_state: Current vehicle state.
            path_points: List of path points.
            closest_index: Index of closest path point.
            look_ahead_dist: Desired look-ahead distance in m.

        Returns:
            PathPoint at the look-ahead distance, or None if not found.
        """
        if not path_points:
            return None

        # Search forward from the closest point
        for i in range(closest_index, len(path_points)):
            dx = path_points[i].x - vehicle_state.x
            dy = path_points[i].y - vehicle_state.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist >= look_ahead_dist:
                return path_points[i]

        # Return the last point if look-ahead distance exceeds path length
        return path_points[-1] if path_points else None

    def _compute_curvature_feedforward(
        self,
        curvature: float,
        speed: float,
    ) -> float:
        """Compute feedforward steering from path curvature.

        Args:
            curvature: Path curvature in 1/m.
            speed: Vehicle speed in m/s.

        Returns:
            Feedforward steering angle in radians.
        """
        # Bicycle model feedforward: delta_ff = arctan(L * kappa)
        feedforward = math.atan(self._geometry.wheelbase * curvature)

        # Understeer compensation
        if speed > 1.0:
            lateral_accel = speed ** 2 * curvature
            understeer_correction = self._config.understeer_gain * lateral_accel
            feedforward += understeer_correction

        return feedforward * self._config.curvature_feedforward_gain

    def _limit_by_lateral_accel(
        self,
        steering_angle: float,
        speed: float,
    ) -> float:
        """Limit steering angle based on lateral acceleration constraint.

        Args:
            steering_angle: Desired steering angle in radians.
            speed: Vehicle speed in m/s.

        Returns:
            Limited steering angle in radians.
        """
        if speed < 1.0:
            return steering_angle

        # Approximate lateral acceleration: a_y = v^2 * tan(delta) / L
        max_steer_for_accel = math.atan(
            self._config.max_lateral_accel * self._geometry.wheelbase / (speed ** 2)
        )

        # Use the more restrictive limit
        max_angle = min(
            self._geometry.max_steer_angle,
            max(max_steer_for_accel, math.radians(2.0)),  # Minimum 2 degrees
        )

        return float(np.clip(steering_angle, -max_angle, max_angle))

    def _apply_speed_gain_schedule(self, speed: float) -> None:
        """Apply speed-dependent gain scheduling.

        Args:
            speed: Current vehicle speed in m/s.
        """
        current_gains = self._pid.gains
        kp_factor = self._kp_schedule.lookup(speed)
        kd_factor = self._kd_schedule.lookup(speed)

        new_kp = self._config.kp * kp_factor
        new_kd = self._config.kd * kd_factor

        self._pid.set_gains(PIDGains(
            kp=new_kp,
            ki=self._config.ki,  # Keep ki constant
            kd=new_kd,
        ))

    def update(
        self,
        vehicle_state: VehicleState,
        path_points: list[PathPoint],
        closest_index: int,
        dt: Optional[float] = None,
    ) -> Tuple[float, dict]:
        """Compute steering angle for path tracking.

        Args:
            vehicle_state: Current vehicle state.
            path_points: Reference path points.
            closest_index: Index of the closest path point.
            dt: Optional timestep override.

        Returns:
            Tuple of (steering_angle in radians, diagnostics dict).
        """
        effective_dt = dt if dt is not None else self._config.dt

        if not path_points or closest_index >= len(path_points):
            return 0.0, {"error": "invalid_path"}

        closest_point = path_points[closest_index]

        # Compute errors
        lateral_error = self._compute_lateral_error(vehicle_state, closest_point)
        heading_error = self._compute_heading_error(vehicle_state, closest_point)

        # Apply speed-dependent gain scheduling
        self._apply_speed_gain_schedule(vehicle_state.speed)

        # Compute steering based on mode
        steering_angle = 0.0

        if self._mode == SteeringMode.STANLEY:
            steering_angle = self._stanley_steering(
                heading_error, lateral_error, vehicle_state.speed
            )

        elif self._mode == SteeringMode.PURE_PURSUIT:
            steering_angle = self._pure_pursuit_steering(
                vehicle_state, path_points, closest_index
            )

        elif self._mode == SteeringMode.PID_ONLY:
            pid_output = self._pid.update(
                setpoint=0.0,
                measurement=lateral_error,
                dt=effective_dt,
            )
            steering_angle = heading_error + pid_output

        elif self._mode == SteeringMode.COMBINED:
            # Weighted combination of all methods
            stanley_steer = self._stanley_steering(
                heading_error, lateral_error, vehicle_state.speed
            )
            pp_steer = self._pure_pursuit_steering(
                vehicle_state, path_points, closest_index
            )
            pid_output = self._pid.update(
                setpoint=0.0,
                measurement=lateral_error,
                dt=effective_dt,
            )

            # Speed-dependent weighting
            speed = max(vehicle_state.speed, 1.0)
            w_stanley = max(0.3, 1.0 - speed / 40.0)
            w_pp = min(0.5, speed / 30.0)
            w_pid = 0.2

            # Normalize weights
            total_w = w_stanley + w_pp + w_pid
            w_stanley /= total_w
            w_pp /= total_w
            w_pid /= total_w

            steering_angle = w_stanley * stanley_steer + w_pp * pp_steer + w_pid * pid_output

        # Add curvature feedforward
        feedforward = self._compute_curvature_feedforward(
            closest_point.curvature, vehicle_state.speed
        )
        steering_angle += feedforward

        # Apply lateral acceleration limiting
        steering_angle = self._limit_by_lateral_accel(steering_angle, vehicle_state.speed)

        # Apply steering rate limiting
        steering_angle = self._steer_rate_limiter.update(steering_angle, effective_dt)

        # Apply low-pass filter for smoothness
        steering_angle = self._steer_filter.update(steering_angle, effective_dt)

        # Final saturation
        steering_angle = float(np.clip(
            steering_angle,
            -self._geometry.max_steer_angle,
            self._geometry.max_steer_angle,
        ))

        self._current_steering_angle = steering_angle
        self._prev_lateral_error = lateral_error
        self._prev_heading_error = heading_error

        # Build diagnostics
        diagnostics = {
            "steering_angle_deg": math.degrees(steering_angle),
            "lateral_error_m": lateral_error,
            "heading_error_deg": math.degrees(heading_error),
            "curvature_1m": closest_point.curvature,
            "feedforward_deg": math.degrees(feedforward),
            "mode": self._mode.value,
            "pid_diagnostics": {
                "p_term": self._pid.diagnostics.p_term,
                "i_term": self._pid.diagnostics.i_term,
                "d_term": self._pid.diagnostics.d_term,
                "saturated": self._pid.diagnostics.saturated,
            },
        }

        return steering_angle, diagnostics

    def reset(self) -> None:
        """Reset the steering controller state."""
        self._pid.reset()
        self._steer_rate_limiter.reset()
        self._steer_filter.reset()
        self._prev_lateral_error = 0.0
        self._prev_heading_error = 0.0
        self._integral_lateral_error = 0.0
        self._current_steering_angle = 0.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SteeringPIDController(mode={self._mode.value}, "
            f"kp={self._config.kp}, ki={self._config.ki}, kd={self._config.kd})"
        )
