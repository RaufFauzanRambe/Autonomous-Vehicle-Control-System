"""
Speed PID Controller for Autonomous Vehicle Longitudinal Control

Implements speed control with:
  - Throttle and brake mapping
  - Cruise control mode
  - Grade compensation feedforward
  - Aerodynamic drag compensation
  - Acceleration/deceleration limits
  - Smooth throttle/brake transitions

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .pid_controller import (
    AntiWindupMode,
    DerivativeMode,
    PIDController,
    PIDGains,
    PIDLimits,
    PIDFilterParams,
    ControllerMode,
)
from .controller_utils import RateLimiter, DeadZone, LowPassFilter, Saturate


class SpeedControlMode(Enum):
    """Speed controller operating mode."""
    CRUISE = "cruise"           # Standard cruise control
    ACCELERATE = "accelerate"   # Aggressive acceleration mode
    DECELERATE = "decelerate"   # Controlled deceleration mode
    COAST = "coast"             # No throttle, no brake (coasting)
    EMERGENCY = "emergency"     # Emergency braking override


@dataclass
class VehicleParams:
    """Vehicle physical parameters for feedforward calculations.

    Attributes:
        mass: Vehicle mass in kg.
        frontal_area: Frontal cross-section area in m^2.
        drag_coefficient: Aerodynamic drag coefficient (Cd).
        rolling_resistance: Rolling resistance coefficient.
        wheel_radius: Effective wheel radius in m.
        max_throttle: Maximum throttle command (0-1).
        max_brake: Maximum brake command (0-1).
        max_engine_torque: Maximum engine torque in Nm.
        max_brake_torque: Maximum brake torque in Nm.
        transmission_ratio: Overall transmission ratio.
        air_density: Air density in kg/m^3.
        gravity: Gravitational acceleration in m/s^2.
    """
    mass: float = 1800.0
    frontal_area: float = 2.5
    drag_coefficient: float = 0.3
    rolling_resistance: float = 0.015
    wheel_radius: float = 0.33
    max_throttle: float = 1.0
    max_brake: float = 1.0
    max_engine_torque: float = 350.0
    max_brake_torque: float = 12000.0
    transmission_ratio: float = 4.5
    air_density: float = 1.225
    gravity: float = 9.81


@dataclass
class SpeedControllerConfig:
    """Configuration for the speed PID controller.

    Attributes:
        kp: Proportional gain for speed error.
        ki: Integral gain for speed error.
        kd: Derivative gain for speed error.
        output_min: Minimum PID output.
        output_max: Maximum PID output.
        throttle_deadzone: Dead zone for throttle activation.
        brake_deadzone: Dead zone for brake activation.
        throttle_brake_overlap: Overlap region where both can be active.
        max_acceleration: Maximum allowed acceleration in m/s^2.
        max_deceleration: Maximum allowed deceleration in m/s^2.
        coast_deceleration: Deceleration threshold for coasting mode.
        dt: Controller timestep in seconds.
    """
    kp: float = 1.5
    ki: float = 0.3
    kd: float = 0.05
    output_min: float = -1.0
    output_max: float = 1.0
    throttle_deadzone: float = 0.02
    brake_deadzone: float = 0.02
    throttle_brake_overlap: float = 0.01
    max_acceleration: float = 3.0
    max_deceleration: float = -6.0
    coast_deceleration: float = -0.3
    dt: float = 0.01


@dataclass
class SpeedControllerState:
    """Internal state of the speed controller.

    Attributes:
        current_mode: Current control mode.
        throttle_command: Current throttle command (0-1).
        brake_command: Current brake command (0-1).
        speed_error: Current speed error.
        acceleration: Estimated longitudinal acceleration.
        feedforward_torque: Current feedforward torque.
        grade_angle: Current road grade angle.
    """
    current_mode: SpeedControlMode = SpeedControlMode.CRUISE
    throttle_command: float = 0.0
    brake_command: float = 0.0
    speed_error: float = 0.0
    acceleration: float = 0.0
    feedforward_torque: float = 0.0
    grade_angle: float = 0.0


class SpeedPIDController:
    """Speed PID controller with throttle/brake mapping for autonomous vehicles.

    This controller takes a target speed and current speed, computes the
    appropriate throttle and brake commands using PID control with feedforward
    compensation for aerodynamic drag, rolling resistance, and road grade.

    The output of the internal PID is a normalized torque demand (-1 to 1),
    where positive values map to throttle and negative values map to brake.

    Example:
        >>> config = SpeedControllerConfig()
        >>> vehicle = VehicleParams()
        >>> ctrl = SpeedPIDController(config=config, vehicle=vehicle)
        >>> throttle, brake = ctrl.update(
        ...     target_speed=25.0, current_speed=20.0, grade_angle=0.0
        ... )
    """

    def __init__(
        self,
        config: SpeedControllerConfig = SpeedControllerConfig(),
        vehicle: VehicleParams = VehicleParams(),
    ) -> None:
        """Initialize the speed PID controller.

        Args:
            config: Speed controller configuration.
            vehicle: Vehicle physical parameters.
        """
        self._config = config
        self._vehicle = vehicle
        self._state = SpeedControllerState()

        # Internal PID controller
        self._pid = PIDController(
            gains=PIDGains(kp=config.kp, ki=config.ki, kd=config.kd),
            limits=PIDLimits(
                output_min=config.output_min,
                output_max=config.output_max,
                integral_min=-0.5,
                integral_max=0.5,
            ),
            filter_params=PIDFilterParams(derivative_filter_coeff=0.15),
            dt=config.dt,
            derivative_mode=DerivativeMode.ON_MEASUREMENT,
            anti_windup_mode=AntiWindupMode.BACK_CALCULATION,
            name="speed_pid",
        )

        # Rate limiters for smooth transitions
        self._throttle_rate_limiter = RateLimiter(
            rate_limit=2.0,  # throttle units/s
            initial_value=0.0,
        )
        self._brake_rate_limiter = RateLimiter(
            rate_limit=5.0,  # brake units/s
            initial_value=0.0,
        )

        # Dead zones
        self._throttle_deadzone = DeadZone(
            lower_limit=-config.throttle_deadzone,
            upper_limit=config.throttle_deadzone,
        )
        self._brake_deadzone = DeadZone(
            lower_limit=-config.brake_deadzone,
            upper_limit=config.brake_deadzone,
        )

        # Low-pass filter for speed measurement
        self._speed_filter = LowPassFilter(
            cutoff_freq=5.0,
            dt=config.dt,
            initial_value=0.0,
        )

        # Acceleration estimator
        self._prev_speed = 0.0
        self._accel_filter = LowPassFilter(
            cutoff_freq=2.0,
            dt=config.dt,
            initial_value=0.0,
        )

    @property
    def state(self) -> SpeedControllerState:
        """Return current controller state."""
        return self._state

    @property
    def pid(self) -> PIDController:
        """Return the internal PID controller for tuning access."""
        return self._pid

    def set_mode(self, mode: SpeedControlMode) -> None:
        """Set the speed control mode.

        Args:
            mode: Desired speed control mode.
        """
        self._state.current_mode = mode
        if mode == SpeedControlMode.COAST:
            self._state.throttle_command = 0.0
            self._state.brake_command = 0.0
        elif mode == SpeedControlMode.EMERGENCY:
            self._state.throttle_command = 0.0
            self._state.brake_command = self._vehicle.max_brake

    def _compute_aero_drag_force(self, speed: float) -> float:
        """Compute aerodynamic drag force.

        Args:
            speed: Vehicle speed in m/s.

        Returns:
            Aerodynamic drag force in N.
        """
        drag_force = (
            0.5 * self._vehicle.air_density
            * self._vehicle.drag_coefficient
            * self._vehicle.frontal_area
            * speed ** 2
        )
        return drag_force

    def _compute_rolling_resistance_force(self, speed: float, grade_angle: float = 0.0) -> float:
        """Compute rolling resistance force.

        Args:
            speed: Vehicle speed in m/s.
            grade_angle: Road grade angle in radians.

        Returns:
            Rolling resistance force in N.
        """
        normal_force = self._vehicle.mass * self._vehicle.gravity * np.cos(grade_angle)
        rolling_force = self._vehicle.rolling_resistance * normal_force
        return rolling_force if speed > 0.1 else 0.0

    def _compute_grade_force(self, grade_angle: float) -> float:
        """Compute gravitational force component along the road.

        Args:
            grade_angle: Road grade angle in radians (positive = uphill).

        Returns:
            Grade resistance force in N (positive when going uphill).
        """
        return self._vehicle.mass * self._vehicle.gravity * np.sin(grade_angle)

    def _compute_feedforward_torque(
        self,
        speed: float,
        grade_angle: float,
    ) -> float:
        """Compute feedforward torque for disturbance compensation.

        Args:
            speed: Vehicle speed in m/s.
            grade_angle: Road grade angle in radians.

        Returns:
            Feedforward torque in Nm.
        """
        aero_force = self._compute_aero_drag_force(speed)
        rolling_force = self._compute_rolling_resistance_force(speed, grade_angle)
        grade_force = self._compute_grade_force(grade_angle)

        total_resistance = aero_force + rolling_force + grade_force
        feedforward_torque = total_resistance * self._vehicle.wheel_radius / self._vehicle.transmission_ratio

        return feedforward_torque

    def _torque_to_throttle(self, torque_demand: float) -> float:
        """Map positive torque demand to throttle command.

        Args:
            torque_demand: Positive torque demand in Nm.

        Returns:
            Throttle command (0-1).
        """
        max_wheel_torque = (
            self._vehicle.max_engine_torque
            * self._vehicle.transmission_ratio
            / self._vehicle.wheel_radius
        )
        throttle = torque_demand / max_wheel_torque if max_wheel_torque > 0 else 0.0
        return float(np.clip(throttle, 0.0, self._vehicle.max_throttle))

    def _torque_to_brake(self, torque_demand: float) -> float:
        """Map negative torque demand to brake command.

        Args:
            torque_demand: Negative torque demand in Nm (absolute value used).

        Returns:
            Brake command (0-1).
        """
        max_brake_force = self._vehicle.max_brake_torque / self._vehicle.wheel_radius
        brake_force = abs(torque_demand) / self._vehicle.wheel_radius
        brake = brake_force / max_brake_force if max_brake_force > 0 else 0.0
        return float(np.clip(brake, 0.0, self._vehicle.max_brake))

    def _map_pid_output_to_commands(
        self,
        pid_output: float,
        feedforward_torque: float,
    ) -> tuple[float, float]:
        """Map the normalized PID output to throttle and brake commands.

        The PID output is in [-1, 1], representing normalized torque demand.
        Positive values are mapped to throttle, negative to brake.

        Args:
            pid_output: PID controller output.
            feedforward_torque: Feedforward torque demand in Nm.

        Returns:
            Tuple of (throttle_command, brake_command), both in [0, 1].
        """
        throttle = 0.0
        brake = 0.0

        # Convert normalized PID output to torque
        max_engine_torque = self._vehicle.max_engine_torque
        torque_demand = pid_output * max_engine_torque + feedforward_torque

        if torque_demand > 0:
            throttle = self._torque_to_throttle(torque_demand)
            # Apply dead zone
            if throttle < self._config.throttle_deadzone:
                throttle = 0.0
        else:
            brake = self._torque_to_brake(torque_demand)
            # Apply dead zone
            if brake < self._config.brake_deadzone:
                brake = 0.0

        # Apply rate limiting
        throttle = self._throttle_rate_limiter.update(throttle, self._config.dt)
        brake = self._brake_rate_limiter.update(brake, self._config.dt)

        return throttle, brake

    def _determine_mode(self, speed_error: float, current_speed: float) -> SpeedControlMode:
        """Determine the appropriate speed control mode.

        Args:
            speed_error: Speed error (target - current).
            current_speed: Current vehicle speed.

        Returns:
            Appropriate speed control mode.
        """
        if self._state.current_mode == SpeedControlMode.EMERGENCY:
            return SpeedControlMode.EMERGENCY

        if current_speed < 0.5:
            return SpeedControlMode.ACCELERATE if speed_error > 0 else SpeedControlMode.COAST

        if speed_error > 2.0:
            return SpeedControlMode.ACCELERATE
        elif speed_error < -2.0:
            return SpeedControlMode.DECELERATE
        else:
            return SpeedControlMode.CRUISE

    def update(
        self,
        target_speed: float,
        current_speed: float,
        grade_angle: float = 0.0,
        dt: Optional[float] = None,
        external_accel_demand: float = 0.0,
    ) -> tuple[float, float]:
        """Compute throttle and brake commands for speed control.

        Args:
            target_speed: Desired speed in m/s.
            current_speed: Current vehicle speed in m/s.
            grade_angle: Road grade angle in radians (positive = uphill).
            dt: Optional timestep override.
            external_accel_demand: External acceleration demand for
                coordinated control (m/s^2).

        Returns:
            Tuple of (throttle_command, brake_command), both in [0, 1].
        """
        effective_dt = dt if dt is not None else self._config.dt

        # Filter speed measurement
        filtered_speed = self._speed_filter.update(current_speed, effective_dt)

        # Estimate acceleration
        if effective_dt > 0:
            raw_accel = (filtered_speed - self._prev_speed) / effective_dt
            self._state.acceleration = self._accel_filter.update(raw_accel, effective_dt)
        self._prev_speed = filtered_speed

        # Compute feedforward torque
        feedforward_torque = self._compute_feedforward_torque(filtered_speed, grade_angle)
        self._state.feedforward_torque = feedforward_torque
        self._state.grade_angle = grade_angle

        # Normalize feedforward for PID
        max_engine_torque = self._vehicle.max_engine_torque
        feedforward_normalized = feedforward_torque / max_engine_torque if max_engine_torque > 0 else 0.0

        # Compute speed error
        speed_error = target_speed - filtered_speed
        self._state.speed_error = speed_error

        # Determine control mode
        self._state.current_mode = self._determine_mode(speed_error, filtered_speed)

        # Handle coast mode
        if self._state.current_mode == SpeedControlMode.COAST:
            self._state.throttle_command = 0.0
            self._state.brake_command = 0.0
            return 0.0, 0.0

        # Handle emergency mode
        if self._state.current_mode == SpeedControlMode.EMERGENCY:
            self._state.throttle_command = 0.0
            self._state.brake_command = self._vehicle.max_brake
            return 0.0, self._state.brake_command

        # PID update with feedforward
        pid_output = self._pid.update(
            setpoint=target_speed,
            measurement=filtered_speed,
            dt=effective_dt,
            feedforward=feedforward_normalized,
        )

        # Map to throttle/brake
        throttle, brake = self._map_pid_output_to_commands(pid_output, feedforward_torque)

        # Limit acceleration/deceleration
        if self._state.acceleration > self._config.max_acceleration and throttle > self._state.throttle_command:
            throttle = self._state.throttle_command  # Don't increase throttle
        if self._state.acceleration < self._config.max_deceleration and brake > self._state.brake_command:
            pass  # Allow braking even during high deceleration for safety

        self._state.throttle_command = throttle
        self._state.brake_command = brake

        return throttle, brake

    def reset(self) -> None:
        """Reset the speed controller state."""
        self._pid.reset()
        self._state = SpeedControllerState()
        self._prev_speed = 0.0
        self._throttle_rate_limiter.reset()
        self._brake_rate_limiter.reset()
        self._speed_filter.reset()
        self._accel_filter.reset()

    def get_diagnostics(self) -> dict:
        """Return comprehensive diagnostic information.

        Returns:
            Dictionary with controller diagnostics.
        """
        pid_diag = self._pid.diagnostics
        return {
            "mode": self._state.current_mode.value,
            "throttle": self._state.throttle_command,
            "brake": self._state.brake_command,
            "speed_error": self._state.speed_error,
            "acceleration": self._state.acceleration,
            "feedforward_torque": self._state.feedforward_torque,
            "grade_angle_deg": np.degrees(self._state.grade_angle),
            "pid_p_term": pid_diag.p_term,
            "pid_i_term": pid_diag.i_term,
            "pid_d_term": pid_diag.d_term,
            "pid_saturated": pid_diag.saturated,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpeedPIDController(mode={self._state.current_mode.value}, "
            f"kp={self._config.kp}, ki={self._config.ki}, kd={self._config.kd})"
        )
