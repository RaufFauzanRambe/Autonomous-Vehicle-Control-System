"""
Brake PID Controller for Autonomous Vehicle Deceleration Control

Implements brake control with:
  - Deceleration tracking PID
  - ABS (Anti-lock Braking System) integration
  - Emergency braking with maximum deceleration
  - Brake pressure mapping
  - Wheel slip control
  - Regenerative braking coordination (for EVs)
  - Gradient-aware braking

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
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
)
from .controller_utils import RateLimiter, DeadZone, LowPassFilter, Saturate


class BrakeMode(Enum):
    """Brake controller operating mode."""
    NORMAL = "normal"             # Normal deceleration control
    ABS = "abs"                   # Anti-lock braking active
    EMERGENCY = "emergency"       # Emergency stop
    REGENERATIVE = "regenerative"  # Regenerative braking coordination
    HOLD = "hold"                 # Hold brake pressure (e.g., at stop)


@dataclass
class BrakeSystemParams:
    """Brake system physical parameters.

    Attributes:
        max_brake_pressure: Maximum brake hydraulic pressure in bar.
        max_deceleration: Maximum achievable deceleration in m/s^2.
        brake_response_time: Brake system response time in seconds.
        wheelbase: Vehicle wheelbase in m.
        cg_height: Center of gravity height in m.
        front_brake_bias: Front brake bias ratio (0-1).
        num_wheels: Number of braked wheels.
        max_wheel_slip: Maximum allowed wheel slip ratio for ABS.
        abs_cycle_freq: ABS cycling frequency in Hz.
        regen_max_decel: Maximum regenerative deceleration in m/s^2.
        mass: Vehicle mass in kg.
    """
    max_brake_pressure: float = 180.0
    max_deceleration: float = 8.5
    brake_response_time: float = 0.05
    wheelbase: float = 2.85
    cg_height: float = 0.55
    front_brake_bias: float = 0.65
    num_wheels: int = 4
    max_wheel_slip: float = 0.15
    abs_cycle_freq: float = 15.0
    regen_max_decel: float = 3.0
    mass: float = 1800.0


@dataclass
class WheelState:
    """State of an individual wheel for ABS.

    Attributes:
        wheel_speed: Wheel rotational speed converted to linear speed in m/s.
        slip_ratio: Current wheel slip ratio.
        brake_pressure: Brake pressure at this wheel in bar.
        locked: Whether the wheel is locked.
    """
    wheel_speed: float = 0.0
    slip_ratio: float = 0.0
    brake_pressure: float = 0.0
    locked: bool = False


@dataclass
class BrakeControllerConfig:
    """Configuration for the brake PID controller.

    Attributes:
        kp: Proportional gain for deceleration error.
        ki: Integral gain for deceleration error.
        kd: Derivative gain for deceleration error.
        output_min: Minimum PID output (brake pressure in bar).
        output_max: Maximum PID output (brake pressure in bar).
        deadzone_decel: Dead zone for deceleration error in m/s^2.
        abs_enable_threshold: Wheel slip threshold to activate ABS.
        abs_release_threshold: Wheel slip threshold to release ABS.
        emergency_decel: Emergency deceleration target in m/s^2.
        min_speed_for_abs: Minimum speed to enable ABS in m/s.
        dt: Controller timestep in seconds.
    """
    kp: float = 25.0
    ki: float = 5.0
    kd: float = 2.0
    output_min: float = 0.0
    output_max: float = 180.0
    deadzone_decel: float = 0.1
    abs_enable_threshold: float = 0.12
    abs_release_threshold: float = 0.06
    emergency_decel: float = 8.0
    min_speed_for_abs: float = 2.0
    dt: float = 0.01


class BrakePIDController:
    """Brake PID controller with ABS integration for autonomous vehicles.

    This controller provides:
    - Deceleration-tracking PID control
    - ABS wheel slip control to prevent wheel lock
    - Emergency braking with maximum deceleration
    - Brake pressure mapping and distribution
    - Regenerative braking coordination for EVs
    - Weight transfer compensation for gradient braking

    Example:
        >>> config = BrakeControllerConfig()
        >>> params = BrakeSystemParams()
        >>> ctrl = BrakePIDController(config=config, params=params)
        >>> brake_pressure = ctrl.update(
        ...     target_decel=3.0, current_decel=1.5,
        ...     vehicle_speed=20.0, wheel_states=[WheelState()] * 4
        ... )
    """

    def __init__(
        self,
        config: BrakeControllerConfig = BrakeControllerConfig(),
        params: BrakeSystemParams = BrakeSystemParams(),
    ) -> None:
        """Initialize the brake PID controller.

        Args:
            config: Brake controller configuration.
            params: Brake system physical parameters.
        """
        self._config = config
        self._params = params
        self._mode = BrakeMode.NORMAL

        # Deceleration PID controller
        self._pid = PIDController(
            gains=PIDGains(kp=config.kp, ki=config.ki, kd=config.kd),
            limits=PIDLimits(
                output_min=config.output_min,
                output_max=config.output_max,
                integral_min=-50.0,
                integral_max=50.0,
            ),
            filter_params=PIDFilterParams(derivative_filter_coeff=0.2),
            dt=config.dt,
            derivative_mode=DerivativeMode.ON_MEASUREMENT,
            anti_windup_mode=AntiWindupMode.BACK_CALCULATION,
            name="brake_pid",
        )

        # Brake pressure rate limiter
        self._pressure_rate_limiter = RateLimiter(
            rate_limit=5000.0,  # bar/s (fast hydraulic response)
            initial_value=0.0,
        )

        # Dead zone for deceleration error
        self._decel_deadzone = DeadZone(
            lower_limit=-config.deadzone_decel,
            upper_limit=config.deadzone_decel,
        )

        # Low-pass filter for deceleration measurement
        self._decel_filter = LowPassFilter(
            cutoff_freq=10.0,
            dt=config.dt,
            initial_value=0.0,
        )

        # ABS state
        self._abs_active = False
        self._abs_phase = "pressure_build"  # or "pressure_release"
        self._abs_timer = 0.0
        self._abs_duty_cycle = 0.5

        # Brake pressure distribution
        self._front_pressure = 0.0
        self._rear_pressure = 0.0

        # Emergency braking flag
        self._emergency_active = False

        # Hold mode
        self._hold_pressure = 0.0

    @property
    def mode(self) -> BrakeMode:
        """Return current brake mode."""
        return self._mode

    @property
    def abs_active(self) -> bool:
        """Return whether ABS is currently active."""
        return self._abs_active

    @property
    def front_pressure(self) -> float:
        """Return current front brake pressure in bar."""
        return self._front_pressure

    @property
    def rear_pressure(self) -> float:
        """Return current rear brake pressure in bar."""
        return self._rear_pressure

    def _compute_wheel_slip(
        self,
        wheel_speed: float,
        vehicle_speed: float,
    ) -> float:
        """Compute wheel slip ratio.

        Slip ratio = (V_vehicle - V_wheel) / V_vehicle for braking.

        Args:
            wheel_speed: Wheel linear speed in m/s.
            vehicle_speed: Vehicle speed in m/s.

        Returns:
            Slip ratio (positive during braking).
        """
        if abs(vehicle_speed) < 0.5:
            return 0.0
        slip = (vehicle_speed - wheel_speed) / vehicle_speed
        return max(0.0, slip)

    def _check_abs_activation(
        self,
        wheel_states: list[WheelState],
        vehicle_speed: float,
    ) -> bool:
        """Check if ABS should be activated.

        Args:
            wheel_states: List of wheel states.
            vehicle_speed: Vehicle speed in m/s.

        Returns:
            True if ABS should be active.
        """
        if vehicle_speed < self._config.min_speed_for_abs:
            return False

        for wheel in wheel_states:
            if wheel.slip_ratio > self._config.abs_enable_threshold:
                return True
            if wheel.locked:
                return True

        return False

    def _abs_control(
        self,
        base_pressure: float,
        wheel_states: list[WheelState],
        dt: float,
    ) -> float:
        """Apply ABS modulation to brake pressure.

        Implements threshold-based ABS with pressure build/release phases.

        Args:
            base_pressure: Base brake pressure from PID in bar.
            wheel_states: List of wheel states.
            dt: Timestep in seconds.

        Returns:
            ABS-modulated brake pressure in bar.
        """
        abs_period = 1.0 / self._params.abs_cycle_freq
        self._abs_timer += dt

        # Check individual wheel slips
        max_slip = max(w.slip_ratio for w in wheel_states) if wheel_states else 0.0
        any_locked = any(w.locked for w in wheel_states)

        if any_locked or max_slip > self._config.abs_enable_threshold:
            self._abs_phase = "pressure_release"
        elif max_slip < self._config.abs_release_threshold:
            self._abs_phase = "pressure_build"

        if self._abs_phase == "pressure_release":
            # Reduce pressure when slip is too high
            release_factor = 0.5 + 0.5 * (1.0 - min(max_slip / 0.3, 1.0))
            modulated_pressure = base_pressure * release_factor
        else:
            # Build pressure gradually
            ramp_factor = min(1.0, self._abs_timer / (abs_period * 0.5))
            modulated_pressure = base_pressure * ramp_factor

        # Reset timer at end of cycle
        if self._abs_timer >= abs_period:
            self._abs_timer = 0.0

        return max(0.0, modulated_pressure)

    def _distribute_brake_pressure(
        self,
        total_pressure: float,
        deceleration: float,
        grade_angle: float = 0.0,
    ) -> tuple[float, float]:
        """Distribute brake pressure between front and rear axles.

        Accounts for weight transfer during braking.

        Args:
            total_pressure: Total brake pressure in bar.
            deceleration: Current deceleration in m/s^2.
            grade_angle: Road grade angle in radians.

        Returns:
            Tuple of (front_pressure, rear_pressure) in bar.
        """
        if total_pressure <= 0:
            return 0.0, 0.0

        # Weight transfer calculation
        weight = self._params.mass * 9.81
        weight_transfer = (
            self._params.mass * abs(deceleration)
            * self._params.cg_height / self._params.wheelbase
        )

        # Front and rear normal forces
        front_normal = (
            weight * (1.0 - self._params.front_brake_bias) * math.cos(grade_angle)
            + weight_transfer
        )
        rear_normal = (
            weight * self._params.front_brake_bias * math.cos(grade_angle)
            - weight_transfer
        )

        # Ensure non-negative
        front_normal = max(front_normal, 0.0)
        rear_normal = max(rear_normal, 0.0)
        total_normal = front_normal + rear_normal

        if total_normal < 1.0:
            return total_pressure * self._params.front_brake_bias, total_pressure * (1.0 - self._params.front_brake_bias)

        # Distribute pressure proportional to normal force
        front_ratio = front_normal / total_normal
        rear_ratio = rear_normal / total_normal

        # Apply brake bias limits (never go below 40% front for safety)
        front_ratio = max(0.4, min(0.85, front_ratio))
        rear_ratio = 1.0 - front_ratio

        self._front_pressure = total_pressure * front_ratio
        self._rear_pressure = total_pressure * rear_ratio

        return self._front_pressure, self._rear_pressure

    def _emergency_braking(
        self,
        vehicle_speed: float,
        wheel_states: list[WheelState],
        dt: float,
    ) -> float:
        """Compute emergency braking pressure.

        Args:
            vehicle_speed: Vehicle speed in m/s.
            wheel_states: List of wheel states.
            dt: Timestep in seconds.

        Returns:
            Emergency brake pressure in bar.
        """
        # Target maximum deceleration with ABS
        target_pressure = self._params.max_brake_pressure * 0.95

        # Apply ABS if needed
        if vehicle_speed > self._config.min_speed_for_abs:
            if self._check_abs_activation(wheel_states, vehicle_speed):
                self._abs_active = True
                target_pressure = self._abs_control(
                    target_pressure, wheel_states, dt
                )

        return target_pressure

    def update(
        self,
        target_decel: float,
        current_decel: float,
        vehicle_speed: float,
        wheel_states: Optional[list[WheelState]] = None,
        grade_angle: float = 0.0,
        regen_decel_available: float = 0.0,
        dt: Optional[float] = None,
    ) -> dict:
        """Compute brake pressure command for deceleration control.

        Args:
            target_decel: Target deceleration in m/s^2 (positive value).
            current_decel: Current measured deceleration in m/s^2.
            vehicle_speed: Vehicle speed in m/s.
            wheel_states: Optional list of wheel states for ABS.
            grade_angle: Road grade angle in radians.
            regen_decel_available: Available regenerative deceleration in m/s^2.
            dt: Optional timestep override.

        Returns:
            Dictionary with brake commands and diagnostics.
        """
        effective_dt = dt if dt is not None else self._config.dt

        if wheel_states is None:
            wheel_states = [
                WheelState(wheel_speed=vehicle_speed, slip_ratio=0.0)
                for _ in range(self._params.num_wheels)
            ]

        # Update wheel slip ratios
        for wheel in wheel_states:
            wheel.slip_ratio = self._compute_wheel_slip(
                wheel.wheel_speed, vehicle_speed
            )
            wheel.locked = wheel.wheel_speed < 0.5 and vehicle_speed > 2.0

        # Determine mode
        if self._emergency_active:
            self._mode = BrakeMode.EMERGENCY
        elif self._abs_active:
            self._mode = BrakeMode.ABS
        elif regen_decel_available > 0 and target_decel <= self._params.regen_max_decel:
            self._mode = BrakeMode.REGENERATIVE
        elif vehicle_speed < 0.3 and target_decel > 0:
            self._mode = BrakeMode.HOLD
        else:
            self._mode = BrakeMode.NORMAL

        # Handle hold mode
        if self._mode == BrakeMode.HOLD:
            hold_pressure = 20.0  # Minimum hold pressure in bar
            self._hold_pressure = hold_pressure
            front_p, rear_p = self._distribute_brake_pressure(
                hold_pressure, 0.0, grade_angle
            )
            return {
                "total_pressure": hold_pressure,
                "front_pressure": front_p,
                "rear_pressure": rear_p,
                "regen_decel": 0.0,
                "mode": self._mode.value,
                "abs_active": self._abs_active,
            }

        # Handle emergency mode
        if self._mode == BrakeMode.EMERGENCY:
            pressure = self._emergency_braking(
                vehicle_speed, wheel_states, effective_dt
            )
            front_p, rear_p = self._distribute_brake_pressure(
                pressure, self._config.emergency_decel, grade_angle
            )
            return {
                "total_pressure": pressure,
                "front_pressure": front_p,
                "rear_pressure": rear_p,
                "regen_decel": 0.0,
                "mode": self._mode.value,
                "abs_active": self._abs_active,
            }

        # Filter deceleration measurement
        filtered_decel = self._decel_filter.update(current_decel, effective_dt)

        # Regenerative braking coordination
        regen_decel = 0.0
        friction_decel = target_decel

        if self._mode == BrakeMode.REGENERATIVE and regen_decel_available > 0:
            regen_decel = min(target_decel, regen_decel_available)
            friction_decel = max(0.0, target_decel - regen_decel)

        # PID control on friction deceleration
        brake_pressure = self._pid.update(
            setpoint=friction_decel,
            measurement=filtered_decel,
            dt=effective_dt,
        )

        # Ensure non-negative pressure
        brake_pressure = max(0.0, brake_pressure)

        # ABS check and modulation
        self._abs_active = self._check_abs_activation(wheel_states, vehicle_speed)
        if self._abs_active:
            self._mode = BrakeMode.ABS
            brake_pressure = self._abs_control(
                brake_pressure, wheel_states, effective_dt
            )

        # Apply rate limiting
        brake_pressure = self._pressure_rate_limiter.update(
            brake_pressure, effective_dt
        )

        # Final saturation
        brake_pressure = float(np.clip(
            brake_pressure,
            self._config.output_min,
            self._config.output_max,
        ))

        # Distribute pressure
        front_p, rear_p = self._distribute_brake_pressure(
            brake_pressure, current_decel, grade_angle
        )

        return {
            "total_pressure": brake_pressure,
            "front_pressure": front_p,
            "rear_pressure": rear_p,
            "regen_decel": regen_decel,
            "mode": self._mode.value,
            "abs_active": self._abs_active,
        }

    def activate_emergency(self) -> None:
        """Activate emergency braking mode."""
        self._emergency_active = True

    def deactivate_emergency(self) -> None:
        """Deactivate emergency braking mode."""
        self._emergency_active = False
        self._abs_active = False

    def reset(self) -> None:
        """Reset the brake controller state."""
        self._pid.reset()
        self._pressure_rate_limiter.reset()
        self._decel_filter.reset()
        self._abs_active = False
        self._abs_phase = "pressure_build"
        self._abs_timer = 0.0
        self._front_pressure = 0.0
        self._rear_pressure = 0.0
        self._emergency_active = False
        self._hold_pressure = 0.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BrakePIDController(mode={self._mode.value}, "
            f"abs_active={self._abs_active}, "
            f"kp={self._config.kp}, ki={self._config.ki}, kd={self._config.kd})"
        )
