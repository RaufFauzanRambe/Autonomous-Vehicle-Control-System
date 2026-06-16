"""
PID Controller with Anti-Windup, Derivative Filter, and Bumpless Transfer

This module implements a production-grade PID controller suitable for autonomous
vehicle control applications. Key features include:
  - Configurable proportional, integral, and derivative gains
  - Anti-windup via back-calculation and clamping
  - First-order low-pass filter on the derivative term
  - Bumpless transfer between manual and automatic modes
  - Output saturation with tracking integrator
  - Derivative kick elimination (error-on-setpoint option)

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ControllerMode(Enum):
    """Operating mode of the PID controller."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class DerivativeMode(Enum):
    """Derivative computation mode."""
    ON_ERROR = "on_error"        # Derivative on error (standard)
    ON_MEASUREMENT = "on_measurement"  # Derivative on PV (avoids derivative kick)


class AntiWindupMode(Enum):
    """Anti-windup strategy."""
    NONE = "none"
    CLAMPING = "clamping"          # Conditional integration
    BACK_CALCULATION = "back_calculation"  # Back-calculation method


@dataclass
class PIDGains:
    """PID controller gain structure.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain (1/s).
        kd: Derivative gain (s).
    """
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0

    def validate(self) -> None:
        """Validate gains are non-negative."""
        if self.kp < 0:
            raise ValueError(f"Proportional gain kp must be non-negative, got {self.kp}")
        if self.ki < 0:
            raise ValueError(f"Integral gain ki must be non-negative, got {self.ki}")
        if self.kd < 0:
            raise ValueError(f"Derivative gain kd must be non-negative, got {self.kd}")


@dataclass
class PIDLimits:
    """Output and integrator limits.

    Attributes:
        output_min: Minimum controller output.
        output_max: Maximum controller output.
        integral_min: Minimum integrator value (separate from output).
        integral_max: Maximum integrator value.
        rate_limit: Maximum rate of change of output (units/s). 0 = unlimited.
    """
    output_min: float = -100.0
    output_max: float = 100.0
    integral_min: float = -1e6
    integral_max: float = 1e6
    rate_limit: float = 0.0


@dataclass
class PIDFilterParams:
    """Derivative filter parameters.

    Attributes:
        derivative_filter_coeff: Low-pass filter coefficient (0-1).
            0 = no filtering, 1 = full filtering.
        setpoint_filter_coeff: Setpoint reference filter coefficient (0-1).
            0 = no filtering, 1 = full filtering.
    """
    derivative_filter_coeff: float = 0.1
    setpoint_filter_coeff: float = 0.0


@dataclass
class PIDState:
    """Internal state of the PID controller.

    Attributes:
        integral: Current integrator value.
        prev_error: Previous error value.
        prev_measurement: Previous process variable measurement.
        prev_derivative: Previous filtered derivative value.
        prev_output: Previous controller output.
        prev_setpoint: Previous setpoint (for bumpless transfer).
        derivative_filtered: Current filtered derivative term.
    """
    integral: float = 0.0
    prev_error: float = 0.0
    prev_measurement: float = 0.0
    prev_derivative: float = 0.0
    prev_output: float = 0.0
    prev_setpoint: float = 0.0
    derivative_filtered: float = 0.0


@dataclass
class PIDDiagnostics:
    """Diagnostic information from the PID controller.

    Attributes:
        p_term: Proportional term contribution.
        i_term: Integral term contribution.
        d_term: Derivative term contribution.
        error: Current error.
        output_unclamped: Output before saturation.
        output_clamped: Output after saturation.
        saturated: Whether output is saturated.
        integral_windup: Whether integrator is at a limit.
        mode: Current controller mode.
    """
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    error: float = 0.0
    output_unclamped: float = 0.0
    output_clamped: float = 0.0
    saturated: bool = False
    integral_windup: bool = False
    mode: ControllerMode = ControllerMode.AUTOMATIC


class PIDController:
    """Production-grade PID controller for autonomous vehicle applications.

    This controller implements:
    - Standard PID algorithm with configurable gains
    - Anti-windup via clamping or back-calculation
    - First-order low-pass filter on the derivative term
    - Bumpless transfer between manual and automatic modes
    - Rate limiting on the output
    - Derivative kick elimination
    - Setpoint reference filtering

    Example:
        >>> gains = PIDGains(kp=2.0, ki=0.5, kd=0.1)
        >>> limits = PIDLimits(output_min=-10, output_max=10)
        >>> pid = PIDController(gains=gains, limits=limits, dt=0.01)
        >>> output = pid.update(setpoint=100.0, measurement=95.0)
    """

    def __init__(
        self,
        gains: PIDGains = PIDGains(),
        limits: PIDLimits = PIDLimits(),
        filter_params: PIDFilterParams = PIDFilterParams(),
        dt: float = 0.01,
        derivative_mode: DerivativeMode = DerivativeMode.ON_MEASUREMENT,
        anti_windup_mode: AntiWindupMode = AntiWindupMode.BACK_CALCULATION,
        bumpless_transfer: bool = True,
        name: str = "pid_controller",
    ) -> None:
        """Initialize the PID controller.

        Args:
            gains: PID gain structure.
            limits: Output and integrator limits.
            filter_params: Derivative and setpoint filter parameters.
            dt: Controller timestep in seconds.
            derivative_mode: Derivative computation mode.
            anti_windup_mode: Anti-windup strategy.
            bumpless_transfer: Enable bumpless manual/auto transfer.
            name: Controller name for logging.
        """
        gains.validate()
        self._validate_dt(dt)

        self._gains = gains
        self._limits = limits
        self._filter_params = filter_params
        self._dt = dt
        self._derivative_mode = derivative_mode
        self._anti_windup_mode = anti_windup_mode
        self._bumpless_transfer = bumpless_transfer
        self._name = name

        self._state = PIDState()
        self._mode = ControllerMode.AUTOMATIC
        self._manual_output = 0.0
        self._initialized = False
        self._diagnostics = PIDDiagnostics()

        # Back-calculation coefficient
        self._back_calc_gain = 1.0 / max(self._gains.ki, 1e-10) if self._gains.ki > 0 else 0.0

    @staticmethod
    def _validate_dt(dt: float) -> None:
        """Validate the timestep."""
        if dt <= 0:
            raise ValueError(f"Timestep dt must be positive, got {dt}")
        if dt > 10.0:
            raise ValueError(f"Timestep dt seems unreasonably large: {dt}s")

    @property
    def name(self) -> str:
        """Return controller name."""
        return self._name

    @property
    def mode(self) -> ControllerMode:
        """Return current controller mode."""
        return self._mode

    @property
    def gains(self) -> PIDGains:
        """Return current PID gains."""
        return self._gains

    @property
    def state(self) -> PIDState:
        """Return current controller state."""
        return self._state

    @property
    def diagnostics(self) -> PIDDiagnostics:
        """Return latest diagnostics."""
        return self._diagnostics

    @property
    def integral_value(self) -> float:
        """Return current integral term value."""
        return self._state.integral

    def set_gains(self, gains: PIDGains) -> None:
        """Update PID gains dynamically.

        Args:
            gains: New PID gain structure.
        """
        gains.validate()
        self._gains = gains
        self._back_calc_gain = 1.0 / max(self._gains.ki, 1e-10) if self._gains.ki > 0 else 0.0

    def set_mode(self, mode: ControllerMode, manual_output: float = 0.0) -> None:
        """Switch controller mode with bumpless transfer.

        Args:
            mode: Target mode (AUTOMATIC or MANUAL).
            manual_output: Manual output value when switching to MANUAL mode.
        """
        if mode == self._mode:
            return

        if mode == ControllerMode.MANUAL:
            self._manual_output = manual_output
            self._mode = ControllerMode.MANUAL
        else:
            # Switching to AUTO: apply bumpless transfer
            if self._bumpless_transfer and self._gains.ki > 0:
                self._state.integral = self._manual_output / self._gains.ki
                self._clamp_integrator()
            self._mode = ControllerMode.AUTOMATIC
            self._initialized = False  # Force re-initialization

    def reset(self) -> None:
        """Reset the controller state."""
        self._state = PIDState()
        self._initialized = False
        self._diagnostics = PIDDiagnostics()

    def _clamp_integrator(self) -> None:
        """Clamp integrator to its limits."""
        self._state.integral = np.clip(
            self._state.integral,
            self._limits.integral_min,
            self._limits.integral_max,
        )

    def _apply_setpoint_filter(self, setpoint: float) -> float:
        """Apply first-order filter to setpoint reference.

        Args:
            setpoint: Raw setpoint value.

        Returns:
            Filtered setpoint value.
        """
        if self._filter_params.setpoint_filter_coeff <= 0:
            return setpoint

        alpha = self._filter_params.setpoint_filter_coeff
        if not self._initialized:
            return setpoint

        filtered = alpha * setpoint + (1.0 - alpha) * self._state.prev_setpoint
        return filtered

    def _compute_derivative(
        self,
        error: float,
        measurement: float,
    ) -> float:
        """Compute the filtered derivative term.

        Args:
            error: Current error (setpoint - measurement).
            measurement: Current process variable measurement.

        Returns:
            Filtered derivative value.
        """
        if not self._initialized:
            self._state.prev_error = error
            self._state.prev_measurement = measurement
            return 0.0

        if self._derivative_mode == DerivativeMode.ON_ERROR:
            raw_derivative = (error - self._state.prev_error) / self._dt
        else:
            # Derivative on measurement (inverted sign to maintain direction)
            raw_derivative = -(measurement - self._state.prev_measurement) / self._dt

        # Apply first-order low-pass filter
        alpha = self._filter_params.derivative_filter_coeff
        filtered = alpha * raw_derivative + (1.0 - alpha) * self._state.derivative_filtered

        return filtered

    def _apply_anti_windup(
        self,
        error: float,
        output_unclamped: float,
        output_clamped: float,
    ) -> None:
        """Apply anti-windup to the integrator.

        Args:
            error: Current error.
            output_unclamped: Output before saturation.
            output_clamped: Output after saturation.
        """
        if self._anti_windup_mode == AntiWindupMode.NONE:
            # No anti-windup: simple integration
            self._state.integral += self._gains.ki * error * self._dt

        elif self._anti_windup_mode == AntiWindupMode.CLAMPING:
            # Conditional integration: only integrate if not saturated
            # or if integration would reduce saturation
            saturated_high = output_clamped >= self._limits.output_max
            saturated_low = output_clamped <= self._limits.output_min
            error_positive = error > 0
            error_negative = error < 0

            should_integrate = True
            if saturated_high and error_positive:
                should_integrate = False
            if saturated_low and error_negative:
                should_integrate = False

            if should_integrate:
                self._state.integral += self._gains.ki * error * self._dt

        elif self._anti_windup_mode == AntiWindupMode.BACK_CALCULATION:
            # Back-calculation: adjust integrator based on saturation error
            saturation_error = output_clamped - output_unclamped
            if self._gains.ki > 0 and abs(saturation_error) > 1e-10:
                adjustment = (saturation_error / (self._gains.ki * self._dt + 1e-10)) * self._dt
                self._state.integral += self._gains.ki * error * self._dt + adjustment
            else:
                self._state.integral += self._gains.ki * error * self._dt

        self._clamp_integrator()

    def _apply_rate_limit(self, output: float) -> float:
        """Apply rate of change limit to the output.

        Args:
            output: Desired output value.

        Returns:
            Rate-limited output value.
        """
        if self._limits.rate_limit <= 0:
            return output

        if not self._initialized:
            return output

        max_change = self._limits.rate_limit * self._dt
        delta = output - self._state.prev_output
        if abs(delta) > max_change:
            delta = math.copysign(max_change, delta)
        return self._state.prev_output + delta

    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: Optional[float] = None,
        feedforward: float = 0.0,
    ) -> float:
        """Compute the PID controller output.

        Args:
            setpoint: Desired setpoint value.
            measurement: Current process variable measurement.
            dt: Optional override for the timestep. If None, uses init value.
            feedforward: Feedforward term added to the output.

        Returns:
            Controller output value.
        """
        if dt is not None:
            self._validate_dt(dt)
            self._dt = dt

        # Handle manual mode
        if self._mode == ControllerMode.MANUAL:
            self._diagnostics = PIDDiagnostics(mode=ControllerMode.MANUAL)
            self._state.prev_output = self._manual_output
            return self._manual_output

        # Apply setpoint reference filter
        filtered_setpoint = self._apply_setpoint_filter(setpoint)

        # Compute error
        error = filtered_setpoint - measurement

        # Proportional term
        p_term = self._gains.kp * error

        # Derivative term (computed before integral update for state consistency)
        d_filtered = self._compute_derivative(error, measurement)
        d_term = self._gains.kd * d_filtered

        # Integral term (anti-windup applied inside)
        i_term = self._gains.ki * self._state.integral

        # Total output (unclamped)
        output_unclamped = p_term + i_term + d_term + feedforward

        # Apply output saturation
        output_clamped = float(np.clip(
            output_unclamped,
            self._limits.output_min,
            self._limits.output_max,
        ))

        # Apply anti-windup
        self._apply_anti_windup(error, output_unclamped, output_clamped)

        # Re-compute i_term after anti-windup adjustment
        i_term = self._gains.ki * self._state.integral

        # Apply rate limiting
        output_final = self._apply_rate_limit(output_clamped)

        # Update state
        self._state.prev_error = error
        self._state.prev_measurement = measurement
        self._state.prev_derivative = d_filtered
        self._state.derivative_filtered = d_filtered
        self._state.prev_output = output_final
        self._state.prev_setpoint = filtered_setpoint
        self._initialized = True

        # Update diagnostics
        self._diagnostics = PIDDiagnostics(
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            error=error,
            output_unclamped=output_unclamped,
            output_clamped=output_final,
            saturated=(abs(output_clamped - output_unclamped) > 1e-6),
            integral_windup=(
                self._state.integral >= self._limits.integral_max - 1e-6
                or self._state.integral <= self._limits.integral_min + 1e-6
            ),
            mode=ControllerMode.AUTOMATIC,
        )

        return output_final

    def step_response(
        self,
        setpoint: float,
        n_steps: int = 1000,
        plant_gain: float = 1.0,
        plant_tau: float = 1.0,
    ) -> dict:
        """Simulate a step response with a first-order plant.

        Used for quick tuning verification. The plant is modeled as:
            G(s) = plant_gain / (plant_tau * s + 1)

        Args:
            setpoint: Step setpoint value.
            n_steps: Number of simulation steps.
            plant_gain: First-order plant gain.
            plant_tau: First-order plant time constant.

        Returns:
            Dictionary with 'time', 'output', 'setpoint', 'measurement' arrays.
        """
        time_arr = np.zeros(n_steps)
        output_arr = np.zeros(n_steps)
        setpoint_arr = np.full(n_steps, setpoint)
        measurement_arr = np.zeros(n_steps)

        measurement = 0.0
        self.reset()

        for i in range(n_steps):
            output = self.update(setpoint=setpoint, measurement=measurement)
            # First-order plant simulation (Euler integration)
            dmeasurement = (plant_gain * output - measurement) / plant_tau * self._dt
            measurement += dmeasurement

            time_arr[i] = i * self._dt
            output_arr[i] = output
            measurement_arr[i] = measurement

        return {
            "time": time_arr,
            "output": output_arr,
            "setpoint": setpoint_arr,
            "measurement": measurement_arr,
        }

    def get_performance_metrics(
        self,
        time: np.ndarray,
        measurement: np.ndarray,
        setpoint: float,
    ) -> dict:
        """Compute performance metrics from a closed-loop response.

        Args:
            time: Time array.
            measurement: Measurement array.
            setpoint: Constant setpoint value.

        Returns:
            Dictionary with performance metrics.
        """
        error = setpoint - measurement
        abs_error = np.abs(error)

        # Settling time (2% band)
        settling_band = 0.02 * abs(setpoint)
        settled = abs_error <= settling_band
        if np.any(settled):
            settling_idx = np.where(settled)[0][-1]
            # Find first time it stays settled
            for i in range(len(settled)):
                if np.all(settled[i:]):
                    settling_time = time[i]
                    break
            else:
                settling_time = float(time[-1])
        else:
            settling_time = float("inf")

        # Overshoot
        if setpoint >= 0:
            peak = float(np.max(measurement))
            overshoot = max(0, (peak - setpoint) / abs(setpoint) * 100.0) if setpoint != 0 else 0.0
        else:
            valley = float(np.min(measurement))
            overshoot = max(0, (setpoint - valley) / abs(setpoint) * 100.0) if setpoint != 0 else 0.0

        # Steady-state error
        ss_error = float(error[-1]) if len(error) > 0 else 0.0

        # Rise time (10% to 90%)
        low = 0.1 * setpoint
        high = 0.9 * setpoint
        try:
            rise_start_idx = np.where(measurement >= low)[0][0]
            rise_end_idx = np.where(measurement >= high)[0][0]
            rise_time = float(time[rise_end_idx] - time[rise_start_idx])
        except (IndexError, ValueError):
            rise_time = float("inf")

        # IAE, ISE, ITAE
        iae = float(np.trapz(abs_error, time))
        ise = float(np.trapz(error ** 2, time))
        itae = float(np.trapz(time * abs_error, time))

        return {
            "settling_time": settling_time,
            "overshoot_percent": overshoot,
            "steady_state_error": ss_error,
            "rise_time": rise_time,
            "IAE": iae,
            "ISE": ise,
            "ITAE": itae,
        }

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"PIDController(name='{self._name}', "
            f"kp={self._gains.kp}, ki={self._gains.ki}, kd={self._gains.kd}, "
            f"dt={self._dt}, mode={self._mode.value})"
        )
