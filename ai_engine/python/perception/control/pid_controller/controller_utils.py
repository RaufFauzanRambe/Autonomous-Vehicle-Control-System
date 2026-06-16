"""
Controller Utility Functions for Autonomous Vehicle Control

Provides common signal processing and control utilities:
  - Saturation (clamping)
  - Rate limiter
  - Dead zone
  - Low-pass and high-pass filters
  - Moving average filter
  - Interpolating lookup table
  - Signal normalization
  - Wrap-to-pi angle normalization
  - First-order lag element
  - Lead-lag compensator

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


class Saturate:
    """Saturation (clamping) utility."""

    @staticmethod
    def apply(value: float, lower: float, upper: float) -> float:
        """Saturate a value between lower and upper limits.

        Args:
            value: Input value.
            lower: Lower limit.
            upper: Upper limit.

        Returns:
            Clamped value.
        """
        return float(np.clip(value, lower, upper))

    @staticmethod
    def is_saturated(value: float, lower: float, upper: float, tol: float = 1e-6) -> bool:
        """Check if a value is at a saturation limit.

        Args:
            value: Input value.
            lower: Lower limit.
            upper: Upper limit.
            tol: Tolerance for saturation detection.

        Returns:
            True if the value is at either limit.
        """
        return value <= lower + tol or value >= upper - tol


class RateLimiter:
    """Rate of change limiter for smooth signal transitions.

    Limits how fast a signal can change per unit time.

    Example:
        >>> rl = RateLimiter(rate_limit=5.0, initial_value=0.0)
        >>> output = rl.update(10.0, dt=0.1)  # Limited to 0.5 change
    """

    def __init__(
        self,
        rate_limit: float,
        initial_value: float = 0.0,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            rate_limit: Maximum rate of change (units/second). 0 = no limit.
            initial_value: Initial output value.
        """
        self._rate_limit = rate_limit
        self._prev_value = initial_value

    @property
    def rate_limit(self) -> float:
        """Return current rate limit."""
        return self._rate_limit

    @rate_limit.setter
    def rate_limit(self, value: float) -> None:
        """Set the rate limit."""
        self._rate_limit = max(0.0, value)

    def update(self, input_value: float, dt: float) -> float:
        """Apply rate limiting to the input.

        Args:
            input_value: Desired input value.
            dt: Timestep in seconds.

        Returns:
            Rate-limited output value.
        """
        if dt <= 0:
            return self._prev_value

        if self._rate_limit <= 0:
            self._prev_value = input_value
            return input_value

        max_change = self._rate_limit * dt
        delta = input_value - self._prev_value

        if abs(delta) > max_change:
            delta = math.copysign(max_change, delta)

        output = self._prev_value + delta
        self._prev_value = output
        return output

    def reset(self, value: float = 0.0) -> None:
        """Reset the rate limiter state.

        Args:
            value: Value to reset to.
        """
        self._prev_value = value


class DeadZone:
    """Dead zone utility for eliminating small signal noise.

    Values within the dead zone are output as zero. Values outside
    are shifted by the dead zone width.

    Example:
        >>> dz = DeadZone(lower_limit=-0.1, upper_limit=0.1)
        >>> dz.apply(0.05)  # Returns 0.0 (within dead zone)
        >>> dz.apply(0.3)   # Returns 0.2 (shifted by upper limit)
    """

    def __init__(
        self,
        lower_limit: float = -0.1,
        upper_limit: float = 0.1,
    ) -> None:
        """Initialize the dead zone.

        Args:
            lower_limit: Lower limit of the dead zone.
            upper_limit: Upper limit of the dead zone.

        Raises:
            ValueError: If lower_limit > upper_limit.
        """
        if lower_limit > upper_limit:
            raise ValueError(
                f"lower_limit ({lower_limit}) must be <= upper_limit ({upper_limit})"
            )
        self._lower = lower_limit
        self._upper = upper_limit

    def apply(self, value: float) -> float:
        """Apply dead zone to the input value.

        Args:
            value: Input value.

        Returns:
            Value with dead zone applied.
        """
        if value > self._upper:
            return value - self._upper
        elif value < self._lower:
            return value - self._lower
        else:
            return 0.0

    def apply_no_shift(self, value: float) -> float:
        """Apply dead zone without shifting (output is clamped to zero in zone).

        Args:
            value: Input value.

        Returns:
            Zero if in dead zone, original value otherwise.
        """
        if self._lower <= value <= self._upper:
            return 0.0
        return value

    @property
    def width(self) -> float:
        """Return the dead zone width."""
        return self._upper - self._lower


class LowPassFilter:
    """First-order low-pass filter (exponential smoothing).

    Implements: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    where alpha = dt / (tau + dt), tau = 1 / (2 * pi * fc)

    Example:
        >>> lpf = LowPassFilter(cutoff_freq=5.0, dt=0.01)
        >>> filtered = lpf.update(raw_value, dt=0.01)
    """

    def __init__(
        self,
        cutoff_freq: float,
        dt: float = 0.01,
        initial_value: float = 0.0,
    ) -> None:
        """Initialize the low-pass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            dt: Default timestep in seconds.
            initial_value: Initial filter output.

        Raises:
            ValueError: If cutoff_freq is not positive.
        """
        if cutoff_freq <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff_freq}")
        self._cutoff_freq = cutoff_freq
        self._default_dt = dt
        self._output = initial_value
        self._initialized = False

    @property
    def cutoff_freq(self) -> float:
        """Return current cutoff frequency."""
        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, freq: float) -> None:
        """Update the cutoff frequency."""
        if freq <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {freq}")
        self._cutoff_freq = freq

    def _compute_alpha(self, dt: float) -> float:
        """Compute filter coefficient.

        Args:
            dt: Timestep in seconds.

        Returns:
            Filter coefficient alpha.
        """
        tau = 1.0 / (2.0 * math.pi * self._cutoff_freq)
        return dt / (tau + dt)

    def update(self, input_value: float, dt: Optional[float] = None) -> float:
        """Apply low-pass filter to input.

        Args:
            input_value: Raw input value.
            dt: Optional timestep override.

        Returns:
            Filtered output value.
        """
        effective_dt = dt if dt is not None else self._default_dt

        if not self._initialized:
            self._output = input_value
            self._initialized = True
            return self._output

        alpha = self._compute_alpha(effective_dt)
        self._output = alpha * input_value + (1.0 - alpha) * self._output
        return self._output

    def reset(self, value: float = 0.0) -> None:
        """Reset the filter state.

        Args:
            value: Value to reset to.
        """
        self._output = value
        self._initialized = False


class HighPassFilter:
    """First-order high-pass filter.

    Implements: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
    where alpha = RC / (RC + dt), RC = 1 / (2 * pi * fc)
    """

    def __init__(
        self,
        cutoff_freq: float,
        dt: float = 0.01,
        initial_value: float = 0.0,
    ) -> None:
        """Initialize the high-pass filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz.
            dt: Default timestep in seconds.
            initial_value: Initial filter output.
        """
        if cutoff_freq <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff_freq}")
        self._cutoff_freq = cutoff_freq
        self._default_dt = dt
        self._output = initial_value
        self._prev_input = initial_value
        self._initialized = False

    def update(self, input_value: float, dt: Optional[float] = None) -> float:
        """Apply high-pass filter to input.

        Args:
            input_value: Raw input value.
            dt: Optional timestep override.

        Returns:
            Filtered output value.
        """
        effective_dt = dt if dt is not None else self._default_dt

        if not self._initialized:
            self._prev_input = input_value
            self._output = 0.0
            self._initialized = True
            return 0.0

        tau = 1.0 / (2.0 * math.pi * self._cutoff_freq)
        alpha = tau / (tau + effective_dt)
        self._output = alpha * (self._output + input_value - self._prev_input)
        self._prev_input = input_value
        return self._output

    def reset(self) -> None:
        """Reset the filter state."""
        self._output = 0.0
        self._prev_input = 0.0
        self._initialized = False


class MovingAverageFilter:
    """Moving average filter for signal smoothing.

    Maintains a fixed-length window and computes the arithmetic mean.
    """

    def __init__(self, window_size: int = 5, initial_value: float = 0.0) -> None:
        """Initialize the moving average filter.

        Args:
            window_size: Number of samples in the averaging window.
            initial_value: Initial value to fill the buffer.

        Raises:
            ValueError: If window_size < 1.
        """
        if window_size < 1:
            raise ValueError(f"Window size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._buffer: List[float] = [initial_value] * window_size
        self._index = 0

    def update(self, input_value: float) -> float:
        """Add a new sample and compute the moving average.

        Args:
            input_value: New input value.

        Returns:
            Current moving average.
        """
        self._buffer[self._index] = input_value
        self._index = (self._index + 1) % self._window_size
        return sum(self._buffer) / len(self._buffer)

    def reset(self, initial_value: float = 0.0) -> None:
        """Reset the filter buffer.

        Args:
            initial_value: Value to fill the buffer with.
        """
        self._buffer = [initial_value] * self._window_size
        self._index = 0


class InterpolatingTable:
    """1D lookup table with linear interpolation.

    Used for gain scheduling and map-based calibration.

    Example:
        >>> table = InterpolatingTable(
        ...     x_values=[0, 10, 20, 30],
        ...     y_values=[1.0, 0.8, 0.6, 0.4]
        ... )
        >>> table.lookup(15.0)  # Returns 0.7
    """

    def __init__(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        extrapolate: bool = False,
    ) -> None:
        """Initialize the interpolating table.

        Args:
            x_values: X-axis breakpoints (must be sorted ascending).
            y_values: Y-axis values at each breakpoint.
            extrapolate: If True, extrapolate beyond table range.
                If False, hold the endpoint value.

        Raises:
            ValueError: If x_values and y_values have different lengths.
            ValueError: If fewer than 2 breakpoints are provided.
        """
        if len(x_values) != len(y_values):
            raise ValueError(
                f"x_values ({len(x_values)}) and y_values ({len(y_values)}) "
                f"must have the same length"
            )
        if len(x_values) < 2:
            raise ValueError("At least 2 breakpoints are required")

        self._x = list(x_values)
        self._y = list(y_values)
        self._extrapolate = extrapolate

        # Verify sorted
        for i in range(len(self._x) - 1):
            if self._x[i] > self._x[i + 1]:
                raise ValueError(
                    f"x_values must be sorted ascending. "
                    f"Found {self._x[i]} > {self._x[i+1]} at index {i}"
                )

    def lookup(self, x: float) -> float:
        """Look up a value with linear interpolation.

        Args:
            x: Input value to look up.

        Returns:
            Interpolated output value.
        """
        # Below lower bound
        if x <= self._x[0]:
            if self._extrapolate and len(self._x) >= 2:
                slope = (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])
                return self._y[0] + slope * (x - self._x[0])
            return self._y[0]

        # Above upper bound
        if x >= self._x[-1]:
            if self._extrapolate and len(self._x) >= 2:
                n = len(self._x) - 1
                slope = (self._y[n] - self._y[n - 1]) / (self._x[n] - self._x[n - 1])
                return self._y[n] + slope * (x - self._x[n])
            return self._y[-1]

        # Find interval and interpolate
        for i in range(len(self._x) - 1):
            if self._x[i] <= x <= self._x[i + 1]:
                t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
                return self._y[i] + t * (self._y[i + 1] - self._y[i])

        return self._y[-1]

    @property
    def x_values(self) -> List[float]:
        """Return x-axis breakpoints."""
        return self._x.copy()

    @property
    def y_values(self) -> List[float]:
        """Return y-axis values."""
        return self._y.copy()


class FirstOrderLag:
    """First-order lag element (PT1).

    Implements: G(s) = K / (T * s + 1)
    Discrete: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    where alpha = dt / (T + dt)
    """

    def __init__(
        self,
        gain: float = 1.0,
        time_constant: float = 1.0,
        dt: float = 0.01,
        initial_value: float = 0.0,
    ) -> None:
        """Initialize the first-order lag.

        Args:
            gain: Static gain K.
            time_constant: Time constant T in seconds.
            dt: Default timestep in seconds.
            initial_value: Initial output value.
        """
        self._gain = gain
        self._time_constant = max(time_constant, 1e-10)
        self._default_dt = dt
        self._output = initial_value

    def update(self, input_value: float, dt: Optional[float] = None) -> float:
        """Apply first-order lag to input.

        Args:
            input_value: Input signal value.
            dt: Optional timestep override.

        Returns:
            Filtered output value.
        """
        effective_dt = dt if dt is not None else self._default_dt
        alpha = effective_dt / (self._time_constant + effective_dt)
        self._output = alpha * self._gain * input_value + (1.0 - alpha) * self._output
        return self._output

    def reset(self, value: float = 0.0) -> None:
        """Reset the lag element."""
        self._output = value


class LeadLagCompensator:
    """Lead-lag compensator for frequency-dependent gain shaping.

    Implements a discrete lead-lag compensator:
        G(s) = K * (T_z * s + 1) / (T_p * s + 1)

    When T_z > T_p: lead compensator (phase advance)
    When T_z < T_p: lag compensator (phase delay)
    """

    def __init__(
        self,
        gain: float = 1.0,
        t_zero: float = 0.1,
        t_pole: float = 0.01,
        dt: float = 0.01,
    ) -> None:
        """Initialize the lead-lag compensator.

        Args:
            gain: Static gain K.
            t_zero: Zero time constant T_z in seconds.
            t_pole: Pole time constant T_p in seconds.
            dt: Default timestep in seconds.
        """
        self._gain = gain
        self._t_zero = max(t_zero, 1e-10)
        self._t_pole = max(t_pole, 1e-10)
        self._default_dt = dt
        self._prev_input = 0.0
        self._prev_output = 0.0

    def update(self, input_value: float, dt: Optional[float] = None) -> float:
        """Apply lead-lag compensator to input.

        Uses bilinear (Tustin) transform for discretization.

        Args:
            input_value: Input signal value.
            dt: Optional timestep override.

        Returns:
            Compensated output value.
        """
        effective_dt = dt if dt is not None else self._default_dt

        # Bilinear transform coefficients
        a = 2.0 * self._t_zero / effective_dt
        b = 2.0 * self._t_pole / effective_dt

        numerator = self._gain * ((a + 1) * input_value + (a - 1) * self._prev_input)
        denominator = b + 1
        past_term = (b - 1) * self._prev_output

        output = (numerator - past_term) / denominator

        self._prev_input = input_value
        self._prev_output = output
        return output

    def reset(self) -> None:
        """Reset the compensator state."""
        self._prev_input = 0.0
        self._prev_output = 0.0


class DerivativeFilter:
    """Filtered derivative computation.

    Computes the derivative of a signal with built-in low-pass filtering
    to reduce noise amplification.
    """

    def __init__(
        self,
        filter_coeff: float = 0.1,
        dt: float = 0.01,
    ) -> None:
        """Initialize the derivative filter.

        Args:
            filter_coeff: Filter coefficient (0-1). Higher = more filtering.
            dt: Default timestep in seconds.
        """
        self._filter_coeff = max(0.0, min(1.0, filter_coeff))
        self._default_dt = dt
        self._prev_input = 0.0
        self._filtered_derivative = 0.0
        self._initialized = False

    def update(self, input_value: float, dt: Optional[float] = None) -> float:
        """Compute filtered derivative of input.

        Args:
            input_value: Input signal value.
            dt: Optional timestep override.

        Returns:
            Filtered derivative value.
        """
        effective_dt = dt if dt is not None else self._default_dt

        if not self._initialized or effective_dt <= 0:
            self._prev_input = input_value
            self._initialized = True
            return 0.0

        raw_derivative = (input_value - self._prev_input) / effective_dt
        alpha = self._filter_coeff
        self._filtered_derivative = (
            alpha * raw_derivative + (1.0 - alpha) * self._filtered_derivative
        )
        self._prev_input = input_value
        return self._filtered_derivative

    def reset(self) -> None:
        """Reset the derivative filter."""
        self._prev_input = 0.0
        self._filtered_derivative = 0.0
        self._initialized = False


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the range [-pi, pi].

    Args:
        angle: Input angle in radians.

    Returns:
        Wrapped angle in [-pi, pi].
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_to_2pi(angle: float) -> float:
    """Wrap an angle to the range [0, 2*pi).

    Args:
        angle: Input angle in radians.

    Returns:
        Wrapped angle in [0, 2*pi).
    """
    return angle % (2.0 * math.pi)


def normalize_signal(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = -1.0,
    target_max: float = 1.0,
) -> float:
    """Normalize a signal from [min_val, max_val] to [target_min, target_max].

    Args:
        value: Input value.
        min_val: Input range minimum.
        max_val: Input range maximum.
        target_min: Target range minimum.
        target_max: Target range maximum.

    Returns:
        Normalized value.
    """
    if abs(max_val - min_val) < 1e-10:
        return (target_min + target_max) / 2.0
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def linear_interp(
    x: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> float:
    """Simple linear interpolation between two points.

    Args:
        x: Input x value.
        x0: First x breakpoint.
        x1: Second x breakpoint.
        y0: First y value.
        y1: Second y value.

    Returns:
        Interpolated y value.
    """
    if abs(x1 - x0) < 1e-10:
        return (y0 + y1) / 2.0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
