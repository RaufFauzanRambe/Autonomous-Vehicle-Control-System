"""
Gain Scheduler for Autonomous Vehicle Control

Implements gain scheduling strategies:
  - Speed-based gain scheduling
  - Load-based gain scheduling
  - Interpolating lookup tables (1D and 2D)
  - Smooth transition between gain regions
  - Combined scheduling (multi-variable)

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GainSet:
    """A set of controller gains.

    Attributes:
        kp: Proportional gains (input_dim,).
        ki: Integral gains (input_dim,).
        kd: Derivative gains (input_dim,).
        name: Name identifier for this gain set.
    """
    kp: np.ndarray = np.array([1.0])
    ki: np.ndarray = np.array([0.0])
    kd: np.ndarray = np.array([0.0])
    name: str = ""

    def to_vector(self) -> np.ndarray:
        """Convert gains to a flat vector."""
        return np.concatenate([self.kp, self.ki, self.kd])

    @classmethod
    def from_vector(cls, vec: np.ndarray, name: str = "") -> "GainSet":
        """Create GainSet from a flat vector."""
        n = len(vec) // 3
        return cls(
            kp=vec[:n],
            ki=vec[n:2*n],
            kd=vec[2*n:],
            name=name,
        )


class InterpolatingTable1D:
    """1D lookup table with linear interpolation and smooth transitions.

    Used for single-variable gain scheduling (e.g., speed-based).
    """

    def __init__(
        self,
        breakpoints: np.ndarray,
        values: np.ndarray,
        extrapolate: bool = False,
        smooth_transition: bool = True,
        transition_width: float = 0.1,
    ) -> None:
        """Initialize the 1D lookup table.

        Args:
            breakpoints: X-axis breakpoints (must be sorted ascending).
            values: Y-axis values at each breakpoint. Shape (N,) or (N, M).
            extrapolate: If True, extrapolate beyond range.
            smooth_transition: Use smooth (Hermite) interpolation.
            transition_width: Width for smooth blending.
        """
        if len(breakpoints) < 2:
            raise ValueError("At least 2 breakpoints required")

        # Sort by breakpoints
        sort_idx = np.argsort(breakpoints)
        self._bp = breakpoints[sort_idx]
        self._values = np.atleast_2d(values[sort_idx])
        self._extrapolate = extrapolate
        self._smooth = smooth_transition
        self._width = transition_width

    def lookup(self, x: float) -> np.ndarray:
        """Look up value with interpolation.

        Args:
            x: Input value.

        Returns:
            Interpolated output vector.
        """
        # Below range
        if x <= self._bp[0]:
            if self._extrapolate and len(self._bp) >= 2:
                slope = (self._values[1] - self._values[0]) / (self._bp[1] - self._bp[0])
                return self._values[0] + slope * (x - self._bp[0])
            return self._values[0].copy()

        # Above range
        if x >= self._bp[-1]:
            if self._extrapolate and len(self._bp) >= 2:
                n = len(self._bp) - 1
                slope = (self._values[n] - self._values[n-1]) / (self._bp[n] - self._bp[n-1])
                return self._values[n] + slope * (x - self._bp[n])
            return self._values[-1].copy()

        # Find interval
        for i in range(len(self._bp) - 1):
            if self._bp[i] <= x <= self._bp[i + 1]:
                t = (x - self._bp[i]) / (self._bp[i + 1] - self._bp[i])

                if self._smooth:
                    # Smoothstep interpolation (Hermite)
                    t = t * t * (3.0 - 2.0 * t)

                return self._values[i] + t * (self._values[i + 1] - self._values[i])

        return self._values[-1].copy()

    def lookup_batch(self, x_array: np.ndarray) -> np.ndarray:
        """Look up values for an array of inputs.

        Args:
            x_array: Input values.

        Returns:
            Array of interpolated outputs.
        """
        return np.array([self.lookup(x) for x in x_array])


class InterpolatingTable2D:
    """2D lookup table with bilinear interpolation.

    Used for two-variable gain scheduling (e.g., speed × load).
    """

    def __init__(
        self,
        x_breakpoints: np.ndarray,
        y_breakpoints: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Initialize the 2D lookup table.

        Args:
            x_breakpoints: X-axis breakpoints (1D array, sorted).
            y_breakpoints: Y-axis breakpoints (1D array, sorted).
            values: 2D array of values at grid points (len(x) x len(y)).
        """
        self._x_bp = np.sort(x_breakpoints)
        self._y_bp = np.sort(y_breakpoints)

        if values.shape != (len(self._x_bp), len(self._y_bp)):
            raise ValueError(
                f"Values shape {values.shape} doesn't match "
                f"breakpoints ({len(self._x_bp)}, {len(self._y_bp)})"
            )
        self._values = values.copy()

    def lookup(self, x: float, y: float) -> float:
        """Look up value with bilinear interpolation.

        Args:
            x: X input value.
            y: Y input value.

        Returns:
            Interpolated scalar value.
        """
        # Clip to range
        x = np.clip(x, self._x_bp[0], self._x_bp[-1])
        y = np.clip(y, self._y_bp[0], self._y_bp[-1])

        # Find x interval
        xi = np.searchsorted(self._x_bp, x) - 1
        xi = max(0, min(xi, len(self._x_bp) - 2))
        tx = (x - self._x_bp[xi]) / (self._x_bp[xi + 1] - self._x_bp[xi])

        # Find y interval
        yi = np.searchsorted(self._y_bp, y) - 1
        yi = max(0, min(yi, len(self._y_bp) - 2))
        ty = (y - self._y_bp[yi]) / (self._y_bp[yi + 1] - self._y_bp[yi])

        # Bilinear interpolation
        v00 = self._values[xi, yi]
        v10 = self._values[xi + 1, yi]
        v01 = self._values[xi, yi + 1]
        v11 = self._values[xi + 1, yi + 1]

        v0 = v00 + tx * (v10 - v00)
        v1 = v01 + tx * (v11 - v01)

        return v0 + ty * (v1 - v0)


class SpeedGainScheduler:
    """Speed-based gain scheduler for vehicle control.

    Adjusts controller gains based on vehicle speed to maintain
    consistent performance across the speed range.

    At low speeds:
    - Higher proportional gains (more responsive)
    - Lower derivative gains (less noise-sensitive)

    At high speeds:
    - Lower proportional gains (more stable)
    - Higher derivative gains (better damping)
    """

    def __init__(
        self,
        speed_breakpoints: Optional[np.ndarray] = None,
        gain_table: Optional[Dict[str, np.ndarray]] = None,
        dt: float = 0.01,
        transition_smoothing: float = 0.5,
    ) -> None:
        """Initialize the speed gain scheduler.

        Args:
            speed_breakpoints: Speed breakpoints (m/s).
            gain_table: Dictionary with 'kp', 'ki', 'kd' arrays.
            dt: Controller timestep.
            transition_smoothing: Smoothing factor for gain transitions.
        """
        if speed_breakpoints is None:
            speed_breakpoints = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])

        if gain_table is None:
            # Default gain schedule for lateral control
            gain_table = {
                "kp": np.array([
                    [1.5, 1.3, 1.1, 1.0, 0.85, 0.7, 0.6, 0.5],
                ]).T,  # (N, M) format
                "ki": np.array([
                    [0.02, 0.02, 0.015, 0.01, 0.008, 0.006, 0.005, 0.004],
                ]).T,
                "kd": np.array([
                    [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06],
                ]).T,
            }

        self._kp_table = InterpolatingTable1D(speed_breakpoints, gain_table["kp"])
        self._ki_table = InterpolatingTable1D(speed_breakpoints, gain_table["ki"])
        self._kd_table = InterpolatingTable1D(speed_breakpoints, gain_table["kd"])

        self._dt = dt
        self._smoothing = transition_smoothing

        # Current gains (with smoothing)
        self._current_kp = self._kp_table.lookup(0.0)
        self._current_ki = self._ki_table.lookup(0.0)
        self._current_kd = self._kd_table.lookup(0.0)

    def update(self, speed: float, dt: Optional[float] = None) -> GainSet:
        """Get interpolated gains for the current speed.

        Args:
            speed: Vehicle speed (m/s).
            dt: Optional timestep override.

        Returns:
            GainSet with interpolated gains.
        """
        effective_dt = dt if dt is not None else self._dt

        # Look up raw gains
        target_kp = self._kp_table.lookup(speed)
        target_ki = self._ki_table.lookup(speed)
        target_kd = self._kd_table.lookup(speed)

        # Apply smoothing (low-pass filter)
        alpha = min(1.0, effective_dt / (self._smoothing + effective_dt))
        self._current_kp += alpha * (target_kp - self._current_kp)
        self._current_ki += alpha * (target_ki - self._current_ki)
        self._current_kd += alpha * (target_kd - self._current_kd)

        return GainSet(
            kp=self._current_kp.copy(),
            ki=self._current_ki.copy(),
            kd=self._current_kd.copy(),
            name=f"speed_{speed:.1f}ms",
        )

    def get_gains_at_speed(self, speed: float) -> GainSet:
        """Get gains at a specific speed without smoothing.

        Args:
            speed: Vehicle speed (m/s).

        Returns:
            GainSet at the given speed.
        """
        return GainSet(
            kp=self._kp_table.lookup(speed),
            ki=self._ki_table.lookup(speed),
            kd=self._kd_table.lookup(speed),
            name=f"speed_{speed:.1f}ms",
        )


class LoadGainScheduler:
    """Load-based gain scheduler for vehicle control.

    Adjusts gains based on vehicle load (payload, trailer weight)
    to maintain consistent handling characteristics.
    """

    def __init__(
        self,
        load_breakpoints: Optional[np.ndarray] = None,
        gain_table: Optional[Dict[str, np.ndarray]] = None,
        dt: float = 0.01,
    ) -> None:
        """Initialize the load gain scheduler.

        Args:
            load_breakpoints: Load breakpoints (kg).
            gain_table: Dictionary with gain arrays.
            dt: Controller timestep.
        """
        if load_breakpoints is None:
            load_breakpoints = np.array([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0])

        if gain_table is None:
            gain_table = {
                "kp": np.array([[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]).T,
                "ki": np.array([[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]]).T,
                "kd": np.array([[0.1, 0.11, 0.12, 0.13, 0.14, 0.15]]).T,
            }

        self._kp_table = InterpolatingTable1D(load_breakpoints, gain_table["kp"])
        self._ki_table = InterpolatingTable1D(load_breakpoints, gain_table["ki"])
        self._kd_table = InterpolatingTable1D(load_breakpoints, gain_table["kd"])
        self._dt = dt

        self._current_gains = self.get_gains_at_load(0.0)

    def get_gains_at_load(self, load_kg: float) -> GainSet:
        """Get interpolated gains for a given load.

        Args:
            load_kg: Additional load in kg.

        Returns:
            GainSet with interpolated gains.
        """
        return GainSet(
            kp=self._kp_table.lookup(load_kg),
            ki=self._ki_table.lookup(load_kg),
            kd=self._kd_table.lookup(load_kg),
        )

    def update(self, load_kg: float, dt: Optional[float] = None) -> GainSet:
        """Update gains based on current load with smoothing.

        Args:
            load_kg: Current additional load (kg).
            dt: Optional timestep override.

        Returns:
            Smoothed GainSet.
        """
        effective_dt = dt if dt is not None else self._dt

        target = self.get_gains_at_load(load_kg)
        alpha = min(1.0, effective_dt / (0.5 + effective_dt))

        new_kp = self._current_gains.kp + alpha * (target.kp - self._current_gains.kp)
        new_ki = self._current_gains.ki + alpha * (target.ki - self._current_gains.ki)
        new_kd = self._current_gains.kd + alpha * (target.kd - self._current_gains.kd)

        self._current_gains = GainSet(kp=new_kp, ki=new_ki, kd=new_kd)
        return self._current_gains


class CombinedGainScheduler:
    """Combined gain scheduler using both speed and load.

    Uses a 2D lookup table for gain interpolation based on
    both vehicle speed and payload.
    """

    def __init__(
        self,
        speed_breakpoints: Optional[np.ndarray] = None,
        load_breakpoints: Optional[np.ndarray] = None,
        kp_grid: Optional[np.ndarray] = None,
        ki_grid: Optional[np.ndarray] = None,
        kd_grid: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the combined gain scheduler.

        Args:
            speed_breakpoints: Speed breakpoints (m/s).
            load_breakpoints: Load breakpoints (kg).
            kp_grid: 2D gain grid for kp (speed x load).
            ki_grid: 2D gain grid for ki.
            kd_grid: 2D gain grid for kd.
        """
        if speed_breakpoints is None:
            speed_breakpoints = np.array([0.0, 10.0, 20.0, 30.0])
        if load_breakpoints is None:
            load_breakpoints = np.array([0.0, 500.0, 1000.0])

        if kp_grid is None:
            kp_grid = np.array([
                [1.5, 1.7, 1.9],
                [1.0, 1.2, 1.4],
                [0.7, 0.85, 1.0],
                [0.5, 0.6, 0.7],
            ])
        if ki_grid is None:
            ki_grid = np.array([
                [0.02, 0.025, 0.03],
                [0.01, 0.012, 0.015],
                [0.008, 0.01, 0.012],
                [0.005, 0.006, 0.008],
            ])
        if kd_grid is None:
            kd_grid = np.array([
                [0.3, 0.35, 0.4],
                [0.15, 0.18, 0.2],
                [0.1, 0.12, 0.14],
                [0.06, 0.07, 0.08],
            ])

        self._kp_table = InterpolatingTable2D(speed_breakpoints, load_breakpoints, kp_grid)
        self._ki_table = InterpolatingTable2D(speed_breakpoints, load_breakpoints, ki_grid)
        self._kd_table = InterpolatingTable2D(speed_breakpoints, load_breakpoints, kd_grid)

    def update(self, speed: float, load_kg: float) -> GainSet:
        """Get interpolated gains based on speed and load.

        Args:
            speed: Vehicle speed (m/s).
            load_kg: Additional load (kg).

        Returns:
            GainSet with interpolated gains.
        """
        kp_val = self._kp_table.lookup(speed, load_kg)
        ki_val = self._ki_table.lookup(speed, load_kg)
        kd_val = self._kd_table.lookup(speed, load_kg)

        return GainSet(
            kp=np.array([kp_val]),
            ki=np.array([ki_val]),
            kd=np.array([kd_val]),
        )
