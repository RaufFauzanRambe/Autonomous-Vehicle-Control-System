"""
Vehicle Dynamics Model for Autonomous Vehicle Control System.

Provides kinematic and dynamic bicycle models for vehicle state prediction,
linearization, and simulation. Supports both low-speed (kinematic) and
high-speed (dynamic) regimes with configurable vehicle parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class VehicleParameters:
    """Physical and geometric parameters of the vehicle.

    Attributes:
        wheelbase: Distance between front and rear axles [m].
        mass: Total vehicle mass [kg].
        inertia: Yaw moment of inertia [kg·m²].
        front_to_cg: Distance from front axle to center of gravity [m].
        rear_to_cg: Distance from rear axle to center of gravity [m].
        front_tire_stiffness: Cornering stiffness of front tires [N/rad].
        rear_tire_stiffness: Cornering stiffness of rear tires [N/rad].
        max_steer: Maximum steering angle [rad].
        max_accel: Maximum longitudinal acceleration [m/s²].
        max_decel: Maximum longitudinal deceleration [m/s²].
        drag_coefficient: Aerodynamic drag coefficient [N·s²/m²].
        frontal_area: Vehicle frontal area for drag computation [m²].
        air_density: Air density for drag computation [kg/m³].
        rolling_resistance: Rolling resistance coefficient [-].
        gravity: Gravitational acceleration [m/s²].
    """

    wheelbase: float = 2.9
    mass: float = 1500.0
    inertia: float = 2500.0
    front_to_cg: float = 1.4
    rear_to_cg: float = 1.5
    front_tire_stiffness: float = 80000.0
    rear_tire_stiffness: float = 90000.0
    max_steer: float = 0.6
    max_accel: float = 3.0
    max_decel: float = -5.0
    drag_coefficient: float = 0.3
    frontal_area: float = 2.2
    air_density: float = 1.225
    rolling_resistance: float = 0.015
    gravity: float = 9.81

    @property
    def cg_to_front(self) -> float:
        """Distance from CG to front axle [m]."""
        return self.front_to_cg

    @property
    def cg_to_rear(self) -> float:
        """Distance from CG to rear axle [m]."""
        return self.rear_to_cg

    @property
    def front_axle_to_rear(self) -> float:
        """Distance from front axle to rear axle (wheelbase) [m]."""
        return self.wheelbase


class BicycleModel:
    """Bicycle vehicle model with kinematic and dynamic variants.

    The kinematic model assumes no tire slip and is accurate at low speeds.
    The dynamic model incorporates tire slip angles and lateral dynamics,
    making it suitable for higher-speed prediction and control.

    State vector: [x, y, yaw, v, delta]
        x     – rear-axle (kinematic) or CG (dynamic) x position [m]
        y     – rear-axle (kinematic) or CG (dynamic) y position [m]
        yaw   – heading angle [rad]
        v     – longitudinal velocity [m/s]
        delta – steering angle [rad]

    Control inputs: [acceleration, steering_rate] or [acceleration, steering_angle]
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        params: Optional[VehicleParameters] = None,
        model_type: str = "kinematic",
        dt: float = 0.1,
    ) -> None:
        """Initialise the bicycle model.

        Args:
            params: Vehicle parameters. Uses defaults if *None*.
            model_type: Either ``'kinematic'`` or ``'dynamic'``.
            dt: Discrete time step for integration [s].

        Raises:
            ValueError: If *model_type* is not one of the accepted values.
        """
        if model_type not in ("kinematic", "dynamic"):
            raise ValueError(
                f"model_type must be 'kinematic' or 'dynamic', got '{model_type}'"
            )
        self.params = params if params is not None else VehicleParameters()
        self.model_type = model_type
        self.dt = dt

        # State dimension: x, y, yaw, v, delta
        self.n_states: int = 5
        # Extended state for dynamic model: + yaw_rate, + slip_angle
        self.n_states_dynamic: int = 7

        self._state: NDArray[np.float64] = np.zeros(self.n_states)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current state vector (copy)."""
        return self._state.copy()

    @state.setter
    def state(self, value: NDArray[np.float64]) -> None:
        if value.shape[0] != self.n_states:
            raise ValueError(
                f"State vector must have {self.n_states} elements, got {value.shape[0]}"
            )
        self._state = value.astype(np.float64)

    @property
    def x(self) -> float:
        """X position [m]."""
        return float(self._state[0])

    @property
    def y(self) -> float:
        """Y position [m]."""
        return float(self._state[1])

    @property
    def yaw(self) -> float:
        """Heading angle [rad]."""
        return float(self._state[2])

    @property
    def velocity(self) -> float:
        """Longitudinal velocity [m/s]."""
        return float(self._state[3])

    @property
    def steering(self) -> float:
        """Steering angle [rad]."""
        return float(self._state[4])

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def _kinematic_derivatives(
        self, state: NDArray[np.float64], accel: float, steer: float
    ) -> NDArray[np.float64]:
        """Compute state derivatives using the kinematic bicycle model.

        The kinematic model treats the vehicle as a rigid body with no
        slip at the wheels.  The reference point is the rear axle.

        Args:
            state: State vector [x, y, yaw, v, delta].
            accel: Longitudinal acceleration command [m/s²].
            steer: Steering angle command [rad].

        Returns:
            Time derivative of the state vector.
        """
        _, _, yaw, v, _ = state
        delta = np.clip(steer, -self.params.max_steer, self.params.max_steer)
        accel = np.clip(accel, self.params.max_decel, self.params.max_accel)

        dx = v * math.cos(yaw)
        dy = v * math.sin(yaw)
        if abs(v) > 1e-6:
            dyaw = v * math.tan(delta) / self.params.wheelbase
        else:
            dyaw = 0.0
        dv = accel
        ddelta = (delta - state[4]) / self.dt  # steering rate

        return np.array([dx, dy, dyaw, dv, ddelta])

    def _dynamic_derivatives(
        self, state: NDArray[np.float64], accel: float, steer: float
    ) -> NDArray[np.float64]:
        """Compute state derivatives using the dynamic bicycle model.

        Incorporates lateral tire forces via a linear tyre model
        (cornering stiffness).  Reference point is the centre of gravity.

        Args:
            state: State vector [x, y, yaw, v, delta].
            accel: Longitudinal acceleration command [m/s²].
            steer: Steering angle command [rad].

        Returns:
            Time derivative of the state vector.
        """
        x, y, yaw, v, _ = state
        delta = np.clip(steer, -self.params.max_steer, self.params.max_steer)
        accel = np.clip(accel, self.params.max_decel, self.params.max_accel)

        p = self.params

        # Tire slip angles
        if abs(v) > 1e-3:
            # Approximate yaw rate from kinematic relation for the extended state
            r = v * math.tan(delta) / p.wheelbase
            alpha_f = delta - math.atan2(v * math.sin(yaw) + p.front_to_cg * r,
                                          v * math.cos(yaw) + 1e-9)
            alpha_r = -math.atan2(v * math.sin(yaw) - p.rear_to_cg * r,
                                   v * math.cos(yaw) + 1e-9)
        else:
            alpha_f = delta
            alpha_r = 0.0
            r = 0.0

        # Lateral tire forces (linear model)
        fy_front = -p.front_tire_stiffness * alpha_f
        fy_rear = -p.rear_tire_stiffness * alpha_r

        # Aerodynamic drag force
        f_drag = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * v * abs(v)
        # Rolling resistance
        f_roll = p.rolling_resistance * p.mass * p.gravity * (1.0 if v > 0 else -1.0 if v < 0 else 0.0)

        # Equations of motion
        dx = v * math.cos(yaw)
        dy = v * math.sin(yaw)
        dyaw = r
        dv = accel - (f_drag + f_roll) / p.mass

        ddelta = (delta - state[4]) / self.dt

        return np.array([dx, dy, dyaw, dv, ddelta])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        accel: float,
        steer: float,
        dt: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Advance the vehicle state by one time step (RK4 integration).

        Args:
            accel: Longitudinal acceleration command [m/s²].
            steer: Steering angle command [rad].
            dt: Override time step. Uses *self.dt* if *None*.

        Returns:
            Updated state vector.
        """
        h = dt if dt is not None else self.dt
        deriv_fn = (
            self._kinematic_derivatives
            if self.model_type == "kinematic"
            else self._dynamic_derivatives
        )

        s = self._state
        k1 = deriv_fn(s, accel, steer)
        k2 = deriv_fn(s + 0.5 * h * k1, accel, steer)
        k3 = deriv_fn(s + 0.5 * h * k2, accel, steer)
        k4 = deriv_fn(s + h * k3, accel, steer)

        self._state = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce velocity non-negative for kinematic model
        if self.model_type == "kinematic":
            self._state[3] = max(0.0, self._state[3])

        return self._state.copy()

    def predict(
        self,
        state: NDArray[np.float64],
        accel_sequence: NDArray[np.float64],
        steer_sequence: NDArray[np.float64],
        dt: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Predict future state trajectory given control sequences.

        Args:
            state: Initial state vector [x, y, yaw, v, delta].
            accel_sequence: 1-D array of acceleration commands.
            steer_sequence: 1-D array of steering commands.
            dt: Override time step per step.

        Returns:
            Array of shape ``(N+1, n_states)`` where row 0 is *state*
            and row *i* is the state after applying the (*i*-1)-th control.
        """
        if accel_sequence.shape[0] != steer_sequence.shape[0]:
            raise ValueError("Acceleration and steering sequences must have equal length")

        h = dt if dt is not None else self.dt
        n_steps = accel_sequence.shape[0]
        traj = np.zeros((n_steps + 1, self.n_states))
        traj[0] = state.copy()

        saved = self._state.copy()
        self._state = state.copy()

        for i in range(n_steps):
            self.update(accel_sequence[i], steer_sequence[i], dt=h)
            traj[i + 1] = self._state.copy()

        self._state = saved  # restore
        return traj

    def linearize(
        self,
        state: NDArray[np.float64],
        accel: float,
        steer: float,
        dt: Optional[float] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute Jacobian linearisation around (state, control).

        Returns discrete-time matrices ``(A_d, B_d)`` such that

            x_{k+1} ≈ A_d @ x_k + B_d @ u_k + d

        where ``u = [accel, steer]^T`` and ``d`` is an offset term
        capturing the remainder of the first-order Taylor expansion.

        Args:
            state: Operating-point state.
            accel: Operating-point acceleration.
            steer: Operating-point steering angle.
            dt: Override time step.

        Returns:
            Tuple ``(A_d, B_d)`` – discrete state-transition and control
            matrices of shapes ``(n_states, n_states)`` and
            ``(n_states, 2)`` respectively.
        """
        h = dt if dt is not None else self.dt
        eps = 1e-5

        deriv_fn = (
            self._kinematic_derivatives
            if self.model_type == "kinematic"
            else self._dynamic_derivatives
        )

        f0 = deriv_fn(state, accel, steer)
        n = self.n_states
        m = 2  # accel, steer

        A_c = np.zeros((n, n))
        for i in range(n):
            s_plus = state.copy()
            s_plus[i] += eps
            s_minus = state.copy()
            s_minus[i] -= eps
            A_c[:, i] = (deriv_fn(s_plus, accel, steer) - deriv_fn(s_minus, accel, steer)) / (2.0 * eps)

        B_c = np.zeros((n, m))
        u_pts = [accel, steer]
        for i in range(m):
            u_plus = u_pts.copy()
            u_plus[i] += eps
            u_minus = u_pts.copy()
            u_minus[i] -= eps
            B_c[:, i] = (deriv_fn(state, u_plus[0], u_plus[1]) - deriv_fn(state, u_minus[0], u_minus[1])) / (2.0 * eps)

        # Forward-Euler discretisation
        A_d = np.eye(n) + h * A_c
        B_d = h * B_c

        return A_d, B_d

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def front_axle_position(self) -> Tuple[float, float]:
        """Return the (x, y) position of the front axle centre."""
        fx = self._state[0] + self.params.front_to_cg * math.cos(self._state[2])
        fy = self._state[1] + self.params.front_to_cg * math.sin(self._state[2])
        return fx, fy

    def rear_axle_position(self) -> Tuple[float, float]:
        """Return the (x, y) position of the rear axle centre."""
        rx = self._state[0] - self.params.rear_to_cg * math.cos(self._state[2])
        ry = self._state[1] - self.params.rear_to_cg * math.sin(self._state[2])
        return rx, ry

    def reset(self, state: Optional[NDArray[np.float64]] = None) -> None:
        """Reset model to zero or a given state.

        Args:
            state: New state vector. Zeros if *None*.
        """
        if state is not None:
            self._state = state.astype(np.float64).copy()
        else:
            self._state = np.zeros(self.n_states)

    def __repr__(self) -> str:
        return (
            f"BicycleModel(model_type='{self.model_type}', dt={self.dt}, "
            f"state=[{self.x:.2f}, {self.y:.2f}, {self.yaw:.3f}, "
            f"{self.velocity:.2f}, {self.steering:.3f}])"
        )
