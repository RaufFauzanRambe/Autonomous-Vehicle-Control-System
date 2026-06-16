"""
Vehicle Dynamics Models for MPC Controller

Implements multiple vehicle models:
  - Kinematic bicycle model (low-speed, parking)
  - Dynamic bicycle model (high-speed, handling)
  - Linearized models for MPC prediction
  - Tire models (linear, Pacejka, combined slip)
  - Vehicle parameter configurations

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class VehicleParameters:
    """Vehicle physical parameters for dynamics models.

    Attributes:
        mass: Vehicle mass (kg).
        wheelbase: Wheelbase (m).
        lf: CG to front axle distance (m).
        lr: CG to rear axle distance (m).
        track_width: Track width (m).
        cg_height: CG height (m).
        Iz: Yaw moment of inertia (kg*m^2).
        Cf: Front cornering stiffness (N/rad).
        Cr: Rear cornering stiffness (N/rad).
        max_steer: Maximum steering angle (rad).
        max_steer_rate: Maximum steering rate (rad/s).
        max_speed: Maximum speed (m/s).
        max_accel: Maximum longitudinal acceleration (m/s^2).
        max_decel: Maximum deceleration (m/s^2).
        drag_coeff: Aerodynamic drag coefficient.
        frontal_area: Frontal area (m^2).
        air_density: Air density (kg/m^3).
        rolling_resistance: Rolling resistance coefficient.
        gravity: Gravitational acceleration (m/s^2).
    """
    mass: float = 1800.0
    wheelbase: float = 2.85
    lf: float = 1.42
    lr: float = 1.43
    track_width: float = 1.6
    cg_height: float = 0.55
    Iz: float = 3200.0
    Cf: float = 80000.0
    Cr: float = 90000.0
    max_steer: float = math.radians(35)
    max_steer_rate: float = math.radians(120)
    max_speed: float = 50.0
    max_accel: float = 3.0
    max_decel: float = -8.0
    drag_coeff: float = 0.3
    frontal_area: float = 2.5
    air_density: float = 1.225
    rolling_resistance: float = 0.015
    gravity: float = 9.81


class KinematicBicycleModel:
    """Kinematic bicycle model for low-speed vehicle dynamics.

    State: [x, y, psi, v]  (position, heading, speed)
    Input: [accel, delta]   (acceleration, steering angle)

    This model assumes no slip at the wheels and is valid for
    low-speed maneuvers (typically < 5 m/s).

    Equations:
        dx/dt = v * cos(psi)
        dy/dt = v * sin(psi)
        dpsi/dt = v * tan(delta) / L
        dv/dt = accel
    """

    def __init__(self, params: VehicleParameters = VehicleParameters()) -> None:
        """Initialize the kinematic bicycle model.

        Args:
            params: Vehicle parameters.
        """
        self._params = params
        self._state_dim = 4
        self._input_dim = 2

    @property
    def state_dim(self) -> int:
        """Return state dimension."""
        return self._state_dim

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self._input_dim

    def dynamics(
        self,
        state: np.ndarray,
        input: np.ndarray,
    ) -> np.ndarray:
        """Compute state derivative.

        Args:
            state: State vector [x, y, psi, v].
            input: Input vector [accel, delta].

        Returns:
            State derivative [dx, dy, dpsi, dv].
        """
        x, y, psi, v = state
        accel, delta = input

        L = self._params.wheelbase

        # Clip steering angle
        delta = np.clip(delta, -self._params.max_steer, self._params.max_steer)

        dx = v * math.cos(psi)
        dy = v * math.sin(psi)
        dpsi = v * math.tan(delta) / L if L > 0 else 0.0
        dv = accel

        return np.array([dx, dy, dpsi, dv])

    def step(
        self,
        state: np.ndarray,
        input: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Integrate one step using RK4.

        Args:
            state: Current state.
            input: Control input.
            dt: Timestep.

        Returns:
            Next state.
        """
        # RK4 integration
        k1 = self.dynamics(state, input)
        k2 = self.dynamics(state + 0.5 * dt * k1, input)
        k3 = self.dynamics(state + 0.5 * dt * k2, input)
        k4 = self.dynamics(state + dt * k3, input)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure speed is non-negative
        next_state[3] = max(0.0, next_state[3])

        return next_state

    def linearize(
        self,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize the model around an equilibrium point.

        Computes discrete-time linear model:
            x_{k+1} = A_d * x_k + B_d * u_k

        Uses numerical Jacobian computation.

        Args:
            x_eq: Equilibrium state.
            u_eq: Equilibrium input.
            dt: Discretization timestep.

        Returns:
            Tuple of (A_d, B_d) matrices.
        """
        n = self._state_dim
        m = self._input_dim
        eps = 1e-6

        # Compute equilibrium next state
        x_next_eq = self.step(x_eq, u_eq, dt)

        # Numerical Jacobian A = df/dx
        A = np.zeros((n, n))
        for i in range(n):
            x_perturbed = x_eq.copy()
            x_perturbed[i] += eps
            x_next_perturbed = self.step(x_perturbed, u_eq, dt)
            A[:, i] = (x_next_perturbed - x_next_eq) / eps

        # Numerical Jacobian B = df/du
        B = np.zeros((n, m))
        for i in range(m):
            u_perturbed = u_eq.copy()
            u_perturbed[i] += eps
            x_next_perturbed = self.step(x_eq, u_perturbed, dt)
            B[:, i] = (x_next_perturbed - x_next_eq) / eps

        return A, B

    def get_discrete_model(self, v_ref: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get a discrete-time model linearized around a reference speed.

        Args:
            v_ref: Reference speed (m/s).
            dt: Discretization timestep.

        Returns:
            Tuple of (A_d, B_d) matrices.
        """
        x_eq = np.array([0.0, 0.0, 0.0, v_ref])
        u_eq = np.array([0.0, 0.0])  # Zero acceleration, zero steering
        return self.linearize(x_eq, u_eq, dt)


class DynamicBicycleModel:
    """Dynamic bicycle model for high-speed vehicle dynamics.

    State: [x, y, psi, vx, vy, r]  (position, heading, longitudinal/lateral speed, yaw rate)
    Input: [Fx, delta]              (longitudinal force, steering angle)

    This model includes tire slip angles and is valid for higher speeds
    where tire forces are important.

    Equations:
        m * (dvx/dt - vy * r) = Fx - Fyf * sin(delta)
        m * (dvy/dt + vx * r) = Fyf * cos(delta) + Fyr
        Iz * dr/dt = lf * Fyf * cos(delta) - lr * Fyr
    """

    def __init__(self, params: VehicleParameters = VehicleParameters()) -> None:
        """Initialize the dynamic bicycle model.

        Args:
            params: Vehicle parameters.
        """
        self._params = params
        self._state_dim = 6
        self._input_dim = 2

    @property
    def state_dim(self) -> int:
        """Return state dimension."""
        return self._state_dim

    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self._input_dim

    def _compute_tire_forces(
        self,
        state: np.ndarray,
        delta: float,
    ) -> Tuple[float, float, float, float]:
        """Compute tire slip angles and lateral forces.

        Uses linear tire model for small slip angles.

        Args:
            state: State vector [x, y, psi, vx, vy, r].
            delta: Steering angle.

        Returns:
            Tuple of (alpha_f, alpha_r, Fyf, Fyr).
        """
        _, _, _, vx, vy, r = state
        p = self._params

        # Avoid division by zero
        vx_safe = max(vx, 1.0)

        # Front slip angle
        alpha_f = delta - math.atan2(vy + p.lf * r, vx_safe)

        # Rear slip angle
        alpha_r = -math.atan2(vy - p.lr * r, vx_safe)

        # Linear tire forces
        Fyf = p.Cf * alpha_f
        Fyr = p.Cr * alpha_r

        return alpha_f, alpha_r, Fyf, Fyr

    def dynamics(
        self,
        state: np.ndarray,
        input: np.ndarray,
    ) -> np.ndarray:
        """Compute state derivative for the dynamic bicycle model.

        Args:
            state: State vector [x, y, psi, vx, vy, r].
            input: Input vector [Fx, delta].

        Returns:
            State derivative.
        """
        x, y, psi, vx, vy, r = state
        Fx, delta = input
        p = self._params

        delta = np.clip(delta, -p.max_steer, p.max_steer)

        # Tire forces
        alpha_f, alpha_r, Fyf, Fyr = self._compute_tire_forces(state, delta)

        # Equations of motion
        m = p.mass
        Iz = p.Iz

        dx = vx * math.cos(psi) - vy * math.sin(psi)
        dy = vx * math.sin(psi) + vy * math.cos(psi)
        dpsi = r
        dvx = (Fx - Fyf * math.sin(delta)) / m + vy * r
        dvy = (Fyf * math.cos(delta) + Fyr) / m - vx * r
        dr = (p.lf * Fyf * math.cos(delta) - p.lr * Fyr) / Iz

        return np.array([dx, dy, dpsi, dvx, dvy, dr])

    def step(
        self,
        state: np.ndarray,
        input: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Integrate one step using RK4.

        Args:
            state: Current state.
            input: Control input.
            dt: Timestep.

        Returns:
            Next state.
        """
        k1 = self.dynamics(state, input)
        k2 = self.dynamics(state + 0.5 * dt * k1, input)
        k3 = self.dynamics(state + 0.5 * dt * k2, input)
        k4 = self.dynamics(state + dt * k3, input)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ensure longitudinal speed is positive
        next_state[3] = max(0.5, next_state[3])

        return next_state

    def linearize(
        self,
        x_eq: np.ndarray,
        u_eq: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize around an equilibrium point.

        Args:
            x_eq: Equilibrium state.
            u_eq: Equilibrium input.
            dt: Discretization timestep.

        Returns:
            Tuple of (A_d, B_d) matrices.
        """
        n = self._state_dim
        m = self._input_dim
        eps = 1e-6

        x_next_eq = self.step(x_eq, u_eq, dt)

        A = np.zeros((n, n))
        for i in range(n):
            x_perturbed = x_eq.copy()
            x_perturbed[i] += eps
            x_next_perturbed = self.step(x_perturbed, u_eq, dt)
            A[:, i] = (x_next_perturbed - x_next_eq) / eps

        B = np.zeros((n, m))
        for i in range(m):
            u_perturbed = u_eq.copy()
            u_perturbed[i] += eps
            x_next_perturbed = self.step(x_eq, u_perturbed, dt)
            B[:, i] = (x_next_perturbed - x_next_eq) / eps

        return A, B

    def get_error_dynamics_model(
        self,
        v_ref: float,
        curvature: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear error dynamics model for path tracking MPC.

        The error state is [e_y, e_psi, v_s, e_vy] where:
        - e_y: lateral error
        - e_psi: heading error
        - v_s: longitudinal speed error
        - e_vy: lateral velocity error

        Args:
            v_ref: Reference speed (m/s).
            curvature: Path curvature (1/m).
            dt: Discretization timestep.

        Returns:
            Tuple of (A_d, B_d) matrices.
        """
        p = self._params

        # Continuous-time error dynamics matrices
        A_ct = np.array([
            [0, v_ref, 0, 1],
            [0, -(p.Cf + p.Cr) / (p.mass * v_ref), 0, 0],
            [0, 0, 0, 0],
            [0, -(p.lf * p.Cf - p.lr * p.Cr) / (p.Iz * v_ref) - v_ref, 0,
             -(p.lf**2 * p.Cf + p.lr**2 * p.Cr) / (p.Iz * v_ref)],
        ])

        B_ct = np.array([
            [0],
            [p.Cf / (p.mass)],
            [1 / p.mass],
            [p.lf * p.Cf / p.Iz],
        ])

        # Discretize using zero-order hold
        n = A_ct.shape[0]
        m = B_ct.shape[1]

        # Matrix exponential: e^{A*dt}
        A_d = np.eye(n) + A_ct * dt + 0.5 * A_ct @ A_ct * dt**2
        B_d = B_ct * dt + 0.5 * A_ct @ B_ct * dt**2

        return A_d, B_d


class PacejkaTireModel:
    """Pacejka "Magic Formula" tire model.

    Computes tire forces as a function of slip angle and vertical load.

    Fy = D * sin(C * atan(B * alpha - E * (B * alpha - atan(B * alpha))))

    where B, C, D, E are tire-specific coefficients.
    """

    def __init__(
        self,
        B: float = 10.0,
        C: float = 1.9,
        D: float = 1.0,
        E: float = -0.5,
    ) -> None:
        """Initialize the Pacejka tire model.

        Args:
            B: Stiffness factor.
            C: Shape factor.
            D: Peak factor (normalized by Fz).
            E: Curvature factor.
        """
        self._B = B
        self._C = C
        self._D = D
        self._E = E

    def lateral_force(self, alpha: float, Fz: float) -> float:
        """Compute lateral tire force.

        Args:
            alpha: Slip angle in radians.
            Fz: Vertical load in N.

        Returns:
            Lateral force in N.
        """
        B_alpha = self._B * alpha
        inner = B_alpha - self._E * (B_alpha - math.atan(B_alpha))
        Fy = Fz * self._D * math.sin(self._C * math.atan(inner))
        return Fy

    def longitudinal_force(self, kappa: float, Fz: float) -> float:
        """Compute longitudinal tire force from slip ratio.

        Args:
            kappa: Longitudinal slip ratio.
            Fz: Vertical load in N.

        Returns:
            Longitudinal force in N.
        """
        B_kappa = self._B * 0.5 * kappa  # Typically different B for longitudinal
        inner = B_kappa - self._E * (B_kappa - math.atan(B_kappa))
        Fx = Fz * self._D * math.sin(self._C * math.atan(inner))
        return Fx

    def combined_slip(
        self,
        alpha: float,
        kappa: float,
        Fz: float,
    ) -> Tuple[float, float]:
        """Compute combined slip tire forces.

        Uses the friction ellipse concept to limit combined forces.

        Args:
            alpha: Slip angle in radians.
            kappa: Longitudinal slip ratio.
            Fz: Vertical load in N.

        Returns:
            Tuple of (Fx, Fy) forces.
        """
        # Pure slip forces
        Fx_pure = self.longitudinal_force(kappa, Fz)
        Fy_pure = self.lateral_force(alpha, Fz)

        # Maximum friction force
        F_max = Fz * self._D

        # Friction ellipse scaling
        Fx_sq = Fx_pure ** 2
        Fy_sq = Fy_pure ** 2
        F_total = math.sqrt(Fx_sq + Fy_sq)

        if F_total > F_max and F_total > 0:
            scale = F_max / F_total
            Fx_pure *= scale
            Fy_pure *= scale

        return Fx_pure, Fy_pure


class LinearTireModel:
    """Linear tire model for small slip angles.

    Fy = C_alpha * alpha

    Valid for |alpha| < ~5 degrees.
    """

    def __init__(
        self,
        cornering_stiffness_front: float = 80000.0,
        cornering_stiffness_rear: float = 90000.0,
        max_slip_angle: float = math.radians(8),
    ) -> None:
        """Initialize the linear tire model.

        Args:
            cornering_stiffness_front: Front cornering stiffness (N/rad).
            cornering_stiffness_rear: Rear cornering stiffness (N/rad).
            max_slip_angle: Maximum valid slip angle (rad).
        """
        self._Cf = cornering_stiffness_front
        self._Cr = cornering_stiffness_rear
        self._max_alpha = max_slip_angle

    def front_lateral_force(self, alpha_f: float) -> float:
        """Compute front lateral force.

        Args:
            alpha_f: Front slip angle (rad).

        Returns:
            Front lateral force (N).
        """
        alpha_clipped = np.clip(alpha_f, -self._max_alpha, self._max_alpha)
        return self._Cf * alpha_clipped

    def rear_lateral_force(self, alpha_r: float) -> float:
        """Compute rear lateral force.

        Args:
            alpha_r: Rear slip angle (rad).

        Returns:
            Rear lateral force (N).
        """
        alpha_clipped = np.clip(alpha_r, -self._max_alpha, self._max_alpha)
        return self._Cr * alpha_clipped


def create_vehicle_model(
    model_type: str = "kinematic",
    params: VehicleParameters = VehicleParameters(),
) -> object:
    """Factory function to create a vehicle dynamics model.

    Args:
        model_type: Type of model ('kinematic' or 'dynamic').
        params: Vehicle parameters.

    Returns:
        Vehicle dynamics model instance.
    """
    if model_type == "kinematic":
        return KinematicBicycleModel(params)
    elif model_type == "dynamic":
        return DynamicBicycleModel(params)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'kinematic' or 'dynamic'.")
