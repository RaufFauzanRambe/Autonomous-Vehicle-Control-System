"""
Adaptive Controller Base for Autonomous Vehicle Control

Implements multiple adaptive control architectures:
  - Model Reference Adaptive Control (MRAC)
  - L1 Adaptive Control
  - Direct adaptive control with Lyapunov stability
  - Indirect adaptive control with parameter estimation
  - Composite adaptation

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AdaptiveParams:
    """Parameters for adaptive control.

    Attributes:
        state_dim: State dimension.
        input_dim: Input dimension.
        gamma: Adaptation gain (learning rate).
        sigma: e-modification parameter (prevents parameter drift).
        gamma_proj: Projection operator bound.
        adaptation_limit: Maximum parameter estimate magnitude.
        initial_param_estimate: Initial parameter estimate.
        reference_model_bw: Reference model bandwidth (rad/s).
        damping: Reference model damping ratio.
    """
    state_dim: int = 4
    input_dim: int = 2
    gamma: float = 1.0
    sigma: float = 0.01
    gamma_proj: float = 10.0
    adaptation_limit: float = 100.0
    initial_param_estimate: Optional[np.ndarray] = None
    reference_model_bw: float = 5.0
    damping: float = 0.9


class ReferenceModel:
    """Reference model for Model Reference Adaptive Control (MRAC).

    Generates the desired closed-loop trajectory that the plant
    should follow. Typically a second-order system:

        G_ref(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    """

    def __init__(
        self,
        bandwidth: float = 5.0,
        damping: float = 0.9,
        dt: float = 0.01,
    ) -> None:
        """Initialize the reference model.

        Args:
            bandwidth: Natural frequency omega_n (rad/s).
            damping: Damping ratio zeta.
            dt: Timestep in seconds.
        """
        self._omega_n = bandwidth
        self._zeta = damping
        self._dt = dt

        # Discrete state-space realization
        wn = self._omega_n
        zeta = self._zeta

        # Continuous-time state-space
        self._A_c = np.array([
            [0, 1],
            [-wn ** 2, -2 * zeta * wn],
        ])
        self._B_c = np.array([[0], [wn ** 2]])
        self._C = np.array([[1, 0]])
        self._D = np.array([[0]])

        # Discretize using matrix exponential (approximate)
        self._A_d = np.eye(2) + self._A_c * dt + 0.5 * self._A_c @ self._A_c * dt ** 2
        self._B_d = (self._B_c * dt + 0.5 * self._A_c @ self._B_c * dt ** 2)

        # State
        self._state = np.zeros(2)

    @property
    def bandwidth(self) -> float:
        """Return reference model bandwidth."""
        return self._omega_n

    @property
    def damping(self) -> float:
        """Return reference model damping ratio."""
        return self._zeta

    def update(self, command: float, dt: Optional[float] = None) -> np.ndarray:
        """Compute reference model output.

        Args:
            command: Reference command input.
            dt: Optional timestep override.

        Returns:
            Reference state [position, velocity].
        """
        effective_dt = dt if dt is not None else self._dt

        # State update
        self._state = self._A_d @ self._state + self._B_d.flatten() * command

        return self._state.copy()

    def reset(self) -> None:
        """Reset the reference model state."""
        self._state = np.zeros(2)


class MRACController:
    """Model Reference Adaptive Controller (MRAC).

    Implements direct MRAC with the following adaptation law:

        dθ/dt = -Γ * e * Φ(x)

    where:
    - θ: Adaptive parameter vector
    - Γ: Adaptation gain matrix
    - e: Tracking error (plant output - reference output)
    - Φ(x): Regressor vector (basis functions of the state)

    Supports:
    - e-modification for robustness to disturbances
    - Projection operator for bounded parameters
    - Both scalar and vector adaptation

    Example:
        >>> params = AdaptiveParams(gamma=2.0)
        >>> mrac = MRACController(params)
        >>> u_adaptive = mrac.update(state, reference_state, nominal_control)
    """

    def __init__(
        self,
        params: AdaptiveParams = AdaptiveParams(),
        dt: float = 0.01,
        name: str = "mrac",
    ) -> None:
        """Initialize the MRAC controller.

        Args:
            params: Adaptive control parameters.
            dt: Controller timestep in seconds.
            name: Controller name.
        """
        self._params = params
        self._dt = dt
        self._name = name

        # Reference model
        self._ref_model = ReferenceModel(
            bandwidth=params.reference_model_bw,
            damping=params.damping,
            dt=dt,
        )

        # Adaptive parameters
        n_params = params.state_dim * params.input_dim
        if params.initial_param_estimate is not None:
            self._theta = params.initial_param_estimate.copy()
        else:
            self._theta = np.zeros(n_params)

        # Adaptation gain matrix
        self._Gamma = params.gamma * np.eye(n_params)

        # State tracking
        self._tracking_error = np.zeros(params.state_dim)
        self._prev_tracking_error = np.zeros(params.state_dim)
        self._adaptation_rate = 0.0
        self._parameter_history: List[np.ndarray] = []
        self._error_history: List[float] = []

    @property
    def name(self) -> str:
        """Return controller name."""
        return self._name

    @property
    def adaptive_parameters(self) -> np.ndarray:
        """Return current adaptive parameter estimates."""
        return self._theta.copy()

    @property
    def tracking_error(self) -> np.ndarray:
        """Return current tracking error."""
        return self._tracking_error.copy()

    @property
    def reference_model(self) -> ReferenceModel:
        """Return the reference model."""
        return self._ref_model

    def _compute_regressor(self, state: np.ndarray) -> np.ndarray:
        """Compute the regressor vector Φ(x).

        The regressor maps the state to the parameter space.
        Standard choice: Φ(x) = [x₁, x₂, ..., xₙ] ⊗ I_m

        Args:
            state: Current state vector.

        Returns:
            Regressor vector.
        """
        n = self._params.state_dim
        m = self._params.input_dim

        # Simple linear regressor: Φ = x ⊗ I_m
        regressor = np.kron(state, np.eye(m)).flatten()

        # Ensure dimensions match
        if len(regressor) != len(self._theta):
            regressor = np.zeros(len(self._theta))
            for i in range(min(len(regressor), n)):
                for j in range(min(m, len(self._theta) // max(n, 1))):
                    idx = i * m + j
                    if idx < len(regressor):
                        regressor[idx] = state[i] if i < len(state) else 0.0

        return regressor

    def _projection_operator(self, theta: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Projection operator to bound parameter estimates.

        Implements: proj(θ, δ) = δ if ||θ|| < M or θ'δ <= 0
                                   δ - (θ'δ / ||θ||²) * θ otherwise

        Args:
            theta: Current parameter vector.
            delta: Adaptation direction (raw update).

        Returns:
            Projected adaptation direction.
        """
        M = self._params.gamma_proj
        theta_norm_sq = np.dot(theta, theta)

        if theta_norm_sq <= M ** 2:
            return delta
        elif np.dot(theta, delta) <= 0:
            return delta
        else:
            projection = delta - (np.dot(theta, delta) / theta_norm_sq) * theta
            return projection

    def update(
        self,
        plant_state: np.ndarray,
        reference_command: float,
        nominal_control: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute adaptive control signal.

        Args:
            plant_state: Current plant state.
            reference_command: Reference command input.
            nominal_control: Nominal (baseline) control signal.
            dt: Optional timestep override.

        Returns:
            Adaptive control signal (input_dim,).
        """
        effective_dt = dt if dt is not None else self._dt

        # Update reference model
        ref_state = self._ref_model.update(reference_command, effective_dt)

        # Compute tracking error
        self._tracking_error = plant_state - ref_state[:len(plant_state)]
        error_norm = np.linalg.norm(self._tracking_error)

        # Compute regressor
        phi = self._compute_regressor(plant_state)

        # Adaptation law with e-modification
        sigma = self._params.sigma
        raw_update = -self._Gamma @ (phi * error_norm + sigma * error_norm * self._theta)

        # Apply projection
        projected_update = self._projection_operator(self._theta, raw_update)

        # Update parameters
        self._theta += projected_update * effective_dt

        # Clip parameters to adaptation limit
        self._theta = np.clip(
            self._theta,
            -self._params.adaptation_limit,
            self._params.adaptation_limit,
        )

        # Compute adaptive control signal
        u_adaptive = self._theta @ phi

        # Total control: nominal + adaptive
        m = self._params.input_dim
        u_total = nominal_control.copy()
        if len(u_total) >= m:
            for i in range(m):
                u_total[i] += u_adaptive if i == 0 else 0.0

        # Record history
        self._parameter_history.append(self._theta.copy())
        self._error_history.append(error_norm)
        self._adaptation_rate = float(np.linalg.norm(projected_update))

        self._prev_tracking_error = self._tracking_error.copy()

        return u_total

    def reset(self) -> None:
        """Reset the MRAC controller."""
        self._ref_model.reset()
        self._theta = np.zeros(len(self._theta))
        self._tracking_error = np.zeros(self._params.state_dim)
        self._prev_tracking_error = np.zeros(self._params.state_dim)
        self._parameter_history = []
        self._error_history = []
        self._adaptation_rate = 0.0

    def get_diagnostics(self) -> Dict:
        """Return diagnostic information.

        Returns:
            Dictionary with diagnostics.
        """
        return {
            "tracking_error_norm": float(np.linalg.norm(self._tracking_error)),
            "parameter_norm": float(np.linalg.norm(self._theta)),
            "adaptation_rate": self._adaptation_rate,
            "num_adaptations": len(self._parameter_history),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MRACController(name='{self._name}', "
            f"gamma={self._params.gamma}, sigma={self._params.sigma})"
        )


class L1AdaptiveController:
    """L1 Adaptive Controller.

    Implements L1 adaptive control with:
    - State predictor (fast estimation)
    - Adaptation law (piecewise constant)
    - Low-pass filter (ensures bounded control)

    The L1 architecture decouples adaptation from robustness,
    allowing fast adaptation without sacrificing robustness.

    Key equation:
        u(s) = -C(s) * (θ_hat * x_hat(s))

    where C(s) is a low-pass filter.
    """

    def __init__(
        self,
        state_dim: int = 4,
        input_dim: int = 2,
        adaptation_gain: float = 100.0,
        filter_cutoff: float = 10.0,
        dt: float = 0.01,
        name: str = "l1_adaptive",
    ) -> None:
        """Initialize the L1 adaptive controller.

        Args:
            state_dim: State dimension.
            input_dim: Input dimension.
            adaptation_gain: Adaptation gain Γ.
            filter_cutoff: Low-pass filter cutoff frequency (Hz).
            dt: Controller timestep.
            name: Controller name.
        """
        self._state_dim = state_dim
        self._input_dim = input_dim
        self._adaptation_gain = adaptation_gain
        self._filter_cutoff = filter_cutoff
        self._dt = dt
        self._name = name

        # Parameter estimates
        self._theta_hat = np.zeros(state_dim)
        self._sigma_hat = np.zeros(state_dim)  # Disturbance estimate

        # State predictor state
        self._x_hat = np.zeros(state_dim)
        self._prediction_error = np.zeros(state_dim)

        # Low-pass filter state
        self._filtered_control = np.zeros(input_dim)

        # Filter coefficient
        tau = 1.0 / (2.0 * math.pi * filter_cutoff)
        self._filter_alpha = dt / (tau + dt)

        # History
        self._parameter_history: List[np.ndarray] = []

    @property
    def name(self) -> str:
        """Return controller name."""
        return self._name

    def _update_state_predictor(
        self,
        state: np.ndarray,
        control: np.ndarray,
        Am: np.ndarray,
        Bm: np.ndarray,
        dt: float,
    ) -> None:
        """Update the state predictor.

        Predicts: x_hat_dot = Am * x_hat + Bm * u + theta_hat * |x| + sigma_hat

        Args:
            state: Current plant state.
            control: Current control input.
            Am: Nominal state matrix.
            Bm: Nominal input matrix.
            dt: Timestep.
        """
        # Prediction dynamics
        x_hat_dot = (
            Am @ self._x_hat
            + Bm @ control
            + self._theta_hat * np.abs(state)
            + self._sigma_hat
        )
        self._x_hat += x_hat_dot * dt

        # Compute prediction error
        self._prediction_error = state - self._x_hat

    def _adaptation_law(self, dt: float) -> None:
        """Piecewise constant adaptation law.

        θ_hat = -Γ * sign(x_tilde)
        σ_hat = -Γ * sign(x_tilde) * |x|

        Args:
            dt: Timestep.
        """
        Gamma = self._adaptation_gain

        # Update parameter estimates
        for i in range(self._state_dim):
            if abs(self._prediction_error[i]) > 1e-10:
                self._theta_hat[i] = -Gamma * np.sign(self._prediction_error[i])
                self._sigma_hat[i] = -Gamma * np.sign(self._prediction_error[i]) * abs(self._prediction_error[i])

    def _apply_filter(self, raw_control: np.ndarray, dt: float) -> np.ndarray:
        """Apply low-pass filter to the adaptive control signal.

        Args:
            raw_control: Unfiltered adaptive control.
            dt: Timestep.

        Returns:
            Filtered control signal.
        """
        alpha = self._filter_alpha
        self._filtered_control = (
            alpha * raw_control + (1 - alpha) * self._filtered_control
        )
        return self._filtered_control.copy()

    def update(
        self,
        state: np.ndarray,
        reference: np.ndarray,
        nominal_control: np.ndarray,
        Am: Optional[np.ndarray] = None,
        Bm: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute L1 adaptive control signal.

        Args:
            state: Current plant state.
            reference: Reference state.
            nominal_control: Nominal (baseline) control.
            Am: Nominal state matrix. If None, uses identity.
            Bm: Nominal input matrix. If None, uses identity.
            dt: Optional timestep override.

        Returns:
            Adaptive control signal.
        """
        effective_dt = dt if dt is not None else self._dt

        if Am is None:
            Am = -np.eye(self._state_dim) * 5.0  # Stable nominal dynamics
        if Bm is None:
            Bm = np.eye(self._state_dim)[:, :self._input_dim]

        # Update state predictor
        self._update_state_predictor(state, nominal_control, Am, Bm, effective_dt)

        # Adaptation law
        self._adaptation_law(effective_dt)

        # Compute raw adaptive control
        # u_ad = -C(s) * (theta_hat * |x| + sigma_hat)
        uncertainty_estimate = self._theta_hat * np.abs(state) + self._sigma_hat

        # Map uncertainty to control space
        raw_adaptive = np.zeros(self._input_dim)
        for i in range(min(self._input_dim, len(uncertainty_estimate))):
            raw_adaptive[i] = -uncertainty_estimate[i]

        # Apply low-pass filter
        filtered_adaptive = self._apply_filter(raw_adaptive, effective_dt)

        # Total control
        u_total = nominal_control + filtered_adaptive

        # Record
        self._parameter_history.append(self._theta_hat.copy())

        return u_total

    def reset(self) -> None:
        """Reset the L1 controller state."""
        self._theta_hat = np.zeros(self._state_dim)
        self._sigma_hat = np.zeros(self._state_dim)
        self._x_hat = np.zeros(self._state_dim)
        self._prediction_error = np.zeros(self._state_dim)
        self._filtered_control = np.zeros(self._input_dim)
        self._parameter_history = []

    def get_diagnostics(self) -> Dict:
        """Return diagnostic information."""
        return {
            "prediction_error_norm": float(np.linalg.norm(self._prediction_error)),
            "theta_hat_norm": float(np.linalg.norm(self._theta_hat)),
            "sigma_hat_norm": float(np.linalg.norm(self._sigma_hat)),
        }


class CompositeAdaptiveController:
    """Composite Adaptive Controller.

    Combines tracking-error-based and prediction-error-based adaptation
    for improved transient performance and robustness.

    dθ/dt = -Γ_t * (tracking_error * Φ_t) - Γ_p * (prediction_error * Φ_p)
    """

    def __init__(
        self,
        state_dim: int = 4,
        input_dim: int = 2,
        gamma_tracking: float = 1.0,
        gamma_prediction: float = 5.0,
        dt: float = 0.01,
        name: str = "composite_adaptive",
    ) -> None:
        """Initialize the composite adaptive controller.

        Args:
            state_dim: State dimension.
            input_dim: Input dimension.
            gamma_tracking: Tracking error adaptation gain.
            gamma_prediction: Prediction error adaptation gain.
            dt: Timestep.
            name: Controller name.
        """
        self._state_dim = state_dim
        self._input_dim = input_dim
        self._gamma_t = gamma_tracking
        self._gamma_p = gamma_prediction
        self._dt = dt
        self._name = name

        # Parameter estimates
        self._theta = np.zeros(state_dim * input_dim)

        # Reference model
        self._ref_model = ReferenceModel(bandwidth=5.0, damping=0.9, dt=dt)

        # State predictor
        self._x_hat = np.zeros(state_dim)

        # Errors
        self._tracking_error = np.zeros(state_dim)
        self._prediction_error = np.zeros(state_dim)

    def update(
        self,
        state: np.ndarray,
        reference_command: float,
        nominal_control: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute composite adaptive control.

        Args:
            state: Current plant state.
            reference_command: Reference command.
            nominal_control: Nominal control.
            dt: Optional timestep override.

        Returns:
            Adaptive control signal.
        """
        effective_dt = dt if dt is not None else self._dt

        # Reference model
        ref_state = self._ref_model.update(reference_command, effective_dt)

        # Tracking error
        self._tracking_error = state - ref_state[:len(state)]

        # Prediction error
        self._prediction_error = state - self._x_hat

        # Update state predictor
        self._x_hat += (state - self._x_hat) * 0.5  # Simple predictor

        # Regressor
        phi = np.kron(state, np.eye(self._input_dim)).flatten()

        # Composite adaptation law
        d_theta_tracking = -self._gamma_t * phi * np.linalg.norm(self._tracking_error)
        d_theta_prediction = -self._gamma_p * phi * np.linalg.norm(self._prediction_error)

        self._theta += (d_theta_tracking + d_theta_prediction) * effective_dt

        # Adaptive control
        u_adaptive = self._theta @ phi
        u_total = nominal_control.copy()
        if len(u_total) > 0:
            u_total[0] += u_adaptive

        return u_total

    def reset(self) -> None:
        """Reset the controller state."""
        self._theta = np.zeros(len(self._theta))
        self._ref_model.reset()
        self._x_hat = np.zeros(self._state_dim)
        self._tracking_error = np.zeros(self._state_dim)
        self._prediction_error = np.zeros(self._state_dim)
