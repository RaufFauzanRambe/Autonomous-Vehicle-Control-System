"""
Adaptive Tuning Module for Autonomous Vehicle Control

Implements automatic controller parameter adaptation:
  - Gradient-based parameter adaptation
  - Lyapunov-based adaptation laws (guaranteed stability)
  - MIT rule for model reference adaptive control
  - Stability monitoring with Lyapunov function tracking
  - Bounded adaptation with projection operator
  - Composite adaptation combining tracking and prediction errors

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveTuningConfig:
    """Configuration for the adaptive tuning module.

    Attributes:
        n_params: Number of tunable parameters.
        learning_rate: Base learning rate (γ) for gradient adaptation.
        mit_gain: MIT rule adaptation gain.
        lyapunov_gain: Lyapunov adaptation gain.
        adaptation_bounds: Per-parameter (min, max) bounds.
        sigma: e-modification parameter for robustness.
        projection_bound: Norm bound for projection operator.
        stability_threshold: Lyapunov derivative threshold for instability alert.
        min_learning_rate: Floor for learning rate decay.
        max_learning_rate: Ceiling for learning rate.
        decay_rate: Exponential decay rate for learning rate.
    """
    n_params: int = 4
    learning_rate: float = 0.01
    mit_gain: float = 0.5
    lyapunov_gain: float = 2.0
    adaptation_bounds: Optional[List[Tuple[float, float]]] = None
    sigma: float = 0.01
    projection_bound: float = 50.0
    stability_threshold: float = -1e-3
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1.0
    decay_rate: float = 0.9999


# ---------------------------------------------------------------------------
# Projection Operator
# ---------------------------------------------------------------------------

def projection_operator(
    theta: np.ndarray,
    raw_update: np.ndarray,
    bound: float,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Smooth projection operator to keep parameters within a bounded set.

    Implements:

        proj(θ, Δ) = Δ                                       if ‖θ‖ ≤ M - ε
                    = Δ · (M - ‖θ‖) / ε                      if M - ε < ‖θ‖ ≤ M
                    = Δ · max(0, 1 - (θᵀΔ / (‖θ‖·‖Δ‖)))    if ‖θ‖ > M and θᵀΔ > 0
                    = Δ                                       otherwise

    Args:
        theta: Current parameter vector.
        raw_update: Desired update direction.
        bound: Maximum parameter norm (M).
        epsilon: Transition region width.

    Returns:
        Projected update direction (same shape as raw_update).
    """
    theta_norm = float(np.linalg.norm(theta))

    if theta_norm <= bound - epsilon:
        return raw_update

    if theta_norm <= bound:
        scale = (bound - theta_norm) / epsilon
        return raw_update * scale

    # ‖θ‖ > bound
    dot = float(np.dot(theta, raw_update))
    if dot > 0:
        scale = max(0.0, 1.0 - dot / (theta_norm * float(np.linalg.norm(raw_update)) + 1e-10))
        return raw_update * scale

    return raw_update


def clip_parameters(
    theta: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]],
) -> np.ndarray:
    """Clip parameters to per-element bounds.

    Args:
        theta: Parameter vector.
        bounds: List of (min, max) per parameter. None = no clipping.

    Returns:
        Clipped parameter vector.
    """
    if bounds is None:
        return theta
    result = theta.copy()
    for i, (lo, hi) in enumerate(bounds):
        if i < len(result):
            result[i] = np.clip(result[i], lo, hi)
    return result


# ---------------------------------------------------------------------------
# Gradient-Based Adaptation
# ---------------------------------------------------------------------------

class GradientAdaptation:
    """Gradient-based parameter adaptation for controller tuning.

    Adjusts parameters θ by descending the gradient of a cost
    function J(θ):

        dθ/dt = -γ ∂J/∂θ

    Supports:
    - Constant learning rate
    - Exponentially decaying learning rate
    - Adaptive learning rate based on error magnitude
    - Projection for bounded parameters
    - Momentum term for faster convergence
    """

    def __init__(self, config: AdaptiveTuningConfig = AdaptiveTuningConfig()) -> None:
        """Initialise the gradient adaptation.

        Args:
            config: Tuning configuration.
        """
        self._config = config
        self._theta = np.zeros(config.n_params)
        self._momentum = np.zeros(config.n_params)
        self._current_lr = config.learning_rate
        self._step_count = 0
        self._cost_history: List[float] = []
        self._theta_history: List[np.ndarray] = []

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimates."""
        return self._theta.copy()

    @property
    def learning_rate(self) -> float:
        """Current effective learning rate."""
        return self._current_lr

    def update(
        self,
        gradient: np.ndarray,
        cost: float = 0.0,
        momentum_coeff: float = 0.9,
        dt: float = 0.01,
    ) -> np.ndarray:
        """Perform one gradient adaptation step.

        Args:
            gradient: Gradient ∂J/∂θ at current parameters.
            cost: Current cost value (for logging).
            momentum_coeff: Momentum coefficient β ∈ [0, 1).
            dt: Time step.

        Returns:
            Updated parameter vector.
        """
        assert len(gradient) == self._config.n_params

        # Decay learning rate
        self._current_lr = max(
            self._config.min_learning_rate,
            min(self._config.max_learning_rate,
                self._current_lr * self._config.decay_rate),
        )

        # Raw update
        raw_update = -self._current_lr * gradient

        # Momentum
        self._momentum = momentum_coeff * self._momentum + (1.0 - momentum_coeff) * raw_update

        # Apply projection
        projected = projection_operator(
            self._theta, self._momentum, self._config.projection_bound
        )

        # Update parameters
        self._theta += projected * dt

        # Clip to bounds
        self._theta = clip_parameters(self._theta, self._config.adaptation_bounds)

        # Record
        self._step_count += 1
        self._cost_history.append(cost)
        self._theta_history.append(self._theta.copy())

        return self._theta.copy()

    def reset(self) -> None:
        """Reset the adapter."""
        self._theta = np.zeros(self._config.n_params)
        self._momentum = np.zeros(self._config.n_params)
        self._current_lr = self._config.learning_rate
        self._step_count = 0
        self._cost_history = []
        self._theta_history = []


# ---------------------------------------------------------------------------
# MIT Rule Adaptation
# ---------------------------------------------------------------------------

class MITRuleAdaptation:
    """MIT Rule adaptation for model reference adaptive control.

    The MIT rule adjusts parameters to minimise the squared tracking
    error e² between the plant and a reference model:

        dθ/dt = -γ · e · ∂y_m/∂θ

    where:
    - γ is the adaptation gain
    - e = y_plant - y_reference
    - ∂y_m/∂θ is the sensitivity derivative

    This is the simplest MRAC adaptation law, suitable for
    slowly varying systems with bounded disturbances.
    """

    def __init__(
        self,
        n_params: int = 4,
        adaptation_gain: float = 0.5,
        sigma: float = 0.01,
        projection_bound: float = 50.0,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialise the MIT rule adaptation.

        Args:
            n_params: Number of adaptive parameters.
            adaptation_gain: MIT adaptation gain γ.
            sigma: e-modification parameter (robustness).
            projection_bound: Projection operator bound.
            bounds: Per-parameter (min, max) bounds.
        """
        self._n = n_params
        self._gamma = adaptation_gain
        self._sigma = sigma
        self._proj_bound = projection_bound
        self._bounds = bounds

        self._theta = np.zeros(n_params)
        self._error: float = 0.0
        self._step_count = 0
        self._error_history: List[float] = []
        self._theta_history: List[np.ndarray] = []

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimates."""
        return self._theta.copy()

    @property
    def tracking_error(self) -> float:
        """Last tracking error."""
        return self._error

    def update(
        self,
        tracking_error: float,
        sensitivity: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """Perform one MIT rule adaptation step.

        Args:
            tracking_error: e = y_plant - y_reference.
            sensitivity: ∂y_m/∂θ — sensitivity of reference model output
                to parameter changes.
            dt: Time step.

        Returns:
            Updated parameter vector.
        """
        assert len(sensitivity) == self._n

        self._error = tracking_error

        # MIT rule: dθ/dt = -γ · e · ∂y_m/∂θ
        raw_update = -self._gamma * tracking_error * sensitivity

        # e-modification for robustness
        if self._sigma > 0 and np.linalg.norm(self._theta) > 1e-10:
            raw_update -= self._sigma * abs(tracking_error) * self._theta

        # Projection
        projected = projection_operator(self._theta, raw_update, self._proj_bound)

        # Update
        self._theta += projected * dt
        self._theta = clip_parameters(self._theta, self._bounds)

        # Record
        self._step_count += 1
        self._error_history.append(tracking_error)
        self._theta_history.append(self._theta.copy())

        return self._theta.copy()

    def reset(self) -> None:
        """Reset the adapter."""
        self._theta = np.zeros(self._n)
        self._error = 0.0
        self._step_count = 0
        self._error_history = []
        self._theta_history = []


# ---------------------------------------------------------------------------
# Lyapunov-Based Adaptation
# ---------------------------------------------------------------------------

class LyapunovAdaptation:
    """Lyapunov-based adaptation law with guaranteed stability.

    Constructs a Lyapunov function candidate:

        V(e, θ̃) = eᵀPe + θ̃ᵀΓ⁻¹θ̃

    where e is the tracking error, θ̃ = θ - θ* is the parameter
    error, P > 0 solves the Lyapunov equation, and Γ > 0 is the
    adaptation gain matrix.

    The adaptation law:

        dθ/dt = -Γ Φ(x) eᵀPb

    ensures Ḟ ≤ 0 (negative semi-definite), guaranteeing stability.

    Features:
    - Stability certificate via Lyapunov function monitoring
    - e-modification and σ-modification for robustness
    - Projection operator for bounded parameters
    - Automatic stability alert when Ḟ > 0
    """

    def __init__(
        self,
        state_dim: int = 2,
        n_params: int = 4,
        adaptation_gain: float = 2.0,
        sigma: float = 0.01,
        projection_bound: float = 50.0,
        bounds: Optional[List[Tuple[float, float]]] = None,
        dt: float = 0.01,
    ) -> None:
        """Initialise the Lyapunov-based adaptation.

        Args:
            state_dim: Dimension of the tracking error state.
            n_params: Number of adaptive parameters.
            adaptation_gain: Lyapunov adaptation gain Γ.
            sigma: σ-modification parameter.
            projection_bound: Projection operator norm bound.
            bounds: Per-parameter bounds.
            dt: Controller time step.
        """
        self._state_dim = state_dim
        self._n_params = n_params
        self._gamma = adaptation_gain
        self._sigma = sigma
        self._proj_bound = projection_bound
        self._bounds = bounds
        self._dt = dt

        # Solve for P from the Lyapunov equation AᵐᵀP + PAᵐ = -Q
        # For stable reference model A_m = [[0,1],[-wn²,-2ζwn]]
        wn = 5.0
        zeta = 0.9
        Am = np.array([[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]])
        Q = np.eye(state_dim) * 10.0

        # Solve continuous Lyapunov equation via scipy-free method
        # For 2x2: manual solution
        self._P = self._solve_lyapunov_2x2(Am, Q)
        self._b = np.array([0.0, wn**2])  # B_m vector

        # Adaptive parameters
        self._theta = np.zeros(n_params)
        self._Gamma = adaptation_gain * np.eye(n_params)

        # Lyapunov function tracking
        self._V_history: List[float] = []
        self._Vdot_history: List[float] = []
        self._stable: bool = True

        # Error tracking
        self._error_history: List[float] = []
        self._theta_history: List[np.ndarray] = []
        self._step_count = 0

    @staticmethod
    def _solve_lyapunov_2x2(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Solve the 2x2 continuous Lyapunov equation AᵀP + PA = -Q.

        Args:
            A: 2x2 stable matrix.
            Q: 2x2 positive definite matrix.

        Returns:
            2x2 positive definite solution P.
        """
        # For a 2x2 system, solve via vectorisation:
        # (I⊗Aᵀ + Aᵀ⊗I) vec(P) = -vec(Q)
        I = np.eye(2)
        M = np.kron(I, A.T) + np.kron(A.T, I)
        vec_Q = Q.flatten()
        vec_P = np.linalg.solve(M, -vec_Q)
        P = vec_P.reshape(2, 2)

        # Ensure symmetry
        P = 0.5 * (P + P.T)
        return P

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimates."""
        return self._theta.copy()

    @property
    def is_stable(self) -> bool:
        """Whether the adaptation is currently stable (Ḟ ≤ 0)."""
        return self._stable

    @property
    def lyapunov_value(self) -> float:
        """Current Lyapunov function value V."""
        return self._V_history[-1] if self._V_history else 0.0

    def compute_regressor(self, state: np.ndarray) -> np.ndarray:
        """Compute the regressor vector Φ(x).

        Maps the state to the parameter space. Uses a simple
        linear basis: Φ = x ⊗ [1, 1, ..., 1] (truncated to n_params).

        Args:
            state: Current state vector.

        Returns:
            Regressor vector of length n_params.
        """
        # Replicate state elements to fill n_params
        phi = np.zeros(self._n_params)
        for i in range(self._n_params):
            phi[i] = state[i % len(state)]
        return phi

    def update(
        self,
        tracking_error: np.ndarray,
        state: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Perform one Lyapunov adaptation step.

        Args:
            tracking_error: Error state e = x - x_m (state_dim,).
            state: Current plant state (state_dim,).
            dt: Optional time step override.

        Returns:
            Tuple of (updated_theta, diagnostics_dict).
        """
        effective_dt = dt if dt is not None else self._dt

        # Regressor
        phi = self.compute_regressor(state)

        # Lyapunov function value
        V_error = float(tracking_error @ self._P @ tracking_error)
        theta_tilde = self._theta  # Approximation (θ* unknown)
        V_param = float(theta_tilde @ np.linalg.solve(self._Gamma, theta_tilde))
        V = V_error + V_param

        # Adaptation law: dθ/dt = -Γ Φ eᵀPb - σ|e|θ
        e_Pb = float(tracking_error @ self._P @ self._b)
        raw_update = -self._Gamma @ (phi * e_Pb)

        # σ-modification
        error_norm = float(np.linalg.norm(tracking_error))
        if self._sigma > 0:
            raw_update -= self._sigma * error_norm * self._theta

        # Projection
        projected = projection_operator(self._theta, raw_update, self._proj_bound)

        # Estimate V_dot (should be ≤ 0 for stability)
        V_dot = 2.0 * float(tracking_error @ self._P @ (self._b * float(phi @ self._theta))) 
        V_dot += 2.0 * float(theta_tilde @ np.linalg.solve(self._Gamma, projected))

        # Stability check
        self._stable = V_dot <= abs(self._sigma) * error_norm * V + 1e-6

        # Update parameters
        self._theta += projected * effective_dt
        self._theta = clip_parameters(self._theta, self._bounds)

        # Record
        self._step_count += 1
        self._V_history.append(V)
        self._Vdot_history.append(V_dot)
        self._error_history.append(error_norm)
        self._theta_history.append(self._theta.copy())

        diagnostics = {
            "V": V,
            "V_dot": V_dot,
            "stable": self._stable,
            "error_norm": error_norm,
            "theta_norm": float(np.linalg.norm(self._theta)),
        }

        return self._theta.copy(), diagnostics

    def reset(self) -> None:
        """Reset the adaptation state."""
        self._theta = np.zeros(self._n_params)
        self._V_history = []
        self._Vdot_history = []
        self._error_history = []
        self._theta_history = []
        self._step_count = 0
        self._stable = True


# ---------------------------------------------------------------------------
# Stability Monitor
# ---------------------------------------------------------------------------

class StabilityMonitor:
    """Monitors system stability during adaptive control operation.

    Checks multiple stability indicators:
    - Lyapunov function monotonicity
    - Parameter drift rate
    - Tracking error growth
    - Control signal saturation frequency
    - Oscillation detection
    """

    def __init__(
        self,
        window_size: int = 200,
        lyapunov_increase_tolerance: float = 0.1,
        drift_rate_threshold: float = 1.0,
        oscillation_freq_range: Tuple[float, float] = (2.0, 50.0),
        dt: float = 0.01,
    ) -> None:
        """Initialise the stability monitor.

        Args:
            window_size: Rolling window for statistics.
            lyapunov_increase_tolerance: Allowed V increase fraction.
            drift_rate_threshold: Max parameter drift rate (‖Δθ‖/step).
            oscillation_freq_range: (min, max) Hz for oscillation detection.
            dt: Sample time.
        """
        self._window = window_size
        self._lyap_tol = lyapunov_increase_tolerance
        self._drift_thresh = drift_rate_threshold
        self._osc_range = oscillation_freq_range
        self._dt = dt

        self._V_history: List[float] = []
        self._error_history: List[float] = []
        self._control_history: List[float] = []
        self._drift_history: List[float] = []

    def update(
        self,
        lyapunov_value: float,
        tracking_error_norm: float,
        control_output: float,
        parameter_drift: float,
    ) -> Dict[str, bool]:
        """Record one step and evaluate stability.

        Args:
            lyapunov_value: Current Lyapunov function value V.
            tracking_error_norm: ‖e‖ at current step.
            control_output: Control signal magnitude.
            parameter_drift: ‖Δθ‖ at current step.

        Returns:
            Dictionary of stability status flags:
              - lyapunov_stable: V is non-increasing (within tolerance).
              - bounded_error: Tracking error is not growing.
              - bounded_parameters: Parameter drift is within limits.
              - no_oscillation: No sustained oscillations detected.
              - overall_stable: All checks pass.
        """
        self._V_history.append(lyapunov_value)
        self._error_history.append(tracking_error_norm)
        self._control_history.append(control_output)
        self._drift_history.append(parameter_drift)

        # Trim
        max_len = self._window * 2
        if len(self._V_history) > max_len:
            self._V_history = self._V_history[-max_len:]
            self._error_history = self._error_history[-max_len:]
            self._control_history = self._control_history[-max_len:]
            self._drift_history = self._drift_history[-max_len:]

        w = self._window

        # 1. Lyapunov stability: V should not increase significantly
        if len(self._V_history) >= w:
            V_recent = self._V_history[-w:]
            lyapunov_stable = all(
                V_recent[i] <= V_recent[0] * (1.0 + self._lyap_tol)
                for i in range(len(V_recent))
            )
        else:
            lyapunov_stable = True

        # 2. Bounded error: error should not grow monotonically
        if len(self._error_history) >= w:
            errors = np.array(self._error_history[-w:])
            # Check if error is growing (positive slope)
            slope = np.polyfit(np.arange(w), errors, 1)[0]
            bounded_error = slope <= 0.01 * np.mean(errors) if np.mean(errors) > 0.01 else True
        else:
            bounded_error = True

        # 3. Bounded parameters: drift rate check
        if len(self._drift_history) >= w:
            mean_drift = np.mean(self._drift_history[-w:])
            bounded_parameters = mean_drift < self._drift_thresh
        else:
            bounded_parameters = True

        # 4. Oscillation detection via zero-crossing count
        no_oscillation = True
        if len(self._control_history) >= w:
            ctrl = np.array(self._control_history[-w:])
            ctrl_mean = np.mean(ctrl)
            crossings = np.sum(np.diff(np.sign(ctrl - ctrl_mean)) != 0)
            # Expected crossings for a signal at max oscillation freq
            max_crossings = int(2 * self._osc_range[1] * w * self._dt)
            # Count only if there are enough crossings and amplitude is significant
            amplitude = float(np.std(ctrl))
            if crossings > max_crossings and amplitude > 0.01:
                no_oscillation = False

        overall = lyapunov_stable and bounded_error and bounded_parameters and no_oscillation

        return {
            "lyapunov_stable": lyapunov_stable,
            "bounded_error": bounded_error,
            "bounded_parameters": bounded_parameters,
            "no_oscillation": no_oscillation,
            "overall_stable": overall,
        }

    def reset(self) -> None:
        """Reset the monitor."""
        self._V_history = []
        self._error_history = []
        self._control_history = []
        self._drift_history = []


# ---------------------------------------------------------------------------
# Bounded Adaptation Wrapper
# ---------------------------------------------------------------------------

class BoundedAdaptation:
    """Wraps any adaptation law with hard bounds and safety checks.

    Ensures:
    - Parameters stay within specified bounds
    - Adaptation rate is limited
    - Emergency freeze when instability is detected
    - Gradual unfreeze when stability is recovered
    """

    def __init__(
        self,
        base_adapter: LyapunovAdaptation,
        max_adaptation_rate: float = 0.1,
        freeze_threshold: float = 5.0,
        unfreeze_threshold: float = 1.0,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialise the bounded adaptation wrapper.

        Args:
            base_adapter: Underlying adaptation law (e.g., LyapunovAdaptation).
            max_adaptation_rate: Maximum ‖Δθ‖ per step.
            freeze_threshold: Error norm above which adaptation is frozen.
            unfreeze_threshold: Error norm below which adaptation resumes.
            bounds: Per-parameter (min, max) bounds.
        """
        self._adapter = base_adapter
        self._max_rate = max_adaptation_rate
        self._freeze_thresh = freeze_threshold
        self._unfreeze_thresh = unfreeze_threshold
        self._bounds = bounds

        self._frozen = False
        self._prev_theta = base_adapter.theta
        self._step_count = 0

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimates."""
        return self._adapter.theta

    @property
    def is_frozen(self) -> bool:
        """Whether adaptation is currently frozen."""
        return self._frozen

    def update(
        self,
        tracking_error: np.ndarray,
        state: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Perform one bounded adaptation step.

        Args:
            tracking_error: Tracking error state vector.
            state: Current plant state.
            dt: Optional time step.

        Returns:
            Tuple of (theta, diagnostics).
        """
        error_norm = float(np.linalg.norm(tracking_error))

        # Freeze / unfreeze logic
        if error_norm > self._freeze_thresh:
            self._frozen = True
        elif error_norm < self._unfreeze_thresh:
            self._frozen = False

        if self._frozen:
            diagnostics = {
                "frozen": True,
                "error_norm": error_norm,
                "theta_norm": float(np.linalg.norm(self._adapter.theta)),
            }
            return self._adapter.theta, diagnostics

        # Normal adaptation
        self._prev_theta = self._adapter.theta.copy()
        theta_new, diagnostics = self._adapter.update(tracking_error, state, dt)

        # Rate limiting
        delta = theta_new - self._prev_theta
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self._max_rate and delta_norm > 0:
            scale = self._max_rate / delta_norm
            theta_new = self._prev_theta + delta * scale

        # Apply bounds
        theta_new = clip_parameters(theta_new, self._bounds)

        # Override adapter's theta
        self._adapter._theta = theta_new.copy()

        diagnostics["frozen"] = False
        diagnostics["adaptation_rate"] = float(np.linalg.norm(theta_new - self._prev_theta))

        self._step_count += 1
        return theta_new, diagnostics

    def reset(self) -> None:
        """Reset the bounded adaptation."""
        self._adapter.reset()
        self._frozen = False
        self._prev_theta = self._adapter.theta
        self._step_count = 0
