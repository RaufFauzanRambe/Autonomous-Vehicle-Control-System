"""
Online System Identifier for Autonomous Vehicle Control

Implements recursive parameter estimation algorithms:
  - Recursive Least Squares (RLS) with forgetting factor
  - Extended Least Squares (ELS) for ARMAX models
  - Model order selection via AIC / BIC
  - Convergence monitoring and covariance condition tracking
  - Parameter estimation for lateral and longitudinal vehicle dynamics

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Recursive Least Squares (RLS)
# ---------------------------------------------------------------------------

@dataclass
class RLSConfig:
    """Configuration for the Recursive Least Squares estimator.

    Attributes:
        n_params: Number of parameters to estimate.
        forgetting_factor: Exponential forgetting factor λ ∈ (0, 1].
            λ = 1.0  → no forgetting (grows covariance over time).
            λ < 1.0  → discounts old data; typical 0.95–0.999.
        initial_covariance: Diagonal of the initial P-matrix (P₀ = δ·I).
        covariance_floor: Minimum eigenvalue floor for P (prevents singularity).
        parameter_bounds: Optional (min, max) clip bounds per parameter.
        excitation_threshold: Minimum persistency-of-excitation norm.
    """
    n_params: int = 4
    forgetting_factor: float = 0.995
    initial_covariance: float = 1000.0
    covariance_floor: float = 1e-6
    parameter_bounds: Optional[List[Tuple[float, float]]] = None
    excitation_threshold: float = 1e-4


class RecursiveLeastSquares:
    """Recursive Least Squares estimator with forgetting factor.

    Estimates parameters θ of the linear model:

        y(t) = φ(t)ᵀ θ + ε(t)

    using the update:

        K(t) = P(t-1) φ(t) / (λ + φ(t)ᵀ P(t-1) φ(t))
        θ(t) = θ(t-1) + K(t) (y(t) - φ(t)ᵀ θ(t-1))
        P(t) = (I - K(t) φ(t)ᵀ) P(t-1) / λ

    Attributes:
        theta: Current parameter estimate vector.
        P: Covariance matrix.
        error: Last prediction error.
    """

    def __init__(self, config: RLSConfig = RLSConfig()) -> None:
        """Initialise the RLS estimator.

        Args:
            config: RLS configuration.
        """
        self._config = config
        self._n = config.n_params
        self._lambda = config.forgetting_factor

        # Parameter estimate
        self._theta = np.zeros(self._n)

        # Covariance matrix
        self._P = config.initial_covariance * np.eye(self._n)

        # State tracking
        self._error: float = 0.0
        self._gain: np.ndarray = np.zeros(self._n)
        self._step_count: int = 0
        self._error_history: List[float] = []
        self._theta_history: List[np.ndarray] = []
        self._trace_history: List[float] = []

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimates."""
        return self._theta.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current covariance matrix."""
        return self._P.copy()

    @property
    def error(self) -> float:
        """Last prediction error."""
        return self._error

    @property
    def step_count(self) -> int:
        """Number of update steps performed."""
        return self._step_count

    def predict(self, phi: np.ndarray) -> float:
        """Compute one-step-ahead prediction.

        Args:
            phi: Regressor vector at current time step.

        Returns:
            Predicted output ŷ(t) = φᵀ θ.
        """
        return float(phi @ self._theta)

    def update(self, y: float, phi: np.ndarray) -> np.ndarray:
        """Perform one RLS update step.

        Args:
            y: Measured output.
            phi: Regressor vector (same dimension as theta).

        Returns:
            Updated parameter estimate.
        """
        assert len(phi) == self._n, (
            f"Regressor dimension {len(phi)} != parameter dimension {self._n}"
        )

        # Prediction error
        y_hat = self.predict(phi)
        self._error = y - y_hat

        # Kalman gain
        denom = self._lambda + phi @ self._P @ phi
        if abs(denom) < 1e-15:
            denom = 1e-15
        self._gain = (self._P @ phi) / denom

        # Parameter update
        self._theta = self._theta + self._gain * self._error

        # Apply parameter bounds if configured
        if self._config.parameter_bounds is not None:
            for i, (lo, hi) in enumerate(self._config.parameter_bounds):
                self._theta[i] = np.clip(self._theta[i], lo, hi)

        # Covariance update (Joseph form for numerical stability)
        K_phi = np.outer(self._gain, phi)
        I_Kphi = np.eye(self._n) - K_phi
        self._P = (I_Kphi @ self._P @ I_Kphi.T + self._gain @ self._gain.T * self._error**2)
        self._P /= self._lambda

        # Apply covariance floor to prevent singularity
        eigvals = np.linalg.eigvalsh(self._P)
        if np.min(eigvals) < self._config.covariance_floor:
            self._P += (self._config.covariance_floor - np.min(eigvals) + 1e-8) * np.eye(self._n)

        # Record history
        self._step_count += 1
        self._error_history.append(self._error)
        self._theta_history.append(self._theta.copy())
        self._trace_history.append(float(np.trace(self._P)))

        return self._theta.copy()

    def reset(self) -> None:
        """Reset the estimator to initial conditions."""
        self._theta = np.zeros(self._n)
        self._P = self._config.initial_covariance * np.eye(self._n)
        self._error = 0.0
        self._gain = np.zeros(self._n)
        self._step_count = 0
        self._error_history = []
        self._theta_history = []
        self._trace_history = []

    def get_diagnostics(self) -> Dict:
        """Return diagnostic information.

        Returns:
            Dictionary with estimation diagnostics.
        """
        trace_P = float(np.trace(self._P))
        cond_P = float(np.linalg.cond(self._P)) if self._step_count > 0 else float("inf")
        return {
            "step_count": self._step_count,
            "last_error": self._error,
            "trace_P": trace_P,
            "cond_P": cond_P,
            "theta_norm": float(np.linalg.norm(self._theta)),
            "theta": self._theta.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"RecursiveLeastSquares(n={self._n}, "
            f"lambda={self._lambda}, steps={self._step_count})"
        )


# ---------------------------------------------------------------------------
# Extended Least Squares (for ARMAX models)
# ---------------------------------------------------------------------------

class ExtendedLeastSquares:
    """Extended Least Squares estimator for ARMAX-type models.

    Identifies models of the form:

        A(q⁻¹) y(t) = B(q⁻¹) u(t) + C(q⁻¹) e(t)

    by extending the regressor with past prediction errors.

    Attributes:
        theta: Current parameter vector [a₁..aₙₐ, b₀..bₙ_b, c₁..cₙ_c].
    """

    def __init__(
        self,
        na: int = 2,
        nb: int = 2,
        nc: int = 1,
        nk: int = 1,
        forgetting_factor: float = 0.995,
        initial_covariance: float = 1000.0,
    ) -> None:
        """Initialise the ELS estimator.

        Args:
            na: Order of A polynomial.
            nb: Order of B polynomial (number of b parameters).
            nc: Order of C polynomial.
            nk: Input delay (dead time in samples).
            forgetting_factor: RLS forgetting factor.
            initial_covariance: Initial P-matrix diagonal.
        """
        self._na = na
        self._nb = nb
        self._nc = nc
        self._nk = nk
        self._n_params = na + nb + nc

        self._rls = RecursiveLeastSquares(RLSConfig(
            n_params=self._n_params,
            forgetting_factor=forgetting_factor,
            initial_covariance=initial_covariance,
        ))

        # Data buffers
        self._y_buf: List[float] = [0.0] * na
        self._u_buf: List[float] = [0.0] * (nb + nk)
        self._e_buf: List[float] = [0.0] * nc

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimates."""
        return self._rls.theta

    def _build_regressor(self) -> np.ndarray:
        """Construct the extended regressor vector.

        Returns:
            φ = [-y(t-1), ..., -y(t-na), u(t-nk), ..., u(t-nk-nb+1),
                  e(t-1), ..., e(t-nc)]
        """
        phi_parts: List[float] = []

        # Past outputs (negated for A-polynomial)
        for i in range(self._na):
            phi_parts.append(-self._y_buf[i] if i < len(self._y_buf) else 0.0)

        # Past inputs
        for i in range(self._nb):
            idx = self._nk + i
            phi_parts.append(self._u_buf[idx] if idx < len(self._u_buf) else 0.0)

        # Past errors (C-polynomial)
        for i in range(self._nc):
            phi_parts.append(self._e_buf[i] if i < len(self._e_buf) else 0.0)

        return np.array(phi_parts)

    def update(self, y: float, u: float) -> np.ndarray:
        """Perform one ELS update step.

        Args:
            y: Current measured output.
            u: Current input value.

        Returns:
            Updated parameter vector.
        """
        phi = self._build_regressor()

        # RLS update
        self._rls.update(y, phi)

        # Update buffers
        self._y_buf.insert(0, y)
        self._y_buf = self._y_buf[:self._na]

        self._u_buf.insert(0, u)
        self._u_buf = self._u_buf[:self._nb + self._nk]

        self._e_buf.insert(0, self._rls.error)
        self._e_buf = self._e_buf[:self._nc]

        return self._rls.theta

    def get_transfer_function(self) -> Dict[str, List[float]]:
        """Extract identified A, B, C polynomial coefficients.

        Returns:
            Dictionary with keys 'A', 'B', 'C' and coefficient lists.
            A = [1, a₁, a₂, ...], B = [0..nk, b₀, b₁, ...], C = [1, c₁, ...]
        """
        theta = self._rls.theta
        a_coeffs = [1.0] + [-theta[i] for i in range(self._na)]
        b_coeffs = [0.0] * self._nk + [theta[self._na + i] for i in range(self._nb)]
        c_coeffs = [1.0] + [theta[self._na + self._nb + i] for i in range(self._nc)]
        return {"A": a_coeffs, "B": b_coeffs, "C": c_coeffs}

    def reset(self) -> None:
        """Reset the estimator."""
        self._rls.reset()
        self._y_buf = [0.0] * self._na
        self._u_buf = [0.0] * (self._nb + self._nk)
        self._e_buf = [0.0] * self._nc


# ---------------------------------------------------------------------------
# Model Order Selection
# ---------------------------------------------------------------------------

@dataclass
class ModelOrderResult:
    """Result of model order selection.

    Attributes:
        best_na: Selected A-polynomial order.
        best_nb: Selected B-polynomial order.
        best_nk: Selected delay.
        aic_values: Dict mapping (na,nb) to AIC.
        bic_values: Dict mapping (na,nb) to BIC.
    """
    best_na: int = 1
    best_nb: int = 1
    best_nk: int = 1
    aic_values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    bic_values: Dict[Tuple[int, int], float] = field(default_factory=dict)


def select_model_order(
    y_data: np.ndarray,
    u_data: np.ndarray,
    na_range: range = range(1, 5),
    nb_range: range = range(1, 5),
    nk_range: range = range(1, 3),
    forgetting_factor: float = 0.995,
) -> ModelOrderResult:
    """Select the best ARX model order using AIC and BIC.

    Fits ARX models of the form A(q⁻¹)y = B(q⁻¹)u + e for
    various (na, nb, nk) combinations and selects the order that
    minimises both AIC and BIC (BIC preference for parsimony).

    AIC = N * ln(σ²) + 2k
    BIC = N * ln(σ²) + k * ln(N)

    where σ² is the residual variance, N is data length, k = na + nb.

    Args:
        y_data: Output data array.
        u_data: Input data array.
        na_range: Range of A-polynomial orders to test.
        nb_range: Range of B-polynomial orders to test.
        nk_range: Range of delays to test.
        forgetting_factor: RLS forgetting factor.

    Returns:
        ModelOrderResult with selected orders and criterion values.
    """
    N = len(y_data)
    result = ModelOrderResult()
    best_bic = float("inf")
    best_aic = float("inf")
    best_bic_order = (1, 1, 1)
    best_aic_order = (1, 1, 1)

    for nk in nk_range:
        for na in na_range:
            for nb in nb_range:
                n_params = na + nb
                rls = RecursiveLeastSquares(RLSConfig(
                    n_params=n_params,
                    forgetting_factor=forgetting_factor,
                ))

                errors_sq: List[float] = []
                y_buf = [0.0] * na
                u_buf = [0.0] * (nb + nk)

                for t in range(N):
                    # Build regressor
                    phi_parts: List[float] = []
                    for i in range(na):
                        phi_parts.append(-y_buf[i])
                    for i in range(nb):
                        idx = nk + i
                        phi_parts.append(u_buf[idx] if idx < len(u_buf) else 0.0)
                    phi = np.array(phi_parts)

                    rls.update(y_data[t], phi)
                    errors_sq.append(rls.error ** 2)

                    # Update buffers
                    y_buf.insert(0, y_data[t])
                    y_buf = y_buf[:na]
                    u_buf.insert(0, u_data[t])
                    u_buf = u_buf[:nb + nk]

                # Compute criteria on last 50% of data
                start = N // 2
                residual_var = np.mean(errors_sq[start:]) if start < N else 1e10
                if residual_var < 1e-15:
                    residual_var = 1e-15

                k = n_params
                aic = N * math.log(residual_var) + 2 * k
                bic = N * math.log(residual_var) + k * math.log(N)

                result.aic_values[(na, nb)] = aic
                result.bic_values[(na, nb)] = bic

                if bic < best_bic:
                    best_bic = bic
                    best_bic_order = (na, nb, nk)
                if aic < best_aic:
                    best_aic = aic
                    best_aic_order = (na, nb, nk)

    # Prefer BIC-selected order (more conservative)
    result.best_na = best_bic_order[0]
    result.best_nb = best_bic_order[1]
    result.best_nk = best_bic_order[2]

    return result


# ---------------------------------------------------------------------------
# Convergence Monitor
# ---------------------------------------------------------------------------

class ConvergenceMonitor:
    """Monitors parameter convergence for online estimators.

    Tracks:
    - Parameter drift rate (‖Δθ‖ per step).
    - Prediction error statistics (mean, variance).
    - Covariance trace decay.
    - Excitation level (φᵀPφ).

    Raises alerts when parameters have converged or when estimation
    is unreliable (low excitation or diverging).
    """

    def __init__(
        self,
        window_size: int = 100,
        convergence_threshold: float = 1e-4,
        divergence_threshold: float = 10.0,
        low_excitation_threshold: float = 1e-6,
    ) -> None:
        """Initialise the convergence monitor.

        Args:
            window_size: Rolling window size for statistics.
            convergence_threshold: ‖Δθ‖ threshold below which params are converged.
            divergence_threshold: Error threshold above which estimation diverges.
            low_excitation_threshold: φᵀPφ threshold for low excitation alert.
        """
        self._window = window_size
        self._conv_thresh = convergence_threshold
        self._div_thresh = divergence_threshold
        self._exc_thresh = low_excitation_threshold

        self._drift_history: List[float] = []
        self._error_history: List[float] = []
        self._excitation_history: List[float] = []

    def update(
        self,
        theta: np.ndarray,
        prev_theta: np.ndarray,
        prediction_error: float,
        excitation: float = 0.0,
    ) -> Dict[str, bool]:
        """Record one estimation step and check convergence status.

        Args:
            theta: Current parameter vector.
            prev_theta: Previous parameter vector.
            prediction_error: Current prediction error.
            excitation: Current excitation level (φᵀPφ).

        Returns:
            Dictionary of status flags:
              - converged: Parameters have converged.
              - diverging: Error is growing unboundedly.
              - low_excitation: Insufficient excitation for reliable estimation.
        """
        drift = float(np.linalg.norm(theta - prev_theta))
        self._drift_history.append(drift)
        self._error_history.append(prediction_error)
        self._excitation_history.append(excitation)

        # Trim histories
        if len(self._drift_history) > self._window * 2:
            self._drift_history = self._drift_history[-self._window:]
            self._error_history = self._error_history[-self._window:]
            self._excitation_history = self._excitation_history[-self._window:]

        # Convergence check: mean drift over window below threshold
        window_drift = self._drift_history[-self._window:]
        mean_drift = np.mean(window_drift) if window_drift else float("inf")
        converged = mean_drift < self._conv_thresh

        # Divergence check: error magnitude
        window_errors = self._error_history[-self._window:]
        rms_error = math.sqrt(np.mean(np.array(window_errors) ** 2)) if window_errors else 0.0
        diverging = rms_error > self._div_thresh

        # Excitation check
        window_exc = self._excitation_history[-self._window:]
        mean_exc = np.mean(window_exc) if window_exc else 0.0
        low_excitation = mean_exc < self._exc_thresh

        return {
            "converged": converged,
            "diverging": diverging,
            "low_excitation": low_excitation,
        }

    def reset(self) -> None:
        """Reset monitor history."""
        self._drift_history = []
        self._error_history = []
        self._excitation_history = []


# ---------------------------------------------------------------------------
# Vehicle Dynamics Identifier
# ---------------------------------------------------------------------------

@dataclass
class VehicleDynamicsModel:
    """Identified vehicle dynamics model parameters.

    Attributes:
        lateral_stiffness_front: Front cornering stiffness (N/rad).
        lateral_stiffness_rear: Rear cornering stiffness (N/rad).
        yaw_inertia: Yaw moment of inertia (kg·m²).
        mass: Vehicle mass (kg).
        drag_coefficient: Aerodynamic drag coefficient.
        rolling_resistance: Rolling resistance coefficient.
    """
    lateral_stiffness_front: float = 80000.0
    lateral_stiffness_rear: float = 90000.0
    yaw_inertia: float = 3500.0
    mass: float = 1800.0
    drag_coefficient: float = 0.3
    rolling_resistance: float = 0.015


class LateralDynamicsIdentifier:
    """Online identifier for lateral vehicle dynamics.

    Identifies the bicycle model parameters from measured data:
        ẏ = a₁₁·v_y + a₁₂·r + b₁·δ
        ṙ = a₂₁·v_y + a₂₂·r + b₂·δ

    where v_y is lateral velocity, r is yaw rate, δ is steering angle.

    Uses RLS to estimate the four state-space matrix entries from
    measured lateral acceleration and yaw rate data.
    """

    def __init__(
        self,
        dt: float = 0.01,
        forgetting_factor: float = 0.995,
        initial_mass: float = 1800.0,
        initial_wheelbase: float = 2.85,
        cg_to_front: float = 1.42,
        cg_to_rear: float = 1.43,
    ) -> None:
        """Initialise the lateral dynamics identifier.

        Args:
            dt: Sample time in seconds.
            forgetting_factor: RLS forgetting factor.
            initial_mass: Initial vehicle mass estimate (kg).
            initial_wheelbase: Wheelbase (m).
            cg_to_front: CG to front axle distance (m).
            cg_to_rear: CG to rear axle distance (m).
        """
        self._dt = dt
        self._mass = initial_mass
        self._wheelbase = initial_wheelbase
        self._cg_to_front = cg_to_front
        self._cg_to_rear = cg_to_rear

        # We estimate 4 rows of the state-space model
        # Regressor: [v_y, r, δ] → predict [a_y] and [ṙ]
        # Lateral: a_y ≈ a11*v_y + a12*r + b1*δ
        self._rls_lateral = RecursiveLeastSquares(RLSConfig(
            n_params=3,
            forgetting_factor=forgetting_factor,
            initial_covariance=1000.0,
        ))
        # Yaw: ṙ ≈ a21*v_y + a22*r + b2*δ
        self._rls_yaw = RecursiveLeastSquares(RLSConfig(
            n_params=3,
            forgetting_factor=forgetting_factor,
            initial_covariance=1000.0,
        ))

        self._convergence_monitor = ConvergenceMonitor()
        self._prev_theta_lat = np.zeros(3)
        self._prev_theta_yaw = np.zeros(3)
        self._identified_model: Optional[VehicleDynamicsModel] = None

    def update(
        self,
        lateral_velocity: float,
        yaw_rate: float,
        steering_angle: float,
        lateral_acceleration: float,
        yaw_acceleration: float,
    ) -> Dict:
        """Update the lateral dynamics estimates.

        Args:
            lateral_velocity: Measured lateral velocity (m/s).
            yaw_rate: Measured yaw rate (rad/s).
            steering_angle: Steering angle (rad).
            lateral_acceleration: Measured lateral acceleration (m/s²).
            yaw_acceleration: Measured yaw acceleration (rad/s²).

        Returns:
            Dictionary with current parameter estimates and convergence flags.
        """
        phi = np.array([lateral_velocity, yaw_rate, steering_angle])

        # Save previous theta
        self._prev_theta_lat = self._rls_lateral.theta.copy()
        self._prev_theta_yaw = self._rls_yaw.theta.copy()

        # Update RLS
        theta_lat = self._rls_lateral.update(lateral_acceleration, phi)
        theta_yaw = self._rls_yaw.update(yaw_acceleration, phi)

        # Convergence monitoring
        excitation = float(phi @ self._rls_lateral.covariance @ phi)
        status = self._convergence_monitor.update(
            theta=np.concatenate([theta_lat, theta_yaw]),
            prev_theta=np.concatenate([self._prev_theta_lat, self._prev_theta_yaw]),
            prediction_error=self._rls_lateral.error,
            excitation=excitation,
        )

        # Build identified model from converged parameters
        if status["converged"]:
            self._identified_model = self._build_model(theta_lat, theta_yaw)

        return {
            "lateral_params": theta_lat.tolist(),
            "yaw_params": theta_yaw.tolist(),
            "lateral_error": self._rls_lateral.error,
            "yaw_error": self._rls_yaw.error,
            "converged": status["converged"],
            "diverging": status["diverging"],
            "low_excitation": status["low_excitation"],
        }

    def _build_model(
        self,
        theta_lat: np.ndarray,
        theta_yaw: np.ndarray,
    ) -> VehicleDynamicsModel:
        """Construct a VehicleDynamicsModel from identified parameters.

        Args:
            theta_lat: [a11, a12, b1] from lateral equation.
            theta_yaw: [a21, a22, b2] from yaw equation.

        Returns:
            Identified vehicle dynamics model.
        """
        # Extract front/rear cornering stiffness from state-space entries
        # a11 = -(Cf + Cr) / (m * vx), b1 = Cf / (m * vx)
        # a21 = -(lf*Cf - lr*Cr) / (Iz * vx), b2 = lf*Cf / (Iz * vx)
        # Assuming nominal vx = 15 m/s
        vx_nominal = 15.0

        b1 = theta_lat[2]
        Cf_est = abs(b1 * self._mass * vx_nominal)

        b2 = theta_yaw[2]
        lf_Cf = abs(b2 * 3500.0 * vx_nominal)  # Using default Iz

        Cr_est = max(0.0, (Cf_est * self._cg_to_front - (lf_Cf - Cf_est * self._cg_to_front))
                     / self._cg_to_rear) if self._cg_to_rear > 0 else 0.0

        return VehicleDynamicsModel(
            lateral_stiffness_front=max(Cf_est, 1000.0),
            lateral_stiffness_rear=max(Cr_est, 1000.0),
            mass=self._mass,
        )

    @property
    def identified_model(self) -> Optional[VehicleDynamicsModel]:
        """Return the identified model (None if not yet converged)."""
        return self._identified_model

    def reset(self) -> None:
        """Reset the identifier."""
        self._rls_lateral.reset()
        self._rls_yaw.reset()
        self._convergence_monitor.reset()
        self._prev_theta_lat = np.zeros(3)
        self._prev_theta_yaw = np.zeros(3)
        self._identified_model = None

    def __repr__(self) -> str:
        return f"LateralDynamicsIdentifier(steps_lat={self._rls_lateral.step_count}, steps_yaw={self._rls_yaw.step_count})"
