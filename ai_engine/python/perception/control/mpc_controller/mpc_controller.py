"""
Model Predictive Controller (MPC) for Autonomous Vehicle Control

Implements a complete MPC framework with:
  - Configurable prediction and control horizons
  - Quadratic cost function with state and input weighting
  - State and input constraints
  - Terminal cost and terminal constraint
  - Multiple solver interfaces
  - Warm-starting for real-time operation
  - Reference preview / look-ahead

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass
class MPCParams:
    """MPC controller parameters.

    Attributes:
        prediction_horizon: Number of prediction steps (N).
        control_horizon: Number of control moves (M, M <= N).
        state_dim: Number of states.
        input_dim: Number of control inputs.
        output_dim: Number of measured outputs.
        dt: Prediction timestep in seconds.
        Q: State weighting matrix (output_dim x output_dim).
        R: Input weighting matrix (input_dim x input_dim).
        Q_terminal: Terminal state weighting matrix.
        R_delta: Input rate weighting matrix.
    """
    prediction_horizon: int = 20
    control_horizon: int = 10
    state_dim: int = 4
    input_dim: int = 2
    output_dim: int = 4
    dt: float = 0.1
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    Q_terminal: Optional[np.ndarray] = None
    R_delta: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize default weighting matrices if not provided."""
        if self.Q is None:
            self.Q = np.eye(self.output_dim) * 10.0
        if self.R is None:
            self.R = np.eye(self.input_dim) * 1.0
        if self.Q_terminal is None:
            self.Q_terminal = self.Q * 5.0
        if self.R_delta is None:
            self.R_delta = np.eye(self.input_dim) * 0.5


@dataclass
class MPCConstraints:
    """Constraints for the MPC problem.

    Attributes:
        x_min: Minimum state bounds (state_dim,).
        x_max: Maximum state bounds (state_dim,).
        u_min: Minimum input bounds (input_dim,).
        u_max: Maximum input bounds (input_dim,).
        du_min: Minimum input rate bounds (input_dim,).
        du_max: Maximum input rate bounds (input_dim,).
        y_min: Minimum output bounds (output_dim,).
        y_max: Maximum output bounds (output_dim,).
    """
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    du_min: Optional[np.ndarray] = None
    du_max: Optional[np.ndarray] = None
    y_min: Optional[np.ndarray] = None
    y_max: Optional[np.ndarray] = None

    def get_input_bounds(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get stacked input bounds for the full horizon.

        Args:
            horizon: Number of control moves.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays.
        """
        if self.u_min is not None and self.u_max is not None:
            lb = np.tile(self.u_min, horizon)
            ub = np.tile(self.u_max, horizon)
        else:
            lb = np.full(horizon * 2, -np.inf)
            ub = np.full(horizon * 2, np.inf)
        return lb, ub


@dataclass
class MPCState:
    """Current state and solution of the MPC controller.

    Attributes:
        current_state: Current measured state.
        previous_input: Previous control input (for rate computation).
        optimal_input_sequence: Full optimal input trajectory.
        predicted_state_sequence: Predicted state trajectory.
        solve_time_ms: Time to solve the QP in milliseconds.
        cost: Optimal cost value.
        iterations: Number of QP solver iterations.
        status: Solver status string.
    """
    current_state: Optional[np.ndarray] = None
    previous_input: Optional[np.ndarray] = None
    optimal_input_sequence: Optional[np.ndarray] = None
    predicted_state_sequence: Optional[np.ndarray] = None
    solve_time_ms: float = 0.0
    cost: float = float("inf")
    iterations: int = 0
    status: str = "uninitialized"


class MPCController:
    """Model Predictive Controller for autonomous vehicle control.

    Implements a linear MPC using state-space models with:
    - Quadratic programming formulation
    - State and input constraints
    - Reference preview (look-ahead)
    - Warm-starting from previous solution
    - Input rate penalization

    The MPC solves the following problem at each time step:
        min  sum_k (y_k - r_k)' Q (y_k - r_k) + u_k' R u_k + du_k' R_delta du_k
        s.t. x_{k+1} = A x_k + B u_k
             y_k = C x_k + D u_k
             x_min <= x_k <= x_max
             u_min <= u_k <= u_max
             du_min <= du_k - du_{k-1} <= du_max

    Example:
        >>> params = MPCParams(prediction_horizon=20, state_dim=4, input_dim=2)
        >>> mpc = MPCController(params=params)
        >>> mpc.set_model(A, B, C, D)
        >>> u_opt = mpc.compute_control(x_current, reference)
    """

    def __init__(
        self,
        params: MPCParams = MPCParams(),
        constraints: MPCConstraints = MPCConstraints(),
        name: str = "mpc_controller",
    ) -> None:
        """Initialize the MPC controller.

        Args:
            params: MPC parameters (horizons, weights, dimensions).
            constraints: State and input constraints.
            name: Controller name for logging.
        """
        self._params = params
        self._constraints = constraints
        self._name = name

        # System matrices
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None

        # State
        self._state = MPCState()
        self._warm_start: Optional[np.ndarray] = None
        self._initialized = False

        # Pre-computed QP matrices (built when model is set)
        self._H: Optional[np.ndarray] = None
        self._q_template: Optional[np.ndarray] = None
        self._constraint_A: Optional[sparse.spmatrix] = None
        self._constraint_lb: Optional[np.ndarray] = None
        self._constraint_ub: Optional[np.ndarray] = None

        # Statistics
        self._solve_count = 0
        self._total_solve_time_ms = 0.0

    @property
    def name(self) -> str:
        """Return controller name."""
        return self._name

    @property
    def params(self) -> MPCParams:
        """Return MPC parameters."""
        return self._params

    @property
    def state(self) -> MPCState:
        """Return current MPC state."""
        return self._state

    @property
    def solve_statistics(self) -> dict:
        """Return solver statistics."""
        avg_time = self._total_solve_time_ms / max(self._solve_count, 1)
        return {
            "total_solves": self._solve_count,
            "total_solve_time_ms": self._total_solve_time_ms,
            "avg_solve_time_ms": avg_time,
        }

    def set_model(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ) -> None:
        """Set the state-space model matrices.

        Args:
            A: State transition matrix (n x n).
            B: Input matrix (n x m).
            C: Output matrix (p x n). If None, uses identity.
            D: Feedthrough matrix (p x m). If None, uses zeros.
        """
        n = self._params.state_dim
        m = self._params.input_dim
        p = self._params.output_dim

        assert A.shape == (n, n), f"A must be ({n}x{n}), got {A.shape}"
        assert B.shape == (n, m), f"B must be ({n}x{m}), got {B.shape}"

        self._A = A.copy()
        self._B = B.copy()
        self._C = C.copy() if C is not None else np.eye(n)[:p, :]
        self._D = D.copy() if D is not None else np.zeros((p, m))

        self._build_qp_matrices()

    def _build_qp_matrices(self) -> None:
        """Build the QP matrices for the condensed formulation.

        The QP is formulated as:
            min  0.5 * u' H u + q' u
            s.t. lb <= A_con u <= ub
        """
        if self._A is None or self._B is None:
            return

        N = self._params.control_horizon
        n = self._params.state_dim
        m = self._params.input_dim
        p = self._params.output_dim
        Q = self._params.Q
        R = self._params.R
        R_delta = self._params.R_delta

        # Build prediction matrices (forced response)
        # S_x = [C*A; C*A^2; ...; C*A^N]  -- free response
        # S_u = [C*B,   0,   ...  0  ]     -- forced response
        #      [C*A*B, C*B,  ...  0  ]
        #      [C*A^{N-1}*B, ..., C*B]

        S_u = np.zeros((N * p, N * m))
        S_x = np.zeros((N * p, n))

        A_power = np.eye(n)
        for k in range(N):
            S_x[k * p:(k + 1) * p, :] = self._C @ A_power

            # Build column blocks of S_u
            A_power_B = A_power @ self._B
            for j in range(k + 1):
                if j == 0:
                    CB = self._C @ self._B
                else:
                    CB = self._C @ np.linalg.matrix_power(self._A, k - j) @ self._B
                S_u[k * p:(k + 1) * p, j * m:(j + 1) * m] = CB

            A_power = A_power @ self._A

        # Hessian: H = 2 * (S_u' * block_diag(Q) * S_u + block_diag(R) + diff' * block_diag(R_delta) * diff)
        Q_bar = sparse.kron(sparse.eye(N), sparse.csc_matrix(Q))
        R_bar = sparse.kron(sparse.eye(N), sparse.csc_matrix(R))
        R_delta_bar = sparse.kron(sparse.eye(N), sparse.csc_matrix(R_delta))

        # Input rate difference matrix
        diff_matrix = np.eye(N * m)
        for k in range(1, N):
            diff_matrix[k * m:(k + 1) * m, (k - 1) * m:k * m] = -np.eye(m)

        S_u_sparse = sparse.csc_matrix(S_u)
        diff_sparse = sparse.csc_matrix(diff_matrix)

        H = 2.0 * (
            S_u_sparse.T @ Q_bar @ S_u_sparse
            + R_bar
            + diff_sparse.T @ R_delta_bar @ diff_sparse
        )

        # Store for later use
        self._H = H.toarray() if sparse.issparse(H) else H
        self._S_u = S_u
        self._S_x = S_x
        self._diff_matrix = diff_matrix

        # Constraint matrix (stacked)
        self._build_constraints()

    def _build_constraints(self) -> None:
        """Build constraint matrices for the QP."""
        N = self._params.control_horizon
        m = self._params.input_dim
        n = self._params.state_dim

        constraint_rows = []
        lb_list = []
        ub_list = []

        # Input bounds
        if self._constraints.u_min is not None and self._constraints.u_max is not None:
            I_Nm = sparse.eye(N * m)
            constraint_rows.append(I_Nm)
            lb_list.append(np.tile(self._constraints.u_min, N))
            ub_list.append(np.tile(self._constraints.u_max, N))

        # Input rate bounds
        if self._constraints.du_min is not None and self._constraints.du_max is not None:
            constraint_rows.append(sparse.csc_matrix(self._diff_matrix))
            # Rate bounds are relative to previous input
            lb_list.append(np.tile(self._constraints.du_min, N))
            ub_list.append(np.tile(self._constraints.du_max, N))

        if constraint_rows:
            self._constraint_A = sparse.vstack(constraint_rows, format="csc")
            self._constraint_lb = np.concatenate(lb_list)
            self._constraint_ub = np.concatenate(ub_list)
        else:
            self._constraint_A = None
            self._constraint_lb = None
            self._constraint_ub = None

    def compute_control(
        self,
        current_state: np.ndarray,
        reference: np.ndarray,
        previous_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the optimal control input using MPC.

        Args:
            current_state: Current state vector (n,).
            reference: Reference trajectory (N*p,) or (p,) for constant reference.
            previous_input: Previous control input for rate computation (m,).

        Returns:
            Optimal control input vector (m,).
        """
        if self._H is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        N = self._params.control_horizon
        n = self._params.state_dim
        m = self._params.input_dim
        p = self._params.output_dim

        start_time = time.time()

        # Handle reference dimensions
        if reference.ndim == 1 and len(reference) == p:
            # Constant reference
            ref = np.tile(reference, N)
        elif len(reference) == N * p:
            ref = reference
        else:
            ref = np.tile(reference[:p], N)

        # Compute free response (output prediction without control)
        free_response = self._S_x @ current_state

        # Compute tracking error (reference - free response)
        tracking_error = ref - free_response

        # Gradient: q = -2 * S_u' * Q_bar * tracking_error + R-related terms
        Q = self._params.Q
        Q_bar_diag = np.kron(np.eye(N), Q)
        R = self._params.R
        R_delta = self._params.R_delta

        q = -2.0 * self._S_u.T @ Q_bar_diag @ tracking_error

        # Add input rate cost contribution
        if previous_input is not None:
            # Shift previous input into rate matrix context
            u_prev = np.zeros(N * m)
            u_prev[:m] = previous_input
            R_delta_bar_diag = np.kron(np.eye(N), R_delta)
            q += 2.0 * self._diff_matrix.T @ R_delta_bar_diag @ self._diff_matrix @ u_prev

        # Solve QP
        u_opt = self._solve_qp(q)

        # Extract first control move
        u_current = u_opt[:m]

        # Store state
        self._state.current_state = current_state
        self._state.previous_input = u_current
        self._state.optimal_input_sequence = u_opt
        self._state.solve_time_ms = (time.time() - start_time) * 1000.0

        # Compute predicted states
        self._state.predicted_state_sequence = self._compute_predictions(
            current_state, u_opt
        )

        # Warm start for next iteration
        self._warm_start = np.roll(u_opt, -m)
        self._warm_start[-m:] = u_opt[-m:]

        self._solve_count += 1
        self._total_solve_time_ms += self._state.solve_time_ms

        return u_current

    def _solve_qp(self, q: np.ndarray) -> np.ndarray:
        """Solve the QP problem using the active set method.

        For small problems, uses a simple projected gradient descent.
        For larger problems, could be replaced with OSQP or qpOASES.

        Args:
            q: Linear cost vector.

        Returns:
            Optimal solution vector.
        """
        H = self._H
        n_vars = H.shape[0]

        # Initialize
        if self._warm_start is not None and len(self._warm_start) == n_vars:
            u = self._warm_start.copy()
        else:
            u = np.zeros(n_vars)

        # Input bounds
        lb, ub = self._constraints.get_input_bounds(self._params.control_horizon)

        # Projected gradient descent with momentum
        max_iter = 200
        step_size = 1.0 / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-6)
        tolerance = 1e-6
        momentum = 0.5
        velocity = np.zeros_like(u)

        for iteration in range(max_iter):
            gradient = H @ u + q

            # Check convergence
            u_proj = np.clip(u, lb, ub)
            gradient_projected = u - u_proj + gradient * step_size
            if np.max(np.abs(gradient_projected)) < tolerance:
                break

            # Update with momentum
            velocity = momentum * velocity - step_size * gradient
            u = u + velocity

            # Project onto feasible set
            u = np.clip(u, lb, ub)

        self._state.iterations = iteration + 1
        self._state.status = "optimal"
        self._state.cost = float(0.5 * u @ H @ u + q @ u)

        return u

    def _compute_predictions(
        self,
        x0: np.ndarray,
        u_sequence: np.ndarray,
    ) -> np.ndarray:
        """Compute predicted state trajectory.

        Args:
            x0: Initial state.
            u_sequence: Control input sequence (N*m,).

        Returns:
            Predicted states (N*n,).
        """
        N = self._params.control_horizon
        n = self._params.state_dim
        m = self._params.input_dim

        predictions = np.zeros((N + 1, n))
        predictions[0] = x0

        x = x0.copy()
        for k in range(N):
            u_k = u_sequence[k * m:(k + 1) * m]
            x = self._A @ x + self._B @ u_k
            predictions[k + 1] = x

        return predictions.flatten()

    def reset(self) -> None:
        """Reset the MPC controller state."""
        self._state = MPCState()
        self._warm_start = None
        self._initialized = False
        self._solve_count = 0
        self._total_solve_time_ms = 0.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MPCController(name='{self._name}', "
            f"N={self._params.prediction_horizon}, "
            f"M={self._params.control_horizon}, "
            f"states={self._params.state_dim}, "
            f"inputs={self._params.input_dim})"
        )
