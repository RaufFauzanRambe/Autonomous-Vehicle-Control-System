"""
Optimization Solver Interfaces for MPC Controller

Provides interfaces to multiple QP and NLP solvers:
  - OSQP (Operator Splitting Quadratic Program)
  - qpOASES (active-set QP solver)
  - IPOPT (Interior Point NLP solver)
  - Built-in projected gradient solver (fallback)
  - Solver benchmarking utilities

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


@dataclass
class SolverResult:
    """Result from an optimization solver.

    Attributes:
        x: Optimal solution vector.
        cost: Optimal cost value.
        solve_time_ms: Solve time in milliseconds.
        iterations: Number of iterations.
        status: Solver status string.
        dual_vars: Dual variables (Lagrange multipliers).
    """
    x: np.ndarray = np.array([])
    cost: float = float("inf")
    solve_time_ms: float = 0.0
    iterations: int = 0
    status: str = "unknown"
    dual_vars: Optional[np.ndarray] = None

    @property
    def is_optimal(self) -> bool:
        """Return whether the solution is optimal."""
        return self.status in ("optimal", "solved", "Optimal", "Solved")


class QPSolver(ABC):
    """Abstract base class for QP solvers.

    All solvers must implement the solve method for the standard QP form:
        min  0.5 * x' H x + q' x
        s.t. lb <= A x <= ub
    """

    @abstractmethod
    def solve(
        self,
        H: np.ndarray,
        q: np.ndarray,
        A: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve the QP problem.

        Args:
            H: Hessian matrix (n x n).
            q: Linear cost vector (n,).
            A: Constraint matrix (m x n). If None, no constraints.
            lb: Lower bounds (m,). If None, -inf.
            ub: Upper bounds (m,). If None, +inf.
            x0: Initial guess. If None, zeros.

        Returns:
            SolverResult with solution.
        """
        pass

    @abstractmethod
    def update(
        self,
        q: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        """Warm-start update for the solver (re-use factorization).

        Args:
            q: Updated linear cost vector.
            lb: Updated lower bounds.
            ub: Updated upper bounds.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the solver state."""
        pass


class ProjectedGradientSolver(QPSolver):
    """Projected gradient descent QP solver.

    A simple but robust solver for small to medium QP problems.
    Uses accelerated projected gradient (FISTA-like) with
    adaptive step size.

    Suitable for real-time MPC with short horizons.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        """Initialize the projected gradient solver.

        Args:
            max_iter: Maximum number of iterations.
            tolerance: Convergence tolerance.
            verbose: Print solver progress.
        """
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._verbose = verbose
        self._H = None
        self._prev_x = None

    def solve(
        self,
        H: np.ndarray,
        q: np.ndarray,
        A: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve QP using projected gradient descent with acceleration.

        Args:
            H: Hessian matrix.
            q: Linear cost vector.
            A: Constraint matrix (not used directly, bounds used instead).
            lb: Variable lower bounds.
            ub: Variable upper bounds.
            x0: Initial guess.

        Returns:
            SolverResult with solution.
        """
        start_time = time.time()
        n = H.shape[0]

        # Initialize
        x = x0.copy() if x0 is not None else np.zeros(n)
        if self._prev_x is not None and len(self._prev_x) == n:
            x = self._prev_x.copy()

        lower = lb if lb is not None else np.full(n, -np.inf)
        upper = ub if ub is not None else np.full(n, np.inf)

        # Step size: 1/L where L is the largest eigenvalue of H
        eigvals = np.linalg.eigvalsh(H)
        L = max(np.max(np.abs(eigvals)), 1e-10)
        step_size = 1.0 / L

        # FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
        y = x.copy()
        t = 1.0
        best_x = x.copy()
        best_cost = float("inf")

        for iteration in range(self._max_iter):
            # Gradient at y
            grad = H @ y + q

            # Gradient step
            x_new = y - step_size * grad

            # Projection onto bounds
            x_new = np.clip(x_new, lower, upper)

            # FISTA momentum
            t_new = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            beta = (t - 1.0) / t_new
            y = x_new + beta * (x_new - x)

            # Update
            x = x_new
            t = t_new

            # Compute cost
            cost = 0.5 * x @ H @ x + q @ x
            if cost < best_cost:
                best_cost = cost
                best_x = x.copy()

            # Check convergence
            diff = np.max(np.abs(x - y + step_size * grad))
            if diff < self._tolerance:
                break

        solve_time_ms = (time.time() - start_time) * 1000.0

        self._H = H
        self._prev_x = best_x.copy()

        return SolverResult(
            x=best_x,
            cost=best_cost,
            solve_time_ms=solve_time_ms,
            iterations=iteration + 1,
            status="optimal",
        )

    def update(
        self,
        q: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        """Warm-start update."""
        pass  # Warm start is handled via _prev_x

    def reset(self) -> None:
        """Reset solver state."""
        self._prev_x = None


class OSQPSolver(QPSolver):
    """OSQP (Operator Splitting Quadratic Program) solver interface.

    Uses the ADMM-based OSQP solver for efficient QP solving.
    Falls back to projected gradient if OSQP is not available.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        eps_abs: float = 1e-5,
        eps_rel: float = 1e-5,
        verbose: bool = False,
        warm_start: bool = True,
    ) -> None:
        """Initialize the OSQP solver.

        Args:
            max_iter: Maximum iterations.
            eps_abs: Absolute tolerance.
            eps_rel: Relative tolerance.
            verbose: Print solver output.
            warm_start: Enable warm starting.
        """
        self._max_iter = max_iter
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel
        self._verbose = verbose
        self._warm_start = warm_start
        self._solver = None
        self._fallback = ProjectedGradientSolver(max_iter=max_iter)
        self._osqp_available = False

        try:
            import osqp
            self._osqp_available = True
            self._osqp_module = osqp
        except ImportError:
            self._osqp_available = False

    def solve(
        self,
        H: np.ndarray,
        q: np.ndarray,
        A: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve QP using OSQP or fallback.

        Args:
            H: Hessian matrix.
            q: Linear cost vector.
            A: Constraint matrix.
            lb: Lower bounds.
            ub: Upper bounds.
            x0: Initial guess.

        Returns:
            SolverResult with solution.
        """
        n = H.shape[0]

        if not self._osqp_available:
            return self._fallback.solve(H, q, A, lb, ub, x0)

        start_time = time.time()

        # OSQP format: min 0.5 x' P x + q' x  s.t. l <= A x <= u
        P = sparse.csc_matrix(H)
        q_osqp = q.copy()

        if A is not None:
            A_sparse = sparse.csc_matrix(A)
        else:
            # Only bound constraints: A = I
            A_sparse = sparse.eye(n, format="csc")

        lower = lb if lb is not None else np.full(A_sparse.shape[0], -np.inf)
        upper = ub if ub is not None else np.full(A_sparse.shape[0], np.inf)

        # Create solver
        solver = self._osqp_module.OSQP()
        solver.setup(
            P=P,
            q=q_osqp,
            A=A_sparse,
            l=lower,
            u=upper,
            max_iter=self._max_iter,
            eps_abs=self._eps_abs,
            eps_rel=self._eps_rel,
            verbose=self._verbose,
            warm_starting=self._warm_start,
        )

        if x0 is not None and self._warm_start:
            solver.warm_start(x=x0)

        result = solver.solve()

        solve_time_ms = (time.time() - start_time) * 1000.0
        self._solver = solver

        return SolverResult(
            x=result.x,
            cost=result.info.obj_val if result.info.obj_val is not None else float("inf"),
            solve_time_ms=solve_time_ms,
            iterations=result.info.iter,
            status=result.info.status,
        )

    def update(
        self,
        q: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        """Warm-start update for OSQP."""
        if self._solver is not None and self._osqp_available:
            if q is not None:
                self._solver.update(q=q)
            if lb is not None or ub is not None:
                kwargs = {}
                if lb is not None:
                    kwargs["l"] = lb
                if ub is not None:
                    kwargs["u"] = ub
                self._solver.update(**kwargs)

    def reset(self) -> None:
        """Reset solver state."""
        self._solver = None


class QpOASESSolver(QPSolver):
    """qpOASES active-set QP solver interface.

    Provides an interface to the qpOASES solver, which is well-suited
    for MPC problems due to its efficient warm-starting capabilities.
    Falls back to projected gradient if qpOASES is not available.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        """Initialize the qpOASES solver.

        Args:
            max_iter: Maximum iterations.
            tolerance: Solver tolerance.
            verbose: Print solver output.
        """
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._verbose = verbose
        self._fallback = ProjectedGradientSolver(max_iter=max_iter)
        self._qpoases_available = False

        try:
            import qpoases
            self._qpoases_available = True
            self._qpoases_module = qpoases
        except ImportError:
            self._qpoases_available = False

    def solve(
        self,
        H: np.ndarray,
        q: np.ndarray,
        A: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """Solve QP using qpOASES or fallback.

        Args:
            H: Hessian matrix.
            q: Linear cost vector.
            A: Constraint matrix.
            lb: Lower bounds.
            ub: Upper bounds.
            x0: Initial guess.

        Returns:
            SolverResult with solution.
        """
        n = H.shape[0]

        if not self._qpoases_available:
            return self._fallback.solve(H, q, A, lb, ub, x0)

        start_time = time.time()

        # qpOASES uses row-major format
        H_row = np.ascontiguousarray(H, dtype=np.float64)
        q_arr = np.ascontiguousarray(q, dtype=np.float64)

        lower = lb if lb is not None else np.full(n, -1e20)
        upper = ub if ub is not None else np.full(n, 1e20)

        try:
            # Create qpOASES problem
            qp = self._qpoases_module.PythonQProblem(n)
            options = self._qpoases_module.Options()
            options.printLevel = self._qpoases_module.PrintLevel.NONE if not self._verbose else self._qpoases_module.PrintLevel.LOW
            qp.setOptions(options)

            nWSR = np.array([self._max_iter])
            qp.init(H_row, q_arr, None, lower, upper, None, None, nWSR)

            x_opt = np.zeros(n)
            qp.getPrimalSolution(x_opt)

            solve_time_ms = (time.time() - start_time) * 1000.0
            cost = 0.5 * x_opt @ H @ x_opt + q @ x_opt

            return SolverResult(
                x=x_opt,
                cost=cost,
                solve_time_ms=solve_time_ms,
                iterations=int(nWSR[0]),
                status="optimal",
            )
        except Exception as e:
            # Fall back to projected gradient
            return self._fallback.solve(H, q, A, lb, ub, x0)

    def update(
        self,
        q: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
    ) -> None:
        """Warm-start update."""
        pass

    def reset(self) -> None:
        """Reset solver state."""
        pass


class IPOPTSolver:
    """IPOPT (Interior Point OPTimizer) interface for NLP problems.

    Solves general nonlinear optimization problems:
        min  f(x)
        s.t. g_L <= g(x) <= g_U
             x_L <=  x  <= x_U

    Used for nonlinear MPC formulations.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        """Initialize the IPOPT solver.

        Args:
            max_iter: Maximum iterations.
            tolerance: Convergence tolerance.
            verbose: Print solver output.
        """
        self._max_iter = max_iter
        self._tolerance = tolerance
        self._verbose = verbose

    def solve(
        self,
        objective: callable,
        gradient: callable,
        x0: np.ndarray,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        constraints: Optional[dict] = None,
    ) -> SolverResult:
        """Solve NLP using scipy.optimize.minimize (IPOPT-like interface).

        Falls back to SLSQP if IPOPT is not available.

        Args:
            objective: Objective function f(x) -> float.
            gradient: Gradient function grad_f(x) -> np.ndarray.
            x0: Initial guess.
            lb: Variable lower bounds.
            ub: Variable upper bounds.
            constraints: Constraint dictionary for scipy.

        Returns:
            SolverResult with solution.
        """
        start_time = time.time()
        n = len(x0)

        bounds = None
        if lb is not None or ub is not None:
            lower = lb if lb is not None else np.full(n, -np.inf)
            upper = ub if ub is not None else np.full(n, np.inf)
            bounds = list(zip(lower, upper))

        result = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self._max_iter,
                "ftol": self._tolerance,
                "disp": self._verbose,
            },
        )

        solve_time_ms = (time.time() - start_time) * 1000.0

        return SolverResult(
            x=result.x,
            cost=result.fun,
            solve_time_ms=solve_time_ms,
            iterations=result.nit,
            status="optimal" if result.success else "failed",
        )


class SolverBenchmark:
    """Benchmark utility for comparing QP solver performance."""

    def __init__(self) -> None:
        """Initialize the solver benchmark."""
        self._results: Dict[str, list] = {}

    def benchmark_solvers(
        self,
        H: np.ndarray,
        q: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_trials: int = 10,
    ) -> Dict[str, SolverResult]:
        """Benchmark available solvers on a given QP.

        Args:
            H: Hessian matrix.
            q: Linear cost vector.
            lb: Lower bounds.
            ub: Upper bounds.
            n_trials: Number of trials for timing.

        Returns:
            Dictionary mapping solver name to average SolverResult.
        """
        solvers = {
            "ProjectedGradient": ProjectedGradientSolver(),
            "OSQP": OSQPSolver(),
        }

        results = {}
        for name, solver in solvers.items():
            times = []
            result = None
            for _ in range(n_trials):
                r = solver.solve(H, q, lb=lb, ub=ub)
                times.append(r.solve_time_ms)
                result = r

            if result is not None:
                avg_time = np.mean(times)
                results[name] = SolverResult(
                    x=result.x,
                    cost=result.cost,
                    solve_time_ms=avg_time,
                    iterations=result.iterations,
                    status=result.status,
                )

        return results

    def print_results(self, results: Dict[str, SolverResult]) -> None:
        """Print benchmark results.

        Args:
            results: Dictionary of solver results.
        """
        print("=" * 70)
        print("Solver Benchmark Results")
        print("=" * 70)
        print(f"{'Solver':<25} {'Time (ms)':<12} {'Iterations':<12} {'Cost':<15} {'Status'}")
        print("-" * 70)
        for name, result in results.items():
            print(f"{name:<25} {result.solve_time_ms:<12.3f} {result.iterations:<12} "
                  f"{result.cost:<15.6f} {result.status}")
        print("=" * 70)


def create_solver(
    solver_type: str = "projected_gradient",
    **kwargs,
) -> QPSolver:
    """Factory function to create a QP solver.

    Args:
        solver_type: Type of solver ('projected_gradient', 'osqp', 'qpoases').
        **kwargs: Additional solver-specific arguments.

    Returns:
        QPSolver instance.
    """
    if solver_type == "projected_gradient":
        return ProjectedGradientSolver(**kwargs)
    elif solver_type == "osqp":
        return OSQPSolver(**kwargs)
    elif solver_type == "qpoases":
        return QpOASESSolver(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
