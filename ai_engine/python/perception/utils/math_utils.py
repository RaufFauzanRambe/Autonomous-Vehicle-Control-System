"""
Math utilities for the Autonomous Vehicle Control System.

Provides interpolation, polynomial fitting, eigenvalue decomposition, SVD,
and statistical functions used throughout perception, planning, and control.

All functions operate on plain NumPy arrays for maximum interoperability.

Usage:
    from utils.math_utils import cubic_spline_interpolate, fit_polynomial, compute_pca

    waypoints = cubic_spline_interpolate(control_points, num_points=200)
    coeffs = fit_polynomial(x_data, y_data, degree=3)
    components = compute_pca(data_matrix, n_components=3)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.linalg import eig, eigh, svd, inv, det, norm


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def linear_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Piecewise linear interpolation.

    Args:
        x: (N,) independent variable – must be monotonically increasing.
        y: (N,) dependent variable.
        x_new: (M,) query points.

    Returns:
        (M,) interpolated values. Values outside the range of *x* are
        extrapolated from the nearest segment.
    """
    return np.interp(x_new, x, y)


def cubic_spline_interpolate(
    control_points: np.ndarray,
    num_points: int = 200,
    periodic: bool = False,
    bc_type: str = "clamped",
) -> np.ndarray:
    """Cubic spline interpolation through 2D control points.

    Uses natural boundary conditions by default (second derivative = 0 at
    endpoints). Computes a cumulative-distance parameterisation so that
    output points are approximately evenly spaced in arc-length.

    Args:
        control_points: (N, 2) array of [x, y] waypoints.
        num_points: Number of interpolated output points.
        periodic: If True, close the spline by repeating the first point.
        bc_type: Boundary condition – ``"clamped"`` or ``"natural"``.

    Returns:
        (num_points, 2) array of smoothly interpolated [x, y] points.
    """
    from scipy.interpolate import CubicSpline

    pts = np.asarray(control_points, dtype=float)
    if periodic:
        pts = np.vstack([pts, pts[0]])

    # Cumulative arc-length parameter
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.zeros(len(pts))
    t[1:] = np.cumsum(seg_lengths)

    if t[-1] == 0:
        return np.tile(pts[0], (num_points, 1))

    t_uniform = np.linspace(0, t[-1], num_points)
    bc = bc_type if not periodic else "periodic"

    cs_x = CubicSpline(t, pts[:, 0], bc_type=bc)
    cs_y = CubicSpline(t, pts[:, 1], bc_type=bc)

    return np.column_stack([cs_x(t_uniform), cs_y(t_uniform)])


def bilinear_interpolate(
    grid: np.ndarray,
    x: float,
    y: float,
) -> float:
    """Bilinear interpolation on a 2D grid.

    Args:
        grid: (H, W) 2D array with values at integer grid positions.
        x: Row coordinate (can be fractional).
        y: Column coordinate (can be fractional).

    Returns:
        Interpolated value.
    """
    h, w = grid.shape
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, h - 1)
    y1 = min(y0 + 1, w - 1)
    x0 = max(0, x0)
    y0 = max(0, y0)

    dx = x - math.floor(x)
    dy = y - math.floor(y)

    val = (
        grid[x0, y0] * (1 - dx) * (1 - dy)
        + grid[x1, y0] * dx * (1 - dy)
        + grid[x0, y1] * (1 - dx) * dy
        + grid[x1, y1] * dx * dy
    )
    return float(val)


# ---------------------------------------------------------------------------
# Polynomial fitting
# ---------------------------------------------------------------------------

def fit_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit a polynomial to (x, y) data using least-squares.

    Args:
        x: Independent variable.
        y: Dependent variable.
        degree: Polynomial degree.
        weights: Optional per-point weights for weighted least-squares.

    Returns:
        Coefficients array of shape (degree + 1,), highest power first
        (compatible with :func:`numpy.polyval`).
    """
    return np.polyfit(x, y, degree, w=weights)


def evaluate_polynomial(
    coeffs: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate a polynomial given coefficients (highest power first)."""
    return np.polyval(coeffs, x)


def polynomial_derivative(
    coeffs: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Compute coefficients of the *order*-th derivative of a polynomial."""
    return np.polyder(coeffs, order)


def polynomial_roots(
    coeffs: np.ndarray,
    real_only: bool = True,
) -> np.ndarray:
    """Find roots of a polynomial, optionally filtering for real roots only.

    Args:
        coeffs: Polynomial coefficients (highest power first).
        real_only: If True, return only roots with negligible imaginary part.

    Returns:
        Array of roots.
    """
    roots = np.roots(coeffs)
    if real_only:
        roots = roots[np.abs(roots.imag) < 1e-8].real
    return roots


def fit_cubic_bezier(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a cubic Bézier curve through four or more points using
    least-squares.

    Returns the four control points P0, P1, P2, P3.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    # Chord-length parameterisation
    diffs = np.diff(pts, axis=0)
    chord_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.zeros(n)
    t[1:] = np.cumsum(chord_lengths)
    if t[-1] > 0:
        t /= t[-1]

    # Basis matrix for cubic Bézier: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    t_col = t[:, None]
    basis = np.column_stack([
        (1 - t) ** 3,
        3 * (1 - t) ** 2 * t,
        3 * (1 - t) * t ** 2,
        t ** 3,
    ])

    # P0 = first point, P3 = last point – solve for P1, P2
    P0 = pts[0]
    P3 = pts[-1]
    rhs = pts - basis[:, 0:1] * P0 - basis[:, 3:4] * P3
    inner = basis[:, 1:3]

    # Least-squares solve
    result, _, _, _ = np.linalg.lstsq(inner, rhs, rcond=None)
    P1, P2 = result[0], result[1]

    return P0, P1, P2, P3


# ---------------------------------------------------------------------------
# Linear algebra – eigenvalue, SVD, PCA
# ---------------------------------------------------------------------------

def eigen_decomposition(
    matrix: np.ndarray,
    symmetric: bool = True,
    sort_descending: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors.

    Args:
        matrix: Square matrix (N, N).
        symmetric: Use faster ``eigh`` for Hermitian/symmetric matrices.
        sort_descending: Sort eigenvalues from largest to smallest.

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors are columns.
    """
    if symmetric:
        values, vectors = eigh(matrix)
    else:
        values, vectors = eig(matrix)
        values = values.real
        vectors = vectors.real

    if sort_descending:
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]

    return values, vectors


def singular_value_decomposition(
    matrix: np.ndarray,
    full_matrices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD decomposition.

    Returns:
        (U, S, Vt) such that ``matrix ≈ U @ diag(S) @ Vt``.
    """
    return svd(matrix, full_matrices=full_matrices)


def compute_pca(
    data: np.ndarray,
    n_components: Optional[int] = None,
    explained_variance_ratio: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Principal Component Analysis.

    Args:
        data: (N, D) data matrix (N samples, D features).
        n_components: Number of components to keep. If None, keep all.
        explained_variance_ratio: If set (0 < r ≤ 1), keep enough components
            to explain at least this fraction of total variance. Overrides
            *n_components* if both are given.

    Returns:
        Dict with keys:
            - ``components``: (K, D) principal component vectors.
            - ``explained_variance``: (K,) eigenvalues.
            - ``explained_variance_ratio``: (K,) proportion of total variance.
            - ``mean``: (D,) feature mean.
            - ``transformed``: (N, K) projected data.
    """
    data = np.asarray(data, dtype=float)
    mean = data.mean(axis=0)
    centered = data - mean

    # Covariance matrix
    cov = (centered.T @ centered) / (len(centered) - 1)
    eigenvalues, eigenvectors = eigen_decomposition(cov, symmetric=True, sort_descending=True)

    total_var = eigenvalues.sum()
    ratios = eigenvalues / total_var if total_var > 0 else eigenvalues

    k = len(eigenvalues)
    if explained_variance_ratio is not None:
        cumulative = np.cumsum(ratios)
        k = int(np.searchsorted(cumulative, explained_variance_ratio) + 1)
        k = min(k, len(eigenvalues))
    elif n_components is not None:
        k = min(n_components, len(eigenvalues))

    return {
        "components": eigenvectors[:, :k].T,
        "explained_variance": eigenvalues[:k],
        "explained_variance_ratio": ratios[:k],
        "mean": mean,
        "transformed": centered @ eigenvectors[:, :k],
    }


def pseudo_inverse(matrix: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Compute the Moore-Penrose pseudo-inverse using SVD."""
    U, S, Vt = svd(matrix, full_matrices=False)
    S_inv = np.where(S > rcond * S.max(), 1.0 / S, 0.0)
    return Vt.T @ np.diag(S_inv) @ U.T


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def descriptive_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute a standard set of descriptive statistics.

    Args:
        data: 1-D numeric array.

    Returns:
        Dict with count, mean, std, min, max, median, and percentiles.
    """
    arr = np.asarray(data, dtype=float).ravel()
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "var": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.ptp(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "skewness": float(_skewness(arr)),
        "kurtosis": float(_kurtosis(arr)),
    }


def _skewness(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(np.sum(((arr - m) / s) ** 3) * n / ((n - 1) * (n - 2)))


def _kurtosis(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    k4 = np.sum(((arr - m) / s) ** 4) / n
    return float(k4 - 3.0)  # Excess kurtosis


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Simple moving average with uniform weights.

    Pads the edges so the output has the same length as the input.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    kernel = np.ones(window_size) / window_size
    # Pad to maintain length
    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(data)]


def exponential_moving_average(
    data: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """Exponential moving average.

    Args:
        data: 1-D input array.
        alpha: Smoothing factor (0 < alpha ≤ 1). Higher values give less smoothing.

    Returns:
        Smoothed array of same length.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    result = np.empty_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def weighted_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    W: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve the weighted least-squares problem:  min ||W(Ax - b)||^2.

    Args:
        A: (M, N) design matrix.
        b: (M,) observations.
        W: (M,) diagonal weights. If None, unweighted.

    Returns:
        (N,) solution vector x.
    """
    if W is not None:
        sqrt_W = np.sqrt(W)
        A_w = A * sqrt_W[:, None]
        b_w = b * sqrt_W
    else:
        A_w = A
        b_w = b
    x, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
    return x


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Compute the Mahalanobis distance between *x* and a distribution
    described by *mean* and *covariance*.

    Args:
        x: (D,) observation.
        mean: (D,) distribution mean.
        covariance: (D, D) covariance matrix.

    Returns:
        Scalar distance.
    """
    diff = x - mean
    try:
        cov_inv = inv(covariance)
    except np.linalg.LinAlgError:
        cov_inv = pseudo_inverse(covariance)
    return float(np.sqrt(diff @ cov_inv @ diff))


def normalize(
    data: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize data along an axis.

    Args:
        data: Input array.
        method: ``"zscore"`` (zero-mean, unit-variance) or ``"minmax"`` (0–1).
        axis: Axis along which to compute statistics.
        eps: Small constant to avoid division by zero.

    Returns:
        (normalized_data, params) where *params* can be used to reverse the
        transformation later.
    """
    if method == "zscore":
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True) + eps
        result = (data - mean) / std
        params = {"method": "zscore", "mean": mean, "std": std}
    elif method == "minmax":
        vmin = data.min(axis=axis, keepdims=True)
        vmax = data.max(axis=axis, keepdims=True)
        rng = vmax - vmin + eps
        result = (data - vmin) / rng
        params = {"method": "minmax", "min": vmin, "max": vmax}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return result, params


def denormalize(
    data: np.ndarray,
    params: Dict[str, np.ndarray],
) -> np.ndarray:
    """Reverse normalization using parameters from :func:`normalize`."""
    method = params["method"]
    if method == "zscore":
        return data * params["std"] + params["mean"]
    elif method == "minmax":
        rng = params["max"] - params["min"]
        return data * rng + params["min"]
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def angle_diff(target: float, current: float) -> float:
    """Shortest angular difference from *current* to *target* in [-π, π]."""
    return normalize_angle(target - current)


def lerp_angle(a: float, b: float, t: float) -> float:
    """Linear interpolation between angles, taking the shortest path."""
    diff = angle_diff(b, a)
    return normalize_angle(a + diff * t)


def rad2deg(angle: float) -> float:
    return math.degrees(angle)


def deg2rad(angle: float) -> float:
    return math.radians(angle)
