"""Tests for the localization filters (module 04).

Covered behavior: GPS low-pass convergence onto a constant measurement,
and the complementary fusion of GPS + IMU dead reckoning achieving a
lower RMSE than the raw GPS track.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless rendering for plotting tests

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_localization_demo"))

from localization.gps_processor import GPSProcessor
from localization.imu_processor import IMUProcessor
from localization.sensor_fusion import SensorFusion


def _rmse(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error over all elements of two aligned arrays.

    Args:
        estimated: Estimated values, shape ``(N, 2)`` (x, y per step).
        truth: Ground-truth values with the same shape.

    Returns:
        The RMSE [m].
    """
    estimated_arr = np.asarray(estimated, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((estimated_arr - truth_arr) ** 2)))


def test_gps_low_pass_converges() -> None:
    """A constant measurement must pull the low-pass estimate onto itself.

    The processor seeds its estimate with the first measurement, so after
    20 calls of ``process((100.0, 50.0))`` with ``low_pass_alpha = 0.6``
    the residual is bounded by ``100 * (1 - alpha)^19 = 2.75e-6 m`` on the
    x axis (y starts at 50 m and converges twice as fast). The estimate is
    therefore asserted to within a 5e-6 m tolerance, i.e. far tighter than
    any GPS accuracy of interest while respecting that geometric bound.
    """
    processor = GPSProcessor({"low_pass_alpha": 0.6})
    estimate: tuple[float, float] = (0.0, 0.0)
    for _ in range(20):
        estimate = processor.process((100.0, 50.0))
    assert estimate[0] == pytest.approx(100.0, abs=5e-6)
    assert estimate[1] == pytest.approx(50.0, abs=5e-6)


def test_fusion_beats_raw_gps() -> None:
    """Complementary GPS + IMU fusion must beat the raw GPS track in RMSE.

    Scenario: straight drive at 5 m/s for 200 steps of 0.1 s. The IMU is
    bias-free with small acceleration noise (0.02 m/s^2 sigma), the GPS
    is noisy (1.5 m sigma). The fused estimate initialized at the first
    GPS fix must track the ground truth more closely than the GPS alone.
    """
    rng = np.random.default_rng(42)
    dt = 0.1
    n_steps = 200

    imu = IMUProcessor({"accel_bias_mpss": 0.0, "gyro_bias_radps": 0.0})
    imu.reset(initial_velocity=(5.0, 0.0))

    gps_x0 = float(rng.normal(0.0, 1.5))
    fusion = SensorFusion({"complementary_alpha": 0.95})
    fusion.reset(initial_position=(gps_x0, 0.0))

    truth_positions: list[tuple[float, float]] = []
    gps_positions: list[tuple[float, float]] = []
    fused_positions: list[tuple[float, float]] = []

    for step in range(1, n_steps + 1):
        truth_x = 5.0 * step * dt
        ax_mpss = float(rng.normal(0.0, 0.02))
        gps_x = truth_x + float(rng.normal(0.0, 1.5))
        gps_y = 0.0
        imu_dxy = imu.update(ax_mpss, 0.0, 0.0, dt)
        fused = fusion.update((gps_x, gps_y), imu_dxy, dt)
        truth_positions.append((truth_x, 0.0))
        gps_positions.append((gps_x, gps_y))
        fused_positions.append((float(fused[0]), float(fused[1])))

    truth_arr = np.asarray(truth_positions)
    gps_arr = np.asarray(gps_positions)
    fused_arr = np.asarray(fused_positions)

    gps_rmse = _rmse(gps_arr, truth_arr)
    fused_rmse = _rmse(fused_arr, truth_arr)
    assert fused_rmse < gps_rmse
