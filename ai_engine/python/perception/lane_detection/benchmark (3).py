"""
MPC Controller Benchmark Suite for Autonomous Vehicle Control

Benchmarks MPC controller performance with metrics:
  - Computation time (solve time, total time)
  - Tracking accuracy (RMS lateral error, heading error, speed error)
  - Control effort (steering effort, acceleration effort)
  - Constraint satisfaction rate
  - Comparison with PID baseline
  - Scalability with horizon length

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mpc_controller import MPCController, MPCParams, MPCConstraints
from .vehicle_model import KinematicBicycleModel, VehicleParameters
from .trajectory_tracker import TrajectoryTracker, TrackerConfig, ReferencePoint
from .state_predictor import StatePredictor, PredictionConfig
from .optimization_solver import SolverBenchmark


@dataclass
class MPCBenchmarkMetrics:
    """MPC benchmark metrics.

    Attributes:
        avg_solve_time_ms: Average QP solve time.
        max_solve_time_ms: Maximum QP solve time.
        rms_lateral_error: RMS lateral tracking error (m).
        max_lateral_error: Maximum lateral tracking error (m).
        rms_heading_error: RMS heading tracking error (rad).
        max_heading_error: Maximum heading tracking error (rad).
        rms_speed_error: RMS speed tracking error (m/s).
        total_control_effort: Total control effort.
        constraint_violations: Number of constraint violations.
        solver_failures: Number of solver failures.
        tracking_accuracy: Combined tracking accuracy score.
    """
    avg_solve_time_ms: float = 0.0
    max_solve_time_ms: float = 0.0
    rms_lateral_error: float = 0.0
    max_lateral_error: float = 0.0
    rms_heading_error: float = 0.0
    max_heading_error: float = 0.0
    rms_speed_error: float = 0.0
    total_control_effort: float = 0.0
    constraint_violations: int = 0
    solver_failures: int = 0
    tracking_accuracy: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "avg_solve_time_ms": f"{self.avg_solve_time_ms:.3f}",
            "max_solve_time_ms": f"{self.max_solve_time_ms:.3f}",
            "rms_lateral_error_m": f"{self.rms_lateral_error:.4f}",
            "max_lateral_error_m": f"{self.max_lateral_error:.4f}",
            "rms_heading_error_deg": f"{math.degrees(self.rms_heading_error):.4f}",
            "rms_speed_error_ms": f"{self.rms_speed_error:.4f}",
            "constraint_violations": self.constraint_violations,
            "solver_failures": self.solver_failures,
            "tracking_accuracy": f"{self.tracking_accuracy:.4f}",
        }


@dataclass
class BenchmarkScenario:
    """A benchmark test scenario.

    Attributes:
        name: Scenario name.
        reference: Reference trajectory points.
        initial_speed: Initial vehicle speed (m/s).
        duration: Test duration (s).
        description: Scenario description.
    """
    name: str = ""
    reference: List[ReferencePoint] = field(default_factory=list)
    initial_speed: float = 15.0
    duration: float = 30.0
    description: str = ""


class MPCBenchmarkSuite:
    """Benchmark suite for MPC controller performance evaluation."""

    def __init__(self, dt: float = 0.01) -> None:
        """Initialize the benchmark suite.

        Args:
            dt: Simulation timestep in seconds.
        """
        self._dt = dt
        self._vehicle_params = VehicleParameters()

    def _generate_straight_reference(self, length: float = 200.0) -> List[ReferencePoint]:
        """Generate a straight line reference trajectory.

        Args:
            length: Path length in meters.

        Returns:
            List of ReferencePoint objects.
        """
        points = []
        spacing = 0.5
        n_points = int(length / spacing)
        for i in range(n_points):
            points.append(ReferencePoint(
                x=i * spacing,
                y=0.0,
                heading=0.0,
                curvature=0.0,
                speed=15.0,
                acceleration=0.0,
            ))
        return points

    def _generate_curve_reference(
        self,
        radius: float = 100.0,
        arc_length: float = 150.0,
    ) -> List[ReferencePoint]:
        """Generate a constant curvature reference trajectory.

        Args:
            radius: Curve radius in meters.
            arc_length: Total arc length in meters.

        Returns:
            List of ReferencePoint objects.
        """
        points = []
        spacing = 0.5
        n_points = int(arc_length / spacing)
        curvature = 1.0 / radius
        heading = 0.0
        x, y = 0.0, 0.0
        speed = 12.0  # Slower for curve

        for i in range(n_points):
            points.append(ReferencePoint(
                x=x, y=y, heading=heading,
                curvature=curvature, speed=speed,
            ))
            x += spacing * math.cos(heading)
            y += spacing * math.sin(heading)
            heading += curvature * spacing

        return points

    def _generate_s_curve_reference(self, length: float = 300.0) -> List[ReferencePoint]:
        """Generate an S-curve reference trajectory.

        Args:
            length: Total path length in meters.

        Returns:
            List of ReferencePoint objects.
        """
        points = []
        spacing = 0.5
        n_points = int(length / spacing)
        heading = 0.0
        x, y = 0.0, 0.0

        for i in range(n_points):
            s = i * spacing
            # S-curve curvature
            if s < 50:
                curvature = 0.0
                speed = 20.0
            elif s < 100:
                curvature = 0.01  # Right curve
                speed = 15.0
            elif s < 150:
                curvature = 0.0  # Straight
                speed = 20.0
            elif s < 200:
                curvature = -0.01  # Left curve
                speed = 15.0
            else:
                curvature = 0.0
                speed = 20.0

            points.append(ReferencePoint(
                x=x, y=y, heading=heading,
                curvature=curvature, speed=speed,
            ))
            x += spacing * math.cos(heading)
            y += spacing * math.sin(heading)
            heading += curvature * spacing

        return points

    def _generate_lane_change_reference(self) -> List[ReferencePoint]:
        """Generate a lane change reference trajectory.

        Returns:
            List of ReferencePoint objects.
        """
        points = []
        spacing = 0.5
        lane_width = 3.5
        x = 0.0

        while x < 200.0:
            y = 0.0
            heading = 0.0
            curvature = 0.0
            speed = 15.0

            if 40.0 <= x <= 70.0:
                t = (x - 40.0) / 30.0
                y = lane_width * 0.5 * (1 - math.cos(math.pi * t))
                heading = math.atan2(
                    lane_width * 0.5 * math.pi / 30.0 * math.sin(math.pi * t), 1.0
                )
                curvature = lane_width * 0.5 * (math.pi / 30.0) ** 2 * math.cos(math.pi * t)
            elif x > 70.0:
                y = lane_width

            points.append(ReferencePoint(
                x=x, y=y, heading=heading,
                curvature=curvature, speed=speed,
            ))
            x += spacing

        return points

    def benchmark_tracking(
        self,
        config: TrackerConfig = TrackerConfig(),
        scenario_name: str = "s_curve",
        n_steps: Optional[int] = None,
    ) -> MPCBenchmarkMetrics:
        """Benchmark MPC tracking performance on a scenario.

        Args:
            config: Tracker configuration.
            scenario_name: Scenario name.
            n_steps: Number of simulation steps. If None, auto-computed.

        Returns:
            MPCBenchmarkMetrics with results.
        """
        # Generate scenario
        if scenario_name == "straight":
            reference = self._generate_straight_reference()
        elif scenario_name == "curve":
            reference = self._generate_curve_reference()
        elif scenario_name == "s_curve":
            reference = self._generate_s_curve_reference()
        elif scenario_name == "lane_change":
            reference = self._generate_lane_change_reference()
        else:
            reference = self._generate_s_curve_reference()

        if n_steps is None:
            n_steps = int(30.0 / self._dt)

        # Initialize tracker
        tracker = TrajectoryTracker(config=config, vehicle_params=self._vehicle_params)

        # Vehicle simulation
        vehicle_model = KinematicBicycleModel(self._vehicle_params)
        initial_speed = reference[0].speed if reference else 15.0
        state = np.array([0.0, 0.3, 0.0, initial_speed])  # Slight offset

        # Tracking data
        solve_times = []
        lateral_errors = []
        heading_errors = []
        speed_errors = []
        control_efforts = []
        violations = 0
        failures = 0

        for i in range(n_steps):
            vehicle_state_dict = {
                "x": state[0], "y": state[1],
                "heading": state[2], "speed": state[3],
            }

            steer, accel, diag = tracker.compute_control(
                vehicle_x=state[0],
                vehicle_y=state[1],
                vehicle_heading=state[2],
                vehicle_speed=state[3],
                reference=reference,
            )

            # Record metrics
            solve_times.append(diag.get("solve_time_ms", 0.0))
            lateral_errors.append(diag.get("lateral_error", 0.0))
            heading_errors.append(diag.get("heading_error_deg", 0.0))
            speed_errors.append(diag.get("speed_error", 0.0))

            # Control effort
            effort = abs(steer) + abs(accel) * 0.1
            control_efforts.append(effort)

            if "fallback" in diag.get("solve_status", ""):
                failures += 1

            # Check constraint violations
            if abs(steer) > self._vehicle_params.max_steer * 1.01:
                violations += 1

            # Simulate vehicle
            u = np.array([accel, steer])
            state = vehicle_model.step(state, u, self._dt)

            # Update closest index
            # (handled internally by tracker)

        # Compute summary metrics
        metrics = MPCBenchmarkMetrics(
            avg_solve_time_ms=float(np.mean(solve_times)) if solve_times else 0.0,
            max_solve_time_ms=float(np.max(solve_times)) if solve_times else 0.0,
            rms_lateral_error=float(np.sqrt(np.mean(np.array(lateral_errors) ** 2))),
            max_lateral_error=float(np.max(np.abs(lateral_errors))),
            rms_heading_error=float(np.sqrt(np.mean(np.radians(np.array(heading_errors)) ** 2))),
            max_heading_error=float(np.max(np.abs(np.radians(np.array(heading_errors))))),
            rms_speed_error=float(np.sqrt(np.mean(np.array(speed_errors) ** 2))),
            total_control_effort=float(np.sum(control_efforts)),
            constraint_violations=violations,
            solver_failures=failures,
        )

        # Compute combined tracking accuracy score (lower is better)
        metrics.tracking_accuracy = (
            0.4 * metrics.rms_lateral_error
            + 0.3 * metrics.rms_heading_error
            + 0.2 * metrics.rms_speed_error
            + 0.1 * metrics.avg_solve_time_ms / 100.0
        )

        return metrics

    def benchmark_computation_time(
        self,
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark MPC computation time vs horizon length.

        Args:
            horizons: List of prediction horizons to test.

        Returns:
            Dictionary mapping horizon to timing results.
        """
        if horizons is None:
            horizons = [5, 10, 15, 20, 25, 30, 40, 50]

        results = {}

        for N in horizons:
            config = TrackerConfig(
                prediction_horizon=N,
                control_horizon=min(N, 15),
            )
            metrics = self.benchmark_tracking(config, "s_curve", n_steps=500)
            results[N] = {
                "avg_solve_time_ms": metrics.avg_solve_time_ms,
                "max_solve_time_ms": metrics.max_solve_time_ms,
                "rms_lateral_error_m": metrics.rms_lateral_error,
            }

        return results

    def compare_with_different_weights(
        self,
    ) -> Dict[str, MPCBenchmarkMetrics]:
        """Compare MPC with different weight configurations.

        Returns:
            Dictionary mapping configuration name to metrics.
        """
        configs = {
            "Default": TrackerConfig(),
            "Aggressive Tracking": TrackerConfig(
                q_lateral=100.0, q_heading=200.0, q_speed=20.0,
                r_steer=2.0, r_accel=0.5,
            ),
            "Smooth Control": TrackerConfig(
                q_lateral=20.0, q_heading=50.0, q_speed=5.0,
                r_steer=20.0, r_accel=5.0,
                rd_steer=200.0, rd_accel=50.0,
            ),
            "Safety Priority": TrackerConfig(
                q_lateral=80.0, q_heading=150.0, q_speed=10.0,
                r_steer=10.0, r_accel=2.0,
                feedforward_gain=0.8,
            ),
        }

        results = {}
        for name, config in configs.items():
            results[name] = self.benchmark_tracking(config, "s_curve")

        return results

    def print_results(self, results: Dict[str, MPCBenchmarkMetrics]) -> None:
        """Print benchmark results in a formatted table.

        Args:
            results: Dictionary mapping name to metrics.
        """
        print("=" * 90)
        print("MPC Controller Benchmark Results")
        print("=" * 90)
        print(f"{'Config':<25} {'Avg Time':<10} {'RMS Lat':<10} {'RMS Head':<10} "
              f"{'RMS Spd':<10} {'Violations':<10} {'Accuracy':<10}")
        print("-" * 90)

        for name, m in results.items():
            print(f"{name:<25} {m.avg_solve_time_ms:<10.3f} "
                  f"{m.rms_lateral_error:<10.4f} "
                  f"{math.degrees(m.rms_heading_error):<10.4f} "
                  f"{m.rms_speed_error:<10.4f} "
                  f"{m.constraint_violations:<10} "
                  f"{m.tracking_accuracy:<10.4f}")

        print("=" * 90)


def run_quick_benchmark() -> Dict[str, MPCBenchmarkMetrics]:
    """Run a quick MPC benchmark with default settings.

    Returns:
        Dictionary mapping scenario name to metrics.
    """
    suite = MPCBenchmarkSuite(dt=0.01)
    config = TrackerConfig()

    scenarios = ["straight", "curve", "s_curve", "lane_change"]
    results = {}

    for scenario in scenarios:
        results[scenario] = suite.benchmark_tracking(config, scenario, n_steps=1000)

    return results
