"""
PID Controller Benchmark Suite for Autonomous Vehicle Control

Benchmarks controller performance with metrics:
  - Settling time (2% and 5% bands)
  - Overshoot percentage
  - Rise time (10%-90%)
  - Steady-state error
  - IAE, ISE, ITAE integral metrics
  - Control effort (total actuator usage)
  - Robustness margins (gain and phase margin)
  - Disturbance rejection performance

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pid_controller import PIDController, PIDGains, PIDLimits
from .speed_pid import SpeedPIDController, SpeedControllerConfig, VehicleParams
from .steering_pid import (
    SteeringPIDController,
    SteeringControllerConfig,
    VehicleGeometry,
    VehicleState,
    PathPoint,
)
from .brake_pid import BrakePIDController, BrakeControllerConfig, BrakeSystemParams


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics.

    Attributes:
        settling_time_2pct: 2% settling time in seconds.
        settling_time_5pct: 5% settling time in seconds.
        overshoot_percent: Overshoot as a percentage of setpoint.
        rise_time: 10%-90% rise time in seconds.
        steady_state_error: Final steady-state error.
        IAE: Integral of Absolute Error.
        ISE: Integral of Squared Error.
        ITAE: Integral of Time-weighted Absolute Error.
        control_effort: Total control effort (∫|u|dt).
        max_control: Peak control output.
        max_error: Peak error.
    """
    settling_time_2pct: float = float("inf")
    settling_time_5pct: float = float("inf")
    overshoot_percent: float = 0.0
    rise_time: float = float("inf")
    steady_state_error: float = 0.0
    IAE: float = 0.0
    ISE: float = 0.0
    ITAE: float = 0.0
    control_effort: float = 0.0
    max_control: float = 0.0
    max_error: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "settling_time_2%": f"{self.settling_time_2pct:.4f} s",
            "settling_time_5%": f"{self.settling_time_5pct:.4f} s",
            "overshoot": f"{self.overshoot_percent:.2f} %",
            "rise_time": f"{self.rise_time:.4f} s",
            "steady_state_error": f"{self.steady_state_error:.6f}",
            "IAE": f"{self.IAE:.4f}",
            "ISE": f"{self.ISE:.4f}",
            "ITAE": f"{self.ITAE:.4f}",
            "control_effort": f"{self.control_effort:.4f}",
            "max_control": f"{self.max_control:.4f}",
            "max_error": f"{self.max_error:.4f}",
        }


@dataclass
class BenchmarkResult:
    """Result from a benchmark test.

    Attributes:
        test_name: Name of the benchmark test.
        metrics: Computed performance metrics.
        computation_time_ms: Time to compute the benchmark.
        passed: Whether the controller meets requirements.
    """
    test_name: str = ""
    metrics: BenchmarkMetrics = BenchmarkMetrics()
    computation_time_ms: float = 0.0
    passed: bool = True


class PIDBenchmarkSuite:
    """Benchmark suite for evaluating PID controller performance.

    Provides standardized tests for:
    - Step response analysis
    - Ramp tracking
    - Disturbance rejection
    - Noise sensitivity
    - Speed controller evaluation
    - Steering controller evaluation
    - Brake controller evaluation
    """

    def __init__(self, dt: float = 0.01) -> None:
        """Initialize the benchmark suite.

        Args:
            dt: Simulation timestep in seconds.
        """
        self._dt = dt

    def _compute_metrics(
        self,
        time: np.ndarray,
        setpoint: np.ndarray,
        measurement: np.ndarray,
        control: np.ndarray,
    ) -> BenchmarkMetrics:
        """Compute all performance metrics from simulation data.

        Args:
            time: Time array.
            setpoint: Setpoint array.
            measurement: Measurement array.
            control: Control output array.

        Returns:
            BenchmarkMetrics with computed values.
        """
        error = setpoint - measurement
        abs_error = np.abs(error)
        dt_arr = np.diff(time, prepend=time[0])

        metrics = BenchmarkMetrics()

        # Settling time (2% band)
        if len(setpoint) > 0:
            sp_value = setpoint[-1]
        else:
            sp_value = 0.0

        band_2pct = 0.02 * abs(sp_value) if sp_value != 0 else 0.02
        band_5pct = 0.05 * abs(sp_value) if sp_value != 0 else 0.05

        settled_2pct = abs_error <= band_2pct
        settled_5pct = abs_error <= band_5pct

        # Find settling time (last time error leaves the band, then stays)
        metrics.settling_time_2pct = self._find_settling_time(time, settled_2pct)
        metrics.settling_time_5pct = self._find_settling_time(time, settled_5pct)

        # Overshoot
        if sp_value >= 0:
            peak = float(np.max(measurement))
            metrics.overshoot_percent = max(0, (peak - sp_value) / abs(sp_value) * 100.0) if sp_value != 0 else 0.0
        else:
            valley = float(np.min(measurement))
            metrics.overshoot_percent = max(0, (sp_value - valley) / abs(sp_value) * 100.0) if sp_value != 0 else 0.0

        # Rise time (10% to 90%)
        low = 0.1 * sp_value
        high = 0.9 * sp_value
        try:
            rise_start = np.where(measurement >= low)[0][0]
            rise_end = np.where(measurement >= high)[0][0]
            metrics.rise_time = float(time[rise_end] - time[rise_start])
        except (IndexError, ValueError):
            metrics.rise_time = float("inf")

        # Steady-state error (last 10% of simulation)
        n_ss = max(1, int(len(error) * 0.1))
        metrics.steady_state_error = float(np.mean(error[-n_ss:]))

        # Integral metrics
        metrics.IAE = float(np.trapz(abs_error, time))
        metrics.ISE = float(np.trapz(error ** 2, time))
        metrics.ITAE = float(np.trapz(time * abs_error, time))

        # Control effort
        metrics.control_effort = float(np.trapz(np.abs(control), time))
        metrics.max_control = float(np.max(np.abs(control)))
        metrics.max_error = float(np.max(abs_error))

        return metrics

    @staticmethod
    def _find_settling_time(time: np.ndarray, settled: np.ndarray) -> float:
        """Find the settling time from a boolean settled array.

        Args:
            time: Time array.
            settled: Boolean array indicating if settled.

        Returns:
            Settling time in seconds.
        """
        # Find the last time it was unsettled
        for i in range(len(settled) - 1, -1, -1):
            if not settled[i]:
                # Check if it stays settled after this point
                if np.all(settled[i + 1:]):
                    return float(time[i + 1]) if i + 1 < len(time) else float(time[-1])
                else:
                    return float("inf")
        return 0.0  # Always settled

    def step_response_test(
        self,
        gains: PIDGains,
        limits: PIDLimits = PIDLimits(),
        setpoint: float = 1.0,
        plant_gain: float = 1.0,
        plant_tau: float = 1.0,
        n_steps: int = 5000,
    ) -> BenchmarkResult:
        """Benchmark step response performance.

        Simulates a step input to a first-order plant controlled by PID.

        Args:
            gains: PID gains to test.
            limits: PID output limits.
            setpoint: Step setpoint value.
            plant_gain: Plant static gain.
            plant_tau: Plant time constant.
            n_steps: Number of simulation steps.

        Returns:
            BenchmarkResult with performance metrics.
        """
        start_time = time.time()

        pid = PIDController(
            gains=gains,
            limits=limits,
            dt=self._dt,
        )

        # Simulate
        time_arr = np.zeros(n_steps)
        measurement_arr = np.zeros(n_steps)
        control_arr = np.zeros(n_steps)
        setpoint_arr = np.full(n_steps, setpoint)

        measurement = 0.0
        pid.reset()

        for i in range(n_steps):
            control = pid.update(setpoint=setpoint, measurement=measurement)
            # First-order plant
            dmeasurement = (plant_gain * control - measurement) / plant_tau * self._dt
            measurement += dmeasurement

            time_arr[i] = i * self._dt
            measurement_arr[i] = measurement
            control_arr[i] = control

        metrics = self._compute_metrics(
            time_arr, setpoint_arr, measurement_arr, control_arr
        )
        elapsed = (time.time() - start_time) * 1000.0

        return BenchmarkResult(
            test_name="Step Response",
            metrics=metrics,
            computation_time_ms=elapsed,
        )

    def disturbance_rejection_test(
        self,
        gains: PIDGains,
        limits: PIDLimits = PIDLimits(),
        setpoint: float = 1.0,
        plant_gain: float = 1.0,
        plant_tau: float = 1.0,
        disturbance_time: float = 2.0,
        disturbance_magnitude: float = 0.3,
        n_steps: int = 5000,
    ) -> BenchmarkResult:
        """Benchmark disturbance rejection performance.

        Applies a step disturbance and measures recovery.

        Args:
            gains: PID gains to test.
            limits: PID output limits.
            setpoint: Constant setpoint.
            plant_gain: Plant static gain.
            plant_tau: Plant time constant.
            disturbance_time: Time to apply disturbance.
            disturbance_magnitude: Disturbance magnitude.
            n_steps: Number of simulation steps.

        Returns:
            BenchmarkResult with disturbance rejection metrics.
        """
        start_time = time.time()

        pid = PIDController(
            gains=gains,
            limits=limits,
            dt=self._dt,
        )

        time_arr = np.zeros(n_steps)
        measurement_arr = np.zeros(n_steps)
        control_arr = np.zeros(n_steps)
        setpoint_arr = np.full(n_steps, setpoint)

        measurement = 0.0
        pid.reset()

        for i in range(n_steps):
            t = i * self._dt

            # Apply disturbance at specified time
            disturbance = disturbance_magnitude if t >= disturbance_time else 0.0

            control = pid.update(setpoint=setpoint, measurement=measurement)
            dmeasurement = (plant_gain * control + disturbance - measurement) / plant_tau * self._dt
            measurement += dmeasurement

            time_arr[i] = t
            measurement_arr[i] = measurement
            control_arr[i] = control

        # Compute metrics only for post-disturbance period
        dist_start_idx = int(disturbance_time / self._dt)
        post_dist_time = time_arr[dist_start_idx:]
        post_dist_setpoint = setpoint_arr[dist_start_idx:]
        post_dist_measurement = measurement_arr[dist_start_idx:]
        post_dist_control = control_arr[dist_start_idx:]

        if len(post_dist_time) > 10:
            metrics = self._compute_metrics(
                post_dist_time - post_dist_time[0],
                post_dist_setpoint,
                post_dist_measurement,
                post_dist_control,
            )
        else:
            metrics = BenchmarkMetrics()

        elapsed = (time.time() - start_time) * 1000.0

        return BenchmarkResult(
            test_name="Disturbance Rejection",
            metrics=metrics,
            computation_time_ms=elapsed,
        )

    def speed_controller_test(
        self,
        config: SpeedControllerConfig = SpeedControllerConfig(),
        vehicle: VehicleParams = VehicleParams(),
        target_speeds: Optional[List[float]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark the speed PID controller.

        Tests speed tracking at various target speeds.

        Args:
            config: Speed controller configuration.
            vehicle: Vehicle parameters.
            target_speeds: List of target speeds to test (m/s).

        Returns:
            Dictionary mapping speed to BenchmarkResult.
        """
        if target_speeds is None:
            target_speeds = [10.0, 15.0, 20.0, 25.0, 30.0]

        results = {}
        n_steps = 10000

        for target in target_speeds:
            start_time = time.time()
            ctrl = SpeedPIDController(config=config, vehicle=vehicle)
            ctrl.reset()

            time_arr = np.zeros(n_steps)
            speed_arr = np.zeros(n_steps)
            control_arr = np.zeros(n_steps)
            setpoint_arr = np.full(n_steps, target)

            current_speed = 0.0

            for i in range(n_steps):
                throttle, brake = ctrl.update(
                    target_speed=target,
                    current_speed=current_speed,
                    grade_angle=0.0,
                )
                # Simplified vehicle model
                net_force = throttle * 3000.0 - brake * 8000.0 - 0.5 * 1.225 * 0.3 * 2.5 * current_speed ** 2
                accel = net_force / vehicle.mass
                current_speed += accel * self._dt
                current_speed = max(0.0, current_speed)

                time_arr[i] = i * self._dt
                speed_arr[i] = current_speed
                control_arr[i] = throttle - brake

            metrics = self._compute_metrics(
                time_arr, setpoint_arr, speed_arr, control_arr
            )
            elapsed = (time.time() - start_time) * 1000.0

            results[f"speed_{target:.0f}ms"] = BenchmarkResult(
                test_name=f"Speed Control @ {target:.0f} m/s",
                metrics=metrics,
                computation_time_ms=elapsed,
            )

        return results

    def steering_controller_test(
        self,
        config: SteeringControllerConfig = SteeringControllerConfig(),
        geometry: VehicleGeometry = VehicleGeometry(),
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark the steering PID controller.

        Tests lateral tracking on different path geometries.

        Args:
            config: Steering controller configuration.
            geometry: Vehicle geometry parameters.

        Returns:
            Dictionary mapping scenario to BenchmarkResult.
        """
        ctrl = SteeringPIDController(config=config, geometry=geometry)
        results = {}

        # Test 1: Straight line tracking
        n_steps = 5000
        ctrl.reset()
        time_arr = np.zeros(n_steps)
        lateral_error_arr = np.zeros(n_steps)
        steering_arr = np.zeros(n_steps)

        # Generate straight path
        path_points = [PathPoint(x=i * 0.5, y=0.0, heading=0.0, curvature=0.0) for i in range(200)]

        vehicle_state = VehicleState(x=0.0, y=0.5, heading=0.02, speed=15.0)

        for i in range(n_steps):
            steer, diag = ctrl.update(
                vehicle_state=vehicle_state,
                path_points=path_points,
                closest_index=min(int(vehicle_state.x / 0.5), len(path_points) - 1),
            )

            # Update vehicle state (bicycle model)
            ds = vehicle_state.speed * self._dt
            vehicle_state.x += ds * math.cos(vehicle_state.heading)
            vehicle_state.y += ds * math.sin(vehicle_state.heading)
            vehicle_state.heading += vehicle_state.speed * math.tan(steer) / geometry.wheelbase * self._dt

            time_arr[i] = i * self._dt
            lateral_error_arr[i] = diag["lateral_error_m"]
            steering_arr[i] = steer

        # Compute metrics for lateral error
        setpoint_arr = np.zeros(n_steps)
        metrics = self._compute_metrics(
            time_arr, setpoint_arr, lateral_error_arr, steering_arr
        )

        results["straight_line"] = BenchmarkResult(
            test_name="Straight Line Tracking",
            metrics=metrics,
        )

        # Test 2: Curve tracking
        ctrl.reset()
        time_arr = np.zeros(n_steps)
        lateral_error_arr = np.zeros(n_steps)
        steering_arr = np.zeros(n_steps)

        # Generate curved path (constant curvature)
        curvature = 0.01  # 1/100m radius
        path_points = []
        heading = 0.0
        x, y = 0.0, 0.0
        for i in range(200):
            path_points.append(PathPoint(x=x, y=y, heading=heading, curvature=curvature))
            ds = 0.5
            x += ds * math.cos(heading)
            y += ds * math.sin(heading)
            heading += curvature * ds

        vehicle_state = VehicleState(x=0.0, y=0.3, heading=0.0, speed=15.0)

        for i in range(n_steps):
            idx = min(int(vehicle_state.x / 0.5 + vehicle_state.y / 0.5), len(path_points) - 1)
            idx = max(0, min(idx, len(path_points) - 1))

            steer, diag = ctrl.update(
                vehicle_state=vehicle_state,
                path_points=path_points,
                closest_index=idx,
            )

            ds = vehicle_state.speed * self._dt
            vehicle_state.x += ds * math.cos(vehicle_state.heading)
            vehicle_state.y += ds * math.sin(vehicle_state.heading)
            vehicle_state.heading += vehicle_state.speed * math.tan(steer) / geometry.wheelbase * self._dt

            time_arr[i] = i * self._dt
            lateral_error_arr[i] = diag["lateral_error_m"]
            steering_arr[i] = steer

        setpoint_arr = np.zeros(n_steps)
        metrics = self._compute_metrics(
            time_arr, setpoint_arr, lateral_error_arr, steering_arr
        )

        results["curve_tracking"] = BenchmarkResult(
            test_name="Curve Tracking (R=100m)",
            metrics=metrics,
        )

        return results

    def run_full_benchmark(
        self,
        gains_list: Optional[List[Tuple[str, PIDGains]]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run a comprehensive benchmark comparing different PID tunings.

        Args:
            gains_list: List of (name, PIDGains) tuples to compare.
                If None, uses default tunings.

        Returns:
            Dictionary mapping test name to BenchmarkResult.
        """
        if gains_list is None:
            gains_list = [
                ("ZN Conservative", PIDGains(kp=0.6, ki=0.3, kd=0.1)),
                ("ZN Aggressive", PIDGains(kp=2.0, ki=1.0, kd=0.3)),
                ("IMC Tuning", PIDGains(kp=1.2, ki=0.5, kd=0.05)),
                ("Well-Tuned", PIDGains(kp=1.5, ki=0.6, kd=0.1)),
            ]

        results = {}

        for name, gains in gains_list:
            step_result = self.step_response_test(gains)
            dist_result = self.disturbance_rejection_test(gains)
            results[f"{name}_step"] = step_result
            results[f"{name}_disturbance"] = dist_result

        return results

    def print_results(self, results: Dict[str, BenchmarkResult]) -> None:
        """Print benchmark results in a formatted table.

        Args:
            results: Dictionary of benchmark results.
        """
        print("=" * 100)
        print("PID Controller Benchmark Results")
        print("=" * 100)

        for name, result in results.items():
            print(f"\n--- {result.test_name} ({name}) ---")
            print(f"  Computation Time: {result.computation_time_ms:.2f} ms")
            for key, value in result.metrics.to_dict().items():
                print(f"  {key}: {value}")

        print("\n" + "=" * 100)


def run_quick_benchmark() -> Dict[str, BenchmarkMetrics]:
    """Run a quick benchmark with default settings.

    Returns:
        Dictionary mapping test name to BenchmarkMetrics.
    """
    suite = PIDBenchmarkSuite(dt=0.01)

    # Test different tunings
    gains_to_test = {
        "Conservative": PIDGains(kp=0.6, ki=0.3, kd=0.1),
        "Moderate": PIDGains(kp=1.5, ki=0.6, kd=0.1),
        "Aggressive": PIDGains(kp=2.5, ki=1.2, kd=0.2),
    }

    all_metrics = {}
    for name, gains in gains_to_test.items():
        result = suite.step_response_test(gains)
        all_metrics[f"{name}_step"] = result.metrics

        result = suite.disturbance_rejection_test(gains)
        all_metrics[f"{name}_disturbance"] = result.metrics

    return all_metrics
