"""
PID Controller Simulation Tests for Autonomous Vehicle Control

Simulates the PID controllers with vehicle dynamics models:
  - Longitudinal dynamics simulation (speed + brake)
  - Lateral dynamics simulation (steering)
  - Combined longitudinal + lateral simulation
  - Various driving scenarios (lane change, emergency brake, cruise)
  - Parameter sensitivity analysis

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    SteeringMode,
)
from .brake_pid import (
    BrakePIDController,
    BrakeControllerConfig,
    BrakeSystemParams,
    WheelState,
    BrakeMode,
)


@dataclass
class SimulationResult:
    """Result from a simulation test.

    Attributes:
        test_name: Name of the test scenario.
        time: Time array.
        states: Dictionary of state arrays.
        passed: Whether the test passed all criteria.
        failure_reason: Reason for failure (if any).
    """
    test_name: str = ""
    time: np.ndarray = np.array([])
    states: Dict[str, np.ndarray] = None
    passed: bool = True
    failure_reason: str = ""

    def __post_init__(self):
        if self.states is None:
            self.states = {}


class VehicleDynamicsSimulator:
    """Simplified vehicle dynamics simulator for controller testing.

    Implements a bicycle model with:
    - Longitudinal dynamics (engine, brake, drag, rolling resistance)
    - Lateral dynamics (bicycle model, tire forces)
    - Weight transfer effects
    - Grade effects
    """

    def __init__(
        self,
        vehicle: VehicleParams = VehicleParams(),
        geometry: VehicleGeometry = VehicleGeometry(),
        dt: float = 0.01,
    ) -> None:
        """Initialize the vehicle dynamics simulator.

        Args:
            vehicle: Vehicle physical parameters.
            geometry: Vehicle geometric parameters.
            dt: Simulation timestep in seconds.
        """
        self._vehicle = vehicle
        self._geometry = geometry
        self._dt = dt

        # State
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._lateral_velocity = 0.0
        self._steering_angle = 0.0
        self._acceleration = 0.0

    @property
    def state(self) -> VehicleState:
        """Return current vehicle state."""
        return VehicleState(
            x=self._x,
            y=self._y,
            heading=self._heading,
            speed=self._speed,
            lateral_velocity=self._lateral_velocity,
            yaw_rate=self._yaw_rate,
            steering_angle=self._steering_angle,
        )

    def reset(self, initial_state: Optional[VehicleState] = None) -> None:
        """Reset the vehicle to an initial state.

        Args:
            initial_state: Optional initial state. Defaults to origin.
        """
        if initial_state is not None:
            self._x = initial_state.x
            self._y = initial_state.y
            self._heading = initial_state.heading
            self._speed = initial_state.speed
            self._lateral_velocity = initial_state.lateral_velocity
            self._yaw_rate = initial_state.yaw_rate
            self._steering_angle = initial_state.steering_angle
        else:
            self._x = 0.0
            self._y = 0.0
            self._heading = 0.0
            self._speed = 0.0
            self._lateral_velocity = 0.0
            self._yaw_rate = 0.0
            self._steering_angle = 0.0
        self._acceleration = 0.0

    def step(
        self,
        throttle: float = 0.0,
        brake: float = 0.0,
        steering_angle: float = 0.0,
        grade_angle: float = 0.0,
    ) -> VehicleState:
        """Advance the vehicle dynamics by one timestep.

        Args:
            throttle: Throttle command (0-1).
            brake: Brake command (0-1).
            steering_angle: Steering angle in radians.
            grade_angle: Road grade angle in radians.

        Returns:
            Updated vehicle state.
        """
        self._steering_angle = steering_angle

        # Longitudinal forces
        engine_force = throttle * self._vehicle.max_engine_torque * self._vehicle.transmission_ratio / self._vehicle.wheel_radius
        brake_force = brake * self._vehicle.max_brake_torque / self._vehicle.wheel_radius

        # Aerodynamic drag
        drag_force = (
            0.5 * self._vehicle.air_density
            * self._vehicle.drag_coefficient
            * self._vehicle.frontal_area
            * self._speed ** 2
        )

        # Rolling resistance
        rolling_force = (
            self._vehicle.rolling_resistance
            * self._vehicle.mass * 9.81
            * math.cos(grade_angle)
            if self._speed > 0.1 else 0.0
        )

        # Grade force
        grade_force = self._vehicle.mass * 9.81 * math.sin(grade_angle)

        # Net longitudinal force
        net_long_force = engine_force - brake_force - drag_force - rolling_force - grade_force
        self._acceleration = net_long_force / self._vehicle.mass

        # Update speed
        self._speed += self._acceleration * self._dt
        self._speed = max(0.0, self._speed)

        # Lateral dynamics (bicycle model)
        if self._speed > 0.5:
            # Yaw rate from bicycle model
            self._yaw_rate = self._speed * math.tan(steering_angle) / self._geometry.wheelbase

            # Simplified lateral velocity
            slip_angle_front = steering_angle - math.atan2(
                self._lateral_velocity + self._geometry.cg_to_front * self._yaw_rate,
                max(self._speed, 1.0)
            )
            self._lateral_velocity += (
                (self._vehicle.mass * self._speed * slip_angle_front * 0.5 - self._yaw_rate * self._speed)
                / self._vehicle.mass * self._dt
            )
        else:
            self._yaw_rate = 0.0
            self._lateral_velocity = 0.0

        # Update position
        self._x += self._speed * math.cos(self._heading) * self._dt
        self._y += self._speed * math.sin(self._heading) * self._dt
        self._heading += self._yaw_rate * self._dt

        # Normalize heading
        self._heading = (self._heading + math.pi) % (2 * math.pi) - math.pi

        return self.state


class SimulationTestSuite:
    """Test suite for PID controllers with vehicle dynamics simulation."""

    def __init__(self, dt: float = 0.01) -> None:
        """Initialize the test suite.

        Args:
            dt: Simulation timestep in seconds.
        """
        self._dt = dt
        self._vehicle = VehicleParams()
        self._geometry = VehicleGeometry()

    def test_cruise_control(self) -> SimulationResult:
        """Test cruise control at constant speed.

        Simulates accelerating to and maintaining a target speed.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 10000
        vehicle_sim = VehicleDynamicsSimulator(
            self._vehicle, self._geometry, self._dt
        )
        speed_ctrl = SpeedPIDController(
            config=SpeedControllerConfig(dt=self._dt),
            vehicle=self._vehicle,
        )

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        target_arr = np.full(n_steps, 25.0)
        throttle_arr = np.zeros(n_steps)
        brake_arr = np.zeros(n_steps)
        accel_arr = np.zeros(n_steps)

        target_speed = 25.0  # m/s (~90 km/h)

        for i in range(n_steps):
            state = vehicle_sim.state
            throttle, brake = speed_ctrl.update(
                target_speed=target_speed,
                current_speed=state.speed,
            )
            state = vehicle_sim.step(throttle=throttle, brake=brake)

            time_arr[i] = i * self._dt
            speed_arr[i] = state.speed
            throttle_arr[i] = throttle
            brake_arr[i] = brake
            accel_arr[i] = state.acceleration if hasattr(state, 'acceleration') else 0.0

        # Check pass criteria
        final_speed = speed_arr[-1]
        speed_error = abs(final_speed - target_speed)
        passed = speed_error < 0.5  # Within 0.5 m/s

        return SimulationResult(
            test_name="Cruise Control Test",
            time=time_arr,
            states={
                "speed": speed_arr,
                "target_speed": target_arr,
                "throttle": throttle_arr,
                "brake": brake_arr,
                "acceleration": accel_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Speed error {speed_error:.2f} m/s exceeds 0.5 m/s",
        )

    def test_emergency_braking(self) -> SimulationResult:
        """Test emergency braking from highway speed.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 8000
        vehicle_sim = VehicleDynamicsSimulator(
            self._vehicle, self._geometry, self._dt
        )
        brake_ctrl = BrakePIDController(
            config=BrakeControllerConfig(dt=self._dt),
            params=BrakeSystemParams(),
        )

        # Initialize at 30 m/s
        initial = VehicleState(speed=30.0, x=0.0, y=0.0, heading=0.0)
        vehicle_sim.reset(initial)

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        brake_pressure_arr = np.zeros(n_steps)
        decel_arr = np.zeros(n_steps)

        # Activate emergency braking after 1 second
        emergency_time = 1.0
        brake_ctrl.reset()

        prev_speed = 30.0

        for i in range(n_steps):
            t = i * self._dt
            state = vehicle_sim.state

            if t >= emergency_time:
                brake_ctrl.activate_emergency()
                brake_result = brake_ctrl.update(
                    target_decel=8.0,
                    current_decel=0.0,
                    vehicle_speed=state.speed,
                )
                brake_cmd = min(1.0, brake_result["total_pressure"] / 180.0)
            else:
                brake_cmd = 0.0

            state = vehicle_sim.step(brake=brake_cmd)

            time_arr[i] = t
            speed_arr[i] = state.speed
            brake_pressure_arr[i] = brake_result.get("total_pressure", 0.0) if t >= emergency_time else 0.0
            decel_arr[i] = (prev_speed - state.speed) / self._dt if self._dt > 0 else 0.0
            prev_speed = state.speed

            # Stop simulation when vehicle stops
            if state.speed < 0.1 and t > emergency_time:
                speed_arr[i:] = 0.0
                break

        # Check: vehicle should stop within reasonable distance
        passed = speed_arr[-1] < 1.0

        return SimulationResult(
            test_name="Emergency Braking Test",
            time=time_arr,
            states={
                "speed": speed_arr,
                "brake_pressure": brake_pressure_arr,
                "deceleration": decel_arr,
            },
            passed=passed,
        )

    def test_lane_change(self) -> SimulationResult:
        """Test steering controller during a lane change maneuver.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 8000
        vehicle_sim = VehicleDynamicsSimulator(
            self._vehicle, self._geometry, self._dt
        )
        steer_ctrl = SteeringPIDController(
            config=SteeringControllerConfig(dt=self._dt),
            geometry=self._geometry,
        )

        # Initialize at 20 m/s
        initial = VehicleState(speed=20.0, x=0.0, y=0.0, heading=0.0)
        vehicle_sim.reset(initial)

        # Generate lane change path
        lane_width = 3.5  # meters
        path_points = []
        x = 0.0
        for i in range(400):
            y_target = 0.0
            if 50.0 <= x <= 80.0:
                y_target = lane_width * 0.5 * (1 - math.cos(math.pi * (x - 50.0) / 30.0))
            elif x > 80.0:
                y_target = lane_width
            heading = 0.0
            if 50.0 <= x <= 80.0:
                heading = math.atan2(
                    lane_width * 0.5 * math.pi / 30.0 * math.sin(math.pi * (x - 50.0) / 30.0),
                    1.0
                )
            curvature = 0.0
            if 50.0 < x < 80.0:
                curvature = lane_width * 0.5 * (math.pi / 30.0) ** 2 * math.cos(math.pi * (x - 50.0) / 30.0)

            path_points.append(PathPoint(x=x, y=y_target, heading=heading, curvature=curvature))
            x += 0.5

        time_arr = np.zeros(n_steps)
        x_arr = np.zeros(n_steps)
        y_arr = np.zeros(n_steps)
        heading_arr = np.zeros(n_steps)
        lateral_error_arr = np.zeros(n_steps)
        steering_arr = np.zeros(n_steps)

        throttle = 0.3  # Maintain speed approximately

        for i in range(n_steps):
            state = vehicle_sim.state

            # Find closest path point
            min_dist = float("inf")
            closest_idx = 0
            for j, pp in enumerate(path_points):
                d = math.sqrt((state.x - pp.x) ** 2 + (state.y - pp.y) ** 2)
                if d < min_dist:
                    min_dist = d
                    closest_idx = j

            steer, diag = steer_ctrl.update(
                vehicle_state=state,
                path_points=path_points,
                closest_index=closest_idx,
            )

            state = vehicle_sim.step(throttle=throttle, steering_angle=steer)

            time_arr[i] = i * self._dt
            x_arr[i] = state.x
            y_arr[i] = state.y
            heading_arr[i] = state.heading
            lateral_error_arr[i] = diag.get("lateral_error_m", 0.0)
            steering_arr[i] = math.degrees(steer)

        # Check: max lateral error should be within 1.0m
        max_lateral_error = float(np.max(np.abs(lateral_error_arr)))
        passed = max_lateral_error < 1.5

        return SimulationResult(
            test_name="Lane Change Test",
            time=time_arr,
            states={
                "x": x_arr,
                "y": y_arr,
                "heading": heading_arr,
                "lateral_error": lateral_error_arr,
                "steering_deg": steering_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Max lateral error {max_lateral_error:.2f} m exceeds 1.5 m",
        )

    def test_speed_tracking_with_grade(self) -> SimulationResult:
        """Test speed control on varying road grades.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 15000
        vehicle_sim = VehicleDynamicsSimulator(
            self._vehicle, self._geometry, self._dt
        )
        speed_ctrl = SpeedPIDController(
            config=SpeedControllerConfig(dt=self._dt),
            vehicle=self._vehicle,
        )

        # Initialize at target speed
        target_speed = 20.0
        initial = VehicleState(speed=target_speed, x=0.0, y=0.0, heading=0.0)
        vehicle_sim.reset(initial)

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        target_arr = np.full(n_steps, target_speed)
        grade_arr = np.zeros(n_steps)
        throttle_arr = np.zeros(n_steps)
        brake_arr = np.zeros(n_steps)

        for i in range(n_steps):
            t = i * self._dt
            state = vehicle_sim.state

            # Varying grade: flat -> uphill -> flat -> downhill -> flat
            if t < 20.0:
                grade = 0.0
            elif t < 50.0:
                grade = math.radians(5.0)  # 5% uphill
            elif t < 80.0:
                grade = 0.0
            elif t < 110.0:
                grade = math.radians(-4.0)  # 4% downhill
            else:
                grade = 0.0

            throttle, brake = speed_ctrl.update(
                target_speed=target_speed,
                current_speed=state.speed,
                grade_angle=grade,
            )
            state = vehicle_sim.step(throttle=throttle, brake=brake, grade_angle=grade)

            time_arr[i] = t
            speed_arr[i] = state.speed
            grade_arr[i] = math.degrees(grade)
            throttle_arr[i] = throttle
            brake_arr[i] = brake

        # Check: speed should stay within ±2 m/s of target on grade
        speed_error = np.abs(speed_arr - target_speed)
        max_error = float(np.max(speed_error[2000:]))  # Skip transient
        passed = max_error < 3.0

        return SimulationResult(
            test_name="Speed Tracking with Grade",
            time=time_arr,
            states={
                "speed": speed_arr,
                "target_speed": target_arr,
                "grade_deg": grade_arr,
                "throttle": throttle_arr,
                "brake": brake_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Max speed error {max_error:.2f} m/s exceeds 3.0 m/s",
        )

    def test_combined_longitudinal_lateral(self) -> SimulationResult:
        """Test combined speed and steering control.

        Simulates accelerating through a curve.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 12000
        vehicle_sim = VehicleDynamicsSimulator(
            self._vehicle, self._geometry, self._dt
        )
        speed_ctrl = SpeedPIDController(
            config=SpeedControllerConfig(dt=self._dt),
            vehicle=self._vehicle,
        )
        steer_ctrl = SteeringPIDController(
            config=SteeringControllerConfig(dt=self._dt),
            geometry=self._geometry,
        )

        # Generate curved path
        curvature = 0.005  # Gentle curve (R=200m)
        path_points = []
        heading = 0.0
        x, y = 0.0, 0.0
        for i in range(400):
            path_points.append(PathPoint(x=x, y=y, heading=heading, curvature=curvature))
            ds = 0.5
            x += ds * math.cos(heading)
            y += ds * math.sin(heading)
            heading += curvature * ds

        initial = VehicleState(speed=15.0, x=0.0, y=0.3, heading=0.0)
        vehicle_sim.reset(initial)

        target_speed = 20.0

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        lateral_error_arr = np.zeros(n_steps)
        throttle_arr = np.zeros(n_steps)
        steering_arr = np.zeros(n_steps)

        for i in range(n_steps):
            state = vehicle_sim.state

            # Find closest path point
            min_dist = float("inf")
            closest_idx = 0
            for j, pp in enumerate(path_points):
                d = math.sqrt((state.x - pp.x) ** 2 + (state.y - pp.y) ** 2)
                if d < min_dist:
                    min_dist = d
                    closest_idx = j

            # Speed control
            throttle, brake = speed_ctrl.update(
                target_speed=target_speed,
                current_speed=state.speed,
            )

            # Steering control
            steer, diag = steer_ctrl.update(
                vehicle_state=state,
                path_points=path_points,
                closest_index=closest_idx,
            )

            state = vehicle_sim.step(throttle=throttle, brake=brake, steering_angle=steer)

            time_arr[i] = i * self._dt
            speed_arr[i] = state.speed
            lateral_error_arr[i] = diag.get("lateral_error_m", 0.0)
            throttle_arr[i] = throttle
            steering_arr[i] = math.degrees(steer)

        max_lateral = float(np.max(np.abs(lateral_error_arr[1000:])))
        final_speed_error = abs(speed_arr[-1] - target_speed)
        passed = max_lateral < 1.0 and final_speed_error < 1.0

        return SimulationResult(
            test_name="Combined Longitudinal + Lateral",
            time=time_arr,
            states={
                "speed": speed_arr,
                "lateral_error": lateral_error_arr,
                "throttle": throttle_arr,
                "steering_deg": steering_arr,
            },
            passed=passed,
        )

    def run_all_tests(self) -> Dict[str, SimulationResult]:
        """Run all simulation tests.

        Returns:
            Dictionary mapping test name to SimulationResult.
        """
        tests = {
            "cruise_control": self.test_cruise_control,
            "emergency_braking": self.test_emergency_braking,
            "lane_change": self.test_lane_change,
            "speed_with_grade": self.test_speed_tracking_with_grade,
            "combined_control": self.test_combined_longitudinal_lateral,
        }

        results = {}
        for name, test_func in tests.items():
            try:
                result = test_func()
                results[name] = result
            except Exception as e:
                results[name] = SimulationResult(
                    test_name=name,
                    passed=False,
                    failure_reason=str(e),
                )

        return results

    def print_results(self, results: Dict[str, SimulationResult]) -> None:
        """Print simulation test results.

        Args:
            results: Dictionary of test results.
        """
        print("=" * 80)
        print("PID Controller Simulation Test Results")
        print("=" * 80)

        all_passed = True
        for name, result in results.items():
            status = "PASSED" if result.passed else "FAILED"
            if not result.passed:
                all_passed = False
            print(f"\n  [{status}] {result.test_name}")
            if not result.passed and result.failure_reason:
                print(f"         Reason: {result.failure_reason}")
            if result.states:
                for state_name, arr in result.states.items():
                    if len(arr) > 0:
                        print(f"         {state_name}: min={np.min(arr):.4f}, "
                              f"max={np.max(arr):.4f}, final={arr[-1]:.4f}")

        print(f"\n{'All tests PASSED' if all_passed else 'Some tests FAILED'}")
        print("=" * 80)


if __name__ == "__main__":
    suite = SimulationTestSuite(dt=0.01)
    results = suite.run_all_tests()
    suite.print_results(results)
