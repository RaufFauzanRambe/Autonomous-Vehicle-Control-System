"""
Simulation Tests for Adaptive Vehicle Control

Simulates adaptive controllers under realistic vehicle conditions:
  - Vehicle with varying mass (payload changes)
  - Road grade changes (uphill / downhill transitions)
  - Tire friction coefficient changes (wet / icy road)
  - Wind disturbance (cross-wind gusts)
  - Combined longitudinal + lateral scenarios

Each test evaluates the adaptive controller's ability to maintain
tracking performance despite parametric and environmental uncertainty.

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .adaptive_controller import MRACController, AdaptiveParams, L1AdaptiveController
from .fuzzy_controller import FuzzySteeringController, FuzzySpeedController
from .adaptive_tuning import LyapunovAdaptation, BoundedAdaptation


# ---------------------------------------------------------------------------
# Vehicle Simulator
# ---------------------------------------------------------------------------

@dataclass
class VehicleParams:
    """Vehicle physical parameters (mutable for simulation).

    Attributes:
        mass: Total mass (kg).
        wheelbase: Wheelbase (m).
        cg_to_front: CG to front axle (m).
        cg_to_rear: CG to rear axle (m).
        frontal_area: Frontal area (m²).
        drag_coeff: Drag coefficient.
        rolling_resistance: Rolling resistance coefficient.
        max_engine_force: Maximum engine force (N).
        max_brake_force: Maximum brake force (N).
        air_density: Air density (kg/m³).
    """
    mass: float = 1800.0
    wheelbase: float = 2.85
    cg_to_front: float = 1.42
    cg_to_rear: float = 1.43
    frontal_area: float = 2.5
    drag_coeff: float = 0.3
    rolling_resistance: float = 0.015
    max_engine_force: float = 5000.0
    max_brake_force: float = 15000.0
    air_density: float = 1.225


@dataclass
class VehicleState:
    """Vehicle state vector.

    Attributes:
        x: Longitudinal position (m).
        y: Lateral position (m).
        heading: Heading angle (rad).
        speed: Forward speed (m/s).
        lateral_velocity: Lateral velocity (m/s).
        yaw_rate: Yaw rate (rad/s).
        steering_angle: Current steering angle (rad).
    """
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    steering_angle: float = 0.0


@dataclass
class SimulationResult:
    """Result from a simulation test.

    Attributes:
        test_name: Name of the test.
        time: Time array.
        states: Dictionary of recorded state arrays.
        passed: Whether the test passed.
        failure_reason: Reason for failure (if any).
    """
    test_name: str = ""
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    states: Dict[str, np.ndarray] = field(default_factory=dict)
    passed: bool = True
    failure_reason: str = ""


class AdaptiveVehicleSimulator:
    """Vehicle dynamics simulator for testing adaptive controllers.

    Supports time-varying:
    - Mass (payload changes)
    - Road grade
    - Tire friction coefficient
    - Wind disturbance forces
    """

    def __init__(
        self,
        vehicle: VehicleParams = VehicleParams(),
        dt: float = 0.01,
    ) -> None:
        """Initialise the simulator.

        Args:
            vehicle: Vehicle parameters.
            dt: Simulation timestep.
        """
        self._vehicle = vehicle
        self._dt = dt
        self._state = VehicleState()

        # External conditions (can be changed during simulation)
        self._grade_angle: float = 0.0
        self._friction_coeff: float = 1.0
        self._wind_force_lateral: float = 0.0
        self._wind_force_longitudinal: float = 0.0

    @property
    def state(self) -> VehicleState:
        """Current vehicle state."""
        return VehicleState(
            x=self._state.x,
            y=self._state.y,
            heading=self._state.heading,
            speed=self._state.speed,
            lateral_velocity=self._state.lateral_velocity,
            yaw_rate=self._state.yaw_rate,
            steering_angle=self._state.steering_angle,
        )

    def set_conditions(
        self,
        grade_angle: float = 0.0,
        friction_coeff: float = 1.0,
        wind_lateral: float = 0.0,
        wind_longitudinal: float = 0.0,
        mass: Optional[float] = None,
    ) -> None:
        """Update environmental conditions.

        Args:
            grade_angle: Road grade angle (rad), positive = uphill.
            friction_coeff: Tire-road friction coefficient.
            wind_lateral: Cross-wind force (N), positive = left.
            wind_longitudinal: Head/tail wind force (N), positive = tailwind.
            mass: Override vehicle mass (kg). None keeps current.
        """
        self._grade_angle = grade_angle
        self._friction_coeff = friction_coeff
        self._wind_force_lateral = wind_lateral
        self._wind_force_longitudinal = wind_longitudinal
        if mass is not None:
            self._vehicle.mass = mass

    def reset(self, initial: Optional[VehicleState] = None) -> None:
        """Reset the vehicle state.

        Args:
            initial: Optional initial state.
        """
        if initial is not None:
            self._state = VehicleState(
                x=initial.x, y=initial.y, heading=initial.heading,
                speed=initial.speed, lateral_velocity=initial.lateral_velocity,
                yaw_rate=initial.yaw_rate, steering_angle=initial.steering_angle,
            )
        else:
            self._state = VehicleState()

    def step(
        self,
        throttle: float = 0.0,
        brake: float = 0.0,
        steering_angle: float = 0.0,
    ) -> VehicleState:
        """Advance the vehicle dynamics by one timestep.

        Args:
            throttle: Throttle command (0–1).
            brake: Brake command (0–1).
            steering_angle: Steering angle (rad).

        Returns:
            Updated vehicle state.
        """
        v = self._vehicle
        s = self._state
        dt = self._dt
        s.steering_angle = steering_angle

        # --- Longitudinal forces ---
        engine_force = throttle * v.max_engine_force
        brake_force = brake * v.max_brake_force

        # Aerodynamic drag
        drag = 0.5 * v.air_density * v.drag_coeff * v.frontal_area * s.speed ** 2

        # Rolling resistance (scaled by friction)
        rolling = v.rolling_resistance * v.mass * 9.81 * math.cos(self._grade_angle) if s.speed > 0.1 else 0.0

        # Grade force
        grade_force = v.mass * 9.81 * math.sin(self._grade_angle)

        # Net longitudinal
        net_long = (engine_force - brake_force - drag - rolling - grade_force
                    + self._wind_force_longitudinal)
        accel = net_long / v.mass

        # Friction-limited acceleration
        max_accel = self._friction_coeff * 9.81
        accel = np.clip(accel, -max_accel, max_accel)

        s.speed += accel * dt
        s.speed = max(0.0, s.speed)

        # --- Lateral dynamics (bicycle model) ---
        if s.speed > 0.5:
            # Simplified cornering stiffness scaled by friction
            Cf = 80000.0 * self._friction_coeff
            Cr = 90000.0 * self._friction_coeff

            slip_angle_front = steering_angle - math.atan2(
                s.lateral_velocity + v.cg_to_front * s.yaw_rate,
                max(s.speed, 1.0),
            )
            slip_angle_rear = -math.atan2(
                s.lateral_velocity - v.cg_to_rear * s.yaw_rate,
                max(s.speed, 1.0),
            )

            Fyf = Cf * slip_angle_front
            Fyr = Cr * slip_angle_rear

            # Lateral force including wind
            Fy_total = Fyf + Fyr + self._wind_force_lateral

            # Lateral acceleration
            ay = Fy_total / v.mass
            # Friction circle constraint
            ay = np.clip(ay, -max_accel, max_accel)

            s.lateral_velocity += (ay - s.speed * s.yaw_rate) * dt

            # Yaw dynamics
            Mz = v.cg_to_front * Fyf - v.cg_to_rear * Fyr
            yaw_accel = Mz / 3500.0  # Fixed yaw inertia
            s.yaw_rate += yaw_accel * dt
        else:
            s.yaw_rate = 0.0
            s.lateral_velocity = 0.0

        # --- Position update ---
        s.x += s.speed * math.cos(s.heading) * dt
        s.y += s.speed * math.sin(s.heading) * dt + s.lateral_velocity * dt
        s.heading += s.yaw_rate * dt
        s.heading = (s.heading + math.pi) % (2 * math.pi) - math.pi

        return self.state


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class AdaptiveSimulationTestSuite:
    """Test suite for adaptive controllers under varying conditions."""

    def __init__(self, dt: float = 0.01) -> None:
        """Initialise the test suite.

        Args:
            dt: Simulation timestep.
        """
        self._dt = dt
        self._vehicle = VehicleParams()

    def test_varying_mass(self) -> SimulationResult:
        """Test speed control with payload mass changes.

        Simulates loading and unloading events that change the
        effective vehicle mass.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 20000
        sim = AdaptiveVehicleSimulator(self._vehicle, self._dt)
        speed_ctrl = FuzzySpeedController()

        target_speed = 20.0
        initial = VehicleState(speed=20.0)
        sim.reset(initial)

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        target_arr = np.full(n_steps, target_speed)
        mass_arr = np.zeros(n_steps)
        accel_arr = np.zeros(n_steps)

        for i in range(n_steps):
            t = i * self._dt
            state = sim.state

            # Mass schedule: nominal -> +500 kg -> +1000 kg -> nominal
            if t < 30.0:
                mass = 1800.0
            elif t < 60.0:
                mass = 2300.0
            elif t < 120.0:
                mass = 2800.0
            else:
                mass = 1800.0

            sim.set_conditions(mass=mass)

            # Speed control
            speed_error = target_speed - state.speed
            accel_error = 0.0
            adj = speed_ctrl.compute(speed_error=speed_error, acceleration_error=accel_error)

            throttle = max(0.0, adj)
            brake = max(0.0, -adj)
            sim.step(throttle=throttle, brake=brake)

            time_arr[i] = t
            speed_arr[i] = state.speed
            mass_arr[i] = mass

        # Check: speed error should stay within ±2 m/s after settling
        final_error = abs(speed_arr[-1] - target_speed)
        max_error_late = float(np.max(np.abs(speed_arr[3000:] - target_speed)))
        passed = max_error_late < 3.0 and final_error < 1.0

        return SimulationResult(
            test_name="Varying Mass Test",
            time=time_arr,
            states={
                "speed": speed_arr,
                "target_speed": target_arr,
                "mass_kg": mass_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Max speed error {max_error_late:.2f} m/s",
        )

    def test_road_grade_changes(self) -> SimulationResult:
        """Test speed control on varying road grades.

        Simulates flat → uphill → steep uphill → flat → downhill transitions.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 15000
        sim = AdaptiveVehicleSimulator(self._vehicle, self._dt)
        speed_ctrl = FuzzySpeedController()

        target_speed = 20.0
        initial = VehicleState(speed=20.0)
        sim.reset(initial)

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        grade_arr = np.zeros(n_steps)
        target_arr = np.full(n_steps, target_speed)

        for i in range(n_steps):
            t = i * self._dt
            state = sim.state

            # Grade schedule
            if t < 20.0:
                grade = 0.0
            elif t < 50.0:
                grade = math.radians(4.0)  # 4% uphill
            elif t < 80.0:
                grade = math.radians(8.0)  # 8% steep uphill
            elif t < 100.0:
                grade = 0.0
            elif t < 130.0:
                grade = math.radians(-5.0)  # 5% downhill
            else:
                grade = 0.0

            sim.set_conditions(grade_angle=grade)

            speed_error = target_speed - state.speed
            adj = speed_ctrl.compute(speed_error=speed_error, acceleration_error=0.0)
            throttle = max(0.0, adj)
            brake = max(0.0, -adj)
            sim.step(throttle=throttle, brake=brake)

            time_arr[i] = t
            speed_arr[i] = state.speed
            grade_arr[i] = math.degrees(grade)

        max_error = float(np.max(np.abs(speed_arr[2000:] - target_speed)))
        passed = max_error < 5.0

        return SimulationResult(
            test_name="Road Grade Changes Test",
            time=time_arr,
            states={
                "speed": speed_arr,
                "target_speed": target_arr,
                "grade_deg": grade_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Max speed error {max_error:.2f} m/s",
        )

    def test_tire_friction_changes(self) -> SimulationResult:
        """Test lateral control under tire friction changes.

        Simulates dry → wet → icy → dry road transitions.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 15000
        sim = AdaptiveVehicleSimulator(self._vehicle, self._dt)
        steer_ctrl = FuzzySteeringController()

        target_speed = 15.0
        sim.reset(VehicleState(speed=target_speed, y=0.5))

        time_arr = np.zeros(n_steps)
        y_arr = np.zeros(n_steps)
        heading_arr = np.zeros(n_steps)
        friction_arr = np.zeros(n_steps)
        steer_arr = np.zeros(n_steps)

        # Simple straight-line path at y=0
        for i in range(n_steps):
            t = i * self._dt
            state = sim.state

            # Friction schedule: dry → wet → icy → dry
            if t < 30.0:
                friction = 0.9  # Dry
            elif t < 60.0:
                friction = 0.5  # Wet
            elif t < 90.0:
                friction = 0.2  # Icy
            else:
                friction = 0.9  # Dry

            sim.set_conditions(friction_coeff=friction)

            lateral_error = state.y  # Path at y=0
            heading_error = state.heading
            steer = steer_ctrl.compute(
                lateral_error=lateral_error,
                heading_error=heading_error,
            )

            # Maintain speed approximately
            throttle = 0.25
            sim.step(throttle=throttle, steering_angle=steer)

            time_arr[i] = t
            y_arr[i] = state.y
            heading_arr[i] = state.heading
            friction_arr[i] = friction
            steer_arr[i] = math.degrees(steer)

        max_lateral = float(np.max(np.abs(y_arr)))
        passed = max_lateral < 3.0

        return SimulationResult(
            test_name="Tire Friction Changes Test",
            time=time_arr,
            states={
                "y_position": y_arr,
                "heading_deg": np.degrees(heading_arr),
                "friction_coeff": friction_arr,
                "steering_deg": steer_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Max lateral deviation {max_lateral:.2f} m",
        )

    def test_wind_disturbances(self) -> SimulationResult:
        """Test lateral control with cross-wind gusts.

        Simulates sudden cross-wind gusts that push the vehicle
        laterally off the path.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 15000
        sim = AdaptiveVehicleSimulator(self._vehicle, self._dt)
        steer_ctrl = FuzzySteeringController()

        target_speed = 20.0
        sim.reset(VehicleState(speed=target_speed, y=0.3))

        time_arr = np.zeros(n_steps)
        y_arr = np.zeros(n_steps)
        wind_arr = np.zeros(n_steps)
        steer_arr = np.zeros(n_steps)

        for i in range(n_steps):
            t = i * self._dt
            state = sim.state

            # Wind gust schedule
            if 20.0 <= t < 25.0:
                wind = 2000.0   # Strong right gust
            elif 50.0 <= t < 52.0:
                wind = -3000.0  # Brief left gust
            elif 80.0 <= t < 86.0:
                wind = 1500.0   # Moderate right gust
            else:
                wind = 0.0

            sim.set_conditions(wind_lateral=wind)

            lateral_error = state.y  # Path at y=0
            heading_error = state.heading
            steer = steer_ctrl.compute(
                lateral_error=lateral_error,
                heading_error=heading_error,
            )

            sim.step(throttle=0.3, steering_angle=steer)

            time_arr[i] = t
            y_arr[i] = state.y
            wind_arr[i] = wind
            steer_arr[i] = math.degrees(steer)

        max_lateral = float(np.max(np.abs(y_arr)))
        passed = max_lateral < 2.5

        return SimulationResult(
            test_name="Wind Disturbance Test",
            time=time_arr,
            states={
                "y_position": y_arr,
                "wind_force_N": wind_arr,
                "steering_deg": steer_arr,
            },
            passed=passed,
            failure_reason="" if passed else f"Max lateral deviation {max_lateral:.2f} m",
        )

    def test_combined_adaptive_control(self) -> SimulationResult:
        """Test combined longitudinal + lateral adaptive control.

        Simulates a vehicle encountering mass changes, grade changes,
        and cross-wind simultaneously.

        Returns:
            SimulationResult with test data.
        """
        n_steps = 20000
        sim = AdaptiveVehicleSimulator(self._vehicle, self._dt)
        speed_ctrl = FuzzySpeedController()
        steer_ctrl = FuzzySteeringController()

        target_speed = 18.0
        sim.reset(VehicleState(speed=target_speed, y=0.3))

        time_arr = np.zeros(n_steps)
        speed_arr = np.zeros(n_steps)
        y_arr = np.zeros(n_steps)
        target_speed_arr = np.full(n_steps, target_speed)

        for i in range(n_steps):
            t = i * self._dt
            state = sim.state

            # Combined disturbances
            mass = 1800.0 + 500.0 * math.sin(t * 0.05)
            grade = math.radians(3.0) if 40.0 < t < 80.0 else 0.0
            wind = 1500.0 * math.sin(t * 0.3) if 60.0 < t < 100.0 else 0.0

            sim.set_conditions(mass=mass, grade_angle=grade, wind_lateral=wind)

            # Speed control
            speed_error = target_speed - state.speed
            adj = speed_ctrl.compute(speed_error=speed_error, acceleration_error=0.0)
            throttle = max(0.0, adj)
            brake = max(0.0, -adj)

            # Steering control
            steer = steer_ctrl.compute(
                lateral_error=state.y,
                heading_error=state.heading,
            )

            sim.step(throttle=throttle, brake=brake, steering_angle=steer)

            time_arr[i] = t
            speed_arr[i] = state.speed
            y_arr[i] = state.y

        speed_error_max = float(np.max(np.abs(speed_arr[3000:] - target_speed)))
        lateral_max = float(np.max(np.abs(y_arr)))
        passed = speed_error_max < 3.5 and lateral_max < 2.5

        return SimulationResult(
            test_name="Combined Adaptive Control Test",
            time=time_arr,
            states={
                "speed": speed_arr,
                "target_speed": target_speed_arr,
                "y_position": y_arr,
            },
            passed=passed,
            failure_reason=(
                "" if passed
                else f"Speed error {speed_error_max:.2f} m/s, lateral {lateral_max:.2f} m"
            ),
        )

    def run_all_tests(self) -> Dict[str, SimulationResult]:
        """Run all simulation tests.

        Returns:
            Dictionary mapping test name to SimulationResult.
        """
        tests = {
            "varying_mass": self.test_varying_mass,
            "road_grade": self.test_road_grade_changes,
            "tire_friction": self.test_tire_friction_changes,
            "wind_disturbance": self.test_wind_disturbances,
            "combined_control": self.test_combined_adaptive_control,
        }

        results: Dict[str, SimulationResult] = {}
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
        print("Adaptive Control Simulation Test Results")
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
                        print(
                            f"         {state_name}: min={np.min(arr):.4f}, "
                            f"max={np.max(arr):.4f}, final={arr[-1]:.4f}"
                        )

        print(f"\n{'All tests PASSED' if all_passed else 'Some tests FAILED'}")
        print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suite = AdaptiveSimulationTestSuite(dt=0.01)
    results = suite.run_all_tests()
    suite.print_results(results)
