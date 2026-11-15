"""Closed-loop vehicle response simulation: speed PID + heading steering.

The longitudinal controller is a discrete PID whose output is a normalized
throttle in [-1, 1] that is consumed directly by the kinematic bicycle
model from module 01 (positive = acceleration, negative = braking). The
lateral controller is a simplified pure-pursuit-style heading controller:
the reference point one lookahead distance ahead on the reference polyline
defines a desired heading, and the steering angle is proportional to the
wrapped heading error, clipped to the vehicle steering limit.

Both controllers run at the fixed integration step ``simulation.dt_s`` and
the plant is :class:`vehicle.vehicle_model.VehicleModel`, imported from
module 01 per the project-wide namespace-package convention.

Extension point: replace the P-only heading law with the geometric pure
pursuit steering ``atan(2 * wheelbase * sin(alpha) / lookahead_m)``, a
Stanley controller or a lateral MPC; replace the single-track PID with a
cascade (position -> speed) or gain-scheduled longitudinal controller.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_CONTROL_DIR = Path(__file__).resolve().parent
MODULE_DIR = _CONTROL_DIR.parent  # 06_motion_control_simulation/
ROOT = MODULE_DIR.parent          # project root (shared utilities, module 01)

sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_basic_simulation"))

from vehicle.vehicle_model import VehicleModel
from vehicle.vehicle_state import VehicleState


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval ``[-pi, pi)``.

    Args:
        angle: Unwrapped angle [rad].

    Returns:
        Equivalent angle [rad] in ``[-pi, pi)``.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PIDController:
    """Discrete proportional-integral-derivative controller.

    The controller acts on the error ``e = setpoint - measurement`` with
    derivative-on-error. Anti-windup is implemented by conditional
    integration: while the clamped output is saturated and the error would
    push the output deeper into saturation, the integral state is frozen
    for that step. The final output is clamped to ``output_limits``.

    Attributes:
        _integral: Current integral of the error [error-unit * s]; kept as
            an instance attribute so callers can inspect the windup state.

    Extension point: add back-calculation (integral bleed via a tracking
    time constant), feed-forward of the inverse plant model, or
    gain scheduling on speed for a research-grade longitudinal controller.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: Tuple[float, float] | None = None,
    ) -> None:
        """Configure the controller gains and saturation limits.

        Args:
            kp: Proportional gain multiplying the speed error [m/s] to
                yield the controller output.
            ki: Integral gain multiplying the accumulated error
                [m/s * s]; see ``update`` for the discrete accumulation.
            kd: Derivative gain multiplying the error rate of change
                [m/s per s].
            output_limits: Optional ``(low, high)`` clamp applied to the
                output (for a throttle: ``(-1.0, 1.0)`` [-]). ``low`` must
                be strictly smaller than ``high``.

        Raises:
            ValueError: If ``output_limits`` is given with ``low >= high``.
        """
        if output_limits is not None:
            low, high = float(output_limits[0]), float(output_limits[1])
            if low >= high:
                raise ValueError(f"output_limits must satisfy low < high, got {output_limits}")
            self._output_limits: Tuple[float, float] | None = (low, high)
        else:
            self._output_limits = None
        self._kp: float = float(kp)
        self._ki: float = float(ki)
        self._kd: float = float(kd)
        self._integral: float = 0.0
        self._prev_error: float | None = None

    @property
    def integral(self) -> float:
        """Current integral state [error-unit * s] (read-only view)."""
        return self._integral

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Advance the controller by one discrete step.

        The derivative term is zero on the very first call (no previous
        error is known), which avoids the classic derivative kick at
        start-up.

        Args:
            setpoint: Desired value (e.g. target speed) [m/s].
            measurement: Current measured value (e.g. tracked speed) [m/s].
            dt: Time step since the previous call [s]; must be positive.

        Returns:
            Clamped controller output [-] (a normalized throttle for the
            speed PID in this module).

        Raises:
            ValueError: If ``dt`` is not strictly positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        error = setpoint - measurement
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt

        tentative_integral = self._integral + error * dt
        raw_output = self._kp * error + self._ki * tentative_integral + self._kd * derivative
        if self._output_limits is not None:
            low, high = self._output_limits
            saturates_deeper = (raw_output > high and error > 0.0) or (
                raw_output < low and error < 0.0
            )
            if saturates_deeper:
                tentative_integral = self._integral
        self._integral = tentative_integral

        output = self._kp * error + self._ki * self._integral + self._kd * derivative
        if self._output_limits is not None:
            low, high = self._output_limits
            output = min(max(output, low), high)
        self._prev_error = error
        return float(output)

    def reset(self) -> None:
        """Reset the integral and derivative memory to the initial state."""
        self._integral = 0.0
        self._prev_error = None


class SimulatedVehicleResponse:
    """Closed-loop longitudinal and lateral response for one scenario.

    The reference path is built from the scenario definition (straight line
    or constant-curvature arc, sampled at 1.0 m), the vehicle plant is the
    module 01 kinematic bicycle model, and the controllers are the module
    PID (speed) plus the pure-pursuit-style heading law (lateral).

    Args (constructor):
        scenario: Scenario dictionary with keys ``name`` [-],
            ``duration_s`` [s], ``initial_state`` (``x`` [m], ``y`` [m],
            ``yaw`` [rad], ``speed`` [m/s]), ``path`` (``type`` [-] of
            ``"straight"`` or ``"constant_curvature"``, plus
            ``heading_rad`` [rad] / ``curvature_1_per_m`` [1/m] and
            ``length_m`` [m]) and ``target_speed_mps`` [m/s].
        vehicle_params: Vehicle parameter dictionary consumed by
            :class:`vehicle.vehicle_model.VehicleModel` (keys:
            ``wheelbase_m`` [m], ``max_steering_rad`` [rad],
            ``max_speed_mps`` [m/s], ``max_accel_mpss`` [m/s^2],
            ``max_decel_mpss`` [m/s^2], ``drag_coefficient`` [1/s],
            ``mass_kg`` [kg]).
        control_config: Control and simulation configuration with
            ``control.pid_speed`` (``kp``, ``ki``, ``kd``,
            ``output_limits``), ``control.steering_gain`` [-],
            ``control.lookahead_m`` [m] and ``simulation.dt_s`` [s].

    Extension point: add replanning of the reference at runtime (e.g. from
    a module 05 planner output) and keep the same ``run`` interface so
    closed-loop experiments remain drop-in comparable.
    """

    REFERENCE_SAMPLE_DS_M: float = 1.0  # [m] arc-length sampling of the reference polyline

    def __init__(
        self,
        scenario: Dict[str, object],
        vehicle_params: Dict[str, float],
        control_config: Dict[str, object],
    ) -> None:
        """Validate and store the scenario, plant and controller setup.

        Args:
            scenario: See class docstring for the required keys and units.
            vehicle_params: Vehicle model parameter dictionary.
            control_config: Control/simulation configuration dictionary.

        Raises:
            ValueError: If mandatory scenario keys are missing or the path
                type is not supported.
        """
        for key in ("name", "duration_s", "initial_state", "path", "target_speed_mps"):
            if key not in scenario:
                raise ValueError(f"Scenario is missing mandatory key: {key!r}")
        path_type = str(scenario["path"]["type"])
        if path_type not in ("straight", "constant_curvature"):
            raise ValueError(f"Unsupported path type: {path_type!r}")

        self.scenario: Dict[str, object] = dict(scenario)
        self.vehicle_params: Dict[str, float] = dict(vehicle_params)
        self.control_config: Dict[str, object] = dict(control_config)
        self.vehicle: VehicleModel = VehicleModel(self.vehicle_params)

    def _build_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the reference polyline from the scenario geometry.

        ``"straight"`` paths run from the initial ``x``/``y`` position along
        the constant ``heading_rad``; ``"constant_curvature"`` paths
        integrate ``heading += curvature * ds`` and
        ``position += [cos(heading), sin(heading)] * ds`` from the initial
        pose, both with ``ds = 1.0`` m.

        Returns:
            Tuple ``(reference_x, reference_y)``: polyline coordinates [m],
            each of shape ``(N,)`` with ``N = length_m / ds + 1``.
        """
        path = self.scenario["path"]
        initial = self.scenario["initial_state"]
        x0 = float(initial["x"])  # type: ignore[index]
        y0 = float(initial["y"])  # type: ignore[index]
        yaw0 = float(initial["yaw"])  # type: ignore[index]
        length_m = float(path["length_m"])  # type: ignore[index]
        n_samples = max(1, int(round(length_m / self.REFERENCE_SAMPLE_DS_M)))
        ds = length_m / n_samples

        path_type = str(path["type"])
        reference_x = np.zeros(n_samples + 1, dtype=float)
        reference_y = np.zeros(n_samples + 1, dtype=float)
        reference_x[0] = x0
        reference_y[0] = y0

        if path_type == "straight":
            heading = float(path["heading_rad"])  # type: ignore[index]
            arc_lengths = np.arange(1, n_samples + 1, dtype=float) * ds
            reference_x[1:] = x0 + arc_lengths * math.cos(heading)
            reference_y[1:] = y0 + arc_lengths * math.sin(heading)
            return reference_x, reference_y

        curvature = float(path["curvature_1_per_m"])  # type: ignore[index]
        heading = yaw0
        for index in range(1, n_samples + 1):
            heading += curvature * ds
            reference_x[index] = reference_x[index - 1] + math.cos(heading) * ds
            reference_y[index] = reference_y[index - 1] + math.sin(heading) * ds
        return reference_x, reference_y

    @staticmethod
    def _nearest_reference_state(
        reference_x: np.ndarray,
        reference_y: np.ndarray,
        cumulative_length_m: np.ndarray,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """Project a pose onto the reference polyline.

        The projection is computed analytically per segment (the closest
        point on each segment is clamped to the segment bounds) and the
        segment with the smallest squared distance wins.

        Args:
            reference_x: Polyline x coordinates [m], shape ``(N,)``.
            reference_y: Polyline y coordinates [m], shape ``(N,)``.
            cumulative_length_m: Cumulative arc length at every polyline
                vertex [m], shape ``(N,)``.
            x: Ego x position [m].
            y: Ego y position [m].

        Returns:
            Tuple ``(arc_length_m [m], cross_track_error_m [m])`` where
            ``arc_length_m`` locates the projection on the polyline and
            ``cross_track_error_m`` is the perpendicular distance to the
            nearest reference segment.
        """
        if reference_x.size < 2:
            distance = math.hypot(x - float(reference_x[0]), y - float(reference_y[0]))
            return 0.0, distance

        segment_dx = np.diff(reference_x)
        segment_dy = np.diff(reference_y)
        segment_length = np.hypot(segment_dx, segment_dy)
        squared_length = np.where(segment_length > 0.0, segment_length**2, 1.0)
        projection = np.clip(
            ((x - reference_x[:-1]) * segment_dx + (y - reference_y[:-1]) * segment_dy)
            / squared_length,
            0.0,
            1.0,
        )
        closest_x = reference_x[:-1] + projection * segment_dx
        closest_y = reference_y[:-1] + projection * segment_dy
        squared_distance = (x - closest_x) ** 2 + (y - closest_y) ** 2
        nearest_index = int(np.argmin(squared_distance))
        arc_length = float(
            cumulative_length_m[nearest_index]
            + projection[nearest_index] * segment_length[nearest_index]
        )
        cross_track_error = float(math.sqrt(squared_distance[nearest_index]))
        return arc_length, cross_track_error

    def run(self) -> Dict[str, object]:
        """Simulate the closed loop for the scenario duration.

        The simulated duration is clamped to ``scenario.duration_s`` [s]
        and the episode additionally terminates as soon as the ego reaches
        the end of the reference polyline (whichever comes first). The
        lateral controller steers toward the reference point located
        ``control.lookahead_m`` [m] ahead of the ego projection; near the
        end of the polyline the lookahead point saturates at the final
        vertex.

        Returns:
            Dictionary with keys ``name`` [-] (scenario identifier),
            ``time`` [s], ``speed`` [m/s], ``target_speed`` [m/s],
            ``steering`` [rad], ``x`` [m], ``y`` [m], ``yaw`` [rad],
            ``cross_track_error`` [m] (all except ``name`` are numpy
            arrays of identical length) and ``reference_x`` / ``reference_y``
            [m] (the reference polyline used for plotting).

        Raises:
            ValueError: If ``simulation.dt_s`` is not strictly positive.

        Extension point: log the full per-step controller state (integral
        term, lookahead point, desired heading) to a CSV/parquet trace for
        controller-identification studies, and terminate on the true stop
        condition of the maneuver instead of the polyline end.
        """
        dt = float(self.control_config["simulation"]["dt_s"])  # type: ignore[index]
        if dt <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt}")
        pid_settings = self.control_config["control"]["pid_speed"]  # type: ignore[index]
        steering_gain = float(self.control_config["control"]["steering_gain"])  # type: ignore[index]
        lookahead_m = float(self.control_config["control"]["lookahead_m"])  # type: ignore[index]

        pid = PIDController(
            kp=float(pid_settings["kp"]),  # type: ignore[index]
            ki=float(pid_settings["ki"]),  # type: ignore[index]
            kd=float(pid_settings["kd"]),  # type: ignore[index]
            output_limits=tuple(pid_settings["output_limits"]),  # type: ignore[index]
        )

        target_speed = float(self.scenario["target_speed_mps"])
        max_duration_s = float(self.scenario["duration_s"])
        max_steering_rad = float(self.vehicle_params["max_steering_rad"])
        n_steps_max = max(1, int(round(max_duration_s / dt)))

        reference_x, reference_y = self._build_reference()
        segment_length = np.hypot(np.diff(reference_x), np.diff(reference_y))
        cumulative_length = np.concatenate(([0.0], np.cumsum(segment_length)))
        total_length_m = float(cumulative_length[-1])

        initial = self.scenario["initial_state"]
        state = VehicleState(
            timestamp=0.0,
            x=float(initial["x"]),  # type: ignore[index]
            y=float(initial["y"]),  # type: ignore[index]
            yaw=float(initial["yaw"]),  # type: ignore[index]
            speed=float(initial["speed"]),  # type: ignore[index]
            acceleration=0.0,
            steering_angle=0.0,
        )

        time = np.zeros(n_steps_max + 1, dtype=float)
        speed = np.zeros(n_steps_max + 1, dtype=float)
        target_speed_array = np.full(n_steps_max + 1, target_speed, dtype=float)
        steering = np.zeros(n_steps_max + 1, dtype=float)
        x_array = np.zeros(n_steps_max + 1, dtype=float)
        y_array = np.zeros(n_steps_max + 1, dtype=float)
        yaw_array = np.zeros(n_steps_max + 1, dtype=float)
        cross_track = np.zeros(n_steps_max + 1, dtype=float)

        n_recorded = 0
        for step_index in range(n_steps_max + 1):
            arc_length, cross_track_error = self._nearest_reference_state(
                reference_x, reference_y, cumulative_length, state.x, state.y
            )
            time[step_index] = step_index * dt
            speed[step_index] = state.speed
            steering[step_index] = state.steering_angle
            x_array[step_index] = state.x
            y_array[step_index] = state.y
            yaw_array[step_index] = state.yaw
            cross_track[step_index] = cross_track_error
            n_recorded = step_index + 1

            if step_index == n_steps_max:
                break
            if arc_length >= total_length_m:
                break  # reference fully traversed before the time limit

            throttle = pid.update(target_speed, state.speed, dt)

            lookahead_arc = min(arc_length + lookahead_m, total_length_m)
            lookahead_x = float(np.interp(lookahead_arc, cumulative_length, reference_x))
            lookahead_y = float(np.interp(lookahead_arc, cumulative_length, reference_y))
            delta_x = lookahead_x - state.x
            delta_y = lookahead_y - state.y
            if math.hypot(delta_x, delta_y) < 1e-9:
                desired_heading = state.yaw  # ego sits on the lookahead point
            else:
                desired_heading = math.atan2(delta_y, delta_x)
            heading_error = wrap_angle(desired_heading - state.yaw)
            steering_command = float(
                min(
                    max(steering_gain * heading_error, -max_steering_rad),
                    max_steering_rad,
                )
            )

            state = self.vehicle.step(state, throttle, steering_command, dt)

        return {
            "name": str(self.scenario["name"]),
            "time": time[:n_recorded],
            "speed": speed[:n_recorded],
            "target_speed": target_speed_array[:n_recorded],
            "steering": steering[:n_recorded],
            "x": x_array[:n_recorded],
            "y": y_array[:n_recorded],
            "yaw": yaw_array[:n_recorded],
            "cross_track_error": cross_track[:n_recorded],
            "reference_x": reference_x,
            "reference_y": reference_y,
        }
