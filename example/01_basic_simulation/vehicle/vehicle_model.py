"""Kinematic bicycle vehicle model with Euler integration.

This module implements the simplest runnable plant model for the example
suite: a front-wheel-steering kinematic bicycle driven by a normalized
throttle/brake input and a front wheel angle. Assumptions:

* No tire slip: the vehicle instantaneously follows the bicycle kinematics.
* Front-wheel steering, rear-axle reference point is implicit in the
  chosen integration scheme (position advanced along the resulting heading).
* Flat road, no grade resistance, no pitch/roll dynamics.
* Speed is non-negative (reverse gear is not modelled).

Extension point: replace the Euler integration with RK4 and the throttle
mapping with a first-order actuator lag plus a dynamic bicycle (tire
cornering stiffness) for research-grade fidelity.
"""

from __future__ import annotations

import math
from typing import Dict

from vehicle.vehicle_state import VehicleState


class VehicleModel:
    """Kinematic bicycle model with explicit longitudinal limits.

    Args:
        params: Parameter dictionary with keys (defaults in
            ``vehicle/vehicle_parameters.yaml``):
            ``wheelbase_m`` [m], ``max_steering_rad`` [rad],
            ``max_speed_mps`` [m/s], ``max_accel_mpss`` [m/s^2],
            ``max_decel_mpss`` [m/s^2], ``drag_coefficient`` [1/s],
            ``mass_kg`` [kg] (kept for future dynamic upgrades; unused by
            the kinematic update).
    """

    def __init__(self, params: Dict[str, float]) -> None:
        """Store model parameters with safe fallbacks.

        Args:
            params: See class docstring. Unknown extra keys are ignored.
        """
        self.wheelbase_m: float = float(params.get("wheelbase_m", 2.7))
        self.max_steering_rad: float = float(params.get("max_steering_rad", 0.6))
        self.max_speed_mps: float = float(params.get("max_speed_mps", 33.0))
        self.max_accel_mpss: float = float(params.get("max_accel_mpss", 3.0))
        self.max_decel_mpss: float = float(params.get("max_decel_mpss", 8.0))
        self.drag_coefficient: float = float(params.get("drag_coefficient", 0.02))
        self.mass_kg: float = float(params.get("mass_kg", 1500.0))

    def step(
        self,
        state: VehicleState,
        throttle: float,
        steering_angle: float,
        dt: float,
    ) -> VehicleState:
        """Advance the state by one Euler step of the kinematic bicycle.

        Longitudinal update:
            ``accel = throttle * max_accel_mpss`` if ``throttle >= 0`` else
            ``throttle * max_decel_mpss`` (throttle in [-1, 1]).
            ``speed_next = clip(speed + (accel - drag_coefficient * speed) * dt,
            0.0, max_speed_mps)``.

        Heading and position update:
            ``steering = clip(steering_angle, +/- max_steering_rad)``,
            ``yaw_next = yaw + (speed / wheelbase_m) * tan(steering) * dt``,
            ``x_next = x + cos(yaw_next) * speed_next * dt``,
            ``y_next = y + sin(yaw_next) * speed_next * dt``.

        The reported acceleration is the realized change of speed
        ``(speed_next - speed) / dt`` (guarded for ``dt > 0``).

        The input ``state`` is never mutated; a new :class:`VehicleState` is
        returned.

        Args:
            state: Current vehicle state.
            throttle: Normalized longitudinal input [-], -1..1, negative
                = braking.
            steering_angle: Requested front wheel angle [rad], positive
                = left; clipped to the parameterized steering range.
            dt: Integration step [s], must be > 0 for a meaningful update.

        Returns:
            A new :class:`VehicleState` at ``timestamp + dt``.

        Raises:
            ValueError: If ``dt`` is not strictly positive.

        Extension point: replace the throttle->acceleration map with an
        engine/brake actuator model and integrate heading with the midpoint
        scheme ``yaw_next = yaw + v/L * tan(steer) * dt`` evaluated at the
        average speed for higher accuracy.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Longitudinal dynamics: commanded acceleration minus linear drag.
        if throttle >= 0.0:
            accel_cmd = throttle * self.max_accel_mpss
        else:
            accel_cmd = throttle * self.max_decel_mpss
        speed_next = state.speed + (accel_cmd - self.drag_coefficient * state.speed) * dt
        speed_next = float(min(max(speed_next, 0.0), self.max_speed_mps))

        # Steering is limited by the physical steering rack.
        steering = float(
            min(max(steering_angle, -self.max_steering_rad), self.max_steering_rad)
        )

        # Kinematic bicycle yaw update (uses the pre-step speed).
        yaw_next = state.yaw + (state.speed / self.wheelbase_m) * math.tan(steering) * dt

        # Advance the position along the updated heading (flat world).
        x_next = state.x + math.cos(yaw_next) * speed_next * dt
        y_next = state.y + math.sin(yaw_next) * speed_next * dt

        # Realized acceleration (guards the very first step where dt > 0 is
        # already enforced; keep the guard for numerical safety).
        realized_accel = (speed_next - state.speed) / dt if dt > 0.0 else 0.0

        return VehicleState(
            timestamp=state.timestamp + dt,
            x=x_next,
            y=y_next,
            yaw=yaw_next,
            speed=speed_next,
            acceleration=realized_accel,
            steering_angle=steering,
        )
