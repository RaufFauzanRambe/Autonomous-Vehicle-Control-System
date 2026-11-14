"""Kinematic vehicle state container for the basic simulation module.

The state is a plain flat 2-D pose plus scalar longitudinal dynamics; it is
the single exchange format between the vehicle model (``vehicle_model.py``),
the road environment and any downstream consumer (plots, summaries, logging).

Extension point: swap this dataclass for a full 6-DOF state (roll/pitch,
world-frame velocities, wheel slip angles) once the bicycle model is
replaced by a dynamic model with tire force computation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class VehicleState:
    """Kinematic vehicle state in a flat 2-D world frame.

    Attributes:
        timestamp: Simulation time [s].
        x: World-frame x position [m].
        y: World-frame y position [m].
        yaw: Heading angle [rad], CCW positive, 0 rad aligned with the +x axis.
        speed: Body-frame longitudinal velocity [m/s] (always >= 0 here).
        acceleration: Body-frame longitudinal acceleration [m/s^2].
        steering_angle: Front wheel steering angle [rad], positive = left.

    Extension point: add lateral velocity and yaw rate as explicit fields
    when moving from the kinematic to the dynamic bicycle model.
    """

    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    steering_angle: float = 0.0

    def to_dict(self) -> dict:
        """Convert the state to a plain dictionary.

        Returns:
            Dictionary with all seven state fields, keyed by attribute name
            and holding plain ``float`` values (JSON/pandas friendly).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleState":
        """Build a state from a dictionary, tolerating missing keys.

        Missing keys fall back to the dataclass defaults and every present
        value is coerced to ``float`` so that YAML/JSON/CSV round trips are
        robust against integer-typed input.

        Args:
            d: Mapping possibly containing any subset of the seven state
                keys (``timestamp``, ``x``, ``y``, ``yaw``, ``speed``,
                ``acceleration``, ``steering_angle``).

        Returns:
            A new :class:`VehicleState` instance.

        Extension point: add schema validation against
        ``shared/schemas/vehicle_state.schema.json`` before coercion.
        """
        defaults = cls()
        return cls(
            timestamp=float(d.get("timestamp", defaults.timestamp)),
            x=float(d.get("x", defaults.x)),
            y=float(d.get("y", defaults.y)),
            yaw=float(d.get("yaw", defaults.yaw)),
            speed=float(d.get("speed", defaults.speed)),
            acceleration=float(d.get("acceleration", defaults.acceleration)),
            steering_angle=float(d.get("steering_angle", defaults.steering_angle)),
        )
