"""Environment model for the rule-based driving decision demo.

This module defines the structured view of the world that the decision
layer consumes:

- the ego vehicle state, reusing the ``VehicleState`` dataclass from
  module ``01_basic_simulation`` so that both modules share one physics
  vocabulary,
- dynamic actors (other vehicles and pedestrians) placed in the same
  local planning frame,
- static road features (traffic lights, pedestrian crossings, speed
  limits) expressed as a distance ahead of the ego vehicle.

All positions are metres in a common local planning frame; ``yaw`` and
``heading`` are radians measured counter-clockwise from the positive
x-axis; speeds are metres per second.

Extension point: add lane topology, occupancy grids or per-actor
prediction traces so that downstream decision rules can reason about
future conflict zones instead of the current snapshot only.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

MODULE_DIR = Path(__file__).resolve().parent            # .../07_ai_decision_demo/decision
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vehicle.vehicle_state import VehicleState  # noqa: E402  (path setup above)

ALLOWED_ACTOR_TYPES: tuple[str, ...] = ("vehicle", "pedestrian")
ALLOWED_FEATURE_TYPES: tuple[str, ...] = ("traffic_light", "pedestrian_crossing", "speed_limit")


@dataclass
class Actor:
    """A dynamic traffic participant observed around the ego vehicle.

    Attributes:
        id: Unique actor label, e.g. ``"veh_1"`` or ``"ped_1"`` [-].
        type: Actor class, one of ``{"vehicle", "pedestrian"}`` [-].
        x: Position along the planning-frame x-axis [m].
        y: Position along the planning-frame y-axis [m].
        speed: Scalar speed magnitude [m/s].
        heading: Travel direction, counter-clockwise from +x [rad].

    Raises:
        ValueError: If ``type`` is not an allowed actor type.
    """

    id: str
    type: str
    x: float
    y: float
    speed: float
    heading: float

    def __post_init__(self) -> None:
        if self.type not in ALLOWED_ACTOR_TYPES:
            raise ValueError(
                f"Actor '{self.id}' has unknown type '{self.type}'; "
                f"allowed types: {ALLOWED_ACTOR_TYPES}"
            )


@dataclass
class RoadFeature:
    """A static road feature relevant to the decision layer.

    Attributes:
        type: Feature class, one of ``{"traffic_light",
            "pedestrian_crossing", "speed_limit"}`` [-].
        distance_m: Distance ahead of the ego along its heading [m].
        state: Feature state label; e.g. ``"red"``/``"yellow"``/``"green"``
            for a traffic light, ``"occupied"``/``"clear"`` for a
            pedestrian crossing, or a numeric speed-limit label such as
            ``"50"`` [-].

    Raises:
        ValueError: If ``type`` is not an allowed road-feature type.
    """

    type: str
    distance_m: float
    state: str

    def __post_init__(self) -> None:
        if self.type not in ALLOWED_FEATURE_TYPES:
            raise ValueError(
                f"Road feature has unknown type '{self.type}'; "
                f"allowed types: {ALLOWED_FEATURE_TYPES}"
            )


@dataclass
class EnvironmentState:
    """A single snapshot of the driving environment.

    Attributes:
        ego: Ego vehicle state shared with module 01
            (x/y [m], yaw [rad], speed [m/s]).
        actors: Dynamic actors around the ego (may be empty).
        features: Static road features ahead of the ego (may be empty).
    """

    ego: VehicleState
    actors: List[Actor] = field(default_factory=list)
    features: List[RoadFeature] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, d: Dict[str, Any]) -> "EnvironmentState":
        """Build an environment snapshot from a raw scenario dictionary.

        Args:
            d: Scenario dictionary with keys ``"ego_state"`` (mapping with
                ``x`` [m], ``y`` [m], ``yaw`` [rad], ``speed`` [m/s]),
                ``"actors"`` (list of :class:`Actor`-shaped mappings) and
                ``"road_features"`` (list of :class:`RoadFeature`-shaped
                mappings). Extra keys (id, name, ...) are ignored.

        Returns:
            A fully populated :class:`EnvironmentState`. The ego state is
            created at ``timestamp=0.0`` with zero acceleration and zero
            steering angle because decision snapshots carry pose only.
        """
        ego_raw = d["ego_state"]
        ego = VehicleState(
            timestamp=0.0,
            x=float(ego_raw["x"]),
            y=float(ego_raw["y"]),
            yaw=float(ego_raw["yaw"]),
            speed=float(ego_raw["speed"]),
            acceleration=0.0,
            steering_angle=0.0,
        )
        actors = [
            Actor(
                id=str(raw["id"]),
                type=str(raw["type"]),
                x=float(raw["x"]),
                y=float(raw["y"]),
                speed=float(raw["speed"]),
                heading=float(raw["heading"]),
            )
            for raw in d.get("actors", [])
        ]
        features = [
            RoadFeature(
                type=str(raw["type"]),
                distance_m=float(raw["distance_m"]),
                state=str(raw["state"]),
            )
            for raw in d.get("road_features", [])
        ]
        return cls(ego=ego, actors=actors, features=features)

    def distance_to(self, actor: Actor) -> float:
        """Euclidean distance between the ego and an actor.

        Args:
            actor: Actor to measure against.

        Returns:
            Euclidean distance in the planning frame [m].
        """
        return float(math.hypot(actor.x - self.ego.x, actor.y - self.ego.y))

    def time_to_collision(self, lead: Actor) -> float | None:
        """Time-to-collision with a lead actor under constant speeds.

        Args:
            lead: Lead actor assumed to keep its current speed [m/s].

        Returns:
            Time until the ego reaches the lead actor's current position
            under constant closing speed, in seconds [s]; ``None`` when
            the closing speed is non-positive (the ego is not catching
            up), in which case no collision can occur at constant speed.

        Extension point: replace the constant-speed assumption with a
        predicted-trajectory formulation once actor predictions are
        available.
        """
        closing_speed_mps = self.ego.speed - lead.speed
        if closing_speed_mps <= 0.0:
            return None
        return float(self.distance_to(lead) / closing_speed_mps)
