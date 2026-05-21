"""
Behavior Planner Module for Autonomous Vehicle Control System.

Implements a Finite State Machine (FSM) for high-level driving behavior
selection, cost-based maneuver evaluation, and state transition logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DrivingState(Enum):
    """Finite State Machine states for vehicle behavior.

    Each state corresponds to a distinct driving maneuver category.
    """
    LANE_KEEPING = auto()
    LANE_CHANGE_LEFT = auto()
    LANE_CHANGE_RIGHT = auto()
    INTERSECTION = auto()
    EMERGENCY_STOP = auto()
    PARKING = auto()


class ManeuverType(Enum):
    """Atomic maneuver types that the planner may select."""
    KEEP_LANE = auto()
    CHANGE_LEFT = auto()
    CHANGE_RIGHT = auto()
    YIELD = auto()
    STOP = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    U_TURN = auto()
    PARK = auto()
    EMERGENCY_BRAKE = auto()


class TransitionResult(Enum):
    """Outcome of a state transition evaluation."""
    TRANSITION = auto()
    REMAIN = auto()
    INVALID = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Snapshot of the ego vehicle's dynamic state.

    Attributes:
        x: Longitudinal position (m).
        y: Lateral position (m).
        heading: Heading angle (rad), 0 = east, pi/2 = north.
        speed: Forward speed (m/s).
        acceleration: Longitudinal acceleration (m/s^2).
        lane_id: Current lane identifier (0-indexed from leftmost).
    """
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    lane_id: int = 0


@dataclass
class ObstacleInfo:
    """Information about a detected obstacle / dynamic agent.

    Attributes:
        obstacle_id: Unique identifier.
        x: Position x (m).
        y: Position y (m).
        vx: Velocity x (m/s).
        vy: Velocity y (m/s).
        length: Bounding-box length (m).
        width: Bounding-box width (m).
        is_moving: Whether the obstacle is dynamic.
        lane_id: Lane the obstacle occupies.
        time_to_collision: Estimated TTC (s); inf if not on collision course.
    """
    obstacle_id: int = 0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.5
    width: float = 1.8
    is_moving: bool = False
    lane_id: int = 0
    time_to_collision: float = math.inf


@dataclass
class Maneuver:
    """A candidate maneuver with boundary conditions and metadata.

    Attributes:
        maneuver_type: The type of maneuver.
        start_state: Vehicle state at maneuver start.
        end_state: Expected vehicle state at maneuver end.
        duration: Expected duration (s).
        lateral_displacement: Net lateral displacement (m); negative = left.
        longitudinal_distance: Net longitudinal distance (m).
        target_lane_id: Lane to be in after the maneuver.
        cost: Computed cost of this maneuver (lower = better).
        is_emergency: Whether this is an emergency maneuver.
    """
    maneuver_type: ManeuverType = ManeuverType.KEEP_LANE
    start_state: VehicleState = field(default_factory=VehicleState)
    end_state: VehicleState = field(default_factory=VehicleState)
    duration: float = 1.0
    lateral_displacement: float = 0.0
    longitudinal_distance: float = 0.0
    target_lane_id: int = 0
    cost: float = math.inf
    is_emergency: bool = False


@dataclass
class RoadContext:
    """Environmental context provided to the behavior planner each tick.

    Attributes:
        num_lanes: Total number of lanes in the current road segment.
        lane_width: Width of each lane (m).
        speed_limit: Speed limit (m/s).
        is_intersection: Whether the vehicle is approaching an intersection.
        intersection_type: 'traffic_light', 'stop_sign', 'yield', or ''.
        distance_to_intersection: Distance to the stop line (m); inf if N/A.
        is_parking_zone: Whether a parking zone is nearby.
        parking_available: Whether a parking spot is detected.
        road_curvature: Curvature of the current road segment (1/m).
    """
    num_lanes: int = 3
    lane_width: float = 3.5
    speed_limit: float = 13.9
    is_intersection: bool = False
    intersection_type: str = ""
    distance_to_intersection: float = math.inf
    is_parking_zone: bool = False
    parking_available: bool = False
    road_curvature: float = 0.0


# ---------------------------------------------------------------------------
# FSM Transition Table
# ---------------------------------------------------------------------------

# Defines which target states are reachable from each source state.
_TRANSITION_TABLE: Dict[DrivingState, Set[DrivingState]] = {
    DrivingState.LANE_KEEPING: {
        DrivingState.LANE_CHANGE_LEFT,
        DrivingState.LANE_CHANGE_RIGHT,
        DrivingState.INTERSECTION,
        DrivingState.EMERGENCY_STOP,
        DrivingState.PARKING,
    },
    DrivingState.LANE_CHANGE_LEFT: {
        DrivingState.LANE_KEEPING,
        DrivingState.EMERGENCY_STOP,
    },
    DrivingState.LANE_CHANGE_RIGHT: {
        DrivingState.LANE_KEEPING,
        DrivingState.EMERGENCY_STOP,
    },
    DrivingState.INTERSECTION: {
        DrivingState.LANE_KEEPING,
        DrivingState.EMERGENCY_STOP,
    },
    DrivingState.EMERGENCY_STOP: {
        DrivingState.LANE_KEEPING,
        DrivingState.PARKING,
    },
    DrivingState.PARKING: {
        DrivingState.LANE_KEEPING,
    },
}


# ---------------------------------------------------------------------------
# BehaviorPlanner
# ---------------------------------------------------------------------------

class BehaviorPlanner:
    """Finite-State-Machine behavior planner for autonomous driving.

    Evaluates available maneuvers, selects the lowest-cost one, and
    manages state transitions. Cost-based decision making considers
    safety proximity, lane-keeping preference, traffic-rule compliance,
    and passenger comfort.

    Example::

        bp = BehaviorPlanner(num_lanes=3)
        bp.update(ego_state, obstacles, road_ctx)
        maneuver = bp.select_maneuver()
        print(bp.current_state)
    """

    def __init__(
        self,
        num_lanes: int = 3,
        lane_width: float = 3.5,
        safe_ttc_threshold: float = 3.0,
        lane_change_duration: float = 3.0,
        emergency_decel: float = 6.0,
        comfort_lateral_accel: float = 2.0,
    ) -> None:
        """Initialise the BehaviorPlanner.

        Args:
            num_lanes: Number of lanes on the current road.
            lane_width: Width of each lane in metres.
            safe_ttc_threshold: Minimum safe time-to-collision (s).
            lane_change_duration: Nominal lane-change duration (s).
            emergency_decel: Deceleration used for emergency stop (m/s^2).
            comfort_lateral_accel: Comfortable lateral acceleration (m/s^2).
        """
        self._current_state: DrivingState = DrivingState.LANE_KEEPING
        self._num_lanes = num_lanes
        self._lane_width = lane_width
        self._safe_ttc = safe_ttc_threshold
        self._lane_change_duration = lane_change_duration
        self._emergency_decel = emergency_decel
        self._comfort_lateral_accel = comfort_lateral_accel

        # Maneuver cache
        self._candidate_maneuvers: List[Maneuver] = []
        self._selected_maneuver: Optional[Maneuver] = None

        # Transition history for hysteresis
        self._state_entry_time: float = 0.0
        self._min_state_duration: float = 1.0  # avoid oscillation

        # Cost weights
        self._w_safety: float = 5.0
        self._w_efficiency: float = 2.0
        self._w_comfort: float = 1.5
        self._w_rule: float = 2.5
        self._w_lane_keep: float = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> DrivingState:
        """Current FSM state."""
        return self._current_state

    @property
    def selected_maneuver(self) -> Optional[Maneuver]:
        """Most recently selected maneuver."""
        return self._selected_maneuver

    @property
    def candidates(self) -> List[Maneuver]:
        """Candidate maneuvers from the last planning tick."""
        return list(self._candidate_maneuvers)

    # ------------------------------------------------------------------
    # Core planning loop
    # ------------------------------------------------------------------

    def update(
        self,
        ego_state: VehicleState,
        obstacles: List[ObstacleInfo],
        road_ctx: RoadContext,
        current_time: float = 0.0,
    ) -> None:
        """Run one planning tick: generate, score, and select a maneuver.

        Args:
            ego_state: Current ego vehicle state.
            obstacles: List of detected obstacles.
            road_ctx: Current road environment context.
            current_time: Elapsed time for hysteresis checks.
        """
        # 1. Check for emergency
        if self._detect_emergency(ego_state, obstacles):
            self._transition_to(DrivingState.EMERGENCY_STOP, current_time)
            self._selected_maneuver = self._build_emergency_maneuver(ego_state)
            self._candidate_maneuvers = [self._selected_maneuver]
            return

        # 2. Generate candidate maneuvers for current state
        self._candidate_maneuvers = self._generate_maneuvers(
            ego_state, obstacles, road_ctx
        )

        # 3. Score each candidate
        for m in self._candidate_maneuvers:
            m.cost = self._score_maneuver(m, ego_state, obstacles, road_ctx)

        # 4. Select lowest-cost maneuver
        if self._candidate_maneuvers:
            self._selected_maneuver = min(self._candidate_maneuvers, key=lambda m: m.cost)
        else:
            self._selected_maneuver = self._build_keep_lane_maneuver(ego_state, road_ctx)

        # 5. Determine FSM transition
        target_state = self._maneuver_to_state(self._selected_maneuver.maneuver_type)
        self._try_transition(target_state, current_time)

    def select_maneuver(self) -> Maneuver:
        """Return the currently selected maneuver (convenience method)."""
        if self._selected_maneuver is None:
            return Maneuver()
        return self._selected_maneuver

    # ------------------------------------------------------------------
    # Emergency detection
    # ------------------------------------------------------------------

    def _detect_emergency(
        self, ego: VehicleState, obstacles: List[ObstacleInfo]
    ) -> bool:
        """Check whether any obstacle violates the safe TTC threshold."""
        for obs in obstacles:
            if obs.time_to_collision < self._safe_ttc and obs.time_to_collision > 0:
                return True
            # Also check very close stationary obstacles
            dist = math.hypot(obs.x - ego.x, obs.y - ego.y)
            if dist < 3.0 and not obs.is_moving and ego.speed > 1.0:
                return True
        return False

    # ------------------------------------------------------------------
    # Maneuver generation
    # ------------------------------------------------------------------

    def _generate_maneuvers(
        self,
        ego: VehicleState,
        obstacles: List[ObstacleInfo],
        ctx: RoadContext,
    ) -> List[Maneuver]:
        """Generate all feasible maneuvers given the current state and context."""
        maneuvers: List[Maneuver] = []

        # Always consider lane keeping
        maneuvers.append(self._build_keep_lane_maneuver(ego, ctx))

        # Lane changes (if allowed by FSM and road geometry)
        if self._current_state in (
            DrivingState.LANE_KEEPING,
            DrivingState.LANE_CHANGE_LEFT,
            DrivingState.LANE_CHANGE_RIGHT,
        ):
            if ego.lane_id > 0:
                maneuvers.append(
                    self._build_lane_change_maneuver(ego, ctx, direction="left")
                )
            if ego.lane_id < ctx.num_lanes - 1:
                maneuvers.append(
                    self._build_lane_change_maneuver(ego, ctx, direction="right")
                )

        # Intersection maneuvers
        if ctx.is_intersection:
            maneuvers.append(self._build_intersection_maneuver(ego, ctx))

        # Parking
        if ctx.is_parking_zone and ctx.parking_available:
            maneuvers.append(self._build_parking_maneuver(ego, ctx))

        # Filter out maneuvers that would collide with obstacles
        maneuvers = [m for m in maneuvers if self._is_maneuver_safe(m, obstacles)]

        return maneuvers

    def _build_keep_lane_maneuver(self, ego: VehicleState, ctx: RoadContext) -> Maneuver:
        """Build a lane-keeping maneuver."""
        target_speed = min(ego.speed + 1.0, ctx.speed_limit)
        return Maneuver(
            maneuver_type=ManeuverType.KEEP_LANE,
            start_state=VehicleState(**ego.__dict__),
            end_state=VehicleState(
                x=ego.x + target_speed * 2.0,
                y=ego.y,
                heading=ego.heading,
                speed=target_speed,
                lane_id=ego.lane_id,
            ),
            duration=2.0,
            lateral_displacement=0.0,
            longitudinal_distance=target_speed * 2.0,
            target_lane_id=ego.lane_id,
        )

    def _build_lane_change_maneuver(
        self, ego: VehicleState, ctx: RoadContext, direction: str
    ) -> Maneuver:
        """Build a lane-change maneuver (left or right)."""
        lateral = ctx.lane_width if direction == "left" else -ctx.lane_width
        target_lane = ego.lane_id - 1 if direction == "left" else ego.lane_id + 1
        mtype = ManeuverType.CHANGE_LEFT if direction == "left" else ManeuverType.CHANGE_RIGHT

        return Maneuver(
            maneuver_type=mtype,
            start_state=VehicleState(**ego.__dict__),
            end_state=VehicleState(
                x=ego.x + ego.speed * self._lane_change_duration,
                y=ego.y + lateral,
                heading=ego.heading,
                speed=ego.speed,
                lane_id=target_lane,
            ),
            duration=self._lane_change_duration,
            lateral_displacement=lateral,
            longitudinal_distance=ego.speed * self._lane_change_duration,
            target_lane_id=target_lane,
        )

    def _build_intersection_maneuver(self, ego: VehicleState, ctx: RoadContext) -> Maneuver:
        """Build a maneuver for intersection traversal."""
        return Maneuver(
            maneuver_type=ManeuverType.YIELD,
            start_state=VehicleState(**ego.__dict__),
            end_state=VehicleState(
                x=ego.x + ctx.distance_to_intersection,
                y=ego.y,
                heading=ego.heading,
                speed=0.0,
                lane_id=ego.lane_id,
            ),
            duration=max(ego.speed / self._emergency_decel, 1.0) if ego.speed > 0 else 1.0,
            lateral_displacement=0.0,
            longitudinal_distance=ctx.distance_to_intersection,
            target_lane_id=ego.lane_id,
        )

    def _build_parking_maneuver(self, ego: VehicleState, ctx: RoadContext) -> Maneuver:
        """Build a parking maneuver."""
        return Maneuver(
            maneuver_type=ManeuverType.PARK,
            start_state=VehicleState(**ego.__dict__),
            end_state=VehicleState(
                x=ego.x,
                y=ego.y + ctx.lane_width,
                heading=ego.heading + math.pi / 2,
                speed=0.0,
                lane_id=ego.lane_id,
            ),
            duration=5.0,
            lateral_displacement=ctx.lane_width,
            longitudinal_distance=0.0,
            target_lane_id=ego.lane_id,
        )

    def _build_emergency_maneuver(self, ego: VehicleState) -> Maneuver:
        """Build an emergency-brake maneuver."""
        stop_distance = ego.speed ** 2 / (2.0 * self._emergency_decel)
        stop_duration = ego.speed / self._emergency_decel if self._emergency_decel > 0 else 1.0
        return Maneuver(
            maneuver_type=ManeuverType.EMERGENCY_BRAKE,
            start_state=VehicleState(**ego.__dict__),
            end_state=VehicleState(
                x=ego.x + stop_distance,
                y=ego.y,
                heading=ego.heading,
                speed=0.0,
                lane_id=ego.lane_id,
            ),
            duration=stop_duration,
            lateral_displacement=0.0,
            longitudinal_distance=stop_distance,
            target_lane_id=ego.lane_id,
            is_emergency=True,
            cost=0.0,
        )

    # ------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------

    def _is_maneuver_safe(self, maneuver: Maneuver, obstacles: List[ObstacleInfo]) -> bool:
        """Check that a maneuver does not immediately collide with obstacles."""
        for obs in obstacles:
            # Check proximity at the maneuver's midpoint
            mid_x = (maneuver.start_state.x + maneuver.end_state.x) / 2.0
            mid_y = (maneuver.start_state.y + maneuver.end_state.y) / 2.0
            dist = math.hypot(obs.x - mid_x, obs.y - mid_y)
            min_safe = (obs.length / 2.0 + 2.5)  # 2.5 m ego half-length buffer
            if dist < min_safe:
                return False
        return True

    # ------------------------------------------------------------------
    # Cost scoring
    # ------------------------------------------------------------------

    def _score_maneuver(
        self,
        maneuver: Maneuver,
        ego: VehicleState,
        obstacles: List[ObstacleInfo],
        ctx: RoadContext,
    ) -> float:
        """Compute the cost of a candidate maneuver.

        Cost = w_safety * C_safety + w_efficiency * C_efficiency
             + w_comfort * C_comfort + w_rule * C_rule
             + w_lane_keep * C_lane_keep
        """
        c_safety = self._cost_safety(maneuver, obstacles)
        c_efficiency = self._cost_efficiency(maneuver, ctx)
        c_comfort = self._cost_comfort(maneuver)
        c_rule = self._cost_rule(maneuver, ctx)
        c_lane = self._cost_lane_preference(maneuver, ego)

        return (
            self._w_safety * c_safety
            + self._w_efficiency * c_efficiency
            + self._w_comfort * c_comfort
            + self._w_rule * c_rule
            + self._w_lane_keep * c_lane
        )

    def _cost_safety(self, maneuver: Maneuver, obstacles: List[ObstacleInfo]) -> float:
        """Safety cost: inversely related to min obstacle distance during maneuver."""
        if not obstacles:
            return 0.0
        min_dist = math.inf
        mid_x = (maneuver.start_state.x + maneuver.end_state.x) / 2.0
        mid_y = (maneuver.start_state.y + maneuver.end_state.y) / 2.0
        for obs in obstacles:
            d = math.hypot(obs.x - mid_x, obs.y - mid_y)
            if d < min_dist:
                min_dist = d
        # Cost rises steeply as distance decreases
        safe_buffer = 5.0
        if min_dist >= safe_buffer:
            return 0.0
        return (1.0 - min_dist / safe_buffer) ** 2

    def _cost_efficiency(self, maneuver: Maneuver, ctx: RoadContext) -> float:
        """Efficiency cost: penalises slow speeds and long durations."""
        avg_speed = maneuver.longitudinal_distance / max(maneuver.duration, 0.1)
        speed_ratio = avg_speed / max(ctx.speed_limit, 1.0)
        return 1.0 - min(speed_ratio, 1.0)

    def _cost_comfort(self, maneuver: Maneuver) -> float:
        """Comfort cost: penalises large lateral accelerations."""
        if maneuver.duration <= 0:
            return 1.0
        lateral_accel = abs(maneuver.lateral_displacement) / (maneuver.duration ** 2)
        if lateral_accel <= self._comfort_lateral_accel:
            return lateral_accel / max(self._comfort_lateral_accel, 1e-6)
        return 1.0 + (lateral_accel - self._comfort_lateral_accel)

    def _cost_rule(self, maneuver: Maneuver, ctx: RoadContext) -> float:
        """Rule-compliance cost: penalises speeding and wrong-direction maneuvers."""
        cost = 0.0
        end_speed = maneuver.end_state.speed
        if end_speed > ctx.speed_limit:
            cost += (end_speed - ctx.speed_limit) / max(ctx.speed_limit, 1.0)
        return cost

    def _cost_lane_preference(self, maneuver: Maneuver, ego: VehicleState) -> float:
        """Lane-keeping preference: small penalty for unnecessary lane changes."""
        if maneuver.maneuver_type == ManeuverType.KEEP_LANE:
            return 0.0
        return 0.5  # mild preference to stay in lane

    # ------------------------------------------------------------------
    # FSM transition
    # ------------------------------------------------------------------

    @staticmethod
    def _maneuver_to_state(mtype: ManeuverType) -> DrivingState:
        """Map a ManeuverType to the corresponding DrivingState."""
        mapping: Dict[ManeuverType, DrivingState] = {
            ManeuverType.KEEP_LANE: DrivingState.LANE_KEEPING,
            ManeuverType.CHANGE_LEFT: DrivingState.LANE_CHANGE_LEFT,
            ManeuverType.CHANGE_RIGHT: DrivingState.LANE_CHANGE_RIGHT,
            ManeuverType.YIELD: DrivingState.INTERSECTION,
            ManeuverType.STOP: DrivingState.INTERSECTION,
            ManeuverType.TURN_LEFT: DrivingState.INTERSECTION,
            ManeuverType.TURN_RIGHT: DrivingState.INTERSECTION,
            ManeuverType.U_TURN: DrivingState.INTERSECTION,
            ManeuverType.PARK: DrivingState.PARKING,
            ManeuverType.EMERGENCY_BRAKE: DrivingState.EMERGENCY_STOP,
        }
        return mapping.get(mtype, DrivingState.LANE_KEEPING)

    def _transition_to(self, new_state: DrivingState, current_time: float) -> None:
        """Force a state transition (used for emergency)."""
        self._current_state = new_state
        self._state_entry_time = current_time

    def _try_transition(self, target: DrivingState, current_time: float) -> TransitionResult:
        """Attempt a state transition respecting the transition table and hysteresis."""
        if target == self._current_state:
            return TransitionResult.REMAIN

        reachable = _TRANSITION_TABLE.get(self._current_state, set())
        if target not in reachable:
            return TransitionResult.INVALID

        # Hysteresis: enforce minimum dwell time in the current state
        dwell = current_time - self._state_entry_time
        if dwell < self._min_state_duration and self._current_state != DrivingState.EMERGENCY_STOP:
            return TransitionResult.REMAIN

        self._current_state = target
        self._state_entry_time = current_time
        return TransitionResult.TRANSITION

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BehaviorPlanner(state={self._current_state.name}, "
            f"lanes={self._num_lanes})"
        )
