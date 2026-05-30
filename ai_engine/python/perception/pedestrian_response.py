"""
Pedestrian Response Module for Autonomous Vehicle Behavior Planning.

Handles pedestrian interaction including trajectory prediction, yielding
decisions, crosswalk detection, school zone handling, crowd behavior
analysis, and minimum safe distance computation.

References:
  - Rasouli & Tsotsos, "Autonomous Vehicles that Interact with
    Pedestrians: A Survey of Theory and Practice", IEEE T-ITS 2019
  - Alahi et al., "Social LSTM: Human Trajectory Prediction in
    Crowded Spaces", CVPR 2016
  - ISO 21448 (SOTIF) — Safety of the Intended Functionality
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PedestrianAction(Enum):
    """Predicted pedestrian actions."""
    WALKING = "walking"
    STANDING = "standing"
    CROSSING = "crossing"
    WAITING = "waiting"
    JAYWALKING = "jaywalking"
    RUNNING = "running"
    ENTERING_ROAD = "entering_road"
    EXITING_ROAD = "exiting_road"
    UNKNOWN = "unknown"


class PedestrianRiskLevel(Enum):
    """Risk level associated with a pedestrian."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class YieldDecision(Enum):
    """Yielding decision for pedestrian interaction."""
    YIELD_FULL_STOP = "yield_full_stop"
    YIELD_SLOW_DOWN = "yield_slow_down"
    CREEP = "creep"
    PROCEED = "proceed"
    EMERGENCY_STOP = "emergency_stop"


class CrowdFormation(Enum):
    """Types of crowd formations."""
    SCATTERED = "scattered"          # Individuals spread apart
    GROUP = "group"                  # Small cluster (2-5 people)
    CROWD = "crowd"                  # Large gathering (6+ people)
    QUEUE = "queue"                  # Line formation
    FLOW = "flow"                    # Unidirectional flow
    BIDIRECTIONAL = "bidirectional"  # Two-way pedestrian traffic


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PedestrianConfig:
    """Configuration for pedestrian response."""
    # Minimum safe distances
    min_lateral_distance: float = 1.5       # m lateral clearance from pedestrian
    min_longitudinal_distance: float = 3.0  # m following distance behind pedestrian
    min_stopping_distance: float = 4.0      # m stopping distance from pedestrian path

    # Time parameters
    min_time_gap: float = 2.0               # s minimum time gap to pedestrian
    ttc_critical: float = 2.0               # s TTC threshold for critical risk
    ttc_warning: float = 4.0               # s TTC threshold for high risk

    # Speed thresholds
    creep_speed: float = 2.0                # m/s creeping speed near pedestrians
    school_zone_speed: float = 8.33         # m/s (30 km/h) school zone speed
    crosswalk_approach_speed: float = 10.0  # m/s max approach speed to crosswalk

    # Prediction
    prediction_horizon: float = 3.0         # s prediction horizon
    prediction_dt: float = 0.1              # s prediction time step
    max_prediction_modes: int = 3           # number of trajectory modes

    # Crosswalk
    crosswalk_detection_range: float = 30.0  # m detection range for crosswalks
    crosswalk_yield_distance: float = 20.0   # m start yielding before crosswalk
    crosswalk_clear_wait_time: float = 2.0   # s wait after crosswalk clears

    # School zone
    school_zone_detection_range: float = 50.0  # m detection range
    school_zone_speed_limit: float = 8.33      # m/s (30 km/h)
    school_zone_extra_margin: float = 1.0       # m extra distance margin

    # Crowd
    crowd_density_threshold: float = 0.3  # pedestrians/m^2 for "crowd" classification
    crowd_extra_margin: float = 2.0       # m extra margin for crowds


@dataclass
class PedestrianInfo:
    """Information about a detected pedestrian."""
    pedestrian_id: str
    x: float = 0.0                          # position [m]
    y: float = 0.0
    heading: float = 0.0                    # heading [rad]
    speed: float = 0.0                      # speed [m/s]
    acceleration: float = 0.0               # acceleration [m/s^2]
    distance_to_ego: float = float("inf")   # distance to ego [m]
    distance_to_crosswalk: float = float("inf")  # distance to nearest crosswalk [m]
    is_on_crosswalk: bool = False
    is_on_sidewalk: bool = True
    is_in_road: bool = False
    is_facing_ego: bool = False
    predicted_action: PedestrianAction = PedestrianAction.UNKNOWN
    age_group: str = "adult"                # "child", "adult", "elderly"
    has_intent_to_cross: bool = False
    confidence: float = 0.5


@dataclass
class TrajectoryPrediction:
    """Predicted trajectory for a pedestrian."""
    pedestrian_id: str
    positions: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    probability: float = 1.0
    predicted_action: PedestrianAction = PedestrianAction.UNKNOWN
    lateral_distance_at_closest: float = float("inf")
    ttc: float = float("inf")


@dataclass
class PedestrianResponse:
    """Output of pedestrian response planning."""
    yield_decision: YieldDecision = YieldDecision.PROCEED
    target_speed: float = 0.0
    safe_distance: float = 5.0
    risk_level: PedestrianRiskLevel = PedestrianRiskLevel.NONE
    critical_pedestrian_id: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0
    predictions: List[TrajectoryPrediction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pedestrian Trajectory Predictor
# ---------------------------------------------------------------------------

class PedestrianTrajectoryPredictor:
    """
    Predicts future pedestrian trajectories using a constant-velocity
    model with social force adjustments.

    For each pedestrian, generates multiple trajectory modes representing
    different possible intentions (e.g., continue walking, stop, cross).

    Simplified from Social LSTM / Social GAN for real-time operation.
    """

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config or PedestrianConfig()

    def predict(
        self,
        pedestrian: PedestrianInfo,
        other_pedestrians: Optional[List[PedestrianInfo]] = None,
    ) -> List[TrajectoryPrediction]:
        """
        Predict future trajectories for a pedestrian.

        Args:
            pedestrian: The pedestrian to predict.
            other_pedestrians: Nearby pedestrians for social force.

        Returns:
            List of TrajectoryPrediction objects (multiple modes).
        """
        other_pedestrians = other_pedestrians or []
        predictions: List[TrajectoryPrediction] = []

        # Mode 1: Continue current motion
        pred_continue = self._predict_constant_velocity(pedestrian)
        predictions.append(pred_continue)

        # Mode 2: Pedestrian stops
        pred_stop = self._predict_stop(pedestrian)
        predictions.append(pred_stop)

        # Mode 3: Pedestrian changes direction (crosses road)
        if pedestrian.has_intent_to_cross or pedestrian.is_facing_ego:
            pred_cross = self._predict_crossing(pedestrian)
            predictions.append(pred_cross)

        # Sort by probability (highest first)
        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions[:self.config.max_prediction_modes]

    def _predict_constant_velocity(
        self, pedestrian: PedestrianInfo,
    ) -> TrajectoryPrediction:
        """Predict trajectory assuming constant velocity."""
        positions: List[Tuple[float, float]] = []
        timestamps: List[float] = []
        dt = self.config.prediction_dt
        x, y = pedestrian.x, pedestrian.y
        vx = pedestrian.speed * math.cos(pedestrian.heading)
        vy = pedestrian.speed * math.sin(pedestrian.heading)

        t = 0.0
        for _ in range(int(self.config.prediction_horizon / dt)):
            t += dt
            x += vx * dt
            y += vy * dt
            positions.append((x, y))
            timestamps.append(t)

        # Probability based on consistency of motion
        prob = 0.6 if pedestrian.speed > 0.3 else 0.3
        action = PedestrianAction.WALKING if pedestrian.speed > 0.3 else PedestrianAction.STANDING

        return TrajectoryPrediction(
            pedestrian_id=pedestrian.pedestrian_id,
            positions=positions,
            timestamps=timestamps,
            probability=prob,
            predicted_action=action,
        )

    def _predict_stop(
        self, pedestrian: PedestrianInfo,
    ) -> TrajectoryPrediction:
        """Predict trajectory assuming pedestrian stops."""
        positions: List[Tuple[float, float]] = []
        timestamps: List[float] = []
        dt = self.config.prediction_dt
        x, y = pedestrian.x, pedestrian.y

        # Decelerate to stop
        v = pedestrian.speed
        t = 0.0
        for _ in range(int(self.config.prediction_horizon / dt)):
            t += dt
            v = max(0.0, v - 1.5 * dt)  # decelerate at ~1.5 m/s^2
            x += v * math.cos(pedestrian.heading) * dt
            y += v * math.sin(pedestrian.heading) * dt
            positions.append((x, y))
            timestamps.append(t)

        prob = 0.25 if pedestrian.speed > 0.5 else 0.5

        return TrajectoryPrediction(
            pedestrian_id=pedestrian.pedestrian_id,
            positions=positions,
            timestamps=timestamps,
            probability=prob,
            predicted_action=PedestrianAction.WAITING,
        )

    def _predict_crossing(
        self, pedestrian: PedestrianInfo,
    ) -> TrajectoryPrediction:
        """Predict trajectory assuming pedestrian crosses the road."""
        positions: List[Tuple[float, float]] = []
        timestamps: List[float] = []
        dt = self.config.prediction_dt
        x, y = pedestrian.x, pedestrian.y

        # Crossing speed: typically 1.2-1.5 m/s
        crossing_speed = 1.4 if pedestrian.age_group != "elderly" else 1.0
        if pedestrian.predicted_action == PedestrianAction.RUNNING:
            crossing_speed = 3.0

        # Direction: perpendicular to road (towards road center)
        crossing_heading = pedestrian.heading
        if pedestrian.is_on_sidewalk and pedestrian.is_facing_ego:
            # Turn towards road
            crossing_heading = pedestrian.heading + math.pi / 4

        vx = crossing_speed * math.cos(crossing_heading)
        vy = crossing_speed * math.sin(crossing_heading)

        t = 0.0
        for _ in range(int(self.config.prediction_horizon / dt)):
            t += dt
            x += vx * dt
            y += vy * dt
            positions.append((x, y))
            timestamps.append(t)

        prob = 0.15 if not pedestrian.has_intent_to_cross else 0.4

        return TrajectoryPrediction(
            pedestrian_id=pedestrian.pedestrian_id,
            positions=positions,
            timestamps=timestamps,
            probability=prob,
            predicted_action=PedestrianAction.CROSSING,
        )


# ---------------------------------------------------------------------------
# Safe Distance Computer
# ---------------------------------------------------------------------------

class SafeDistanceComputer:
    """
    Computes minimum safe distance to pedestrians based on
    relative kinematics, pedestrian behavior, and environmental factors.

    The safe distance model accounts for:
      - Ego braking capability
      - Pedestrian unpredictability factor
      - Age group (children and elderly get extra margin)
      - Speed differential
    """

    # Unpredictability factors by age group
    UNPREDICTABILITY_FACTOR: Dict[str, float] = {
        "child": 2.0,
        "adult": 1.0,
        "elderly": 1.5,
        "unknown": 1.5,
    }

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config or PedestrianConfig()

    def compute(
        self,
        ego_speed: float,
        pedestrian: PedestrianInfo,
        road_friction: float = 0.8,
    ) -> float:
        """
        Compute minimum safe distance to a pedestrian.

        Args:
            ego_speed: Ego vehicle speed [m/s].
            pedestrian: Pedestrian information.
            road_friction: Road friction coefficient.

        Returns:
            Minimum safe distance [m].
        """
        # Base braking distance
        max_decel = 6.0 * road_friction  # m/s^2
        braking_distance = ego_speed ** 2 / (2 * max_decel) if max_decel > 0 else 0.0

        # Reaction distance (1.0 s reaction time)
        reaction_distance = ego_speed * 1.0

        # Unpredictability margin
        unpredictability = self.UNPREDICTABILITY_FACTOR.get(
            pedestrian.age_group, 1.5
        )

        # Speed-based margin (higher speed = more margin)
        speed_margin = ego_speed * 0.2 * unpredictability

        # Direction margin (pedestrian facing ego is more dangerous)
        direction_margin = 0.0
        if pedestrian.is_facing_ego:
            direction_margin = 2.0 * unpredictability

        # Motion margin (moving pedestrians are less predictable)
        motion_margin = 0.0
        if pedestrian.speed > 0.5:
            motion_margin = pedestrian.speed * 0.5 * unpredictability

        # Base minimum distance
        min_distance = self.config.min_stopping_distance

        safe_distance = (
            braking_distance
            + reaction_distance
            + speed_margin
            + direction_margin
            + motion_margin
            + min_distance
        )

        return max(safe_distance, self.config.min_lateral_distance)

    def compute_lateral_safe_distance(
        self,
        ego_speed: float,
        pedestrian: PedestrianInfo,
    ) -> float:
        """
        Compute minimum lateral safe distance when passing a pedestrian.

        Args:
            ego_speed: Ego vehicle speed [m/s].
            pedestrian: Pedestrian information.

        Returns:
            Minimum lateral safe distance [m].
        """
        base = self.config.min_lateral_distance
        unpredictability = self.UNPREDICTABILITY_FACTOR.get(pedestrian.age_group, 1.5)
        speed_factor = 1.0 + ego_speed * 0.03
        return base * unpredictability * speed_factor


# ---------------------------------------------------------------------------
# Crosswalk Detector
# ---------------------------------------------------------------------------

class CrosswalkDetector:
    """
    Detects crosswalks and manages yielding behavior.

    Handles:
      - Crosswalk detection from map data
      - Pedestrian presence on crosswalk
      - Yielding decision at crosswalks
      - Wait time after crosswalk clears
    """

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config or PedestrianConfig()
        self._crosswalks: List[Dict] = []
        self._clear_since: Optional[float] = None

    def add_crosswalk(
        self,
        crosswalk_id: str,
        position_s: float,
        width: float = 4.0,
        has_signal: bool = False,
    ) -> None:
        """Register a crosswalk."""
        self._crosswalks.append({
            "id": crosswalk_id,
            "position_s": position_s,
            "width": width,
            "has_signal": has_signal,
        })

    def detect_nearest(
        self,
        ego_position_s: float,
    ) -> Optional[Dict]:
        """
        Find the nearest crosswalk ahead of ego.

        Args:
            ego_position_s: Ego arc-length position [m].

        Returns:
            Crosswalk dict or None if none within detection range.
        """
        nearest = None
        min_dist = float("inf")

        for cw in self._crosswalks:
            dist = cw["position_s"] - ego_position_s
            if 0 < dist < min_dist and dist < self.config.crosswalk_detection_range:
                min_dist = dist
                nearest = {**cw, "distance": dist}

        return nearest

    def check_pedestrians_on_crosswalk(
        self,
        pedestrians: List[PedestrianInfo],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if any pedestrians are on the crosswalk.

        Args:
            pedestrians: List of detected pedestrians.

        Returns:
            Tuple of (pedestrians_present, nearest_ped_id).
        """
        for ped in pedestrians:
            if ped.is_on_crosswalk or (ped.is_in_road and ped.has_intent_to_cross):
                return True, ped.pedestrian_id
        return False, None

    def should_yield(
        self,
        ego_speed: float,
        ego_position_s: float,
        pedestrians: List[PedestrianInfo],
        current_time: float,
    ) -> Tuple[bool, YieldDecision, float]:
        """
        Determine if ego should yield at a crosswalk.

        Args:
            ego_speed: Current ego speed [m/s].
            ego_position_s: Ego position [m].
            pedestrians: Detected pedestrians.
            current_time: Current timestamp [s].

        Returns:
            Tuple of (should_yield, yield_decision, target_speed).
        """
        nearest_cw = self.detect_nearest(ego_position_s)
        if nearest_cw is None:
            return False, YieldDecision.PROCEED, ego_speed

        peds_present, ped_id = self.check_pedestrians_on_crosswalk(pedestrians)

        if not peds_present:
            # Check if any pedestrian is about to enter
            for ped in pedestrians:
                if (ped.distance_to_crosswalk < 3.0
                        and ped.is_facing_ego
                        and ped.has_intent_to_cross):
                    peds_present = True
                    break

        if not peds_present:
            # Wait after clearing
            if self._clear_since is not None:
                elapsed = current_time - self._clear_since
                if elapsed < self.config.crosswalk_clear_wait_time:
                    return True, YieldDecision.CREEP, self.config.creep_speed
                self._clear_since = None
            return False, YieldDecision.PROCEED, ego_speed

        # Pedestrians present: must yield
        self._clear_since = None

        cw_distance = nearest_cw["distance"]
        if cw_distance < 10.0:
            return True, YieldDecision.YIELD_FULL_STOP, 0.0
        elif cw_distance < self.config.crosswalk_yield_distance:
            return True, YieldDecision.YIELD_SLOW_DOWN, self.config.creep_speed
        else:
            return True, YieldDecision.YIELD_SLOW_DOWN, min(ego_speed, self.config.crosswalk_approach_speed)


# ---------------------------------------------------------------------------
# School Zone Handler
# ---------------------------------------------------------------------------

class SchoolZoneHandler:
    """
    Handles school zone speed limits and heightened pedestrian awareness.

    School zones require:
      - Reduced speed limit (typically 30 km/h)
      - Extra margin for pedestrian safe distance
      - Particular attention to children who may dart into the road
    """

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config or PedestrianConfig()
        self._active_zones: List[Dict] = []

    def add_zone(
        self,
        zone_id: str,
        start_s: float,
        end_s: float,
        speed_limit: float = 8.33,
        active_hours: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Register a school zone."""
        self._active_zones.append({
            "id": zone_id,
            "start_s": start_s,
            "end_s": end_s,
            "speed_limit": speed_limit,
            "active_hours": active_hours,  # (start_hour, end_hour) 24h format
        })

    def check_in_zone(
        self,
        ego_position_s: float,
        current_hour: int = 12,
    ) -> Tuple[bool, float, Dict]:
        """
        Check if ego is in a school zone.

        Args:
            ego_position_s: Ego arc-length position [m].
            current_hour: Current hour (0-23).

        Returns:
            Tuple of (in_zone, speed_limit, zone_info).
        """
        for zone in self._active_zones:
            if zone["start_s"] <= ego_position_s <= zone["end_s"]:
                # Check active hours
                if zone["active_hours"] is not None:
                    start_h, end_h = zone["active_hours"]
                    if not (start_h <= current_hour <= end_h):
                        continue
                return True, zone["speed_limit"], zone

        return False, float("inf"), {}

    def adjust_response(
        self,
        response: PedestrianResponse,
        in_school_zone: bool,
        pedestrians: List[PedestrianInfo],
    ) -> PedestrianResponse:
        """
        Adjust pedestrian response for school zone.

        Args:
            response: Current pedestrian response.
            in_school_zone: Whether ego is in a school zone.
            pedestrians: Detected pedestrians.

        Returns:
            Adjusted PedestrianResponse.
        """
        if not in_school_zone:
            return response

        # Enforce school zone speed limit
        response.target_speed = min(response.target_speed, self.config.school_zone_speed)

        # Increase safe distance
        response.safe_distance += self.config.school_zone_extra_margin

        # Check for children
        has_children = any(p.age_group == "child" for p in pedestrians)
        if has_children:
            response.safe_distance += 1.5  # extra margin for children
            if response.risk_level == PedestrianRiskLevel.LOW:
                response.risk_level = PedestrianRiskLevel.MEDIUM
            response.reason += " | children_detected_extra_margin"

        # Any pedestrian in school zone is higher risk
        if pedestrians and response.risk_level == PedestrianRiskLevel.NONE:
            response.risk_level = PedestrianRiskLevel.LOW

        return response


# ---------------------------------------------------------------------------
# Crowd Behavior Analyzer
# ---------------------------------------------------------------------------

class CrowdBehaviorAnalyzer:
    """
    Analyzes crowd behavior near the road.

    Classifies crowd formations and predicts collective behavior
    to inform yielding and speed decisions.
    """

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config or PedestrianConfig()

    def analyze(
        self,
        pedestrians: List[PedestrianInfo],
        area_dimensions: Tuple[float, float] = (20.0, 10.0),
    ) -> Tuple[CrowdFormation, float]:
        """
        Analyze crowd formation and density.

        Args:
            pedestrians: List of detected pedestrians.
            area_dimensions: (length, width) of observation area [m].

        Returns:
            Tuple of (CrowdFormation, density in pedestrians/m^2).
        """
        if not pedestrians:
            return CrowdFormation.SCATTERED, 0.0

        area = area_dimensions[0] * area_dimensions[1]
        density = len(pedestrians) / max(area, 1.0)

        if len(pedestrians) < 2:
            return CrowdFormation.SCATTERED, density

        # Compute centroid
        cx = np.mean([p.x for p in pedestrians])
        cy = np.mean([p.y for p in pedestrians])

        # Compute spread
        distances = [math.hypot(p.x - cx, p.y - cy) for p in pedestrians]
        avg_spread = np.mean(distances)

        # Compute heading consistency
        headings = [p.heading for p in pedestrians]
        heading_cos = np.mean([math.cos(h) for h in headings])
        heading_sin = np.mean([math.sin(h) for h in headings])
        heading_consistency = math.hypot(heading_cos, heading_sin)

        # Classify
        if len(pedestrians) >= 6:
            if heading_consistency > 0.7:
                formation = CrowdFormation.FLOW
            elif avg_spread < 3.0:
                formation = CrowdFormation.CROWD
            else:
                formation = CrowdFormation.SCATTERED
        elif len(pedestrians) >= 2:
            if avg_spread < 2.0:
                formation = CrowdFormation.GROUP
            elif heading_consistency > 0.7:
                # Check for bidirectional flow
                heading_diffs = [abs(h - headings[0]) for h in headings]
                if any(d > math.pi / 2 for d in heading_diffs):
                    formation = CrowdFormation.BIDIRECTIONAL
                else:
                    formation = CrowdFormation.FLOW
            else:
                formation = CrowdFormation.SCATTERED
        else:
            formation = CrowdFormation.SCATTERED

        return formation, density

    def compute_crowd_risk(
        self,
        formation: CrowdFormation,
        density: float,
        pedestrians: List[PedestrianInfo],
        ego_speed: float,
    ) -> PedestrianRiskLevel:
        """
        Compute risk level from crowd behavior.

        Args:
            formation: Detected crowd formation.
            density: Pedestrian density [ped/m^2].
            pedestrians: List of pedestrians.
            ego_speed: Ego speed [m/s].

        Returns:
            PedestrianRiskLevel.
        """
        if not pedestrians:
            return PedestrianRiskLevel.NONE

        # Base risk from density
        if density > self.config.crowd_density_threshold:
            risk_score = 3.0
        elif density > 0.1:
            risk_score = 1.5
        else:
            risk_score = 0.5

        # Formation risk
        formation_risk = {
            CrowdFormation.SCATTERED: 0.5,
            CrowdFormation.GROUP: 1.0,
            CrowdFormation.CROWD: 2.0,
            CrowdFormation.QUEUE: 0.8,
            CrowdFormation.FLOW: 1.5,
            CrowdFormation.BIDIRECTIONAL: 2.0,
        }
        risk_score += formation_risk.get(formation, 1.0)

        # Intent risk
        crossing_count = sum(1 for p in pedestrians if p.has_intent_to_cross or p.is_in_road)
        risk_score += crossing_count * 1.0

        # Speed risk
        if ego_speed > 10.0:
            risk_score += 1.0

        if risk_score >= 5.0:
            return PedestrianRiskLevel.CRITICAL
        elif risk_score >= 3.5:
            return PedestrianRiskLevel.HIGH
        elif risk_score >= 2.0:
            return PedestrianRiskLevel.MEDIUM
        else:
            return PedestrianRiskLevel.LOW


# ---------------------------------------------------------------------------
# Main Pedestrian Response Planner
# ---------------------------------------------------------------------------

class PedestrianResponsePlanner:
    """
    Main pedestrian response planner integrating all sub-modules.

    Usage:
        planner = PedestrianResponsePlanner()
        response = planner.plan(ego_speed, pedestrians, ego_s, ...)
        print(response.yield_decision, response.target_speed)
    """

    def __init__(self, config: Optional[PedestrianConfig] = None):
        self.config = config or PedestrianConfig()
        self.predictor = PedestrianTrajectoryPredictor(config)
        self.safe_distance = SafeDistanceComputer(config)
        self.crosswalk = CrosswalkDetector(config)
        self.school_zone = SchoolZoneHandler(config)
        self.crowd_analyzer = CrowdBehaviorAnalyzer(config)

    def plan(
        self,
        ego_speed: float,
        ego_position: Tuple[float, float],
        ego_heading: float,
        pedestrians: List[PedestrianInfo],
        ego_s: float = 0.0,
        current_time: float = 0.0,
        current_hour: int = 12,
        road_friction: float = 0.8,
    ) -> PedestrianResponse:
        """
        Plan pedestrian response.

        Args:
            ego_speed: Current ego speed [m/s].
            ego_position: Ego (x, y) position.
            ego_heading: Ego heading [rad].
            pedestrians: List of detected pedestrians.
            ego_s: Ego arc-length position [m].
            current_time: Current timestamp [s].
            current_hour: Current hour (0-23).
            road_friction: Road friction coefficient.

        Returns:
            PedestrianResponse with yielding decision.
        """
        if not pedestrians:
            return PedestrianResponse(
                yield_decision=YieldDecision.PROCEED,
                target_speed=ego_speed,
                risk_level=PedestrianRiskLevel.NONE,
                reason="no_pedestrians_detected",
            )

        # Step 1: Predict trajectories
        all_predictions: List[TrajectoryPrediction] = []
        for ped in pedestrians:
            preds = self.predictor.predict(ped, pedestrians)
            all_predictions.extend(preds)

        # Step 2: Compute TTC and risk for each pedestrian
        risk_level = PedestrianRiskLevel.NONE
        critical_ped: Optional[str] = None
        min_safe_dist = float("inf")

        for ped in pedestrians:
            ttc = self._compute_ttc(ego_speed, ego_position, ego_heading, ped)
            ped_safe_dist = self.safe_distance.compute(ego_speed, ped, road_friction)

            if ped_safe_dist < min_safe_dist:
                min_safe_dist = ped_safe_dist

            ped_risk = self._assess_risk(ttc, ped)
            if ped_risk.value in ("high", "critical") and (
                risk_level == PedestrianRiskLevel.NONE
                or ped_risk.value > risk_level.value
            ):
                risk_level = ped_risk
                critical_ped = ped.pedestrian_id

        # Step 3: Crosswalk check
        should_yield_cw, cw_decision, cw_speed = self.crosswalk.should_yield(
            ego_speed, ego_s, pedestrians, current_time,
        )

        # Step 4: School zone check
        in_school, school_limit, _ = self.school_zone.check_in_zone(ego_s, current_hour)

        # Step 5: Crowd analysis
        formation, density = self.crowd_analyzer.analyze(pedestrians)
        crowd_risk = self.crowd_analyzer.compute_crowd_risk(
            formation, density, pedestrians, ego_speed,
        )

        # Take the highest risk level
        risk_levels = [risk_level, crowd_risk]
        if in_school and pedestrians:
            risk_levels.append(PedestrianRiskLevel.MEDIUM)
        risk_level = max(risk_levels, key=lambda r: list(PedestrianRiskLevel).index(r))

        # Step 6: Make yield decision
        response = self._make_decision(
            ego_speed, risk_level, critical_ped, min_safe_dist,
            should_yield_cw, cw_decision, cw_speed,
            in_school, school_limit,
        )

        # Step 7: Attach predictions
        response.predictions = all_predictions

        # Step 8: School zone adjustment
        if in_school:
            response = self.school_zone.adjust_response(response, True, pedestrians)

        return response

    def _compute_ttc(
        self,
        ego_speed: float,
        ego_pos: Tuple[float, float],
        ego_heading: float,
        pedestrian: PedestrianInfo,
    ) -> float:
        """Compute time-to-collision with a pedestrian."""
        dx = pedestrian.x - ego_pos[0]
        dy = pedestrian.y - ego_pos[1]
        dist = math.hypot(dx, dy)

        if dist < 0.1:
            return 0.0

        # Relative velocity
        ego_vx = ego_speed * math.cos(ego_heading)
        ego_vy = ego_speed * math.sin(ego_heading)
        ped_vx = pedestrian.speed * math.cos(pedestrian.heading)
        ped_vy = pedestrian.speed * math.sin(pedestrian.heading)

        rel_vx = ped_vx - ego_vx
        rel_vy = ped_vy - ego_vy

        # Closing speed
        closing_speed = -(dx * rel_vx + dy * rel_vy) / dist

        if closing_speed <= 0:
            return float("inf")

        return dist / closing_speed

    def _assess_risk(
        self,
        ttc: float,
        pedestrian: PedestrianInfo,
    ) -> PedestrianRiskLevel:
        """Assess risk level from a single pedestrian."""
        if ttc < self.config.ttc_critical:
            return PedestrianRiskLevel.CRITICAL
        if ttc < self.config.ttc_warning:
            return PedestrianRiskLevel.HIGH
        if pedestrian.is_in_road or pedestrian.has_intent_to_cross:
            return PedestrianRiskLevel.MEDIUM
        if pedestrian.is_facing_ego and pedestrian.distance_to_ego < 15.0:
            return PedestrianRiskLevel.LOW
        return PedestrianRiskLevel.NONE

    def _make_decision(
        self,
        ego_speed: float,
        risk_level: PedestrianRiskLevel,
        critical_ped_id: Optional[str],
        safe_distance: float,
        should_yield_cw: bool,
        cw_decision: YieldDecision,
        cw_speed: float,
        in_school: bool,
        school_limit: float,
    ) -> PedestrianResponse:
        """Make the final yield/proceed decision."""
        response = PedestrianResponse(
            risk_level=risk_level,
            critical_pedestrian_id=critical_ped_id,
            safe_distance=safe_distance,
        )

        # Emergency
        if risk_level == PedestrianRiskLevel.CRITICAL:
            response.yield_decision = YieldDecision.EMERGENCY_STOP
            response.target_speed = 0.0
            response.reason = "critical_pedestrian_risk_emergency_stop"
            response.confidence = 0.9
            return response

        # High risk: full yield
        if risk_level == PedestrianRiskLevel.HIGH:
            response.yield_decision = YieldDecision.YIELD_FULL_STOP
            response.target_speed = 0.0
            response.reason = "high_pedestrian_risk_yield"
            response.confidence = 0.8
            return response

        # Crosswalk yield
        if should_yield_cw:
            response.yield_decision = cw_decision
            response.target_speed = cw_speed
            response.reason = "yielding_at_crosswalk"
            response.confidence = 0.85
            return response

        # Medium risk: slow down
        if risk_level == PedestrianRiskLevel.MEDIUM:
            response.yield_decision = YieldDecision.YIELD_SLOW_DOWN
            response.target_speed = min(ego_speed * 0.5, self.config.creep_speed + 2.0)
            response.reason = "medium_pedestrian_risk_slow_down"
            response.confidence = 0.7
            return response

        # Low risk: creep
        if risk_level == PedestrianRiskLevel.LOW:
            response.yield_decision = YieldDecision.CREEP
            response.target_speed = min(ego_speed, self.config.creep_speed + 3.0)
            response.reason = "low_pedestrian_risk_cautious_approach"
            response.confidence = 0.6
            return response

        # No risk: proceed
        response.yield_decision = YieldDecision.PROCEED
        response.target_speed = ego_speed
        response.reason = "no_pedestrian_risk_proceed"
        response.confidence = 0.9

        if in_school:
            response.target_speed = min(response.target_speed, school_limit)

        return response


# ---------------------------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = PedestrianConfig()
    planner = PedestrianResponsePlanner(config)

    # Scenario 1: Pedestrian on crosswalk
    peds1 = [PedestrianInfo(
        pedestrian_id="ped_01",
        x=15.0, y=1.5,
        speed=1.2, heading=math.pi / 2,
        distance_to_ego=15.0,
        is_on_crosswalk=True,
        is_in_road=True,
        is_facing_ego=True,
        has_intent_to_cross=True,
        predicted_action=PedestrianAction.CROSSING,
    )]
    resp1 = planner.plan(10.0, (0.0, 0.0), 0.0, peds1, ego_s=100.0)
    print(f"Scenario 1 (ped on crosswalk): decision={resp1.yield_decision.value}, "
          f"speed={resp1.target_speed:.1f}, risk={resp1.risk_level.value}")

    # Scenario 2: Child near road in school zone
    planner.school_zone.add_zone("sz_01", 80.0, 120.0)
    peds2 = [PedestrianInfo(
        pedestrian_id="child_01",
        x=8.0, y=3.0,
        speed=0.5, heading=0.0,
        distance_to_ego=8.0,
        is_on_sidewalk=True,
        is_facing_ego=True,
        age_group="child",
        has_intent_to_cross=True,
    )]
    resp2 = planner.plan(8.0, (0.0, 0.0), 0.0, peds2, ego_s=90.0, current_hour=8)
    print(f"Scenario 2 (child in school zone): decision={resp2.yield_decision.value}, "
          f"speed={resp2.target_speed:.1f}, safe_dist={resp2.safe_distance:.1f}m")

    # Scenario 3: No pedestrians
    resp3 = planner.plan(12.0, (0.0, 0.0), 0.0, [], ego_s=50.0)
    print(f"Scenario 3 (no peds): decision={resp3.yield_decision.value}, "
          f"speed={resp3.target_speed:.1f}")

    # Safe distance computation
    sdc = SafeDistanceComputer(config)
    dist = sdc.compute(15.0, PedestrianInfo(
        pedestrian_id="test", age_group="child", is_facing_ego=True, speed=1.5,
        distance_to_ego=10.0,
    ))
    print(f"\nSafe distance (child, facing ego, 15 m/s): {dist:.1f}m")
