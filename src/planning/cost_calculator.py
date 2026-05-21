"""
Cost Calculator Module for Autonomous Vehicle Control System.

Provides comprehensive cost computation for path evaluation, behavior selection,
and trajectory scoring. Supports multiple cost components with configurable
weights per driving scenario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DrivingScenario(Enum):
    """Predefined driving scenarios with default weight profiles."""
    HIGHWAY = auto()
    URBAN = auto()
    RESIDENTIAL = auto()
    PARKING = auto()
    EMERGENCY = auto()


class CostComponent(Enum):
    """Individual cost components used in aggregated cost."""
    DISTANCE = "distance"
    TIME = "time"
    COMFORT = "comfort"
    SAFETY = "safety"
    RULE_COMPLIANCE = "rule_compliance"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CostWeights:
    """Weight configuration for each cost component.

    All weights should be non-negative. The aggregate cost is the weighted
    sum of component costs, optionally normalised.

    Attributes:
        distance: Weight for path length / travel distance.
        time: Weight for estimated travel time.
        comfort: Weight for passenger comfort (jerk, lateral accel).
        safety: Weight for collision proximity and risk metrics.
        rule_compliance: Weight for traffic-rule adherence score.
    """
    distance: float = 1.0
    time: float = 1.0
    comfort: float = 1.0
    safety: float = 3.0
    rule_compliance: float = 2.0

    def as_vector(self) -> NDArray[np.float64]:
        """Return weights as a numpy vector in canonical component order."""
        return np.array([
            self.distance,
            self.time,
            self.comfort,
            self.safety,
            self.rule_compliance,
        ], dtype=np.float64)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a single candidate plan.

    Attributes:
        distance_cost: Raw distance cost value.
        time_cost: Raw time cost value.
        comfort_cost: Raw comfort cost value.
        safety_cost: Raw safety cost value.
        rule_compliance_cost: Raw rule-compliance cost value.
        total_cost: Weighted aggregate cost.
        weights_used: Snapshot of weights applied during computation.
    """
    distance_cost: float = 0.0
    time_cost: float = 0.0
    comfort_cost: float = 0.0
    safety_cost: float = 0.0
    rule_compliance_cost: float = 0.0
    total_cost: float = 0.0
    weights_used: Optional[CostWeights] = None

    def component_vector(self) -> NDArray[np.float64]:
        """Return component costs as a numpy vector in canonical order."""
        return np.array([
            self.distance_cost,
            self.time_cost,
            self.comfort_cost,
            self.safety_cost,
            self.rule_compliance_cost,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Scenario weight presets
# ---------------------------------------------------------------------------

_SCENARIO_WEIGHTS: Dict[DrivingScenario, CostWeights] = {
    DrivingScenario.HIGHWAY: CostWeights(
        distance=1.0, time=1.5, comfort=1.0, safety=4.0, rule_compliance=2.5,
    ),
    DrivingScenario.URBAN: CostWeights(
        distance=1.0, time=1.0, comfort=1.5, safety=3.5, rule_compliance=3.0,
    ),
    DrivingScenario.RESIDENTIAL: CostWeights(
        distance=0.8, time=0.8, comfort=2.0, safety=4.0, rule_compliance=3.5,
    ),
    DrivingScenario.PARKING: CostWeights(
        distance=1.5, time=0.5, comfort=2.0, safety=3.0, rule_compliance=1.5,
    ),
    DrivingScenario.EMERGENCY: CostWeights(
        distance=0.5, time=2.0, comfort=0.3, safety=5.0, rule_compliance=1.0,
    ),
}


# ---------------------------------------------------------------------------
# CostCalculator
# ---------------------------------------------------------------------------

class CostCalculator:
    """Computes aggregate costs for path, behavior, and trajectory candidates.

    The calculator maintains a set of configurable weights for five cost
    components: distance, time, comfort, safety, and rule_compliance.
    Weights can be loaded from driving-scenario presets or overridden
    individually at runtime.

    Example::

        calc = CostCalculator(scenario=DrivingScenario.HIGHWAY)
        breakdown = calc.compute_path_cost(waypoints, obstacles)
        print(breakdown.total_cost)
    """

    def __init__(
        self,
        scenario: DrivingScenario = DrivingScenario.URBAN,
        custom_weights: Optional[CostWeights] = None,
        vehicle_max_speed: float = 30.0,          # m/s
        vehicle_max_accel: float = 3.0,           # m/s^2
        vehicle_max_jerk: float = 5.0,            # m/s^3
        collision_check_radius: float = 2.0,      # metres
        speed_limit: float = 13.9,                # m/s  (~50 km/h)
    ) -> None:
        """Initialise the CostCalculator.

        Args:
            scenario: Driving scenario used to select default weights.
            custom_weights: If provided, overrides the scenario weights.
            vehicle_max_speed: Maximum vehicle speed in m/s.
            vehicle_max_accel: Maximum comfortable acceleration in m/s^2.
            vehicle_max_jerk: Maximum comfortable jerk in m/s^3.
            collision_check_radius: Minimum distance to obstacles for safety cost.
            speed_limit: Reference speed limit for rule-compliance cost.
        """
        self._weights: CostWeights = (
            custom_weights if custom_weights is not None
            else _SCENARIO_WEIGHTS[scenario].__class__(
                **_SCENARIO_WEIGHTS[scenario].__dict__
            )
        )
        self._scenario = scenario
        self._vehicle_max_speed = vehicle_max_speed
        self._vehicle_max_accel = vehicle_max_accel
        self._vehicle_max_jerk = vehicle_max_jerk
        self._collision_check_radius = collision_check_radius
        self._speed_limit = speed_limit

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> CostWeights:
        """Current cost weights."""
        return self._weights

    @weights.setter
    def weights(self, value: CostWeights) -> None:
        self._weights = value

    @property
    def scenario(self) -> DrivingScenario:
        """Active driving scenario."""
        return self._scenario

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_scenario(self, scenario: DrivingScenario) -> None:
        """Switch to a scenario's preset weights.

        Args:
            scenario: The new driving scenario.
        """
        self._scenario = scenario
        preset = _SCENARIO_WEIGHTS[scenario]
        self._weights = CostWeights(
            distance=preset.distance,
            time=preset.time,
            comfort=preset.comfort,
            safety=preset.safety,
            rule_compliance=preset.rule_compliance,
        )

    def set_weight(self, component: CostComponent, value: float) -> None:
        """Override a single component weight.

        Args:
            component: The cost component to adjust.
            value: New weight (must be >= 0).
        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError(f"Weight must be non-negative, got {value}")
        setattr(self._weights, component.value, value)

    # ------------------------------------------------------------------
    # Public cost computation APIs
    # ------------------------------------------------------------------

    def compute_path_cost(
        self,
        waypoints: NDArray[np.float64],
        obstacles: Optional[NDArray[np.float64]] = None,
        desired_speed: float = 10.0,
    ) -> CostBreakdown:
        """Compute the full cost breakdown for a path.

        Args:
            waypoints: Nx2 or Nx3 array of (x, y[, z]) waypoints in metres.
            obstacles: Mx2 array of (x, y) obstacle positions. May be None.
            desired_speed: Target cruise speed in m/s.

        Returns:
            A CostBreakdown with per-component and total costs.
        """
        if waypoints.shape[0] < 2:
            return CostBreakdown(weights_used=CostWeights(**self._weights.__dict__))

        dist_cost = self._distance_cost(waypoints)
        time_cost = self._time_cost(waypoints, desired_speed)
        comfort_cost = self._comfort_cost_path(waypoints)
        safety_cost = self._safety_cost(waypoints, obstacles)
        rule_cost = self._rule_compliance_cost(waypoints, desired_speed)

        breakdown = CostBreakdown(
            distance_cost=dist_cost,
            time_cost=time_cost,
            comfort_cost=comfort_cost,
            safety_cost=safety_cost,
            rule_compliance_cost=rule_cost,
            weights_used=CostWeights(**self._weights.__dict__),
        )
        breakdown.total_cost = self._aggregate(breakdown)
        return breakdown

    def compute_behavior_cost(
        self,
        maneuver_length: float,
        maneuver_duration: float,
        lateral_acceleration: float,
        min_obstacle_distance: float,
        speed_excess: float,
    ) -> CostBreakdown:
        """Compute cost for a behavior / maneuver candidate.

        Args:
            maneuver_length: Path length of the maneuver in metres.
            maneuver_duration: Duration of the maneuver in seconds.
            lateral_acceleration: Peak lateral acceleration in m/s^2.
            min_obstacle_distance: Closest approach to any obstacle in metres.
            speed_excess: Amount over the speed limit in m/s (0 if compliant).

        Returns:
            A CostBreakdown for the behavior.
        """
        dist_cost = self._normalise(maneuver_length, 0.0, 500.0)
        time_cost = self._normalise(maneuver_duration, 0.0, 30.0)
        comfort_cost = self._comfort_cost_behavior(lateral_acceleration)
        safety_cost = self._safety_cost_scalar(min_obstacle_distance)
        rule_cost = self._rule_cost_scalar(speed_excess)

        breakdown = CostBreakdown(
            distance_cost=dist_cost,
            time_cost=time_cost,
            comfort_cost=comfort_cost,
            safety_cost=safety_cost,
            rule_compliance_cost=rule_cost,
            weights_used=CostWeights(**self._weights.__dict__),
        )
        breakdown.total_cost = self._aggregate(breakdown)
        return breakdown

    def rank_candidates(
        self,
        candidates: List[NDArray[np.float64]],
        obstacles: Optional[NDArray[np.float64]] = None,
        desired_speed: float = 10.0,
    ) -> List[Tuple[int, CostBreakdown]]:
        """Rank path candidates by ascending total cost.

        Args:
            candidates: List of waypoint arrays, one per candidate.
            obstacles: Mx2 obstacle positions.
            desired_speed: Target speed in m/s.

        Returns:
            List of (original_index, CostBreakdown) sorted by total_cost.
        """
        scored: List[Tuple[int, CostBreakdown]] = []
        for idx, wp in enumerate(candidates):
            bd = self.compute_path_cost(wp, obstacles, desired_speed)
            scored.append((idx, bd))
        scored.sort(key=lambda t: t[1].total_cost)
        return scored

    # ------------------------------------------------------------------
    # Internal cost component implementations
    # ------------------------------------------------------------------

    def _distance_cost(self, waypoints: NDArray[np.float64]) -> float:
        """Raw distance cost: normalised total path length."""
        diffs = np.diff(waypoints[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total = float(np.sum(segment_lengths))
        # Normalise to [0, 1] w.r.t. a 500 m reference
        return self._normalise(total, 0.0, 500.0)

    def _time_cost(self, waypoints: NDArray[np.float64], desired_speed: float) -> float:
        """Raw time cost: estimated travel time normalised."""
        diffs = np.diff(waypoints[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        if desired_speed <= 0:
            desired_speed = 1.0
        total_time = float(np.sum(segment_lengths) / desired_speed)
        return self._normalise(total_time, 0.0, 50.0)

    def _comfort_cost_path(self, waypoints: NDArray[np.float64]) -> float:
        """Comfort cost based on curvature / heading changes along the path.

        Approximates lateral acceleration as v^2 * curvature and penalises
        values exceeding the comfort threshold.
        """
        if waypoints.shape[0] < 3:
            return 0.0

        pts = waypoints[:, :2]
        # Vectors between consecutive points
        v1 = pts[1:-1] - pts[:-2]   # (N-2, 2)
        v2 = pts[2:] - pts[1:-1]     # (N-2, 2)

        # Signed heading change (yaw delta) via cross product
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        lengths1 = np.linalg.norm(v1, axis=1)
        lengths2 = np.linalg.norm(v2, axis=1)
        denom = lengths1 * lengths2
        denom = np.where(denom < 1e-9, 1e-9, denom)
        sin_theta = np.clip(cross / denom, -1.0, 1.0)
        heading_changes = np.arcsin(sin_theta)

        # Approximate curvature: kappa ≈ dtheta / avg_segment_length
        avg_len = (lengths1 + lengths2) / 2.0
        avg_len = np.where(avg_len < 1e-9, 1e-9, avg_len)
        curvature = np.abs(heading_changes) / avg_len

        # Assume a nominal speed of 10 m/s for lateral-accel estimate
        nominal_speed = 10.0
        lateral_accel = nominal_speed ** 2 * curvature

        # Penalise deviations beyond comfort threshold
        excess = np.maximum(lateral_accel - self._vehicle_max_accel, 0.0)
        comfort_cost = float(np.mean(excess)) / max(self._vehicle_max_accel, 1e-6)
        return min(comfort_cost, 1.0)

    def _comfort_cost_behavior(self, lateral_acceleration: float) -> float:
        """Comfort cost for a single lateral-acceleration value."""
        excess = max(lateral_acceleration - self._vehicle_max_accel, 0.0)
        return self._normalise(excess, 0.0, self._vehicle_max_accel * 2)

    def _safety_cost(
        self,
        waypoints: NDArray[np.float64],
        obstacles: Optional[NDArray[np.float64]],
    ) -> float:
        """Safety cost based on proximity of the path to obstacles.

        Uses an inverse-distance model: cost grows sharply as the minimum
        distance to any obstacle approaches the collision-check radius.
        """
        if obstacles is None or obstacles.shape[0] == 0:
            return 0.0

        pts = waypoints[:, :2]
        # Compute pairwise distances between path points and obstacles
        # Shape: (N, M)
        diff = pts[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dists = np.min(dists, axis=1)  # closest obstacle per waypoint

        # Inverse-distance cost per waypoint
        r = self._collision_check_radius
        # C(d) = max(0, 1 - d/r)  => 1 at d=0, 0 at d>=r
        costs = np.maximum(1.0 - min_dists / r, 0.0)
        # Use the 95th-percentile to emphasise worst-case regions
        safety_cost = float(np.percentile(costs, 95))
        return safety_cost

    def _safety_cost_scalar(self, min_obstacle_distance: float) -> float:
        """Safety cost from a single proximity value."""
        r = self._collision_check_radius
        return max(1.0 - min_obstacle_distance / r, 0.0)

    def _rule_compliance_cost(
        self,
        waypoints: NDArray[np.float64],
        desired_speed: float,
    ) -> float:
        """Rule-compliance cost penalising excess speed over the limit."""
        excess = max(desired_speed - self._speed_limit, 0.0)
        return self._rule_cost_scalar(excess)

    def _rule_cost_scalar(self, speed_excess: float) -> float:
        """Rule cost from speed-excess scalar."""
        return self._normalise(speed_excess, 0.0, self._speed_limit)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, breakdown: CostBreakdown) -> float:
        """Weighted sum of component costs."""
        w = self._weights.as_vector()
        c = breakdown.component_vector()
        return float(np.dot(w, c))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(value: float, lo: float, hi: float) -> float:
        """Min-max normalise *value* into [0, 1] given a reference range."""
        if hi <= lo:
            return 0.0
        return max(min((value - lo) / (hi - lo), 1.0), 0.0)

    def __repr__(self) -> str:
        return (
            f"CostCalculator(scenario={self._scenario.name}, "
            f"weights={self._weights})"
        )
