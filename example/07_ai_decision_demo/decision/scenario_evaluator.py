"""Rule-based decision evaluation with pinned rule precedence.

The :class:`ScenarioEvaluator` maps an :class:`EnvironmentState` snapshot
to exactly one action from the vocabulary ``{"stop", "follow", "cruise"}``
by trying four ordered private rules; the first rule that matches wins:

1. ``_rule_traffic_light``: red traffic light inside the stop distance.
2. ``_rule_pedestrian``: occupied crossing or pedestrian inside the stop
   distance.
3. ``_rule_follow``: slower lead vehicle inside the desired time headway.
4. ``_rule_cruise``: unconditional fallback.

Rule precedence is therefore: traffic light > pedestrian > follow >
cruise.

Extension point: replace the ordered first-hit rule list with
utility-based scoring (each rule returns a score, argmax wins) or with a
learned policy trained on logged ``EnvironmentState`` -> ``Decision``
pairs, keeping this rule engine as a guardrail.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

MODULE_DIR = Path(__file__).resolve().parent            # .../07_ai_decision_demo/decision
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, MODULE_DIR.parent, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from decision.environment_state import EnvironmentState  # noqa: E402  (path setup above)
from shared.utilities.config_loader import deep_update  # noqa: E402  (path setup above)

ALLOWED_ACTIONS: tuple[str, ...] = ("stop", "follow", "cruise")

#: Built-in defaults, matching ``config/decision_config.yaml`` exactly so
#: that ``ScenarioEvaluator()`` without arguments behaves identically to
#: ``ScenarioEvaluator(load_yaml("config/decision_config.yaml"))``.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "rules": {
        "stop_distance_m": 20.0,
        "follow_time_s": 2.0,
        "brake_deceleration_mpss": 3.0,
        "cruise_speed_mps": 14.0,
    },
    "thresholds": {
        "ttc_stop_s": 4.0,
    },
    "evaluation": {
        "require_reason_strings": True,
    },
}


@dataclass
class Decision:
    """The outcome of evaluating one environment snapshot.

    Attributes:
        action: Chosen action, one of ``{"stop", "follow", "cruise"}`` [-].
        reason: Human-readable audit trail explaining the choice [-].
        target_speed_mps: Speed the action aims for [m/s]; ``None`` when
            the action carries no speed target.
        confidence: Rule confidence in the closed interval [0, 1] [-].

    Raises:
        ValueError: If ``action`` is outside the vocabulary or
            ``confidence`` is outside [0, 1].
    """

    action: str
    reason: str
    target_speed_mps: Optional[float]
    confidence: float

    def __post_init__(self) -> None:
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"Unknown action '{self.action}'; allowed actions: {ALLOWED_ACTIONS}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must lie in [0, 1]; got {self.confidence}"
            )


class ScenarioEvaluator:
    """Ordered rule evaluator producing one :class:`Decision` per snapshot.

    Args:
        config: Optional configuration dictionary shaped like
            ``config/decision_config.yaml`` (keys ``rules``,
            ``thresholds``, ``evaluation``). When ``None`` or empty, the
            built-in defaults are used; provided keys override defaults
            via a deep merge.

    Attributes:
        stop_distance_m: Stop-rule activation distance [m].
        follow_time_s: Desired time headway [s].
        brake_deceleration_mpss: Comfortable deceleration used in audit
            reason strings [m/s^2].
        cruise_speed_mps: Fallback cruise target speed [m/s].
        ttc_stop_s: Time-to-collision advisory threshold [s].
        require_reason_strings: When True, decisions with empty reason
            strings raise :class:`ValueError` [-].
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        merged = deep_update(copy.deepcopy(_DEFAULT_CONFIG), config or {})
        rules = merged["rules"]
        thresholds = merged["thresholds"]
        evaluation = merged["evaluation"]
        self.stop_distance_m = float(rules["stop_distance_m"])
        self.follow_time_s = float(rules["follow_time_s"])
        self.brake_deceleration_mpss = float(rules["brake_deceleration_mpss"])
        self.cruise_speed_mps = float(rules["cruise_speed_mps"])
        self.ttc_stop_s = float(thresholds["ttc_stop_s"])
        self.require_reason_strings = bool(evaluation["require_reason_strings"])

    def evaluate(self, env: EnvironmentState) -> Decision:
        """Evaluate one environment snapshot under the pinned precedence.

        The ordered rules are tried in turn and the first non-``None``
        result is returned (first hit wins).

        Args:
            env: Environment snapshot to evaluate.

        Returns:
            The winning :class:`Decision`.

        Raises:
            ValueError: If ``require_reason_strings`` is enabled and the
                winning decision carries an empty reason string.

        Extension point: swap the ordered rule list for utility-based
        scoring (score per rule, argmax wins) or a learned policy; keep
        this deterministic engine as a fallback guardrail.
        """
        for rule_method in (
            self._rule_traffic_light,
            self._rule_pedestrian,
            self._rule_follow,
            self._rule_cruise,
        ):
            decision = rule_method(env)
            if decision is not None:
                if self.require_reason_strings and not decision.reason.strip():
                    raise ValueError(
                        f"Decision for action '{decision.action}' carries an "
                        "empty reason string but "
                        "evaluation.require_reason_strings is enabled"
                    )
                return decision
        raise RuntimeError(
            "No rule matched; _rule_cruise is an unconditional fallback and "
            "must always match."
        )

    def _rule_traffic_light(self, env: EnvironmentState) -> Decision | None:
        """Stop for a red traffic light inside the stop distance.

        Args:
            env: Environment snapshot to evaluate.

        Returns:
            A ``stop`` :class:`Decision` (confidence 0.95, target speed
            0.0 [m/s]) for the first red light within ``stop_distance_m``,
            or ``None`` when no such light exists.
        """
        ego = env.ego
        for feature in env.features:
            if feature.type != "traffic_light" or feature.state != "red":
                continue
            if feature.distance_m > self.stop_distance_m:
                continue
            if feature.distance_m > 0.0 and ego.speed > 0.0:
                required_decel_mpss = ego.speed ** 2 / (2.0 * feature.distance_m)
                decel_note = (
                    f"required deceleration {required_decel_mpss:.2f} m/s^2 "
                    f"vs comfortable limit {self.brake_deceleration_mpss:.2f} m/s^2"
                )
            else:
                decel_note = "ego already at the stop line or at rest"
            reason = (
                f"Red traffic light {feature.distance_m:.1f} m ahead "
                f"({decel_note})"
            )
            return Decision(
                action="stop",
                reason=reason,
                target_speed_mps=0.0,
                confidence=0.95,
            )
        return None

    def _rule_pedestrian(self, env: EnvironmentState) -> Decision | None:
        """Stop for an occupied crossing or a pedestrian in reach.

        Args:
            env: Environment snapshot to evaluate.

        Returns:
            A ``stop`` :class:`Decision` (confidence 0.9, target speed
            0.0 [m/s]) when an occupied pedestrian crossing or a
            pedestrian actor lies within ``stop_distance_m``, or ``None``
            otherwise.
        """
        for feature in env.features:
            if (
                feature.type == "pedestrian_crossing"
                and feature.state == "occupied"
                and feature.distance_m <= self.stop_distance_m
            ):
                return Decision(
                    action="stop",
                    reason=(
                        f"Occupied pedestrian crossing {feature.distance_m:.1f} m ahead"
                    ),
                    target_speed_mps=0.0,
                    confidence=0.9,
                )
        for actor in env.actors:
            if actor.type != "pedestrian":
                continue
            distance_m = env.distance_to(actor)
            if distance_m <= self.stop_distance_m:
                return Decision(
                    action="stop",
                    reason=(
                        f"Pedestrian {actor.id} {distance_m:.1f} m from ego"
                    ),
                    target_speed_mps=0.0,
                    confidence=0.9,
                )
        return None

    def _rule_follow(self, env: EnvironmentState) -> Decision | None:
        """Follow a slower lead vehicle inside the desired headway.

        The follow threshold is ``ego.speed * follow_time_s`` [m]; the
        lead must be slower than ``0.9 * ego.speed`` [m/s] so that a
        barely-slower vehicle does not trigger speed chatter.

        Args:
            env: Environment snapshot to evaluate.

        Returns:
            A ``follow`` :class:`Decision` (confidence 0.85, target speed
            equal to the lead speed [m/s]) for the first matching vehicle
            actor, or ``None`` otherwise.
        """
        ego = env.ego
        headway_threshold_m = ego.speed * self.follow_time_s
        for actor in env.actors:
            if actor.type != "vehicle":
                continue
            distance_m = env.distance_to(actor)
            if distance_m <= headway_threshold_m and actor.speed < 0.9 * ego.speed:
                ttc_s = env.time_to_collision(actor)
                ttc_note = (
                    f"ttc {ttc_s:.1f} s" if ttc_s is not None else "not closing"
                )
                reason = (
                    f"Vehicle {actor.id} {distance_m:.1f} m ahead within desired "
                    f"headway {headway_threshold_m:.1f} m ({ttc_note}; advisory "
                    f"ttc stop threshold {self.ttc_stop_s:.1f} s), matching its "
                    f"{actor.speed:.1f} m/s"
                )
                return Decision(
                    action="follow",
                    reason=reason,
                    target_speed_mps=float(actor.speed),
                    confidence=0.85,
                )
        return None

    def _rule_cruise(self, env: EnvironmentState) -> Decision:
        """Unconditional cruise fallback.

        Args:
            env: Environment snapshot (unused; kept for rule signature
                symmetry).

        Returns:
            A ``cruise`` :class:`Decision` (confidence 0.8, target speed
            ``cruise_speed_mps`` [m/s]).
        """
        reason = (
            "No signals, pedestrians or slower leads within operational "
            f"limits; cruising at {self.cruise_speed_mps:.1f} m/s"
        )
        return Decision(
            action="cruise",
            reason=reason,
            target_speed_mps=float(self.cruise_speed_mps),
            confidence=0.8,
        )
