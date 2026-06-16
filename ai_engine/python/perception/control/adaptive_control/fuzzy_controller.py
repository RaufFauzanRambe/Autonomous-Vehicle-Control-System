"""
Fuzzy Logic Controller for Autonomous Vehicle Control

Implements a complete fuzzy inference system for vehicle control:
  - Membership functions: triangle, trapezoid, Gaussian
  - Fuzzification of crisp inputs
  - Rule base definition and evaluation
  - Mamdani inference engine (min implication, max aggregation)
  - Defuzzification: centroid, bisector, mean of maxima
  - Pre-configured rule bases for steering and speed control

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Membership Functions
# ---------------------------------------------------------------------------

class MembershipFunctionType(Enum):
    """Supported membership function types."""
    TRIANGLE = "triangle"
    TRAPEZOID = "trapezoid"
    GAUSSIAN = "gaussian"


@dataclass
class TriangleMF:
    """Triangle membership function defined by three points [a, b, c].

    Membership is 0 at a, rises linearly to 1 at b, then falls linearly
    to 0 at c.

    Attributes:
        a: Left foot (membership = 0).
        b: Peak (membership = 1).
        c: Right foot (membership = 0).
        name: Linguistic label.
    """
    a: float
    b: float
    c: float
    name: str = ""

    def evaluate(self, x: float) -> float:
        """Compute membership degree for crisp input *x*.

        Args:
            x: Crisp input value.

        Returns:
            Membership degree in [0, 1].
        """
        if x <= self.a or x >= self.c:
            return 0.0
        if self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 1.0
        # self.b < x < self.c
        return (self.c - x) / (self.c - self.b) if self.c != self.b else 1.0

    def __repr__(self) -> str:
        return f"TriangleMF(a={self.a}, b={self.b}, c={self.c}, name='{self.name}')"


@dataclass
class TrapezoidMF:
    """Trapezoidal membership function defined by [a, b, c, d].

    Membership rises from 0 at *a* to 1 at *b*, stays 1 until *c*,
    then falls to 0 at *d*.

    Attributes:
        a: Left foot.
        b: Left shoulder (membership becomes 1).
        c: Right shoulder (membership starts falling).
        d: Right foot.
        name: Linguistic label.
    """
    a: float
    b: float
    c: float
    d: float
    name: str = ""

    def evaluate(self, x: float) -> float:
        """Compute membership degree for crisp input *x*.

        Args:
            x: Crisp input value.

        Returns:
            Membership degree in [0, 1].
        """
        if x <= self.a or x >= self.d:
            return 0.0
        if self.b <= x <= self.c:
            return 1.0
        if self.a < x < self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 1.0
        # self.c < x < self.d
        return (self.d - x) / (self.d - self.c) if self.d != self.c else 1.0

    def __repr__(self) -> str:
        return f"TrapezoidMF(a={self.a}, b={self.b}, c={self.c}, d={self.d}, name='{self.name}')"


@dataclass
class GaussianMF:
    """Gaussian membership function.

    mu(x) = exp(-(x - mean)^2 / (2 * sigma^2))

    Attributes:
        mean: Centre of the Gaussian.
        sigma: Standard deviation (spread).
        name: Linguistic label.
    """
    mean: float
    sigma: float
    name: str = ""

    def evaluate(self, x: float) -> float:
        """Compute membership degree for crisp input *x*.

        Args:
            x: Crisp input value.

        Returns:
            Membership degree in [0, 1].
        """
        return math.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

    def __repr__(self) -> str:
        return f"GaussianMF(mean={self.mean}, sigma={self.sigma}, name='{self.name}')"


# Type alias – any membership function instance
MembershipFunction = TriangleMF | TrapezoidMF | GaussianMF


# ---------------------------------------------------------------------------
# Linguistic Variable
# ---------------------------------------------------------------------------

@dataclass
class LinguisticVariable:
    """A linguistic variable comprising multiple fuzzy sets.

    Attributes:
        name: Variable name (e.g. "lateral_error").
        universe_min: Minimum value of the universe of discourse.
        universe_max: Maximum value of the universe of discourse.
        terms: Mapping from linguistic label to membership function.
    """
    name: str
    universe_min: float
    universe_max: float
    terms: Dict[str, MembershipFunction] = field(default_factory=dict)

    def add_term(self, label: str, mf: MembershipFunction) -> None:
        """Register a linguistic term with its membership function.

        Args:
            label: Linguistic label (e.g. "NegativeBig").
            mf: Membership function instance.
        """
        self.terms[label] = mf

    def fuzzify(self, value: float) -> Dict[str, float]:
        """Fuzzify a crisp value into membership degrees.

        Args:
            value: Crisp input value.

        Returns:
            Dictionary mapping term labels to membership degrees.
        """
        return {label: mf.evaluate(value) for label, mf in self.terms.items()}

    def get_term(self, label: str) -> Optional[MembershipFunction]:
        """Retrieve a specific membership function by label."""
        return self.terms.get(label)

    def __repr__(self) -> str:
        return (
            f"LinguisticVariable(name='{self.name}', "
            f"terms={list(self.terms.keys())})"
        )


# ---------------------------------------------------------------------------
# Fuzzy Rule
# ---------------------------------------------------------------------------

@dataclass
class FuzzyRule:
    """A single fuzzy IF-THEN rule.

    Antecedent: conjunction of (variable_name, term_label) pairs.
    Consequent: (variable_name, term_label) pair.

    Attributes:
        antecedent: List of (variable_name, term_label) tuples (AND logic).
        consequent: (variable_name, term_label) tuple.
        weight: Rule weight / certainty factor in (0, 1].
    """
    antecedent: List[Tuple[str, str]]
    consequent: Tuple[str, str]
    weight: float = 1.0

    def __repr__(self) -> str:
        ante = " AND ".join(f"{v} IS {t}" for v, t in self.antecedent)
        cons_var, cons_term = self.consequent
        return f"IF {ante} THEN {cons_var} IS {cons_term} (w={self.weight})"


# ---------------------------------------------------------------------------
# Defuzzification Methods
# ---------------------------------------------------------------------------

class DefuzzificationMethod(Enum):
    """Supported defuzzification strategies."""
    CENTROID = "centroid"
    BISECTOR = "bisector"
    MEAN_OF_MAXIMA = "mean_of_maxima"


def _discrete_aggregated_output(
    output_var: LinguisticVariable,
    activations: Dict[str, float],
    resolution: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the aggregated output fuzzy set over a discrete grid.

    Uses Mamdani min-implication and max-aggregation.

    Args:
        output_var: Output linguistic variable.
        activations: Mapping from term label to firing strength.
        resolution: Number of discrete points.

    Returns:
        Tuple of (x_grid, mu_aggregated) arrays.
    """
    x_grid = np.linspace(output_var.universe_min, output_var.universe_max, resolution)
    mu_aggregated = np.zeros(resolution)

    for term_label, firing_strength in activations.items():
        mf = output_var.terms.get(term_label)
        if mf is None:
            continue
        # Min-implication: clip the membership function at firing_strength
        mu_clipped = np.array([min(mf.evaluate(float(x)), firing_strength) for x in x_grid])
        # Max-aggregation
        mu_aggregated = np.maximum(mu_aggregated, mu_clipped)

    return x_grid, mu_aggregated


def defuzzify_centroid(
    output_var: LinguisticVariable,
    activations: Dict[str, float],
    resolution: int = 200,
) -> float:
    """Centroid defuzzification.

    Computes the centre of gravity of the aggregated output fuzzy set:
        x* = ∫ x · μ(x) dx / ∫ μ(x) dx

    Args:
        output_var: Output linguistic variable.
        activations: Term label -> firing strength mapping.
        resolution: Number of discrete grid points.

    Returns:
        Crisp output value.
    """
    x_grid, mu = _discrete_aggregated_output(output_var, activations, resolution)
    total_area = np.trapz(mu, x_grid)
    if total_area < 1e-12:
        # Fallback: midpoint of universe
        return (output_var.universe_min + output_var.universe_max) / 2.0
    return float(np.trapz(x_grid * mu, x_grid) / total_area)


def defuzzify_bisector(
    output_var: LinguisticVariable,
    activations: Dict[str, float],
    resolution: int = 200,
) -> float:
    """Bisector defuzzification.

    Finds the point that divides the aggregated area into two equal halves.

    Args:
        output_var: Output linguistic variable.
        activations: Term label -> firing strength mapping.
        resolution: Number of discrete grid points.

    Returns:
        Crisp output value.
    """
    x_grid, mu = _discrete_aggregated_output(output_var, activations, resolution)
    dx = x_grid[1] - x_grid[0]
    cumulative_area = np.cumsum(mu * dx)
    total_area = cumulative_area[-1]
    if total_area < 1e-12:
        return (output_var.universe_min + output_var.universe_max) / 2.0
    half = total_area / 2.0
    idx = np.searchsorted(cumulative_area, half)
    idx = min(idx, len(x_grid) - 1)
    return float(x_grid[idx])


def defuzzify_mean_of_maxima(
    output_var: LinguisticVariable,
    activations: Dict[str, float],
    resolution: int = 200,
) -> float:
    """Mean of Maxima (MoM) defuzzification.

    Averages all x values where the aggregated membership reaches its maximum.

    Args:
        output_var: Output linguistic variable.
        activations: Term label -> firing strength mapping.
        resolution: Number of discrete grid points.

    Returns:
        Crisp output value.
    """
    x_grid, mu = _discrete_aggregated_output(output_var, activations, resolution)
    max_mu = np.max(mu)
    if max_mu < 1e-12:
        return (output_var.universe_min + output_var.universe_max) / 2.0
    max_indices = np.where(np.abs(mu - max_mu) < 1e-9)[0]
    return float(np.mean(x_grid[max_indices]))


# ---------------------------------------------------------------------------
# Mamdani Fuzzy Inference Engine
# ---------------------------------------------------------------------------

class MamdaniInferenceEngine:
    """Mamdani-style fuzzy inference engine.

    Implements the classic Mamdani inference process:
    1. Fuzzify inputs.
    2. Evaluate rule antecedents (AND = min).
    3. Apply implication (min clipping).
    4. Aggregate outputs (max).
    5. Defuzzify using the selected method.
    """

    def __init__(
        self,
        input_variables: Dict[str, LinguisticVariable],
        output_variables: Dict[str, LinguisticVariable],
        rules: List[FuzzyRule],
        defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID,
        resolution: int = 200,
    ) -> None:
        """Initialise the inference engine.

        Args:
            input_variables: Mapping from variable name to LinguisticVariable.
            output_variables: Mapping from variable name to LinguisticVariable.
            rules: List of fuzzy rules.
            defuzz_method: Defuzzification strategy.
            resolution: Number of grid points for defuzzification.
        """
        self._inputs = input_variables
        self._outputs = output_variables
        self._rules = rules
        self._defuzz_method = defuzz_method
        self._resolution = resolution

        # Cache for last inference results
        self._last_fuzzified: Dict[str, Dict[str, float]] = {}
        self._last_rule_firings: List[Tuple[FuzzyRule, float]] = []

    @property
    def rules(self) -> List[FuzzyRule]:
        """Return the rule base."""
        return self._rules

    def add_rule(self, rule: FuzzyRule) -> None:
        """Append a rule to the rule base."""
        self._rules.append(rule)

    def infer(
        self,
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        """Run the full fuzzy inference pipeline.

        Args:
            inputs: Mapping from input variable name to crisp value.

        Returns:
            Mapping from output variable name to crisp defuzzified value.
        """
        # Step 1 – Fuzzify inputs
        self._last_fuzzified = {}
        for var_name, crisp_val in inputs.items():
            lv = self._inputs.get(var_name)
            if lv is not None:
                self._last_fuzzified[var_name] = lv.fuzzify(crisp_val)

        # Step 2 & 3 – Evaluate rules and collect firing strengths
        # For each output variable we collect: term_label -> max firing
        output_activations: Dict[str, Dict[str, float]] = {
            name: {} for name in self._outputs
        }
        self._last_rule_firings = []

        for rule in self._rules:
            # Evaluate antecedent (AND = min)
            firing = rule.weight
            for var_name, term_label in rule.antecedent:
                lv = self._inputs.get(var_name)
                if lv is None:
                    firing = 0.0
                    break
                degree = lv.terms.get(term_label)
                if degree is None:
                    firing = 0.0
                    break
                fuzzified = self._last_fuzzified.get(var_name, {})
                membership = fuzzified.get(term_label, 0.0)
                firing = min(firing, membership)
                if firing <= 0.0:
                    break

            self._last_rule_firings.append((rule, firing))

            if firing > 0.0:
                out_var, out_term = rule.consequent
                if out_var not in output_activations:
                    output_activations[out_var] = {}
                # Max-aggregation across rules
                current = output_activations[out_var].get(out_term, 0.0)
                output_activations[out_var][out_term] = max(current, firing)

        # Step 4 – Defuzzify each output variable
        defuzz_func = {
            DefuzzificationMethod.CENTROID: defuzzify_centroid,
            DefuzzificationMethod.BISECTOR: defuzzify_bisector,
            DefuzzificationMethod.MEAN_OF_MAXIMA: defuzzify_mean_of_maxima,
        }[self._defuzz_method]

        results: Dict[str, float] = {}
        for out_name, activations in output_activations.items():
            out_var = self._outputs.get(out_name)
            if out_var is not None and any(v > 0 for v in activations.values()):
                results[out_name] = defuzz_func(out_var, activations, self._resolution)
            elif out_var is not None:
                results[out_name] = (out_var.universe_min + out_var.universe_max) / 2.0

        return results

    def get_firing_report(self) -> List[Tuple[str, float]]:
        """Return a human-readable firing report for the last inference.

        Returns:
            List of (rule_description, firing_strength) tuples.
        """
        return [(str(rule), strength) for rule, strength in self._last_rule_firings]


# ---------------------------------------------------------------------------
# Pre-built Vehicle Fuzzy Controllers
# ---------------------------------------------------------------------------

def _build_steering_rule_base() -> Tuple[
    Dict[str, LinguisticVariable],
    Dict[str, LinguisticVariable],
    List[FuzzyRule],
]:
    """Build the linguistic variables and rule base for steering control.

    Inputs:
      - lateral_error: cross-track error (m)
      - heading_error: yaw angle error (rad)

    Output:
      - steering_correction: corrective steering angle (rad)

    Returns:
        Tuple of (input_vars, output_vars, rules).
    """
    # --- Input: lateral_error ---
    lat_err = LinguisticVariable(
        name="lateral_error",
        universe_min=-3.0,
        universe_max=3.0,
    )
    lat_err.add_term("NB", TrapezoidMF(-3.0, -3.0, -2.0, -1.0, name="NB"))
    lat_err.add_term("NS", TriangleMF(-2.0, -1.0, 0.0, name="NS"))
    lat_err.add_term("ZE", TriangleMF(-0.5, 0.0, 0.5, name="ZE"))
    lat_err.add_term("PS", TriangleMF(0.0, 1.0, 2.0, name="PS"))
    lat_err.add_term("PB", TrapezoidMF(1.0, 2.0, 3.0, 3.0, name="PB"))

    # --- Input: heading_error ---
    hdg_err = LinguisticVariable(
        name="heading_error",
        universe_min=-0.5,
        universe_max=0.5,
    )
    hdg_err.add_term("NB", TrapezoidMF(-0.5, -0.5, -0.3, -0.15, name="NB"))
    hdg_err.add_term("NS", TriangleMF(-0.3, -0.15, 0.0, name="NS"))
    hdg_err.add_term("ZE", TriangleMF(-0.1, 0.0, 0.1, name="ZE"))
    hdg_err.add_term("PS", TriangleMF(0.0, 0.15, 0.3, name="PS"))
    hdg_err.add_term("PB", TrapezoidMF(0.15, 0.3, 0.5, 0.5, name="PB"))

    # --- Output: steering_correction ---
    steer = LinguisticVariable(
        name="steering_correction",
        universe_min=-0.15,
        universe_max=0.15,
    )
    steer.add_term("NB", TrapezoidMF(-0.15, -0.15, -0.10, -0.05, name="NB"))
    steer.add_term("NS", TriangleMF(-0.10, -0.05, 0.0, name="NS"))
    steer.add_term("ZE", TriangleMF(-0.02, 0.0, 0.02, name="ZE"))
    steer.add_term("PS", TriangleMF(0.0, 0.05, 0.10, name="PS"))
    steer.add_term("PB", TrapezoidMF(0.05, 0.10, 0.15, 0.15, name="PB"))

    # --- Rules (5 x 5 = 25 rules) ---
    terms = ["NB", "NS", "ZE", "PS", "PB"]
    rule_map = {
        # lateral_error  heading_error  =>  steering_correction
        ("NB", "NB"): "PB", ("NB", "NS"): "PB", ("NB", "ZE"): "PB", ("NB", "PS"): "PS", ("NB", "PB"): "ZE",
        ("NS", "NB"): "PB", ("NS", "NS"): "PS", ("NS", "ZE"): "PS", ("NS", "PS"): "ZE", ("NS", "PB"): "NS",
        ("ZE", "NB"): "PS", ("ZE", "NS"): "PS", ("ZE", "ZE"): "ZE", ("ZE", "PS"): "NS", ("ZE", "PB"): "NB",
        ("PS", "NB"): "PS", ("PS", "NS"): "ZE", ("PS", "ZE"): "NS", ("PS", "PS"): "NS", ("PS", "PB"): "NB",
        ("PB", "NB"): "ZE", ("PB", "NS"): "NS", ("PB", "ZE"): "NB", ("PB", "PS"): "NB", ("PB", "PB"): "NB",
    }

    rules: List[FuzzyRule] = []
    for (le_term, he_term), sc_term in rule_map.items():
        rules.append(FuzzyRule(
            antecedent=[("lateral_error", le_term), ("heading_error", he_term)],
            consequent=("steering_correction", sc_term),
        ))

    input_vars = {"lateral_error": lat_err, "heading_error": hdg_err}
    output_vars = {"steering_correction": steer}
    return input_vars, output_vars, rules


def _build_speed_rule_base() -> Tuple[
    Dict[str, LinguisticVariable],
    Dict[str, LinguisticVariable],
    List[FuzzyRule],
]:
    """Build the linguistic variables and rule base for speed control.

    Inputs:
      - speed_error: target - actual speed (m/s)
      - acceleration_error: desired - actual acceleration (m/s^2)

    Output:
      - throttle_adjustment: throttle/brake adjustment (-1 to +1)

    Returns:
        Tuple of (input_vars, output_vars, rules).
    """
    # --- Input: speed_error ---
    spd_err = LinguisticVariable(name="speed_error", universe_min=-10.0, universe_max=10.0)
    spd_err.add_term("NB", TrapezoidMF(-10.0, -10.0, -6.0, -3.0, name="NB"))
    spd_err.add_term("NS", TriangleMF(-6.0, -3.0, 0.0, name="NS"))
    spd_err.add_term("ZE", TriangleMF(-1.5, 0.0, 1.5, name="ZE"))
    spd_err.add_term("PS", TriangleMF(0.0, 3.0, 6.0, name="PS"))
    spd_err.add_term("PB", TrapezoidMF(3.0, 6.0, 10.0, 10.0, name="PB"))

    # --- Input: acceleration_error ---
    acc_err = LinguisticVariable(name="acceleration_error", universe_min=-3.0, universe_max=3.0)
    acc_err.add_term("N", TrapezoidMF(-3.0, -3.0, -1.5, -0.3, name="N"))
    acc_err.add_term("ZE", TriangleMF(-0.5, 0.0, 0.5, name="ZE"))
    acc_err.add_term("P", TrapezoidMF(0.3, 1.5, 3.0, 3.0, name="P"))

    # --- Output: throttle_adjustment ---
    throttle = LinguisticVariable(name="throttle_adjustment", universe_min=-1.0, universe_max=1.0)
    throttle.add_term("NB", TrapezoidMF(-1.0, -1.0, -0.6, -0.3, name="NB"))
    throttle.add_term("NS", TriangleMF(-0.6, -0.3, 0.0, name="NS"))
    throttle.add_term("ZE", TriangleMF(-0.1, 0.0, 0.1, name="ZE"))
    throttle.add_term("PS", TriangleMF(0.0, 0.3, 0.6, name="PS"))
    throttle.add_term("PB", TrapezoidMF(0.3, 0.6, 1.0, 1.0, name="PB"))

    # --- Rules (5 x 3 = 15 rules) ---
    rule_map = {
        ("NB", "N"): "NB", ("NB", "ZE"): "NB", ("NB", "P"): "NS",
        ("NS", "N"): "NB", ("NS", "ZE"): "NS", ("NS", "P"): "ZE",
        ("ZE", "N"): "NS", ("ZE", "ZE"): "ZE", ("ZE", "P"): "PS",
        ("PS", "N"): "ZE", ("PS", "ZE"): "PS", ("PS", "P"): "PB",
        ("PB", "N"): "PS", ("PB", "ZE"): "PB", ("PB", "P"): "PB",
    }

    rules: List[FuzzyRule] = []
    for (se_term, ae_term), th_term in rule_map.items():
        rules.append(FuzzyRule(
            antecedent=[("speed_error", se_term), ("acceleration_error", ae_term)],
            consequent=("throttle_adjustment", th_term),
        ))

    input_vars = {"speed_error": spd_err, "acceleration_error": acc_err}
    output_vars = {"throttle_adjustment": throttle}
    return input_vars, output_vars, rules


# ---------------------------------------------------------------------------
# High-Level Fuzzy Controller Wrappers
# ---------------------------------------------------------------------------

class FuzzySteeringController:
    """Fuzzy logic steering controller for autonomous vehicles.

    Uses lateral error and heading error to compute a corrective
    steering angle via Mamdani inference.

    Example:
        >>> ctrl = FuzzySteeringController()
        >>> correction = ctrl.compute(lateral_error=-0.8, heading_error=0.05)
    """

    def __init__(
        self,
        defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID,
        resolution: int = 200,
        output_gain: float = 1.0,
        output_offset: float = 0.0,
    ) -> None:
        """Initialise the fuzzy steering controller.

        Args:
            defuzz_method: Defuzzification method to use.
            resolution: Number of grid points for defuzzification.
            output_gain: Multiplicative gain on the defuzzified output.
            output_offset: Additive offset on the defuzzified output.
        """
        inp, out, rules = _build_steering_rule_base()
        self._engine = MamdaniInferenceEngine(inp, out, rules, defuzz_method, resolution)
        self._output_gain = output_gain
        self._output_offset = output_offset
        self._last_output: float = 0.0

    def compute(self, lateral_error: float, heading_error: float) -> float:
        """Compute the steering correction.

        Args:
            lateral_error: Cross-track error in metres (positive = right).
            heading_error: Heading error in radians (positive = right).

        Returns:
            Corrective steering angle in radians.
        """
        result = self._engine.infer({
            "lateral_error": lateral_error,
            "heading_error": heading_error,
        })
        raw = result.get("steering_correction", 0.0)
        self._last_output = raw * self._output_gain + self._output_offset
        return self._last_output

    @property
    def last_output(self) -> float:
        """Return the most recent steering correction."""
        return self._last_output

    def get_firing_report(self) -> List[Tuple[str, float]]:
        """Return rule firing report from last inference."""
        return self._engine.get_firing_report()

    def reset(self) -> None:
        """Reset internal state."""
        self._last_output = 0.0


class FuzzySpeedController:
    """Fuzzy logic speed controller for autonomous vehicles.

    Uses speed error and acceleration error to compute a throttle
    adjustment signal via Mamdani inference.

    Example:
        >>> ctrl = FuzzySpeedController()
        >>> adj = ctrl.compute(speed_error=3.0, acceleration_error=-0.5)
    """

    def __init__(
        self,
        defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID,
        resolution: int = 200,
        output_gain: float = 1.0,
    ) -> None:
        """Initialise the fuzzy speed controller.

        Args:
            defuzz_method: Defuzzification method to use.
            resolution: Number of grid points for defuzzification.
            output_gain: Multiplicative gain on the defuzzified output.
        """
        inp, out, rules = _build_speed_rule_base()
        self._engine = MamdaniInferenceEngine(inp, out, rules, defuzz_method, resolution)
        self._output_gain = output_gain
        self._last_output: float = 0.0

    def compute(self, speed_error: float, acceleration_error: float) -> float:
        """Compute the throttle / brake adjustment.

        Args:
            speed_error: (target_speed - current_speed) in m/s.
            acceleration_error: (desired_accel - current_accel) in m/s^2.

        Returns:
            Throttle adjustment in [-1, 1] (negative = brake).
        """
        result = self._engine.infer({
            "speed_error": speed_error,
            "acceleration_error": acceleration_error,
        })
        raw = result.get("throttle_adjustment", 0.0)
        self._last_output = raw * self._output_gain
        return self._last_output

    @property
    def last_output(self) -> float:
        """Return the most recent throttle adjustment."""
        return self._last_output

    def get_firing_report(self) -> List[Tuple[str, float]]:
        """Return rule firing report from last inference."""
        return self._engine.get_firing_report()

    def reset(self) -> None:
        """Reset internal state."""
        self._last_output = 0.0
