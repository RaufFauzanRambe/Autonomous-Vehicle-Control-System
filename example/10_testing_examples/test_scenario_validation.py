"""Validation tests for the driving-scenario dataset and evaluator (module 07).

Covered behavior: structural schema of ``driving_scenarios.json`` and
agreement between the rule-based ``ScenarioEvaluator`` decisions and the
``expected_decision`` annotations of every scenario.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless rendering for plotting tests

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "07_ai_decision_demo"))

from shared.utilities.config_loader import load_json
from decision.environment_state import EnvironmentState
from decision.scenario_evaluator import ScenarioEvaluator

SCENARIO_PATH = ROOT / "07_ai_decision_demo" / "sample_data" / "driving_scenarios.json"
DATA = load_json(SCENARIO_PATH)

REQUIRED_SCENARIO_KEYS = (
    "id",
    "name",
    "ego_state",
    "actors",
    "road_features",
    "expected_decision",
)
REQUIRED_EGO_KEYS = ("x", "y", "yaw", "speed")
VALID_DECISIONS = {"cruise", "stop", "follow"}


def test_scenario_file_structure() -> None:
    """The scenario dataset must follow the documented structure."""
    scenarios = DATA["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 4
    for scenario in scenarios:
        assert isinstance(scenario, dict)
        for key in REQUIRED_SCENARIO_KEYS:
            assert key in scenario
        ego_state = scenario["ego_state"]
        for key in REQUIRED_EGO_KEYS:
            assert key in ego_state
        assert scenario["expected_decision"] in VALID_DECISIONS


def test_evaluator_matches_expected() -> None:
    """The rule-based evaluator must reproduce every expected decision."""
    evaluator = ScenarioEvaluator()
    for scenario in DATA["scenarios"]:
        environment = EnvironmentState.from_scenario(scenario)
        decision = evaluator.evaluate(environment)
        assert decision.action == scenario["expected_decision"]
