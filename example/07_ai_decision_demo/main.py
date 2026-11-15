"""Entry point for the rule-based driving decision demo (module 07).

Flow: parse CLI arguments -> load the YAML decision configuration ->
load the JSON scenario catalog -> evaluate every scenario with the
pinned rule precedence (traffic light > pedestrian > follow > cruise) ->
render one top-down schematic per scenario -> write ``run_summary.json``
with per-decision records and the match count against the
``expected_decision`` labels.

All runtime paths are resolved against this module directory, never
against the current working directory, so the script behaves identically
when invoked from anywhere.

Extension point: batch-evaluate larger scenario catalogs (or recordings
from modules 02-04) and emit per-rule hit statistics for rule-coverage
analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

MODULE_DIR = Path(__file__).resolve().parent            # .../07_ai_decision_demo
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from decision.environment_state import EnvironmentState  # noqa: E402  (path setup above)
from decision.scenario_evaluator import ScenarioEvaluator  # noqa: E402  (path setup above)
from shared.utilities.config_loader import load_json, load_yaml  # noqa: E402
from shared.utilities.data_loader import save_json  # noqa: E402  (path setup above)

DEFAULT_CONFIG_PATH = MODULE_DIR / "config" / "decision_config.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs"
SCENARIOS_PATH = MODULE_DIR / "sample_data" / "driving_scenarios.json"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``--config``, ``--show``, ``--output-dir``
        and ``--seed`` populated (all paths as :class:`pathlib.Path`).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the rule-based driving decision benchmark: evaluate all "
            "scenarios, render schematics and write run_summary.json."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the decision configuration YAML (default: module config).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each schematic in an interactive window after saving.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for decision PNG files and run_summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Reserved RNG seed for CLI symmetry; the rule evaluation is "
            "fully deterministic and consumes no randomness."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the full decision benchmark.

    Returns:
        Process exit code, 0 on success.
    """
    args = _parse_args()
    if not args.show:
        matplotlib.use("Agg")  # headless backend, set before any pyplot import
    import matplotlib.pyplot as plt  # noqa: F401  (imported here so the backend decision above applies first)
    from decision.decision_visualizer import DecisionVisualizer

    config = load_yaml(args.config)
    scenarios = load_json(SCENARIOS_PATH)["scenarios"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ScenarioEvaluator(config)
    visualizer = DecisionVisualizer()

    records: List[Dict[str, Any]] = []
    for scenario in scenarios:
        env = EnvironmentState.from_scenario(scenario)
        decision = evaluator.evaluate(env)
        figure_path = visualizer.plot_scenario(
            env,
            decision,
            out_path=output_dir / f"decision_{scenario['id']}.png",
            show=args.show,
        )
        record = {
            "id": str(scenario["id"]),
            "action": decision.action,
            "reason": decision.reason,
            "target_speed_mps": decision.target_speed_mps,
            "confidence": decision.confidence,
            "expected_decision": str(scenario.get("expected_decision", "")),
            "matches_expected": bool(decision.action == scenario.get("expected_decision")),
        }
        records.append(record)
        print(
            f"[{record['id']}] action={decision.action} "
            f"expected={record['expected_decision']} "
            f"match={record['matches_expected']} "
            f"target={decision.target_speed_mps} m/s "
            f"confidence={decision.confidence:.2f} "
            f"figure={figure_path.name}"
        )

    summary = {
        "decisions": records,
        "n_matches": int(sum(1 for r in records if r["matches_expected"])),
        "n_scenarios": len(records),
    }
    summary_path = save_json(summary, output_dir / "run_summary.json")

    print(
        f"Decisions matching expected: {summary['n_matches']}/{summary['n_scenarios']}"
    )
    print(f"Run summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
