"""Entry point for the end-to-end autonomy pipeline demo (module 08).

Runs the full sense -> localize -> plan -> act loop for every scenario
configured in ``config/system_config.yaml``:

1. sense: the perception interface returns synthetic vehicle detections
   around the ground-truth state,
2. localize: the localization interface perturbs the ground-truth pose
   with GNSS-style Gaussian noise,
3. plan: the planning interface computes a speed/heading target toward
   the goal from the noisy estimate,
4. act: a proportional surrogate controller (a stand-in for the module
   06 PID controller) converts the target into throttle and steering,
   and the module 01 kinematic bicycle model advances the simulation by
   one tick.

Per-scenario tick logs land in ``logs/<scenario>.log``; per-scenario
result dictionaries and the overall ``run_summary.json`` land in the
results output directory (default ``results/``).

All runtime paths are resolved against this module directory, never
against the current working directory, so the script behaves identically
when invoked from anywhere.

Extension point: swap the stub interfaces for modules 02-06, add
adversarial actors, and sweep seeds/thresholds for batch experiments.
"""

from __future__ import annotations

import argparse
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent            # .../08_full_system_demo
PROJECT_ROOT = MODULE_DIR.parents[1]                    # repository root
for _path in (MODULE_DIR, PROJECT_ROOT, PROJECT_ROOT / "01_basic_simulation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from components.localization_interface import LocalizationInterface  # noqa: E402
from components.perception_interface import PerceptionInterface  # noqa: E402
from components.planning_interface import PlanningInterface, wrap_angle  # noqa: E402
from components.simulation_interface import SimulationInterface  # noqa: E402
from shared.utilities.config_loader import load_yaml  # noqa: E402  (path setup above)
from shared.utilities.data_loader import save_json  # noqa: E402  (path setup above)
from shared.utilities.logging_utils import get_logger  # noqa: E402  (path setup above)

DEFAULT_CONFIG_PATH = MODULE_DIR / "config" / "system_config.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "results"

#: Goal tolerance in metres; must match SimulationInterface._GOAL_TOLERANCE_M
#: so that the reported ``goal_reached`` metric agrees with the loop's
#: termination check.
_GOAL_TOLERANCE_M: float = 5.0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``--config``, ``--show``, ``--output-dir``
        and ``--seed`` populated (paths as :class:`pathlib.Path`).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the module 08 full-system autonomy pipeline: sense -> "
            "localize -> plan -> act for every configured scenario, "
            "writing tick logs, per-scenario results and run_summary.json."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the system configuration YAML (default: module config).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Accepted for CLI symmetry; module 08 produces no plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-scenario result JSON files and run_summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base RNG seed; each scenario adds a deterministic offset "
            "derived from its name (crc32 modulo 1000)."
        ),
    )
    return parser.parse_args()


def _scenario_seed_offset(name: str) -> int:
    """Derive a deterministic per-scenario RNG seed offset.

    Args:
        name: Scenario name [-].

    Returns:
        Offset in [0, 999] [-] computed as ``crc32(name) % 1000``.
        ``zlib.crc32`` is used instead of the built-in ``hash()`` because
        the latter is salted per Python process, which would break
        run-to-run reproducibility of the seeded noise.
    """
    return zlib.crc32(name.encode("utf-8")) % 1000


def _run_scenario(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
    log_directory: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one scenario through the full pipeline and persist results.

    Args:
        scenario: Parsed scenario dictionary (name, start, goal, ...).
        config: Full system configuration dictionary.
        args: Parsed CLI arguments (seed and related flags).
        log_directory: Directory for per-scenario tick logs.
        output_dir: Directory for per-scenario result JSON files.

    Returns:
        Summary dictionary with the scenario name, tick count and all
        run metrics (duration, distance, mean speed, final position,
        detection total, goal flag).
    """
    name = str(scenario["name"])
    goal_x = float(scenario["goal"]["x"])
    goal_y = float(scenario["goal"]["y"])

    perception = PerceptionInterface(config["modules"]["perception"])
    localization = LocalizationInterface(config["modules"]["localization"])
    planning = PlanningInterface(config["modules"]["planning"])
    planning.set_goal(goal_x, goal_y)
    simulation = SimulationInterface(config, scenario)

    logger = get_logger(
        f"08_full_system_demo.{name}",
        log_file=log_directory / f"{name}.log",
        level=str(config["logging"]["level"]),
    )

    rng = np.random.default_rng(args.seed + _scenario_seed_offset(name))

    steering_gain = float(config["modules"]["planning"]["steering_gain"])
    tick_records: List[Dict[str, Any]] = []

    logger.info(
        "Scenario '%s' started: start=(%.1f, %.1f) m, goal=(%.1f, %.1f) m",
        name,
        float(scenario["start"]["x"]),
        float(scenario["start"]["y"]),
        goal_x,
        goal_y,
    )

    while not simulation.is_complete():
        truth = simulation.truth_state
        detections = perception.sense(truth, rng)
        estimate = localization.get_pose(truth, rng)
        target = planning.compute_target(estimate)

        # Proportional surrogate for the module 06 PID speed controller:
        # throttle gain 0.5 maps a 2 m/s speed error to full-scale
        # command, then clamps to the [-1, 1] actuator range.
        throttle = float(
            np.clip(
                0.5 * (float(target["target_speed_mps"]) - truth.speed),
                -1.0,
                1.0,
            )
        )
        # Heading tracking: steering gain times the wrapped heading
        # error, clamped to the vehicle model's max steering (0.6 rad).
        steering = float(
            np.clip(
                steering_gain
                * wrap_angle(float(target["target_heading_rad"]) - truth.yaw),
                -0.6,
                0.6,
            )
        )

        truth = simulation.step(truth, {"throttle": throttle, "steering": steering})

        record = {
            "t": round(float(simulation.time_s), 3),
            "x": round(float(truth.x), 4),
            "y": round(float(truth.y), 4),
            "speed": round(float(truth.speed), 4),
            "target_speed_mps": round(float(target["target_speed_mps"]), 4),
            "distance_m": round(float(simulation.distance_m), 4),
            "n_detections": len(detections),
        }
        tick_records.append(record)

        logger.info(
            "t=%6.2f s | x=%8.2f m | y=%7.2f m | speed=%6.2f m/s | detections=%d",
            simulation.time_s,
            truth.x,
            truth.y,
            truth.speed,
            len(detections),
        )

    final_truth = simulation.truth_state
    goal_distance_m = math.hypot(final_truth.x - goal_x, final_truth.y - goal_y)
    speeds = [float(record["speed"]) for record in tick_records]
    metrics = {
        "duration_s": float(simulation.time_s),
        "distance_m": float(simulation.distance_m),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "final_position_xy": [float(final_truth.x), float(final_truth.y)],
        "detections_total": int(sum(int(record["n_detections"]) for record in tick_records)),
        "goal_reached": bool(goal_distance_m < _GOAL_TOLERANCE_M),
    }

    result = {
        "scenario": name,
        "n_ticks": len(tick_records),
        "tick_records": tick_records,
        "metrics": metrics,
    }
    results_path = save_json(result, output_dir / f"{name}_results.json")
    logger.info("Scenario '%s' finished, results written to %s", name, results_path)

    print(
        f"[{name}] ticks={len(tick_records)} "
        f"duration={metrics['duration_s']:.2f} s "
        f"distance={metrics['distance_m']:.2f} m "
        f"mean_speed={metrics['mean_speed_mps']:.2f} m/s "
        f"goal_reached={metrics['goal_reached']} "
        f"detections_total={metrics['detections_total']}"
    )
    return {"name": name, "n_ticks": len(tick_records), **metrics}


def main() -> int:
    """Run the full pipeline for every configured scenario.

    Returns:
        Process exit code, 0 on success.
    """
    args = _parse_args()
    config = load_yaml(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_directory = MODULE_DIR / str(config.get("logging", {}).get("directory", "logs"))

    scenario_summaries: List[Dict[str, Any]] = []
    for scenario_file in config["scenarios"]["files"]:
        scenario = load_yaml(MODULE_DIR / "scenarios" / str(scenario_file))
        scenario_summaries.append(
            _run_scenario(scenario, config, args, log_directory, output_dir)
        )

    summary_path = save_json({"scenarios": scenario_summaries}, output_dir / "run_summary.json")

    print("=" * 72)
    print("Module 08 full-system demo summary")
    for entry in scenario_summaries:
        print(
            f"  {entry['name']:<10} ticks={entry['n_ticks']:>4}  "
            f"duration={entry['duration_s']:6.2f} s  "
            f"distance={entry['distance_m']:8.2f} m  "
            f"mean_speed={entry['mean_speed_mps']:6.2f} m/s  "
            f"goal_reached={entry['goal_reached']}  "
            f"detections={entry['detections_total']}"
        )
    print(f"Results directory: {output_dir}")
    print(f"Run summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
