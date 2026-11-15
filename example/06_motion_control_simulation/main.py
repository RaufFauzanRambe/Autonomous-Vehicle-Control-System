"""Entry point for module 06: closed-loop motion control simulation.

For every scenario configured in ``config/simulation_config.yaml`` the
script runs a closed-loop simulation of the module 01 kinematic bicycle
vehicle: a speed PID (normalized throttle in [-1, 1]) tracks the scenario
``target_speed_mps`` and a simplified pure-pursuit-style heading controller
steers toward a reference point one lookahead distance ahead on the
scenario path. Each run produces a three-panel response figure and per-
scenario metrics that are collected in ``run_summary.json``.

The vehicle parameters are kept inline in this file because module 01 owns
the canonical vehicle parameter file (``01_basic_simulation/vehicle/
vehicle_parameters.yaml``); the inline dictionary mirrors exactly the
canonical values so the two modules stay consistent.

Typical usage (from the project root)::

    python 06_motion_control_simulation/main.py
    python 06_motion_control_simulation/main.py --show

Extension point: read the vehicle parameters from the module 01 canonical
file once a shared parameter registry exists, and add parameter-sweep /
Monte Carlo runners that call :class:`SimulatedVehicleResponse` repeatedly
to study controller robustness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent  # project root (contains shared/ and module 01)

sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_basic_simulation"))

from control.control_visualizer import ControlVisualizer
from control.simulated_vehicle_response import SimulatedVehicleResponse
from shared.utilities.config_loader import load_yaml
from shared.utilities.data_loader import save_json
from shared.utilities.logging_utils import get_logger

LOGGER = get_logger("06_motion_control_simulation")

DEFAULT_CONFIG_PATH = MODULE_DIR / "config" / "simulation_config.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs"

# Kept inline because module 01 owns the canonical vehicle parameter file;
# values mirror 01_basic_simulation/vehicle/vehicle_parameters.yaml.
VEHICLE_PARAMS: Dict[str, float] = {
    "wheelbase_m": 2.7,          # [m] axle distance
    "max_steering_rad": 0.6,     # [rad] steering rack limit
    "max_speed_mps": 33.0,       # [m/s] top speed
    "max_accel_mpss": 3.0,       # [m/s^2] full-throttle acceleration
    "max_decel_mpss": 8.0,       # [m/s^2] full-brake deceleration
    "drag_coefficient": 0.02,    # [1/s] linear speed drag
    "mass_kg": 1500.0,           # [kg] vehicle mass (reserved for dynamic models)
}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument strings to parse; ``None`` uses ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``config``, ``show``, ``output_dir`` and
        ``seed`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simulate closed-loop speed (PID) and heading (pure-pursuit "
            "style) control of the module 01 vehicle model over configured "
            "scenarios and write per-scenario metrics."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Simulation configuration YAML (default: module config; relative paths resolve against the module directory).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive Matplotlib windows in addition to saving PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG figures and run_summary.json (default: module outputs; relative paths resolve against the module directory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed kept for CLI uniformity across modules (simulation is deterministic).",
    )
    return parser.parse_args(argv)


def resolve_module_path(path: Path) -> Path:
    """Resolve a possibly relative CLI path against the module directory.

    Args:
        path: User-supplied path; absolute paths are returned unchanged.

    Returns:
        Absolute path anchored either at the supplied absolute location or
        at the module directory.
    """
    return path if path.is_absolute() else MODULE_DIR / path


def steady_state_speed_rmse(speed: np.ndarray, target_speed_mps: float) -> float:
    """RMSE of the tracked speed against the target speed setpoint.

    The error series is evaluated from the first sample at which the
    tracked speed enters a +/-10 % band around the target; the cold-start
    transient before that moment is excluded because a ramp from standstill
    to the setpoint at the vehicle's maximum acceleration (3.0 m/s^2) alone
    dominates the full-run error and would mask the steady tracking
    quality this metric is meant to quantify.

    Args:
        speed: Tracked speed samples [m/s].
        target_speed_mps: Constant speed setpoint [m/s].

    Returns:
        Root-mean-square speed tracking error [m/s] over the settled
        portion of the run (0.0 for empty input).
    """
    if speed.size == 0:
        return 0.0
    band_mps = 0.1 * abs(target_speed_mps) if target_speed_mps != 0.0 else 0.1
    settled_indices = np.nonzero(np.abs(speed - target_speed_mps) <= band_mps)[0]
    start_index = int(settled_indices[0]) if settled_indices.size > 0 else 0
    error = speed[start_index:] - target_speed_mps
    if error.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(error))))


def simulate_scenario(
    scenario_file: Path,
    vehicle_params: Dict[str, float],
    config: Dict[str, Any],
    output_dir: Path,
    show: bool,
) -> Dict[str, Any]:
    """Run one scenario end to end and collect its metrics.

    Args:
        scenario_file: Absolute path to the scenario YAML file.
        vehicle_params: Vehicle parameter dictionary for the module 01
            plant model.
        config: Full simulation/control configuration dictionary.
        output_dir: Directory that receives the response PNG.
        show: Passed through to the visualiser (interactive windows).

    Returns:
        Metric dictionary with keys ``scenario`` [-], ``duration_s`` [s],
        ``distance_m`` [m], ``speed_rmse_mps`` [m/s],
        ``max_cross_track_error_m`` [m] and ``final_cross_track_error_m``
        [m].
    """
    scenario = load_yaml(scenario_file)
    result = SimulatedVehicleResponse(scenario, vehicle_params, config).run()

    scenario_name = str(result["name"])
    figure_path = output_dir / f"control_{scenario_name}.png"
    ControlVisualizer.plot_response(result, title=scenario_name, out_path=figure_path, show=show)

    time = np.asarray(result["time"], dtype=float)
    speed = np.asarray(result["speed"], dtype=float)
    cross_track = np.asarray(result["cross_track_error"], dtype=float)

    # Trapezoidal odometry: distance travelled [m] as the time integral of speed.
    distance_m = float(np.sum(0.5 * (speed[1:] + speed[:-1]) * np.diff(time)))

    metrics: Dict[str, Any] = {
        "scenario": scenario_name,
        "duration_s": round(float(time[-1]), 3),
        "distance_m": round(distance_m, 3),
        "speed_rmse_mps": round(steady_state_speed_rmse(speed, float(scenario["target_speed_mps"])), 4),
        "max_cross_track_error_m": round(float(np.max(np.abs(cross_track))), 4),
        "final_cross_track_error_m": round(float(abs(cross_track[-1])), 4),
    }
    LOGGER.info(
        "Scenario %s: duration %.2f s | distance %.2f m | speed RMSE %.3f m/s | "
        "max cross-track %.3f m | final cross-track %.3f m",
        metrics["scenario"],
        metrics["duration_s"],
        metrics["distance_m"],
        metrics["speed_rmse_mps"],
        metrics["max_cross_track_error_m"],
        metrics["final_cross_track_error_m"],
    )
    return metrics


def main(argv: List[str] | None = None) -> int:
    """Run the control simulation for all configured scenarios.

    Args:
        argv: Argument strings; ``None`` uses ``sys.argv[1:]``.

    Returns:
        Process exit code, ``0`` on success.
    """
    args = parse_args(argv)
    LOGGER.info("Random seed %d accepted; the simulation is deterministic and the seed does not alter results.", args.seed)

    # Fix the backend before any pyplot import: headless runs must not
    # require a display, interactive runs honour --show.
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (locks the backend choice in)

    config = load_yaml(resolve_module_path(args.config))
    output_dir = resolve_module_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_settings = config["scenarios"]
    scenario_directory = resolve_module_path(Path(str(scenario_settings["directory"])))
    LOGGER.info("Vehicle params: %s", VEHICLE_PARAMS)
    LOGGER.info("Scenarios: %d file(s) from %s", len(scenario_settings["files"]), scenario_directory)

    scenario_records: List[Dict[str, Any]] = []
    for scenario_file_name in scenario_settings["files"]:
        scenario_records.append(
            simulate_scenario(
                scenario_file=scenario_directory / str(scenario_file_name),
                vehicle_params=VEHICLE_PARAMS,
                config=config,
                output_dir=output_dir,
                show=args.show,
            )
        )

    summary_path = save_json({"scenarios": scenario_records}, output_dir / "run_summary.json")

    LOGGER.info(
        "Summary: %d scenario(s) | total distance %.2f m | worst speed RMSE %.3f m/s | "
        "worst max cross-track %.3f m",
        len(scenario_records),
        sum(record["distance_m"] for record in scenario_records),
        max(record["speed_rmse_mps"] for record in scenario_records),
        max(record["max_cross_track_error_m"] for record in scenario_records),
    )
    LOGGER.info("Figures and summary written to %s", output_dir)
    LOGGER.info("run_summary.json: %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
