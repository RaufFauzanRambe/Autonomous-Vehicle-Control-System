"""Open-loop kinematic bicycle simulation on a straight test track.

This script drives the :class:`VehicleModel` with constant open-loop
controls from ``simulation_config.yaml``, queries the
:class:`RoadEnvironment` at every step (on-road check, speed limit) and
produces a two-panel trajectory/speed figure plus a machine-readable
``run_summary.json``.

Run from the project root (no working-directory assumptions are made; all
paths are resolved relative to this file)::

    python 01_basic_simulation/main.py

Extension point: replace the constant controls with a PID longitudinal
controller tracking ``env.get_speed_limit`` and a pure-pursuit lateral
controller tracking the nearest segment centerline.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent  # project root (contains shared/)
sys.path.insert(0, str(MODULE_DIR))   # sibling imports: from vehicle.vehicle_model import VehicleModel
sys.path.insert(0, str(ROOT))         # shared utilities: from shared.utilities.config_loader import load_yaml

import matplotlib.axes  # noqa: E402  (imported for the Axes type annotation)

from shared.utilities.config_loader import load_yaml  # noqa: E402
from shared.utilities.data_loader import save_json  # noqa: E402
from environment.road_environment import RoadEnvironment  # noqa: E402
from vehicle.vehicle_model import VehicleModel  # noqa: E402
from vehicle.vehicle_state import VehicleState  # noqa: E402


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``config``, ``show``, ``output_dir`` and
        ``seed`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Open-loop kinematic bicycle simulation on a straight test track."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_DIR / "simulation_config.yaml",
        help="Path to the simulation YAML config (default: module simulation_config.yaml).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window instead of headless rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "outputs",
        help="Directory for the PNG figure and run_summary.json (default: module outputs/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (deterministic; reserved for future noise injection, default: 42).",
    )
    return parser.parse_args(argv)


def _draw_road_band(ax: matplotlib.axes.Axes, segments: List[Dict[str, Any]]) -> None:
    """Draw every road segment as a gray rectangular band on an axes.

    Args:
        ax: Matplotlib axes to draw on.
        segments: Segment dictionaries from :class:`RoadEnvironment` with
            ``start`` [m], ``end`` [m] and ``width_m`` [m] fields.
    """
    for segment in segments:
        start = np.array([float(segment["start"][0]), float(segment["start"][1])])
        end = np.array([float(segment["end"][0]), float(segment["end"][1])])
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        half_width = float(segment["width_m"]) / 2.0
        corners = np.array(
            [
                start + normal * half_width,
                end + normal * half_width,
                end - normal * half_width,
                start - normal * half_width,
            ]
        )
        ax.fill(corners[:, 0], corners[:, 1], color="0.75", zorder=1)


def run_simulation(
    model: VehicleModel,
    environment: RoadEnvironment,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
    """Execute the open-loop simulation loop.

    The generator ``rng`` is built in ``main`` from ``--seed`` and stays
    deterministic for a fixed seed; this demo uses constant controls, so
    the generator is reserved for future noise injection (actuator lag,
    wind disturbance) without changing the loop structure.

    Args:
        model: Vehicle plant model.
        environment: Road environment used for on-road checks and the
            speed-limit profile [m/s].
        config: Parsed ``simulation_config.yaml`` dictionary.
        rng: Deterministic random generator (fixed seed).

    Returns:
        Tuple of (recorded state dictionaries, per-step speed-limit values
        [m/s], metrics dictionary). The metrics dictionary contains
        ``duration_s`` [s], ``dt_s`` [s], ``n_steps`` [-],
        ``distance_m`` [m], ``final_speed_mps`` [m/s],
        ``mean_speed_mps`` [m/s] and ``off_road_steps`` [-].

    Raises:
        ValueError: If ``duration_s <= 0`` or ``dt_s <= 0``.
    """
    simulation_cfg = config["simulation"]
    duration_s = float(simulation_cfg["duration_s"])
    dt_s = float(simulation_cfg["dt_s"])
    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("simulation.duration_s and simulation.dt_s must be positive")

    controls = config["controls"]
    throttle = float(controls["throttle"])
    steering_angle = float(controls["steering_angle"])
    record_every_n_steps = max(1, int(config["logging"]["record_every_n_steps"]))

    initial_state = VehicleState.from_dict(config["initial_state"])

    n_steps = int(round(duration_s / dt_s))
    records: List[Dict[str, Any]] = [initial_state.to_dict()]

    state = initial_state
    distance_m = 0.0
    off_road_steps = 0
    speed_limits: List[float] = [environment.get_speed_limit(state.x, state.y)]

    for step_index in range(1, n_steps + 1):
        previous = state
        state = model.step(state, throttle, steering_angle, dt_s)
        distance_m += math.hypot(state.x - previous.x, state.y - previous.y)
        if not environment.is_on_road(state.x, state.y):
            off_road_steps += 1
        speed_limits.append(environment.get_speed_limit(state.x, state.y))
        if step_index % record_every_n_steps == 0:
            records.append(state.to_dict())

    speeds = np.array([record["speed"] for record in records], dtype=float)
    metrics: Dict[str, Any] = {
        "duration_s": duration_s,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "distance_m": distance_m,
        "final_speed_mps": float(state.speed),
        "mean_speed_mps": float(np.mean(speeds)) if speeds.size else 0.0,
        "off_road_steps": off_road_steps,
    }
    return records, speed_limits, metrics


def plot_results(
    records: List[Dict[str, Any]],
    environment: RoadEnvironment,
    speed_limits: List[float],
    output_path: Path,
    show: bool,
) -> None:
    """Render the two-panel trajectory and speed figure.

    Top panel: world-frame trajectory colored by speed with the road drawn
    as a gray band. Bottom panel: speed vs time with the recorded
    speed-limit profile drawn as horizontal line(s).

    Args:
        records: Recorded state dictionaries (stride applied at runtime).
        environment: Road environment providing segment geometry [m].
        speed_limits: Speed-limit value per simulation step [m/s].
        output_path: Destination PNG path (saved at dpi=150).
        show: True opens an interactive window; False only saves the PNG.
    """
    import matplotlib.pyplot as plt

    x = np.array([record["x"] for record in records], dtype=float)
    y = np.array([record["y"] for record in records], dtype=float)
    time = np.array([record["timestamp"] for record in records], dtype=float)
    speed = np.array([record["speed"] for record in records], dtype=float)

    fig, (ax_traj, ax_speed) = plt.subplots(2, 1, figsize=(8, 8))

    _draw_road_band(ax_traj, environment.segments)
    points = ax_traj.scatter(x, y, c=speed, cmap="viridis", s=6, zorder=2)
    fig.colorbar(points, ax=ax_traj, label="speed [m/s]")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Vehicle trajectory (colored by speed)")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.grid(True, alpha=0.3)

    ax_speed.plot(time, speed, color="tab:blue", label="vehicle speed")
    for unique_limit in sorted(set(float(value) for value in speed_limits)):
        ax_speed.axhline(
            unique_limit, color="tab:red", linestyle="--",
            label=f"speed limit [{unique_limit:.1f} m/s]",
        )
    ax_speed.set_xlabel("time [s]")
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.set_title("Speed profile vs speed limit")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    """Entry point: simulate, plot and summarize one open-loop run.

    The matplotlib backend is switched to "Agg" for headless runs before
    ``matplotlib.pyplot`` is imported inside this function.

    Args:
        argv: Optional argument list overriding ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    # pyplot is imported only after the backend decision above.
    import matplotlib.pyplot as plt  # noqa: F401  (binds the chosen backend)

    rng = np.random.default_rng(args.seed)

    config = load_yaml(args.config)
    params = load_yaml(MODULE_DIR / "vehicle" / "vehicle_parameters.yaml")
    model = VehicleModel(params)
    environment = RoadEnvironment(MODULE_DIR / "environment" / "road_map.json")

    records, speed_limits, metrics = run_simulation(model, environment, config, rng)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "basic_simulation_trajectory.png"
    plot_results(records, environment, speed_limits, figure_path, args.show)

    summary = {
        "duration_s": metrics["duration_s"],
        "dt_s": metrics["dt_s"],
        "n_steps": metrics["n_steps"],
        "distance_m": metrics["distance_m"],
        "final_speed_mps": metrics["final_speed_mps"],
        "mean_speed_mps": metrics["mean_speed_mps"],
        "off_road_steps": metrics["off_road_steps"],
    }
    summary_path = save_json(summary, output_dir / "run_summary.json")

    print("=== 01_basic_simulation run summary ===")
    print(f"duration:        {summary['duration_s']:.1f} s (dt={summary['dt_s']:.3f} s, {summary['n_steps']} steps)")
    print(f"distance:        {summary['distance_m']:.2f} m")
    print(f"final speed:     {summary['final_speed_mps']:.2f} m/s")
    print(f"mean speed:      {summary['mean_speed_mps']:.2f} m/s")
    print(f"off-road steps:  {summary['off_road_steps']} / {summary['n_steps']}")
    print(f"figure:          {figure_path}")
    print(f"summary json:    {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
