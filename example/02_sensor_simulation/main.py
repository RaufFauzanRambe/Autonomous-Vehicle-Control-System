"""Multi-sensor simulation over a straight-line ego ground truth.

This script generates a simple ground-truth ego trajectory (constant speed
straight-line motion), feeds it through five synthetic sensors (camera,
lidar, radar, GPS, IMU), logs their envelope readings to a JSONL file and
renders a six-panel sensor trace figure plus a machine-readable
``run_summary.json``.

Run from the project root (no working-directory assumptions are made; all
paths are resolved relative to this file)::

    python 02_sensor_simulation/main.py

Extension point: replace the analytic ground-truth generator with states
replayed from ``01_basic_simulation`` (or recorded rosbags) and validate
every JSONL line against ``shared/schemas/sensor_data.schema.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent  # project root (contains shared/ and the digit-leading module folders)
sys.path.insert(0, str(MODULE_DIR))   # sibling imports: from sensors.camera_sensor import CameraSensor
sys.path.insert(0, str(ROOT))         # shared utilities: from shared.utilities.config_loader import load_yaml

from shared.utilities.config_loader import load_yaml  # noqa: E402
from shared.utilities.data_loader import save_json  # noqa: E402
from sensors.camera_sensor import CameraSensor  # noqa: E402
from sensors.gps_sensor import GPSSensor  # noqa: E402
from sensors.imu_sensor import IMUSensor  # noqa: E402
from sensors.lidar_sensor import LidarSensor  # noqa: E402
from sensors.radar_sensor import RadarSensor  # noqa: E402

# VehicleState lives in a digit-leading namespace package and therefore
# cannot be reached with a regular import statement; see project README.
VehicleState = importlib.import_module(
    "01_basic_simulation.vehicle.vehicle_state"
).VehicleState


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``config``, ``show``, ``output_dir`` and
        ``seed`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Multi-sensor simulation over a straight-line ego ground truth."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_DIR / "config" / "sensor_config.yaml",
        help="Path to the sensor YAML config (default: module config/sensor_config.yaml).",
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
        help="Directory for the PNG figure, JSONL log and run_summary.json "
             "(default: module outputs/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared by every sensor's noise generator (default: 42).",
    )
    return parser.parse_args(argv)


def ground_truth_state(t_s: float, ego_speed_mps: float) -> VehicleState:
    """Build the ego ground-truth state for straight-line motion.

    Deliberately simple ground-truth generator: the vehicle moves along the
    world +x axis with constant speed, so ``x = ego_speed * t``, ``y = 0``,
    ``yaw = 0``, ``acceleration = 0`` and ``steering_angle = 0``.

    Args:
        t_s: Simulation time [s] (>= 0).
        ego_speed_mps: Constant longitudinal ego speed [m/s].

    Returns:
        The ground-truth :class:`VehicleState` at time ``t_s``.

    Extension point: replay states from a vehicle-model run of
    ``01_basic_simulation`` (or from recorded logs) instead of this
    analytic straight line.
    """
    return VehicleState(
        timestamp=float(t_s),
        x=ego_speed_mps * float(t_s),
        y=0.0,
        yaw=0.0,
        speed=ego_speed_mps,
        acceleration=0.0,
        steering_angle=0.0,
    )


def run_simulation(
    config: Dict[str, Any],
    seed: int,
    jsonl_path: Path,
) -> Dict[str, Any]:
    """Run the sensor loop and write the JSONL log.

    All five sensors are updated at every step; only steps matching the
    ``record_every_n_steps`` stride are appended to the JSONL file (one
    ``json.dumps`` line per envelope reading). Full-resolution per-sensor
    traces are kept in memory for plotting.

    Args:
        config: Parsed ``config/sensor_config.yaml`` dictionary.
        seed: Random seed passed to every sensor constructor.
        jsonl_path: Destination path for ``sensor_readings.jsonl``.

    Returns:
        Dictionary with per-sensor trace arrays and scalar counters:
        ``gps_x_m``, ``gps_y_m`` [m], ``true_x_m`` [m], ``time_s`` [s],
        ``ego_speed_mps`` [m/s], ``imu_ax_mpss`` [m/s^2],
        ``camera_counts`` [-], ``radar`` (target_id -> (times, ranges)),
        ``lidar_points_xy_m`` (last scan, sensor frame [m]),
        ``n_readings`` [-], ``lidar_points_last_scan`` [-],
        ``radar_target_count`` [-], ``camera_detections_total`` [-].

    Raises:
        ValueError: If ``duration_s <= 0`` or ``dt_s <= 0``.
    """
    simulation_cfg = config["simulation"]
    duration_s = float(simulation_cfg["duration_s"])
    dt_s = float(simulation_cfg["dt_s"])
    ego_speed_mps = float(simulation_cfg["ego_speed_mps"])
    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("simulation.duration_s and simulation.dt_s must be positive")
    record_every_n_steps = max(1, int(config["logging"]["record_every_n_steps"]))

    camera = CameraSensor(config["camera"], seed=seed)
    lidar = LidarSensor(config["lidar"], seed=seed)
    radar = RadarSensor(config["radar"], seed=seed)
    gps = GPSSensor(config["gps"], seed=seed)
    imu = IMUSensor(config["imu"], seed=seed)

    n_steps = int(round(duration_s / dt_s))
    time_s: List[float] = []
    true_x_m: List[float] = []
    gps_x_m: List[float] = []
    gps_y_m: List[float] = []
    ego_speed_mps_list: List[float] = []
    imu_ax_mpss: List[float] = []
    camera_counts: List[int] = []
    radar_ranges: Dict[str, List[float]] = {}
    radar_times: Dict[str, List[float]] = {}

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n_readings = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for step in range(n_steps):
            t_s = step * dt_s
            state = ground_truth_state(t_s, ego_speed_mps)

            camera.update(state, dt_s)
            lidar.update(state, dt_s)
            radar.update(state, dt_s)
            gps.update(state, dt_s)
            imu.update(state, dt_s)

            readings = {
                "camera": camera.get_reading(),
                "lidar": lidar.get_reading(),
                "radar": radar.get_reading(),
                "gps": gps.get_reading(),
                "imu": imu.get_reading(),
            }

            # Full-resolution in-memory traces for the figure panels.
            time_s.append(t_s)
            true_x_m.append(state.x)
            gps_x_m.append(float(readings["gps"]["data"]["x_m"]))
            gps_y_m.append(float(readings["gps"]["data"]["y_m"]))
            ego_speed_mps_list.append(state.speed)
            imu_ax_mpss.append(float(readings["imu"]["data"]["ax_mpss"]))
            camera_counts.append(len(readings["camera"]["data"]["detections"]))
            for target in readings["radar"]["data"]["targets"]:
                radar_ranges.setdefault(target["target_id"], []).append(
                    float(target["range_m"])
                )
                radar_times.setdefault(target["target_id"], []).append(t_s)

            if step % record_every_n_steps == 0:
                for reading in readings.values():
                    jsonl_handle.write(json.dumps(reading) + "\n")
                    n_readings += 1

    lidar_reading = lidar.get_reading()
    radar_reading = radar.get_reading()

    return {
        "time_s": time_s,
        "true_x_m": true_x_m,
        "gps_x_m": gps_x_m,
        "gps_y_m": gps_y_m,
        "ego_speed_mps": ego_speed_mps_list,
        "imu_ax_mpss": imu_ax_mpss,
        "camera_counts": camera_counts,
        "radar_ranges": radar_ranges,
        "radar_times": radar_times,
        "lidar_points_xy_m": lidar_reading["data"]["points_xy_m"],
        "n_readings": n_readings,
        "lidar_points_last_scan": int(lidar_reading["data"]["n_points"]),
        "radar_target_count": len(radar_reading["data"]["targets"]),
        "camera_detections_total": int(np.sum(camera_counts)) if camera_counts else 0,
    }


def plot_results(traces: Dict[str, Any], output_path: Path, show: bool) -> None:
    """Render the 2x3 sensor trace figure.

    Panels: (1) GPS measured positions vs true trajectory, (2) last lidar
    scan top-down in the sensor frame, (3) radar range vs time per target
    id, (4) IMU longitudinal acceleration vs time, (5) camera detection
    count vs time, (6) ego speed vs time.

    Args:
        traces: Trace dictionary produced by :func:`run_simulation`.
        output_path: Destination PNG path (saved at dpi=150).
        show: True opens an interactive window; False only saves the PNG.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    (ax_gps, ax_lidar, ax_radar), (ax_imu, ax_camera, ax_speed) = axes

    time_s = np.asarray(traces["time_s"], dtype=float)
    true_x_m = np.asarray(traces["true_x_m"], dtype=float)
    gps_x_m = np.asarray(traces["gps_x_m"], dtype=float)
    gps_y_m = np.asarray(traces["gps_y_m"], dtype=float)

    ax_gps.plot(true_x_m, np.zeros_like(true_x_m), color="tab:green", label="true trajectory")
    ax_gps.scatter(gps_x_m, gps_y_m, s=8, color="tab:purple", label="GPS measured")
    ax_gps.set_xlabel("x [m]")
    ax_gps.set_ylabel("y [m]")
    ax_gps.set_title("GPS positions vs ground truth")
    ax_gps.set_aspect("equal", adjustable="datalim")
    ax_gps.grid(True, alpha=0.3)
    ax_gps.legend(loc="upper left", fontsize=8)

    lidar_points = np.asarray(traces["lidar_points_xy_m"], dtype=float)
    if lidar_points.size:
        ax_lidar.scatter(lidar_points[:, 0], lidar_points[:, 1], s=6, color="tab:cyan")
    ax_lidar.set_xlabel("x forward [m]")
    ax_lidar.set_ylabel("y left [m]")
    ax_lidar.set_title("Lidar last scan (sensor frame)")
    ax_lidar.set_aspect("equal", adjustable="datalim")
    ax_lidar.grid(True, alpha=0.3)

    for target_id, ranges in traces["radar_ranges"].items():
        ax_radar.plot(traces["radar_times"][target_id], ranges, marker=".", label=f"target {target_id}")
    ax_radar.set_xlabel("time [s]")
    ax_radar.set_ylabel("range [m]")
    ax_radar.set_title("Radar range per target")
    ax_radar.grid(True, alpha=0.3)
    ax_radar.legend(fontsize=8)

    ax_imu.plot(time_s, traces["imu_ax_mpss"], color="tab:red")
    ax_imu.set_xlabel("time [s]")
    ax_imu.set_ylabel("longitudinal acceleration [m/s^2]")
    ax_imu.set_title("IMU longitudinal acceleration")
    ax_imu.grid(True, alpha=0.3)

    ax_camera.plot(time_s, traces["camera_counts"], color="tab:orange")
    ax_camera.set_xlabel("time [s]")
    ax_camera.set_ylabel("detections [-]")
    ax_camera.set_title("Camera detection count")
    ax_camera.set_ylim(-0.2, 1.6)
    ax_camera.grid(True, alpha=0.3)

    ax_speed.plot(time_s, traces["ego_speed_mps"], color="tab:blue")
    ax_speed.set_xlabel("time [s]")
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.set_title("Ego speed (ground truth)")
    ax_speed.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    """Entry point: simulate the sensor suite, log JSONL and plot traces.

    The matplotlib backend is switched to "Agg" for headless runs before
    ``matplotlib.pyplot`` is imported inside :func:`plot_results`.

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

    config = load_yaml(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = run_simulation(config, seed=args.seed, jsonl_path=output_dir / "sensor_readings.jsonl")

    figure_path = output_dir / "sensor_trace.png"
    plot_results(traces, figure_path, args.show)

    simulation_cfg = config["simulation"]
    summary = {
        "duration_s": float(simulation_cfg["duration_s"]),
        "dt_s": float(simulation_cfg["dt_s"]),
        "n_readings": traces["n_readings"],
        "lidar_points_last_scan": traces["lidar_points_last_scan"],
        "radar_target_count": traces["radar_target_count"],
        "camera_detections_total": traces["camera_detections_total"],
    }
    summary_path = save_json(summary, output_dir / "run_summary.json")

    print("=== 02_sensor_simulation run summary ===")
    print(f"duration:        {summary['duration_s']:.1f} s (dt={summary['dt_s']:.3f} s)")
    print(f"jsonl readings:  {summary['n_readings']}")
    print(f"lidar points:    {summary['lidar_points_last_scan']} (last scan)")
    print(f"radar targets:   {summary['radar_target_count']}")
    print(f"camera detections total: {summary['camera_detections_total']}")
    print(f"figure:          {figure_path}")
    print(f"summary json:    {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
