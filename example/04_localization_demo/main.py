"""Localization demo: GPS low-pass filtering, IMU dead reckoning, complementary fusion.

The demo replays the recorded log ``sample_data/sample_position.csv``
(raw GPS fixes with ground truth and a biased IMU), runs the three
estimators in lockstep and evaluates 2-D position RMSE against ground
truth. All paths are resolved relative to this module directory, never
to the current working directory. The pipeline is fully deterministic;
``--seed`` is accepted for CLI symmetry with the other modules.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[0]  # project root, holds shared/ utilities
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))  # shared utilities: from shared.utilities.data_loader import load_csv

from localization.gps_processor import GPSProcessor  # noqa: E402
from localization.imu_processor import IMUProcessor  # noqa: E402
from localization.sensor_fusion import SensorFusion  # noqa: E402
from shared.utilities.config_loader import load_yaml  # noqa: E402
from shared.utilities.data_loader import load_csv, save_json  # noqa: E402

CONFIG_PATH = MODULE_DIR / "config" / "localization_config.yaml"
DATASET_PATH = MODULE_DIR / "sample_data" / "sample_position.csv"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs"
TRUE_SPEED_MPS = 5.0  # [m/s] ground truth of the log: constant speed along +x


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Parser with ``--config``, ``--show``, ``--output-dir`` and
        ``--seed`` options.
    """
    parser = argparse.ArgumentParser(
        description="Run the GPS/IMU localization demo on the recorded sample log."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to the module YAML configuration file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open matplotlib windows in addition to saving files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the trace figure and run_summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (accepted for CLI symmetry; the replay pipeline is deterministic).",
    )
    return parser


def estimate_initial_speed(t_s: np.ndarray, gps_x: np.ndarray, gps_y: np.ndarray) -> float:
    """Estimate the initial speed from GPS displacement over a time baseline.

    The documented trick is ``v0 = hypot(dx, dy) / dt`` from GPS
    displacement. Applied to only the first two fixes, its standard
    deviation is ``sqrt(2) * sigma_gps / dt`` (about 15 m/s for this log
    with sigma_gps near 1.2 m and dt = 0.1 s), which is unusable. The
    same displacement-over-time trick applied over the full logged
    baseline divides the fix noise by the whole interval, giving a speed
    error of roughly ``sqrt(2) * sigma_gps / (t_end - t_start)`` (about
    0.3 m/s here). The demo therefore uses the full-baseline estimate.

    Args:
        t_s: Sample timestamps [s], length n.
        gps_x: Raw GPS x coordinates [m], length n.
        gps_y: Raw GPS y coordinates [m], length n.

    Returns:
        Estimated initial speed [m/s] as the GPS displacement magnitude
        divided by the elapsed time.
    """
    elapsed_s = float(t_s[-1] - t_s[0])
    displacement_m = math.hypot(float(gps_x[-1] - gps_x[0]), float(gps_y[-1] - gps_y[0]))
    return displacement_m / elapsed_s


def position_rmse(estimates: np.ndarray, reference: np.ndarray) -> float:
    """Compute 2-D position RMSE over all samples.

    Args:
        estimates: Array ``(n, 2)`` of estimated positions [m].
        reference: Array ``(n, 2)`` of reference positions [m].

    Returns:
        RMSE [m] defined as the square root of the mean squared Euclidean
        distance over all samples.
    """
    squared_distances = np.sum((estimates - reference) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distances)))


def _save_trace_figure(
    t_s: np.ndarray,
    gps_raw: np.ndarray,
    gps_filtered: np.ndarray,
    fused: np.ndarray,
    ground_truth: np.ndarray,
    speed_mps: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    """Write the three-panel localization trace figure.

    Panel 1 shows trajectories (raw GPS, filtered GPS, fused, ground
    truth), panel 2 the 2-D position error over time for raw GPS and the
    fused estimate, panel 3 the estimated speed from the IMU processor
    against the true constant speed.

    Args:
        t_s: Timestamps [s], length n.
        gps_raw: Raw GPS positions [m], ``(n, 2)``.
        gps_filtered: Low-pass filtered GPS positions [m], ``(n, 2)``.
        fused: Fused positions [m], ``(n, 2)``.
        ground_truth: Reference positions [m], ``(n, 2)``.
        speed_mps: Estimated speed [m/s] from the IMU velocity, length n.
        output_path: Destination PNG path.
        show: When True, open an interactive window after saving.
    """
    import matplotlib.pyplot as plt  # imported after the backend decision in main()

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), constrained_layout=True)

    ax = axes[0]
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], "k-", lw=2.0, label="ground truth")
    ax.plot(gps_raw[:, 0], gps_raw[:, 1], ".", ms=4, color="0.6", label="raw GPS")
    ax.plot(
        gps_filtered[:, 0],
        gps_filtered[:, 1],
        "--",
        color="tab:orange",
        lw=1.5,
        label="filtered GPS (low-pass)",
    )
    ax.plot(fused[:, 0], fused[:, 1], "-", color="tab:blue", lw=1.5, label="fused (complementary)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectories")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1]
    gps_error_m = np.linalg.norm(gps_raw - ground_truth, axis=1)
    fused_error_m = np.linalg.norm(fused - ground_truth, axis=1)
    ax.plot(t_s, gps_error_m, color="0.4", lw=1.5, label="raw GPS error")
    ax.plot(t_s, fused_error_m, color="tab:blue", lw=1.5, label="fused error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("2-D position error vs time")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t_s, speed_mps, color="tab:green", lw=1.5, label="estimated speed (IMU |v|)")
    ax.axhline(TRUE_SPEED_MPS, color="k", ls="--", lw=1.0, label=f"true speed ({TRUE_SPEED_MPS} m/s)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Estimated speed vs time")
    ax.grid(True)
    ax.legend(loc="best")

    fig.suptitle("GPS / IMU complementary fusion on the recorded log")
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the localization demo end to end.

    Flow: parse arguments -> load config -> load the CSV log -> estimate
    the initial speed from GPS displacement -> reset the estimators ->
    iterate the samples through GPS filter, IMU dead reckoning and
    fusion -> evaluate RMSE metrics against ground truth -> write the
    trace figure and run_summary.json -> print a console summary.

    Args:
        argv: Optional command-line arguments (defaults to ``sys.argv``).

    Returns:
        Process exit code, 0 on success.
    """
    args = build_parser().parse_args(argv)
    if not args.show:
        # Headless backend must be selected before pyplot is imported.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (backend decision precedes this import)

    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_csv(DATASET_PATH)
    t_s = frame["timestamp"].to_numpy(dtype=float)
    gps_x = frame["gps_x"].to_numpy(dtype=float)
    gps_y = frame["gps_y"].to_numpy(dtype=float)
    ground_truth_x = frame["ground_truth_x"].to_numpy(dtype=float)
    ground_truth_y = frame["ground_truth_y"].to_numpy(dtype=float)
    imu_ax = frame["imu_ax"].to_numpy(dtype=float)
    imu_ay = frame["imu_ay"].to_numpy(dtype=float)
    imu_yaw_rate = frame["imu_yaw_rate"].to_numpy(dtype=float)
    n_samples = len(t_s)
    ground_truth = np.column_stack([ground_truth_x, ground_truth_y])

    gps_processor = GPSProcessor(config["gps"])
    imu_processor = IMUProcessor(config["imu"])
    fusion = SensorFusion(config["fusion"])

    # Initial-speed estimation from GPS displacement (see
    # estimate_initial_speed for why the full baseline is used), then
    # seed both estimators with the first GPS fix.
    v0_mps = estimate_initial_speed(t_s, gps_x, gps_y)
    imu_processor.reset(initial_velocity=(v0_mps, 0.0))
    fusion.reset(initial_position=(float(gps_x[0]), float(gps_y[0])))

    gps_raw = np.column_stack([gps_x, gps_y])
    gps_filtered = np.zeros((n_samples, 2), dtype=float)
    fused = np.zeros((n_samples, 2), dtype=float)
    speed_mps = np.zeros(n_samples, dtype=float)

    for k in range(n_samples):
        # The first sample initializes the filters; no time has elapsed.
        dt_s = 0.0 if k == 0 else float(t_s[k] - t_s[k - 1])
        filtered = gps_processor.process((gps_x[k], gps_y[k]))
        imu_dxy = imu_processor.update(imu_ax[k], imu_ay[k], imu_yaw_rate[k], dt_s)
        fused_position = fusion.update((gps_x[k], gps_y[k]), imu_dxy, dt_s)
        gps_filtered[k, :] = filtered
        fused[k, :] = fused_position
        speed_mps[k] = math.hypot(imu_processor.velocity[0], imu_processor.velocity[1])

    gps_rmse_m = position_rmse(gps_raw, ground_truth)
    filtered_gps_rmse_m = position_rmse(gps_filtered, ground_truth)
    fused_rmse_m = position_rmse(fused, ground_truth)
    improvement_pct = 100.0 * (1.0 - fused_rmse_m / gps_rmse_m)
    final_drift_m = math.hypot(
        float(fused[-1, 0] - ground_truth_x[-1]),
        float(fused[-1, 1] - ground_truth_y[-1]),
    )

    _save_trace_figure(
        t_s,
        gps_raw,
        gps_filtered,
        fused,
        ground_truth,
        speed_mps,
        output_dir / "localization_trace.png",
        show=args.show,
    )

    summary_values = {
        "n_samples": n_samples,
        "gps_rmse_m": gps_rmse_m,
        "filtered_gps_rmse_m": filtered_gps_rmse_m,
        "fused_rmse_m": fused_rmse_m,
        "improvement_pct": improvement_pct,
        "final_drift_m": final_drift_m,
        "estimated_initial_speed_mps": v0_mps,
    }
    run_summary = {
        "n_samples": n_samples,
        "gps_rmse_m": gps_rmse_m,
        "filtered_gps_rmse_m": filtered_gps_rmse_m,
        "fused_rmse_m": fused_rmse_m,
        "improvement_pct": improvement_pct,
        "final_drift_m": final_drift_m,
    }
    summary_path = save_json(run_summary, output_dir / "run_summary.json")

    print("Localization demo summary")
    print(f"  samples: {n_samples} (log duration {t_s[-1] - t_s[0]:.1f} s)")
    print(f"  estimated initial speed: {v0_mps:.3f} m/s (true {TRUE_SPEED_MPS:.1f} m/s)")
    for key in config["evaluation"]["metrics"]:
        print(f"  {key}: {summary_values[key]:.4f}")
    print(f"  filtered_gps_rmse_m: {filtered_gps_rmse_m:.4f}")
    print(f"  run summary written to: {summary_path}")
    print(f"  outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
