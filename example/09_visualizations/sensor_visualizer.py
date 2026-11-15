"""Synthetic sensor plotting helpers (lidar, GPS, radar).

:class:`SensorVisualizer` groups three static plotting functions used to
inspect sensor simulation output from module ``02_sensor_simulation``:

* ``plot_lidar_scan`` renders a lidar point cloud given in the sensor
  frame, transformed into the world frame by an ego pose.
* ``plot_gps_noise`` overlays a noisy GPS track on its ground truth.
* ``plot_radar_ranges`` plots range-over-time histories per target.

The standalone demo generates seeded synthetic datasets for all three
plotters so the module runs without any input files.

Extension point: add covariance ellipses, occupancy-grid overlays, or a
bird's-eye-view renderer for camera detections to this class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[0]  # project root (this file sits one level below it)
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = MODULE_DIR / "plots"


def _get_pyplot(show: bool):
    """Return the ``matplotlib.pyplot`` module with a suitable backend.

    Args:
        show: True when an interactive window was requested on the CLI.
            When False, the headless ``Agg`` backend is selected before
            pyplot is imported so the code also runs on display-less
            machines and inside test runners.

    Returns:
        The imported ``matplotlib.pyplot`` module.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class SensorVisualizer:
    """Static plotting helpers for synthetic sensor output.

    All methods follow the same save convention: figures are written at
    ``dpi=150``, the figure is closed afterwards, and the resolved output
    path is returned.
    """

    @staticmethod
    def plot_lidar_scan(
        points_xy: np.ndarray,
        ego_pose: tuple[float, float, float],
        out_path: Path,
        show: bool = False,
    ) -> Path:
        """Plot a lidar scan in the world frame.

        Sensor-frame points ``(px, py)`` are rotated by ``yaw`` and
        translated by ``(x, y)`` of the ego pose:
        ``world = R(yaw) @ point + (x, y)``.

        Args:
            points_xy: ``(N, 2)`` array of sensor-frame points [m].
            ego_pose: Ego ``(x [m], y [m], yaw [rad])``; yaw is CCW
                positive with 0 rad pointing along the +x axis.
            out_path: Destination PNG path; parent folders are created.
            show: When True an interactive window is opened after saving.

        Returns:
            The resolved output path.
        """
        plt = _get_pyplot(show)
        points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        x_ego, y_ego, yaw = (float(value) for value in ego_pose)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        world_x = x_ego + cos_yaw * points[:, 0] - sin_yaw * points[:, 1]
        world_y = y_ego + sin_yaw * points[:, 0] + cos_yaw * points[:, 1]

        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        ax.scatter(world_x, world_y, s=6, color="tab:blue",
                   label="lidar points (world frame)")
        ax.scatter([x_ego], [y_ego], marker="^", color="black", s=90,
                   zorder=3, label="ego")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Lidar scan in world frame")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best")

        resolved = Path(out_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return resolved

    @staticmethod
    def plot_gps_noise(
        true_xy: np.ndarray,
        measured_xy: np.ndarray,
        out_path: Path,
        show: bool = False,
    ) -> Path:
        """Plot a noisy GPS track against its ground-truth trajectory.

        Args:
            true_xy: ``(N, 2)`` array of ground-truth positions [m].
            measured_xy: ``(N, 2)`` array of measured (noisy) positions [m].
            out_path: Destination PNG path; parent folders are created.
            show: When True an interactive window is opened after saving.

        Returns:
            The resolved output path.
        """
        plt = _get_pyplot(show)
        truth = np.asarray(true_xy, dtype=float).reshape(-1, 2)
        measured = np.asarray(measured_xy, dtype=float).reshape(-1, 2)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.plot(truth[:, 0], truth[:, 1], "k--", linewidth=1.5,
                label="ground truth")
        ax.plot(measured[:, 0], measured[:, 1], ".", color="0.55",
                markersize=4, label="GPS measured")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("GPS measurement noise vs ground truth")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best")

        resolved = Path(out_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return resolved

    @staticmethod
    def plot_radar_ranges(
        ranges_by_target: dict[str, np.ndarray],
        time_s: np.ndarray,
        out_path: Path,
        show: bool = False,
    ) -> Path:
        """Plot radar range histories, one line per target.

        Args:
            ranges_by_target: Mapping from target label to range samples [m].
            time_s: ``(N,)`` array of sample times [s]; every range array
                must have the same length as ``time_s``.
            out_path: Destination PNG path; parent folders are created.
            show: When True an interactive window is opened after saving.

        Returns:
            The resolved output path.
        """
        plt = _get_pyplot(show)
        time = np.asarray(time_s, dtype=float)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for name in sorted(ranges_by_target):
            ranges = np.asarray(ranges_by_target[name], dtype=float)
            ax.plot(time, ranges, marker=".", markersize=3, linewidth=1.0,
                    label=name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("range [m]")
        ax.set_title("Radar range histories per target")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()

        resolved = Path(out_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return resolved


def _parse_args() -> argparse.Namespace:
    """Parse standalone CLI arguments.

    Returns:
        Parsed argument namespace with ``show``, ``output_dir`` and ``seed``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate seeded synthetic lidar, GPS and radar datasets and "
            "save one demo figure per sensor."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive figure windows after saving.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the saved figures (default: 09_visualization/plots).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the synthetic sensor data (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the standalone demo: save one figure per synthetic sensor set.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    # Lidar: circular scan of ~30 m radius around a non-trivial ego pose.
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    radii = 30.0 + rng.normal(0.0, 0.5, size=angles.shape)
    points_sensor_frame = np.column_stack(
        [radii * np.cos(angles), radii * np.sin(angles)]
    )
    ego_pose = (50.0, 25.0, 0.3)
    saved_lidar = SensorVisualizer.plot_lidar_scan(
        points_sensor_frame,
        ego_pose,
        args.output_dir / "sensor_lidar_demo.png",
        show=args.show,
    )

    # GPS: 100 samples of a straight 5 m/s path with 1.5 m sigma noise.
    sample_times = np.arange(100.0)
    true_xy = np.column_stack([5.0 * sample_times, np.zeros_like(sample_times)])
    measured_xy = true_xy + rng.normal(0.0, 1.5, size=true_xy.shape)
    saved_gps = SensorVisualizer.plot_gps_noise(
        true_xy,
        measured_xy,
        args.output_dir / "sensor_gps_demo.png",
        show=args.show,
    )

    # Radar: two targets at ~40 m and ~70 m range over 100 s of samples.
    time_s = np.linspace(0.0, 100.0, 100)
    ranges_by_target = {
        "target_a": 40.0 + rng.normal(0.0, 0.4, size=time_s.shape),
        "target_b": 70.0 + rng.normal(0.0, 0.4, size=time_s.shape),
    }
    saved_radar = SensorVisualizer.plot_radar_ranges(
        ranges_by_target,
        time_s,
        args.output_dir / "sensor_radar_demo.png",
        show=args.show,
    )

    for saved_path in (saved_lidar, saved_gps, saved_radar):
        print(f"Saved sensor figure: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
