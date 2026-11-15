"""Trajectory plotting for sequences of vehicle states.

This module provides :class:`VehicleVisualizer`, a small plotting helper
that renders 2-D vehicle trajectories (position polyline, start and end
markers, heading arrows) and can be used both as a library from other
modules and as a standalone demo script over the shared sample dataset.

Extension point: replace the static matplotlib rendering with an
animated (``FuncAnimation``) or interactive (Plotly) backend; the public
``plot_trajectory`` signature is intentionally kept backend agnostic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[0]  # project root (this file sits one level below it)
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_basic_simulation"))

from shared.utilities.data_loader import load_csv_records
from vehicle.vehicle_state import VehicleState

SAMPLE_CSV_PATH = ROOT / "shared" / "sample_data" / "sample_vehicle_data.csv"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "plots"
DEFAULT_FIGURE_NAME = "sample_trajectory.png"


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


class VehicleVisualizer:
    """Render 2-D vehicle trajectories from state sequences.

    The visualizer accepts both :class:`VehicleState` objects and plain
    dictionaries (as produced by ``shared.utilities.data_loader``),
    so it can sit directly at the end of any simulation pipeline.
    """

    @staticmethod
    def _coerce_states(
        states: list[VehicleState] | list[dict[str, Any]],
    ) -> list[VehicleState]:
        """Normalize a state sequence into :class:`VehicleState` objects.

        Args:
            states: Sequence of :class:`VehicleState` instances or plain
                dictionaries accepted by ``VehicleState.from_dict``.

        Returns:
            List of :class:`VehicleState` objects in the original order.

        Raises:
            TypeError: If an element is neither ``VehicleState`` nor ``dict``.
        """
        coerced: list[VehicleState] = []
        for index, item in enumerate(states):
            if isinstance(item, VehicleState):
                coerced.append(item)
            elif isinstance(item, dict):
                coerced.append(VehicleState.from_dict(item))
            else:
                raise TypeError(
                    f"states[{index}] must be VehicleState or dict, "
                    f"got {type(item).__name__}"
                )
        return coerced

    def plot_trajectory(
        self,
        states: list[VehicleState] | list[dict[str, Any]],
        out_path: Path,
        show: bool = False,
        title: str = "Vehicle trajectory",
    ) -> Path:
        """Plot a 2-D trajectory with start/end markers and heading arrows.

        The figure shows the trajectory polyline, a green circle at the
        start, a red square at the end, and heading arrows (yaw in rad,
        0 rad = +x axis) drawn every ``max(1, n // 20)`` states.

        Args:
            states: Vehicle states (objects or dicts), ordered in time.
            out_path: Destination PNG path; parent folders are created.
            show: When True an interactive window is opened after saving.
            title: Figure title.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If ``states`` is empty.
        """
        plt = _get_pyplot(show)
        coerced = self._coerce_states(states)
        if not coerced:
            raise ValueError("states must contain at least one state")

        xs = np.asarray([state.x for state in coerced], dtype=float)
        ys = np.asarray([state.y for state in coerced], dtype=float)
        yaws = np.asarray([state.yaw for state in coerced], dtype=float)
        n_states = len(coerced)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.plot(xs, ys, color="tab:blue", linewidth=1.8, label="trajectory")
        ax.scatter(xs[0], ys[0], color="tab:green", marker="o", s=90,
                   zorder=3, label="start")
        ax.scatter(xs[-1], ys[-1], color="tab:red", marker="s", s=90,
                   zorder=3, label="end")

        step = max(1, n_states // 20)
        span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
        arrow_len = 0.05 * span
        ax.quiver(
            xs[::step],
            ys[::step],
            np.cos(yaws[::step]),
            np.sin(yaws[::step]),
            angles="xy",
            scale_units="xy",
            scale=1.0 / arrow_len,
            color="tab:orange",
            width=0.004,
            alpha=0.9,
            label="heading",
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
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


def _parse_args() -> argparse.Namespace:
    """Parse standalone CLI arguments.

    Returns:
        Parsed argument namespace with ``show``, ``output_dir`` and ``csv``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot the shared sample vehicle trajectory "
            "(position polyline, start/end markers and heading arrows)."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive figure window after saving.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the saved figure (default: 09_visualization/plots).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SAMPLE_CSV_PATH,
        help="Trajectory CSV to plot (default: shared/sample_data/sample_vehicle_data.csv).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the standalone demo: plot the shared sample trajectory.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args()
    records = load_csv_records(args.csv)
    visualizer = VehicleVisualizer()
    saved = visualizer.plot_trajectory(
        records,
        out_path=args.output_dir / DEFAULT_FIGURE_NAME,
        show=args.show,
        title="Sample vehicle trajectory",
    )
    print(f"Saved trajectory figure: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
