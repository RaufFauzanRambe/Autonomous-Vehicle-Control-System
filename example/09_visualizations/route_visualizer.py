"""Bare waypoint route plotting.

:class:`RouteVisualizer` in this module renders plain waypoint
polylines (an ``(N, 2)`` NumPy array of planar positions). It is
deliberately map-agnostic and therefore distinct from the map-aware
visualizer in ``05_path_planning_demo/planning/route_visualizer.py``
(``RouteVisualizer.plot_map_with_route``), which additionally draws the
road-graph nodes and links around a planned route. Use this module when
only a waypoint list is available, e.g. after smoothing a planned path.

Extension point: add curvature or arc-length color scales, and annotate
per-segment speed limits once waypoint metadata becomes available.
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
DEFAULT_FIGURE_NAME = "route_demo.png"


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


class RouteVisualizer:
    """Render bare waypoint polylines without any map context."""

    @staticmethod
    def plot_route(
        waypoints: np.ndarray,
        out_path: Path,
        show: bool = False,
        title: str = "Planned route",
    ) -> Path:
        """Plot a waypoint polyline with start/goal annotations.

        The figure shows the waypoint polyline with point markers, a
        green ``start`` annotation at the first waypoint and a magenta
        ``goal`` annotation at the last waypoint.

        Args:
            waypoints: ``(N, 2)`` array of planar waypoints [m], with
                ``N >= 2``; the row order defines the route direction.
            out_path: Destination PNG path; parent folders are created.
            show: When True an interactive window is opened after saving.
            title: Figure title.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If ``waypoints`` has fewer than 2 rows or is not
                of shape ``(N, 2)``.
        """
        plt = _get_pyplot(show)
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"waypoints must have shape (N, 2), got {points.shape}")
        if points.shape[0] < 2:
            raise ValueError("waypoints must contain at least 2 points")

        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        ax.plot(points[:, 0], points[:, 1], "-", color="tab:blue",
                linewidth=1.8, label="waypoint polyline")
        ax.plot(points[:, 0], points[:, 1], ".", color="tab:blue", markersize=5)
        ax.scatter(points[0, 0], points[0, 1], color="tab:green", marker="o",
                   s=90, zorder=3)
        ax.annotate("start", (points[0, 0], points[0, 1]),
                    textcoords="offset points", xytext=(6, 6),
                    color="tab:green")
        ax.scatter(points[-1, 0], points[-1, 1], color="magenta", marker="*",
                   s=140, zorder=3)
        ax.annotate("goal", (points[-1, 0], points[-1, 1]),
                    textcoords="offset points", xytext=(6, 6),
                    color="magenta")

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
        Parsed argument namespace with ``show`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot a synthetic S-curve waypoint polyline (bare waypoints, "
            "no road map) and save it as a PNG figure."
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
    return parser.parse_args()


def main() -> int:
    """Run the standalone demo: plot a synthetic S-curve route.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args()
    x = np.linspace(0.0, 100.0, 50)
    y = 10.0 * np.sin(x / 15.0)
    waypoints = np.column_stack([x, y])
    saved = RouteVisualizer.plot_route(
        waypoints,
        out_path=args.output_dir / DEFAULT_FIGURE_NAME,
        show=args.show,
        title="Planned S-curve route",
    )
    print(f"Saved route figure: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
