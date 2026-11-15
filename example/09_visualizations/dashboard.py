"""Run-artifact dashboard across the example modules.

:class:`VisualizationDashboard` scans the ``outputs/`` and ``results/``
folders of modules ``01_basic_simulation`` .. ``08_full_system_demo``
under a configurable root and renders a single 2x2 PNG summary:

* PNG artifact counts per module (bar chart, zero for missing folders),
* the newest result-summary file per module with its modification time,
* total artifact statistics,
* guidance on how to populate the dashboard.

The dashboard is read-only with respect to the other modules and works
on a fresh checkout where no module has ever produced outputs.

Extension point: publish the same summary as an interactive HTML page
(Plotly / Dash) or push the metrics to a monitoring system instead of
rendering a static figure.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[0]  # project root (this file sits one level below it)
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = MODULE_DIR / "plots"
DEFAULT_FIGURE_NAME = "dashboard.png"

MONITORED_MODULES: tuple[str, ...] = (
    "01_basic_simulation",
    "02_sensor_simulation",
    "03_perception_demo",
    "04_localization_demo",
    "05_path_planning_demo",
    "06_motion_control_simulation",
    "07_ai_decision_demo",
    "08_full_system_demo",
)
RESULT_FILE_PATTERNS: tuple[str, ...] = ("run_summary.json", "*_results.json")
ARTIFACT_SUBDIRS: tuple[str, ...] = ("outputs", "results")


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


class VisualizationDashboard:
    """Aggregate and render run artifacts of the example modules.

    The dashboard never writes outside its own output figure and never
    modifies other module folders; it only reads them.
    """

    def __init__(self, outputs_root: Path) -> None:
        """Store the root under which module folders are scanned.

        Args:
            outputs_root: Repository root containing the module folders
                ``01_basic_simulation`` .. ``08_full_system_demo``.
        """
        self.outputs_root = Path(outputs_root).resolve()

    def collect(self) -> dict[str, Any]:
        """Scan module output folders and summarize artifacts.

        For every monitored module the ``outputs/`` and ``results/``
        sub-folders (either may be absent) are searched for PNG files
        and for result-summary JSON files (``run_summary.json`` and
        ``*_results.json``). Missing folders contribute zero counts and
        no result entry.

        Returns:
            Dictionary with keys:

            * ``outputs_root``: the resolved scan root as ``str``.
            * ``modules``: mapping of module name to a dictionary with
              ``png_count`` (int), ``result_file`` (str or None) and
              ``result_mtime_iso`` (ISO 8601 string or None).
        """
        modules: dict[str, dict[str, Any]] = {}
        for name in MONITORED_MODULES:
            png_count = 0
            newest_mtime: float | None = None
            newest_file: Path | None = None
            for sub_dir_name in ARTIFACT_SUBDIRS:
                folder = self.outputs_root / name / sub_dir_name
                if not folder.is_dir():
                    continue
                png_count += sum(1 for png in folder.glob("*.png") if png.is_file())
                for pattern in RESULT_FILE_PATTERNS:
                    for candidate in sorted(folder.glob(pattern)):
                        if not candidate.is_file():
                            continue
                        mtime = candidate.stat().st_mtime
                        if newest_mtime is None or mtime > newest_mtime:
                            newest_mtime = mtime
                            newest_file = candidate
            modules[name] = {
                "png_count": png_count,
                "result_file": newest_file.name if newest_file is not None else None,
                "result_mtime_iso": (
                    datetime.fromtimestamp(newest_mtime).isoformat(timespec="seconds")
                    if newest_mtime is not None
                    else None
                ),
            }
        return {"outputs_root": str(self.outputs_root), "modules": modules}

    def build(self, out_path: Path, show: bool = False) -> Path:
        """Render the 2x2 dashboard figure.

        Args:
            out_path: Destination PNG path; parent folders are created.
            show: When True an interactive window is opened after saving.

        Returns:
            The resolved output path.
        """
        plt = _get_pyplot(show)
        summary = self.collect()
        modules: dict[str, dict[str, Any]] = summary["modules"]
        names = list(modules.keys())
        counts = [int(modules[name]["png_count"]) for name in names]

        fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
        (ax_bars, ax_results), (ax_totals, ax_help) = axes

        # (0, 0): PNG artifacts per module.
        ax_bars.bar(range(len(names)), counts, color="tab:blue")
        ax_bars.set_xticks(range(len(names)))
        ax_bars.set_xticklabels([name.split("_", 1)[0] for name in names])
        ax_bars.set_ylabel("PNG file count")
        ax_bars.set_title("PNG artifacts per module")
        ax_bars.grid(True, axis="y", alpha=0.4)
        ax_bars.set_ylim(0, max(1, max(counts) * 1.2))

        # (0, 1): newest result summary per module.
        result_lines: list[str] = []
        for name in names:
            info = modules[name]
            if info["result_mtime_iso"] is None:
                result_lines.append(f"{name:32s} no results yet")
            else:
                result_lines.append(
                    f"{name:32s} {info['result_file']} @ {info['result_mtime_iso']}"
                )
        ax_results.text(
            0.0, 1.0, "\n".join(result_lines), va="top", ha="left",
            family="monospace", fontsize=8, transform=ax_results.transAxes,
        )
        ax_results.set_title("Newest result summary per module")
        ax_results.axis("off")

        # (1, 0): total artifact statistics.
        total_pngs = sum(counts)
        modules_with_artifacts = sum(1 for count in counts if count > 0)
        totals_text = (
            f"Outputs root: {self.outputs_root}\n"
            f"Total PNG artifacts: {total_pngs}\n"
            f"Modules with PNG artifacts: {modules_with_artifacts} / {len(names)}\n"
            f"Generated: {datetime.now().isoformat(timespec='seconds')}"
        )
        ax_totals.text(
            0.0, 1.0, totals_text, va="top", ha="left",
            family="monospace", fontsize=10, transform=ax_totals.transAxes,
        )
        ax_totals.set_title("Artifact totals")
        ax_totals.axis("off")

        # (1, 1): guidance for populating the dashboard.
        help_text = (
            "How to populate this dashboard\n\n"
            "Run <module>/main.py to populate outputs, e.g.:\n"
            "  python 01_basic_simulation/main.py\n"
            "  python 08_full_system_demo/main.py\n\n"
            "Then re-run:\n"
            "  python 09_visualization/dashboard.py"
        )
        ax_help.text(
            0.0, 1.0, help_text, va="top", ha="left",
            family="monospace", fontsize=10, transform=ax_help.transAxes,
        )
        ax_help.set_title("Guidance")
        ax_help.axis("off")

        fig.suptitle("Autonomous Vehicle Control System - example suite artifact dashboard")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

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
        Parsed argument namespace with ``outputs_root``, ``output_dir``
        and ``show``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Scan module outputs/results folders of the example suite and "
            "render a single dashboard PNG."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT,
        help="Root containing the module folders (default: project root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the dashboard PNG (default: 09_visualization/plots).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window after saving.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the standalone demo: build and save the dashboard PNG.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args()
    dashboard = VisualizationDashboard(args.outputs_root)
    summary = dashboard.collect()
    total_pngs = sum(info["png_count"] for info in summary["modules"].values())
    print(
        f"Scanned {len(summary['modules'])} modules under "
        f"{summary['outputs_root']}: {total_pngs} PNG artifacts found."
    )
    saved = dashboard.build(args.output_dir / DEFAULT_FIGURE_NAME, show=args.show)
    print(f"Saved dashboard figure: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
