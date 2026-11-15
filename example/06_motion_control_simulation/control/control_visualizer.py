"""Matplotlib rendering of closed-loop control response results.

The figure stacks three panels: speed tracking versus time, steering angle
versus time, and the planar path with the reference polyline plus the
tracked path coloured by the absolute cross-track error (with a colourbar).
Legends are placed outside the axes (``bbox_to_anchor``) where they would
otherwise overlap data.

``matplotlib.pyplot`` is imported lazily inside the plotting method so the
caller can fix the Matplotlib backend (headless ``Agg`` vs interactive)
before any figure machinery is initialised.

Extension point: add a fourth panel with the longitudinal acceleration
[m/s^2] and throttle [-] traces, or animate the run and export a video
for maneuver-quality reviews.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parents[1]


class ControlVisualizer:
    """Static rendering helpers for closed-loop control results."""

    @staticmethod
    def plot_response(
        result: dict,
        title: str,
        out_path: Path,
        show: bool = False,
    ) -> Path:
        """Render speed, steering and path panels for one simulation run.

        Args:
            result: Result dictionary produced by
                :meth:`control.simulated_vehicle_response.SimulatedVehicleResponse.run`
                with ``time`` [s], ``speed`` [m/s], ``target_speed`` [m/s],
                ``steering`` [rad], ``x`` [m], ``y`` [m],
                ``cross_track_error`` [m], ``reference_x`` [m] and
                ``reference_y`` [m] entries.
            title: Figure title [-].
            out_path: Destination ``.png`` path; relative paths are
                resolved against the module directory. Parent directories
                are created as needed.
            show: If ``True`` call ``matplotlib.pyplot.show()`` (interactive
                backends only); the figure is always closed afterwards.

        Returns:
            The resolved destination path of the written PNG file.
        """
        import matplotlib.pyplot as plt

        time = np.asarray(result["time"], dtype=float)
        speed = np.asarray(result["speed"], dtype=float)
        target_speed = np.asarray(result["target_speed"], dtype=float)
        steering = np.asarray(result["steering"], dtype=float)
        x = np.asarray(result["x"], dtype=float)
        y = np.asarray(result["y"], dtype=float)
        cross_track = np.abs(np.asarray(result["cross_track_error"], dtype=float))
        reference_x = np.asarray(result["reference_x"], dtype=float)
        reference_y = np.asarray(result["reference_y"], dtype=float)

        figure, (speed_axis, steering_axis, path_axis) = plt.subplots(
            3, 1, figsize=(9, 12)
        )

        speed_axis.plot(time, speed, color="tab:blue", linewidth=1.8, label="Tracked speed")
        speed_axis.plot(
            time,
            target_speed,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="Target speed",
        )
        speed_axis.set_ylabel("Speed [m/s]")
        speed_axis.grid(True, linestyle=":", alpha=0.6)
        speed_axis.set_title(title)
        speed_axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )

        steering_axis.plot(time, steering, color="tab:green", linewidth=1.5)
        steering_axis.axhline(0.0, color="0.6", linewidth=0.8)
        steering_axis.set_ylabel("Steering angle [rad]")
        steering_axis.set_xlabel("Time [s]")
        steering_axis.grid(True, linestyle=":", alpha=0.6)

        path_axis.plot(
            reference_x,
            reference_y,
            color="0.55",
            linestyle="--",
            linewidth=1.6,
            zorder=1,
            label="Reference path",
        )
        error_scatter = path_axis.scatter(
            x,
            y,
            c=cross_track,
            cmap="viridis",
            s=8,
            zorder=3,
            label="Tracked path",
        )
        path_axis.plot(
            x[0],
            y[0],
            marker="o",
            markersize=10,
            linestyle="",
            markerfacecolor="limegreen",
            markeredgecolor="black",
            zorder=5,
            label="Start",
        )
        colorbar = figure.colorbar(error_scatter, ax=path_axis, pad=0.02)
        colorbar.set_label("|cross-track error| [m]")
        path_axis.set_aspect("equal", adjustable="datalim")
        path_axis.set_xlabel("x [m]")
        path_axis.set_ylabel("y [m]")
        path_axis.grid(True, linestyle=":", alpha=0.6)
        # The colourbar occupies the right side of this panel, so the legend
        # stays inside the axes instead of using bbox_to_anchor.
        path_axis.legend(loc="upper left", fontsize=9)

        figure.tight_layout()

        destination = Path(out_path)
        if not destination.is_absolute():
            destination = _MODULE_DIR / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)
        return destination
