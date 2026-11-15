"""Top-down matplotlib schematics for driving decisions.

:class:`DecisionVisualizer` renders one environment snapshot plus the
winning decision as a top-down schematic: the ego vehicle as a blue
rectangle with a heading arrow, actors as markers, road features along
the ego heading line, and a decision text box for audit purposes.

The figure backend is deliberately NOT fixed here: ``main.py`` selects a
non-interactive backend (``Agg``) before ``matplotlib.pyplot`` is first
imported, so this module defers its pyplot import to first use.

Extension point: swap the static schematic for an animated player, add
predicted actor trajectories and sensor field-of-view overlays, or
export vector formats (PDF/SVG) for publications.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import List, Tuple

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from decision.environment_state import Actor, EnvironmentState, RoadFeature
from decision.scenario_evaluator import Decision


class DecisionVisualizer:
    """Render one environment snapshot and its decision as a schematic.

    All drawing happens in the local planning frame (metres, x forward,
    y left); headings are radians counter-clockwise from +x.
    """

    EGO_LENGTH_M: float = 4.4      # [m] drawn ego rectangle length, typical car 4-5
    EGO_WIDTH_M: float = 1.8       # [m] drawn ego rectangle width, typical car 1.6-2.0
    HEADING_ARROW_M: float = 7.0   # [m] heading arrow length ahead of ego
    LIMIT_PADDING_M: float = 15.0  # [m] axis padding around all drawn entities
    CROSSING_WIDTH_M: float = 6.0  # [m] drawn pedestrian crossing width across the road
    CROSSING_DEPTH_M: float = 1.2  # [m] drawn pedestrian crossing depth along the road
    REASON_WRAP_WIDTH: int = 56    # [-] characters per line in the decision text box

    def plot_scenario(
        self,
        env: EnvironmentState,
        decision: Decision,
        out_path: Path,
        show: bool = False,
    ) -> Path:
        """Draw one scenario snapshot plus its decision and save it.

        Args:
            env: Environment snapshot to draw (ego pose [m, rad, m/s],
                actors [m, m/s, rad], road features [m]).
            decision: Winning decision to annotate (action [-], target
                speed [m/s], confidence [-], reason [-]).
            out_path: Destination PNG path; parent directories are
                created automatically.
            show: When True, call ``plt.show()`` after saving (requires
                an interactive backend chosen by the caller).

        Returns:
            The resolved destination path.

        Extension point: add per-actor velocity vectors and predicted
        paths, or replace matplotlib with an interactive web viewer for
        dataset exploration.
        """
        import matplotlib.pyplot as plt  # deferred: caller must fix the backend first

        ego = env.ego
        heading_x = math.cos(ego.yaw)
        heading_y = math.sin(ego.yaw)

        fig, ax = plt.subplots(figsize=(8.0, 6.0))

        feature_points = self._draw_features(ax, env, heading_x, heading_y)
        self._draw_ego(ax, ego.x, ego.y, ego.yaw)
        self._draw_actors(ax, env.actors)

        self._set_limits(ax, ego, env.actors, feature_points)
        self._add_decision_box(ax, decision)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Decision: {decision.action} (confidence {decision.confidence:.2f})")
        ax.legend(
            handles=self._legend_handles(),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

        resolved_path = Path(out_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return resolved_path

    def _draw_ego(self, ax, x_m: float, y_m: float, yaw_rad: float) -> None:
        """Draw the ego rectangle and heading arrow.

        Args:
            ax: Target matplotlib axes.
            x_m: Ego x position [m].
            y_m: Ego y position [m].
            yaw_rad: Ego heading, counter-clockwise from +x [rad].
        """
        ego_rect = Rectangle(
            (x_m - self.EGO_LENGTH_M / 2.0, y_m - self.EGO_WIDTH_M / 2.0),
            self.EGO_LENGTH_M,
            self.EGO_WIDTH_M,
            facecolor="tab:blue",
            edgecolor="black",
            zorder=4,
        )
        rotation = Affine2D().rotate_deg_around(
            x_m, y_m, math.degrees(yaw_rad)
        ) + ax.transData
        ego_rect.set_transform(rotation)
        ax.add_patch(ego_rect)

        ax.annotate(
            "",
            xy=(
                x_m + self.HEADING_ARROW_M * math.cos(yaw_rad),
                y_m + self.HEADING_ARROW_M * math.sin(yaw_rad),
            ),
            xytext=(x_m, y_m),
            arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.0),
        )

    def _draw_actors(self, ax, actors: List[Actor]) -> None:
        """Draw dynamic actors.

        Args:
            ax: Target matplotlib axes.
            actors: Actors to draw (positions [m]).
        """
        for actor in actors:
            if actor.type == "vehicle":
                ax.scatter(
                    actor.x, actor.y, s=140, c="tab:red",
                    edgecolors="black", marker="o", zorder=4,
                )
            elif actor.type == "pedestrian":
                ax.scatter(
                    actor.x, actor.y, s=140, c="tab:orange",
                    edgecolors="black", marker="^", zorder=4,
                )

    def _draw_features(
        self,
        ax,
        env: EnvironmentState,
        heading_x: float,
        heading_y: float,
    ) -> List[Tuple[float, float]]:
        """Draw road features along the ego heading line.

        Args:
            ax: Target matplotlib axes.
            env: Environment snapshot providing the features.
            heading_x: Cosine of the ego heading [-].
            heading_y: Sine of the ego heading [-].

        Returns:
            List of ``(x, y)`` positions [m] where features were drawn,
            used later for axis-limit computation.
        """
        points: List[Tuple[float, float]] = []
        for feature in env.features:
            fx = env.ego.x + feature.distance_m * heading_x
            fy = env.ego.y + feature.distance_m * heading_y
            points.append((fx, fy))
            if feature.type == "traffic_light":
                ax.scatter(
                    fx, fy, s=180, c="gold",
                    edgecolors="black", marker="s", zorder=4,
                )
            elif feature.type == "pedestrian_crossing":
                bar = Rectangle(
                    (fx - self.CROSSING_DEPTH_M / 2.0, fy - self.CROSSING_WIDTH_M / 2.0),
                    self.CROSSING_DEPTH_M,
                    self.CROSSING_WIDTH_M,
                    facecolor="lightgray",
                    edgecolor="gray",
                    hatch="///",
                    zorder=2,
                )
                bar.set_transform(
                    Affine2D().rotate_deg_around(fx, fy, math.degrees(env.ego.yaw))
                    + ax.transData
                )
                ax.add_patch(bar)
            elif feature.type == "speed_limit":
                ax.scatter(
                    fx, fy, s=200, facecolors="white",
                    edgecolors="black", marker="o", zorder=4,
                )
                ax.text(
                    fx, fy, str(feature.state),
                    ha="center", va="center", fontsize=7, zorder=5,
                )
        return points

    def _set_limits(self, ax, ego, actors: List[Actor],
                    feature_points: List[Tuple[float, float]]) -> None:
        """Fix axis limits around every drawn entity with padding.

        Args:
            ax: Target matplotlib axes.
            ego: Ego vehicle state (position [m]).
            actors: Actors contributing to the extent (positions [m]).
            feature_points: Feature positions contributing to the extent [m].
        """
        xs = [ego.x] + [actor.x for actor in actors] + [p[0] for p in feature_points]
        ys = [ego.y] + [actor.y for actor in actors] + [p[1] for p in feature_points]
        ax.set_xlim(min(xs) - self.LIMIT_PADDING_M, max(xs) + self.LIMIT_PADDING_M)
        ax.set_ylim(min(ys) - self.LIMIT_PADDING_M, max(ys) + self.LIMIT_PADDING_M)
        ax.set_aspect("equal", adjustable="box")

    def _add_decision_box(self, ax, decision: Decision) -> None:
        """Add the decision audit text box to the bottom-left corner.

        Args:
            ax: Target matplotlib axes.
            decision: Decision to annotate (action, target speed [m/s],
                reason).
        """
        target_text = (
            "n/a"
            if decision.target_speed_mps is None
            else f"{decision.target_speed_mps:.2f}"
        )
        reason_text = textwrap.fill(decision.reason, width=self.REASON_WRAP_WIDTH)
        ax.text(
            0.02,
            0.02,
            f"action={decision.action}, target={target_text} m/s\n{reason_text}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="0.6",
                alpha=0.9,
            ),
            zorder=6,
        )

    def _legend_handles(self) -> List[Line2D]:
        """Build constant legend proxy handles for every entity class.

        Returns:
            List of proxy :class:`Line2D` handles covering ego, actors
            and road features.
        """
        return [
            Line2D([0], [0], marker="s", linestyle="None", markersize=10,
                   markerfacecolor="tab:blue", markeredgecolor="black",
                   label="ego vehicle"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=10,
                   markerfacecolor="tab:red", markeredgecolor="black",
                   label="vehicle actor"),
            Line2D([0], [0], marker="^", linestyle="None", markersize=10,
                   markerfacecolor="tab:orange", markeredgecolor="black",
                   label="pedestrian actor"),
            Line2D([0], [0], marker="s", linestyle="None", markersize=10,
                   markerfacecolor="gold", markeredgecolor="black",
                   label="traffic light"),
            Line2D([0], [0], marker="s", linestyle="None", markersize=10,
                   markerfacecolor="lightgray", markeredgecolor="gray",
                   label="pedestrian crossing"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=10,
                   markerfacecolor="white", markeredgecolor="black",
                   label="speed limit"),
        ]
