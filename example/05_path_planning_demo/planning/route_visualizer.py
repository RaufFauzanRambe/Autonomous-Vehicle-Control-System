"""Matplotlib rendering of road-graph maps with planned routes.

The visualiser draws the raw road network (gray centerlines), the planned
route (thick red polyline through the route nodes), all nodes with small
identifier labels and dedicated start/goal markers. Legends are placed
outside the axes (``bbox_to_anchor``) so they never cover map geometry.

``matplotlib.pyplot`` is imported lazily inside the plotting method so the
caller can fix the Matplotlib backend (headless ``Agg`` vs interactive)
before any figure machinery is initialised.

Extension point: add link attributes to the visual encoding, e.g. colour
edges by speed limit [m/s], width edges by lane count [-], or overlay the
resampled waypoints of :meth:`AStarPlanner.waypoints_from_path` as small
crosses along the route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from planning.map_loader import MapLink, MapNode

_MODULE_DIR = Path(__file__).resolve().parents[1]


class RouteVisualizer:
    """Static rendering helpers for road-graph maps and planned routes."""

    @staticmethod
    def plot_map_with_route(
        nodes: List[MapNode],
        links: List[MapLink],
        path_ids: List[str],
        title: str,
        out_path: Path,
        show: bool = False,
        show_nodes: bool = True,
        node_label_size: int = 6,
    ) -> Path:
        """Render a road map and the planned route, and save it as PNG.

        The start and goal of the route are inferred from the first and
        last entries of ``path_ids``; when ``path_ids`` is empty (unreachable
        goal) the map is still rendered, without route line and without
        start/goal markers.

        Args:
            nodes: Road-graph nodes with coordinates [m].
            links: Road-graph links; drawn as gray centerlines.
            path_ids: Ordered node ids [-] of the planned route; may be
                empty.
            title: Figure title [-].
            out_path: Destination ``.png`` path; relative paths are
                resolved against the module directory. Parent directories
                are created as needed.
            show: If ``True`` call ``matplotlib.pyplot.show()`` (interactive
                backends only); the figure is always closed afterwards.
            show_nodes: If ``True`` draw node dots and id labels.
            node_label_size: Font size of the node id labels [pt].

        Returns:
            The resolved destination path of the written PNG file.

        Raises:
            ValueError: If a link or route node references an id that is
                not part of ``nodes``.
        """
        import matplotlib.pyplot as plt

        lookup: Dict[str, MapNode] = {node.id: node for node in nodes}

        figure, axis = plt.subplots(figsize=(9, 7))

        for index, link in enumerate(links):
            if link.from_id not in lookup or link.to_id not in lookup:
                raise ValueError(
                    f"Link '{link.id}' references nodes outside the node list "
                    f"({link.from_id!r} -> {link.to_id!r})"
                )
            start = lookup[link.from_id]
            end = lookup[link.to_id]
            axis.plot(
                [start.x, end.x],
                [start.y, end.y],
                color="0.75",
                linewidth=1.5,
                zorder=1,
                label="Road links" if index == 0 else None,
            )

        if path_ids:
            missing = [node_id for node_id in path_ids if node_id not in lookup]
            if missing:
                raise ValueError(f"Route references unknown node ids: {missing}")
            route_coordinates = np.array(
                [[lookup[node_id].x, lookup[node_id].y] for node_id in path_ids],
                dtype=float,
            )
            axis.plot(
                route_coordinates[:, 0],
                route_coordinates[:, 1],
                color="red",
                linewidth=3.0,
                zorder=3,
                label="Planned route",
            )
            start_node = lookup[path_ids[0]]
            goal_node = lookup[path_ids[-1]]
            axis.plot(
                start_node.x,
                start_node.y,
                marker="o",
                markersize=12,
                linestyle="",
                markerfacecolor="limegreen",
                markeredgecolor="black",
                zorder=5,
                label=f"Start ({start_node.id})",
            )
            axis.plot(
                goal_node.x,
                goal_node.y,
                marker="*",
                markersize=18,
                linestyle="",
                markerfacecolor="magenta",
                markeredgecolor="black",
                zorder=5,
                label=f"Goal ({goal_node.id})",
            )

        if show_nodes:
            axis.scatter(
                [node.x for node in nodes],
                [node.y for node in nodes],
                s=18,
                color="0.15",
                zorder=4,
                label="Nodes",
            )
            for node in nodes:
                axis.annotate(
                    node.id,
                    (node.x, node.y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=node_label_size,
                    color="0.25",
                )

        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.set_axisbelow(True)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(title)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )
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
