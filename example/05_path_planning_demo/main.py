"""Entry point for module 05: A* route planning on small road-graph maps.

For every map configured in ``config/planning_config.yaml`` the script:

1. loads the road graph (nodes + links) from its JSON map file,
2. builds the undirected adjacency map,
3. plans a route from the configured start node to the goal node with a
   textbook A* search (cost = accumulated link length [m], admissible
   Euclidean heuristic [m]),
4. resamples the route into waypoints spaced ``vehicle.waypoint_spacing_m``
   metres apart,
5. renders one PNG figure per map, and
6. appends per-map planning metrics to a machine-readable
   ``run_summary.json``.

Typical usage (from the project root)::

    python 05_path_planning_demo/main.py
    python 05_path_planning_demo/main.py --show

Extension point: add alternative planners (weighted A*, hybrid A*, RRT*)
in ``planning/path_planner_demo.py`` and select them through the
``planning.algorithm`` configuration key instead of the fixed A* call.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent  # project root (contains shared/ and the other modules)

sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))

from planning.map_loader import MapLink, MapLoader, MapNode
from planning.path_planner_demo import AStarPlanner
from planning.route_visualizer import RouteVisualizer
from shared.utilities.config_loader import load_yaml
from shared.utilities.data_loader import save_json
from shared.utilities.logging_utils import get_logger

LOGGER = get_logger("05_path_planning_demo")

DEFAULT_CONFIG_PATH = MODULE_DIR / "config" / "planning_config.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument strings to parse; ``None`` uses ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``config``, ``show``, ``output_dir`` and
        ``seed`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plan routes on road-graph maps with A* search, render the "
            "routes and write per-map planning metrics."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Planning configuration YAML (default: module config; relative paths resolve against the module directory).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive Matplotlib windows in addition to saving PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG figures and run_summary.json (default: module outputs; relative paths resolve against the module directory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed kept for CLI uniformity across modules (planning is deterministic).",
    )
    return parser.parse_args(argv)


def resolve_module_path(path: Path) -> Path:
    """Resolve a possibly relative CLI path against the module directory.

    Args:
        path: User-supplied path; absolute paths are returned unchanged.

    Returns:
        Absolute path anchored either at the supplied absolute location or
        at the module directory.
    """
    return path if path.is_absolute() else MODULE_DIR / path


def plan_single_map(
    map_entry: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    show: bool,
) -> Dict[str, Any]:
    """Load one map, plan the route, render it and collect the metrics.

    Args:
        map_entry: Configuration dictionary with ``map_file`` (module
            relative), ``start_node`` and ``goal_node``.
        config: Full planning configuration dictionary.
        output_dir: Directory that receives the route PNG.
        show: Passed through to the visualiser (interactive windows).

    Returns:
        Metric dictionary for the map with keys ``map`` [-], ``start_node``
        [-], ``goal_node`` [-], ``n_nodes_in_map`` [-],
        ``n_nodes_in_path`` [-], ``path_length_m`` [m], ``n_waypoints`` [-],
        ``straightness_ratio`` [-], ``nodes_expanded`` [-] and
        ``unreachable`` [-].
    """
    map_path = resolve_module_path(Path(map_entry["map_file"]))
    nodes, links = MapLoader.load(map_path)
    map_name = MapLoader.map_name(map_path)
    adjacency = MapLoader.build_adjacency(nodes, links)

    planner = AStarPlanner(nodes, adjacency)
    start_node = str(map_entry["start_node"])
    goal_node = str(map_entry["goal_node"])
    path_ids = planner.plan(start_node, goal_node)

    spacing_m = float(config["vehicle"]["waypoint_spacing_m"])
    waypoints = AStarPlanner.waypoints_from_path(nodes, path_ids, spacing_m)

    visualization = config.get("visualization", {})
    figure_path = output_dir / f"planned_route_{map_path.stem}.png"
    RouteVisualizer.plot_map_with_route(
        nodes,
        links,
        path_ids,
        title=map_name,
        out_path=figure_path,
        show=show,
        show_nodes=bool(visualization.get("show_nodes", True)),
        node_label_size=int(visualization.get("node_label_size", 6)),
    )

    metrics = build_map_metrics(
        map_stem=map_path.stem,
        start_node=start_node,
        goal_node=goal_node,
        nodes=nodes,
        path_ids=path_ids,
        waypoints=waypoints,
        nodes_expanded=planner.nodes_expanded,
    )
    LOGGER.info(
        "Map %s: %s -> %s | path nodes %d | length %.2f m | waypoints %d | "
        "straightness %.4f | expanded %d | unreachable %s",
        metrics["map"],
        metrics["start_node"],
        metrics["goal_node"],
        metrics["n_nodes_in_path"],
        metrics["path_length_m"],
        metrics["n_waypoints"],
        -1.0 if metrics["straightness_ratio"] is None else metrics["straightness_ratio"],
        metrics["nodes_expanded"],
        metrics["unreachable"],
    )
    return metrics


def build_map_metrics(
    map_stem: str,
    start_node: str,
    goal_node: str,
    nodes: List[MapNode],
    path_ids: List[str],
    waypoints: Any,
    nodes_expanded: int,
) -> Dict[str, Any]:
    """Assemble the per-map planning metrics.

    Args:
        map_stem: File stem of the map document [-].
        start_node: Start node id [-].
        goal_node: Goal node id [-].
        nodes: All nodes of the map.
        path_ids: Planned route node ids [-]; empty when unreachable.
        waypoints: Resampled waypoint array of shape ``(N, 2)`` [m].
        nodes_expanded: A* expansion counter of the last plan [-].

    Returns:
        Metric dictionary; ``straightness_ratio`` is the Euclidean
        start-goal distance [m] divided by the planned path length [m]
        (1.0 = perfectly straight) and ``None`` when no path exists.
    """
    coordinates = {node.id: (node.x, node.y) for node in nodes}
    metrics: Dict[str, Any] = {
        "map": map_stem,
        "start_node": start_node,
        "goal_node": goal_node,
        "n_nodes_in_map": len(nodes),
        "n_nodes_in_path": len(path_ids),
        "path_length_m": 0.0,
        "n_waypoints": int(waypoints.shape[0]),
        "straightness_ratio": None,
        "nodes_expanded": int(nodes_expanded),
        "unreachable": len(path_ids) == 0,
    }
    if path_ids:
        path_length_m = sum(
            math.hypot(
                coordinates[segment_end][0] - coordinates[segment_start][0],
                coordinates[segment_end][1] - coordinates[segment_start][1],
            )
            for segment_start, segment_end in zip(path_ids[:-1], path_ids[1:])
        )
        straight_distance_m = math.hypot(
            coordinates[goal_node][0] - coordinates[start_node][0],
            coordinates[goal_node][1] - coordinates[start_node][1],
        )
        metrics["path_length_m"] = round(path_length_m, 3)
        if path_length_m > 0.0:
            metrics["straightness_ratio"] = round(straight_distance_m / path_length_m, 4)
        elif start_node == goal_node:
            metrics["straightness_ratio"] = 1.0
    return metrics


def main(argv: List[str] | None = None) -> int:
    """Run the planning demo for all configured maps.

    Args:
        argv: Argument strings; ``None`` uses ``sys.argv[1:]``.

    Returns:
        Process exit code, ``0`` on success.
    """
    args = parse_args(argv)
    LOGGER.info("Random seed %d accepted; the planner is deterministic and the seed does not alter results.", args.seed)

    # Fix the backend before any pyplot import: headless runs must not
    # require a display, interactive runs honour --show.
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (locks the backend choice in)

    config = load_yaml(resolve_module_path(args.config))
    output_dir = resolve_module_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    planning_config = config["planning"]
    LOGGER.info("Algorithm: %s | maps: %d", planning_config["algorithm"], len(planning_config["maps"]))

    plan_records: List[Dict[str, Any]] = []
    for map_entry in planning_config["maps"]:
        plan_records.append(plan_single_map(map_entry, config, output_dir, show=args.show))

    summary_path = save_json({"plans": plan_records}, output_dir / "run_summary.json")

    reachable = [record for record in plan_records if not record["unreachable"]]
    LOGGER.info(
        "Summary: %d/%d maps planned | total planned length %.2f m | total waypoints %d",
        len(reachable),
        len(plan_records),
        sum(record["path_length_m"] for record in reachable),
        sum(record["n_waypoints"] for record in plan_records),
    )
    LOGGER.info("Figures and summary written to %s", output_dir)
    LOGGER.info("run_summary.json: %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
