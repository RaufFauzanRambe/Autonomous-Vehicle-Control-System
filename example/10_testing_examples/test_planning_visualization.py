"""Tests for the planning stack and map-aware route rendering (module 05).

Covered behavior: JSON road-map loading and adjacency construction,
A* route validity on the sample map, and PNG generation by the
map-aware route visualizer (written only into the pytest tmp folder).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless rendering for plotting tests

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "05_path_planning_demo"))

from planning.map_loader import MapLoader
from planning.path_planner_demo import AStarPlanner
from planning.route_visualizer import RouteVisualizer

MAP_PATH = ROOT / "05_path_planning_demo" / "maps" / "simple_road_map.json"


def _load_map() -> tuple[list, list, dict[str, list[tuple[str, float]]]]:
    """Load the sample map and build its adjacency structure.

    Returns:
        Tuple of (nodes, links, adjacency) exactly as produced by
        ``MapLoader.load`` and ``MapLoader.build_adjacency``.
    """
    nodes, links = MapLoader.load(MAP_PATH)
    adjacency = MapLoader.build_adjacency(nodes, links)
    return nodes, links, adjacency


def test_map_loader_loads_nodes_and_links() -> None:
    """The sample map must expose nodes, links and a populated adjacency."""
    nodes, links, adjacency = _load_map()
    assert len(nodes) >= 8
    assert len(links) >= 7
    assert "n0" in adjacency


def test_astar_finds_path() -> None:
    """A* must return a valid n0 -> n7 route following adjacency edges."""
    nodes, _links, adjacency = _load_map()
    planner = AStarPlanner(nodes, adjacency)
    path = planner.plan("n0", "n7")
    assert path
    assert path[0] == "n0"
    assert path[-1] == "n7"
    for from_id, to_id in zip(path, path[1:]):
        neighbor_ids = [neighbor_id for neighbor_id, _cost in adjacency[from_id]]
        assert to_id in neighbor_ids


def test_route_png_created(tmp_path) -> None:
    """The map-aware route visualizer must create a non-empty PNG in tmp_path."""
    nodes, links, adjacency = _load_map()
    planner = AStarPlanner(nodes, adjacency)
    path = planner.plan("n0", "n7")
    out_path = tmp_path / "route.png"
    saved = RouteVisualizer.plot_map_with_route(
        nodes,
        links,
        path,
        title="test",
        out_path=out_path,
    )
    saved_path = Path(saved)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
