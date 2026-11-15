"""Textbook A* route planning over a road graph plus waypoint resampling.

The planner minimises accumulated link length [m] and uses the straight-line
Euclidean distance between node coordinates [m] as the heuristic ``h``.
Because the shortest connection between two planar points is the straight
line, ``h`` never overestimates the true remaining cost (it is admissible),
so A* is guaranteed to return the cost-optimal route for the given graph.

The search uses a binary heap (``heapq``) keyed by ``f = g + h`` where
``g`` is the accumulated link length from the start node. Stale heap
entries (nodes whose ``g`` value improved after they were queued) are
recognised by comparing against the best known cost and skipped.

Extension point: replace the planner with weighted A* (inflate ``h`` by a
factor ``epsilon > 1`` for bounded-suboptimal, faster searches), hybrid A*
(continuous state space with steering primitives for kinematically
feasible paths) or RRT* / Informed RRT* (sampling-based asymptotically
optimal planning) inside the same planner interface.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Dict, List, Tuple

import numpy as np

from planning.map_loader import MapNode


class AStarPlanner:
    """A* shortest-path planner on a node/link road graph.

    Attributes:
        nodes_expanded: Number of nodes that were dequeued from the open
            priority queue and expanded during the most recent
            :meth:`plan` call [-]. Stale duplicate queue entries are
            skipped and not counted. The counter resets on every call.

    Extension point: keep ``nodes_expanded`` history across calls to
    benchmark heuristics and cost functions in Monte Carlo studies.
    """

    def __init__(
        self,
        nodes: List[MapNode],
        adjacency: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """Build the planner from typed graph data.

        Args:
            nodes: Road-graph nodes with planar coordinates [m].
            adjacency: Mapping node id [-] to a list of
                ``(neighbor_id [-], length_m [m])`` tuples, as produced by
                :meth:`planning.map_loader.MapLoader.build_adjacency`.

        Raises:
            ValueError: If duplicate node ids are supplied or the
                adjacency references ids outside ``nodes``.
        """
        self._nodes: Dict[str, MapNode] = {}
        for node in nodes:
            if node.id in self._nodes:
                raise ValueError(f"Duplicate node id in map: {node.id!r}")
            self._nodes[node.id] = node
        for source_id, neighbors in adjacency.items():
            if source_id not in self._nodes:
                raise ValueError(f"Adjacency references unknown node: {source_id!r}")
            for neighbor_id, _length_m in neighbors:
                if neighbor_id not in self._nodes:
                    raise ValueError(
                        f"Adjacency of node {source_id!r} references unknown "
                        f"neighbor: {neighbor_id!r}"
                    )
        self._adjacency: Dict[str, List[Tuple[str, float]]] = adjacency
        self.nodes_expanded: int = 0

    def _heuristic(self, node_id: str, goal_id: str) -> float:
        """Admissible Euclidean heuristic between two nodes.

        Args:
            node_id: Identifier of the node to evaluate [-].
            goal_id: Identifier of the goal node [-].

        Returns:
            Straight-line distance between the two node coordinates [m].
        """
        source = self._nodes[node_id]
        goal = self._nodes[goal_id]
        return math.hypot(goal.x - source.x, goal.y - source.y)

    def plan(self, start_id: str, goal_id: str) -> List[str]:
        """Plan the cheapest route from ``start_id`` to ``goal_id``.

        Args:
            start_id: Identifier of the start node [-].
            goal_id: Identifier of the goal node [-].

        Returns:
            Ordered list of node ids [-] from start to goal (both
            inclusive). An empty list signals that the goal is not
            reachable from the start.

        Raises:
            ValueError: If ``start_id`` or ``goal_id`` is unknown, the
                adjacency contains unknown nodes, or a link cost is
                negative (A* requires non-negative edge costs).
        """
        for node_id, label in ((start_id, "start"), (goal_id, "goal")):
            if node_id not in self._nodes:
                raise ValueError(f"Unknown {label} node id: {node_id!r}")

        self.nodes_expanded = 0
        tie_breaker = itertools.count()
        open_heap: List[Tuple[float, int, str]] = [
            (self._heuristic(start_id, goal_id), next(tie_breaker), start_id)
        ]
        g_score: Dict[str, float] = {start_id: 0.0}
        came_from: Dict[str, str] = {}
        expanded: set[str] = set()

        while open_heap:
            _f_value, _tie, current_id = heapq.heappop(open_heap)
            if current_id in expanded:
                continue
            self.nodes_expanded += 1
            if current_id == goal_id:
                return self._reconstruct_path(came_from, start_id, goal_id)
            expanded.add(current_id)
            for neighbor_id, length_m in self._adjacency.get(current_id, []):
                if neighbor_id not in self._nodes:
                    raise ValueError(
                        f"Adjacency references unknown node: {neighbor_id!r}"
                    )
                if length_m < 0.0:
                    raise ValueError(
                        "Negative link length between "
                        f"{current_id!r} and {neighbor_id!r} breaks A* optimality"
                    )
                tentative_g = g_score[current_id] + length_m
                if tentative_g < g_score.get(neighbor_id, math.inf):
                    g_score[neighbor_id] = tentative_g
                    came_from[neighbor_id] = current_id
                    heapq.heappush(
                        open_heap,
                        (
                            tentative_g + self._heuristic(neighbor_id, goal_id),
                            next(tie_breaker),
                            neighbor_id,
                        ),
                    )
        return []

    @staticmethod
    def _reconstruct_path(came_from: Dict[str, str], start_id: str, goal_id: str) -> List[str]:
        """Walk the predecessor chain backwards from the goal.

        Args:
            came_from: Mapping node id [-] to its predecessor on the best
                path found so far.
            start_id: Identifier of the start node [-].
            goal_id: Identifier of the goal node [-].

        Returns:
            Ordered node ids [-] from start to goal, both inclusive.
        """
        path = [goal_id]
        while path[-1] != start_id:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    @staticmethod
    def waypoints_from_path(
        nodes: List[MapNode],
        path_ids: List[str],
        spacing_m: float,
    ) -> np.ndarray:
        """Resample a node route into evenly spaced waypoints.

        The route polyline is walked segment by segment and a waypoint is
        emitted every ``spacing_m`` metres of arc length; the final route
        node is always included as the last waypoint, even when the last
        partial segment is shorter than ``spacing_m``.

        Args:
            nodes: Road-graph nodes providing the coordinates [m].
            path_ids: Ordered node ids [-] of the route (as returned by
                :meth:`plan`); may be empty.
            spacing_m: Resampling distance between consecutive waypoints
                [m]; must be strictly positive.

        Returns:
            Array of shape ``(N, 2)`` with waypoint ``x``/``y`` coordinates
            [m]. For an empty route an array of shape ``(0, 2)`` is
            returned.

        Raises:
            ValueError: If ``spacing_m`` is not positive or a path id does
                not match any node in ``nodes``.

        Extension point: replace the linear resampling with clothoid or
        cubic-spline smoothing (e.g. ``scipy.interpolate.splprep``) so the
        waypoint sequence carries continuous curvature for the downstream
        motion controller.
        """
        if spacing_m <= 0.0:
            raise ValueError(f"spacing_m must be positive, got {spacing_m}")
        if not path_ids:
            return np.zeros((0, 2), dtype=float)

        lookup: Dict[str, MapNode] = {node.id: node for node in nodes}
        points: List[np.ndarray] = []
        for node_id in path_ids:
            if node_id not in lookup:
                raise ValueError(f"Path references unknown node id: {node_id!r}")
            node = lookup[node_id]
            points.append(np.array([node.x, node.y], dtype=float))

        emitted: List[np.ndarray] = [points[0].copy()]
        current = points[0]
        carry_m = 0.0
        for target in points[1:]:
            segment = target - current
            segment_length = float(np.linalg.norm(segment))
            while segment_length > 0.0 and carry_m + segment_length >= spacing_m:
                step_m = spacing_m - carry_m
                current = current + (step_m / segment_length) * segment
                segment = target - current
                segment_length = float(np.linalg.norm(segment))
                carry_m = 0.0
                emitted.append(current.copy())
            carry_m += segment_length
            current = target

        if np.linalg.norm(emitted[-1] - points[-1]) > 1e-9:
            emitted.append(points[-1].copy())
        return np.asarray(emitted, dtype=float)
