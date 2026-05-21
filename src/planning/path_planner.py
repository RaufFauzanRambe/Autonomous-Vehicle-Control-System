"""
Path Planner Module for Autonomous Vehicle Control System.

Provides A* occupancy-grid planning and lattice-based planning for
structured roads, along with path data structures and smoothing utilities.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HeuristicType(Enum):
    """Supported heuristic functions for A* search."""
    EUCLIDEAN = auto()
    MANHATTAN = auto()
    CHEBYSHEV = auto()


# ---------------------------------------------------------------------------
# Path dataclass
# ---------------------------------------------------------------------------

@dataclass
class Path:
    """Represents a planned path with metadata.

    Attributes:
        waypoints: Nx2 (or Nx3) float array of (x, y[, z]) coordinates.
        costs: Optional per-segment cost values (length N-1).
        total_cost: Aggregate cost of the full path.
        planner_name: Name of the planner that produced this path.
        metadata: Arbitrary key-value metadata (e.g. grid resolution).
        is_smooth: Whether gradient-descent smoothing has been applied.
    """
    waypoints: NDArray[np.float64]
    costs: Optional[NDArray[np.float64]] = None
    total_cost: float = 0.0
    planner_name: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)
    is_smooth: bool = False

    @property
    def length(self) -> float:
        """Total Euclidean path length."""
        if self.waypoints.shape[0] < 2:
            return 0.0
        diffs = np.diff(self.waypoints[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def num_waypoints(self) -> int:
        return self.waypoints.shape[0]

    def interpolated_point(self, fraction: float) -> NDArray[np.float64]:
        """Return an interpolated point at *fraction* along the path.

        Args:
            fraction: Value in [0, 1] where 0 = start, 1 = end.

        Returns:
            1-D array of the interpolated (x, y[, z]) coordinates.
        """
        if self.waypoints.shape[0] < 2:
            return self.waypoints[0].copy()
        fraction = max(0.0, min(fraction, 1.0))
        diffs = np.diff(self.waypoints[:, :2], axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        target = fraction * cum[-1]
        idx = int(np.searchsorted(cum, target, side="right")) - 1
        idx = max(0, min(idx, len(seg_lens) - 1))
        seg_frac = (target - cum[idx]) / max(seg_lens[idx], 1e-12)
        seg_frac = max(0.0, min(seg_frac, 1.0))
        pt = self.waypoints[idx] + seg_frac * (self.waypoints[idx + 1] - self.waypoints[idx])
        return pt


# ---------------------------------------------------------------------------
# Heuristic functions
# ---------------------------------------------------------------------------

def _heuristic_euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _heuristic_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _heuristic_chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


_HEURISTIC_MAP: Dict[HeuristicType, Callable] = {
    HeuristicType.EUCLIDEAN: _heuristic_euclidean,
    HeuristicType.MANHATTAN: _heuristic_manhattan,
    HeuristicType.CHEBYSHEV: _heuristic_chebyshev,
}


# ---------------------------------------------------------------------------
# AStarPlanner
# ---------------------------------------------------------------------------

class AStarPlanner:
    """A* pathfinding on a 2-D occupancy grid.

    The occupancy grid is a 2-D numpy array where 0 = free and 1 = occupied.
    8-connected neighbours are explored by default.

    Example::

        grid = np.zeros((100, 100), dtype=np.int8)
        grid[40:60, 40:60] = 1   # obstacle block
        planner = AStarPlanner(grid, resolution=0.5)
        path = planner.plan((5, 5), (90, 90))
    """

    # 8-connected movement deltas and costs
    _DELTAS_8: List[Tuple[int, int, float]] = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def __init__(
        self,
        occupancy_grid: NDArray[np.int8],
        resolution: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        heuristic: HeuristicType = HeuristicType.EUCLIDEAN,
        allow_diagonal: bool = True,
        obstacle_inflation: int = 0,
    ) -> None:
        """Initialise the A* planner.

        Args:
            occupancy_grid: 2-D array with 0=free, 1=occupied.
            resolution: Metres per grid cell.
            origin: World-frame origin of grid cell (0, 0).
            heuristic: Heuristic function to use.
            allow_diagonal: Whether 8-connectivity is allowed.
            obstacle_inflation: Number of cells to inflate obstacles by.
        """
        if obstacle_inflation > 0:
            self._grid = self._inflate_grid(occupancy_grid, obstacle_inflation)
        else:
            self._grid = occupancy_grid.copy()

        self._resolution = resolution
        self._origin = origin
        self._heuristic_type = heuristic
        self._heuristic_fn = _HEURISTIC_MAP[heuristic]
        self._deltas = self._DELTAS_8 if allow_diagonal else self._DELTAS_8[:4]
        self._rows, self._cols = self._grid.shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Path:
        """Run A* search from *start* to *goal* on the occupancy grid.

        Args:
            start: (row, col) grid indices for the start cell.
            goal: (row, col) grid indices for the goal cell.

        Returns:
            A Path object. If no path is found, waypoints contains only
            the start position.
        """
        self._validate_cell(start)
        self._validate_cell(goal)

        if self._grid[start[0], start[1]] == 1 or self._grid[goal[0], goal[1]] == 1:
            return self._empty_path(start)

        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0  # tie-breaker for equal f-values
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: Set[Tuple[int, int]] = set()

        h0 = self._heuristic_fn(start, goal)
        heapq.heappush(open_heap, (h0, counter, start))
        counter += 1

        while open_heap:
            f_val, _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, current, start, g_score[current])

            if current in closed:
                continue
            closed.add(current)

            for dr, dc, step_cost in self._deltas:
                nr, nc = current[0] + dr, current[1] + dc
                neighbour = (nr, nc)

                if not (0 <= nr < self._rows and 0 <= nc < self._cols):
                    continue
                if self._grid[nr, nc] == 1 or neighbour in closed:
                    continue

                tentative_g = g_score[current] + step_cost * self._resolution

                if tentative_g < g_score.get(neighbour, math.inf):
                    g_score[neighbour] = tentative_g
                    came_from[neighbour] = current
                    f = tentative_g + self._heuristic_fn(neighbour, goal)
                    heapq.heappush(open_heap, (f, counter, neighbour))
                    counter += 1

        # No path found
        return self._empty_path(start)

    def update_grid(self, new_grid: NDArray[np.int8]) -> None:
        """Replace the occupancy grid (e.g. after a sensor update)."""
        self._grid = new_grid.copy()
        self._rows, self._cols = self._grid.shape

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_cell(self, cell: Tuple[int, int]) -> None:
        if not (0 <= cell[0] < self._rows and 0 <= cell[1] < self._cols):
            raise ValueError(
                f"Cell {cell} out of grid bounds ({self._rows}, {self._cols})"
            )

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
        start: Tuple[int, int],
        total_g: float,
    ) -> Path:
        """Trace back from goal to start and build a Path object."""
        cells: List[Tuple[int, int]] = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()

        # Convert grid cells to world coordinates
        waypoints = np.array(
            [
                [self._origin[0] + c * self._resolution,
                 self._origin[1] + r * self._resolution]
                for r, c in cells
            ],
            dtype=np.float64,
        )

        seg_costs: Optional[NDArray[np.float64]] = None
        if waypoints.shape[0] > 1:
            seg_costs = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        return Path(
            waypoints=waypoints,
            costs=seg_costs,
            total_cost=total_g,
            planner_name="AStarPlanner",
            metadata={
                "resolution": self._resolution,
                "heuristic": self._heuristic_type.name,
                "grid_shape": self._grid.shape,
            },
        )

    def _empty_path(self, start: Tuple[int, int]) -> Path:
        """Return a degenerate path when no route exists."""
        wp = np.array(
            [[self._origin[0] + start[1] * self._resolution,
              self._origin[1] + start[0] * self._resolution]],
            dtype=np.float64,
        )
        return Path(
            waypoints=wp,
            total_cost=math.inf,
            planner_name="AStarPlanner",
            metadata={"no_path_found": True},
        )

    @staticmethod
    def _inflate_grid(grid: NDArray[np.int8], radius: int) -> NDArray[np.int8]:
        """Morphologically inflate obstacles by *radius* cells."""
        if radius <= 0:
            return grid.copy()
        inflated = grid.copy()
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    r_lo = max(0, r - radius)
                    r_hi = min(rows, r + radius + 1)
                    c_lo = max(0, c - radius)
                    c_hi = min(cols, c + radius + 1)
                    inflated[r_lo:r_hi, c_lo:c_hi] = 1
        return inflated


# ---------------------------------------------------------------------------
# LatticePlanner
# ---------------------------------------------------------------------------

@dataclass
class LatticeNode:
    """A node in the lattice graph.

    Attributes:
        x: Longitudinal position (m).
        y: Lateral position (m).
        theta: Heading angle (rad).
        layer: Layer index along the road (longitudinal step).
    """
    x: float
    y: float
    theta: float
    layer: int


class LatticePlanner:
    """Lattice-based planner for structured road environments.

    Constructs a lattice of candidate trajectories by discretising
    longitudinal and lateral positions and connecting them with
    cubic-polynomial arcs. Then selects the lowest-cost path through
    the lattice via dynamic programming.

    Example::

        planner = LatticePlanner(
            road_length=100.0, num_layers=20,
            lateral_offsets=[-3.5, 0.0, 3.5],
        )
        path = planner.plan(start_state=(0, 0, 0), goal_state=(100, 0, 0))
    """

    def __init__(
        self,
        road_length: float = 100.0,
        num_layers: int = 20,
        lateral_offsets: Optional[List[float]] = None,
        heading_samples: int = 5,
        max_lateral_offset: float = 7.0,
        step_cost_weight: float = 1.0,
        lateral_change_weight: float = 2.0,
        heading_weight: float = 0.5,
    ) -> None:
        """Initialise the LatticePlanner.

        Args:
            road_length: Total longitudinal length of the planning horizon (m).
            num_layers: Number of longitudinal layers (discretisation).
            lateral_offsets: Explicit lateral offset positions (m). If None,
                evenly spaced offsets from -max_lateral_offset to
                +max_lateral_offset (7 positions by default).
            heading_samples: Number of heading samples per lattice node.
            max_lateral_offset: Maximum lateral offset if auto-generating.
            step_cost_weight: Weight for longitudinal step cost.
            lateral_change_weight: Weight for lateral-change penalty.
            heading_weight: Weight for heading deviation penalty.
        """
        self._road_length = road_length
        self._num_layers = num_layers
        self._layer_spacing = road_length / max(num_layers - 1, 1)
        self._lateral_offsets = lateral_offsets or np.linspace(
            -max_lateral_offset, max_lateral_offset, 7
        ).tolist()
        self._heading_samples = heading_samples
        self._max_lateral_offset = max_lateral_offset
        self._step_cost_weight = step_cost_weight
        self._lateral_change_weight = lateral_change_weight
        self._heading_weight = heading_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start_state: Tuple[float, float, float],
        goal_state: Tuple[float, float, float],
        obstacles: Optional[NDArray[np.float64]] = None,
    ) -> Path:
        """Plan a path through the lattice from start to goal.

        Args:
            start_state: (x, y, theta) at the start.
            goal_state: (x, y, theta) at the goal.
            obstacles: Optional Mx2 array of obstacle (x, y) positions.

        Returns:
            A Path through the lattice, or a degenerate path if none found.
        """
        # Build lattice layers
        lattice = self._build_lattice(start_state, goal_state)

        # Dynamic-programming search through layers
        path_nodes = self._dp_search(lattice, obstacles)

        if path_nodes is None or len(path_nodes) < 2:
            return Path(
                waypoints=np.array([[start_state[0], start_state[1]]], dtype=np.float64),
                total_cost=math.inf,
                planner_name="LatticePlanner",
                metadata={"no_path_found": True},
            )

        # Interpolate between lattice nodes with cubic polynomials
        waypoints = self._interpolate_nodes(path_nodes)
        total_cost = self._compute_path_cost(path_nodes)

        return Path(
            waypoints=waypoints,
            total_cost=total_cost,
            planner_name="LatticePlanner",
            metadata={
                "road_length": self._road_length,
                "num_layers": self._num_layers,
                "lateral_offsets": self._lateral_offsets,
            },
        )

    # ------------------------------------------------------------------
    # Lattice construction
    # ------------------------------------------------------------------

    def _build_lattice(
        self,
        start_state: Tuple[float, float, float],
        goal_state: Tuple[float, float, float],
    ) -> List[List[LatticeNode]]:
        """Construct the lattice: a list of layers, each containing nodes."""
        lattice: List[List[LatticeNode]] = []

        # Layer 0: start node
        lattice.append([
            LatticeNode(
                x=start_state[0], y=start_state[1],
                theta=start_state[2], layer=0,
            )
        ])

        # Intermediate layers
        headings = np.linspace(-0.3, 0.3, self._heading_samples).tolist()
        for layer_idx in range(1, self._num_layers - 1):
            x = start_state[0] + layer_idx * self._layer_spacing
            nodes: List[LatticeNode] = []
            for y_off in self._lateral_offsets:
                for theta in headings:
                    nodes.append(LatticeNode(
                        x=x, y=y_off, theta=theta, layer=layer_idx,
                    ))
            lattice.append(nodes)

        # Last layer: goal node
        lattice.append([
            LatticeNode(
                x=goal_state[0], y=goal_state[1],
                theta=goal_state[2], layer=self._num_layers - 1,
            )
        ])

        return lattice

    # ------------------------------------------------------------------
    # DP search
    # ------------------------------------------------------------------

    def _dp_search(
        self,
        lattice: List[List[LatticeNode]],
        obstacles: Optional[NDArray[np.float64]],
    ) -> Optional[List[LatticeNode]]:
        """Dynamic-programming search: cheapest path from layer 0 to last layer.

        Returns the sequence of LatticeNodes, or None if unreachable.
        """
        num_layers = len(lattice)
        if num_layers == 0:
            return None

        # cost_to[layer][node_idx] = best cost to reach this node
        cost_to: List[Dict[int, float]] = [{} for _ in range(num_layers)]
        parent: List[Dict[int, int]] = [{} for _ in range(num_layers)]

        # Initialise first layer
        for j, node in enumerate(lattice[0]):
            cost_to[0][j] = 0.0

        for i in range(num_layers - 1):
            for j, node_curr in enumerate(lattice[i]):
                if j not in cost_to[i]:
                    continue
                base_cost = cost_to[i][j]
                for k, node_next in enumerate(lattice[i + 1]):
                    edge_cost = self._edge_cost(node_curr, node_next, obstacles)
                    if edge_cost is None:
                        continue  # collision
                    new_cost = base_cost + edge_cost
                    if new_cost < cost_to[i + 1].get(k, math.inf):
                        cost_to[i + 1][k] = new_cost
                        parent[i + 1][k] = j

        # Find best goal node
        last = num_layers - 1
        if not cost_to[last]:
            return None

        best_idx = min(cost_to[last], key=lambda k: cost_to[last][k])

        # Reconstruct
        path_indices: List[Tuple[int, int]] = [(last, best_idx)]
        layer = last
        idx = best_idx
        while layer > 0:
            idx = parent[layer].get(idx)
            if idx is None:
                return None
            layer -= 1
            path_indices.append((layer, idx))
        path_indices.reverse()

        return [lattice[l][j] for l, j in path_indices]

    # ------------------------------------------------------------------
    # Edge cost
    # ------------------------------------------------------------------

    def _edge_cost(
        self,
        src: LatticeNode,
        dst: LatticeNode,
        obstacles: Optional[NDArray[np.float64]],
    ) -> Optional[float]:
        """Cost of transitioning from *src* to *dst*. Returns None on collision."""
        dx = dst.x - src.x
        dy = dst.y - src.y
        step_cost = math.hypot(dx, dy) * self._step_cost_weight
        lateral_cost = abs(dy) * self._lateral_change_weight
        heading_cost = abs(dst.theta - src.theta) * self._heading_weight

        # Simple collision check: sample midpoint
        mid_x = (src.x + dst.x) / 2.0
        mid_y = (src.y + dst.y) / 2.0
        if obstacles is not None and obstacles.shape[0] > 0:
            dists = np.sqrt((obstacles[:, 0] - mid_x) ** 2 + (obstacles[:, 1] - mid_y) ** 2)
            if np.min(dists) < 2.0:
                return None

        return step_cost + lateral_cost + heading_cost

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _interpolate_nodes(self, nodes: List[LatticeNode]) -> NDArray[np.float64]:
        """Interpolate between lattice nodes using cubic polynomials.

        For each pair of consecutive nodes, fit a cubic polynomial y(x)
        satisfying position and heading boundary conditions, then sample
        points along it.
        """
        points: List[NDArray[np.float64]] = []

        for i in range(len(nodes) - 1):
            n0 = nodes[i]
            n1 = nodes[i + 1]
            x0, y0, t0 = n0.x, n0.y, n0.theta
            x1, y1, t1 = n1.x, n1.y, n1.theta

            dx = x1 - x0
            if abs(dx) < 1e-6:
                # Nearly vertical: linear interpolation
                n_samples = max(int(abs(y1 - y0) / 0.5), 2)
                for k in range(n_samples):
                    frac = k / n_samples
                    points.append(np.array([x0 + frac * dx, y0 + frac * (y1 - y0)]))
                continue

            # Cubic polynomial: y = a + b*x_rel + c*x_rel^2 + d*x_rel^3
            # Boundary conditions:
            #   y(0) = y0, y'(0) = tan(t0), y(1) = y1, y'(1) = tan(t1)
            # where x_rel = (x - x0) / dx  in [0, 1]
            dy0 = math.tan(t0) * dx
            dy1 = math.tan(t1) * dx

            a = y0
            b = dy0
            c = 3.0 * (y1 - y0) - 2.0 * dy0 - dy1
            d = -2.0 * (y1 - y0) + dy0 + dy1

            n_samples = max(int(dx / 0.5), 2)
            for k in range(n_samples):
                t = k / n_samples
                y_val = a + b * t + c * t ** 2 + d * t ** 3
                x_val = x0 + t * dx
                points.append(np.array([x_val, y_val]))

        # Append final point
        points.append(np.array([nodes[-1].x, nodes[-1].y]))

        return np.array(points, dtype=np.float64)

    # ------------------------------------------------------------------
    # Path cost
    # ------------------------------------------------------------------

    def _compute_path_cost(self, nodes: List[LatticeNode]) -> float:
        """Sum of step distances along the selected lattice nodes."""
        cost = 0.0
        for i in range(len(nodes) - 1):
            dx = nodes[i + 1].x - nodes[i].x
            dy = nodes[i + 1].y - nodes[i].y
            cost += math.hypot(dx, dy)
        return cost


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------

def smooth_path(
    path: Path,
    weight_data: float = 0.3,
    weight_smooth: float = 0.5,
    tolerance: float = 1e-5,
    max_iterations: int = 500,
    obstacle_grid: Optional[NDArray[np.int8]] = None,
    grid_resolution: float = 1.0,
    grid_origin: Tuple[float, float] = (0.0, 0.0),
) -> Path:
    """Smooth a path using gradient descent.

    The algorithm iteratively adjusts interior waypoints to balance
    closeness to the original data (weight_data) with path smoothness
    (weight_smooth), similar to the approach in Thrun et al.

    Args:
        path: The original Path to smooth.
        weight_data: Weight for data fidelity term (0..1).
        weight_smooth: Weight for smoothness term (0..1).
        tolerance: Convergence threshold on max waypoint change.
        max_iterations: Maximum gradient-descent iterations.
        obstacle_grid: If provided, prevents smoothed points from entering
            occupied cells.
        grid_resolution: Metres per cell of the obstacle grid.
        grid_origin: World-frame origin of grid cell (0, 0).

    Returns:
        A new Path with smoothed waypoints and is_smooth=True.
    """
    if path.waypoints.shape[0] <= 2:
        return Path(
            waypoints=path.waypoints.copy(),
            total_cost=path.total_cost,
            planner_name=path.planner_name,
            metadata={**path.metadata, "smoothing_applied": True},
            is_smooth=True,
        )

    new_wp = path.waypoints.copy().astype(np.float64)

    for _ in range(max_iterations):
        max_change = 0.0
        for i in range(1, new_wp.shape[0] - 1):
            for d in range(new_wp.shape[1]):
                old_val = new_wp[i, d]
                data_term = weight_data * (path.waypoints[i, d] - new_wp[i, d])
                smooth_term = weight_smooth * (
                    new_wp[i - 1, d] + new_wp[i + 1, d] - 2.0 * new_wp[i, d]
                )
                new_wp[i, d] += data_term + smooth_term

                # Obstacle constraint: revert if entering occupied cell
                if obstacle_grid is not None and d < 2:
                    col = int((new_wp[i, 0] - grid_origin[0]) / grid_resolution)
                    row = int((new_wp[i, 1] - grid_origin[1]) / grid_resolution)
                    if 0 <= row < obstacle_grid.shape[0] and 0 <= col < obstacle_grid.shape[1]:
                        if obstacle_grid[row, col] == 1:
                            new_wp[i, d] = old_val  # revert

                max_change = max(max_change, abs(new_wp[i, d] - old_val))

        if max_change < tolerance:
            break

    seg_costs = np.linalg.norm(np.diff(new_wp[:, :2], axis=0), axis=1) if new_wp.shape[0] > 1 else None

    return Path(
        waypoints=new_wp,
        costs=seg_costs,
        total_cost=float(np.sum(seg_costs)) if seg_costs is not None else 0.0,
        planner_name=path.planner_name,
        metadata={**path.metadata, "smoothing_applied": True},
        is_smooth=True,
    )
