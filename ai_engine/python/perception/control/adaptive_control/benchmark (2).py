"""
Benchmark Suite for Path Planning Algorithms.

Provides comprehensive benchmarking capabilities:
- Multiple planning scenarios (open, maze, narrow passages, parking)
- Comparative metrics: time, path quality, optimality, memory
- Statistical analysis across multiple trials
- Benchmark result reporting and visualization data
- Regression testing against baseline performance
"""

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from planning.path_planning.planner_utils import (
    Path,
    Pose2D,
    euclidean_distance,
    compute_path_smoothness,
    compute_path_curvature_variance,
)
from planning.path_planning.astar_planner import AStarPlanner, AStarConfig, AStarResult
from planning.path_planning.dijkstra_planner import DijkstraPlanner, DijkstraConfig, DijkstraResult
from planning.path_planning.rrt_planner import RRTPlanner, RRTConfig, RRTResult

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Data Structures
# =============================================================================

@dataclass
class BenchmarkScenario:
    """A benchmark scenario with map, start, and goal.

    Attributes:
        name: Scenario name.
        description: Scenario description.
        grid: Occupancy grid for the scenario.
        origin_x: Grid origin x.
        origin_y: Grid origin y.
        resolution: Grid resolution.
        start: Start pose.
        goal: Goal pose.
        optimal_cost: Known optimal path cost (if available).
        category: Scenario category (open, maze, narrow, parking).
        difficulty: Difficulty rating (1-5).
    """
    name: str
    description: str = ""
    grid: Optional[np.ndarray] = None
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.1
    start: Pose2D = field(default_factory=Pose2D)
    goal: Pose2D = field(default_factory=Pose2D)
    optimal_cost: float = float('inf')
    category: str = "general"
    difficulty: int = 3


@dataclass
class TrialResult:
    """Result from a single benchmark trial.

    Attributes:
        planner_name: Name of the planner.
        scenario_name: Name of the scenario.
        success: Whether planning succeeded.
        planning_time: Time for planning in seconds.
        path_length: Length of the found path in meters.
        path_smoothness: Smoothness metric (lower is smoother).
        curvature_variance: Curvature variance metric.
        optimality_gap: Ratio of found cost to optimal cost.
        explored_nodes: Number of nodes/states explored.
        iterations: Number of iterations used.
        memory_estimate: Estimated memory usage in bytes.
    """
    planner_name: str = ""
    scenario_name: str = ""
    success: bool = False
    planning_time: float = 0.0
    path_length: float = 0.0
    path_smoothness: float = 0.0
    curvature_variance: float = 0.0
    optimality_gap: float = float('inf')
    explored_nodes: int = 0
    iterations: int = 0
    memory_estimate: int = 0


@dataclass
class BenchmarkResult:
    """Aggregated results for a planner-scenario combination.

    Attributes:
        planner_name: Name of the planner.
        scenario_name: Name of the scenario.
        num_trials: Number of trials run.
        success_rate: Fraction of successful trials.
        avg_planning_time: Average planning time.
        std_planning_time: Standard deviation of planning time.
        min_planning_time: Minimum planning time.
        max_planning_time: Maximum planning time.
        avg_path_length: Average path length.
        avg_smoothness: Average smoothness.
        avg_optimality_gap: Average optimality gap.
        avg_explored: Average number of explored nodes.
    """
    planner_name: str = ""
    scenario_name: str = ""
    num_trials: int = 0
    success_rate: float = 0.0
    avg_planning_time: float = 0.0
    std_planning_time: float = 0.0
    min_planning_time: float = 0.0
    max_planning_time: float = 0.0
    avg_path_length: float = 0.0
    std_path_length: float = 0.0
    avg_smoothness: float = 0.0
    avg_curvature_variance: float = 0.0
    avg_optimality_gap: float = float('inf')
    avg_explored: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "planner": self.planner_name,
            "scenario": self.scenario_name,
            "trials": self.num_trials,
            "success_rate": self.success_rate,
            "avg_time": self.avg_planning_time,
            "std_time": self.std_planning_time,
            "min_time": self.min_planning_time,
            "max_time": self.max_planning_time,
            "avg_length": self.avg_path_length,
            "std_length": self.std_path_length,
            "avg_smoothness": self.avg_smoothness,
            "avg_curvature_var": self.avg_curvature_variance,
            "avg_optimality_gap": self.avg_optimality_gap,
            "avg_explored": self.avg_explored,
        }


# =============================================================================
# Scenario Generator
# =============================================================================

class ScenarioGenerator:
    """Generates benchmark scenarios of varying difficulty.

    Creates realistic test environments for evaluating planner
    performance across different conditions.
    """

    @staticmethod
    def generate_open_space(width: int = 100, height: int = 100,
                            resolution: float = 0.1,
                            obstacle_ratio: float = 0.05,
                            seed: int = 42) -> BenchmarkScenario:
        """Generate an open space scenario with few obstacles.

        Args:
            width: Grid width.
            height: Grid height.
            resolution: Grid resolution.
            obstacle_ratio: Ratio of obstacle cells.
            seed: Random seed.

        Returns:
            BenchmarkScenario with open space map.
        """
        rng = np.random.RandomState(seed)
        grid = np.zeros((height, width), dtype=np.int8)

        # Sparse random obstacles
        mask = rng.random((height, width)) < obstacle_ratio
        grid[mask] = 100

        # Add borders
        grid[0, :] = 100
        grid[-1, :] = 100
        grid[:, 0] = 100
        grid[:, -1] = 100

        start = Pose2D(5.0, 5.0, 0.0)
        goal = Pose2D(width * resolution - 5.0, height * resolution - 5.0, 0.0)

        return BenchmarkScenario(
            name="open_space",
            description="Open space with sparse obstacles",
            grid=grid,
            resolution=resolution,
            start=start,
            goal=goal,
            category="open",
            difficulty=1,
        )

    @staticmethod
    def generate_maze(width: int = 100, height: int = 100,
                      resolution: float = 0.1,
                      corridor_width: int = 3,
                      seed: int = 42) -> BenchmarkScenario:
        """Generate a maze scenario.

        Uses recursive backtracking to create a proper maze.

        Args:
            width: Grid width.
            height: Grid height.
            resolution: Grid resolution.
            corridor_width: Width of corridors in cells.
            seed: Random seed.

        Returns:
            BenchmarkScenario with maze map.
        """
        rng = np.random.RandomState(seed)
        grid = np.full((height, width), 100, dtype=np.int8)  # All walls

        # Maze generation using recursive backtracking
        cell_size = corridor_width + 1
        maze_w = width // cell_size
        maze_h = height // cell_size

        # Track visited cells
        visited = np.zeros((maze_h, maze_w), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < maze_w and 0 <= ny < maze_h and not visited[ny, nx]:
                    neighbors.append((nx, ny, dx, dy))

            if neighbors:
                nx, ny, dx, dy = neighbors[rng.randint(len(neighbors))]
                # Carve passage
                base_x = cx * cell_size + 1
                base_y = cy * cell_size + 1
                for i in range(corridor_width):
                    for j in range(corridor_width):
                        grid[base_y + j, base_x + i] = 0
                # Carve connecting passage
                conn_x = (cx * cell_size + 1) + dx * cell_size
                conn_y = (cy * cell_size + 1) + dy * cell_size
                for i in range(corridor_width):
                    for j in range(corridor_width):
                        px = conn_x + i if dx != 0 else base_x + i
                        py = conn_y + j if dy != 0 else base_y + j
                        if 0 <= px < width and 0 <= py < height:
                            grid[py, px] = 0

                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()

        # Set start and goal in open areas
        start = Pose2D(1.0, 1.0, 0.0)
        goal = Pose2D((width - 2) * resolution, (height - 2) * resolution, 0.0)

        # Ensure start and goal are free
        grid[1:4, 1:4] = 0
        grid[-4:-1, -4:-1] = 0

        return BenchmarkScenario(
            name="maze",
            description="Generated maze with narrow corridors",
            grid=grid,
            resolution=resolution,
            start=start,
            goal=goal,
            category="maze",
            difficulty=4,
        )

    @staticmethod
    def generate_narrow_passage(width: int = 150, height: int = 100,
                                 resolution: float = 0.1,
                                 passage_width: int = 5,
                                 num_passages: int = 3,
                                 seed: int = 42) -> BenchmarkScenario:
        """Generate a scenario with narrow passages.

        Creates a map with walls that have narrow gaps, testing the
        planner's ability to find and navigate through tight spaces.

        Args:
            width: Grid width.
            height: Grid height.
            resolution: Grid resolution.
            passage_width: Width of each passage in cells.
            num_passages: Number of wall passages.
            seed: Random seed.

        Returns:
            BenchmarkScenario with narrow passages.
        """
        rng = np.random.RandomState(seed)
        grid = np.zeros((height, width), dtype=np.int8)

        # Add borders
        grid[0, :] = 100
        grid[-1, :] = 100
        grid[:, 0] = 100
        grid[:, -1] = 100

        # Add vertical walls with narrow passages
        spacing = width // (num_passages + 1)
        for i in range(num_passages):
            wall_x = spacing * (i + 1)
            grid[:, wall_x] = 100
            grid[:, wall_x + 1] = 100

            # Create passage at random y position
            passage_y = rng.randint(passage_width + 5, height - passage_width - 5)
            grid[passage_y - passage_width // 2:passage_y + passage_width // 2,
                 wall_x:wall_x + 2] = 0

        start = Pose2D(2.0, height * resolution / 2.0, 0.0)
        goal = Pose2D((width - 2) * resolution, height * resolution / 2.0, 0.0)

        return BenchmarkScenario(
            name="narrow_passage",
            description=f"Narrow passages (width={passage_width})",
            grid=grid,
            resolution=resolution,
            start=start,
            goal=goal,
            category="narrow",
            difficulty=3,
        )

    @staticmethod
    def generate_parking_lot(width: int = 80, height: int = 60,
                              resolution: float = 0.1,
                              seed: int = 42) -> BenchmarkScenario:
        """Generate a parking lot scenario.

        Creates a realistic parking lot with parked cars and lanes.

        Args:
            width: Grid width.
            height: Grid height.
            resolution: Grid resolution.
            seed: Random seed.

        Returns:
            BenchmarkScenario with parking lot.
        """
        rng = np.random.RandomState(seed)
        grid = np.zeros((height, width), dtype=np.int8)

        # Borders
        grid[0, :] = 100
        grid[-1, :] = 100
        grid[:, 0] = 100
        grid[:, -1] = 100

        # Parking rows (cars as obstacles)
        car_length = int(4.5 / resolution)
        car_width = int(1.8 / resolution)
        lane_width = int(6.0 / resolution)
        car_spacing = int(0.5 / resolution)

        y = 10
        while y + car_length < height - 10:
            # Row of cars
            x = 5
            while x + car_width < width - 5:
                if rng.random() > 0.2:  # 80% occupied
                    grid[y:y + car_length, x:x + car_width] = 100
                x += car_width + car_spacing

            y += car_length + lane_width

        start = Pose2D(width * resolution / 2.0, 2.0, math.pi / 2.0)
        goal = Pose2D(width * resolution / 2.0, (height - 5) * resolution, math.pi / 2.0)

        return BenchmarkScenario(
            name="parking_lot",
            description="Parking lot with randomly parked cars",
            grid=grid,
            resolution=resolution,
            start=start,
            goal=goal,
            category="parking",
            difficulty=4,
        )


# =============================================================================
# Benchmark Suite
# =============================================================================

class BenchmarkSuite:
    """Comprehensive benchmark suite for path planning algorithms.

    Runs multiple planners on multiple scenarios with statistical
    analysis of results.

    Usage:
        suite = BenchmarkSuite()
        suite.add_planner("A*", AStarPlanner())
        suite.add_scenario(ScenarioGenerator.generate_maze())
        results = suite.run(num_trials=10)
        suite.report(results)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize benchmark suite.

        Args:
            config: Configuration dictionary.
        """
        self._config = config or {}
        self._num_trials = self._config.get("num_trials", 10)
        self._timeout = self._config.get("timeout_seconds", 30.0)
        self._output_dir = self._config.get("output_dir", "./benchmark_results")
        self._metrics = self._config.get("metrics", [
            "planning_time", "path_length", "path_smoothness",
            "optimality_gap", "memory_usage", "success_rate",
        ])

        self._planners: Dict[str, Any] = {}
        self._scenarios: List[BenchmarkScenario] = []
        self._results: List[BenchmarkResult] = []

    def add_planner(self, name: str, planner: Any) -> None:
        """Add a planner to benchmark.

        Args:
            name: Planner name for reporting.
            planner: Planner instance with plan() method.
        """
        self._planners[name] = planner

    def add_scenario(self, scenario: BenchmarkScenario) -> None:
        """Add a scenario to benchmark.

        Args:
            scenario: BenchmarkScenario instance.
        """
        self._scenarios.append(scenario)

    def add_default_scenarios(self) -> None:
        """Add a default set of benchmark scenarios."""
        self._scenarios = [
            ScenarioGenerator.generate_open_space(),
            ScenarioGenerator.generate_maze(),
            ScenarioGenerator.generate_narrow_passage(),
            ScenarioGenerator.generate_parking_lot(),
        ]

    def add_default_planners(self) -> None:
        """Add a default set of planners."""
        self._planners = {
            "AStar_Euclidean": AStarPlanner(AStarConfig(heuristic="euclidean")),
            "AStar_Octile": AStarPlanner(AStarConfig(heuristic="octile")),
            "AStar_Weighted": AStarPlanner(AStarConfig(weight=2.0)),
            "Dijkstra": DijkstraPlanner(DijkstraConfig()),
            "RRT": RRTPlanner(RRTConfig(use_rrt_star=False, max_iterations=10000)),
            "RRTStar": RRTPlanner(RRTConfig(use_rrt_star=True, max_iterations=10000)),
        }

    # =========================================================================
    # Benchmark Execution
    # =========================================================================

    def run(self, num_trials: Optional[int] = None) -> List[BenchmarkResult]:
        """Run all planner-scenario combinations.

        Args:
            num_trials: Override number of trials per combination.

        Returns:
            List of BenchmarkResult objects.
        """
        trials = num_trials or self._num_trials
        self._results = []

        total_runs = len(self._planners) * len(self._scenarios) * trials
        completed = 0

        logger.info(f"Starting benchmark: {len(self._planners)} planners, "
                     f"{len(self._scenarios)} scenarios, {trials} trials each")

        for planner_name, planner in self._planners.items():
            for scenario in self._scenarios:
                trial_results: List[TrialResult] = []

                # Set map for planner
                if hasattr(planner, 'set_map') and scenario.grid is not None:
                    planner.set_map(
                        scenario.grid,
                        scenario.origin_x,
                        scenario.origin_y,
                        scenario.resolution,
                    )

                for trial in range(trials):
                    result = self._run_trial(planner, planner_name, scenario)
                    trial_results.append(result)
                    completed += 1

                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total_runs} "
                                     f"({100 * completed / total_runs:.0f}%)")

                # Aggregate results
                benchmark_result = self._aggregate_results(
                    planner_name, scenario.name, trial_results
                )
                self._results.append(benchmark_result)

        logger.info(f"Benchmark complete: {len(self._results)} results")
        return self._results

    def _run_trial(self, planner: Any, planner_name: str,
                   scenario: BenchmarkScenario) -> TrialResult:
        """Run a single benchmark trial.

        Args:
            planner: Planner instance.
            planner_name: Name for reporting.
            scenario: Benchmark scenario.

        Returns:
            TrialResult with metrics.
        """
        result = TrialResult(
            planner_name=planner_name,
            scenario_name=scenario.name,
        )

        try:
            start_time = time.time()
            plan_result = planner.plan(scenario.start, scenario.goal)
            elapsed = time.time() - start_time

            if elapsed > self._timeout:
                result.success = False
                result.planning_time = elapsed
                return result

            result.planning_time = elapsed
            result.success = plan_result.path.is_valid if hasattr(plan_result, 'path') else False

            if result.success:
                path = plan_result.path
                result.path_length = path.length
                result.path_smoothness = compute_path_smoothness(path)
                result.curvature_variance = compute_path_curvature_variance(path)

                if scenario.optimal_cost < float('inf'):
                    result.optimality_gap = path.length / scenario.optimal_cost

                if hasattr(plan_result, 'explored_cells'):
                    result.explored_nodes = plan_result.explored_cells
                elif hasattr(plan_result, 'explored_states'):
                    result.explored_nodes = plan_result.explored_states
                elif hasattr(plan_result, 'tree_size'):
                    result.explored_nodes = plan_result.tree_size

                if hasattr(plan_result, 'iterations_used'):
                    result.iterations = plan_result.iterations_used

                # Memory estimate (rough)
                result.memory_estimate = len(path.poses) * 64  # ~64 bytes per Pose2D

        except Exception as e:
            logger.error(f"Trial failed: {planner_name} on {scenario.name}: {e}")
            result.success = False

        return result

    def _aggregate_results(self, planner_name: str, scenario_name: str,
                           trial_results: List[TrialResult]) -> BenchmarkResult:
        """Aggregate trial results into benchmark statistics.

        Args:
            planner_name: Planner name.
            scenario_name: Scenario name.
            trial_results: List of trial results.

        Returns:
            BenchmarkResult with aggregated statistics.
        """
        successful = [r for r in trial_results if r.success]
        n_success = len(successful)
        n_total = len(trial_results)

        result = BenchmarkResult(
            planner_name=planner_name,
            scenario_name=scenario_name,
            num_trials=n_total,
            success_rate=n_success / max(n_total, 1),
        )

        if n_success == 0:
            return result

        # Compute statistics
        times = [r.planning_time for r in successful]
        lengths = [r.path_length for r in successful]
        smoothness = [r.path_smoothness for r in successful]
        curv_vars = [r.curvature_variance for r in successful]
        opt_gaps = [r.optimality_gap for r in successful if r.optimality_gap < float('inf')]
        explored = [r.explored_nodes for r in successful]

        result.avg_planning_time = np.mean(times)
        result.std_planning_time = np.std(times)
        result.min_planning_time = np.min(times)
        result.max_planning_time = np.max(times)

        result.avg_path_length = np.mean(lengths)
        result.std_path_length = np.std(lengths)

        result.avg_smoothness = np.mean(smoothness)
        result.avg_curvature_variance = np.mean(curv_vars)
        result.avg_optimality_gap = np.mean(opt_gaps) if opt_gaps else float('inf')
        result.avg_explored = np.mean(explored)

        return result

    # =========================================================================
    # Reporting
    # =========================================================================

    def report(self, results: Optional[List[BenchmarkResult]] = None) -> str:
        """Generate a text report of benchmark results.

        Args:
            results: Results to report. Uses stored results if None.

        Returns:
            Formatted report string.
        """
        results = results or self._results
        if not results:
            return "No benchmark results to report."

        lines = []
        lines.append("=" * 80)
        lines.append("PATH PLANNING BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Group by scenario
        by_scenario: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        for r in results:
            by_scenario[r.scenario_name].append(r)

        for scenario_name, scenario_results in by_scenario.items():
            lines.append(f"\n--- Scenario: {scenario_name} ---")
            lines.append(f"{'Planner':<25} {'Success':>8} {'Avg Time':>10} "
                         f"{'Avg Len':>10} {'Smooth':>8} {'Opt Gap':>8} {'Explored':>10}")
            lines.append("-" * 80)

            for r in sorted(scenario_results, key=lambda x: x.avg_planning_time):
                opt_str = (f"{r.avg_optimality_gap:.3f}"
                           if r.avg_optimality_gap < float('inf') else "N/A")
                lines.append(
                    f"{r.planner_name:<25} {r.success_rate:>7.1%} "
                    f"{r.avg_planning_time:>9.4f}s "
                    f"{r.avg_path_length:>9.1f}m "
                    f"{r.avg_smoothness:>7.3f} "
                    f"{opt_str:>8} "
                    f"{r.avg_explored:>9.0f}"
                )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_results(self, filepath: Optional[str] = None) -> None:
        """Save benchmark results to a JSON file.

        Args:
            filepath: Output file path. Uses config default if None.
        """
        filepath = filepath or os.path.join(self._output_dir, "benchmark_results.json")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        data = [r.to_dict() for r in self._results]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def get_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """Get a comparison table of planners across all scenarios.

        Returns:
            Dictionary mapping planner names to metric averages.
        """
        by_planner: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        for r in self._results:
            by_planner[r.planner_name].append(r)

        comparison = {}
        for planner_name, planner_results in by_planner.items():
            successful = [r for r in planner_results if r.success_rate > 0]
            if not successful:
                continue

            comparison[planner_name] = {
                "avg_success_rate": np.mean([r.success_rate for r in successful]),
                "avg_planning_time": np.mean([r.avg_planning_time for r in successful]),
                "avg_path_length": np.mean([r.avg_path_length for r in successful]),
                "avg_smoothness": np.mean([r.avg_smoothness for r in successful]),
                "avg_explored": np.mean([r.avg_explored for r in successful]),
            }

        return comparison


# =============================================================================
# Main Entry Point
# =============================================================================

def run_default_benchmark() -> str:
    """Run the default benchmark suite and return the report.

    Returns:
        Formatted benchmark report string.
    """
    suite = BenchmarkSuite({"num_trials": 5, "timeout_seconds": 10.0})
    suite.add_default_planners()
    suite.add_default_scenarios()
    results = suite.run()
    report = suite.report(results)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_default_benchmark()
    print(report)
