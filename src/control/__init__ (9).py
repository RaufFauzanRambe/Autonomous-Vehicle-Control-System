"""
Planning Module for Autonomous Vehicle Control System.

Provides path planning, behavior planning, trajectory optimization,
and cost calculation capabilities.

Exports:
    PathPlanner: Alias for the primary A* path planner class.
    BehaviorPlanner: FSM-based behavior planner for driving decisions.
    TrajectoryOptimizer: QP-based trajectory optimizer.
    CostCalculator: Multi-component cost calculator with scenario presets.

Submodules:
    path_planner: A* and lattice-based path planning with smoothing.
    behavior_planner: Finite state machine behavior planning.
    trajectory_optimizer: Quadratic programming trajectory optimization.
    cost_calculator: Weighted multi-component cost computation.
"""

from planning.path_planner import (
    AStarPlanner,
    HeuristicType,
    LatticePlanner,
    Path,
    smooth_path,
)
from planning.behavior_planner import (
    BehaviorPlanner,
    DrivingState,
    Maneuver,
    ManeuverType,
    ObstacleInfo,
    RoadContext,
    TransitionResult,
    VehicleState,
)
from planning.trajectory_optimizer import (
    OptimizerConfig,
    Trajectory,
    TrajectoryOptimizer,
    TrajectoryPoint,
)
from planning.cost_calculator import (
    CostBreakdown,
    CostCalculator,
    CostComponent,
    CostWeights,
    DrivingScenario,
)

# Primary path-planner alias: use A* by default for general occupancy-grid planning.
PathPlanner = AStarPlanner

__all__ = [
    # Path planning
    "PathPlanner",
    "AStarPlanner",
    "LatticePlanner",
    "Path",
    "HeuristicType",
    "smooth_path",
    # Behavior planning
    "BehaviorPlanner",
    "DrivingState",
    "Maneuver",
    "ManeuverType",
    "ObstacleInfo",
    "RoadContext",
    "TransitionResult",
    "VehicleState",
    # Trajectory optimization
    "TrajectoryOptimizer",
    "Trajectory",
    "TrajectoryPoint",
    "OptimizerConfig",
    # Cost calculation
    "CostCalculator",
    "CostBreakdown",
    "CostComponent",
    "CostWeights",
    "DrivingScenario",
]
