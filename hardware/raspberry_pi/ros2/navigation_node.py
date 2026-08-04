"""
File:        ros2/navigation_node.py
Brief:       NavigationNode — ROS 2 lifecycle node that implements
             A*/Dijkstra path planning on an occupancy grid, Nav2
             integration hooks, and waypoint following.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
class PlannerKind(str, Enum):
    """Available path planners."""

    ASTAR = "astar"
    DIJKSTRA = "dijkstra"


@dataclass(order=True)
class _AStarNode:
    """Heap entry for A*/Dijkstra priority queue."""

    cost: float
    state: Tuple[int, int] = field(compare=False)


# ----------------------------------------------------------------------
# NavigationNode
# ----------------------------------------------------------------------
class NavigationNode(LifecycleNode):
    """ROS 2 lifecycle node producing ``/navigation/path``.

    Subscriptions:
        /perception/obstacle_map   (nav_msgs/OccupancyGrid, RELIABLE)
        /localization/odom         (nav_msgs/Odometry, RELIABLE)
        /goal_pose                 (geometry_msgs/PoseStamped, RELIABLE,
                                    transient_local)

    Publishers:
        /navigation/path           (nav_msgs/Path, RELIABLE)
    """

    QOS_RELIABLE = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
    QOS_LATCHED = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    # 8-connected grid moves
    MOVES_8: List[Tuple[int, int, float]] = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
    ]

    def __init__(self, node_name: str = "navigation") -> None:
        super().__init__(node_name)
        self._declare_parameters()
        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._grid: Optional[np.ndarray] = None
        self._grid_info = None
        self._current_pose: Optional[PoseStamped] = None
        self._goal: Optional[PoseStamped] = None
        self._lock = threading.RLock()
        self._last_plan_time: float = 0.0
        self._plan_count: int = 0

        # Subs/Pubs
        self._sub_map = None
        self._sub_odom = None
        self._sub_goal = None
        self._pub_path = None
        self._timer = None
        logger.info("NavigationNode created (lifecycle)")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("planner", "astar")
        self.declare_parameter("grid_resolution_m", 0.10)
        self.declare_parameter("grid_width_m", 20.0)
        self.declare_parameter("grid_height_m", 20.0)
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("path_topic", "/navigation/path")
        self.declare_parameter("map_topic", "/perception/obstacle_map")
        self.declare_parameter("odom_topic", "/localization/odom")
        self.declare_parameter("replan_rate_hz", 1.0)
        self.declare_parameter("waypoint_tolerance_m", 0.20)
        self.declare_parameter("obstacle_inflation_m", 0.10)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        logger.info("Configuring navigation node")
        self._sub_map = self.create_subscription(
            OccupancyGrid, self.get_parameter("map_topic").value,
            self._on_map, self.QOS_RELIABLE, callback_group=self._cb_group,
        )
        self._sub_odom = self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self._on_odom, self.QOS_RELIABLE, callback_group=self._cb_group,
        )
        self._sub_goal = self.create_subscription(
            PoseStamped, self.get_parameter("goal_topic").value,
            self._on_goal, self.QOS_LATCHED, callback_group=self._cb_group,
        )
        self._pub_path = self.create_lifecycle_publisher(
            Path, self.get_parameter("path_topic").value, self.QOS_RELIABLE,
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Activating navigation node")
        rate = float(self.get_parameter("replan_rate_hz").value)
        self._timer = self.create_timer(
            1.0 / max(0.1, rate), self._on_timer,
            callback_group=self._cb_group,
        )
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Deactivating navigation node")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        logger.info("Cleaning up navigation node")
        for sub in (self._sub_map, self._sub_odom, self._sub_goal):
            if sub is not None:
                self.destroy_subscription(sub)
        if self._pub_path is not None:
            self.destroy_publisher(self._pub_path)
        self._sub_map = self._sub_odom = self._sub_goal = None
        self._pub_path = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGrid) -> None:
        with self._lock:
            shape = (msg.info.height, msg.info.width)
            self._grid = np.array(msg.data, dtype=np.int8).reshape(shape)
            self._grid_info = msg.info
            # Inflate obstacles
            self._inflate_obstacles(float(
                self.get_parameter("obstacle_inflation_m").value))

    def _inflate_obstacles(self, inflation_m: float) -> None:
        """Dilate occupied cells by ``inflation_m``."""
        if self._grid is None or self._grid_info is None:
            return
        radius = max(1, int(inflation_m / self._grid_info.resolution))
        # Simple dilation: scan a sliding window
        occupied = (self._grid >= 50).astype(np.uint8)
        try:
            import cv2
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            inflated = cv2.dilate(occupied, kernel)
            self._grid = np.where(inflated.astype(bool), 100, self._grid)
        except ImportError:
            # Fallback: scipy
            try:
                from scipy.ndimage import binary_dilation
                structure = np.ones((2 * radius + 1, 2 * radius + 1),
                                    dtype=bool)
                inflated = binary_dilation(occupied, structure=structure)
                self._grid = np.where(inflated, 100, self._grid)
            except ImportError:
                logger.warning("Neither cv2 nor scipy available — "
                               "skipping obstacle inflation")

    def _on_odom(self, msg: Odometry) -> None:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        with self._lock:
            self._current_pose = pose

    def _on_goal(self, msg: PoseStamped) -> None:
        with self._lock:
            self._goal = msg
        logger.info("New goal received: ({:.2f}, {:.2f})",
                    msg.pose.position.x, msg.pose.position.y)
        self._replan()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        now = time.monotonic()
        rate = float(self.get_parameter("replan_rate_hz").value)
        if now - self._last_plan_time < 1.0 / rate:
            return
        self._last_plan_time = now
        self._replan()

    def _replan(self) -> None:
        if (self._grid is None or self._grid_info is None
                or self._current_pose is None or self._goal is None):
            return
        with self._lock:
            grid = self._grid.copy()
            info = self._grid_info
            start = self._world_to_grid(self._current_pose.pose.position, info)
            goal = self._world_to_grid(self._goal.pose.position, info)

        if not self._in_bounds(start, grid):
            logger.warning("Start out of bounds: {}", start)
            return
        if not self._in_bounds(goal, grid):
            logger.warning("Goal out of bounds: {}", goal)
            return
        if grid[goal[0], goal[1]] >= 50:
            logger.warning("Goal is occupied: {}", goal)
            return

        planner = PlannerKind(self.get_parameter("planner").value)
        path_cells = self._plan(grid, start, goal, planner)
        if not path_cells:
            logger.warning("No path found from {} to {}", start, goal)
            return

        path_msg = self._cells_to_path_msg(path_cells, info)
        if self._pub_path is not None and self._pub_path.is_activated():
            self._pub_path.publish(path_msg)
            self._plan_count += 1
            logger.info("Published path ({} waypoints)", len(path_msg.poses))

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _world_to_grid(p: Point, info) -> Tuple[int, int]:
        """Convert a world point to (row, col) in the grid."""
        col = int((p.x - info.origin.position.x) / info.resolution)
        row = int((p.y - info.origin.position.y) / info.resolution)
        return row, col

    @staticmethod
    def _grid_to_world(row: int, col: int, info) -> Point:
        p = Point()
        p.x = info.origin.position.x + (col + 0.5) * info.resolution
        p.y = info.origin.position.y + (row + 0.5) * info.resolution
        return p

    @staticmethod
    def _in_bounds(state: Tuple[int, int], grid: np.ndarray) -> bool:
        row, col = state
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------
    def _plan(self, grid: np.ndarray, start: Tuple[int, int],
              goal: Tuple[int, int], planner: PlannerKind
              ) -> List[Tuple[int, int]]:
        """Run A* or Dijkstra. Returns list of (row, col) cells."""
        rows, cols = grid.shape
        open_heap: List[_AStarNode] = []
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        heapq.heappush(open_heap, _AStarNode(0.0, start))
        closed = set()

        def heuristic(s: Tuple[int, int]) -> float:
            if planner == PlannerKind.ASTAR:
                dx = abs(s[1] - goal[1])
                dy = abs(s[0] - goal[0])
                # Octile distance
                return (min(dx, dy) * math.sqrt(2.0)
                        + abs(dx - dy) * 1.0)
            else:
                return 0.0  # Dijkstra

        while open_heap:
            current = heapq.heappop(open_heap).state
            if current == goal:
                return self._reconstruct(came_from, current)
            if current in closed:
                continue
            closed.add(current)

            for dr, dc, cost in self.MOVES_8:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(neighbor, grid):
                    continue
                if grid[neighbor[0], neighbor[1]] >= 50:
                    continue
                tentative = g_score[current] + cost
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + heuristic(neighbor)
                    heapq.heappush(open_heap, _AStarNode(f, neighbor))

        return []

    @staticmethod
    def _reconstruct(came_from: Dict[Tuple[int, int], Tuple[int, int]],
                     current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path message conversion
    # ------------------------------------------------------------------
    def _cells_to_path_msg(self, cells: List[Tuple[int, int]], info) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        # Downsample: 1 pose per cell is too dense for large maps
        step = max(1, len(cells) // 200)
        for i, (row, col) in enumerate(cells):
            if i % step != 0 and i != len(cells) - 1:
                continue
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = self._grid_to_world(row, col, info)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "plan_count": self._plan_count,
            "has_goal": self._goal is not None,
            "has_grid": self._grid is not None,
            "grid_shape": (self._grid.shape if self._grid is not None else None),
        }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        node.configure()
        node.activate()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("Navigation node crashed: {}", exc)
    finally:
        node.deactivate()
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
