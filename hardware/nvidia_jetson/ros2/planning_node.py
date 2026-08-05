#!/usr/bin/env python3
# =============================================================================
# File: ros2/planning_node.py
# Brief: PlanningNode — behavior planner + trajectory generator for the
#        Jetson AV stack. Implements a finite-state machine
#        (CRUISE / FOLLOW / OVERTAKE / STOP / EMERGENCY) and publishes a
#        trajectory (visualization_msgs/MarkerArray) plus a control command
#        (ackermann_msgs/AckermannDrive) consumed by the low-level
#        controller on the STM32 / Arduino.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Behavior planning and trajectory generation ROS2 node.

Topic interface
---------------
  Subscriptions:
    /perception/objects   [vision_msgs/Detection3DArray]  — fused obstacles
    /perception/obstacles [nav_msgs/OccupancyGrid]        — 2D obstacle map
    /odom                  [nav_msgs/Odometry]              — ego pose + velocity

  Publishers:
    /planning/trajectory   [visualization_msgs/MarkerArray] — visualization
    /planning/control_cmd  [ackermann_msgs/AckermannDrive]  — drive command
    /planning/state        [std_msgs/String]                — current FSM state
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Header, ColorRGBA
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

try:
    from nav_msgs.msg import OccupancyGrid, Odometry
    from ackermann_msgs.msg import AckermannDrive
    from vision_msgs.msg import Detection3DArray
    _FULL_MSGS = True
except ImportError:  # pragma: no cover
    _FULL_MSGS = False
    OccupancyGrid = None  # type: ignore
    Odometry = None  # type: ignore
    AckermannDrive = None  # type: ignore
    Detection3DArray = None  # type: ignore


# -----------------------------------------------------------------------------
# FSM states
# -----------------------------------------------------------------------------
class PlannerState(Enum):
    """Behavior planner states."""

    CRUISE = auto()       # nominal — follow the lane centerline
    FOLLOW = auto()       # a slow vehicle is ahead — match its speed
    OVERTAKE = auto()     # change lane / steer around a slow vehicle
    STOP = auto()         # stopped at an intersection / red light
    EMERGENCY = auto()    # hard brake — obstacle too close


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class EgoState:
    """Ego vehicle state."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0  # m/s


@dataclass
class Obstacle:
    """A single obstacle in ego frame."""

    class_id: int
    position: np.ndarray  # (3,)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 0.0


@dataclass
class ControlCommand:
    """Drive command sent to the low-level controller."""

    speed: float       # m/s
    steering: float    # rad (positive = left)
    brake: float = 0.0  # 0–1


# -----------------------------------------------------------------------------
# PlanningNode
# -----------------------------------------------------------------------------
class PlanningNode(Node):
    """Behavior planner + trajectory generator."""

    # Tunable parameters (defaults are conservative).
    MAX_SPEED = 8.0          # m/s
    SAFE_FOLLOW_DISTANCE = 6.0   # m
    EMERGENCY_BRAKE_DISTANCE = 3.0  # m
    LANE_WIDTH = 3.7         # m
    LOOKAHEAD_DISTANCE = 8.0  # m for pure-pursuit
    WHEELBASE = 2.6          # m
    TRAJECTORY_HORIZON_S = 2.0
    TRAJECTORY_DT = 0.1      # 10 points

    def __init__(self) -> None:
        super().__init__("planning_node")

        # --- Parameters ---------------------------------------------------
        self.declare_parameter("objects_topic", "/perception/objects")
        self.declare_parameter("obstacles_topic", "/perception/obstacles")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("trajectory_topic", "/planning/trajectory")
        self.declare_parameter("control_topic", "/planning/control_cmd")
        self.declare_parameter("state_topic", "/planning/state")
        self.declare_parameter("max_speed", self.MAX_SPEED)
        self.declare_parameter("control_rate_hz", 20.0)

        objects_topic = self.get_parameter("objects_topic").value
        obstacles_topic = self.get_parameter("obstacles_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        traj_topic = self.get_parameter("trajectory_topic").value
        ctrl_topic = self.get_parameter("control_topic").value
        state_topic = self.get_parameter("state_topic").value
        self.max_speed = float(self.get_parameter("max_speed").value)
        rate_hz = float(self.get_parameter("control_rate_hz").value)

        # --- QoS ----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # --- Subscribers --------------------------------------------------
        if _FULL_MSGS:
            self.sub_objects = self.create_subscription(
                Detection3DArray, objects_topic, self._on_objects, sensor_qos)
            self.sub_obstacles = self.create_subscription(
                OccupancyGrid, obstacles_topic, self._on_obstacles, sensor_qos)
            self.sub_odom = self.create_subscription(
                Odometry, odom_topic, self._on_odom, sensor_qos)
        else:
            self.sub_objects = self.sub_obstacles = self.sub_odom = None
            self.get_logger().warning(
                "Required message types not available — running in stub mode.")

        # --- Publishers ---------------------------------------------------
        self.pub_traj = self.create_publisher(
            MarkerArray, traj_topic, reliable_qos)
        if _FULL_MSGS:
            self.pub_cmd = self.create_publisher(
                AckermannDrive, ctrl_topic, reliable_qos)
        else:
            self.pub_cmd = None
        self.pub_state = self.create_publisher(
            String, state_topic, reliable_qos)

        # --- Internal state ----------------------------------------------
        self.ego = EgoState()
        self.obstacles: List[Obstacle] = []
        self.state = PlannerState.CRUISE
        self.target_speed = self.max_speed
        self._cmd_history: Deque[ControlCommand] = deque(maxlen=20)

        # Control loop timer.
        self._timer = self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

        self.get_logger().info(
            f"PlanningNode ready — max_speed={self.max_speed} m/s, "
            f"rate={rate_hz:.1f} Hz, initial_state={self.state.name}")

    # ------------------------------------------------------------------ #
    # Subscribers
    # ------------------------------------------------------------------ #
    def _on_objects(self, msg) -> None:
        """Cache the latest 3D obstacle list."""
        self.obstacles = []
        for det in msg.detections:
            pos = np.array([
                det.bbox.center.position.x,
                det.bbox.center.position.y,
                det.bbox.center.position.z,
            ], dtype=np.float32)
            dims = np.array([
                det.bbox.size.x, det.bbox.size.y, det.bbox.size.z],
                dtype=np.float32)
            hyp = det.results[0] if det.results else None
            class_id = int(hyp.id) if hyp and hyp.id.isdigit() else -1
            score = float(hyp.score) if hyp else 0.0
            self.obstacles.append(Obstacle(
                class_id=class_id, position=pos, dimensions=dims,
                confidence=score))

    def _on_obstacles(self, msg) -> None:
        """(Optional) cache the occupancy grid for planning. Currently the
        planner relies on the object list; the grid is consumed when
        planning lane changes."""
        pass

    def _on_odom(self, msg) -> None:
        """Update ego state from odometry."""
        self.ego.x = msg.pose.pose.position.x
        self.ego.y = msg.pose.pose.position.y
        # Yaw from quaternion.
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ego.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.ego.speed = math.sqrt(
            msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)

    # ------------------------------------------------------------------ #
    # Main control loop
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        """One planning iteration: update FSM, generate trajectory, publish."""
        # Find the closest in-lane obstacle ahead.
        lead = self._find_lead_obstacle()
        lead_distance = float(np.linalg.norm(lead.position[:2])) if lead else float("inf")

        # --- State machine ----------------------------------------------
        new_state = self._transition(lead, lead_distance)
        if new_state != self.state:
            self.get_logger().info(
                f"State transition: {self.state.name} -> {new_state.name}")
            self.state = new_state

        # --- Target speed selection -------------------------------------
        if self.state == PlannerState.STOP:
            self.target_speed = 0.0
        elif self.state == PlannerState.EMERGENCY:
            self.target_speed = 0.0
        elif self.state == PlannerState.FOLLOW and lead is not None:
            # Match lead speed at safe distance.
            rel_v = float(lead.velocity[0]) if lead.velocity.size else 0.0
            desired = max(0.0, rel_v + self.ego.speed - 0.5)
            # Slow down if too close.
            if lead_distance < self.SAFE_FOLLOW_DISTANCE:
                desired *= lead_distance / self.SAFE_FOLLOW_DISTANCE
            self.target_speed = min(desired, self.max_speed)
        elif self.state == PlannerState.OVERTAKE:
            self.target_speed = self.max_speed
        else:  # CRUISE
            self.target_speed = self.max_speed

        # --- Trajectory generation (pure-pursuit) -----------------------
        trajectory = self._generate_trajectory(lead)

        # --- Control command --------------------------------------------
        cmd = self._compute_control(trajectory)
        self._cmd_history.append(cmd)

        # --- Publish ----------------------------------------------------
        self._publish_trajectory(trajectory)
        self._publish_command(cmd)
        self._publish_state()

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #
    def _transition(
        self, lead: Optional[Obstacle], lead_distance: float
    ) -> PlannerState:
        """Decide the next FSM state from the current situation."""
        if lead is not None and lead_distance < self.EMERGENCY_BRAKE_DISTANCE:
            return PlannerState.EMERGENCY
        if lead is not None and lead_distance < self.SAFE_FOLLOW_DISTANCE:
            # If we are slower than the lead, just follow.
            if self.ego.speed > float(lead.velocity[0] if lead.velocity.size else 0.0) + 1.0:
                return PlannerState.OVERTAKE
            return PlannerState.FOLLOW
        return PlannerState.CRUISE

    # ------------------------------------------------------------------ #
    # Perception helpers
    # ------------------------------------------------------------------ #
    def _find_lead_obstacle(self) -> Optional[Obstacle]:
        """Return the closest obstacle within ±1 m of the lane centerline."""
        best: Optional[Obstacle] = None
        best_dist = float("inf")
        for obs in self.obstacles:
            # In ego frame: x is forward, y is lateral.
            if obs.position[0] <= 0:
                continue
            if abs(obs.position[1]) > 1.0:
                continue
            d = float(obs.position[0])
            if d < best_dist:
                best, best_dist = obs, d
        return best

    # ------------------------------------------------------------------ #
    # Trajectory generation
    # ------------------------------------------------------------------ #
    def _generate_trajectory(
        self, lead: Optional[Obstacle]
    ) -> List[Tuple[float, float, float]]:
        """Generate a list of (x, y, yaw) waypoints in ego frame.

        Uses a simple pure-pursuit lookahead along the lane centerline;
        when overtaking, the trajectory is shifted laterally.
        """
        n_points = int(self.TRAJECTORY_HORIZON_S / self.TRAJECTORY_DT)
        trajectory: List[Tuple[float, float, float]] = []
        # Lateral offset for overtake maneuver.
        lateral_offset = 0.0
        if self.state == PlannerState.OVERTAKE and lead is not None:
            # Shift to the left of the lead vehicle.
            lateral_offset = self.LANE_WIDTH * 0.8

        # Build a smooth cubic curve that joins the ego pose to a
        # lookahead point ahead.
        x_target = self.LOOKAHEAD_DISTANCE
        y_target = lateral_offset
        for i in range(n_points):
            t = (i + 1) / n_points
            x = x_target * t
            y = y_target * (3 * t ** 2 - 2 * t ** 3)
            yaw = math.atan2(
                6 * y_target * t * (1 - t) / max(x_target, 1e-3), 1.0)
            trajectory.append((x, y, yaw))
        return trajectory

    # ------------------------------------------------------------------ #
    # Control computation
    # ------------------------------------------------------------------ #
    def _compute_control(
        self, trajectory: List[Tuple[float, float, float]]
    ) -> ControlCommand:
        """Convert trajectory into a (speed, steering, brake) command."""
        if not trajectory:
            return ControlCommand(speed=0.0, steering=0.0, brake=1.0)
        # Pure pursuit — steer toward the lookahead point.
        x_l, y_l, _ = trajectory[0]
        ld = max(math.hypot(x_l, y_l), 1e-3)
        alpha = math.atan2(y_l, x_l)
        steering = math.atan2(
            2.0 * self.WHEELBASE * math.sin(alpha), ld)

        # Speed command.
        speed = self.target_speed
        # Emergency brake.
        if self.state == PlannerState.EMERGENCY:
            return ControlCommand(speed=0.0, steering=steering, brake=1.0)
        # Smooth acceleration toward target.
        if self.ego.speed < self.target_speed - 0.1:
            brake = 0.0
        elif self.ego.speed > self.target_speed + 0.1:
            speed = self.target_speed
            brake = 0.3
        else:
            brake = 0.0
        return ControlCommand(speed=speed, steering=steering, brake=brake)

    # ------------------------------------------------------------------ #
    # Publishers
    # ------------------------------------------------------------------ #
    def _publish_trajectory(
        self, trajectory: List[Tuple[float, float, float]]
    ) -> None:
        """Publish a MarkerArray visualization of the planned trajectory."""
        msg = MarkerArray()
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "trajectory"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.1  # line width
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)
        m.lifetime.sec = 0
        m.lifetime.nanosec = int(0.2 * 1e9)
        for x, y, _ in trajectory:
            m.points.append(Point(x=float(x), y=float(y), z=0.0))
        msg.markers.append(m)
        self.pub_traj.publish(msg)

    def _publish_command(self, cmd: ControlCommand) -> None:
        """Publish an AckermannDrive control command."""
        if not _FULL_MSGS or self.pub_cmd is None:
            return
        msg = AckermannDrive()
        msg.speed = float(cmd.speed)
        msg.steering_angle = float(cmd.steering)
        msg.acceleration = 0.0
        msg.jerk = 0.0
        msg.steering_angle_velocity = 0.0
        self.pub_cmd.publish(msg)

    def _publish_state(self) -> None:
        """Publish the current FSM state as a String."""
        msg = String()
        msg.data = self.state.name
        self.pub_state.publish(msg)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(args=None) -> None:  # pragma: no cover
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
