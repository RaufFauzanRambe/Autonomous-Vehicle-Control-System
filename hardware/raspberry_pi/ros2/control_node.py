"""
File:        ros2/control_node.py
Brief:       ControlNode — ROS 2 lifecycle node that subscribes to
             path/obstacles/odom and publishes cmd_vel (Twist). Runs
             PID loops on linear and angular velocity, implements a
             simple finite-state machine (IDLE → FOLLOW_PATH →
             AVOID_OBSTACLE → ESTOP), and includes a watchdog.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, Pose2D
from nav_msgs.msg import Odometry, Path
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool, Empty

# Reuse the project's PID controller
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from utils import PIDController, clamp, angular_error_deg


# ----------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------
class ControlState(enum.IntEnum):
    """Finite-state machine for the controller."""

    IDLE = 0
    FOLLOW_PATH = 1
    AVOID_OBSTACLE = 2
    ESTOP = 3


# ----------------------------------------------------------------------
# Control node
# ----------------------------------------------------------------------
class ControlNode(LifecycleNode):
    """ROS 2 lifecycle node that produces ``/cmd_vel``.

    Subscriptions (QoS-aware):
        /navigation/path          (nav_msgs/Path, RELIABLE)
        /perception/obstacles     (vision_msgs/Detection2DArray, RELIABLE)
        /localization/odom        (nav_msgs/Odometry, RELIABLE)
        /estop                    (std_msgs/Bool, RELIABLE, transient_local)

    Publishers:
        /cmd_vel                  (geometry_msgs/Twist, RELIABLE, KEEP_LAST 1)
        /control/state            (std_msgs/String, RELIABLE)
        /control/diagnostics      (std_msgs/String, RELIABLE)
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
    QOS_CMD_VEL = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )

    def __init__(self, node_name: str = "control") -> None:
        super().__init__(node_name)
        self._declare_parameters()

        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._state: ControlState = ControlState.IDLE
        self._current_pose: Optional[Pose2D] = None
        self._current_path: Optional[Path] = None
        self._path_idx: int = 0
        self._obstacle_distance_m: float = float("inf")
        self._estop_engaged: bool = False
        self._last_cmd_time: float = 0.0
        self._loop_count: int = 0

        # PID loops
        self._pid_linear = PIDController(
            kp=float(self.get_parameter("pid_linear_kp").value),
            ki=float(self.get_parameter("pid_linear_ki").value),
            kd=float(self.get_parameter("pid_linear_kd").value),
            output_limits=(0.0,
                           float(self.get_parameter("max_speed_mps").value)),
        )
        self._pid_angular = PIDController(
            kp=float(self.get_parameter("pid_angular_kp").value),
            ki=float(self.get_parameter("pid_angular_ki").value),
            kd=float(self.get_parameter("pid_angular_kd").value),
            output_limits=(
                -math.radians(float(self.get_parameter("max_steering_deg").value)),
                math.radians(float(self.get_parameter("max_steering_deg").value)),
            ),
        )

        # Subs/Pubs (created in on_configure)
        self._sub_path = None
        self._sub_obstacles = None
        self._sub_odom = None
        self._sub_estop = None
        self._pub_cmd = None
        self._pub_state = None
        self._pub_diag = None
        self._timer = None
        logger.info("ControlNode created (lifecycle)")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("loop_rate_hz", 30)
        self.declare_parameter("cruise_speed_mps", 0.4)
        self.declare_parameter("max_speed_mps", 1.5)
        self.declare_parameter("max_steering_deg", 27.0)
        self.declare_parameter("obstacle_safe_distance_m", 0.30)
        self.declare_parameter("pid_linear_kp", 0.8)
        self.declare_parameter("pid_linear_ki", 0.05)
        self.declare_parameter("pid_linear_kd", 0.01)
        self.declare_parameter("pid_angular_kp", 1.2)
        self.declare_parameter("pid_angular_ki", 0.0)
        self.declare_parameter("pid_angular_kd", 0.05)
        self.declare_parameter("estop_on_watchdog_timeout_s", 0.25)
        self.declare_parameter("waypoint_tolerance_m", 0.20)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        logger.info("Configuring control node")
        self._sub_path = self.create_subscription(
            Path, "/navigation/path", self._on_path,
            self.QOS_RELIABLE, callback_group=self._cb_group,
        )
        self._sub_obstacles = self.create_subscription(
            Detection2DArray, "/perception/obstacles", self._on_obstacles,
            self.QOS_RELIABLE, callback_group=self._cb_group,
        )
        self._sub_odom = self.create_subscription(
            Odometry, "/localization/odom", self._on_odom,
            self.QOS_RELIABLE, callback_group=self._cb_group,
        )
        self._sub_estop = self.create_subscription(
            Bool, "/estop", self._on_estop,
            self.QOS_LATCHED, callback_group=self._cb_group,
        )
        self._pub_cmd = self.create_lifecycle_publisher(
            Twist, "/cmd_vel", self.QOS_CMD_VEL,
        )
        self._pub_state = self.create_lifecycle_publisher(
            Bool, "/control/state", self.QOS_LATCHED,
        )
        self._pub_diag = self.create_lifecycle_publisher(
            Empty, "/control/heartbeat", self.QOS_RELIABLE,
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Activating control node")
        rate_hz = int(self.get_parameter("loop_rate_hz").value)
        self._timer = self.create_timer(
            1.0 / max(1, rate_hz), self._on_timer,
            callback_group=self._cb_group,
        )
        self._last_cmd_time = time.monotonic()
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Deactivating control node")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        # Stop the vehicle
        self._publish_cmd(0.0, 0.0)
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        logger.info("Cleaning up control node")
        for sub in (self._sub_path, self._sub_obstacles,
                    self._sub_odom, self._sub_estop):
            if sub is not None:
                self.destroy_subscription(sub)
        for pub in (self._pub_cmd, self._pub_state, self._pub_diag):
            if pub is not None:
                self.destroy_publisher(pub)
        self._sub_path = self._sub_obstacles = self._sub_odom = self._sub_estop = None
        self._pub_cmd = self._pub_state = self._pub_diag = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------
    def _on_path(self, msg: Path) -> None:
        self._current_path = msg
        self._path_idx = 0
        if msg.poses:
            self._set_state(ControlState.FOLLOW_PATH)
            logger.info("Received path with {} poses", len(msg.poses))
        else:
            self._set_state(ControlState.IDLE)

    def _on_obstacles(self, msg: Detection2DArray) -> None:
        # Use the closest detection's distance (if available)
        min_distance = float("inf")
        for det in msg.detections:
            for result in det.results:
                pose = result.pose.pose
                dist = math.sqrt(pose.position.x ** 2 + pose.position.y ** 2)
                if dist < min_distance:
                    min_distance = dist
        self._obstacle_distance_m = min_distance

    def _on_odom(self, msg: Odometry) -> None:
        pose = Pose2D()
        pose.x = msg.pose.pose.position.x
        pose.y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        pose.theta = math.atan2(siny_cosp, cosy_cosp)
        self._current_pose = pose

    def _on_estop(self, msg: Bool) -> None:
        self._estop_engaged = bool(msg.data)
        if self._estop_engaged:
            self._set_state(ControlState.ESTOP)
            logger.warning("E-STOP engaged")
        else:
            self._set_state(ControlState.IDLE)
            logger.info("E-STOP released")

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        self._loop_count += 1

        # Watchdog: if no cmd_vel for too long, estop
        watchdog = float(self.get_parameter("estop_on_watchdog_timeout_s").value)
        if time.monotonic() - self._last_cmd_time > watchdog and not self._estop_engaged:
            logger.error("Watchdog timeout — engaging E-stop")
            self._estop_engaged = True
            self._set_state(ControlState.ESTOP)

        # FSM
        if self._state == ControlState.ESTOP or self._estop_engaged:
            self._publish_cmd(0.0, 0.0)
            return

        if self._state == ControlState.IDLE:
            self._publish_cmd(0.0, 0.0)
            return

        if self._current_pose is None or self._current_path is None:
            return

        # Find closest waypoint ahead
        tolerance = float(self.get_parameter("waypoint_tolerance_m").value)
        poses = self._current_path.poses
        if self._path_idx >= len(poses):
            logger.info("Path complete — entering IDLE")
            self._set_state(ControlState.IDLE)
            self._publish_cmd(0.0, 0.0)
            return

        target = poses[self._path_idx].pose.position
        dx = target.x - self._current_pose.x
        dy = target.y - self._current_pose.y
        dist = math.hypot(dx, dy)
        if dist < tolerance:
            self._path_idx += 1
            return

        # Obstacle check
        safe = float(self.get_parameter("obstacle_safe_distance_m").value)
        if self._obstacle_distance_m < safe:
            if self._state != ControlState.AVOID_OBSTACLE:
                logger.warning("Obstacle at {:.2f} m — entering AVOID",
                               self._obstacle_distance_m)
            self._set_state(ControlState.AVOID_OBSTACLE)
            self._publish_cmd(0.0, 0.0)   # simple brake
            return
        elif self._state == ControlState.AVOID_OBSTACLE:
            self._set_state(ControlState.FOLLOW_PATH)

        # Path-following: pure pursuit (simplified)
        target_heading = math.atan2(dy, dx)
        heading_err = angular_error_deg(math.degrees(target_heading),
                                        math.degrees(self._current_pose.theta))
        self._pid_angular.setpoint = 0.0
        angular_correction = self._pid_angular.update(
            math.radians(heading_err)
        )
        cruise = float(self.get_parameter("cruise_speed_mps").value)
        # Slow down when turning sharply
        speed = cruise * max(0.2, 1.0 - abs(angular_correction) / 2.0)
        self._publish_cmd(speed, angular_correction)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_cmd(self, linear_mps: float, angular_rps: float) -> None:
        if self._pub_cmd is None or not self._pub_cmd.is_activated():
            return
        msg = Twist()
        msg.linear.x = float(linear_mps)
        msg.angular.z = float(angular_rps)
        self._pub_cmd.publish(msg)
        self._last_cmd_time = time.monotonic()
        if self._loop_count % 30 == 0:
            logger.debug("cmd_vel: linear={:.2f} angular={:.2f}",
                         linear_mps, angular_rps)

    def _set_state(self, new_state: ControlState) -> None:
        if new_state == self._state:
            return
        logger.info("State: {} → {}", self._state.name, new_state.name)
        self._state = new_state
        if self._pub_state is not None and self._pub_state.is_activated():
            self._pub_state.publish(Bool(data=(new_state == ControlState.ESTOP)))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    try:
        node.configure()
        node.activate()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("Control node crashed: {}", exc)
    finally:
        node.deactivate()
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
