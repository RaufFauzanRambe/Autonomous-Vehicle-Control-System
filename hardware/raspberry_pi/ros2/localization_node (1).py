"""
File:        ros2/localization_node.py
Brief:       LocalizationNode — ROS 2 lifecycle node that fuses GPS + IMU
             using a simple Extended Kalman Filter and publishes
             nav_msgs/Odometry and the odom → base_link transform.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_ros import TransformBroadcaster, TransformStamped

from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
from std_msgs.msg import Header

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from utils import haversine_m, bearing_to_waypoint_deg


# ----------------------------------------------------------------------
# State vector: [x, y, theta, vx, vy, omega]
# ----------------------------------------------------------------------
STATE_X = 0
STATE_Y = 1
STATE_THETA = 2
STATE_VX = 3
STATE_VY = 4
STATE_OMEGA = 5
STATE_DIM = 6


@dataclass(slots=True)
class EKFState:
    """EKF state and covariance."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    P: np.ndarray = field(default_factory=lambda: np.eye(STATE_DIM) * 1.0)

    def reset(self) -> None:
        self.x = np.zeros(STATE_DIM)
        self.P = np.eye(STATE_DIM) * 1.0


# ----------------------------------------------------------------------
# LocalizationNode
# ----------------------------------------------------------------------
class LocalizationNode(LifecycleNode):
    """GPS + IMU fusion node producing Odometry + tf.

    The filter is a 6-state EKF:
        state = [x, y, theta, vx, vy, omega]
    Prediction uses IMU gyro + kinematic model.
    Update uses GPS position (lat/lon → ENU around the initial fix).
    """

    QOS_SENSOR = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=DurabilityPolicy.VOLATILE,
    )
    QOS_RELIABLE = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    EARTH_RADIUS_M = 6_371_000.0

    def __init__(self, node_name: str = "localization") -> None:
        super().__init__(node_name)
        self._declare_parameters()
        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._state = EKFState()
        self._origin_lat: Optional[float] = None
        self._origin_lon: Optional[float] = None
        self._last_imu_time: Optional[float] = None
        self._last_gps_time: Optional[float] = None
        self._imu_count = 0
        self._gps_count = 0
        self._ekf_alpha = float(self.get_parameter("ekf_alpha").value)

        # Process noise
        q_pos = 0.05
        q_theta = 0.01
        q_vel = 0.5
        self._Q = np.diag([q_pos, q_pos, q_theta, q_vel, q_vel, q_theta])

        # Measurement noise for GPS (meters)
        self._R_gps = np.diag([2.0, 2.0])

        # Subs/Pubs
        self._sub_gps = None
        self._sub_imu = None
        self._pub_odom = None
        self._tf_broadcaster: Optional[TransformBroadcaster] = None
        self._timer = None
        self._lock = threading.RLock()
        logger.info("LocalizationNode created (lifecycle)")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("gps_topic", "/gps/fix")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("odom_topic", "/localization/odom")
        self.declare_parameter("publish_rate_hz", 30)
        self.declare_parameter("ekf_alpha", 0.95)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("initial_lat", 0.0)
        self.declare_parameter("initial_lon", 0.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        logger.info("Configuring localization node")
        self._sub_gps = self.create_subscription(
            NavSatFix, self.get_parameter("gps_topic").value,
            self._on_gps, self.QOS_SENSOR, callback_group=self._cb_group,
        )
        self._sub_imu = self.create_subscription(
            Imu, self.get_parameter("imu_topic").value,
            self._on_imu, self.QOS_SENSOR, callback_group=self._cb_group,
        )
        self._pub_odom = self.create_lifecycle_publisher(
            Odometry, self.get_parameter("odom_topic").value,
            self.QOS_RELIABLE,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        # Set origin from parameters if provided
        lat0 = float(self.get_parameter("initial_lat").value)
        lon0 = float(self.get_parameter("initial_lon").value)
        if lat0 != 0.0 or lon0 != 0.0:
            self._origin_lat = lat0
            self._origin_lon = lon0
            logger.info("GPS origin: ({:.6f}, {:.6f})", lat0, lon0)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Activating localization node")
        rate = int(self.get_parameter("publish_rate_hz").value)
        self._timer = self.create_timer(
            1.0 / max(1, rate), self._on_timer,
            callback_group=self._cb_group,
        )
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        logger.info("Deactivating localization node")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        logger.info("Cleaning up localization node")
        for sub in (self._sub_gps, self._sub_imu):
            if sub is not None:
                self.destroy_subscription(sub)
        if self._pub_odom is not None:
            self.destroy_publisher(self._pub_odom)
        self._sub_gps = self._sub_imu = None
        self._pub_odom = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # GPS callback → position update
    # ------------------------------------------------------------------
    def _on_gps(self, msg: NavSatFix) -> None:
        self._gps_count += 1
        if self._origin_lat is None or self._origin_lon is None:
            # Capture the first fix as origin
            self._origin_lat = msg.latitude
            self._origin_lon = msg.longitude
            logger.info("GPS origin captured: ({:.6f}, {:.6f})",
                        self._origin_lat, self._origin_lon)
            return

        # Convert lat/lon → local ENU (East-North-Up, flat-Earth approx)
        east, north = self._latlon_to_enu(msg.latitude, msg.longitude)
        # Update step
        with self._lock:
            z = np.array([east, north])
            H = np.zeros((2, STATE_DIM))
            H[0, STATE_X] = 1.0
            H[1, STATE_Y] = 1.0
            y = z - H @ self._state.x
            S = H @ self._state.P @ H.T + self._R_gps
            try:
                K = self._state.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                logger.warning("EKF S matrix singular; skipping GPS update")
                return
            self._state.x = self._state.x + K @ y
            self._state.P = (np.eye(STATE_DIM) - K @ H) @ self._state.P
        self._last_gps_time = time.monotonic()

    def _latlon_to_enu(self, lat: float, lon: float) -> tuple[float, float]:
        """Flat-Earth conversion from lat/lon to East/North meters."""
        if self._origin_lat is None or self._origin_lon is None:
            return (0.0, 0.0)
        # East  = R * cos(lat0) * (lon - lon0) * pi/180
        # North = R * (lat - lat0) * pi/180
        lat0_rad = math.radians(self._origin_lat)
        east  = self.EARTH_RADIUS_M * math.cos(lat0_rad) \
                * math.radians(lon - self._origin_lon)
        north = self.EARTH_RADIUS_M * math.radians(lat - self._origin_lat)
        return east, north

    # ------------------------------------------------------------------
    # IMU callback → prediction step
    # ------------------------------------------------------------------
    def _on_imu(self, msg: Imu) -> None:
        self._imu_count += 1
        now = time.monotonic()
        dt = (now - self._last_imu_time) if self._last_imu_time else 0.0
        self._last_imu_time = now
        if dt <= 0 or dt > 1.0:
            return

        # Extract yaw rate (body-frame z-axis angular velocity)
        omega = float(msg.angular_velocity.z)
        # Extract yaw from quaternion (assuming msg.orientation is valid)
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_imu = math.atan2(siny_cosp, cosy_cosp)

        with self._lock:
            theta = self._state.x[STATE_THETA]
            vx = self._state.x[STATE_VX]
            vy = self._state.x[STATE_VY]

            # State transition function f(x)
            x_new = self._state.x.copy()
            x_new[STATE_X]     += (vx * math.cos(theta)
                                   - vy * math.sin(theta)) * dt
            x_new[STATE_Y]     += (vx * math.sin(theta)
                                   + vy * math.cos(theta)) * dt
            x_new[STATE_THETA] += omega * dt
            # Velocity carries forward (no input model)
            # x_new[STATE_VX..] unchanged

            # Jacobian F (numerical/analytic)
            F = np.eye(STATE_DIM)
            F[STATE_X, STATE_VX] = math.cos(theta) * dt
            F[STATE_X, STATE_VY] = -math.sin(theta) * dt
            F[STATE_X, STATE_THETA] = (-vx * math.sin(theta)
                                       - vy * math.cos(theta)) * dt
            F[STATE_Y, STATE_VX] = math.sin(theta) * dt
            F[STATE_Y, STATE_VY] = math.cos(theta) * dt
            F[STATE_Y, STATE_THETA] = (vx * math.cos(theta)
                                       - vy * math.sin(theta)) * dt
            F[STATE_THETA, STATE_OMEGA] = dt

            self._state.x = x_new
            self._state.P = F @ self._state.P @ F.T + self._Q * dt

            # Complementary fusion of yaw from IMU directly
            alpha = self._ekf_alpha
            self._state.x[STATE_THETA] = (
                alpha * self._state.x[STATE_THETA]
                + (1.0 - alpha) * yaw_imu
            )

    # ------------------------------------------------------------------
    # Timer → publish odom + tf
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        if self._pub_odom is None or not self._pub_odom.is_activated():
            return
        with self._lock:
            x = float(self._state.x[STATE_X])
            y = float(self._state.x[STATE_Y])
            theta = float(self._state.x[STATE_THETA])
            vx = float(self._state.x[STATE_VX])
            vy = float(self._state.x[STATE_VY])
            vtheta = float(self._state.x[STATE_OMEGA])

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.get_parameter("frame_id").value
        odom.child_frame_id = self.get_parameter("child_frame_id").value
        odom.pose.pose = self._pose_from_xytheta(x, y, theta)
        odom.twist.twist = Twist(
            linear=Vector3(x=vx, y=vy, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=vtheta),
        )
        # Add covariance (diagonal from P)
        for i in range(6):
            odom.pose.covariance[i * 6 + i] = float(self._state.P[i, i])
        self._pub_odom.publish(odom)

        # Broadcast tf
        if self._tf_broadcaster is not None:
            t = TransformStamped()
            t.header = odom.header
            t.child_frame_id = odom.child_frame_id
            t.header.frame_id = odom.header.frame_id
            t.transform.translation = Vector3(x=x, y=y, z=0.0)
            t.transform.rotation = odom.pose.pose.orientation
            self._tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_from_xytheta(x: float, y: float, theta: float) -> Pose:
        pose = Pose()
        pose.position = Point(x=x, y=y, z=0.0)
        # Yaw → quaternion
        half = theta * 0.5
        pose.orientation = Quaternion(
            x=0.0, y=0.0,
            z=float(math.sin(half)),
            w=float(math.cos(half)),
        )
        return pose

    def stats(self) -> dict:
        return {
            "imu_count": self._imu_count,
            "gps_count": self._gps_count,
            "origin_lat": self._origin_lat,
            "origin_lon": self._origin_lon,
            "state": self._state.x.tolist(),
        }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        node.configure()
        node.activate()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("Localization node crashed: {}", exc)
    finally:
        node.deactivate()
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
