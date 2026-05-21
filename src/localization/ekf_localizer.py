"""
Extended Kalman Filter Localizer for Autonomous Vehicle

Implements EKF-based state estimation fusing GPS, IMU, wheel odometry,
and LiDAR localization for robust vehicle pose estimation.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Full vehicle state estimate."""
    x: float = 0.0             # Position x (meters)
    y: float = 0.0             # Position y (meters)
    z: float = 0.0             # Position z (meters)
    roll: float = 0.0          # Roll angle (radians)
    pitch: float = 0.0         # Pitch angle (radians)
    yaw: float = 0.0           # Yaw angle (radians)
    vx: float = 0.0            # Velocity x (m/s)
    vy: float = 0.0            # Velocity y (m/s)
    vz: float = 0.0            # Velocity z (m/s)
    vroll: float = 0.0         # Roll rate (rad/s)
    vpitch: float = 0.0        # Pitch rate (rad/s)
    vyaw: float = 0.0          # Yaw rate (rad/s)
    ax: float = 0.0            # Acceleration x (m/s^2)
    ay: float = 0.0            # Acceleration y (m/s^2)
    timestamp: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def orientation(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.z,
            self.roll, self.pitch, self.yaw,
            self.vx, self.vy, self.vz,
            self.vroll, self.vpitch, self.vyaw,
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray, timestamp: float = 0.0) -> 'VehicleState':
        return cls(
            x=vec[0], y=vec[1], z=vec[2],
            roll=vec[3], pitch=vec[4], yaw=vec[5],
            vx=vec[6], vy=vec[7], vz=vec[8],
            vroll=vec[9], vpitch=vec[10], vyaw=vec[11],
            timestamp=timestamp,
        )


class EKFLocalizer:
    """
    EKF-based vehicle localizer fusing multiple sensor sources.

    State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
    (12 dimensions)

    Supports updates from:
    - GPS: position measurements
    - IMU: orientation and angular velocity
    - Wheel odometry: velocity
    - LiDAR: position correction via map matching
    """

    STATE_DIM = 12

    def __init__(
        self,
        initial_state: Optional[VehicleState] = None,
        initial_covariance: Optional[np.ndarray] = None,
    ):
        """
        Initialize the EKF localizer.

        Args:
            initial_state: Initial vehicle state. Defaults to origin.
            initial_covariance: Initial state covariance. Defaults to high uncertainty.
        """
        if initial_state is not None:
            self._state = initial_state.to_vector()
        else:
            self._state = np.zeros(self.STATE_DIM)

        if initial_covariance is not None:
            self._covariance = initial_covariance
        else:
            self._covariance = np.eye(self.STATE_DIM) * 10.0
            # Higher uncertainty on velocities and angular rates
            self._covariance[6:12, 6:12] = np.eye(6) * 100.0

        # Process noise parameters
        self._process_noise_pos = 0.5
        self._process_noise_rot = 0.01
        self._process_noise_vel = 1.0
        self._process_noise_angvel = 0.1

        # Measurement noise parameters
        self._gps_noise = np.diag([2.0, 2.0, 5.0])          # GPS position noise
        self._imu_orient_noise = np.diag([0.01, 0.01, 0.01]) # IMU orientation noise
        self._imu_angvel_noise = np.diag([0.005, 0.005, 0.005])  # IMU angular vel noise
        self._odom_noise = np.diag([0.1, 0.1, 0.1])          # Odometry velocity noise
        self._lidar_noise = np.diag([0.1, 0.1, 0.1, 0.005, 0.005, 0.01])  # LiDAR pose noise

    @property
    def state(self) -> VehicleState:
        """Get current state estimate."""
        return VehicleState.from_vector(self._state)

    @property
    def covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self._covariance.copy()

    def predict(self, dt: float, imu_accel: Optional[np.ndarray] = None) -> None:
        """
        EKF predict step using motion model.

        Args:
            dt: Time step in seconds.
            imu_accel: IMU acceleration [ax, ay, az] in body frame.
        """
        # State transition matrix (constant velocity model)
        F = np.eye(self.STATE_DIM)
        F[0, 6] = dt   # x += vx * dt
        F[1, 7] = dt   # y += vy * dt
        F[2, 8] = dt   # z += vz * dt
        F[3, 9] = dt   # roll += vroll * dt
        F[4, 10] = dt  # pitch += vpitch * dt
        F[5, 11] = dt  # yaw += vyaw * dt

        # Rotate velocity to world frame if IMU acceleration available
        if imu_accel is not None:
            R = self._rotation_matrix(self._state[3:6])
            world_accel = R @ imu_accel
            # Apply acceleration to state
            self._state[6:9] += world_accel * dt

        # Predict state
        self._state = F @ self._state

        # Process noise covariance
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        Q[0:3, 0:3] = np.eye(3) * self._process_noise_pos * dt**2
        Q[3:6, 3:6] = np.eye(3) * self._process_noise_rot * dt**2
        Q[6:9, 6:9] = np.eye(3) * self._process_noise_vel * dt**2
        Q[9:12, 9:12] = np.eye(3) * self._process_noise_angvel * dt**2

        # Predict covariance
        self._covariance = F @ self._covariance @ F.T + Q

    def update_gps(self, gps_position: np.ndarray) -> None:
        """
        EKF update with GPS position measurement.

        Args:
            gps_position: GPS measured position [x, y, z].
        """
        H = np.zeros((3, self.STATE_DIM))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z

        self._update(H, gps_position, self._gps_noise)

    def update_imu_orientation(self, orientation: np.ndarray) -> None:
        """
        EKF update with IMU orientation measurement.

        Args:
            orientation: IMU orientation [roll, pitch, yaw].
        """
        H = np.zeros((3, self.STATE_DIM))
        H[0, 3] = 1.0  # roll
        H[1, 4] = 1.0  # pitch
        H[2, 5] = 1.0  # yaw

        self._update(H, orientation, self._imu_orient_noise)

    def update_imu_angular_velocity(self, angular_vel: np.ndarray) -> None:
        """
        EKF update with IMU angular velocity measurement.

        Args:
            angular_vel: Angular velocity [vroll, vpitch, vyaw].
        """
        H = np.zeros((3, self.STATE_DIM))
        H[0, 9] = 1.0   # vroll
        H[1, 10] = 1.0  # vpitch
        H[2, 11] = 1.0  # vyaw

        self._update(H, angular_vel, self._imu_angvel_noise)

    def update_odometry(self, velocity: np.ndarray) -> None:
        """
        EKF update with wheel odometry velocity measurement.

        Args:
            velocity: Wheel odometry velocity [vx, vy, vz] in body frame.
        """
        # Transform body velocity to world frame
        R = self._rotation_matrix(self._state[3:6])
        world_vel = R @ velocity

        H = np.zeros((3, self.STATE_DIM))
        H[0, 6] = 1.0  # vx
        H[1, 7] = 1.0  # vy
        H[2, 8] = 1.0  # vz

        self._update(H, world_vel, self._odom_noise)

    def update_lidar(self, pose: np.ndarray) -> None:
        """
        EKF update with LiDAR localization pose measurement.

        Args:
            pose: LiDAR-estimated pose [x, y, z, roll, pitch, yaw].
        """
        H = np.zeros((6, self.STATE_DIM))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        H[3, 3] = 1.0  # roll
        H[4, 4] = 1.0  # pitch
        H[5, 5] = 1.0  # yaw

        self._update(H, pose, self._lidar_noise)

    def _update(self, H: np.ndarray, measurement: np.ndarray, R: np.ndarray) -> None:
        """Generic EKF update step."""
        # Innovation
        y = measurement - H @ self._state

        # Innovation covariance
        S = H @ self._covariance @ H.T + R

        # Kalman gain
        K = self._covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self._state = self._state + K @ y

        # Update covariance (Joseph form)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self._covariance = I_KH @ self._covariance @ I_KH.T + K @ R @ K.T

        # Ensure symmetry
        self._covariance = (self._covariance + self._covariance.T) / 2

    @staticmethod
    def _rotation_matrix(rpy: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix from roll-pitch-yaw angles.

        Uses ZYX convention (yaw-pitch-roll).
        """
        roll, pitch, yaw = rpy

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        return R
