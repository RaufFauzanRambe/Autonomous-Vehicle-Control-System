"""
IMU Processor Module for Autonomous Vehicle Localization

Handles IMU data preprocessing, bias estimation, and integration
for use in the EKF localization pipeline.
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class IMUReading:
    """Raw IMU measurement."""
    linear_acceleration: np.ndarray   # [ax, ay, az] m/s^2 in body frame
    angular_velocity: np.ndarray      # [wx, wy, wz] rad/s in body frame
    orientation: Optional[np.ndarray] = None  # [roll, pitch, yaw] if available
    timestamp: float = 0.0


class IMUProcessor:
    """
    IMU data processor for autonomous vehicle localization.

    Provides bias estimation and correction, noise filtering,
    and orientation estimation from raw IMU measurements.
    """

    def __init__(
        self,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
        accel_noise_std: float = 0.01,
        gyro_noise_std: float = 0.001,
        accel_bias_std: float = 0.0001,
        gyro_bias_std: float = 0.00001,
        gravity: float = 9.81,
        stationary_threshold: float = 0.05,
        bias_estimation_samples: int = 500,
    ):
        """
        Initialize the IMU processor.

        Args:
            accel_bias: Initial accelerometer bias [bx, by, bz].
            gyro_bias: Initial gyroscope bias [bx, by, bz].
            accel_noise_std: Accelerometer noise standard deviation.
            gyro_noise_std: Gyroscope noise standard deviation.
            accel_bias_std: Accelerometer bias random walk std.
            gyro_bias_std: Gyroscope bias random walk std.
            gravity: Local gravity magnitude.
            stationary_threshold: Velocity threshold for stationary detection.
            bias_estimation_samples: Samples needed for bias calibration.
        """
        self.accel_bias = accel_bias if accel_bias is not None else np.zeros(3)
        self.gyro_bias = gyro_bias if gyro_bias is not None else np.zeros(3)
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_std = accel_bias_std
        self.gyro_bias_std = gyro_bias_std
        self.gravity = gravity
        self.stationary_threshold = stationary_threshold
        self.bias_estimation_samples = bias_estimation_samples

        self._calibration_buffer: List[IMUReading] = []
        self._is_calibrated = accel_bias is not None and gyro_bias is not None
        self._prev_reading: Optional[IMUReading] = None
        self._integrated_velocity = np.zeros(3)
        self._integrated_position = np.zeros(3)

    def calibrate(self, readings: List[IMUReading]) -> bool:
        """
        Calibrate IMU biases from stationary readings.

        When the vehicle is stationary, the accelerometer should
        read only gravity, and the gyroscope should read zero.
        Any deviation from these values is estimated as bias.

        Args:
            readings: List of IMU readings collected while stationary.

        Returns:
            True if calibration succeeded with enough samples.
        """
        if len(readings) < self.bias_estimation_samples:
            print(f"Need {self.bias_estimation_samples} samples, got {len(readings)}")
            return False

        # Average accelerometer readings (should be [0, 0, g] when level)
        accel_samples = np.array([r.linear_acceleration for r in readings])
        mean_accel = accel_samples.mean(axis=0)

        # Estimate accelerometer bias (gravity points in z-axis)
        self.accel_bias = mean_accel - np.array([0, 0, self.gravity])

        # Average gyroscope readings (should be zero when stationary)
        gyro_samples = np.array([r.angular_velocity for r in readings])
        self.gyro_bias = gyro_samples.mean(axis=0)

        self._is_calibrated = True
        return True

    def process(self, reading: IMUReading) -> Dict:
        """
        Process a raw IMU reading.

        Applies bias correction and computes derived quantities
        for use in the localization EKF.

        Args:
            reading: Raw IMU reading.

        Returns:
            Dictionary with corrected measurements and quality metrics.
        """
        # Bias-corrected measurements
        corrected_accel = reading.linear_acceleration - self.accel_bias
        corrected_gyro = reading.angular_velocity - self.gyro_bias

        # Remove gravity from acceleration (if orientation known)
        gravity_corrected_accel = corrected_accel.copy()
        if reading.orientation is not None:
            R = self._rpy_to_rotation_matrix(reading.orientation)
            gravity_world = np.array([0, 0, self.gravity])
            gravity_body = R.T @ gravity_world
            gravity_corrected_accel = corrected_accel - gravity_body

        # Integrate for velocity and position (simple Euler integration)
        dt = 0.0
        if self._prev_reading is not None:
            dt = reading.timestamp - self._prev_reading.timestamp
            if dt > 0 and dt < 0.1:  # Sanity check
                self._integrated_velocity += gravity_corrected_accel * dt
                self._integrated_position += self._integrated_velocity * dt

        # Build measurement noise covariance
        accel_cov = np.diag([self.accel_noise_std**2] * 3)
        gyro_cov = np.diag([self.gyro_noise_std**2] * 3)

        result = {
            'corrected_acceleration': gravity_corrected_accel,
            'corrected_angular_velocity': corrected_gyro,
            'orientation': reading.orientation,
            'accel_covariance': accel_cov,
            'gyro_covariance': gyro_cov,
            'integrated_velocity': self._integrated_velocity.copy(),
            'integrated_position': self._integrated_position.copy(),
            'dt': dt,
            'timestamp': reading.timestamp,
            'is_calibrated': self._is_calibrated,
        }

        self._prev_reading = reading
        return result

    def reset_integration(self) -> None:
        """Reset the dead-reckoning integration."""
        self._integrated_velocity = np.zeros(3)
        self._integrated_position = np.zeros(3)

    @staticmethod
    def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
        """Convert roll-pitch-yaw to rotation matrix (ZYX convention)."""
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
