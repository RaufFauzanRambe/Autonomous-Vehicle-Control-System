"""
Physical constants, sensor specifications, vehicle parameters, and coordinate
frame definitions for the Autonomous Vehicle Control System.

All values represent realistic production specifications used throughout
perception, planning, and control modules.

Usage:
    from utils.constants import VehicleParams, SensorSpecs, PhysicsConstants

    wheel_base = VehicleParams.WHEEL_BASE
    max_decel = PhysicsConstants.MAX_COMFORTABLE_DECELERATION
    lidar_range = SensorSpecs.LIDAR_RANGE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

class PhysicsConstants:
    """Fundamental physical constants relevant to autonomous driving."""

    GRAVITY: float = 9.80665  # m/s^2 – standard gravitational acceleration
    GRAVITY_CM_S2: float = 980.665  # cm/s^2

    EARTH_RADIUS_M: float = 6_371_000.0  # Mean Earth radius in metres

    SPEED_OF_LIGHT: float = 299_792_458.0  # m/s

    AIR_DENSITY_SEA_LEVEL: float = 1.225  # kg/m^3 at 15°C, sea level

    # Tyre-road friction coefficients (dry asphalt)
    TYRE_FRICTION_DRY: float = 0.7  # Conservative estimate for dry asphalt
    TYRE_FRICTION_WET: float = 0.4  # Wet asphalt
    TYRE_FRICTION_SNOW: float = 0.2  # Snow-covered road
    TYRE_FRICTION_ICE: float = 0.1  # Icy road

    # Comfortable acceleration/deceleration limits (passenger comfort)
    MAX_COMFORTABLE_ACCELERATION: float = 2.5  # m/s^2
    MAX_COMFORTABLE_DECELERATION: float = 3.0  # m/s^2
    MAX_EMERGENCY_DECELERATION: float = 8.0  # m/s^2 (ABS threshold)

    # Lateral acceleration limits
    MAX_COMFORTABLE_LATERAL_ACCEL: float = 2.0  # m/s^2
    MAX_LATERAL_ACCEL_DRY: float = 4.0  # m/s^2 before tyre slip (dry)

    # Jerk limits (rate of change of acceleration)
    MAX_COMFORTABLE_JERK: float = 5.0  # m/s^3

    # Typical reaction times
    HUMAN_REACTION_TIME_S: float = 1.5  # Average human driver reaction time
    SYSTEM_REACTION_TIME_S: float = 0.3  # AV system sensor-to-actuator latency

    # Speed limits
    URBAN_SPEED_LIMIT_MS: float = 50.0 / 3.6  # 50 km/h → m/s
    HIGHWAY_SPEED_LIMIT_MS: float = 130.0 / 3.6  # 130 km/h → m/s
    RESIDENTIAL_SPEED_LIMIT_MS: float = 30.0 / 3.6  # 30 km/h → m/s

    # Stopping distance constants (v^2 / 2 * a)
    @staticmethod
    def stopping_distance(speed_ms: float, deceleration_ms2: float = 3.0) -> float:
        """Compute minimum stopping distance: d = v² / (2a)."""
        if deceleration_ms2 <= 0:
            return float("inf")
        return (speed_ms ** 2) / (2.0 * deceleration_ms2)

    @staticmethod
    def safe_following_distance(speed_ms: float, reaction_time_s: float = 1.0, deceleration_ms2: float = 3.0) -> float:
        """Safe following distance = reaction distance + braking distance."""
        reaction_dist = speed_ms * reaction_time_s
        braking_dist = PhysicsConstants.stopping_distance(speed_ms, deceleration_ms2)
        return reaction_dist + braking_dist


# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------

class VehicleParams:
    """Parameters for a typical mid-size autonomous vehicle (e.g., sedan).

    Based on a combination of Lincoln MKZ, Toyota Prius, and similar
    platforms commonly used for autonomous driving research.
    """

    # Dimensions
    LENGTH: float = 4.926  # m
    WIDTH: float = 1.864  # m (without mirrors)
    HEIGHT: float = 1.465  # m
    GROUND_CLEARANCE: float = 0.15  # m

    # Mass
    CURB_MASS: float = 1_850.0  # kg (vehicle without passengers/cargo)
    GROSS_MASS: float = 2_300.0  # kg (maximum loaded mass)

    # Wheel / axle
    WHEEL_BASE: float = 2.789  # m (distance between front and rear axles)
    FRONT_OVERHANG: float = 0.97  # m
    REAR_OVERHANG: float = 1.17  # m
    TRACK_WIDTH: float = 1.580  # m (distance between left and right wheels)
    WHEEL_RADIUS: float = 0.33  # m (including tyre)
    TYRE_WIDTH: float = 0.225  # m

    # Steering
    MIN_TURNING_RADIUS: float = 5.5  # m
    MAX_STEERING_ANGLE: float = math.radians(35.0)  # rad (front wheel angle)
    MAX_STEERING_RATE: float = math.radians(60.0)  # rad/s (steering wheel rate)
    STEERING_RATIO: float = 15.9  # steering wheel to front wheel ratio

    # Performance
    MAX_SPEED_MS: float = 180.0 / 3.6  # m/s (180 km/h)
    MAX_ACCELERATION: float = 3.5  # m/s^2
    MAX_DECELERATION: float = 9.0  # m/s^2 (emergency, ABS)
    ENGINE_MAX_POWER_KW: float = 200.0  # kW

    # Drag
    DRAG_COEFFICIENT: float = 0.28  # Cd
    FRONTAL_AREA_M2: float = 2.25  # m^2

    # Brake
    BRAKE_RESPONSE_TIME_S: float = 0.1  # s (hydraulic system latency)
    BRAKE_MAX_PRESSURE_BAR: float = 180.0  # bar

    # Battery / electric (for EV platforms)
    BATTERY_CAPACITY_KWH: float = 75.0  # kWh
    BATTERY_VOLTAGE_V: float = 355.2  # V (nominal)

    # Bounding box (for collision checking)
    @staticmethod
    def bounding_box(length: float = LENGTH, width: float = WIDTH) -> Tuple[float, float]:
        return (length, width)

    @staticmethod
    def turning_circle_radius(wheel_base: float = WHEEL_BASE, max_steer: float = MAX_STEERING_ANGLE) -> float:
        """Minimum turning circle radius: R = L / tan(δ)."""
        if max_steer <= 0:
            return float("inf")
        return wheel_base / math.tan(max_steer)


# ---------------------------------------------------------------------------
# Sensor specifications
# ---------------------------------------------------------------------------

class SensorSpecs:
    """Specifications for common AV sensors.

    Values are representative of production-grade sensors used in
    autonomous vehicle stacks (Velodyne, Ouster, Hesai LiDARs; FLIR,
    Allied Vision cameras; Delphi/Continental radars).
    """

    # ---- LiDAR ----
    LIDAR_RANGE: float = 120.0  # m (max detection range)
    LIDAR_RANGE_ACCURACY: float = 0.02  # m (±2 cm)
    LIDAR_HORIZONTAL_FOV: float = 360.0  # degrees
    LIDAR_VERTICAL_FOV: float = 40.0  # degrees (typical for 64/128-channel)
    LIDAR_HORIZONTAL_RESOLUTION: float = 0.2  # degrees
    LIDAR_VERTICAL_RESOLUTION: float = 0.33  # degrees
    LIDAR_SCAN_RATE_HZ: float = 10.0  # Hz
    LIDAR_POINTS_PER_FRAME: int = 130_000  # typical for 64-channel
    LIDAR_DUAL_RETURN: bool = True

    # ---- Camera ----
    CAMERA_RESOLUTION: Tuple[int, int] = (1920, 1200)  # (width, height)
    CAMERA_FPS: float = 30.0  # Hz
    CAMERA_HORIZONTAL_FOV: float = 60.0  # degrees
    CAMERA_VERTICAL_FOV: float = 38.0  # degrees
    CAMERA_FOCAL_LENGTH_PX: float = 1050.0  # pixels (approx. 6mm lens)
    CAMERA_EXPOSURE_TIME_MS: float = 5.0  # ms (typical auto-exposure)
    CAMERA_BIT_DEPTH: int = 12  # bits per channel
    CAMERA_SHUTTER_TYPE: str = "global"  # "global" or "rolling"

    # ---- Radar ----
    RADAR_RANGE: float = 200.0  # m (long-range radar)
    RADAR_RANGE_SHORT: float = 80.0  # m (short-range/corner radar)
    RADAR_RANGE_ACCURACY: float = 0.5  # m
    RADAR_VELOCITY_RANGE_MS: float = 75.0  # m/s (max measurable relative velocity)
    RADAR_VELOCITY_ACCURACY: float = 0.1  # m/s
    RADAR_HORIZONTAL_FOV: float = 18.0  # degrees (long-range)
    RADAR_HORIZONTAL_FOV_SHORT: float = 150.0  # degrees (corner)
    RADAR_ANGULAR_RESOLUTION: float = 1.0  # degrees
    RADAR_SCAN_RATE_HZ: float = 13.0  # Hz

    # ---- IMU ----
    IMU_ACCEL_RANGE_G: float = 16.0  # ±16 g
    IMU_ACCEL_NOISE_DENSITY: float = 100e-6  # g/√Hz
    IMU_GYRO_RANGE_DEG_S: float = 500.0  # ±500 °/s
    IMU_GYRO_NOISE_DENSITY: float = 0.01  # °/s/√Hz
    IMU_SAMPLE_RATE_HZ: float = 200.0  # Hz
    IMU_ACCEL_BIAS_STABILITY: float = 0.04  # mg

    # ---- GNSS/GPS ----
    GNSS_POSITION_ACCURACY_M: float = 2.5  # m (standalone GPS)
    GNSS_RTK_POSITION_ACCURACY_M: float = 0.02  # m (RTK fixed)
    GNSS_VELOCITY_ACCURACY_MS: float = 0.05  # m/s
    GNSS_UPDATE_RATE_HZ: float = 10.0  # Hz
    GNSS_LATENCY_MS: float = 50.0  # ms

    # ---- Ultrasonic ----
    ULTRASONIC_RANGE: float = 5.0  # m
    ULTRASONIC_FOV: float = 120.0  # degrees
    ULTRASONIC_ACCURACY: float = 0.03  # m

    # ---- Sensor placement (typical 5-camera + LiDAR + radar setup) ----
    SENSOR_POSITIONS: Dict[str, Tuple[float, float, float, float]] = {
        # name: (x, y, z, heading) relative to rear-axle, in metres/radians
        "lidar_top": (1.395, 0.0, 1.73, 0.0),
        "camera_front_wide": (2.35, 0.0, 1.30, 0.0),
        "camera_front_narrow": (2.35, 0.0, 1.30, 0.0),
        "camera_front_left": (2.05, 0.45, 1.20, math.radians(45)),
        "camera_front_right": (2.05, -0.45, 1.20, math.radians(-45)),
        "camera_rear": (-0.85, 0.0, 1.25, math.radians(180)),
        "radar_front_long": (2.50, 0.0, 0.55, 0.0),
        "radar_front_left": (1.80, 0.90, 0.50, math.radians(45)),
        "radar_front_right": (1.80, -0.90, 0.50, math.radians(-45)),
        "radar_rear_left": (-0.90, 0.85, 0.50, math.radians(135)),
        "radar_rear_right": (-0.90, -0.85, 0.50, math.radians(-135)),
    }


# ---------------------------------------------------------------------------
# Coordinate frame definitions
# ---------------------------------------------------------------------------

class CoordinateFrames:
    """Definitions and conventions for coordinate frames used in the AV stack.

    Follows the right-hand rule and the ISO 8855 / SAE J670 convention:

    - **EGO / Base Link**: Origin at rear axle, X forward, Y left, Z up.
    - **MAP / World**: ENU (East-North-Up) or UTM.
    - **Camera**: X right, Y down, Z forward (OpenCV convention).
    - **LiDAR**: X forward, Y left, Z up (same as ego).
    - **IMU**: Same as ego (when mounted at rear axle).
    """

    EGO: str = "ego"
    MAP: str = "map"
    CAMERA: str = "camera"
    LIDAR: str = "lidar"
    IMU: str = "imu"
    GPS: str = "gps"
    REAR_AXLE: str = "rear_axle"
    FRONT_AXLE: str = "front_axle"

    # Standard frame IDs for ROS2 TF
    FRAME_IDS: Dict[str, str] = {
        EGO: "base_link",
        MAP: "map",
        CAMERA: "camera_front_optical",
        LIDAR: "lidar_top",
        IMU: "imu_link",
        GPS: "gps_link",
        REAR_AXLE: "rear_axle",
        FRONT_AXLE: "front_axle",
    }

    # Axis conventions
    EGO_AXIS: str = "FLU"  # Forward-Left-Up
    CAMERA_AXIS: str = "RDF"  # Right-Down-Forward (OpenCV)
    LIDAR_AXIS: str = "FLU"  # Forward-Left-Up
    MAP_AXIS: str = "ENU"  # East-North-Up

    @staticmethod
    def ego_to_camera_transform() -> Dict[str, Any]:
        """Rigid transform from ego (FLU) to camera (RDF) frame.

        Rotation: X_cam = Y_ego, Y_cam = Z_ego, Z_cam = X_ego
        """
        return {
            "rotation": [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
            "translation": [0.0, 0.0, 0.0],
        }

    @staticmethod
    def camera_to_ego_transform() -> Dict[str, Any]:
        """Inverse of ego_to_camera (RDF → FLU)."""
        return {
            "rotation": [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            "translation": [0.0, 0.0, 0.0],
        }


# ---------------------------------------------------------------------------
# Detection / classification constants
# ---------------------------------------------------------------------------

class DetectionClasses:
    """Object detection class definitions and IDs."""

    UNKNOWN: int = 0
    CAR: int = 1
    TRUCK: int = 2
    BUS: int = 3
    MOTORCYCLE: int = 4
    BICYCLE: int = 5
    PEDESTRIAN: int = 6
    ANIMAL: int = 7
    BARRIER: int = 8
    TRAFFIC_CONE: int = 9
    TRAFFIC_SIGN: int = 10
    TRAFFIC_LIGHT: int = 11

    NAME_MAP: Dict[int, str] = {
        0: "unknown",
        1: "car",
        2: "truck",
        3: "bus",
        4: "motorcycle",
        5: "bicycle",
        6: "pedestrian",
        7: "animal",
        8: "barrier",
        9: "traffic_cone",
        10: "traffic_sign",
        11: "traffic_light",
    }

    # Typical bounding box sizes (length, width, height) in metres
    TYPICAL_DIMS: Dict[int, Tuple[float, float, float]] = {
        1: (4.5, 1.8, 1.5),    # car
        2: (10.0, 2.5, 3.0),   # truck
        3: (12.0, 2.6, 3.2),   # bus
        4: (2.2, 0.8, 1.4),    # motorcycle
        5: (1.8, 0.6, 1.6),    # bicycle
        6: (0.6, 0.6, 1.7),    # pedestrian
        7: (1.2, 0.5, 0.8),    # animal
        8: (1.0, 0.2, 0.8),    # barrier
        9: (0.3, 0.3, 0.9),    # traffic cone
    }


# ---------------------------------------------------------------------------
# Planning constants
# ---------------------------------------------------------------------------

class PlanningConstants:
    """Parameters for motion planning and trajectory generation."""

    # Time horizons
    PLANNING_HORIZON_S: float = 8.0  # seconds
    TRAJECTORY_RESOLUTION_S: float = 0.1  # seconds between trajectory points
    EMERGENCY_HORIZON_S: float = 3.0  # seconds for emergency manoeuvres

    # Distance thresholds
    SAFETY_BUFFER_M: float = 0.5  # m (minimum clearance to obstacles)
    LANE_CHANGE_DISTANCE_M: float = 30.0  # m (look-ahead for lane change)
    MIN_OBSTACLE_DISTANCE_M: float = 2.0  # m (minimum following distance at rest)

    # Speed thresholds
    CREEP_SPEED_MS: float = 1.5  # m/s (parking / slow manoeuvres)
    YIELD_SPEED_MS: float = 3.0  # m/s (approaching yield sign)

    # Cost weights (for optimisation-based planners)
    COST_WEIGHT_SPEED: float = 1.0
    COST_WEIGHT_ACCEL: float = 10.0
    COST_WEIGHT_JERK: float = 50.0
    COST_WEIGHT_LATERAL_OFFSET: float = 5.0
    COST_WEIGHT_HEADING_ERROR: float = 10.0
    COST_WEIGHT_OBSTACLE_PROXIMITY: float = 100.0

    # Lattice planner
    LATTICE_VELOCITY_STEPS: int = 15
    LATTICE_TIME_STEPS: int = 8
    LATTICE_LATERAL_OFFSETS: Tuple[float, ...] = (-1.5, -0.75, 0.0, 0.75, 1.5)

    # Sampling
    TRAJECTORY_SAMPLES: int = 50  # number of candidate trajectories
    BEST_N_TRAJECTORIES: int = 5  # top-N to consider


# ---------------------------------------------------------------------------
# Network / communication
# ---------------------------------------------------------------------------

class NetworkConstants:
    """Network and communication parameters for the AV system."""

    CONTROL_LOOP_RATE_HZ: float = 100.0  # Hz (10 ms control period)
    PERCEPTION_LOOP_RATE_HZ: float = 10.0  # Hz
    PLANNING_LOOP_RATE_HZ: float = 10.0  # Hz
    LOCALIZATION_LOOP_RATE_HZ: float = 50.0  # Hz

    MAX_SENSOR_LATENCY_MS: float = 100.0  # ms
    MAX_CONTROL_LATENCY_MS: float = 20.0  # ms
    MAX_ACTUATOR_LATENCY_MS: float = 10.0  # ms

    ROS_DOMAIN_ID: int = 42  # Default DDS domain ID
