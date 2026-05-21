"""
Autonomous Vehicle Control System - Localization Module

Provides state estimation, mapping, and localization capabilities
using sensor fusion of GPS, IMU, LiDAR, and odometry data.
"""

__version__ = "0.1.0"

from .ekf_localizer import EKFLocalizer
from .gps_processor import GPSProcessor
from .imu_processor import IMUProcessor
from .map_manager import MapManager
from .slam_node import SLAMNode
