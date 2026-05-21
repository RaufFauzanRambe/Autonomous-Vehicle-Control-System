"""
Autonomous Vehicle Control System - Perception Module

This module provides object detection, tracking, and sensor fusion
capabilities for autonomous vehicle perception pipelines.
"""

__version__ = "0.1.0"
__author__ = "AVCS Team"

from .object_detector import ObjectDetector
from .object_tracker import ObjectTracker
from .sensor_fusion import SensorFusionNode
from .lidar_processor import LidarProcessor
from .camera_processor import CameraProcessor
