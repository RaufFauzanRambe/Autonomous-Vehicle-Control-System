#!/usr/bin/env python3
# =============================================================================
# File: ros2/launch.py
# Brief: ROS2 launch file for the Jetson AV stack — brings up AINode,
#        PerceptionNode, and PlanningNode with GPU parameters, image
#        transport remappings, and parameter file loading.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Launch file: bring up the full Jetson AV perception + planning stack.

Usage:
  ros2 launch autonomous_vehicle_hardware jetson_av.launch.py \\
      engine_path:=/models/yolov5s.engine \\
      input_topic:=/camera/image_raw
"""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration, PythonExpression, PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# -----------------------------------------------------------------------------
# Default parameter values — overridable via launch arguments.
# -----------------------------------------------------------------------------
DEFAULT_ENGINE_PATH = "/models/yolov5s.engine"
DEFAULT_CLASS_NAMES = "/models/coco.names"
DEFAULT_INPUT_TOPIC = "/camera/image_raw"
DEFAULT_LIDAR_TOPIC = "/lidar/points"
DEFAULT_CONTROL_TOPIC = "/planning/control_cmd"
DEFAULT_MAX_SPEED = "8.0"


# -----------------------------------------------------------------------------
# Launch description
# -----------------------------------------------------------------------------
def generate_launch_description() -> LaunchDescription:
    """Build the launch description for the AV Jetson stack."""

    # --- Launch arguments -----------------------------------------------
    engine_path_arg = DeclareLaunchArgument(
        "engine_path", default_value=DEFAULT_ENGINE_PATH,
        description="Path to the TensorRT .engine file.")
    class_names_arg = DeclareLaunchArgument(
        "class_names", default_value=DEFAULT_CLASS_NAMES,
        description="Path to a class names file (JSON or one-per-line).")
    input_topic_arg = DeclareLaunchArgument(
        "input_topic", default_value=DEFAULT_INPUT_TOPIC,
        description="Image topic to subscribe to.")
    lidar_topic_arg = DeclareLaunchArgument(
        "lidar_topic", default_value=DEFAULT_LIDAR_TOPIC,
        description="LiDAR PointCloud2 topic.")
    control_topic_arg = DeclareLaunchArgument(
        "control_topic", default_value=DEFAULT_CONTROL_TOPIC,
        description="Topic the STM32 / Arduino controller subscribes to.")
    max_speed_arg = DeclareLaunchArgument(
        "max_speed", default_value=DEFAULT_MAX_SPEED,
        description="Maximum planner speed (m/s).")
    device_id_arg = DeclareLaunchArgument(
        "device_id", default_value="0",
        description="CUDA device index (always 0 on Jetson).")
    publish_annotated_arg = DeclareLaunchArgument(
        "publish_annotated", default_value="true",
        description="If true, AINode publishes a visualized image.")
    enable_perception_arg = DeclareLaunchArgument(
        "enable_perception", default_value="true",
        description="If true, start the perception (fusion) node.")
    enable_planning_arg = DeclareLaunchArgument(
        "enable_planning", default_value="true",
        description="If true, start the planning node.")
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info",
        description="ROS2 log level (debug/info/warn/error).")

    # --- AINode ---------------------------------------------------------
    ai_node = Node(
        package="autonomous_vehicle_hardware",
        executable="ai_node",
        name="ai_node",
        output="screen",
        parameters=[{
            "engine_path": LaunchConfiguration("engine_path"),
            "class_names": LaunchConfiguration("class_names"),
            "input_topic": LaunchConfiguration("input_topic"),
            "detection_topic": "/ai/detections",
            "annotated_topic": "/ai/image_annotated",
            "fps_topic": "/ai/fps",
            "publish_annotated": LaunchConfiguration("publish_annotated"),
            "confidence": 0.45,
            "device_id": LaunchConfiguration("device_id"),
            "yolo_version": "v8",
            "max_queue_size": 4,
        }],
        remappings=[
            ("/ai/detections", "/perception/detections_2d"),
        ],
        arguments=["--ros-args", "--log-level",
                    LaunchConfiguration("log_level")],
        condition=UnlessCondition(
            PythonExpression(["'", LaunchConfiguration("engine_path"),
                              "' == ''"])),
    )

    # --- PerceptionNode -------------------------------------------------
    perception_node = Node(
        package="autonomous_vehicle_hardware",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[{
            "det_topic": "/perception/detections_2d",
            "lidar_topic": LaunchConfiguration("lidar_topic"),
            "camera_info_topic": "/camera/camera_info",
            "objects_topic": "/perception/objects",
            "obstacles_topic": "/perception/obstacles",
            "drivable_topic": "/perception/drivable",
            "grid_resolution": 0.2,
            "grid_size_x": 100,
            "grid_size_y": 100,
            "max_object_age_s": 1.0,
            "publish_rate_hz": 20.0,
        }],
        arguments=["--ros-args", "--log-level",
                    LaunchConfiguration("log_level")],
        condition=IfCondition(LaunchConfiguration("enable_perception")),
    )

    # --- PlanningNode ---------------------------------------------------
    planning_node = Node(
        package="autonomous_vehicle_hardware",
        executable="planning_node",
        name="planning_node",
        output="screen",
        parameters=[{
            "objects_topic": "/perception/objects",
            "obstacles_topic": "/perception/obstacles",
            "odom_topic": "/odom",
            "trajectory_topic": "/planning/trajectory",
            "control_topic": LaunchConfiguration("control_topic"),
            "state_topic": "/planning/state",
            "max_speed": LaunchConfiguration("max_speed"),
            "control_rate_hz": 20.0,
        }],
        arguments=["--ros-args", "--log-level",
                    LaunchConfiguration("log_level")],
        condition=IfCondition(LaunchConfiguration("enable_planning")),
    )

    # --- Diagnostic log -------------------------------------------------
    startup_log = LogInfo(msg=[
        "Launching AV Jetson stack — engine=",
        LaunchConfiguration("engine_path"),
        " input=", LaunchConfiguration("input_topic"),
        " lidar=", LaunchConfiguration("lidar_topic"),
        " control=", LaunchConfiguration("control_topic"),
    ])

    return LaunchDescription([
        engine_path_arg,
        class_names_arg,
        input_topic_arg,
        lidar_topic_arg,
        control_topic_arg,
        max_speed_arg,
        device_id_arg,
        publish_annotated_arg,
        enable_perception_arg,
        enable_planning_arg,
        log_level_arg,
        startup_log,
        ai_node,
        perception_node,
        planning_node,
    ])
