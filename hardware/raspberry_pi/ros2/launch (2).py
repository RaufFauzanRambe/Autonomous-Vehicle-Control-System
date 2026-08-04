"""
File:        ros2/launch.py
Brief:       ROS 2 launch file for the Autonomous Vehicle Control
             System on Raspberry Pi. Brings up perception, control,
             localization, and navigation nodes with parameters,
             remappings, and namespaces.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from launch import LaunchDescription, LaunchContext
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _declare_arg(name: str, default: str, description: str = "") -> DeclareLaunchArgument:
    return DeclareLaunchArgument(
        name, default_value=default, description=description,
    )


def _config_path(vehicle_id: LaunchConfiguration) -> PythonExpression:
    """Return the YAML config path for the given vehicle id."""
    return PythonExpression([
        "'", PathJoinSubstitution([
            FindPackageShare("av_bringup"), "config",
        ]).perform(None) if False else "",  # placeholder; we resolve at runtime
        "'",
    ])


# ----------------------------------------------------------------------
# Node factories
# ----------------------------------------------------------------------
def _perception_node(vehicle_id: LaunchConfiguration,
                     use_sim_time: LaunchConfiguration) -> Node:
    return Node(
        package="av_perception",
        executable="perception_node",
        name="perception",
        namespace=vehicle_id,
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            "use_sim_time": use_sim_time,
            "camera_topic": "/camera/image_raw",
            "lidar_topic":  "/lidar/scan",
            "detection_confidence": 0.5,
            "max_obstacle_distance_m": 8.0,
            "publish_rate_hz": 10,
        }],
        remappings=[
            ("obstacles", "/perception/obstacles"),
            ("obstacle_map", "/perception/obstacle_map"),
        ],
    )


def _control_node(vehicle_id: LaunchConfiguration,
                  use_sim_time: LaunchConfiguration) -> Node:
    return Node(
        package="av_control",
        executable="control_node",
        name="control",
        namespace=vehicle_id,
        output="screen",
        respawn=True,
        respawn_delay=1.0,
        parameters=[{
            "use_sim_time": use_sim_time,
            "loop_rate_hz": 30,
            "cruise_speed_mps": 0.4,
            "max_speed_mps": 1.5,
            "max_steering_deg": 27.0,
            "obstacle_safe_distance_m": 0.30,
            "pid_linear_kp": 0.8,
            "pid_linear_ki": 0.05,
            "pid_linear_kd": 0.01,
            "pid_angular_kp": 1.2,
            "pid_angular_ki": 0.0,
            "pid_angular_kd": 0.05,
            "estop_on_watchdog_timeout_s": 0.25,
        }],
        remappings=[
            ("cmd_vel", "/cmd_vel"),
            ("estop", "/estop"),
        ],
    )


def _localization_node(vehicle_id: LaunchConfiguration,
                       use_sim_time: LaunchConfiguration) -> Node:
    return Node(
        package="av_localization",
        executable="localization_node",
        name="localization",
        namespace=vehicle_id,
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            "use_sim_time": use_sim_time,
            "gps_topic": "/gps/fix",
            "imu_topic": "/imu/data",
            "odom_topic": "/localization/odom",
            "publish_rate_hz": 30,
            "ekf_alpha": 0.95,
            "frame_id": "odom",
            "child_frame_id": "base_link",
            "initial_lat": 0.0,
            "initial_lon": 0.0,
        }],
    )


def _navigation_node(vehicle_id: LaunchConfiguration,
                     use_sim_time: LaunchConfiguration,
                     enable_nav2: LaunchConfiguration) -> Node:
    return Node(
        package="av_navigation",
        executable="navigation_node",
        name="navigation",
        namespace=vehicle_id,
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        condition=UnlessCondition(enable_nav2),
        parameters=[{
            "use_sim_time": use_sim_time,
            "planner": "astar",
            "grid_resolution_m": 0.10,
            "grid_width_m": 20.0,
            "grid_height_m": 20.0,
            "goal_topic": "/goal_pose",
            "path_topic": "/navigation/path",
            "waypoint_tolerance_m": 0.20,
        }],
    )


def _nav2_bringup(enable_nav2: LaunchConfiguration) -> Node:
    """Optional Nav2 stack (heavier — use a Jetson if available)."""
    return Node(
        package="nav2_bringup",
        executable="navigation_launch",
        name="nav2",
        output="screen",
        condition=IfCondition(enable_nav2),
        parameters=[{
            "use_sim_time": False,
            "controller_server.ros__parameters": {
                "controller_frequency": 20.0,
            },
        }],
    )


# ----------------------------------------------------------------------
# Launch description
# ----------------------------------------------------------------------
def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for the AV bringup."""
    vehicle_id = LaunchConfiguration("vehicle_id")
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_nav2 = LaunchConfiguration("enable_nav2")
    log_level = LaunchConfiguration("log_level")

    # ---- Declare arguments ----
    declare_args: List = [
        _declare_arg("vehicle_id", "av-01", "Unique vehicle identifier"),
        _declare_arg("use_sim_time", "false", "Use simulation clock"),
        _declare_arg("enable_nav2", "false", "Bring up the full Nav2 stack"),
        _declare_arg("log_level", "info", "ROS log level (debug/info/warn/error)"),
    ]

    # ---- Environment ----
    env = [
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "0"),
        SetEnvironmentVariable(
            "ROS_LOG_DIR",
            PathJoinSubstitution([
                os.environ.get("HOME", "/tmp"), "ros2_logs",
            ]),
        ),
    ]

    # ---- Log info ----
    info = LogInfo(msg=["Bringing up autonomous vehicle '", vehicle_id, "'",
                        " (sim_time=", use_sim_time,
                        " nav2=", enable_nav2, ")"]),

    # ---- Nodes ----
    perception = _perception_node(vehicle_id, use_sim_time)
    control = _control_node(vehicle_id, use_sim_time)
    localization = _localization_node(vehicle_id, use_sim_time)
    navigation = _navigation_node(vehicle_id, use_sim_time, enable_nav2)
    nav2 = _nav2_bringup(enable_nav2)

    # ---- Group with namespace ----
    group = GroupAction(
        actions=[
            PushRosNamespace(vehicle_id),
            perception,
            control,
            localization,
            navigation,
            nav2,
        ],
        on_exit=Shutdown(),
    )

    return LaunchDescription(declare_args + env + list(info) + [group])


# ----------------------------------------------------------------------
# When invoked directly (python3 ros2/launch.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from launch import LaunchService
    from launch import LaunchDescription
    from launch.testing import LaunchContextManager

    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    raise SystemExit(ls.run())
