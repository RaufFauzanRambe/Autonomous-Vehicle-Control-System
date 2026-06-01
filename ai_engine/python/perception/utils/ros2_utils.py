"""
ROS2 utilities for the Autonomous Vehicle Control System.

Provides node lifecycle management, message conversion helpers,
QoS profile definitions, and parameter handling utilities for
ROS2 Humble and later distributions.

This module is designed to work whether or not ``rclpy`` is installed.
When ROS2 is unavailable, public APIs degrade gracefully (e.g., message
conversion falls back to dict-based representations).

Usage:
    from utils.ros2_utils import QoSProfiles, msg_to_dict, declare_parameters

    # In a ROS2 node:
    self.qos = QoSProfiles.SENSOR_DATA
    params = declare_parameters(self, [("max_speed", 30.0), ("frame_id", "base_link")])
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

# Attempt to import ROS2 – degrade gracefully if not available
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from rclpy.parameter import Parameter
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    Node = object  # type: ignore[misc, assignment]
    LifecycleNode = object  # type: ignore[misc, assignment]
    QoSProfile = object  # type: ignore[misc, assignment]
    Parameter = object  # type: ignore[misc, assignment]
    TransitionCallbackReturn = object  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# QoS Profiles
# ---------------------------------------------------------------------------

class QoSProfiles:
    """Pre-defined QoS profiles for common AV communication patterns.

    Uses the actual ``rclpy.qos.QoSProfile`` when ROS2 is available;
    otherwise returns a plain dict describing the profile for documentation.
    """

    @staticmethod
    def _make_profile(
        depth: int,
        reliability: str,
        history: str,
        durability: str,
    ) -> Any:
        if not _ROS2_AVAILABLE:
            return {
                "depth": depth,
                "reliability": reliability,
                "history": history,
                "durability": durability,
            }

        rel_map = {
            "reliable": QoSReliabilityPolicy.RELIABLE,
            "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
        }
        hist_map = {
            "keep_last": QoSHistoryPolicy.KEEP_LAST,
            "keep_all": QoSHistoryPolicy.KEEP_ALL,
        }
        dur_map = {
            "volatile": QoSDurabilityPolicy.VOLATILE,
            "transient_local": QoSDurabilityPolicy.TRANSIENT_LOCAL,
        }

        return QoSProfile(
            depth=depth,
            reliability=rel_map[reliability],
            history=hist_map[history],
            durability=dur_map[durability],
        )

    # Standard profiles
    SENSOR_DATA = property(lambda self: QoSProfiles._make_profile(
        5, "best_effort", "keep_last", "volatile",
    ))
    CONTROL_COMMANDS = property(lambda self: QoSProfiles._make_profile(
        10, "reliable", "keep_last", "volatile",
    ))
    STATE_ESTIMATION = property(lambda self: QoSProfiles._make_profile(
        1, "reliable", "keep_last", "transient_local",
    ))
    DIAGNOSTICS = property(lambda self: QoSProfiles._make_profile(
        20, "reliable", "keep_last", "volatile",
    ))
    LIDAR_SCAN = property(lambda self: QoSProfiles._make_profile(
        5, "best_effort", "keep_last", "volatile",
    ))
    CAMERA_IMAGE = property(lambda self: QoSProfiles._make_profile(
        1, "best_effort", "keep_last", "volatile",
    ))
    PLANNING_PATH = property(lambda self: QoSProfiles._make_profile(
        1, "reliable", "keep_last", "transient_local",
    ))

    def get(self, name: str) -> Any:
        """Retrieve a profile by name string."""
        profiles = {
            "sensor_data": self.SENSOR_DATA,
            "control_commands": self.CONTROL_COMMANDS,
            "state_estimation": self.STATE_ESTIMATION,
            "diagnostics": self.DIAGNOSTICS,
            "lidar_scan": self.LIDAR_SCAN,
            "camera_image": self.CAMERA_IMAGE,
            "planning_path": self.PLANNING_PATH,
        }
        key = name.lower()
        if key not in profiles:
            raise KeyError(f"Unknown QoS profile: {name}. Available: {list(profiles.keys())}")
        return profiles[key]


# Singleton instance
qos_profiles = QoSProfiles()


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def msg_to_dict(msg: Any) -> Dict[str, Any]:
    """Convert a ROS2 message to a plain Python dictionary.

    Handles nested messages, arrays, and primitive types.
    Works with any ROS2 message type that follows the ``get_fields_and_field_types``
    convention, or falls back to ``__slots__`` / ``__dict__`` introspection.

    Args:
        msg: A ROS2 message instance.

    Returns:
        Dictionary representation.
    """
    if msg is None:
        return {}

    # If it's already a dict or primitive
    if isinstance(msg, (int, float, str, bool)):
        return {"value": msg}
    if isinstance(msg, dict):
        return msg

    result: Dict[str, Any] = {}

    # Try the ROS2 standard introspection API
    if hasattr(msg, "get_fields_and_field_types"):
        fields = msg.get_fields_and_field_types()
        for field_name in fields:
            value = getattr(msg, field_name, None)
            result[field_name] = _convert_field(value)
    elif hasattr(msg, "__slots__"):
        for slot in msg.__slots__:
            field_name = slot.lstrip("_")
            value = getattr(msg, slot, None)
            if value is not None:
                result[field_name] = _convert_field(value)
    elif hasattr(msg, "__dict__"):
        result = {k: _convert_field(v) for k, v in msg.__dict__.items() if not k.startswith("_")}
    else:
        result = {"raw": str(msg)}

    return result


def _convert_field(value: Any) -> Any:
    """Recursively convert a message field value to a Python-native type."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_convert_field(v) for v in value]
    # Assume nested ROS2 message
    if hasattr(value, "get_fields_and_field_types") or hasattr(value, "__slots__"):
        return msg_to_dict(value)
    return str(value)


def dict_to_msg(msg_type: Any, data: Dict[str, Any]) -> Any:
    """Populate a ROS2 message from a dictionary.

    Args:
        msg_type: The ROS2 message class (e.g., ``std_msgs.msg.String``).
        data: Dictionary of field names → values.

    Returns:
        Populated message instance.
    """
    if not _ROS2_AVAILABLE:
        raise RuntimeError("ROS2 (rclpy) is required for dict_to_msg")

    msg = msg_type()
    for key, value in data.items():
        if hasattr(msg, key):
            setattr(msg, key, value)
    return msg


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def declare_parameters(
    node: Any,
    parameters: Sequence[Tuple[str, Any]],
    descriptors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Declare and retrieve parameters from a ROS2 node.

    Args:
        node: A ``rclpy.node.Node`` instance.
        parameters: Sequence of ``(name, default_value)`` tuples.
        descriptors: Optional parameter descriptor overrides.

    Returns:
        Dict mapping parameter names to their current values.
    """
    if not _ROS2_AVAILABLE:
        # Return defaults when ROS2 is not available
        return {name: default for name, default in parameters}

    result: Dict[str, Any] = {}
    for name, default_value in parameters:
        param_type = _infer_param_type(default_value)
        desc = descriptors.get(name) if descriptors else None

        if desc is not None:
            node.declare_parameter(name, default_value, descriptor=desc)
        else:
            node.declare_parameter(name, default_value)

        result[name] = node.get_parameter(name).value

    return result


def _infer_param_type(value: Any) -> str:
    """Infer the ROS2 parameter type from a Python value."""
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "double"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        if not value:
            return "double_array"
        first = value[0]
        if isinstance(first, bool):
            return "bool_array"
        elif isinstance(first, int):
            return "int_array"
        elif isinstance(first, float):
            return "double_array"
        elif isinstance(first, str):
            return "string_array"
    return "string"


# ---------------------------------------------------------------------------
# Node lifecycle helpers
# ---------------------------------------------------------------------------

class NodeState(enum.Enum):
    """Simplified node lifecycle states."""
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR = "error"


@dataclass
class NodeHealth:
    """Track the health and state of a managed node."""
    name: str
    state: NodeState = NodeState.UNCONFIGURED
    last_heartbeat: float = 0.0
    error_count: int = 0
    last_error: str = ""
    startup_time: float = 0.0

    def heartbeat(self) -> None:
        """Record a heartbeat to signal the node is alive."""
        self.last_heartbeat = time.monotonic()

    def mark_active(self) -> None:
        self.state = NodeState.ACTIVE
        self.heartbeat()

    def mark_error(self, message: str = "") -> None:
        self.state = NodeState.ERROR
        self.error_count += 1
        self.last_error = message
        self.heartbeat()

    def is_stale(self, timeout: float = 5.0) -> bool:
        """Return True if no heartbeat received within *timeout* seconds."""
        return (time.monotonic() - self.last_heartbeat) > timeout


class NodeManager:
    """Simple manager that tracks multiple ROS2 node health states.

    Useful for a system-level monitor that needs to know which subsystem
    nodes are alive, active, or in error.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeHealth] = {}

    def register(self, name: str) -> NodeHealth:
        health = NodeHealth(name=name, startup_time=time.monotonic())
        self._nodes[name] = health
        return health

    def get(self, name: str) -> Optional[NodeHealth]:
        return self._nodes.get(name)

    def all_healthy(self, heartbeat_timeout: float = 5.0) -> bool:
        """Return True if every registered node is active and not stale."""
        return all(
            h.state == NodeState.ACTIVE and not h.is_stale(heartbeat_timeout)
            for h in self._nodes.values()
        )

    def unhealthy_nodes(self, heartbeat_timeout: float = 5.0) -> List[NodeHealth]:
        """Return a list of nodes that are not healthy."""
        return [
            h for h in self._nodes.values()
            if h.state != NodeState.ACTIVE or h.is_stale(heartbeat_timeout)
        ]

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Return a summary dict for all nodes."""
        return {
            name: {
                "state": h.state.value,
                "error_count": h.error_count,
                "last_error": h.last_error,
                "stale": h.is_stale(),
                "uptime": time.monotonic() - h.startup_time,
            }
            for name, h in self._nodes.items()
        }


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def ros2_time_to_float(stamp: Any) -> float:
    """Convert a ROS2 ``builtin_interfaces.msg.Time`` to a float (seconds)."""
    if stamp is None:
        return 0.0
    if isinstance(stamp, (int, float)):
        return float(stamp)
    sec = getattr(stamp, "sec", 0)
    nanosec = getattr(stamp, "nanosec", 0)
    return float(sec) + float(nanosec) * 1e-9


def float_to_ros2_duration(seconds: float) -> Dict[str, int]:
    """Convert float seconds to ROS2 duration dict {sec, nanosec}."""
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1e9)
    return {"sec": sec, "nanosec": nanosec}


def float_to_ros2_time(seconds: float) -> Dict[str, int]:
    """Convert float seconds to ROS2 time dict {sec, nanosec}."""
    return float_to_ros2_duration(seconds)


# ---------------------------------------------------------------------------
# Topic name helpers
# ---------------------------------------------------------------------------

def resolve_topic(namespace: str, topic: str) -> str:
    """Resolve a topic name within a namespace.

    Ensures proper leading slashes and no double slashes.
    """
    ns = namespace.strip("/")
    t = topic.strip("/")
    if ns:
        return f"/{ns}/{t}"
    return f"/{t}"


def topic_parts(topic: str) -> List[str]:
    """Split a fully-qualified topic name into its parts."""
    return [p for p in topic.split("/") if p]
