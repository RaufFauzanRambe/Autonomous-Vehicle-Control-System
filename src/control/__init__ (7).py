"""
Communication Module — Autonomous Vehicle Control System

This module provides V2X (Vehicle-to-Everything) communication capabilities
for the autonomous vehicle platform. It includes:

- **V2XNode**: Central V2X communication hub with publish/subscribe messaging,
  peer management, and priority queue handling for V2V and V2I links.

- **MessageParser**: J2735 DSRC message parsing and serialization for BSM
  (Basic Safety Message), MAP (Map Data), and SPAT (Signal Phase and Timing).

- **DSRCInterface**: Low-level 5.9 GHz DSRC radio interface with channel
  management, transmission power control, and WAVE protocol support.

- **CAMPProcessor**: Cooperative Awareness Message processing with ego vehicle
  broadcasting, remote vehicle tracking, and reception quality monitoring.

Typical usage::

    from communication import V2XNode, MessageParser, DSRCInterface, CAMPProcessor

    # Initialize communication stack
    dsrc = DSRCInterface(device_index=0)
    dsrc.initialize()

    parser = MessageParser()
    node = V2XNode(node_id="AV-001")
    node.start()

    camp = CAMPProcessor(station_id=1001)
    camp.start()
"""

from communication.v2x_node import V2XNode
from communication.message_parser import MessageParser
from communication.dsrc_interface import DSRCInterface
from communication.camp_processor import CAMPProcessor

__all__ = [
    "V2XNode",
    "MessageParser",
    "DSRCInterface",
    "CAMPProcessor",
]

__version__ = "1.0.0"
