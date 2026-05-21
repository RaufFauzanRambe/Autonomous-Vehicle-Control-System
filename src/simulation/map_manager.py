"""
Map Manager Module for Autonomous Vehicle Localization

Handles loading, storing, and querying HD maps for localization
and planning. Supports lane-level maps, semantic maps, and
occupancy grid maps.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import struct


class MapType(Enum):
    """Supported map types."""
    LANE = "lane"
    SEMANTIC = "semantic"
    OCCUPANCY = "occupancy"
    POINT_CLOUD = "point_cloud"
    TOPOLOGY = "topology"


@dataclass
class LaneSegment:
    """Represents a single lane segment in the HD map."""
    lane_id: str
    left_boundary: np.ndarray    # (N, 2) polyline [x, y]
    right_boundary: np.ndarray   # (N, 2) polyline [x, y]
    center_line: np.ndarray      # (N, 2) polyline [x, y]
    speed_limit: float           # m/s
    lane_type: str = "driving"   # driving, shoulder, parking, etc.
    direction: str = "forward"   # forward, backward, bidirectional
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)
    left_neighbor_id: Optional[str] = None
    right_neighbor_id: Optional[str] = None

    @property
    def length(self) -> float:
        """Approximate lane segment length."""
        diffs = np.diff(self.center_line, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

    @property
    def width(self) -> float:
        """Average lane width."""
        if len(self.left_boundary) > 0 and len(self.right_boundary) > 0:
            distances = np.linalg.norm(
                self.left_boundary[:len(self.right_boundary)] - self.right_boundary[:len(self.left_boundary)],
                axis=1
            )
            return float(np.mean(distances))
        return 3.5  # Default lane width


@dataclass
class RoadSegment:
    """Represents a road consisting of multiple lanes."""
    road_id: str
    lanes: List[LaneSegment]
    road_type: str = "urban"     # urban, highway, rural
    junction_id: Optional[str] = None


@dataclass
class TrafficSignal:
    """Traffic signal/light in the map."""
    signal_id: str
    position: np.ndarray         # [x, y, z]
    orientation: float           # Heading angle in radians
    controlled_lanes: List[str]  # Lane IDs controlled by this signal
    signal_type: str = "traffic_light"  # traffic_light, stop_sign, yield_sign


class MapManager:
    """
    HD Map manager for autonomous vehicle localization and planning.

    Provides efficient spatial queries for lane information,
    road topology, and traffic signal data. Supports loading
    from multiple map formats including OpenDRIVE and Apollo HD Map.
    """

    def __init__(self, map_path: Optional[str] = None):
        """
        Initialize the map manager.

        Args:
            map_path: Path to the map file to load.
        """
        self._lanes: Dict[str, LaneSegment] = {}
        self._roads: Dict[str, RoadSegment] = {}
        self._signals: Dict[str, TrafficSignal] = {}
        self._bounds: Optional[Tuple[float, float, float, float]] = None

        if map_path:
            self.load_map(map_path)

    def load_map(self, map_path: str) -> bool:
        """
        Load an HD map from file.

        Supports JSON-based map format. OpenDRIVE and Apollo
        format support would require additional parsers.

        Args:
            map_path: Path to the map file.

        Returns:
            True if the map was loaded successfully.
        """
        try:
            with open(map_path, 'r') as f:
                map_data = json.load(f)

            self._parse_map_data(map_data)
            self._compute_bounds()
            return True
        except Exception as e:
            print(f"Failed to load map from {map_path}: {e}")
            return False

    def add_lane(self, lane: LaneSegment) -> None:
        """Add a lane segment to the map."""
        self._lanes[lane.lane_id] = lane

    def add_road(self, road: RoadSegment) -> None:
        """Add a road segment to the map."""
        self._roads[road.road_id] = road
        for lane in road.lanes:
            self.add_lane(lane)

    def add_signal(self, signal: TrafficSignal) -> None:
        """Add a traffic signal to the map."""
        self._signals[signal.signal_id] = signal

    def get_lane(self, lane_id: str) -> Optional[LaneSegment]:
        """Get a lane segment by its ID."""
        return self._lanes.get(lane_id)

    def get_nearest_lane(self, position: np.ndarray) -> Optional[LaneSegment]:
        """
        Find the nearest lane to a given position.

        Args:
            position: 2D position [x, y] or 3D position [x, y, z].

        Returns:
            The nearest LaneSegment, or None if no lanes loaded.
        """
        if not self._lanes:
            return None

        pos_2d = position[:2]
        best_lane = None
        best_dist = float('inf')

        for lane in self._lanes.values():
            dist = self._distance_to_lane(pos_2d, lane)
            if dist < best_dist:
                best_dist = dist
                best_lane = lane

        return best_lane

    def get_lanes_in_radius(self, position: np.ndarray, radius: float) -> List[LaneSegment]:
        """
        Get all lane segments within a radius of a position.

        Args:
            position: 2D/3D position.
            radius: Search radius in meters.

        Returns:
            List of lane segments within the radius.
        """
        pos_2d = position[:2]
        result = []

        for lane in self._lanes.values():
            if self._distance_to_lane(pos_2d, lane) <= radius:
                result.append(lane)

        return result

    def get_signals_in_radius(self, position: np.ndarray, radius: float) -> List[TrafficSignal]:
        """
        Get all traffic signals within a radius of a position.

        Args:
            position: 2D/3D position.
            radius: Search radius in meters.

        Returns:
            List of traffic signals within the radius.
        """
        pos = position[:2]
        result = []

        for signal in self._signals.values():
            dist = np.linalg.norm(signal.position[:2] - pos)
            if dist <= radius:
                result.append(signal)

        return result

    def get_route_lanes(self, start_lane_id: str, end_lane_id: str) -> List[LaneSegment]:
        """
        Find a sequence of connected lanes from start to end.

        Uses BFS on the lane topology graph.

        Args:
            start_lane_id: Starting lane ID.
            end_lane_id: Destination lane ID.

        Returns:
            Ordered list of lane segments forming the route.
        """
        if start_lane_id not in self._lanes or end_lane_id not in self._lanes:
            return []

        # BFS
        visited = {start_lane_id}
        queue = [(start_lane_id, [start_lane_id])]

        while queue:
            current_id, path = queue.pop(0)
            if current_id == end_lane_id:
                return [self._lanes[lid] for lid in path]

            lane = self._lanes[current_id]
            neighbors = lane.successor_ids + lane.predecessor_ids
            if lane.left_neighbor_id:
                neighbors.append(lane.left_neighbor_id)
            if lane.right_neighbor_id:
                neighbors.append(lane.right_neighbor_id)

            for neighbor_id in neighbors:
                if neighbor_id not in visited and neighbor_id in self._lanes:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return []  # No route found

    @staticmethod
    def _distance_to_lane(position: np.ndarray, lane: LaneSegment) -> float:
        """Compute minimum distance from a point to a lane center line."""
        if len(lane.center_line) == 0:
            return float('inf')

        diffs = lane.center_line - position
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return float(np.min(distances))

    def _parse_map_data(self, data: Dict) -> None:
        """Parse map data from dictionary."""
        for lane_data in data.get('lanes', []):
            lane = LaneSegment(
                lane_id=lane_data['id'],
                left_boundary=np.array(lane_data.get('left_boundary', [])),
                right_boundary=np.array(lane_data.get('right_boundary', [])),
                center_line=np.array(lane_data.get('center_line', [])),
                speed_limit=lane_data.get('speed_limit', 13.9),
                lane_type=lane_data.get('type', 'driving'),
                direction=lane_data.get('direction', 'forward'),
                predecessor_ids=lane_data.get('predecessors', []),
                successor_ids=lane_data.get('successors', []),
                left_neighbor_id=lane_data.get('left_neighbor'),
                right_neighbor_id=lane_data.get('right_neighbor'),
            )
            self.add_lane(lane)

        for signal_data in data.get('signals', []):
            signal = TrafficSignal(
                signal_id=signal_data['id'],
                position=np.array(signal_data['position']),
                orientation=signal_data.get('orientation', 0.0),
                controlled_lanes=signal_data.get('controlled_lanes', []),
                signal_type=signal_data.get('type', 'traffic_light'),
            )
            self.add_signal(signal)

    def _compute_bounds(self) -> None:
        """Compute the bounding box of the map."""
        if not self._lanes:
            return

        all_points = []
        for lane in self._lanes.values():
            if len(lane.center_line) > 0:
                all_points.append(lane.center_line)

        if all_points:
            points = np.vstack(all_points)
            self._bounds = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )

    @property
    def num_lanes(self) -> int:
        return len(self._lanes)

    @property
    def num_signals(self) -> int:
        return len(self._signals)

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        return self._bounds
