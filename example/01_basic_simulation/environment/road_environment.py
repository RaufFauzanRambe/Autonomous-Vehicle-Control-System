"""Road environment: straight road segments with widths and speed limits.

The environment answers three geometric queries used by the simulation
loop and the plotting code:

* ``is_on_road(x, y)``: whether a world point lies within any segment.
* ``distance_to_centerline(x, y)``: distance to the nearest segment axis.
* ``get_speed_limit(x, y)``: the speed limit of the nearest segment.

Distances are Euclidean point-to-segment distances in a flat 2-D world
frame; all units are meters and seconds.

Extension point: load a polyline graph instead of straight segments and
replace the brute-force minimum over segments with a KD-tree / R-tree
spatial index once maps exceed a few hundred segments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class RoadEnvironment:
    """Collection of straight road segments with geometric queries.

    The map file is a JSON document of the shape::

        {
          "map_name": "straight_test_track",
          "units": "m",
          "road_segments": [
            {"id": "seg_00", "start": [x0, y0], "end": [x1, y1],
             "width_m": 7.0, "speed_limit_mps": 20.0}
          ]
        }

    Args:
        map_path: Path to the road map JSON file.

    Attributes:
        map_name: Name taken from the map file.
        segments: Raw segment dictionaries as stored in the file.

    Raises:
        FileNotFoundError: If ``map_path`` does not exist.
        ValueError: If the file contains no road segments or uses units
            other than meters.
    """

    def __init__(self, map_path: str | Path) -> None:
        """Load and validate the road map file.

        Args:
            map_path: Path to the road map JSON file (units must be "m").
        """
        with Path(map_path).open("r", encoding="utf-8") as handle:
            map_data: Dict[str, Any] = json.load(handle)

        self.map_name: str = str(map_data.get("map_name", "unnamed_map"))
        units: str = str(map_data.get("units", "m"))
        if units != "m":
            raise ValueError(f"Expected road map units 'm', got '{units}'")

        segments: List[Dict[str, Any]] = list(map_data.get("road_segments", []))
        if not segments:
            raise ValueError(f"Road map '{map_path}' contains no road segments")
        self.segments: List[Dict[str, Any]] = segments

    @staticmethod
    def _point_segment_distance(
        px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> float:
        """Euclidean distance from a point to a straight segment [m].

        Uses numpy 2-D vector math: the projection parameter of the point
        onto the segment line is clamped to [0, 1], so the closest point is
        guaranteed to lie on the segment (not on the infinite line).

        Args:
            px: Point x coordinate [m].
            py: Point y coordinate [m].
            ax: Segment start x coordinate [m].
            ay: Segment start y coordinate [m].
            bx: Segment end x coordinate [m].
            by: Segment end y coordinate [m].

        Returns:
            Shortest distance from the point to the segment [m]; 0.0 for a
            zero-length segment coincident with the point.
        """
        point = np.array([px, py], dtype=float)
        start = np.array([ax, ay], dtype=float)
        end = np.array([bx, by], dtype=float)
        direction = end - start
        length_sq = float(direction @ direction)
        if length_sq == 0.0:
            # Degenerate segment: treat as a point.
            return float(np.linalg.norm(point - start))
        t = float(((point - start) @ direction) / length_sq)
        t_clamped = min(1.0, max(0.0, t))
        closest = start + t_clamped * direction
        return float(np.linalg.norm(point - closest))

    def _nearest_segment_index(self, x: float, y: float) -> int:
        """Index of the segment whose centerline is closest to the point.

        Args:
            x: World-frame x coordinate [m].
            y: World-frame y coordinate [m].

        Returns:
            Index into ``self.segments`` of the nearest segment.
        """
        best_index = 0
        best_distance = float("inf")
        for index, segment in enumerate(self.segments):
            distance = self._point_segment_distance(
                x, y,
                float(segment["start"][0]), float(segment["start"][1]),
                float(segment["end"][0]), float(segment["end"][1]),
            )
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _segment_endpoints(self, index: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Start and end coordinates of a segment as float tuples [m].

        Args:
            index: Segment index into ``self.segments``.

        Returns:
            ((start_x, start_y), (end_x, end_y)) in meters.
        """
        segment = self.segments[index]
        return (
            (float(segment["start"][0]), float(segment["start"][1])),
            (float(segment["end"][0]), float(segment["end"][1])),
        )

    def distance_to_centerline(self, x: float, y: float) -> float:
        """Minimum point-to-segment distance over all segments [m].

        Args:
            x: World-frame x coordinate [m].
            y: World-frame y coordinate [m].

        Returns:
            Distance to the nearest centerline [m] (>= 0).
        """
        (ax, ay), (bx, by) = self._segment_endpoints(self._nearest_segment_index(x, y))
        return self._point_segment_distance(x, y, ax, ay, bx, by)

    def is_on_road(self, x: float, y: float) -> bool:
        """Check whether a world point lies on any road segment.

        A point is on the road when its distance to the nearest segment
        centerline is at most half that segment's width.

        Args:
            x: World-frame x coordinate [m].
            y: World-frame y coordinate [m].

        Returns:
            True if the point is on the road, False otherwise.
        """
        index = self._nearest_segment_index(x, y)
        (ax, ay), (bx, by) = self._segment_endpoints(index)
        distance = self._point_segment_distance(x, y, ax, ay, bx, by)
        return bool(distance <= float(self.segments[index]["width_m"]) / 2.0)

    def get_speed_limit(self, x: float, y: float) -> float:
        """Speed limit of the segment nearest to the point [m/s].

        Args:
            x: World-frame x coordinate [m].
            y: World-frame y coordinate [m].

        Returns:
            Speed limit of the nearest segment [m/s]. Falls back to the
            first segment's limit if no nearest segment can be determined
            (defensive branch; the constructor guarantees at least one
            segment, so this normally returns the nearest segment's limit).
        """
        index = self._nearest_segment_index(x, y)
        if index < 0:
            return float(self.segments[0]["speed_limit_mps"])
        return float(self.segments[index]["speed_limit_mps"])
