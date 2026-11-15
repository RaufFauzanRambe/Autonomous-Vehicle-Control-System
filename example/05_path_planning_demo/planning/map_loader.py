"""Road-graph map loading for the path-planning demo (module 05).

A map is a JSON document describing a small road network as a graph:

* ``nodes`` carry planar coordinates ``x`` and ``y`` in metres.
* ``links`` connect two nodes; each link stores a speed limit [m/s] and a
  lane count [-]. The geometric link length [m] is recomputed from the node
  coordinates whenever it is not stored explicitly, so the JSON schema stays
  minimal and can never disagree with the node geometry.

Roads are treated as undirected by default: a single link entry is expanded
into both travel directions by :meth:`MapLoader.build_adjacency`. Passing
``directed=True`` keeps only the stored direction, which models one-way
streets.

Extension point: replace the flat JSON schema with lane-level edges
(per-lane connectivity and turn manoeuvres at intersections) or add SUMO /
OpenDRIVE importers for research-grade network topologies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class MapNode:
    """A planar node of the road graph.

    Attributes:
        id: Unique node identifier [-].
        x: World-frame east-west coordinate [m].
        y: World-frame north-south coordinate [m].
    """

    id: str
    x: float
    y: float


@dataclass
class MapLink:
    """A road segment connecting two nodes.

    Attributes:
        id: Unique link identifier [-].
        from_id: Identifier of the start node [-].
        to_id: Identifier of the end node [-].
        length_m: Geometric length of the segment [m].
        speed_limit_mps: Legal speed limit on the segment [m/s].
        lanes: Number of lanes [-].
    """

    id: str
    from_id: str
    to_id: str
    length_m: float
    speed_limit_mps: float
    lanes: int


class MapLoader:
    """Factory helpers that turn JSON map documents into typed graph data.

    All methods are static so the loader can be used without instance
    state; maps are small enough (tens of nodes) that no caching layer is
    required.
    """

    @staticmethod
    def load(map_path: str | Path) -> Tuple[List[MapNode], List[MapLink]]:
        """Load a road graph from a JSON map document.

        The expected document layout is::

            {"map_name": str, "units": str,
             "nodes": [{"id": str, "x": float, "y": float}, ...],
             "links": [{"id": str, "from_id": str, "to_id": str,
                        "speed_limit_mps": float, "lanes": int}, ...]}

        Args:
            map_path: Path to the map ``.json`` file. Relative paths are
                resolved by the caller against the module directory.

        Returns:
            Tuple ``(nodes, links)`` where ``nodes`` is a list of
            :class:`MapNode` (coordinates in [m]) and ``links`` is a list
            of :class:`MapLink` (lengths in [m], speed limits in [m/s],
            lane counts [-]).

        Raises:
            FileNotFoundError: If ``map_path`` does not exist.
            ValueError: If the document is structurally invalid (unknown
                link endpoints, missing mandatory fields, non-numeric
                coordinates).

        Extension point: validate the loaded document against a JSON
        schema (see ``shared/schemas/``) before building the dataclasses.
        """
        file_path = Path(map_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Map file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)

        nodes: List[MapNode] = []
        for entry in document.get("nodes", []):
            try:
                nodes.append(
                    MapNode(id=str(entry["id"]), x=float(entry["x"]), y=float(entry["y"]))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid node entry {entry!r} in {file_path}: {exc}") from exc

        lookup: Dict[str, MapNode] = {node.id: node for node in nodes}
        links: List[MapLink] = []
        for entry in document.get("links", []):
            try:
                from_id = str(entry["from_id"])
                to_id = str(entry["to_id"])
                if from_id not in lookup or to_id not in lookup:
                    raise ValueError(
                        f"link '{entry.get('id', '?')}' references unknown node "
                        f"({from_id!r} -> {to_id!r})"
                    )
                if "speed_limit_mps" not in entry:
                    raise ValueError(f"link '{entry.get('id', '?')}' is missing 'speed_limit_mps'")
                start_node = lookup[from_id]
                end_node = lookup[to_id]
                raw_length = entry.get("length_m")
                length_m = (
                    float(raw_length)
                    if raw_length is not None
                    else math.hypot(end_node.x - start_node.x, end_node.y - start_node.y)
                )
                links.append(
                    MapLink(
                        id=str(entry["id"]),
                        from_id=from_id,
                        to_id=to_id,
                        length_m=length_m,
                        speed_limit_mps=float(entry["speed_limit_mps"]),
                        lanes=int(entry.get("lanes", 1)),
                    )
                )
            except ValueError:
                raise
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid link entry {entry!r} in {file_path}: {exc}") from exc

        return nodes, links

    @staticmethod
    def build_adjacency(
        nodes: List[MapNode],
        links: List[MapLink],
        directed: bool = False,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Build a node-id to neighbour-list adjacency map.

        Args:
            nodes: Graph nodes; every node id becomes a key, isolated
                nodes map to an empty list.
            links: Graph links providing the edge costs.
            directed: If ``False`` (default) every link is expanded into
                both travel directions; if ``True`` only the stored
                ``from_id -> to_id`` direction is added (one-way streets).

        Returns:
            Dictionary mapping every node id [-] to a list of tuples
            ``(neighbor_id [-], length_m [m])``. The list is owned by the
            caller after return and may be modified freely.

        Raises:
            ValueError: If a link references a node id that is not part of
                ``nodes``.
        """
        adjacency: Dict[str, List[Tuple[str, float]]] = {node.id: [] for node in nodes}
        for link in links:
            if link.from_id not in adjacency or link.to_id not in adjacency:
                raise ValueError(
                    f"Link '{link.id}' references nodes outside the node list "
                    f"({link.from_id!r} -> {link.to_id!r})"
                )
            adjacency[link.from_id].append((link.to_id, link.length_m))
            if not directed:
                adjacency[link.to_id].append((link.from_id, link.length_m))
        return adjacency

    @staticmethod
    def map_name(map_path: str | Path) -> str:
        """Return the human-readable map name stored in the JSON document.

        Args:
            map_path: Path to the map ``.json`` file.

        Returns:
            The ``map_name`` field [-] if present, otherwise the file stem
            (e.g. ``"simple_road_map"``).

        Raises:
            FileNotFoundError: If ``map_path`` does not exist.
        """
        file_path = Path(map_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Map file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        return str(document.get("map_name", file_path.stem))
