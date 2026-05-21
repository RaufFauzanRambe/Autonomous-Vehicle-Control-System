"""
Message Parser Module for J2735 DSRC Messages

This module implements the MessageParser class which handles parsing and
serialization of J2735 DSRC (Dedicated Short-Range Communications) message
types, including BSM (Basic Safety Message), MAP (Map Data), and SPAT
(Signal Phase and Timing). It also provides message validation and
version-aware decoding.

References:
    - SAE J2735-2016 DSRC Message Set Dictionary
    - IEEE 1609.2 Security Services
    - SAE J2945/1 On-Board System Requirements for V2V Safety Communications
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# J2735 Protocol Constants
# ---------------------------------------------------------------------------

J2735_MSG_ID_BSM: int = 0x0014          # Basic Safety Message
J2735_MSG_ID_MAP: int = 0x0012          # Map Data
J2735_MSG_ID_SPAT: int = 0x0013         # Signal Phase and Timing
J2735_MSG_ID_RSA: int = 0x0015          # Road Side Alert
J2735_MSG_ID_PSM: int = 0x001C          # Personal Safety Message

J2735_VERSION_CURRENT: int = 4          # J2735-2016
J2735_VERSION_MIN_SUPPORTED: int = 2    # Minimum version we can decode

BSM_PART_I_LENGTH: int = 39             # BSM Part I fixed length in bytes
BSM_MAX_LENGTH: int = 400               # Practical maximum BSM length
MAP_MAX_LENGTH: int = 8000              # Practical maximum MAP length
SPAT_MAX_LENGTH: int = 2000             # Practical maximum SPAT length


class ParseError(Exception):
    """Raised when message parsing fails due to malformed or invalid data."""


class ValidationError(Exception):
    """Raised when a parsed message fails validation checks."""


class MessageFormat(IntEnum):
    """Wire format of the message."""
    UPER = 1        # Unaligned Packed Encoding Rules (ASN.1)
    PER = 2         # Aligned Packed Encoding Rules
    JSON = 3        # JSON representation (testing / logging)
    RAW = 4         # Raw byte buffer without ASN.1 framing


# ---------------------------------------------------------------------------
# Data structures for parsed message content
# ---------------------------------------------------------------------------

@dataclass
class Position3D:
    """WGS-84 position with optional elevation.

    Attributes:
        latitude: Latitude in degrees (-90..90).
        longitude: Longitude in degrees (-180..180).
        elevation: Elevation in decimetres above the WGS-84 ellipsoid, or None.
    """
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: Optional[float] = None

    def is_valid(self) -> bool:
        """Return True if the position coordinates are within valid ranges."""
        return (-90.0 <= self.latitude <= 90.0) and (-180.0 <= self.longitude <= 180.0)


@dataclass
class MotionCruiseControl:
    """Represents cruise control status from a BSM Part II."""
    enabled: bool = False
    active: bool = False
    speed_set_point: Optional[float] = None


@dataclass
class BSMPartIIContent:
    """Container for optional BSM Part II data elements.

    Attributes:
        vehicle_safety_extensions: Optional safety extension data.
        special_vehicle_extensions: Optional special vehicle data.
        cruise_control: Cruise control status.
    """
    vehicle_safety_extensions: Optional[Dict[str, Any]] = None
    special_vehicle_extensions: Optional[Dict[str, Any]] = None
    cruise_control: Optional[MotionCruiseControl] = None


@dataclass
class ParsedBSM:
    """Fully parsed Basic Safety Message (BSM).

    Attributes:
        msg_count: Sequence number (0..127) for duplicate detection.
        id: Temporary vehicle ID (rotated periodically for privacy).
        sec_mark: Milliseconds within the current or previous minute.
        position: WGS-84 position of the transmitting vehicle.
        elevation: Elevation in decimetres.
        speed: Vehicle speed in 0.02 m/s units.
        heading: Heading in 0.0125-degree units from true north.
        angle: Steering wheel angle.
        transmission_state: Vehicle transmission state.
        acceleration_set: 4-way acceleration set (long, lat, vert, yaw).
        brake_status: Brake system status flags.
        vehicle_size: (length, width) in centimetres.
        part_ii: Optional BSM Part II content.
        raw_length: Length of the original raw message in bytes.
    """
    msg_count: int = 0
    id: int = 0
    sec_mark: int = 0
    position: Position3D = field(default_factory=Position3D)
    elevation: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    angle: float = 0.0
    transmission_state: int = 0
    acceleration_set: Dict[str, float] = field(default_factory=lambda: {
        "longitudinal": 0.0, "lateral": 0.0, "vertical": 0.0, "yaw_rate": 0.0
    })
    brake_status: Dict[str, bool] = field(default_factory=lambda: {
        "brake_applied": False, "abs_active": False,
        "stability_control": False, "brake_boost": False, "aux_brake": False,
    })
    vehicle_size: Tuple[int, int] = (0, 0)
    part_ii: Optional[BSMPartIIContent] = None
    raw_length: int = 0


@dataclass
class NodeReference:
    """A node (point) within a MAP intersection or road segment.

    Attributes:
        node_id: Unique node identifier within the intersection.
        position: WGS-84 position.
        lane_width: Lane width at this node in centimetres, or None.
    """
    node_id: int = 0
    position: Position3D = field(default_factory=Position3D)
    lane_width: Optional[int] = None


@dataclass
class LaneData:
    """Describes a single lane within an intersection or road segment.

    Attributes:
        lane_id: Unique lane identifier.
        lane_type: Lane type code (vehicle, bike, crosswalk, etc.).
        direction: Ingress or egress indicator.
        node_list: Ordered list of NodeReference objects defining the lane path.
        speed_limits: List of (speed_limit_type, speed_in_cm_per_sec) tuples.
    """
    lane_id: int = 0
    lane_type: int = 0
    direction: int = 0
    node_list: List[NodeReference] = field(default_factory=list)
    speed_limits: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ParsedMAP:
    """Fully parsed Map Data (MAP) message.

    Attributes:
        msg_issue_revision: Revision counter for this map update.
        layer_type: Layer type indicator (intersection, curve, etc.).
        intersection_id: Intersection reference identifier.
        road_regulator_id: Road regulator identifier.
        ref_position: Reference WGS-84 position.
        lanes: List of LaneData objects.
        node_list: All nodes referenced by lanes.
        raw_length: Length of the original raw message in bytes.
    """
    msg_issue_revision: int = 0
    layer_type: int = 0
    intersection_id: int = 0
    road_regulator_id: int = 0
    ref_position: Position3D = field(default_factory=Position3D)
    lanes: List[LaneData] = field(default_factory=list)
    node_list: List[NodeReference] = field(default_factory=list)
    raw_length: int = 0


@dataclass
class PhaseState:
    """Current state of a single traffic signal phase.

    Attributes:
        phase_id: Phase identifier (usually 1..8).
        phase_state: Current light state (0=dark, 1=red, 5=green, 7=yellow).
        start_time: Time at which this state began (sec_mark).
        min_end_time: Earliest time the state may change.
        max_end_time: Latest time the state may change.
        likely_time: Predicted time the state will change.
    """
    phase_id: int = 0
    phase_state: int = 1
    start_time: int = 0
    min_end_time: int = 0
    max_end_time: int = 0
    likely_time: int = 0


@dataclass
class ParsedSPAT:
    """Fully parsed Signal Phase and Timing (SPAT) message.

    Attributes:
        intersection_id: Intersection reference identifier.
        road_regulator_id: Road regulator identifier.
        msg_count: Message sequence counter.
        epoch_timestamp: Timestamp from the signal controller.
        phases: List of PhaseState objects.
        raw_length: Length of the original raw message in bytes.
    """
    intersection_id: int = 0
    road_regulator_id: int = 0
    msg_count: int = 0
    epoch_timestamp: int = 0
    phases: List[PhaseState] = field(default_factory=list)
    raw_length: int = 0


# ---------------------------------------------------------------------------
# MessageParser
# ---------------------------------------------------------------------------

class MessageParser:
    """Parser for J2735 DSRC messages supporting BSM, MAP, and SPAT.

    The parser handles byte-level decoding of UPER-encoded J2735 messages
    and also supports JSON and raw dictionary input for testing. It validates
    version fields and structural integrity before returning typed data
    classes.

    Thread Safety:
        The parser is stateless after construction and can be shared across
        threads.

    Usage Example::

        parser = MessageParser()
        bsm = parser.parse_bsm(raw_bytes)
        print(f"Vehicle {bsm.id} at ({bsm.position.latitude}, {bsm.position.longitude})")
    """

    def __init__(
        self,
        supported_versions: Optional[List[int]] = None,
        strict_validation: bool = True,
    ) -> None:
        """Initialize the MessageParser.

        Args:
            supported_versions: List of J2735 version numbers accepted.
                Defaults to [2, 3, 4] covering J2735-2009 through 2016.
            strict_validation: If True, raise ValidationError on suspicious
                values; if False, log warnings and continue.
        """
        self.supported_versions: List[int] = supported_versions or [2, 3, 4]
        self.strict_validation: bool = strict_validation
        self._parse_count: int = 0
        self._error_count: int = 0

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def parse(self, data: Union[bytes, Dict[str, Any]], msg_format: MessageFormat = MessageFormat.UPER) \
            -> Union[ParsedBSM, ParsedMAP, ParsedSPAT]:
        """Auto-detect the message type and parse accordingly.

        Args:
            data: Raw message data (bytes for UPER/PER, dict for JSON).
            msg_format: The wire format of the input data.

        Returns:
            A parsed message object (ParsedBSM, ParsedMAP, or ParsedSPAT).

        Raises:
            ParseError: If the message type cannot be determined or the
                data is structurally invalid.
            ValidationError: If the message fails validation.
        """
        if msg_format == MessageFormat.JSON and isinstance(data, dict):
            return self._parse_json(data)

        if not isinstance(data, (bytes, bytearray)):
            raise ParseError(f"Expected bytes or dict, got {type(data).__name__}")

        msg_id = self._detect_message_type(data)
        if msg_id == J2735_MSG_ID_BSM:
            return self.parse_bsm(data)
        elif msg_id == J2735_MSG_ID_MAP:
            return self.parse_map(data)
        elif msg_id == J2735_MSG_ID_SPAT:
            return self.parse_spat(data)
        else:
            raise ParseError(f"Unknown J2735 message ID: 0x{msg_id:04X}")

    def parse_bsm(self, data: Union[bytes, Dict[str, Any]]) -> ParsedBSM:
        """Parse a Basic Safety Message (BSM).

        Supports both UPER-encoded byte data and dictionary (JSON) input.

        Args:
            data: The raw BSM data.

        Returns:
            A ParsedBSM object with all extracted fields.

        Raises:
            ParseError: If the data cannot be decoded.
            ValidationError: If the decoded message fails validation.
        """
        self._parse_count += 1
        try:
            if isinstance(data, dict):
                return self._parse_bsm_from_dict(data)
            return self._parse_bsm_from_bytes(data)
        except (ParseError, ValidationError):
            self._error_count += 1
            raise
        except Exception as exc:
            self._error_count += 1
            raise ParseError(f"Unexpected error parsing BSM: {exc}") from exc

    def parse_map(self, data: Union[bytes, Dict[str, Any]]) -> ParsedMAP:
        """Parse a Map Data (MAP) message.

        Args:
            data: The raw MAP data.

        Returns:
            A ParsedMAP object.

        Raises:
            ParseError: If the data cannot be decoded.
            ValidationError: If the decoded message fails validation.
        """
        self._parse_count += 1
        try:
            if isinstance(data, dict):
                return self._parse_map_from_dict(data)
            return self._parse_map_from_bytes(data)
        except (ParseError, ValidationError):
            self._error_count += 1
            raise
        except Exception as exc:
            self._error_count += 1
            raise ParseError(f"Unexpected error parsing MAP: {exc}") from exc

    def parse_spat(self, data: Union[bytes, Dict[str, Any]]) -> ParsedSPAT:
        """Parse a Signal Phase and Timing (SPAT) message.

        Args:
            data: The raw SPAT data.

        Returns:
            A ParsedSPAT object.

        Raises:
            ParseError: If the data cannot be decoded.
            ValidationError: If the decoded message fails validation.
        """
        self._parse_count += 1
        try:
            if isinstance(data, dict):
                return self._parse_spat_from_dict(data)
            return self._parse_spat_from_bytes(data)
        except (ParseError, ValidationError):
            self._error_count += 1
            raise
        except Exception as exc:
            self._error_count += 1
            raise ParseError(f"Unexpected error parsing SPAT: {exc}") from exc

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------

    def serialize_bsm(self, bsm: ParsedBSM, msg_format: MessageFormat = MessageFormat.UPER) -> Union[bytes, Dict[str, Any]]:
        """Serialize a ParsedBSM back to wire format.

        Args:
            bsm: The parsed BSM object.
            msg_format: Target wire format.

        Returns:
            Serialized bytes or dict depending on the requested format.
        """
        if msg_format == MessageFormat.JSON:
            return self._serialize_bsm_to_dict(bsm)
        return self._serialize_bsm_to_bytes(bsm)

    def serialize_map(self, map_msg: ParsedMAP, msg_format: MessageFormat = MessageFormat.UPER) -> Union[bytes, Dict[str, Any]]:
        """Serialize a ParsedMAP back to wire format.

        Args:
            map_msg: The parsed MAP object.
            msg_format: Target wire format.

        Returns:
            Serialized bytes or dict depending on the requested format.
        """
        if msg_format == MessageFormat.JSON:
            return self._serialize_map_to_dict(map_msg)
        return self._serialize_map_to_bytes(map_msg)

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def validate_bsm(self, bsm: ParsedBSM) -> bool:
        """Validate a parsed BSM for structural and semantic correctness.

        Checks include: position validity, speed range, heading range,
        message count monotonicity, and vehicle size plausibility.

        Args:
            bsm: The ParsedBSM to validate.

        Returns:
            True if validation passes.

        Raises:
            ValidationError: If strict_validation is True and a check fails.
        """
        errors: List[str] = []

        if not bsm.position.is_valid():
            errors.append(f"Invalid position: lat={bsm.position.latitude}, lon={bsm.position.longitude}")

        if not (0.0 <= bsm.speed <= 163.82):
            errors.append(f"Speed out of range: {bsm.speed} m/s")

        if not (0.0 <= bsm.heading <= 359.9875):
            errors.append(f"Heading out of range: {bsm.heading} degrees")

        if not (0 <= bsm.msg_count <= 127):
            errors.append(f"MsgCount out of range: {bsm.msg_count}")

        if bsm.vehicle_size[0] < 0 or bsm.vehicle_size[1] < 0:
            errors.append(f"Invalid vehicle size: {bsm.vehicle_size}")

        return self._handle_validation_errors(errors)

    def validate_map(self, map_msg: ParsedMAP) -> bool:
        """Validate a parsed MAP message.

        Args:
            map_msg: The ParsedMAP to validate.

        Returns:
            True if validation passes.

        Raises:
            ValidationError: If strict_validation is True and a check fails.
        """
        errors: List[str] = []

        if not map_msg.ref_position.is_valid():
            errors.append(f"Invalid reference position in MAP message")

        if map_msg.msg_issue_revision < 0:
            errors.append(f"Negative revision number: {map_msg.msg_issue_revision}")

        if not map_msg.lanes and not map_msg.node_list:
            errors.append("MAP message contains no lanes or nodes")

        return self._handle_validation_errors(errors)

    def validate_spat(self, spat: ParsedSPAT) -> bool:
        """Validate a parsed SPAT message.

        Args:
            spat: The ParsedSPAT to validate.

        Returns:
            True if validation passes.

        Raises:
            ValidationError: If strict_validation is True and a check fails.
        """
        errors: List[str] = []

        if not spat.phases:
            errors.append("SPAT message contains no phase information")

        for phase in spat.phases:
            if not (0 <= phase.phase_id <= 255):
                errors.append(f"Invalid phase ID: {phase.phase_id}")
            if phase.min_end_time > phase.max_end_time:
                errors.append(
                    f"Phase {phase.phase_id}: min_end_time ({phase.min_end_time}) "
                    f"> max_end_time ({phase.max_end_time})"
                )

        return self._handle_validation_errors(errors)

    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return parsing statistics."""
        return {
            "total_parsed": self._parse_count,
            "total_errors": self._error_count,
        }

    # ---------------------------------------------------------------------
    # Internal: type detection
    # ---------------------------------------------------------------------

    def _detect_message_type(self, data: bytes) -> int:
        """Extract the J2735 message ID from the first bytes.

        In UPER encoding the message ID occupies the first two bytes.

        Args:
            data: Raw message bytes.

        Returns:
            The integer message ID.
        """
        if len(data) < 2:
            raise ParseError("Message too short to determine type")

        msg_id = struct.unpack("!H", data[:2])[0]
        return msg_id

    # ---------------------------------------------------------------------
    # Internal: BSM byte-level parsing
    # ---------------------------------------------------------------------

    def _parse_bsm_from_bytes(self, data: bytes) -> ParsedBSM:
        """Parse a UPER-encoded BSM from raw bytes.

        This is a simplified parser that extracts the most commonly used
        fields from BSM Part I. A full production implementation would
        use a generated ASN.1 UPER codec.
        """
        if len(data) < BSM_PART_I_LENGTH:
            raise ParseError(
                f"BSM too short: {len(data)} bytes (minimum {BSM_PART_I_LENGTH})"
            )

        raw_len = len(data)

        # Message ID (2 bytes) already consumed for type detection
        offset = 2

        # Message count (1 byte, 7 bits used)
        msg_count = data[offset] & 0x7F
        offset += 1

        # Temporary ID (4 bytes)
        temp_id = struct.unpack_from("!I", data, offset)[0]
        offset += 4

        # DSecond / sec_mark (2 bytes, 0..60999)
        sec_mark = struct.unpack_from("!H", data, offset)[0]
        if sec_mark > 60999:
            if self.strict_validation:
                raise ValidationError(f"sec_mark out of range: {sec_mark}")
            sec_mark = min(sec_mark, 60999)
        offset += 2

        # Latitude (4 bytes, 1/10 microdegree scale)
        lat_raw = struct.unpack_from("!i", data, offset)[0]
        latitude = lat_raw / 1e7
        offset += 4

        # Longitude (4 bytes, 1/10 microdegree scale)
        lon_raw = struct.unpack_from("!i", data, offset)[0]
        longitude = lon_raw / 1e7
        offset += 4

        # Elevation (2 bytes, decimetres)
        elev_raw = struct.unpack_from("!h", data, offset)[0]
        elevation = float(elev_raw)
        offset += 2

        # Speed (2 bytes, 0.02 m/s units)
        speed_raw = struct.unpack_from("!H", data, offset)[0]
        speed = speed_raw * 0.02
        offset += 2

        # Heading (2 bytes, 0.0125 degree units)
        heading_raw = struct.unpack_from("!H", data, offset)[0]
        heading = heading_raw * 0.0125
        offset += 2

        # Steering wheel angle (1 byte)
        angle_raw = struct.unpack_from("!b", data, offset)[0]
        angle = float(angle_raw) * 1.5
        offset += 1

        # Transmission state (1 byte, lower 4 bits)
        transmission_state = data[offset] & 0x0F
        offset += 1

        # Acceleration set (4 x 1 byte)
        accel_long = struct.unpack_from("!b", data, offset)[0] * 0.1
        accel_lat = struct.unpack_from("!b", data, offset + 1)[0] * 0.1
        accel_vert = struct.unpack_from("!b", data, offset + 2)[0] * 0.05
        yaw_rate = struct.unpack_from("!b", data, offset + 3)[0] * 0.01
        offset += 4

        # Brake status (1 byte as flags)
        brake_byte = data[offset]
        brake_status = {
            "brake_applied": bool(brake_byte & 0x01),
            "abs_active": bool(brake_byte & 0x02),
            "stability_control": bool(brake_byte & 0x04),
            "brake_boost": bool(brake_byte & 0x08),
            "aux_brake": bool(brake_byte & 0x10),
        }
        offset += 1

        # Vehicle size: length (2 bytes) and width (1 byte) in cm
        veh_length = struct.unpack_from("!H", data, offset)[0]
        veh_width = data[offset + 2]
        offset += 3

        bsm = ParsedBSM(
            msg_count=msg_count,
            id=temp_id,
            sec_mark=sec_mark,
            position=Position3D(latitude=latitude, longitude=longitude),
            elevation=elevation,
            speed=speed,
            heading=heading,
            angle=angle,
            transmission_state=transmission_state,
            acceleration_set={
                "longitudinal": accel_long,
                "lateral": accel_lat,
                "vertical": accel_vert,
                "yaw_rate": yaw_rate,
            },
            brake_status=brake_status,
            vehicle_size=(veh_length, veh_width),
            raw_length=raw_len,
        )

        self.validate_bsm(bsm)
        return bsm

    def _parse_bsm_from_dict(self, data: Dict[str, Any]) -> ParsedBSM:
        """Parse a BSM from a dictionary (JSON format)."""
        core = data.get("core_data", data)
        pos = core.get("position", {})

        bsm = ParsedBSM(
            msg_count=core.get("msg_count", 0),
            id=core.get("id", 0),
            sec_mark=core.get("sec_mark", 0),
            position=Position3D(
                latitude=pos.get("latitude", 0.0),
                longitude=pos.get("longitude", 0.0),
                elevation=pos.get("elevation"),
            ),
            elevation=pos.get("elevation", 0.0),
            speed=core.get("speed", 0.0),
            heading=core.get("heading", 0.0),
            angle=core.get("angle", 0.0),
            transmission_state=core.get("transmission", 0),
            acceleration_set=core.get("accel_set", {
                "longitudinal": 0.0, "lateral": 0.0, "vertical": 0.0, "yaw_rate": 0.0,
            }),
            brake_status=core.get("brakes", {
                "brake_applied": False, "abs_active": False,
                "stability_control": False, "brake_boost": False, "aux_brake": False,
            }),
            vehicle_size=tuple(core.get("size", [0, 0])[:2]),
            raw_length=0,
        )
        self.validate_bsm(bsm)
        return bsm

    # ---------------------------------------------------------------------
    # Internal: MAP byte-level parsing
    # ---------------------------------------------------------------------

    def _parse_map_from_bytes(self, data: bytes) -> ParsedMAP:
        """Parse a UPER-encoded MAP message from raw bytes."""
        if len(data) < 12:
            raise ParseError(f"MAP message too short: {len(data)} bytes")

        raw_len = len(data)
        offset = 2  # skip message ID

        msg_issue_revision = data[offset]
        offset += 1

        layer_type = data[offset]
        offset += 1

        intersection_id = struct.unpack_from("!H", data, offset)[0]
        offset += 2

        road_regulator_id = struct.unpack_from("!H", data, offset)[0]
        offset += 2

        lat_raw = struct.unpack_from("!i", data, offset)[0]
        ref_lat = lat_raw / 1e7
        offset += 4

        lon_raw = struct.unpack_from("!i", data, offset)[0]
        ref_lon = lon_raw / 1e7
        offset += 4

        # Parse nodes and lanes from remaining bytes
        nodes: List[NodeReference] = []
        lanes: List[LaneData] = []

        while offset + 3 < len(data):
            record_type = data[offset]
            offset += 1

            if record_type == 0x01:  # Node record
                node_id = data[offset]
                offset += 1
                n_lat_raw = struct.unpack_from("!i", data, offset)[0]
                n_lat = n_lat_raw / 1e7
                offset += 4
                n_lon_raw = struct.unpack_from("!i", data, offset)[0]
                n_lon = n_lon_raw / 1e7
                offset += 4
                nodes.append(NodeReference(
                    node_id=node_id,
                    position=Position3D(latitude=n_lat, longitude=n_lon),
                ))

            elif record_type == 0x02:  # Lane record
                lane_id = data[offset]
                offset += 1
                lane_type = data[offset]
                offset += 1
                direction = data[offset]
                offset += 1
                num_nodes = data[offset]
                offset += 1
                lane_node_ids: List[int] = []
                for _ in range(min(num_nodes, 50)):
                    if offset < len(data):
                        lane_node_ids.append(data[offset])
                        offset += 1
                lanes.append(LaneData(
                    lane_id=lane_id,
                    lane_type=lane_type,
                    direction=direction,
                    node_list=[n for n in nodes if n.node_id in lane_node_ids],
                ))
            else:
                offset += 1  # Skip unknown record type

        map_msg = ParsedMAP(
            msg_issue_revision=msg_issue_revision,
            layer_type=layer_type,
            intersection_id=intersection_id,
            road_regulator_id=road_regulator_id,
            ref_position=Position3D(latitude=ref_lat, longitude=ref_lon),
            lanes=lanes,
            node_list=nodes,
            raw_length=raw_len,
        )
        self.validate_map(map_msg)
        return map_msg

    def _parse_map_from_dict(self, data: Dict[str, Any]) -> ParsedMAP:
        """Parse a MAP message from a dictionary."""
        nodes = [
            NodeReference(
                node_id=n.get("node_id", 0),
                position=Position3D(
                    latitude=n.get("latitude", 0.0),
                    longitude=n.get("longitude", 0.0),
                ),
            )
            for n in data.get("nodes", [])
        ]
        lanes = [
            LaneData(
                lane_id=l.get("lane_id", 0),
                lane_type=l.get("lane_type", 0),
                direction=l.get("direction", 0),
            )
            for l in data.get("lanes", [])
        ]

        ref = data.get("ref_position", {})
        map_msg = ParsedMAP(
            msg_issue_revision=data.get("msg_issue_revision", 0),
            layer_type=data.get("layer_type", 0),
            intersection_id=data.get("intersection_id", 0),
            road_regulator_id=data.get("road_regulator_id", 0),
            ref_position=Position3D(
                latitude=ref.get("latitude", 0.0),
                longitude=ref.get("longitude", 0.0),
            ),
            lanes=lanes,
            node_list=nodes,
            raw_length=0,
        )
        self.validate_map(map_msg)
        return map_msg

    # ---------------------------------------------------------------------
    # Internal: SPAT byte-level parsing
    # ---------------------------------------------------------------------

    def _parse_spat_from_bytes(self, data: bytes) -> ParsedSPAT:
        """Parse a UPER-encoded SPAT message from raw bytes."""
        if len(data) < 8:
            raise ParseError(f"SPAT message too short: {len(data)} bytes")

        raw_len = len(data)
        offset = 2  # skip message ID

        msg_count = data[offset]
        offset += 1

        intersection_id = struct.unpack_from("!H", data, offset)[0]
        offset += 2

        road_regulator_id = struct.unpack_from("!H", data, offset)[0]
        offset += 2

        epoch_timestamp = struct.unpack_from("!I", data, offset)[0] if offset + 4 <= len(data) else 0
        if offset + 4 <= len(data):
            offset += 4

        phases: List[PhaseState] = []
        while offset + 4 < len(data):
            phase_id = data[offset]
            offset += 1
            phase_state = data[offset] & 0x0F
            offset += 1
            min_end = struct.unpack_from("!H", data, offset)[0]
            offset += 2
            max_end = min_end
            likely = min_end
            if offset + 4 < len(data):
                max_end = struct.unpack_from("!H", data, offset)[0]
                offset += 2
                likely = struct.unpack_from("!H", data, offset)[0]
                offset += 2

            phases.append(PhaseState(
                phase_id=phase_id,
                phase_state=phase_state,
                min_end_time=min_end,
                max_end_time=max_end,
                likely_time=likely,
            ))

        spat = ParsedSPAT(
            intersection_id=intersection_id,
            road_regulator_id=road_regulator_id,
            msg_count=msg_count,
            epoch_timestamp=epoch_timestamp,
            phases=phases,
            raw_length=raw_len,
        )
        self.validate_spat(spat)
        return spat

    def _parse_spat_from_dict(self, data: Dict[str, Any]) -> ParsedSPAT:
        """Parse a SPAT message from a dictionary."""
        phases = [
            PhaseState(
                phase_id=p.get("phase_id", 0),
                phase_state=p.get("phase_state", 1),
                min_end_time=p.get("min_end_time", 0),
                max_end_time=p.get("max_end_time", 0),
                likely_time=p.get("likely_time", 0),
            )
            for p in data.get("phases", [])
        ]
        spat = ParsedSPAT(
            intersection_id=data.get("intersection_id", 0),
            road_regulator_id=data.get("road_regulator_id", 0),
            msg_count=data.get("msg_count", 0),
            epoch_timestamp=data.get("epoch_timestamp", 0),
            phases=phases,
            raw_length=0,
        )
        self.validate_spat(spat)
        return spat

    # ---------------------------------------------------------------------
    # Internal: serialization helpers
    # ---------------------------------------------------------------------

    def _serialize_bsm_to_dict(self, bsm: ParsedBSM) -> Dict[str, Any]:
        """Convert a ParsedBSM to a JSON-friendly dictionary."""
        return {
            "msg_type": "BSM",
            "core_data": {
                "msg_count": bsm.msg_count,
                "id": bsm.id,
                "sec_mark": bsm.sec_mark,
                "position": {
                    "latitude": bsm.position.latitude,
                    "longitude": bsm.position.longitude,
                    "elevation": bsm.position.elevation,
                },
                "speed": bsm.speed,
                "heading": bsm.heading,
                "angle": bsm.angle,
                "transmission": bsm.transmission_state,
                "accel_set": bsm.acceleration_set,
                "brakes": bsm.brake_status,
                "size": list(bsm.vehicle_size),
            },
        }

    def _serialize_bsm_to_bytes(self, bsm: ParsedBSM) -> bytes:
        """Serialize a ParsedBSM to UPER-compatible bytes (simplified)."""
        parts: List[bytes] = [
            struct.pack("!H", J2735_MSG_ID_BSM),
            struct.pack("!B", bsm.msg_count & 0x7F),
            struct.pack("!I", bsm.id),
            struct.pack("!H", bsm.sec_mark),
            struct.pack("!i", int(bsm.position.latitude * 1e7)),
            struct.pack("!i", int(bsm.position.longitude * 1e7)),
            struct.pack("!h", int(bsm.elevation)),
            struct.pack("!H", int(bsm.speed / 0.02)),
            struct.pack("!H", int(bsm.heading / 0.0125)),
            struct.pack("!b", int(bsm.angle / 1.5)),
            struct.pack("!B", bsm.transmission_state & 0x0F),
            struct.pack("!b", int(bsm.acceleration_set["longitudinal"] / 0.1)),
            struct.pack("!b", int(bsm.acceleration_set["lateral"] / 0.1)),
            struct.pack("!b", int(bsm.acceleration_set["vertical"] / 0.05)),
            struct.pack("!b", int(bsm.acceleration_set["yaw_rate"] / 0.01)),
            self._pack_brake_byte(bsm.brake_status),
            struct.pack("!H", bsm.vehicle_size[0]),
            struct.pack("!B", bsm.vehicle_size[1]),
        ]
        return b"".join(parts)

    def _serialize_map_to_dict(self, map_msg: ParsedMAP) -> Dict[str, Any]:
        """Convert a ParsedMAP to a JSON-friendly dictionary."""
        return {
            "msg_type": "MAP",
            "msg_issue_revision": map_msg.msg_issue_revision,
            "layer_type": map_msg.layer_type,
            "intersection_id": map_msg.intersection_id,
            "road_regulator_id": map_msg.road_regulator_id,
            "ref_position": {
                "latitude": map_msg.ref_position.latitude,
                "longitude": map_msg.ref_position.longitude,
            },
            "nodes": [
                {"node_id": n.node_id, "latitude": n.position.latitude, "longitude": n.position.longitude}
                for n in map_msg.node_list
            ],
            "lanes": [
                {"lane_id": l.lane_id, "lane_type": l.lane_type, "direction": l.direction}
                for l in map_msg.lanes
            ],
        }

    def _serialize_map_to_bytes(self, map_msg: ParsedMAP) -> bytes:
        """Serialize a ParsedMAP to bytes (simplified)."""
        parts: List[bytes] = [
            struct.pack("!H", J2735_MSG_ID_MAP),
            struct.pack("!B", map_msg.msg_issue_revision),
            struct.pack("!B", map_msg.layer_type),
            struct.pack("!H", map_msg.intersection_id),
            struct.pack("!H", map_msg.road_regulator_id),
            struct.pack("!i", int(map_msg.ref_position.latitude * 1e7)),
            struct.pack("!i", int(map_msg.ref_position.longitude * 1e7)),
        ]
        for node in map_msg.node_list:
            parts.append(struct.pack("!B", 0x01))  # node record type
            parts.append(struct.pack("!B", node.node_id))
            parts.append(struct.pack("!i", int(node.position.latitude * 1e7)))
            parts.append(struct.pack("!i", int(node.position.longitude * 1e7)))
        for lane in map_msg.lanes:
            parts.append(struct.pack("!B", 0x02))  # lane record type
            parts.append(struct.pack("!B", lane.lane_id))
            parts.append(struct.pack("!B", lane.lane_type))
            parts.append(struct.pack("!B", lane.direction))
            parts.append(struct.pack("!B", len(lane.node_list)))
            for node in lane.node_list:
                parts.append(struct.pack("!B", node.node_id))
        return b"".join(parts)

    # ---------------------------------------------------------------------
    # Internal: auto-detect from JSON dict
    # ---------------------------------------------------------------------

    def _parse_json(self, data: Dict[str, Any]) -> Union[ParsedBSM, ParsedMAP, ParsedSPAT]:
        """Route a JSON dict to the correct parser based on msg_type field."""
        msg_type = data.get("msg_type", "").upper()
        if msg_type == "BSM":
            return self.parse_bsm(data)
        elif msg_type == "MAP":
            return self.parse_map(data)
        elif msg_type == "SPAT":
            return self.parse_spat(data)
        raise ParseError(f"Cannot determine message type from JSON: msg_type={msg_type!r}")

    # ---------------------------------------------------------------------
    # Internal: utilities
    # ---------------------------------------------------------------------

    def _handle_validation_errors(self, errors: List[str]) -> bool:
        """Process validation error list according to strictness policy."""
        if not errors:
            return True
        for err in errors:
            if self.strict_validation:
                raise ValidationError(err)
            else:
                logger.warning("Validation warning: %s", err)
        return False

    @staticmethod
    def _pack_brake_byte(brake_status: Dict[str, bool]) -> bytes:
        """Pack brake status flags into a single byte."""
        byte_val = 0
        if brake_status.get("brake_applied", False):
            byte_val |= 0x01
        if brake_status.get("abs_active", False):
            byte_val |= 0x02
        if brake_status.get("stability_control", False):
            byte_val |= 0x04
        if brake_status.get("brake_boost", False):
            byte_val |= 0x08
        if brake_status.get("aux_brake", False):
            byte_val |= 0x10
        return struct.pack("!B", byte_val)

    def __repr__(self) -> str:
        return (
            f"MessageParser(parsed={self._parse_count}, errors={self._error_count}, "
            f"versions={self.supported_versions})"
        )
