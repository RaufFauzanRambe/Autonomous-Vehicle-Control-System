"""
Collective Awareness Message Processing Module

This module implements the CAMPProcessor class for processing Cooperative
Awareness Messages (CAM) as defined in ETSI EN 302 637-2. It handles
ego vehicle status broadcasting, remote vehicle status reception and
tracking, and reception quality monitoring. The processor maintains a
dynamic Local Dynamic Map (LDM) of nearby vehicles.

References:
    - ETSI EN 302 637-2: Cooperative Awareness Basic Service
    - ETSI TS 102 894-2: CAM Transmission Parameters
    - ETSI TS 103 579: Communication Congestion Control
    - ISO 21217: ITS Station Reference Architecture
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAM_DEFAULT_INTERVAL_S: float = 1.0        # 1 Hz default broadcast rate
CAM_MIN_INTERVAL_S: float = 0.1            # 10 Hz maximum rate
CAM_MAX_INTERVAL_S: float = 1.0            # 1 Hz minimum rate

CAM_STATION_TYPE_PASSENGER_CAR: int = 5
CAM_STATION_TYPE_BUS: int = 6
CAM_STATION_TYPE_TRUCK: int = 7
CAM_STATION_TYPE_MOTORCYCLE: int = 8
CAM_STATION_TYPE_RSU: int = 15

RECEPTION_QUALITY_WINDOW_S: float = 10.0   # Sliding window for quality metrics
REMOTE_VEHICLE_TIMEOUT_S: float = 5.0      # Timeout for stale remote vehicles
MAX_REMOTE_VEHICLES: int = 200             # Maximum tracked remote vehicles
LDM_MAX_ENTRIES: int = 500                 # LDM capacity limit


class CAMGenerationTrigger(Enum):
    """Triggers for CAM generation per ETSI TS 102 894-2."""
    PERIODIC = "periodic"                   # Timer-based periodic trigger
    HEADING_CHANGE = "heading_change"       # Heading changed > 4°
    POSITION_CHANGE = "position_change"     # Distance > 5 m from last CAM
    SPEED_CHANGE = "speed_change"           # Speed changed > 0.5 m/s


class VehicleRole(IntEnum):
    """Vehicle role as defined in ETSI CAM."""
    DEFAULT = 0
    PUBLIC_TRANSPORT = 1
    SPECIAL_TRANSPORT = 2
    DANGEROUS_GOODS = 3
    ROAD_WORK = 4
    RESCUE = 5
    EMERGENCY = 6
    SAFETY_CAR = 7
    AGRICULTURAL = 8
    COMMERCIAL = 9
    MILITARY = 10
    ROAD_SIDE_ASSISTANCE = 11


class ReceptionQualityLevel(Enum):
    """Qualitative reception quality rating."""
    EXCELLENT = "excellent"     # Packet loss < 1%
    GOOD = "good"               # Packet loss 1-5%
    FAIR = "fair"               # Packet loss 5-15%
    POOR = "poor"               # Packet loss 15-30%
    CRITICAL = "critical"       # Packet loss > 30%


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EgoVehicleStatus:
    """Current status of the ego vehicle for CAM broadcasting.

    Attributes:
        station_id: Temporary station identifier (privacy-rotated).
        station_type: ETSI station type code.
        latitude: WGS-84 latitude in degrees.
        longitude: WGS-84 longitude in degrees.
        altitude: Altitude above WGS-84 ellipsoid in metres.
        speed: Vehicle speed in m/s.
        heading: Heading in degrees from true north (0..360).
        drive_direction: Forward (0) or reverse (1).
        vehicle_length: Vehicle length in decimetres.
        vehicle_width: Vehicle width in decimetres.
        curvature: Path curvature in 1/m (positive = left turn).
        curvature_calculation_mode: How curvature was calculated.
        yaw_rate: Yaw rate in degrees/s.
        acceleration: Longitudinal acceleration in m/s².
        lane_position: Lane position code.
        steering_angle: Steering wheel angle in degrees.
        lateral_acceleration: Lateral acceleration in m/s².
        vertical_acceleration: Vertical acceleration in m/s².
        confidence: Position confidence indicator (0..15).
        timestamp: Time of this status measurement.
    """
    station_id: int = 0
    station_type: int = CAM_STATION_TYPE_PASSENGER_CAR
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    drive_direction: int = 0
    vehicle_length: float = 45.0   # 4.5 m = 45 dm
    vehicle_width: float = 18.0    # 1.8 m = 18 dm
    curvature: float = 0.0
    curvature_calculation_mode: int = 0
    yaw_rate: float = 0.0
    acceleration: float = 0.0
    lane_position: int = 0
    steering_angle: float = 0.0
    lateral_acceleration: float = 0.0
    vertical_acceleration: float = 0.0
    confidence: int = 15
    timestamp: float = field(default_factory=time.time)

    def distance_to(self, other_lat: float, other_lon: float) -> float:
        """Estimate horizontal distance to a given lat/lon using the Haversine formula.

        Args:
            other_lat: Target latitude in degrees.
            other_lon: Target longitude in degrees.

        Returns:
            Distance in metres.
        """
        R = 6_371_000.0  # Earth radius in metres
        dlat = math.radians(other_lat - self.latitude)
        dlon = math.radians(other_lon - self.longitude)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(self.latitude)) * math.cos(math.radians(other_lat)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


@dataclass
class RemoteVehicleStatus:
    """Status of a remote vehicle as received via CAM.

    Attributes:
        station_id: Remote station identifier.
        station_type: ETSI station type code.
        latitude: WGS-84 latitude.
        longitude: WGS-84 longitude.
        altitude: Altitude in metres.
        speed: Speed in m/s.
        heading: Heading in degrees.
        drive_direction: Forward or reverse.
        vehicle_length: Length in decimetres.
        vehicle_width: Width in decimetres.
        curvature: Path curvature in 1/m.
        yaw_rate: Yaw rate in degrees/s.
        acceleration: Longitudinal acceleration in m/s².
        last_update: Timestamp of the most recently received CAM.
        first_seen: Timestamp when the vehicle was first detected.
        cam_count: Number of CAMs received from this vehicle.
        generation_time: Generation timestamp from the last CAM.
        seq_number: Sequence number from the last CAM.
    """
    station_id: int = 0
    station_type: int = CAM_STATION_TYPE_PASSENGER_CAR
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    drive_direction: int = 0
    vehicle_length: float = 0.0
    vehicle_width: float = 0.0
    curvature: float = 0.0
    yaw_rate: float = 0.0
    acceleration: float = 0.0
    last_update: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    cam_count: int = 1
    generation_time: float = 0.0
    seq_number: int = 0

    def age_seconds(self) -> float:
        """Return the number of seconds since the last update."""
        return time.time() - self.last_update

    def is_stale(self, timeout: float = REMOTE_VEHICLE_TIMEOUT_S) -> bool:
        """Return True if this vehicle entry has not been updated within timeout."""
        return self.age_seconds() > timeout


@dataclass
class ReceptionMetrics:
    """Reception quality metrics for a remote station.

    Attributes:
        station_id: Remote station identifier.
        expected_cam_count: Number of CAMs expected in the measurement window.
        received_cam_count: Number of CAMs actually received.
        avg_rssi_dbm: Average RSSI across received CAMs.
        packet_loss_rate: Estimated packet loss ratio (0..1).
        avg_latency_ms: Average CAM delivery latency in milliseconds.
        last_received: Timestamp of the most recently received CAM.
    """
    station_id: int = 0
    expected_cam_count: int = 0
    received_cam_count: int = 0
    avg_rssi_dbm: float = -80.0
    packet_loss_rate: float = 0.0
    avg_latency_ms: float = 0.0
    last_received: float = field(default_factory=time.time)

    @property
    def quality_level(self) -> ReceptionQualityLevel:
        """Map packet loss rate to a qualitative quality level."""
        if self.packet_loss_rate < 0.01:
            return ReceptionQualityLevel.EXCELLENT
        elif self.packet_loss_rate < 0.05:
            return ReceptionQualityLevel.GOOD
        elif self.packet_loss_rate < 0.15:
            return ReceptionQualityLevel.FAIR
        elif self.packet_loss_rate < 0.30:
            return ReceptionQualityLevel.POOR
        return ReceptionQualityLevel.CRITICAL


@dataclass
class CAMMessage:
    """Serialized Cooperative Awareness Message.

    Attributes:
        station_id: Originating station identifier.
        generation_time: CAM generation timestamp.
        seq_number: Sequence number for duplicate detection.
        payload: Encoded CAM payload bytes.
        rssi: Received signal strength (for inbound CAMs only).
    """
    station_id: int = 0
    generation_time: float = 0.0
    seq_number: int = 0
    payload: bytes = b""
    rssi: float = 0.0


# ---------------------------------------------------------------------------
# CAMPProcessor
# ---------------------------------------------------------------------------

class CAMPProcessor:
    """Processor for Cooperative Awareness Messages (CAM).

    The CAMPProcessor manages the complete CAM lifecycle:
      - Broadcasting ego vehicle status at adaptive rates
      - Receiving and tracking remote vehicle status
      - Maintaining a Local Dynamic Map (LDM) of nearby vehicles
      - Monitoring reception quality for each remote station
      - Detecting and reporting communication quality degradation

    Thread Safety:
        All public methods are thread-safe. The processor runs an
        internal broadcasting thread and a cleanup thread.

    Usage Example::

        processor = CAMPProcessor(station_id=1001)
        processor.start()

        # Update ego vehicle status (called by vehicle state manager)
        processor.update_ego_status(ego_status)

        # Process an incoming CAM from the DSRC interface
        processor.process_received_cam(cam_msg)

        # Query nearby vehicles
        nearby = processor.get_nearby_vehicles(radius_m=100.0)
    """

    def __init__(
        self,
        station_id: int = 0,
        broadcast_interval: float = CAM_DEFAULT_INTERVAL_S,
        vehicle_timeout: float = REMOTE_VEHICLE_TIMEOUT_S,
        max_remote_vehicles: int = MAX_REMOTE_VEHICLES,
        station_type: int = CAM_STATION_TYPE_PASSENGER_CAR,
        vehicle_role: VehicleRole = VehicleRole.DEFAULT,
        enable_adaptive_rate: bool = True,
    ) -> None:
        """Initialize the CAMPProcessor.

        Args:
            station_id: Temporary station identifier (should be periodically rotated).
            broadcast_interval: Default CAM broadcast interval in seconds.
            vehicle_timeout: Seconds before a remote vehicle is considered stale.
            max_remote_vehicles: Maximum number of remote vehicles tracked.
            station_type: ETSI station type code for the ego vehicle.
            vehicle_role: Operational role of the ego vehicle.
            enable_adaptive_rate: Whether to use adaptive CAM rate control.
        """
        self.station_id: int = station_id
        self.broadcast_interval: float = broadcast_interval
        self.vehicle_timeout: float = vehicle_timeout
        self.max_remote_vehicles: int = max_remote_vehicles
        self.station_type: int = station_type
        self.vehicle_role: VehicleRole = vehicle_role
        self.enable_adaptive_rate: bool = enable_adaptive_rate

        self._lock: threading.RLock = threading.RLock()
        self._ego_status: EgoVehicleStatus = EgoVehicleStatus(station_id=station_id, station_type=station_type)
        self._remote_vehicles: OrderedDict[int, RemoteVehicleStatus] = OrderedDict()
        self._reception_metrics: Dict[int, ReceptionMetrics] = {}
        self._last_broadcast_status: Optional[EgoVehicleStatus] = None
        self._last_broadcast_time: float = 0.0
        self._cam_seq_number: int = 0
        self._adaptive_interval: float = broadcast_interval
        self._generation_triggers: List[CAMGenerationTrigger] = []

        # Sliding-window reception tracking
        self._rx_window: Dict[int, List[Tuple[float, float]]] = {}  # station_id -> [(timestamp, rssi)]

        self._running: threading.Event = threading.Event()
        self._broadcast_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None

        # Callback for when a CAM should be transmitted
        self._transmit_callback: Optional[Callable[[CAMMessage], None]] = None
        # Callback for when remote vehicle data changes
        self._ldm_update_callback: Optional[Callable[[Dict[int, RemoteVehicleStatus]], None]] = None

        self._stats: Dict[str, int] = {
            "cam_sent": 0,
            "cam_received": 0,
            "cam_duplicates": 0,
            "vehicles_added": 0,
            "vehicles_expired": 0,
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the CAMP processor and its background threads.

        Spawns a periodic broadcast thread and a stale-vehicle cleanup
        thread.
        """
        logger.info("CAMPProcessor starting (station_id=%d)", self.station_id)
        self._running.set()

        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name=f"camp-broadcast-{self.station_id}",
            daemon=True,
        )
        self._broadcast_thread.start()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name=f"camp-cleanup-{self.station_id}",
            daemon=True,
        )
        self._cleanup_thread.start()

        logger.info("CAMPProcessor started")

    def stop(self) -> None:
        """Stop the CAMP processor and all background threads."""
        logger.info("CAMPProcessor stopping (station_id=%d)", self.station_id)
        self._running.clear()

        if self._broadcast_thread is not None:
            self._broadcast_thread.join(timeout=5.0)
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5.0)

        with self._lock:
            self._remote_vehicles.clear()
            self._reception_metrics.clear()

        logger.info("CAMPProcessor stopped")

    # -------------------------------------------------------------------------
    # Ego vehicle status
    # -------------------------------------------------------------------------

    def update_ego_status(self, status: EgoVehicleStatus) -> None:
        """Update the ego vehicle status used for CAM generation.

        This method should be called whenever the vehicle's kinematic
        state changes (e.g., from the localization or control module).

        Args:
            status: The current ego vehicle status.
        """
        with self._lock:
            self._ego_status = status
            self._ego_status.timestamp = time.time()

            if self.enable_adaptive_rate:
                self._check_generation_triggers()

    def get_ego_status(self) -> EgoVehicleStatus:
        """Return the current ego vehicle status."""
        with self._lock:
            return EgoVehicleStatus(**self._ego_status.__dict__)

    # -------------------------------------------------------------------------
    # CAM broadcasting
    # -------------------------------------------------------------------------

    def set_transmit_callback(self, callback: Callable[[CAMMessage], None]) -> None:
        """Set the callback invoked when a CAM is ready for transmission.

        The callback is responsible for handing the CAM to the V2X
        communication stack for over-the-air transmission.

        Args:
            callback: Function accepting a CAMMessage.
        """
        with self._lock:
            self._transmit_callback = callback

    def generate_cam(self) -> Optional[CAMMessage]:
        """Generate a CAM from the current ego vehicle status.

        Applies adaptive rate control by checking whether enough time
        has passed since the last broadcast. Returns None if the
        current interval has not elapsed.

        Returns:
            A CAMMessage if a CAM should be sent, None otherwise.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_broadcast_time
            if elapsed < self._adaptive_interval:
                return None

            self._cam_seq_number = (self._cam_seq_number + 1) % 65536
            cam = CAMMessage(
                station_id=self._ego_status.station_id,
                generation_time=now,
                seq_number=self._cam_seq_number,
                payload=self._encode_cam(self._ego_status),
            )

            self._last_broadcast_status = EgoVehicleStatus(**self._ego_status.__dict__)
            self._last_broadcast_time = now
            self._stats["cam_sent"] += 1
            self._generation_triggers.clear()

            return cam

    def _broadcast_loop(self) -> None:
        """Background loop that periodically generates and transmits CAMs."""
        logger.debug("CAM broadcast loop started")
        while self._running.is_set():
            cam = self.generate_cam()
            if cam is not None:
                with self._lock:
                    if self._transmit_callback is not None:
                        try:
                            self._transmit_callback(cam)
                        except Exception:
                            logger.exception("Transmit callback error for CAM seq=%d", cam.seq_number)
                    else:
                        logger.debug("No transmit callback; CAM seq=%d generated but not sent", cam.seq_number)

            # Sleep for a fraction of the adaptive interval to allow
            # event-triggered early transmission
            time.sleep(min(self._adaptive_interval / 4, 0.05))

        logger.debug("CAM broadcast loop stopped")

    # -------------------------------------------------------------------------
    # Remote vehicle reception
    # -------------------------------------------------------------------------

    def process_received_cam(self, cam: CAMMessage) -> bool:
        """Process a received Cooperative Awareness Message.

        Updates the remote vehicle table, reception metrics, and
        triggers the LDM update callback if the vehicle data changed.

        Args:
            cam: The received CAMMessage.

        Returns:
            True if the CAM was processed (not a duplicate), False otherwise.
        """
        with self._lock:
            self._stats["cam_received"] += 1
            remote = self._decode_cam(cam)

            # Check for duplicate
            existing = self._remote_vehicles.get(remote.station_id)
            if existing is not None and existing.seq_number == remote.seq_number:
                self._stats["cam_duplicates"] += 1
                return False

            # Update or add remote vehicle
            if existing is not None:
                remote.first_seen = existing.first_seen
                remote.cam_count = existing.cam_count + 1
                self._remote_vehicles.move_to_end(remote.station_id)
            else:
                if len(self._remote_vehicles) >= self.max_remote_vehicles:
                    self._evict_oldest_vehicle()
                self._stats["vehicles_added"] += 1

            self._remote_vehicles[remote.station_id] = remote

            # Update reception metrics
            self._update_reception_metrics(cam)

            # Notify LDM subscribers
            if self._ldm_update_callback is not None:
                try:
                    self._ldm_update_callback(dict(self._remote_vehicles))
                except Exception:
                    logger.exception("LDM update callback error")

            return True

    def get_nearby_vehicles(self, radius_m: float = 100.0) -> List[RemoteVehicleStatus]:
        """Return remote vehicles within the specified radius of the ego vehicle.

        Args:
            radius_m: Maximum distance in metres from the ego vehicle.

        Returns:
            List of RemoteVehicleStatus objects within the radius.
        """
        with self._lock:
            result: List[RemoteVehicleStatus] = []
            ego = self._ego_status
            for rv in self._remote_vehicles.values():
                dist = ego.distance_to(rv.latitude, rv.longitude)
                if dist <= radius_m:
                    result.append(RemoteVehicleStatus(**rv.__dict__))
            return result

    def get_remote_vehicle(self, station_id: int) -> Optional[RemoteVehicleStatus]:
        """Look up a specific remote vehicle by station ID.

        Args:
            station_id: The remote station identifier.

        Returns:
            RemoteVehicleStatus if found and not stale, None otherwise.
        """
        with self._lock:
            rv = self._remote_vehicles.get(station_id)
            if rv is None or rv.is_stale(self.vehicle_timeout):
                return None
            return RemoteVehicleStatus(**rv.__dict__)

    def get_all_remote_vehicles(self) -> Dict[int, RemoteVehicleStatus]:
        """Return a snapshot of all tracked remote vehicles."""
        with self._lock:
            return {sid: RemoteVehicleStatus(**rv.__dict__) for sid, rv in self._remote_vehicles.items()}

    def set_ldm_update_callback(self, callback: Callable[[Dict[int, RemoteVehicleStatus]], None]) -> None:
        """Register a callback for LDM (Local Dynamic Map) updates.

        The callback is invoked whenever the remote vehicle table changes
        due to a received CAM or a vehicle expiry.

        Args:
            callback: Function accepting a dict mapping station_id -> RemoteVehicleStatus.
        """
        with self._lock:
            self._ldm_update_callback = callback

    # -------------------------------------------------------------------------
    # Reception quality monitoring
    # -------------------------------------------------------------------------

    def get_reception_quality(self, station_id: int) -> Optional[ReceptionMetrics]:
        """Return reception quality metrics for a specific remote station.

        Args:
            station_id: The remote station identifier.

        Returns:
            ReceptionMetrics if available, None otherwise.
        """
        with self._lock:
            metrics = self._reception_metrics.get(station_id)
            if metrics is None:
                return None
            return ReceptionMetrics(**metrics.__dict__)

    def get_all_reception_quality(self) -> Dict[int, ReceptionMetrics]:
        """Return reception quality metrics for all tracked remote stations."""
        with self._lock:
            return {sid: ReceptionMetrics(**m.__dict__) for sid, m in self._reception_metrics.items()}

    def get_overall_quality(self) -> ReceptionQualityLevel:
        """Return the aggregate reception quality across all remote stations.

        Computed as the worst-case quality level among all active peers.
        """
        with self._lock:
            if not self._reception_metrics:
                return ReceptionQualityLevel.EXCELLENT

            worst = ReceptionQualityLevel.EXCELLENT
            quality_order = list(ReceptionQualityLevel)
            for metrics in self._reception_metrics.values():
                idx = quality_order.index(metrics.quality_level)
                worst_idx = quality_order.index(worst)
                if idx > worst_idx:
                    worst = metrics.quality_level
            return worst

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return processor statistics."""
        return dict(self._stats)

    def get_tracked_vehicle_count(self) -> int:
        """Return the number of currently tracked remote vehicles."""
        with self._lock:
            return len(self._remote_vehicles)

    # -------------------------------------------------------------------------
    # Internal: adaptive rate control
    # -------------------------------------------------------------------------

    def _check_generation_triggers(self) -> None:
        """Check whether any ETSI-defined trigger conditions are met.

        When a trigger is detected the adaptive interval is shortened
        to increase the CAM broadcast rate, ensuring nearby vehicles
        receive timely updates.
        """
        if self._last_broadcast_status is None:
            return

        ego = self._ego_status
        last = self._last_broadcast_status

        # Heading change > 4°
        heading_diff = abs(ego.heading - last.heading)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        if heading_diff > 4.0:
            self._generation_triggers.append(CAMGenerationTrigger.HEADING_CHANGE)
            self._adaptive_interval = max(CAM_MIN_INTERVAL_S, self.broadcast_interval / 4)
            return

        # Position change > 5 m
        dist = last.distance_to(ego.latitude, ego.longitude)
        if dist > 5.0:
            self._generation_triggers.append(CAMGenerationTrigger.POSITION_CHANGE)
            self._adaptive_interval = max(CAM_MIN_INTERVAL_S, self.broadcast_interval / 4)
            return

        # Speed change > 0.5 m/s
        if abs(ego.speed - last.speed) > 0.5:
            self._generation_triggers.append(CAMGenerationTrigger.SPEED_CHANGE)
            self._adaptive_interval = max(CAM_MIN_INTERVAL_S, self.broadcast_interval / 2)
            return

        # No trigger: gradually restore default interval
        if self._adaptive_interval < self.broadcast_interval:
            self._adaptive_interval = min(
                self.broadcast_interval,
                self._adaptive_interval * 1.1,
            )

    # -------------------------------------------------------------------------
    # Internal: reception metrics
    # -------------------------------------------------------------------------

    def _update_reception_metrics(self, cam: CAMMessage) -> None:
        """Update reception quality metrics based on a received CAM.

        Uses a sliding window to estimate packet loss rate from the
        ratio of received to expected CAMs.
        """
        sid = cam.station_id
        now = time.time()

        if sid not in self._rx_window:
            self._rx_window[sid] = []
        self._rx_window[sid].append((now, cam.rssi))

        # Prune entries outside the measurement window
        self._rx_window[sid] = [
            (t, rssi) for t, rssi in self._rx_window[sid]
            if (now - t) <= RECEPTION_QUALITY_WINDOW_S
        ]

        received = len(self._rx_window[sid])
        rssi_values = [rssi for _, rssi in self._rx_window[sid]]

        # Estimate expected count based on nominal CAM rate
        window_duration = min(RECEPTION_QUALITY_WINDOW_S, now - self._rx_window[sid][0][0])
        expected = max(1, int(window_duration / self.broadcast_interval))

        packet_loss = max(0.0, 1.0 - (received / max(1, expected)))
        avg_rssi = sum(rssi_values) / len(rssi_values) if rssi_values else -80.0

        # Estimate latency from generation time
        latency_ms = (now - cam.generation_time) * 1000.0 if cam.generation_time > 0 else 0.0

        self._reception_metrics[sid] = ReceptionMetrics(
            station_id=sid,
            expected_cam_count=expected,
            received_cam_count=received,
            avg_rssi_dbm=avg_rssi,
            packet_loss_rate=packet_loss,
            avg_latency_ms=max(0.0, latency_ms),
            last_received=now,
        )

    # -------------------------------------------------------------------------
    # Internal: cleanup
    # -------------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Background loop that removes stale remote vehicle entries."""
        logger.debug("CAM cleanup loop started")
        while self._running.is_set():
            time.sleep(self.vehicle_timeout / 2)
            self._remove_stale_vehicles()
        logger.debug("CAM cleanup loop stopped")

    def _remove_stale_vehicles(self) -> None:
        """Remove remote vehicles that have not been updated within the timeout."""
        with self._lock:
            stale_ids: List[int] = []
            for sid, rv in self._remote_vehicles.items():
                if rv.is_stale(self.vehicle_timeout):
                    stale_ids.append(sid)

            for sid in stale_ids:
                del self._remote_vehicles[sid]
                self._rx_window.pop(sid, None)
                self._reception_metrics.pop(sid, None)
                self._stats["vehicles_expired"] += 1
                logger.debug("Expired remote vehicle station_id=%d", sid)

            if stale_ids and self._ldm_update_callback is not None:
                try:
                    self._ldm_update_callback(dict(self._remote_vehicles))
                except Exception:
                    logger.exception("LDM update callback error during cleanup")

    def _evict_oldest_vehicle(self) -> None:
        """Evict the oldest (first-inserted) remote vehicle to make room.

        Called when the remote vehicle table exceeds max_remote_vehicles.
        """
        if self._remote_vehicles:
            evicted_id, _ = self._remote_vehicles.popitem(last=False)
            self._rx_window.pop(evicted_id, None)
            self._reception_metrics.pop(evicted_id, None)
            logger.debug("Evicted oldest remote vehicle station_id=%d", evicted_id)

    # -------------------------------------------------------------------------
    # Internal: CAM encoding / decoding (simplified)
    # -------------------------------------------------------------------------

    def _encode_cam(self, status: EgoVehicleStatus) -> bytes:
        """Encode ego vehicle status into a simplified CAM payload.

        In a production system this would use an ASN.1 UPER codec
        generated from the ETSI CAM ASN.1 module. This implementation
        uses a compact binary format for demonstration purposes.

        Args:
            status: Current ego vehicle status.

        Returns:
            Encoded CAM payload bytes.
        """
        import struct
        payload = struct.pack(
            "!Iiiddffhff",
            status.station_id,
            int(status.latitude * 1e7),
            int(status.longitude * 1e7),
            status.altitude,
            status.speed,
            status.heading,
            int(status.vehicle_length),
            status.curvature,
            status.yaw_rate,
            status.acceleration,
        )
        return payload

    def _decode_cam(self, cam: CAMMessage) -> RemoteVehicleStatus:
        """Decode a received CAM into a RemoteVehicleStatus.

        Args:
            cam: The received CAMMessage.

        Returns:
            A RemoteVehicleStatus populated from the CAM payload.
        """
        import struct

        now = time.time()

        # If the payload has the expected size, decode it
        expected_size = struct.calcsize("!Iiiddffhff")
        if len(cam.payload) >= expected_size:
            vals = struct.unpack("!Iiiddffhff", cam.payload[:expected_size])
            return RemoteVehicleStatus(
                station_id=vals[0],
                latitude=vals[1] / 1e7,
                longitude=vals[2] / 1e7,
                altitude=vals[3],
                speed=vals[4],
                heading=vals[5],
                vehicle_length=float(vals[6]),
                curvature=vals[7],
                yaw_rate=vals[8],
                acceleration=vals[9],
                last_update=now,
                first_seen=now,
                cam_count=1,
                generation_time=cam.generation_time,
                seq_number=cam.seq_number,
            )

        # Fallback: minimal decode from CAM header fields
        return RemoteVehicleStatus(
            station_id=cam.station_id,
            last_update=now,
            first_seen=now,
            cam_count=1,
            generation_time=cam.generation_time,
            seq_number=cam.seq_number,
        )

    def __repr__(self) -> str:
        return (
            f"CAMPProcessor(station_id={self.station_id}, "
            f"tracked={self.get_tracked_vehicle_count()}, "
            f"interval={self._adaptive_interval:.2f}s, "
            f"sent={self._stats['cam_sent']}, "
            f"received={self._stats['cam_received']})"
        )
