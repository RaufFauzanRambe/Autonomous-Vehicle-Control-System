"""
File:        python/lidar.py
Brief:       LidarModule — driver for RPLidar (UART) and Velodyne VLP-16
             (UDP). Parses scans, converts to numpy point clouds, and
             provides a simple obstacle-detection API.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import serial  # pyserial
    _SERIAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    serial = None  # type: ignore
    _SERIAL_AVAILABLE = False


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
class LidarKind(str, Enum):
    """Supported LiDAR models."""

    RPLIDAR = "rplidar"
    VELODYNE = "velodyne"


@dataclass(slots=True)
class LidarScan:
    """One full revolution."""

    angles: np.ndarray          # radians, shape (N,)
    ranges: np.ndarray          # meters, shape (N,)
    intensities: Optional[np.ndarray]  # shape (N,) or None
    timestamp: float
    scan_id: int

    @property
    def points(self) -> np.ndarray:
        """Cartesian points, shape (N, 2)."""
        x = self.ranges * np.cos(self.angles)
        y = self.ranges * np.sin(self.angles)
        return np.stack([x, y], axis=-1)

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of points with finite, non-zero range."""
        return np.isfinite(self.ranges) & (self.ranges > 0.05)


# ----------------------------------------------------------------------
# RPLidar protocol constants (RPLIDAR A1/A2)
# ----------------------------------------------------------------------
RPLIDAR_SYNC_BYTE1 = 0xA5
RPLIDAR_SYNC_BYTE2 = 0x5A

RPLIDAR_CMD_GET_INFO = 0x50
RPLIDAR_CMD_GET_HEALTH = 0x52
RPLIDAR_CMD_STOP = 0x25
RPLIDAR_CMD_SCAN = 0x20
RPLIDAR_CMD_FORCE_SCAN = 0x21

RPLIDAR_RESP_MEASUREMENT_SYNCBIT1 = 0xA1  # bit 0 of quality indicates sync (start of revolution)


# ----------------------------------------------------------------------
# Lidar module
# ----------------------------------------------------------------------
class LidarModule:
    """Driver for RPLidar and Velodyne VLP-16 sensors.

    Provides a background thread that continuously reads scans and exposes
    a thread-safe :meth:`get_scan` accessor. The most recent scan is kept
    in a single-slot buffer.

    Example:
        >>> lidar = LidarModule({"kind": "rplidar", "port": "/dev/rplidar", "baud": 115200})
        >>> lidar.start()
        >>> scan = lidar.get_scan()
        >>> obstacles = lidar.detect_obstacles(min_distance_m=0.5)
    """

    def __init__(self, config: dict) -> None:
        self.kind: LidarKind = LidarKind(config.get("kind", "rplidar"))
        self.port: str = config.get("port", "/dev/rplidar")
        self.baud: int = int(config.get("baud", 115200))
        self.udp_port: int = int(config.get("udp_port", 2368))
        self.udp_host: str = config.get("udp_host", "0.0.0.0")
        self.max_range_m: float = float(config.get("max_range_m", 12.0))
        self.min_range_m: float = float(config.get("min_range_m", 0.10))

        self._serial: Optional[object] = None
        self._udp_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[LidarScan] = None
        self._scan_id: int = 0
        self._info: dict = {}

        self._open()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _open(self) -> None:
        """Open the underlying transport (UART or UDP)."""
        if self.kind == LidarKind.RPLIDAR:
            if not _SERIAL_AVAILABLE:
                logger.warning("pyserial not installed — LiDAR running in stub mode")
                return
            logger.info("Opening RPLidar on {} @ {} baud", self.port, self.baud)
            self._serial = serial.Serial(self.port, self.baud, timeout=1.0)
            self._request_info()
            self._check_health()
        elif self.kind == LidarKind.VELODYNE:
            logger.info("Opening Velodyne UDP socket on {}:{}", self.udp_host, self.udp_port)
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._udp_sock.bind((self.udp_host, self.udp_port))
            self._udp_sock.settimeout(1.0)
        else:
            raise ValueError(f"Unknown LiDAR kind: {self.kind}")

    # ------------------------------------------------------------------
    # RPLidar commands
    # ------------------------------------------------------------------
    def _send_cmd(self, cmd: int, payload: bytes = b"") -> None:
        assert self._serial is not None
        self._serial.write(bytes([RPLIDAR_SYNC_BYTE1, cmd]) + payload)
        # Give the device a moment
        time.sleep(0.001)

    def _read_descriptor(self) -> Tuple[int, int]:
        """Read a response descriptor. Returns (data_size, data_type)."""
        assert self._serial is not None
        desc = self._serial.read(7)
        if len(desc) < 7 or desc[0] != 0xA5 or desc[1] != 0x5A:
            raise IOError("Bad response descriptor")
        size = desc[2] | (desc[3] << 8) | (desc[4] << 16)
        send_mode = desc[5]
        data_type = desc[6]
        return size, data_type

    def _request_info(self) -> None:
        """Query device info (model, firmware, serial)."""
        try:
            self._send_cmd(RPLIDAR_CMD_GET_INFO)
            size, _ = self._read_descriptor()
            data = self._serial.read(size)
            if len(data) >= 20:
                self._info = {
                    "model": data[0],
                    "firmware_minor": data[2],
                    "firmware_major": data[3],
                    "hardware_rev": data[4],
                    "serial": data[4:20].hex(),
                }
                logger.info("RPLidar info: {}", self._info)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read RPLidar info: {}", exc)

    def _check_health(self) -> None:
        """Verify the device is healthy before starting a scan."""
        try:
            self._send_cmd(RPLIDAR_CMD_GET_HEALTH)
            size, _ = self._read_descriptor()
            data = self._serial.read(size)
            if len(data) >= 3:
                status = data[0]
                error_code = (data[1] << 8) | data[2]
                if status == 0:
                    logger.info("RPLidar health: OK")
                elif status == 1:
                    logger.warning("RPLidar health: WARNING (code={})", error_code)
                elif status == 2:
                    logger.error("RPLidar health: ERROR (code={})", error_code)
                    raise RuntimeError("RPLidar reports error status")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Health check failed: {}", exc)

    # ------------------------------------------------------------------
    # Background scan loop
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background scan reader thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        if self.kind == LidarKind.RPLIDAR:
            self._send_cmd(RPLIDAR_CMD_SCAN)
        self._thread = threading.Thread(target=self._scan_loop,
                                        name="lidar-scan", daemon=True)
        self._thread.start()
        logger.info("Lidar scan thread started")

    def stop(self) -> None:
        """Stop scanning and join the reader thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.kind == LidarKind.RPLIDAR and self._serial is not None:
            try:
                self._send_cmd(RPLIDAR_CMD_STOP)
            except Exception:  # noqa: BLE001
                pass

    def _scan_loop(self) -> None:
        """Worker: continuously read scans."""
        while not self._stop_event.is_set():
            try:
                if self.kind == LidarKind.RPLIDAR:
                    scan = self._read_rplidar_scan()
                else:
                    scan = self._read_velodyne_scan()
                if scan is not None:
                    with self._lock:
                        self._latest = scan
            except Exception as exc:  # noqa: BLE001
                logger.exception("Lidar read error: {}", exc)
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # RPLidar scan parsing
    # ------------------------------------------------------------------
    def _read_rplidar_scan(self) -> Optional[LidarScan]:
        """Read one full revolution from an RPLidar.

        Each measurement packet is 5 bytes:
          byte 0   : quality (6 bits) | start_flag (1 bit) at bit 0
          bytes 1-2: angle_q6 (14 bits, little-endian, 1/64 deg units)
          bytes 3-4: distance_q2 (14 bits, little-endian, 1/4 mm units)
        """
        assert self._serial is not None
        angles: List[float] = []
        ranges: List[float] = []
        qualities: List[int] = []
        seen_sync = False

        while not self._stop_event.is_set():
            pkt = self._serial.read(5)
            if len(pkt) < 5:
                time.sleep(0.005)
                continue
            quality = pkt[0] >> 2
            start_flag = pkt[0] & 0x01
            angle_q6 = ((pkt[1] >> 1) | (pkt[2] << 7)) & 0x7FFF
            distance_q2 = ((pkt[3] >> 1) | (pkt[4] << 7)) & 0x7FFF

            if start_flag:
                if seen_sync:
                    # End of previous revolution; emit it
                    break
                seen_sync = True

            if distance_q2 == 0:
                continue

            angle_deg = angle_q6 / 64.0
            distance_m = distance_q2 / 4.0 / 1000.0
            if distance_m < self.min_range_m or distance_m > self.max_range_m:
                continue
            angles.append(math.radians(angle_deg))
            ranges.append(distance_m)
            qualities.append(quality)

        if not angles:
            return None

        self._scan_id += 1
        return LidarScan(
            angles=np.array(angles, dtype=np.float32),
            ranges=np.array(ranges, dtype=np.float32),
            intensities=np.array(qualities, dtype=np.float32),
            timestamp=time.monotonic(),
            scan_id=self._scan_id,
        )

    # ------------------------------------------------------------------
    # Velodyne VLP-16 scan parsing (simplified)
    # ------------------------------------------------------------------
    def _read_velodyne_scan(self) -> Optional[LidarScan]:
        """Read and parse one VLP-16 data packet.

        Each UDP packet is 1206 bytes:
          * 100 blocks of 12 bytes (16 azimuths × 2 layers each),
          * 4 bytes timestamp + 2 bytes factory.
        This is a simplified parser that extracts azimuths and ranges;
        see the Velodyne user manual for full details.
        """
        assert self._udp_sock is not None
        try:
            data, _ = self._udp_sock.recvfrom(2048)
        except socket.timeout:
            return None
        if len(data) < 1206:
            return None

        angles: List[float] = []
        ranges: List[float] = []
        intensities: List[int] = []

        # The 16 vertical angles of VLP-16 in degrees
        vertical_angles_deg = np.array(
            [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15],
            dtype=np.float32,
        )
        vertical_angles = np.deg2rad(vertical_angles_deg)

        # Each packet has 12 blocks × 32 firing pairs
        for block_idx in range(12):
            base = block_idx * 100
            if base + 100 > len(data):
                break
            flag = struct.unpack_from("<H", data, base)[0]
            if flag not in (0xEEFF, 0xDDFF):
                continue
            azimuth_raw = struct.unpack_from("<H", data, base + 2)[0]
            azimuth_deg = azimuth_raw / 100.0
            azimuth = math.radians(azimuth_deg)
            for chan in range(16):
                off = base + 4 + chan * 3
                dist_raw = struct.unpack_from("<H", data, off)[0]
                intensity = data[off + 2]
                dist_m = dist_raw * 0.002  # 2 mm units
                if dist_m < self.min_range_m or dist_m > self.max_range_m:
                    continue
                # Adjust horizontal angle for the channel's vertical angle
                horizontal_angle = azimuth  # simplification
                angles.append(horizontal_angle)
                ranges.append(dist_m)
                intensities.append(intensity)

        if not angles:
            return None
        self._scan_id += 1
        return LidarScan(
            angles=np.array(angles, dtype=np.float32),
            ranges=np.array(ranges, dtype=np.float32),
            intensities=np.array(intensities, dtype=np.float32),
            timestamp=time.monotonic(),
            scan_id=self._scan_id,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_scan(self) -> Optional[LidarScan]:
        """Return the most recent scan (or ``None``)."""
        with self._lock:
            return self._latest

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Return the latest point cloud as an (N, 2) array."""
        scan = self.get_scan()
        if scan is None:
            return None
        return scan.points[scan.valid_mask]

    # ------------------------------------------------------------------
    # Obstacle detection
    # ------------------------------------------------------------------
    def detect_obstacles(
        self,
        min_distance_m: float = 0.30,
        max_distance_m: float = 5.0,
        angular_window_deg: float = 60.0,
    ) -> List[dict]:
        """Detect obstacles within ``[min, max]`` meters and ±window degrees.

        Returns a list of dicts with keys ``angle``, ``distance``,
        ``size_deg`` describing each detected obstacle cluster.
        """
        scan = self.get_scan()
        if scan is None or len(scan.ranges) == 0:
            return []

        mask = (
            scan.valid_mask
            & (scan.ranges >= min_distance_m)
            & (scan.ranges <= max_distance_m)
        )
        if not mask.any():
            return []

        angles = scan.angles[mask]
        ranges = scan.ranges[mask]

        # Sort by angle for clustering
        order = np.argsort(angles)
        angles = angles[order]
        ranges = ranges[order]

        half_window = math.radians(angular_window_deg / 2.0)

        obstacles: List[dict] = []
        cluster_start = 0
        for i in range(1, len(angles)):
            # Cluster break: gap > 5 deg or large range jump
            angle_gap = abs(angles[i] - angles[i - 1])
            range_jump = abs(ranges[i] - ranges[i - 1]) > 0.30
            if angle_gap > math.radians(5.0) or range_jump:
                if i - cluster_start >= 2:
                    obstacles.append(self._summarize_cluster(
                        angles[cluster_start:i], ranges[cluster_start:i], half_window
                    ))
                cluster_start = i
        # Final cluster
        if len(angles) - cluster_start >= 2:
            obstacles.append(self._summarize_cluster(
                angles[cluster_start:], ranges[cluster_start:], half_window
            ))
        return obstacles

    @staticmethod
    def _summarize_cluster(angles: np.ndarray, ranges: np.ndarray,
                           half_window: float) -> dict:
        """Convert a cluster of returns into an obstacle summary."""
        # Keep only those within the forward window
        within = np.abs(angles) <= half_window
        if not within.any():
            within = np.ones_like(angles, dtype=bool)
        a = angles[within]
        r = ranges[within]
        return {
            "angle_deg": float(np.degrees(np.mean(a))),
            "distance_m": float(np.min(r)),
            "size_deg": float(np.degrees(np.ptp(a))) if len(a) > 1 else 0.0,
            "num_returns": int(len(a)),
        }

    # ------------------------------------------------------------------
    # Stats / cleanup
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "kind": self.kind.value,
            "scan_id": self._scan_id,
            "info": self._info,
        }

    def close(self) -> None:
        """Stop the reader thread and release the transport."""
        self.stop()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001
                pass
            self._serial = None
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:  # noqa: BLE001
                pass
            self._udp_sock = None
        logger.info("Lidar closed")
