"""
File:        python/gps.py
Brief:       GpsModule — gpsd-py3 client for the Adafruit Ultimate GPS
             (or any NMEA GPS exposing a fix to gpsd). Provides lat/lon,
             speed, heading, fix quality, waypoint navigation, Haversine
             distance and bearing calculations.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

from loguru import logger

try:
    import gps as gpsd_py3
    _GPSD_AVAILABLE = True
except ImportError:  # pragma: no cover - dev workstation
    gpsd_py3 = None  # type: ignore
    _GPSD_AVAILABLE = False


# ----------------------------------------------------------------------
# Enums and dataclasses
# ----------------------------------------------------------------------
class FixQuality(IntEnum):
    """NMEA fix quality values."""

    INVALID = 0
    GPS = 1
    DGPS = 2
    PPS = 3
    RTK_FIXED = 4
    RTK_FLOAT = 5
    DEAD_RECKONING = 6
    MANUAL = 7
    SIMULATOR = 8


@dataclass(slots=True)
class GpsFix:
    """A single GPS fix sample."""

    latitude: float
    longitude: float
    altitude_m: float
    speed_mps: float          # m/s
    heading_deg: float        # 0..360, course over ground
    fix_quality: int          # FixQuality value
    num_satellites: int
    hdop: float               # horizontal dilution of precision
    timestamp: float          # monotonic seconds
    utc_time: Optional[str] = None

    @property
    def has_fix(self) -> bool:
        return self.fix_quality >= FixQuality.GPS

    @property
    def is_rtk(self) -> bool:
        return self.fix_quality in (FixQuality.RTK_FIXED, FixQuality.RTK_FLOAT)


@dataclass(slots=True)
class Waypoint:
    """A navigation waypoint."""

    latitude: float
    longitude: float
    name: str = ""
    radius_m: float = 2.0     # arrival radius

    def __post_init__(self) -> None:
        # Normalize lat/lon
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"Invalid longitude: {self.longitude}")


# ----------------------------------------------------------------------
# Earth model constants
# ----------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_371_000.0  # mean radius


# ----------------------------------------------------------------------
# GpsModule
# ----------------------------------------------------------------------
class GpsModule:
    """gpsd-backed GPS reader with waypoint navigation helpers.

    Spawns a background thread that streams fixes from gpsd. The most
    recent fix is kept in a single-slot buffer. Waypoint navigation
    uses the Haversine formula and great-circle bearing.
    """

    DEFAULT_HOST: str = "127.0.0.1"
    DEFAULT_PORT: str = "2947"

    def __init__(self, config: dict) -> None:
        self.host: str = config.get("host", self.DEFAULT_HOST)
        self.port: str = config.get("port", self.DEFAULT_PORT)
        self.poll_interval_s: float = float(config.get("poll_interval_s", 0.2))
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx: int = 0

        self._session = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._latest: Optional[GpsFix] = None
        self._fix_count: int = 0

        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Open a gpsd session."""
        if not _GPSD_AVAILABLE:
            logger.warning("gpsd-py3 not installed — running in stub mode")
            return
        try:
            logger.info("Connecting to gpsd at {}:{}", self.host, self.port)
            self._session = gpsd_py3.gps(
                mode=gpsd_py3.WATCH_ENABLE | gpsd_py3.WATCH_NEWSTYLE,
                host=self.host, port=self.port,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to connect to gpsd: {}", exc)
            self._session = None

    def reconnect(self) -> bool:
        """Reconnect to gpsd (e.g. after the daemon restarted)."""
        self._connect()
        return self._session is not None

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background fix-reader thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop,
                                        name="gps-reader", daemon=True)
        self._thread.start()
        logger.info("GPS reader thread started")

    def stop(self) -> None:
        """Stop the reader thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _reader_loop(self) -> None:
        """Worker: read fixes from gpsd in WATCH mode."""
        while not self._stop_event.is_set():
            try:
                if self._session is None:
                    self._connect()
                    if self._session is None:
                        time.sleep(1.0)
                        continue
                # blocking read with timeout
                if not self._session.waiting(timeout=self.poll_interval_s):
                    continue
                report = self._session.next()
                fix = self._parse_report(report)
                if fix is not None:
                    with self._lock:
                        self._latest = fix
                    self._fix_count += 1
            except StopIteration:
                logger.warning("gpsd stream closed; reconnecting")
                self._session = None
            except Exception as exc:  # noqa: BLE001
                logger.exception("GPS read error: {}", exc)
                time.sleep(0.5)

    @staticmethod
    def _parse_report(report) -> Optional[GpsFix]:
        """Convert a gpsd report dict into a :class:`GpsFix`."""
        # TPV reports contain position/velocity
        cls = getattr(report, "class", None)
        if cls != "TPV":
            return None
        mode = getattr(report, "mode", 0)
        if mode < 2:  # no fix
            return GpsFix(
                latitude=0.0, longitude=0.0, altitude_m=0.0,
                speed_mps=0.0, heading_deg=0.0,
                fix_quality=0, num_satellites=0, hdop=99.9,
                timestamp=time.monotonic(),
                utc_time=getattr(report, "time", None),
            )

        lat = float(getattr(report, "lat", 0.0))
        lon = float(getattr(report, "lon", 0.0))
        alt = float(getattr(report, "alt", 0.0) or 0.0)
        speed = float(getattr(report, "speed", 0.0) or 0.0)        # m/s
        track = float(getattr(report, "track", 0.0) or 0.0)        # deg
        eps = float(getattr(report, "eps", 99.9) or 99.9)          # error estimate
        return GpsFix(
            latitude=lat, longitude=lon, altitude_m=alt,
            speed_mps=speed, heading_deg=track,
            fix_quality=1 if mode == 2 else 2,
            num_satellites=0,  # SKY report has this; ignored here
            hdop=eps,
            timestamp=time.monotonic(),
            utc_time=getattr(report, "time", None),
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def read(self) -> Optional[GpsFix]:
        """Return the most recent fix (alias of :meth:`get_fix`)."""
        with self._lock:
            return self._latest

    def get_fix(self) -> Optional[GpsFix]:
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # Waypoint navigation
    # ------------------------------------------------------------------
    def load_waypoints(self, waypoints: List[Waypoint]) -> None:
        """Replace the waypoint list and reset the index."""
        self.waypoints = list(waypoints)
        self.current_waypoint_idx = 0
        logger.info("Loaded {} waypoints (start: '{}')",
                    len(self.waypoints),
                    self.waypoints[0].name if self.waypoints else "—")

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Return the current target waypoint or ``None`` if done."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_idx]

    def advance_waypoint(self) -> Optional[Waypoint]:
        """Move to the next waypoint."""
        if self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1
            logger.info("Advancing to waypoint '{}'",
                        self.waypoints[self.current_waypoint_idx].name)
            return self.waypoints[self.current_waypoint_idx]
        self.current_waypoint_idx = len(self.waypoints)  # past end
        logger.info("Reached final waypoint")
        return None

    def distance_to_waypoint(self, wp: Optional[Waypoint] = None) -> float:
        """Great-circle distance from current fix to the given waypoint."""
        fix = self.get_fix()
        target = wp or self.get_current_waypoint()
        if fix is None or target is None or not fix.has_fix:
            return float("inf")
        return haversine_m(fix.latitude, fix.longitude,
                           target.latitude, target.longitude)

    def bearing_to_waypoint(self, wp: Optional[Waypoint] = None) -> float:
        """Initial great-circle bearing (deg, 0=N, 90=E) to a waypoint."""
        fix = self.get_fix()
        target = wp or self.get_current_waypoint()
        if fix is None or target is None or not fix.has_fix:
            return 0.0
        return bearing_to_waypoint_deg(fix.latitude, fix.longitude,
                                       target.latitude, target.longitude)

    def has_arrived(self, wp: Optional[Waypoint] = None) -> bool:
        """True if within ``wp.radius_m`` of the waypoint."""
        target = wp or self.get_current_waypoint()
        if target is None:
            return False
        return self.distance_to_waypoint(target) <= target.radius_m

    def cross_track_error_m(self, wp_a: Waypoint, wp_b: Waypoint) -> float:
        """Signed cross-track distance (m) from current pos to A→B leg.

        Positive = vehicle is to the right of the leg.
        """
        fix = self.get_fix()
        if fix is None or not fix.has_fix:
            return 0.0
        return cross_track_m(fix.latitude, fix.longitude,
                             wp_a.latitude, wp_a.longitude,
                             wp_b.latitude, wp_b.longitude)

    # ------------------------------------------------------------------
    # Stats / cleanup
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "fix_count": self._fix_count,
            "current_waypoint": (self.get_current_waypoint().__dict__
                                 if self.get_current_waypoint() else None),
            "waypoints_total": len(self.waypoints),
            "waypoints_remaining": max(0, len(self.waypoints)
                                       - self.current_waypoint_idx),
        }

    def close(self) -> None:
        self.stop()
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None
        logger.info("GPS closed")


# ----------------------------------------------------------------------
# Geodesy helpers
# ----------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def bearing_to_waypoint_deg(lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from (lat1, lon1) to (lat2, lon2).

    Returns degrees 0..360 (0 = North, 90 = East).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360.0) % 360.0


def cross_track_m(lat_p: float, lon_p: float,
                  lat_a: float, lon_a: float,
                  lat_b: float, lon_b: float) -> float:
    """Signed cross-track distance from point P to the A→B great circle.

    Positive = P is to the right of A→B.
    """
    d13 = haversine_m(lat_a, lon_a, lat_p, lon_p) / EARTH_RADIUS_M
    theta13 = math.radians(bearing_to_waypoint_deg(lat_a, lon_a, lat_p, lon_p))
    theta12 = math.radians(bearing_to_waypoint_deg(lat_a, lon_a, lat_b, lon_b))
    return math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * EARTH_RADIUS_M
