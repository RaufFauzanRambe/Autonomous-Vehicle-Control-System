"""
GPS Processor Module for Autonomous Vehicle Localization

Handles GPS data parsing, coordinate transformation (LLA to local ENU),
and GPS signal quality assessment for the localization pipeline.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class GPSFixType(Enum):
    """GPS fix quality indicators."""
    NO_FIX = 0
    FIX_2D = 1
    FIX_3D = 2
    DGPS = 3
    RTK_FIXED = 4
    RTK_FLOAT = 5


@dataclass
class GPSReading:
    """Raw GPS measurement."""
    latitude: float       # Degrees
    longitude: float      # Degrees
    altitude: float       # Meters above WGS84 ellipsoid
    fix_type: GPSFixType
    num_satellites: int
    hdop: float           # Horizontal dilution of precision
    vdop: float           # Vertical dilution of precision
    speed: float          # Ground speed in m/s
    heading: float        # Track angle in degrees
    timestamp: float
    covariance: np.ndarray = None  # 3x3 position covariance

    @property
    def is_valid(self) -> bool:
        """Check if GPS fix is valid for localization."""
        return self.fix_type != GPSFixType.NO_FIX and self.num_satellites >= 4

    @property
    def accuracy(self) -> float:
        """Estimated position accuracy in meters."""
        if self.fix_type == GPSFixType.RTK_FIXED:
            return 0.02  # 2cm RTK accuracy
        elif self.fix_type == GPSFixType.RTK_FLOAT:
            return 0.5   # 50cm RTK float
        elif self.fix_type == GPSFixType.DGPS:
            return 1.0   # 1m DGPS
        elif self.fix_type == GPSFixType.FIX_3D:
            return 3.0 * self.hdop  # ~3-10m with HDOP
        else:
            return 10.0  # 10m+ for 2D fix


class GPSProcessor:
    """
    GPS data processor for autonomous vehicle localization.

    Converts raw GPS readings (latitude, longitude, altitude) to
    local ENU (East-North-Up) coordinates relative to a reference
    origin, and provides signal quality assessment.
    """

    # WGS84 ellipsoid parameters
    WGS84_A = 6378137.0           # Semi-major axis (m)
    WGS84_B = 6356752.314245     # Semi-minor axis (m)
    WGS84_E2 = 0.00669437999014  # First eccentricity squared

    def __init__(
        self,
        reference_latitude: Optional[float] = None,
        reference_longitude: Optional[float] = None,
        reference_altitude: Optional[float] = None,
    ):
        """
        Initialize GPS processor with reference origin.

        Args:
            reference_latitude: Reference latitude in degrees. First reading used if None.
            reference_longitude: Reference longitude in degrees.
            reference_altitude: Reference altitude in meters.
        """
        self._ref_lla = None
        if reference_latitude is not None and reference_longitude is not None:
            self._ref_lla = np.array([
                np.radians(reference_latitude),
                np.radians(reference_longitude),
                reference_altitude or 0.0,
            ])

        self._ref_set = self._ref_lla is not None

    def set_reference(self, latitude: float, longitude: float, altitude: float = 0.0) -> None:
        """
        Set the reference origin for LLA-to-ENU conversion.

        Args:
            latitude: Reference latitude in degrees.
            longitude: Reference longitude in degrees.
            altitude: Reference altitude in meters.
        """
        self._ref_lla = np.array([np.radians(latitude), np.radians(longitude), altitude])
        self._ref_set = True

    def lla_to_enu(self, latitude: float, longitude: float, altitude: float) -> np.ndarray:
        """
        Convert LLA (latitude, longitude, altitude) to local ENU coordinates.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
            altitude: Altitude in meters.

        Returns:
            ENU coordinates [east, north, up] in meters relative to reference.
        """
        if not self._ref_set:
            self.set_reference(latitude, longitude, altitude)
            return np.array([0.0, 0.0, 0.0])

        # Convert to radians
        lat = np.radians(latitude)
        lon = np.radians(longitude)

        # Convert LLA to ECEF
        ecef = self._lla_to_ecef(lat, lon, altitude)
        ref_ecef = self._lla_to_ecef(self._ref_lla[0], self._ref_lla[1], self._ref_lla[2])

        # ECEF to ENU transformation
        dx = ecef - ref_ecef
        R = self._ecef_to_enu_rotation(self._ref_lla[0], self._ref_lla[1])
        enu = R @ dx

        return enu

    def process(self, reading: GPSReading) -> Dict:
        """
        Process a GPS reading into localization-compatible format.

        Args:
            reading: Raw GPS reading.

        Returns:
            Dictionary with ENU position, covariance, and quality metrics.
        """
        enu = self.lla_to_enu(reading.latitude, reading.longitude, reading.altitude)

        # Build position covariance from HDOP/VDOP
        if reading.covariance is not None:
            cov = reading.covariance
        else:
            base_var = reading.accuracy ** 2
            cov = np.diag([
                base_var * (reading.hdop ** 2),
                base_var * (reading.hdop ** 2),
                base_var * (reading.vdop ** 2),
            ])

        return {
            'position_enu': enu,
            'covariance': cov,
            'fix_type': reading.fix_type,
            'num_satellites': reading.num_satellites,
            'accuracy': reading.accuracy,
            'speed': reading.speed,
            'heading': reading.heading,
            'timestamp': reading.timestamp,
            'is_valid': reading.is_valid,
        }

    def _lla_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert geodetic LLA to Earth-Centered Earth-Fixed (ECEF)."""
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * sin_lat**2)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - self.WGS84_E2) + alt) * sin_lat

        return np.array([x, y, z])

    def _ecef_to_enu_rotation(self, ref_lat: float, ref_lon: float) -> np.ndarray:
        """Compute rotation matrix from ECEF to ENU at reference point."""
        sin_lat = np.sin(ref_lat)
        cos_lat = np.cos(ref_lat)
        sin_lon = np.sin(ref_lon)
        cos_lon = np.cos(ref_lon)

        R = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ])
        return R
