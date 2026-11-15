"""Tests for the synthetic sensor classes (module 02).

Covered behavior: the common reading envelope (keys and sensor type),
seeded determinism, lidar point-cloud output shape, and a 6-sigma bound
on GPS position noise.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless rendering for plotting tests

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_basic_simulation"))
sys.path.insert(0, str(ROOT / "02_sensor_simulation"))

from sensors.camera_sensor import CameraSensor
from sensors.gps_sensor import GPSSensor
from sensors.imu_sensor import IMUSensor
from sensors.lidar_sensor import LidarSensor
from sensors.radar_sensor import RadarSensor
from vehicle.vehicle_state import VehicleState

READING_ENVELOPE_KEYS = {"sensor_type", "timestamp", "frame_id", "data"}

MINIMAL_CONFIGS: dict[type, dict] = {
    CameraSensor: {
        "image_width_px": 640,
        "image_height_px": 480,
        "fov_deg": 90.0,
        "max_detection_range_m": 100.0,
        "detection_noise_std_px": 2.0,
    },
    LidarSensor: {
        "channels": 16,
        "points_per_scan": 120,
        "min_range_m": 1.0,
        "max_range_m": 100.0,
        "range_noise_std_m": 0.02,
    },
    RadarSensor: {
        "max_range_m": 150.0,
        "fov_deg": 70.0,
        "range_noise_std_m": 0.5,
    },
    GPSSensor: {
        "position_noise_std_m": 1.5,
        "frame_id": "gps_link",
    },
    IMUSensor: {
        "accel_noise_std_mpss": 0.05,
        "gyro_noise_std_radps": 0.01,
        "bias_ax_mpss": 0.1,
        "frame_id": "imu_link",
    },
}

EXPECTED_SENSOR_TYPES: dict[type, str] = {
    CameraSensor: "camera",
    LidarSensor: "lidar",
    RadarSensor: "radar",
    GPSSensor: "gps",
    IMUSensor: "imu",
}


@pytest.mark.parametrize(
    "sensor_cls",
    [CameraSensor, LidarSensor, RadarSensor, GPSSensor, IMUSensor],
)
def test_reading_envelope_keys(sensor_cls: type) -> None:
    """Every sensor returns the shared envelope and its own sensor type."""
    sensor = sensor_cls(MINIMAL_CONFIGS[sensor_cls], seed=42)
    sensor.update(VehicleState(x=10.0, speed=5.0), 0.1)
    reading = sensor.get_reading()
    assert set(reading.keys()) == READING_ENVELOPE_KEYS
    assert reading["sensor_type"] == EXPECTED_SENSOR_TYPES[sensor_cls]


def test_deterministic_with_seed() -> None:
    """Two GPS sensors with identical seeds produce identical readings."""
    state_sequence = [VehicleState(x=0.5 * i, y=0.0, speed=5.0) for i in range(5)]
    readings: list[dict] = []
    for _ in range(2):
        sensor = GPSSensor({"position_noise_std_m": 1.5}, seed=42)
        for state in state_sequence:
            sensor.update(state, 0.1)
        readings.append(sensor.get_reading())
    assert readings[0] == readings[1]


def test_lidar_points_shape() -> None:
    """A lidar scan contains exactly points_per_scan 2-D points."""
    sensor = LidarSensor(
        {
            "points_per_scan": 120,
            "max_range_m": 100.0,
            "min_range_m": 1.0,
            "range_noise_std_m": 0.02,
        },
        seed=42,
    )
    sensor.update(VehicleState(x=10.0, speed=5.0), 0.1)
    points = sensor.get_reading()["data"]["points_xy_m"]
    assert len(points) == 120
    assert all(len(point) == 2 for point in points)


def test_gps_noise_bounded() -> None:
    """GPS measurement error stays inside a 6-sigma envelope (9.0 m)."""
    sigma_m = 1.5
    bound_m = 6.0 * sigma_m
    sensor = GPSSensor({"position_noise_std_m": sigma_m}, seed=42)
    for i in range(50):
        truth_x = 0.4 * i
        sensor.update(VehicleState(x=truth_x, y=0.0, speed=4.0), 0.1)
        reading = sensor.get_reading()
        measured_x = reading["data"]["x_m"]
        assert abs(measured_x - truth_x) < bound_m
