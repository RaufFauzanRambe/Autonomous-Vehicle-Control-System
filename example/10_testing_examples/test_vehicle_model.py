"""Tests for the kinematic vehicle model (module 01).

Covered behavior: positive throttle raises speed, speed clamping at the
configured maximum, steering-induced yaw change, and a full
``VehicleState`` serialization roundtrip.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless rendering for plotting tests

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_basic_simulation"))

from vehicle.vehicle_model import VehicleModel
from vehicle.vehicle_state import VehicleState

DEFAULT_PARAMS = {
    "wheelbase_m": 2.7,
    "max_steering_rad": 0.6,
    "max_speed_mps": 33.0,
    "max_accel_mpss": 3.0,
    "max_decel_mpss": 8.0,
    "drag_coefficient": 0.02,
    "mass_kg": 1500.0,
}


def _run_steps(
    model: VehicleModel,
    state: VehicleState,
    throttle: float,
    steering: float,
    dt: float,
    n_steps: int,
) -> VehicleState:
    """Integrate the model forward for a fixed number of steps.

    Args:
        model: Vehicle model under test.
        state: Initial vehicle state.
        throttle: Throttle command in [0, 1] held for every step.
        steering: Steering command [rad] held for every step.
        dt: Integration time step [s].
        n_steps: Number of integration steps.

    Returns:
        The state after the last step.
    """
    for _ in range(n_steps):
        state = model.step(state, throttle, steering, dt)
    return state


def test_positive_throttle_increases_speed() -> None:
    """A half-open throttle from a rolling start must raise the speed."""
    model = VehicleModel(DEFAULT_PARAMS)
    state = VehicleState(speed=1.0)
    final = _run_steps(model, state, throttle=0.5, steering=0.0, dt=0.1, n_steps=10)
    assert final.speed > 1.0


def test_speed_clamped_to_max() -> None:
    """Full throttle for a long time must never exceed the configured maximum."""
    model = VehicleModel(DEFAULT_PARAMS)
    state = VehicleState(speed=0.0)
    final = _run_steps(model, state, throttle=1.0, steering=0.0, dt=0.1, n_steps=2000)
    assert final.speed <= DEFAULT_PARAMS["max_speed_mps"] + 1e-6


def test_steering_changes_yaw() -> None:
    """Steering while rolling must produce a positive (CCW) yaw change."""
    model = VehicleModel(DEFAULT_PARAMS)
    state = VehicleState(speed=5.0)
    final = _run_steps(model, state, throttle=0.0, steering=0.2, dt=0.1, n_steps=20)
    assert final.yaw > 0.1


def test_state_roundtrip() -> None:
    """to_dict -> from_dict must preserve every vehicle-state field."""
    state = VehicleState(
        timestamp=1.23,
        x=3.4,
        y=-5.6,
        yaw=0.78,
        speed=9.1,
        acceleration=-0.4,
        steering_angle=0.12,
    )
    restored = VehicleState.from_dict(state.to_dict())
    assert restored.timestamp == pytest.approx(state.timestamp)
    assert restored.x == pytest.approx(state.x)
    assert restored.y == pytest.approx(state.y)
    assert restored.yaw == pytest.approx(state.yaw)
    assert restored.speed == pytest.approx(state.speed)
    assert restored.acceleration == pytest.approx(state.acceleration)
    assert restored.steering_angle == pytest.approx(state.steering_angle)
