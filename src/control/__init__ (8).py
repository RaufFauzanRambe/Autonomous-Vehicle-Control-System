"""
Autonomous Vehicle Control System – Control Module.

Provides controllers for longitudinal and lateral vehicle control:

- **PIDController** – Versatile PID with anti-windup and derivative filtering.
- **MPCController** – Model Predictive Control with bicycle model and QP.
- **StanleyController** – Front-axle lateral controller with CTE correction.
- **PurePursuitController** – Classic lookahead lateral path follower.

Usage::

    from control import PIDController, MPCController, StanleyController, PurePursuitController
"""

from .pid_controller import PIDController, PIDGains
from .mpc_controller import MPCController, MPCWeights, MPCConstraints
from .stanley_controller import StanleyController, StanleyParams
from .pure_pursuit_controller import PurePursuitController, PurePursuitParams
from .vehicle_model import BicycleModel, VehicleParameters

__all__ = [
    "PIDController",
    "PIDGains",
    "MPCController",
    "MPCWeights",
    "MPCConstraints",
    "StanleyController",
    "StanleyParams",
    "PurePursuitController",
    "PurePursuitParams",
    "BicycleModel",
    "VehicleParameters",
]
