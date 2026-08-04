"""
File:        python/main.py
Brief:       Entry point for the Raspberry Pi platform of the Autonomous
             Vehicle Control System. Orchestrates all subsystems (camera,
             lidar, GPS, IMU, ultrasonic, motor, steering, telemetry),
             handles CLI arguments, signal-based shutdown, and runs the
             main control loop at a configurable rate.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from utils import Config, clamp
from camera import CameraModule
from lidar import LidarModule
from gps import GpsModule
from imu import ImuModule
from ultrasonic import UltrasonicModule
from motor_driver import MotorDriver
from steering import Steering
from telemetry import Telemetry
from mqtt_client import MqttClient


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_RATE_HZ: int = 30
DEFAULT_CONFIG_PATH: str = "config/av-01.yaml"
SHUTDOWN_TIMEOUT_S: float = 5.0


# ----------------------------------------------------------------------
# Application state container
# ----------------------------------------------------------------------
@dataclass(slots=True)
class AppState:
    """Holds runtime flags shared across threads."""

    running: bool = True
    estop: bool = False
    last_heartbeat: float = field(default_factory=time.monotonic)
    loop_count: int = 0

    def request_shutdown(self) -> None:
        """Signal all threads to stop."""
        self.running = False
        logger.info("Shutdown requested")


# ----------------------------------------------------------------------
# Vehicle controller
# ----------------------------------------------------------------------
class VehicleController:
    """Top-level orchestrator.

    Owns every hardware-facing module and runs the main control loop.
    Each module is responsible for its own threading where needed; this
    class only calls the synchronous `update()` and `read()` methods.
    """

    def __init__(self, config: Config, args: argparse.Namespace) -> None:
        self.config: Config = config
        self.args: argparse.Namespace = args
        self.state: AppState = AppState()
        self.rate_hz: int = args.rate or config.get("control.rate_hz", DEFAULT_RATE_HZ)
        self.period: float = 1.0 / self.rate_hz

        # Subsystem handles (lazy-init so unit tests can stub them out).
        self.camera: Optional[CameraModule] = None
        self.lidar: Optional[LidarModule] = None
        self.gps: Optional[GpsModule] = None
        self.imu: Optional[ImuModule] = None
        self.ultrasonic: Optional[UltrasonicModule] = None
        self.motor: Optional[MotorDriver] = None
        self.steering: Optional[Steering] = None
        self.mqtt: Optional[MqttClient] = None
        self.telemetry: Optional[Telemetry] = None

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Instantiate and configure all subsystems."""
        logger.info("Initializing subsystems (vehicle_id={})",
                    self.config.get("vehicle.id", "av-00"))

        try:
            self.camera = CameraModule(self.config.section("camera"))
            self.lidar = LidarModule(self.config.section("lidar"))
            self.gps = GpsModule(self.config.section("gps"))
            self.imu = ImuModule(self.config.section("imu"))
            self.ultrasonic = UltrasonicModule(self.config.section("ultrasonic"))
            self.motor = MotorDriver(self.config.section("motor"))
            self.steering = Steering(self.config.section("steering"))

            self.mqtt = MqttClient(self.config.section("mqtt"))
            self.mqtt.connect()
            self.mqtt.subscribe("vehicle/+/estop", self._on_estop)
            self.mqtt.subscribe("vehicle/+/cmd/velocity", self._on_cmd_velocity)

            self.telemetry = Telemetry(self.config.section("telemetry"),
                                       mqtt_client=self.mqtt)
            logger.info("All subsystems initialized successfully")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initialization failed: {}", exc)
            raise

    def run(self) -> int:
        """Run the main loop until shutdown is requested."""
        self._install_signal_handlers()
        logger.info("Entering main loop at {} Hz", self.rate_hz)

        try:
            while self.state.running:
                loop_start = time.monotonic()
                self.state.loop_count += 1

                try:
                    self._tick()
                except KeyboardInterrupt:
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Loop iteration failed: {}", exc)
                    # Don't crash the whole vehicle for a single bad frame.
                    if self.state.loop_count % 100 == 0:
                        self._safe_estop()

                # Maintain loop rate
                elapsed = time.monotonic() - loop_start
                sleep_for = self.period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    logger.debug("Loop overrun: {:.1f} ms",
                                 (elapsed - self.period) * 1000.0)
        finally:
            self.shutdown()

        logger.info("Main loop exited after {} iterations",
                    self.state.loop_count)
        return 0

    # ------------------------------------------------------------------
    # Per-iteration work
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        """Run a single control loop iteration."""
        # 1. Read sensors
        imu_data = self.imu.read() if self.imu else None
        gps_data = self.gps.read() if self.gps else None
        ultra_data = self.ultrasonic.read_all() if self.ultrasonic else {}
        lidar_scan = self.lidar.get_scan() if self.lidar else None
        # Camera capture runs in its own thread; just grab the latest frame.
        frame = self.camera.get_latest_frame() if self.camera else None

        # 2. Decide motion setpoints
        cmd_linear, cmd_angular = self._compute_motion(
            imu_data, gps_data, ultra_data, lidar_scan
        )

        # 3. Apply motion (respect E-stop)
        if not self.state.estop:
            self.motor.set_speed(cmd_linear)
            self.steering.set_angle(cmd_angular)
        else:
            self.motor.set_speed(0.0)
            self.steering.set_angle(0.0)

        # 4. Publish telemetry
        self.telemetry.publish({
            "vehicle_id": self.config.get("vehicle.id", "av-00"),
            "loop_count": self.state.loop_count,
            "imu": imu_data.__dict__ if imu_data else None,
            "gps": gps_data.__dict__ if gps_data else None,
            "ultrasonic": ultra_data,
            "cmd_linear": cmd_linear,
            "cmd_angular": cmd_angular,
            "estop": self.state.estop,
            "frame_id": getattr(frame, "frame_id", None),
        })

        self.state.last_heartbeat = time.monotonic()

    # ------------------------------------------------------------------
    # Motion logic (placeholder for higher-level controller)
    # ------------------------------------------------------------------
    def _compute_motion(
        self,
        imu: Any,
        gps: Any,
        ultrasonic: Dict[str, float],
        lidar: Any,
    ) -> tuple[float, float]:
        """Compute desired linear velocity (m/s) and steering angle (deg).

        A simple obstacle-avoidance behaviour:
          * If any ultrasonic sensor reports < 30 cm, brake.
          * Otherwise, maintain a cruise speed of 0.4 m/s.
          * Steer away from the closest obstacle.
        """
        if self.state.estop:
            return 0.0, 0.0

        cruise_speed: float = self.config.get("control.cruise_speed", 0.4)
        safe_distance_m: float = self.config.get("control.safe_distance_m", 0.30)

        min_distance: float = min(ultrasonic.values()) if ultrasonic else float("inf")
        if min_distance < safe_distance_m:
            logger.warning("Obstacle at {:.2f} m — braking", min_distance)
            return 0.0, 0.0

        # Steer based on left/right imbalance (very crude wall-follow)
        left = ultrasonic.get("left", 1.0)
        right = ultrasonic.get("right", 1.0)
        # +angle = turn right, -angle = turn left
        angle = clamp((right - left) * 30.0, -25.0, 25.0)
        return cruise_speed, angle

    # ------------------------------------------------------------------
    # E-stop
    # ------------------------------------------------------------------
    def _safe_estop(self) -> None:
        """Engage software E-stop and hold actuators at neutral."""
        if not self.state.estop:
            logger.error("ENGAGING E-STOP")
            self.state.estop = True
            if self.motor:
                self.motor.set_speed(0.0)
            if self.steering:
                self.steering.set_angle(0.0)
            if self.mqtt:
                self.mqtt.publish("vehicle/av-00/estop/engaged", "true", qos=1)

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------
    def _on_estop(self, topic: str, payload: bytes) -> None:
        try:
            engage = payload.decode().strip().lower() in ("1", "true", "on")
        except Exception:  # noqa: BLE001
            engage = True
        logger.warning("E-stop command received: engage={}", engage)
        self.state.estop = engage

    def _on_cmd_velocity(self, topic: str, payload: bytes) -> None:
        """External velocity command (e.g. from a joystick)."""
        import json
        try:
            msg = json.loads(payload.decode())
            linear = float(msg.get("linear", 0.0))
            angular = float(msg.get("angular", 0.0))
            if self.motor:
                self.motor.set_speed(linear)
            if self.steering:
                self.steering.set_angle(angular)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid /cmd/velocity payload: {}", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop subsystems in reverse-init order."""
        logger.info("Shutting down subsystems")
        self.state.request_shutdown()

        if self.motor:
            self.motor.set_speed(0.0)
            self.motor.close()
        if self.steering:
            self.steering.set_angle(0.0)
            self.steering.close()
        if self.camera:
            self.camera.close()
        if self.lidar:
            self.lidar.close()
        if self.gps:
            self.gps.close()
        if self.imu:
            self.imu.close()
        if self.ultrasonic:
            self.ultrasonic.close()
        if self.telemetry:
            self.telemetry.flush()
        if self.mqtt:
            self.mqtt.disconnect()
        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.warning("Received signal {} — requesting shutdown", signum)
        self.state.request_shutdown()


# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autovehicle",
        description="Autonomous Vehicle Control System — Raspberry Pi main loop",
    )
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to YAML config file")
    parser.add_argument("-r", "--rate", type=int, default=None,
                        help="Loop rate (Hz). Overrides config.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v, -vv)")
    parser.add_argument("--log-file", default=None,
                        help="Optional path to a JSON log file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Initialize then exit (for smoke testing)")
    parser.add_argument("--vehicle-id", default=None,
                        help="Override vehicle_id from config")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
def configure_logging(args: argparse.Namespace) -> None:
    level = "WARNING"
    if args.verbose == 1:
        level = "INFO"
    elif args.verbose >= 2:
        level = "DEBUG"

    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True,
               format=("<green>{time:HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
                       "<level>{message}</level>"))
    if args.log_file:
        logger.add(args.log_file, level=level, rotation="10 MB",
                   retention="7 days", compression="gz", serialize=True)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args)

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config file not found: {}", config_path)
        return 2
    config = Config.from_file(config_path)
    if args.vehicle_id:
        config.set("vehicle.id", args.vehicle_id)
    if args.rate:
        config.set("control.rate_hz", args.rate)

    controller = VehicleController(config, args)
    try:
        controller.initialize()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize: {}", exc)
        return 3

    if args.dry_run:
        logger.info("Dry-run — initialization succeeded, exiting")
        controller.shutdown()
        return 0

    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
