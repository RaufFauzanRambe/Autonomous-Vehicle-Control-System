"""
Lane Departure Warning Module
=============================

Implements a Lane Departure Warning (LDW) system that computes
lateral offset from lane center, estimates time-to-lane-crossing,
and triggers visual/haptic/audible warnings.

Classes:
    WarningLevel - Enum for warning severity
    DepartureEvent - Dataclass for a departure warning event
    LaneDepartureWarning - Main LDW system

Typical usage:
    >>> ldw = LaneDepartureWarning(config)
    >>> warning = ldw.check_departure(center_offset, velocity, curvature)
    >>> if warning.level != WarningLevel.NONE:
    ...     ldw.trigger_warning(warning)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WarningLevel(Enum):
    """Warning severity levels."""
    NONE = "none"
    SOFT = "soft"         # Approaching lane boundary
    WARNING = "warning"    # Likely to depart
    CRITICAL = "critical"  # Imminent departure


@dataclass
class DepartureEvent:
    """Represents a lane departure warning event.

    Attributes:
        level: Warning severity level.
        offset: Current lateral offset from lane center (meters).
        direction: Direction of departure ("left" or "right").
        tlc: Time-to-lane-crossing in seconds (None if not computable).
        speed: Vehicle speed at time of warning (m/s).
        curvature: Road curvature radius at time of warning (meters).
        timestamp: Event timestamp.
        confidence: Confidence in the warning assessment.
    """
    level: WarningLevel = WarningLevel.NONE
    offset: float = 0.0
    direction: str = "none"
    tlc: Optional[float] = None
    speed: float = 0.0
    curvature: float = 0.0
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the departure event."""
        return {
            "level": self.level.value,
            "offset": round(self.offset, 4),
            "direction": self.direction,
            "tlc": round(self.tlc, 2) if self.tlc is not None else None,
            "speed": round(self.speed, 2),
            "curvature": round(self.curvature, 2),
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 3),
        }


class LaneDepartureWarning:
    """Lane Departure Warning (LDW) system.

    Monitors the vehicle's position within the lane and issues
    warnings when the vehicle approaches or crosses lane boundaries.
    Uses lateral offset, time-to-lane-crossing (TLC), and hysteresis
    for robust, non-flickering warnings.

    Attributes:
        config: Configuration dictionary for LDW parameters.
        active: Whether the LDW system is enabled.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the LDW system.

        Args:
            config: Configuration dictionary with lane_departure_warning settings.
        """
        self.config = config
        self.active = config.get("enabled", True)

        # Warning thresholds (meters from lane center)
        self._warning_offset = config.get("warning_threshold", 0.5)
        self._critical_offset = config.get("critical_threshold", 0.8)

        # Time-to-lane-crossing thresholds (seconds)
        self._tlc_warning = config.get("tlc_warning", 2.0)
        self._tlc_critical = config.get("tlc_critical", 0.5)

        # Alert configuration
        alert_cfg = config.get("alert_types", {})
        self._visual_alert = alert_cfg.get("visual", True)
        self._haptic_alert = alert_cfg.get("haptic", True)
        self._audible_alert = alert_cfg.get("audible", True)

        # Cooldown
        self._cooldown_ms = config.get("warning_cooldown_ms", 1000)

        # Vehicle parameters
        vehicle_cfg = config.get("vehicle", {})
        self._vehicle_width = vehicle_cfg.get("width", 1.8)
        self._lane_width = vehicle_cfg.get("lane_width", 3.7)

        # Hysteresis
        hyst_cfg = config.get("hysteresis", {})
        self._activate_offset = hyst_cfg.get("activate_offset", 0.5)
        self._deactivate_offset = hyst_cfg.get("deactivate_offset", 0.35)

        # Internal state
        self._current_level = WarningLevel.NONE
        self._last_warning_time: float = 0.0
        self._warning_history: List[DepartureEvent] = []
        self._max_history = 100

        # Alert callbacks
        self._visual_callback: Optional[Callable] = None
        self._haptic_callback: Optional[Callable] = None
        self._audible_callback: Optional[Callable] = None

        logger.info(
            f"LDW initialized: warning={self._warning_offset}m, "
            f"critical={self._critical_offset}m, "
            f"lane_width={self._lane_width}m"
        )

    def check_departure(
        self,
        center_offset: float,
        lateral_velocity: float = 0.0,
        speed: float = 0.0,
        curvature: float = 0.0,
        confidence: float = 1.0,
    ) -> DepartureEvent:
        """Check for lane departure and compute warning level.

        Uses lateral offset and time-to-lane-crossing to determine
        the appropriate warning level. Applies hysteresis to prevent
        warning flickering.

        Args:
            center_offset: Lateral offset from lane center in meters.
                Positive = right of center, negative = left.
            lateral_velocity: Lateral velocity in m/s (positive = moving right).
            speed: Vehicle forward speed in m/s.
            curvature: Road curvature radius in meters (0 = straight).
            confidence: Lane detection confidence [0, 1].

        Returns:
            DepartureEvent with warning level and details.
        """
        if not self.active:
            return DepartureEvent(confidence=confidence)

        if confidence < 0.3:
            # Low confidence - don't issue warnings
            return DepartureEvent(confidence=confidence)

        # Determine direction of departure
        abs_offset = abs(center_offset)
        direction = "right" if center_offset > 0 else "left"

        # Compute distance to lane boundary
        half_lane = self._lane_width / 2.0
        half_vehicle = self._vehicle_width / 2.0
        distance_to_boundary = half_lane - half_vehicle - abs_offset

        # Compute time-to-lane-crossing (TLC)
        tlc = self._compute_tlc(
            center_offset, lateral_velocity, speed, curvature
        )

        # Determine warning level with hysteresis
        warning_level = self._apply_hysteresis(abs_offset, tlc)

        # Create departure event
        event = DepartureEvent(
            level=warning_level,
            offset=center_offset,
            direction=direction,
            tlc=tlc,
            speed=speed,
            curvature=curvature,
            timestamp=time.time(),
            confidence=confidence,
        )

        # Update internal state
        self._current_level = warning_level

        # Record event
        if warning_level != WarningLevel.NONE:
            self._record_event(event)

        logger.debug(
            f"LDW check: offset={center_offset:.3f}m, "
            f"direction={direction}, TLC={tlc}, "
            f"level={warning_level.value}"
        )
        return event

    def _compute_tlc(
        self,
        center_offset: float,
        lateral_velocity: float,
        speed: float,
        curvature: float,
    ) -> Optional[float]:
        """Compute Time-to-Lane-Crossing (TLC).

        Estimates the time until the vehicle crosses the lane boundary
        based on current lateral offset and velocity.

        Args:
            center_offset: Current lateral offset (meters).
            lateral_velocity: Lateral velocity (m/s).
            speed: Forward speed (m/s).
            curvature: Road curvature radius (meters).

        Returns:
            TLC in seconds, or None if cannot be computed.
        """
        half_lane = self._lane_width / 2.0
        half_vehicle = self._vehicle_width / 2.0
        abs_offset = abs(center_offset)

        # Distance to boundary
        distance_to_boundary = half_lane - half_vehicle - abs_offset

        if distance_to_boundary <= 0:
            return 0.0  # Already outside boundary

        # Method 1: TLC from lateral velocity
        if abs(lateral_velocity) > 0.05:
            tlc_velocity = distance_to_boundary / abs(lateral_velocity)
        else:
            tlc_velocity = None

        # Method 2: TLC from curvature (centripetal drift)
        if speed > 1.0 and curvature > 0:
            # Lateral acceleration due to curvature: a_lat = v² / R
            a_lat = speed ** 2 / curvature
            # Time to reach boundary under constant lateral acceleration
            # d = 0.5 * a * t²  =>  t = sqrt(2d / a)
            if a_lat > 0.01:
                tlc_curvature = np.sqrt(2.0 * distance_to_boundary / a_lat)
            else:
                tlc_curvature = None
        else:
            tlc_curvature = None

        # Take minimum of both estimates
        tlc_values = [t for t in [tlc_velocity, tlc_curvature] if t is not None]

        if not tlc_values:
            return None

        return float(min(tlc_values))

    def _apply_hysteresis(
        self,
        abs_offset: float,
        tlc: Optional[float],
    ) -> WarningLevel:
        """Apply hysteresis to prevent warning flickering.

        Uses different thresholds for activation and deactivation
        to ensure stable warning state transitions.

        Args:
            abs_offset: Absolute lateral offset in meters.
            tlc: Time-to-lane-crossing in seconds.

        Returns:
            Appropriate warning level.
        """
        # Critical conditions
        if abs_offset >= self._critical_offset:
            return WarningLevel.CRITICAL

        if tlc is not None and tlc <= self._tlc_critical:
            return WarningLevel.CRITICAL

        # Warning conditions
        if self._current_level == WarningLevel.NONE:
            # Need to exceed activation threshold
            if abs_offset >= self._activate_offset:
                return WarningLevel.WARNING
            if tlc is not None and tlc <= self._tlc_warning:
                return WarningLevel.WARNING
        else:
            # Already in warning state - use deactivation threshold
            if abs_offset >= self._deactivate_offset:
                return WarningLevel.WARNING
            if tlc is not None and tlc <= self._tlc_warning:
                return WarningLevel.WARNING

        # Soft warning (approaching boundary)
        if abs_offset >= self._warning_offset * 0.7:
            if self._current_level in (WarningLevel.SOFT, WarningLevel.WARNING):
                return WarningLevel.SOFT

        return WarningLevel.NONE

    def trigger_warning(self, event: DepartureEvent) -> None:
        """Trigger the appropriate warning alerts.

        Args:
            event: DepartureEvent to trigger warnings for.
        """
        if event.level == WarningLevel.NONE:
            return

        # Check cooldown
        current_time = time.time()
        elapsed_ms = (current_time - self._last_warning_time) * 1000
        if elapsed_ms < self._cooldown_ms:
            return

        # Visual alert
        if self._visual_alert and self._visual_callback:
            try:
                self._visual_callback(event)
            except Exception as e:
                logger.error(f"Visual alert callback error: {e}")

        # Haptic alert
        if self._haptic_alert and self._haptic_callback:
            try:
                self._haptic_callback(event)
            except Exception as e:
                logger.error(f"Haptic alert callback error: {e}")

        # Audible alert
        if self._audible_alert and self._audible_callback:
            try:
                self._audible_callback(event)
            except Exception as e:
                logger.error(f"Audible alert callback error: {e}")

        self._last_warning_time = current_time
        logger.info(
            f"Warning triggered: level={event.level.value}, "
            f"direction={event.direction}, offset={event.offset:.3f}m"
        )

    def set_visual_callback(self, callback: Callable) -> None:
        """Set the visual alert callback.

        Args:
            callback: Function that takes a DepartureEvent.
        """
        self._visual_callback = callback

    def set_haptic_callback(self, callback: Callable) -> None:
        """Set the haptic alert callback.

        Args:
            callback: Function that takes a DepartureEvent.
        """
        self._haptic_callback = callback

    def set_audible_callback(self, callback: Callable) -> None:
        """Set the audible alert callback.

        Args:
            callback: Function that takes a DepartureEvent.
        """
        self._audible_callback = callback

    def _record_event(self, event: DepartureEvent) -> None:
        """Record a departure event in history.

        Args:
            event: Event to record.
        """
        self._warning_history.append(event)
        if len(self._warning_history) > self._max_history:
            self._warning_history = self._warning_history[-self._max_history:]

    def get_warning_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent warning history.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of serialized departure events.
        """
        events = self._warning_history[-limit:]
        return [e.to_dict() for e in events]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics from warning history.

        Returns:
            Dictionary with warning statistics.
        """
        if not self._warning_history:
            return {
                "total_warnings": 0,
                "current_level": self._current_level.value,
            }

        total = len(self._warning_history)
        critical_count = sum(
            1 for e in self._warning_history if e.level == WarningLevel.CRITICAL
        )
        warning_count = sum(
            1 for e in self._warning_history if e.level == WarningLevel.WARNING
        )
        soft_count = sum(
            1 for e in self._warning_history if e.level == WarningLevel.SOFT
        )
        left_count = sum(
            1 for e in self._warning_history if e.direction == "left"
        )
        right_count = sum(
            1 for e in self._warning_history if e.direction == "right"
        )

        avg_offset = np.mean([abs(e.offset) for e in self._warning_history])

        return {
            "total_warnings": total,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "soft_count": soft_count,
            "left_departures": left_count,
            "right_departures": right_count,
            "avg_offset": round(float(avg_offset), 4),
            "current_level": self._current_level.value,
        }

    def enable(self) -> None:
        """Enable the LDW system."""
        self.active = True
        logger.info("LDW system enabled")

    def disable(self) -> None:
        """Disable the LDW system."""
        self.active = False
        self._current_level = WarningLevel.NONE
        logger.info("LDW system disabled")

    def reset(self) -> None:
        """Reset LDW state."""
        self._current_level = WarningLevel.NONE
        self._last_warning_time = 0.0
        self._warning_history.clear()
        logger.debug("LDW state reset")
