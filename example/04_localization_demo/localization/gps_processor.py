"""First-order low-pass filtering of GPS position fixes.

Extension point: replace the scalar exponential filter with a
measurement-noise-adaptive alpha (driven by ``gps_fix_quality``) or with
a per-axis Kalman filter that also estimates GPS velocity.
"""

from __future__ import annotations


class GPSProcessor:
    """First-order (exponential) low-pass filter for 2-D GPS positions.

    The estimate follows ``est = alpha * measurement + (1 - alpha) * est``
    per axis, where ``alpha`` in [0, 1] trades smoothing (small alpha)
    against responsiveness (large alpha). The first measurement seeds the
    filter, so no prior position is needed. Note that a low-pass filter
    introduces a lag bias toward the past on moving platforms; the fused
    estimate in :mod:`~localization.sensor_fusion` compensates this with
    IMU dead reckoning.

    Extension point: make ``alpha`` adaptive to the reported fix quality
    (the ``gps_fix_quality`` column) and estimate GPS velocity for
    lead compensation of the lag bias.
    """

    def __init__(self, config: dict) -> None:
        """Configure the filter.

        Args:
            config: ``gps`` section of the module YAML with key
                ``low_pass_alpha`` [-] in [0, 1]; higher values trust the
                newest measurement more.
        """
        self._alpha = float(config["low_pass_alpha"])
        if not 0.0 <= self._alpha <= 1.0:
            raise ValueError(f"low_pass_alpha must be in [0, 1], got {self._alpha}")
        self._state: tuple[float, float] | None = None

    def process(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """Filter one GPS fix and return the current estimate.

        Args:
            measurement: Raw GPS position ``(x, y)`` [m] in the flat
                world frame.

        Returns:
            Filtered position ``(x, y)`` [m]. The first call seeds the
            estimate with the measurement itself.
        """
        if self._state is None:
            self._state = (float(measurement[0]), float(measurement[1]))
        else:
            self._state = (
                self._alpha * float(measurement[0]) + (1.0 - self._alpha) * self._state[0],
                self._alpha * float(measurement[1]) + (1.0 - self._alpha) * self._state[1],
            )
        return self._state

    @property
    def state(self) -> tuple[float, float] | None:
        """Current filtered estimate ``(x, y)`` [m], or ``None`` before the first call.

        Returns:
            The last estimate as ``(x, y)`` [m], or ``None`` when no
            measurement has been processed yet.
        """
        return self._state

    def reset(self) -> None:
        """Clear the filter state; the next call to :meth:`process` re-seeds it."""
        self._state = None
