"""Complementary fusion of GPS fixes and IMU dead reckoning.

Extension point: replace the complementary filter with a 15-state
error-state EKF (position, velocity, attitude, gyro and accelerometer
biases) or a factor-graph smoother; the ``update`` signature already
carries ``dt`` for time-dependent propagation models.
"""

from __future__ import annotations


class SensorFusion:
    """Complementary filter blending IMU increments with absolute GPS fixes.

    Intuition: the IMU dead reckoning is smooth and high-rate but drifts
    (unobserved initial velocity, residual biases), while the GPS is
    absolutely anchored but noisy. The complementary filter combines the
    two complementary frequency bands:

    ``fused = alpha * (fused + imu_dx_dy) + (1 - alpha) * gps_xy``

    With ``alpha`` close to 1 the short-term motion comes from the IMU
    increment and the estimate stays smooth, while the ``(1 - alpha)``
    term continuously pulls the estimate back to the noisy GPS anchor,
    bounding long-term drift. When no GPS fix is available the filter
    degrades gracefully to pure dead reckoning.

    Extension point: a 15-state error-state EKF with online bias
    estimation replaces the fixed ``alpha`` with measurement-noise-based
    Kalman gains and additionally estimates velocity and attitude.
    """

    def __init__(self, config: dict) -> None:
        """Configure the filter.

        Args:
            config: ``fusion`` section of the module YAML with key
                ``complementary_alpha`` [-] in [0, 1]; the weight on the
                IMU dead reckoning (0.95 typical for this demo).
        """
        self._alpha = float(config["complementary_alpha"])
        if not 0.0 <= self._alpha <= 1.0:
            raise ValueError(f"complementary_alpha must be in [0, 1], got {self._alpha}")
        self._position: tuple[float, float] | None = None

    def reset(self, initial_position: tuple[float, float] | None = None) -> None:
        """Reset the fused position.

        Args:
            initial_position: Starting position ``(x, y)`` [m]; when
                ``None`` the filter is unseeded and the first
                :meth:`update` seeds it from dead reckoning.
        """
        if initial_position is None:
            self._position = None
        else:
            self._position = (float(initial_position[0]), float(initial_position[1]))

    def update(
        self,
        gps_xy: tuple[float, float] | None,
        imu_dx_dy: tuple[float, float],
        dt: float,
    ) -> tuple[float, float]:
        """Advance the fused estimate by one IMU step and one GPS fix.

        Behavior:

        * the IMU increment is always accumulated into the fused state;
        * when a GPS fix is present, the state is blended as
          ``alpha * (state + increment) + (1 - alpha) * gps``;
        * when ``gps_xy`` is ``None`` the filter performs pure dead
          reckoning (state + increment);
        * when the filter was never seeded (``reset`` without an
          initial position and this is the first call), the state is
          seeded from dead reckoning alone.

        Args:
            gps_xy: Absolute GPS position ``(x, y)`` [m], or ``None`` to
                skip the absolute correction for this step.
            imu_dx_dy: IMU position increment ``(dx, dy)`` [m] for this
                step (already scaled by ``dt`` by the IMU processor).
            dt: Time step [s]; kept in the signature for interface
                stability. The first-order complementary filter is
                dt-independent because the increment arrives pre-scaled,
                but a time-continuous EKF extension propagates with dt.

        Returns:
            Fused position ``(x, y)`` [m].

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        if self._position is None:
            # Never-seeded case: seed the state from dead reckoning alone.
            self._position = (float(imu_dx_dy[0]), float(imu_dx_dy[1]))
        else:
            self._position = (
                self._position[0] + float(imu_dx_dy[0]),
                self._position[1] + float(imu_dx_dy[1]),
            )
        if gps_xy is not None:
            self._position = (
                self._alpha * self._position[0] + (1.0 - self._alpha) * float(gps_xy[0]),
                self._alpha * self._position[1] + (1.0 - self._alpha) * float(gps_xy[1]),
            )
        return self._position

    @property
    def position(self) -> tuple[float, float] | None:
        """Current fused position ``(x, y)`` [m], or ``None`` before the first update.

        Returns:
            The fused position, or ``None`` when the filter has not been
            seeded or updated yet.
        """
        return self._position
