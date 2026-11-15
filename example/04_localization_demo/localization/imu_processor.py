"""Inertial dead reckoning with known-bias compensation.

Extension point: replace the planar Euler integrator with a strapdown
INS using quaternion attitude propagation, body-to-world rotation of the
specific force vector, and gravity compensation; identify biases from
Allan variance instead of hard-coding them.
"""

from __future__ import annotations


class IMUProcessor:
    """Integrate bias-corrected IMU signals into velocity, yaw and increments.

    The processor keeps a planar (2-D) state: velocity ``(vx, vy)`` [m/s],
    yaw [rad] and the integrated dead-reckoned position [m]. Each
    :meth:`update` call subtracts the known biases from the raw signals,
    integrates velocity and yaw with a semi-implicit Euler step, and
    returns the position increment ``(dx, dy)`` [m] over that step.

    Simplification: velocity is integrated in the world frame without
    rotating the acceleration by yaw (valid for the straight-line demo
    log where yaw stays near zero).

    Extension point: full strapdown INS with quaternion attitude and
    body-frame to world-frame rotation of the measured accelerations.
    """

    def __init__(self, config: dict) -> None:
        """Configure the processor and reset the state.

        Args:
            config: ``imu`` section of the module YAML with keys
                ``accel_bias_mpss`` [m/s^2] (known forward-axis
                accelerometer bias, subtracted from ``ax`` before
                integration) and ``gyro_bias_radps`` [rad/s] (known yaw
                rate bias, subtracted from the yaw rate before
                integration).
        """
        self._accel_bias_mpss = float(config["accel_bias_mpss"])
        self._gyro_bias_radps = float(config["gyro_bias_radps"])
        self._velocity: tuple[float, float] = (0.0, 0.0)
        self._yaw = 0.0
        self._position: tuple[float, float] = (0.0, 0.0)

    def reset(self, initial_velocity: tuple[float, float] | None = None) -> None:
        """Reset velocity, yaw and the integrated position.

        Args:
            initial_velocity: Starting velocity ``(vx, vy)`` [m/s]; when
                ``None`` the state starts at rest ``(0.0, 0.0)``. The
                demo seeds this from GPS displacement because a pure IMU
                cannot observe absolute velocity.
        """
        if initial_velocity is None:
            self._velocity = (0.0, 0.0)
        else:
            self._velocity = (float(initial_velocity[0]), float(initial_velocity[1]))
        self._yaw = 0.0
        self._position = (0.0, 0.0)

    def update(
        self,
        ax_mpss: float,
        ay_mpss: float,
        yaw_rate_radps: float,
        dt: float,
    ) -> tuple[float, float]:
        """Integrate one IMU sample and return the position increment.

        Processing order per call: subtract the known biases, integrate
        velocity (``vx += ax_clean * dt``, ``vy += ay_clean * dt``),
        integrate yaw (``yaw += (yaw_rate - gyro_bias) * dt``), compute
        the increment from the updated velocity (``dx = vx * dt``,
        ``dy = vy * dt``) and accumulate it into the internal
        dead-reckoned position.

        Args:
            ax_mpss: Raw forward acceleration [m/s^2]; the known bias
                ``accel_bias_mpss`` is removed first.
            ay_mpss: Raw lateral acceleration [m/s^2]; used as reported
                (the demo log carries no lateral bias).
            yaw_rate_radps: Raw yaw rate [rad/s]; the known bias
                ``gyro_bias_radps`` is removed first.
            dt: Time step [s]; ``0.0`` is allowed and produces a zero
                increment (used for the first log sample).

        Returns:
            Position increment ``(dx, dy)`` [m] over this step in the
            world frame.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        ax_clean = float(ax_mpss) - self._accel_bias_mpss
        ay_clean = float(ay_mpss)
        yaw_rate_clean = float(yaw_rate_radps) - self._gyro_bias_radps

        self._velocity = (
            self._velocity[0] + ax_clean * dt,
            self._velocity[1] + ay_clean * dt,
        )
        self._yaw += yaw_rate_clean * dt
        dx = self._velocity[0] * dt
        dy = self._velocity[1] * dt
        self._position = (self._position[0] + dx, self._position[1] + dy)
        return (dx, dy)

    @property
    def velocity(self) -> tuple[float, float]:
        """Current dead-reckoned velocity ``(vx, vy)`` [m/s].

        Returns:
            Velocity components in the world frame.
        """
        return self._velocity

    @property
    def yaw(self) -> float:
        """Current dead-reckoned yaw [rad], CCW positive, 0 rad = +x axis.

        Returns:
            Integrated yaw angle.
        """
        return self._yaw

    @property
    def position(self) -> tuple[float, float]:
        """Integrated dead-reckoned position ``(x, y)`` [m] since the last reset.

        Returns:
            Cumulative position obtained by summing all increments.
        """
        return self._position
