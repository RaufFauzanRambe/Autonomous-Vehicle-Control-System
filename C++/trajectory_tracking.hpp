/**
 * @file trajectory_tracking.hpp
 * @brief Trajectory tracking controller for autonomous vehicle path following.
 *
 * This module implements a trajectory tracking controller that computes
 * steering, throttle, and brake commands to guide the vehicle along a
 * planned reference trajectory. The controller uses a decoupled lateral
 * and longitudinal control strategy:
 *
 *   - **Lateral control**: PD controller on the heading/cross-track error
 *     to compute the steering angle. The proportional gain (Kp_lateral)
 *     determines the aggressiveness of the heading correction, while the
 *     derivative gain (Kd_lateral) provides damping to reduce oscillations.
 *
 *   - **Longitudinal control**: PI controller on the velocity error to
 *     compute throttle and brake commands. The proportional gain
 *     (Kp_longitudinal) tracks the reference speed, while the integral
 *     gain (Ki_longitudinal) eliminates steady-state speed errors.
 *
 * The module also provides utilities for finding the closest point on
 * the reference trajectory and computing detailed tracking error metrics.
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#ifndef AVCS_TRAJECTORY_TRACKING_HPP_
#define AVCS_TRAJECTORY_TRACKING_HPP_

#include <array>
#include <cstddef>
#include <vector>

namespace avcs {

// Forward declarations — defined in localization.hpp
struct Pose3D;
struct Velocity3D;

/**
 * @brief A single waypoint in the reference trajectory.
 *
 * Each trajectory point specifies the desired position, heading, speed,
 * acceleration, and path curvature at a given time. The controller
 * interpolates between consecutive points for smooth tracking.
 */
struct TrajectoryPoint {
    double x;               ///< X position in world frame (meters)
    double y;               ///< Y position in world frame (meters)
    double z;               ///< Z position in world frame (meters)
    double yaw;             ///< Heading angle (radians)
    double velocity;        ///< Desired longitudinal speed (m/s)
    double acceleration;    ///< Desired longitudinal acceleration (m/s²)
    double curvature;       ///< Path curvature at this point (1/meters)
    double timestamp;       ///< Time at which the vehicle should reach this point (seconds)
};

/**
 * @brief Tracking error decomposition between current pose and target point.
 *
 * Expresses the tracking error in the Frenet frame of the reference
 * trajectory point, providing lateral, longitudinal, heading, and
 * curvature error components that the controller uses to compute
 * corrective actions.
 */
struct TrackingError {
    double lateral_error;        ///< Cross-track error (meters); positive = left of path
    double longitudinal_error;   ///< Along-track error (meters); positive = ahead of target
    double heading_error;        ///< Heading deviation (radians); positive = pointing left
    double curvature_error;      ///< Curvature deviation (1/meters)
};

/**
 * @brief Vehicle control command output from the trajectory tracker.
 *
 * Contains the steering angle, throttle position, brake pressure,
 * and gear selection that the low-level vehicle controllers should
 * apply to follow the reference trajectory.
 */
struct ControlCommand {
    double steering_angle;   ///< Front wheel steering angle (radians); positive = left
    double throttle;         ///< Throttle position [0.0, 1.0]; 0 = no throttle
    double brake;            ///< Brake pressure [0.0, 1.0]; 0 = no braking
    int gear;                ///< Gear selection: -1=reverse, 0=neutral, 1=drive
};

/**
 * @brief Trajectory tracking controller with decoupled lateral/longitudinal control.
 *
 * The TrajectoryTracker computes control commands by:
 *   1. Finding the closest point on the reference trajectory to the
 *      current vehicle pose.
 *   2. Computing the lateral and longitudinal tracking errors in the
 *      Frenet frame.
 *   3. Applying a PD law for lateral (steering) control and a PI law
 *      for longitudinal (speed) control.
 *
 * The controller gains are tunable at runtime via setGains().
 *
 * Thread safety:
 *   The class is NOT thread-safe. External synchronization is required
 *   if accessed from multiple threads.
 *
 * Usage example:
 * @code
 *   avcs::TrajectoryTracker tracker(1.5, 0.3, 1.0, 0.1);
 *   auto cmd = tracker.computeControl(pose, vel, trajectory, target_idx);
 * @endcode
 */
class TrajectoryTracker {
public:
    /**
     * @brief Construct the trajectory tracker with control gains.
     *
     * @param kp_lateral       Proportional gain for lateral (steering) control.
     * @param kd_lateral       Derivative gain for lateral (steering) control.
     * @param kp_longitudinal  Proportional gain for longitudinal (speed) control.
     * @param ki_longitudinal  Integral gain for longitudinal (speed) control.
     */
    TrajectoryTracker(double kp_lateral = 1.5,
                      double kd_lateral = 0.3,
                      double kp_longitudinal = 1.0,
                      double ki_longitudinal = 0.1);

    /**
     * @brief Destructor.
     */
    ~TrajectoryTracker() = default;

    // Default copy and move semantics
    TrajectoryTracker(const TrajectoryTracker&) = default;
    TrajectoryTracker& operator=(const TrajectoryTracker&) = default;
    TrajectoryTracker(TrajectoryTracker&&) = default;
    TrajectoryTracker& operator=(TrajectoryTracker&&) = default;

    /**
     * @brief Compute control commands to track the reference trajectory.
     *
     * Determines the closest trajectory point, calculates tracking errors,
     * and applies the PD/PI control laws to produce steering, throttle,
     * and brake commands.
     *
     * @param current_pose  Current estimated pose of the vehicle.
     * @param current_vel   Current velocity of the vehicle.
     * @param trajectory    The reference trajectory as a vector of TrajectoryPoints.
     * @param target_idx    Index of the target trajectory point (hint for search).
     * @return ControlCommand with steering, throttle, brake, and gear.
     */
    ControlCommand computeControl(const Pose3D& current_pose,
                                  const Velocity3D& current_vel,
                                  const std::vector<TrajectoryPoint>& trajectory,
                                  size_t target_idx);

    /**
     * @brief Compute tracking error between the current pose and a target point.
     *
     * Decomposes the error into lateral, longitudinal, heading, and
     * curvature components in the Frenet frame of the target point.
     *
     * @param pose    Current vehicle pose.
     * @param target  Target trajectory point.
     * @return TrackingError with all error components.
     */
    TrackingError computeError(const Pose3D& pose, const TrajectoryPoint& target);

    /**
     * @brief Find the index of the closest trajectory point to the given pose.
     *
     * Performs a nearest-neighbor search over the trajectory to find
     * the point with minimum Euclidean distance to the vehicle's
     * position. For efficiency, the search can be constrained to a
     * local window around the previous closest point.
     *
     * @param pose        Current vehicle pose.
     * @param trajectory  The reference trajectory.
     * @return Index of the closest trajectory point.
     */
    size_t findClosestPoint(const Pose3D& pose,
                            const std::vector<TrajectoryPoint>& trajectory);

    /**
     * @brief Update the controller gains at runtime.
     *
     * Allows adaptive gain scheduling based on vehicle speed, road
     * conditions, or other contextual factors.
     *
     * @param kp_lat  New proportional gain for lateral control.
     * @param kd_lat  New derivative gain for lateral control.
     * @param kp_lon  New proportional gain for longitudinal control.
     * @param ki_lon  New integral gain for longitudinal control.
     */
    void setGains(double kp_lat, double kd_lat, double kp_lon, double ki_lon);

private:
    /**
     * @brief Lateral control proportional gain.
     */
    double kp_lateral_;

    /**
     * @brief Lateral control derivative gain.
     */
    double kd_lateral_;

    /**
     * @brief Longitudinal control proportional gain.
     */
    double kp_longitudinal_;

    /**
     * @brief Longitudinal control integral gain.
     */
    double ki_longitudinal_;

    /**
     * @brief Accumulated integral error for longitudinal PI controller.
     */
    double integral_error_;

    /**
     * @brief Previous tracking error for derivative computation.
     */
    TrackingError prev_error_;
};

}  // namespace avcs

#endif  // AVCS_TRAJECTORY_TRACKING_HPP_
