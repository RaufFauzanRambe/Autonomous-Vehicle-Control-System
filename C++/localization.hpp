/**
 * @file localization.hpp
 * @brief Extended Kalman Filter (EKF) based localization module.
 *
 * This module provides a 15-state Extended Kalman Filter for vehicle
 * localization that fuses GPS, IMU, and LIDAR pose measurements into a
 * robust, high-frequency pose and velocity estimate. The 15-dimensional
 * state vector encodes:
 *   - Position:     [x, y, z]
 *   - Velocity:     [vx, vy, vz]
 *   - Orientation:  [roll, pitch, yaw]
 *   - Accel bias:   [bx, by, bz]
 *   - Gyro bias:    [gbx, gby, gbz]
 *
 * The filter supports asynchronous multi-rate sensor updates: IMU data
 * drives the prediction step at a high rate (typically 100–500 Hz),
 * while GPS and LIDAR corrections are applied at their respective
 * arrival rates (1–10 Hz and 10–20 Hz).
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#ifndef AVCS_LOCALIZATION_HPP_
#define AVCS_LOCALIZATION_HPP_

#include <Eigen/Dense>
#include <array>
#include <mutex>
#include <string>

namespace avcs {

/**
 * @brief 3D pose representation with full uncertainty characterization.
 *
 * Stores position (x, y, z) and orientation as Euler angles (roll, pitch, yaw)
 * along with 6-element covariance vectors for position and orientation.
 * Covariance elements follow the order [xx, yy, zz, xy, xz, yz] for the
 * upper-triangular portion of the 3×3 covariance matrix.
 */
struct Pose3D {
    double x;                               ///< X position in world frame (meters)
    double y;                               ///< Y position in world frame (meters)
    double z;                               ///< Z position in world frame (meters)
    double roll;                            ///< Roll angle (radians)
    double pitch;                           ///< Pitch angle (radians)
    double yaw;                             ///< Yaw angle (radians)
    std::array<double, 6> position_covariance;     ///< [xx, yy, zz, xy, xz, yz]
    std::array<double, 6> orientation_covariance;  ///< [rr, pp, yy, rp, ry, py]
};

/**
 * @brief 3D velocity representation combining linear and angular components.
 *
 * Linear velocities are expressed in the world frame (m/s) and angular
 * velocities in the body frame (rad/s).
 */
struct Velocity3D {
    double vx;      ///< Linear velocity along x-axis (m/s)
    double vy;      ///< Linear velocity along y-axis (m/s)
    double vz;      ///< Linear velocity along z-axis (m/s)
    double wx;      ///< Angular velocity about x-axis (rad/s)
    double wy;      ///< Angular velocity about y-axis (rad/s)
    double wz;      ///< Angular velocity about z-axis (rad/s)
};

/**
 * @brief GPS measurement data with quality indicators.
 *
 * Encapsulates a single GPS fix including geodetic coordinates, estimated
 * accuracy, number of satellites used, and the fix type (e.g., single-point,
 * DGPS, RTK fixed, RTK float).
 */
struct GPSData {
    double lat;                 ///< Latitude (degrees)
    double lon;                 ///< Longitude (degrees)
    double alt;                 ///< Altitude above WGS-84 ellipsoid (meters)
    double accuracy;            ///< Estimated horizontal accuracy (meters)
    int num_satellites;         ///< Number of satellites used in the fix
    int fix_type;               ///< Fix type: 0=none, 1=GPS, 2=DGPS, 4=RTK-fixed, 5=RTK-float
};

/**
 * @brief IMU measurement data including acceleration, angular rate, and orientation.
 *
 * All values are expressed in the IMU body frame. The orientation field
 * is optional and may be populated by an internal AHRS filter if available.
 */
struct IMUData {
    std::array<double, 3> accel;        ///< Linear acceleration [ax, ay, az] (m/s²)
    std::array<double, 3> gyro;         ///< Angular velocity [wx, wy, wz] (rad/s)
    std::array<double, 3> orientation;  ///< Orientation estimate [roll, pitch, yaw] (rad)
    double timestamp;                   ///< Measurement timestamp in seconds (epoch)
};

/**
 * @brief Extended Kalman Filter (EKF) based localizer for autonomous vehicles.
 *
 * The EKFLocalizer maintains a 15-dimensional error-state vector and
 * provides methods for:
 *   - Prediction using IMU measurements (high rate)
 *   - Correction using GPS position fixes
 *   - Correction using IMU orientation constraints
 *   - Correction using LIDAR-derived pose estimates
 *
 * The error-state formulation linearizes around the current best estimate
 * and is numerically stable for large orientation changes between updates.
 *
 * Thread safety:
 *   All public methods are guarded by an internal mutex. Prediction and
 *   correction calls from different threads (e.g., IMU thread, GPS thread)
 *   are safe.
 *
 * Usage example:
 * @code
 *   avcs::Pose3D initial_pose{0, 0, 0, 0, 0, 0, {}, {}};
 *   avcs::EKFLocalizer loc(initial_pose, 0.1, 0.01);
 *   loc.predict(0.01, imu_data);
 *   loc.updateGPS(gps_data);
 *   auto pose = loc.getPose();
 * @endcode
 */
class EKFLocalizer {
public:
    /**
     * @brief Construct the EKF localizer with initial pose and noise parameters.
     *
     * @param initial_pose      The starting pose estimate.
     * @param process_noise_pos Process noise standard deviation for position (meters).
     * @param process_noise_ori Process noise standard deviation for orientation (radians).
     */
    EKFLocalizer(const Pose3D& initial_pose,
                 double process_noise_pos = 0.1,
                 double process_noise_ori = 0.01);

    /**
     * @brief Destructor.
     */
    ~EKFLocalizer() = default;

    // Non-copyable
    EKFLocalizer(const EKFLocalizer&) = delete;
    EKFLocalizer& operator=(const EKFLocalizer&) = delete;

    // Movable
    EKFLocalizer(EKFLocalizer&&) = default;
    EKFLocalizer& operator=(EKFLocalizer&&) = default;

    /**
     * @brief Prediction step: propagate state using IMU data.
     *
     * Integrates the IMU accelerometer and gyroscope measurements over
     * the time step dt to advance the state estimate. Acceleration and
     * gyroscope biases are estimated and compensated internally.
     *
     * @param dt   Time step in seconds.
     * @param imu  IMU measurement data.
     */
    void predict(double dt, const IMUData& imu);

    /**
     * @brief Correction step: update state with a GPS measurement.
     *
     * Applies a Kalman update using the GPS position fix. The GPS
     * measurement model maps the state position to the observation
     * space and corrects the full 15-state vector accordingly.
     *
     * @param gps  GPS measurement data.
     */
    void updateGPS(const GPSData& gps);

    /**
     * @brief Correction step: update state with an IMU orientation constraint.
     *
     * Uses the IMU's orientation estimate (typically from an internal
     * AHRS/compass fusion) as an observation to correct the orientation
     * and bias states.
     *
     * @param imu  IMU measurement containing orientation data.
     */
    void updateIMU(const IMUData& imu);

    /**
     * @brief Correction step: update state with a LIDAR-derived pose.
     *
     * Applies a Kalman update using a LIDAR scan-matching pose estimate.
     * Both position and orientation components of the LIDAR pose are used.
     *
     * @param lidar_pose  Pose estimate from LIDAR scan matching.
     */
    void updateLidar(const Pose3D& lidar_pose);

    /**
     * @brief Get the current best pose estimate.
     *
     * @return The current Pose3D with position, orientation, and covariances.
     */
    Pose3D getPose() const;

    /**
     * @brief Get the current velocity estimate.
     *
     * @return The current Velocity3D with linear and angular components.
     */
    Velocity3D getVelocity() const;

    /**
     * @brief Get the full 15×15 state covariance matrix.
     *
     * The state ordering is:
     *   [x, y, z, vx, vy, vz, roll, pitch, yaw, bx, by, bz, gbx, gby, gbz]
     *
     * @return 15×15 covariance matrix.
     */
    Eigen::MatrixXd getCovariance() const;

private:
    /**
     * @brief 15-dimensional state vector.
     *
     * Layout: [x, y, z, vx, vy, vz, roll, pitch, yaw, bx, by, bz, gbx, gby, gbz]
     * where bx,by,bz are accelerometer biases and gbx,gby,gbz are gyroscope biases.
     */
    Eigen::VectorXd state_;

    /**
     * @brief 15×15 state covariance matrix.
     */
    Eigen::MatrixXd covariance_;

    /**
     * @brief 15×15 process noise covariance matrix.
     */
    Eigen::MatrixXd process_noise_;

    /**
     * @brief GPS measurement noise covariance (3×3).
     */
    Eigen::MatrixXd gps_measurement_noise_;

    /**
     * @brief IMU orientation measurement noise covariance (3×3).
     */
    Eigen::MatrixXd imu_measurement_noise_;

    /**
     * @brief LIDAR pose measurement noise covariance (6×6).
     */
    Eigen::MatrixXd lidar_measurement_noise_;

    /**
     * @brief Mutex for thread-safe access to state and covariance.
     */
    mutable std::mutex mutex_;
};

}  // namespace avcs

#endif  // AVCS_LOCALIZATION_HPP_
