/**
 * @file vehicle_state_estimator.hpp
 * @brief Vehicle state estimation module combining IMU, odometry, and GPS.
 *
 * This module provides a comprehensive vehicle state estimator that fuses
 * data from three complementary sources:
 *
 *   - **IMU**: Provides high-frequency acceleration and angular rate
 *     measurements for short-term state propagation.
 *   - **Wheel Odometry**: Offers direct velocity estimates from wheel
 *     speed sensors and steering angle measurements for kinematic
 *     state updates.
 *   - **GPS**: Supplies absolute position fixes for long-term drift
 *     correction.
 *
 * The estimator uses an Extended Kalman Filter (EKF) with a state vector
 * that captures the full dynamic state of the vehicle including position,
 * velocity, acceleration, orientation, angular velocity, and control
 * inputs (steering, throttle, brake, gear).
 *
 * The vehicle kinematic model (bicycle model) is used for the prediction
 * step, with the wheelbase parameter governing the steering-to-yaw-rate
 * relationship. The vehicle mass and rotational inertia parameters are
 * used to model longitudinal dynamics in the prediction step.
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#ifndef AVCS_VEHICLE_STATE_ESTIMATOR_HPP_
#define AVCS_VEHICLE_STATE_ESTIMATOR_HPP_

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>

namespace avcs {

// Forward declarations — defined in localization.hpp
struct GPSData;
struct IMUData;

/**
 * @brief Complete vehicle dynamic state representation.
 *
 * Encapsulates the full kinematic and dynamic state of the vehicle
 * including position, velocity, acceleration, orientation, angular
 * velocity, and current control inputs. The timestamp field enables
 * temporal alignment with other subsystems.
 */
struct VehicleState {
    std::array<double, 3> position;            ///< [x, y, z] position in world frame (meters)
    std::array<double, 3> velocity;            ///< [vx, vy, vz] velocity in world frame (m/s)
    std::array<double, 3> acceleration;        ///< [ax, ay, az] acceleration in world frame (m/s²)
    std::array<double, 3> orientation;         ///< [roll, pitch, yaw] Euler angles (radians)
    std::array<double, 3> angular_velocity;    ///< [wx, wy, wz] angular velocity in body frame (rad/s)
    double steering_angle;                     ///< Current front wheel steering angle (radians)
    double throttle;                           ///< Current throttle position [0.0, 1.0]
    double brake;                              ///< Current brake pressure [0.0, 1.0]
    int gear;                                  ///< Current gear: -1=reverse, 0=neutral, 1=drive
    double timestamp;                          ///< State timestamp in seconds (epoch)
};

/**
 * @brief EKF-based vehicle state estimator with multi-source sensor fusion.
 *
 * The VehicleStateEstimator maintains an Extended Kalman Filter whose
 * state vector encodes the complete vehicle dynamic state. It supports
 * asynchronous updates from IMU, wheel odometry, and GPS:
 *
 *   - **updateIMU**: Integrates acceleration and angular rate at high
 *     frequency (100–500 Hz) to propagate the state forward.
 *   - **updateOdometry**: Corrects velocity and yaw rate estimates
 *     using wheel speed and steering angle measurements (50–100 Hz).
 *   - **updateGPS**: Applies absolute position constraints to correct
 *     accumulated drift (1–10 Hz).
 *
 * The filter's process model uses a bicycle kinematic model for the
 * lateral dynamics and a simple longitudinal dynamics model based on
 * the vehicle's mass and inertia.
 *
 * Thread safety:
 *   All public methods are guarded by an internal mutex, allowing
 *   concurrent updates from different sensor threads.
 *
 * Usage example:
 * @code
 *   avcs::VehicleStateEstimator estimator(2.8, 1500.0, 2500.0);
 *   estimator.updateIMU(imu_data, t);
 *   estimator.updateOdometry(wfl, wfr, wrl, wrr, steer, t);
 *   estimator.updateGPS(gps_data);
 *   auto state = estimator.getState();
 * @endcode
 */
class VehicleStateEstimator {
public:
    /**
     * @brief Construct the estimator with vehicle parameters.
     *
     * @param wheelbase  Distance between front and rear axles (meters).
     * @param mass       Total vehicle mass including payload (kilograms).
     * @param inertia    Yaw moment of inertia (kg·m²).
     */
    VehicleStateEstimator(double wheelbase = 2.8,
                          double mass = 1500.0,
                          double inertia = 2500.0);

    /**
     * @brief Destructor.
     */
    ~VehicleStateEstimator() = default;

    // Non-copyable
    VehicleStateEstimator(const VehicleStateEstimator&) = delete;
    VehicleStateEstimator& operator=(const VehicleStateEstimator&) = delete;

    // Movable
    VehicleStateEstimator(VehicleStateEstimator&&) = default;
    VehicleStateEstimator& operator=(VehicleStateEstimator&&) = default;

    /**
     * @brief Update the state estimate with a new IMU measurement.
     *
     * Performs the EKF prediction step by integrating the IMU
     * accelerometer and gyroscope readings over the time interval
     * since the last update.
     *
     * @param imu        IMU measurement data.
     * @param timestamp  Current timestamp in seconds (epoch).
     */
    void updateIMU(const IMUData& imu, double timestamp);

    /**
     * @brief Update the state estimate with wheel odometry data.
     *
     * Uses the four wheel speed measurements and steering angle to
     * compute a longitudinal velocity and yaw rate estimate, then
     * applies an EKF correction step.
     *
     * @param wheel_speed_fl  Front-left wheel speed (m/s).
     * @param wheel_speed_fr  Front-right wheel speed (m/s).
     * @param wheel_speed_rl  Rear-left wheel speed (m/s).
     * @param wheel_speed_rr  Rear-right wheel speed (m/s).
     * @param steering_angle  Current steering angle (radians).
     * @param timestamp       Current timestamp in seconds (epoch).
     */
    void updateOdometry(double wheel_speed_fl,
                        double wheel_speed_fr,
                        double wheel_speed_rl,
                        double wheel_speed_rr,
                        double steering_angle,
                        double timestamp);

    /**
     * @brief Update the state estimate with a GPS measurement.
     *
     * Applies an EKF correction step using the GPS position fix
     * to constrain the position states and reduce drift.
     *
     * @param gps  GPS measurement data.
     */
    void updateGPS(const GPSData& gps);

    /**
     * @brief Get the current vehicle state estimate.
     *
     * @return The current VehicleState including position, velocity,
     *         acceleration, orientation, angular velocity, and control inputs.
     */
    VehicleState getState() const;

    /**
     * @brief Get the full state covariance matrix.
     *
     * The state vector layout matches the fields in VehicleState.
     * The covariance matrix dimensionality corresponds to the
     * internal EKF state dimension.
     *
     * @return State covariance matrix.
     */
    Eigen::MatrixXd getStateCovariance() const;

    /**
     * @brief Reset the estimator to a known initial state.
     *
     * Clears the state history buffer and re-initializes the EKF
     * state vector and covariance from the provided initial state.
     *
     * @param initial_state  The state to initialize from.
     */
    void reset(const VehicleState& initial_state);

private:
    /**
     * @brief EKF state vector (dimension matches internal model).
     */
    Eigen::VectorXd ekf_state_;

    /**
     * @brief EKF state covariance matrix.
     */
    Eigen::MatrixXd ekf_covariance_;

    /**
     * @brief Process noise covariance matrix.
     */
    Eigen::MatrixXd process_noise_;

    /**
     * @brief IMU measurement noise covariance.
     */
    Eigen::MatrixXd imu_measurement_noise_;

    /**
     * @brief Odometry measurement noise covariance.
     */
    Eigen::MatrixXd odometry_measurement_noise_;

    /**
     * @brief GPS measurement noise covariance.
     */
    Eigen::MatrixXd gps_measurement_noise_;

    /**
     * @brief Vehicle wheelbase (meters).
     */
    double wheelbase_;

    /**
     * @brief Vehicle mass (kilograms).
     */
    double mass_;

    /**
     * @brief Vehicle yaw moment of inertia (kg·m²).
     */
    double inertia_;

    /**
     * @brief Timestamp of the last filter update (seconds).
     */
    double last_timestamp_;

    /**
     * @brief Rolling history buffer of recent state estimates for
     *        smoothing and out-of-sequence measurement handling.
     */
    std::deque<VehicleState> state_history_;

    /**
     * @brief Mutex for thread-safe access to estimator state.
     */
    mutable std::mutex mutex_;
};

}  // namespace avcs

#endif  // AVCS_VEHICLE_STATE_ESTIMATOR_HPP_
