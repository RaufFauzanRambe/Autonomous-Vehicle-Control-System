/**
 * @file sensor_fusion.hpp
 * @brief Multi-sensor data fusion module for the Autonomous Vehicle Control System.
 *
 * This module provides a centralized sensor fusion framework that combines
 * measurements from heterogeneous sensors (LIDAR, RADAR, CAMERA, GPS, IMU,
 * ULTRASONIC) into a unified state estimate. It employs an Extended Kalman
 * Filter (EKF) approach with configurable fusion modes and synchronization
 * thresholds to handle out-of-sequence and asynchronous measurements robustly.
 *
 * Key features:
 *   - Supports six sensor modalities with per-sensor covariance handling
 *   - Time-synchronized measurement buffering with configurable sync threshold
 *   - Predict-update cycle for real-time state propagation
 *   - Thread-safe measurement ingestion via mutex-protected buffers
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#ifndef AVCS_SENSOR_FUSION_HPP_
#define AVCS_SENSOR_FUSION_HPP_

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace avcs {

/**
 * @brief Enumeration of supported sensor types in the fusion pipeline.
 *
 * Each sensor modality has distinct noise characteristics, update rates,
 * and observability properties that the fusion filter accounts for when
 * weighting measurements.
 */
enum class SensorType {
    LIDAR,      ///< Light Detection and Ranging — 3D point cloud range sensor
    RADAR,      ///< Radio Detection and Ranging — velocity-aware range sensor
    CAMERA,     ///< Visual camera — appearance-based perception sensor
    GPS,        ///< Global Positioning System — absolute position sensor
    IMU,        ///< Inertial Measurement Unit — acceleration and angular rate sensor
    ULTRASONIC  ///< Ultrasonic — short-range proximity sensor
};

/**
 * @brief Represents a single sensor measurement with associated metadata.
 *
 * Encapsulates the raw data from a sensor along with its type, timestamp,
 * and uncertainty characterization. The covariance is stored as a 3×3
 * upper-triangular matrix in row-major order (9 elements).
 */
struct SensorMeasurement {
    SensorType sensor_type;               ///< Type of the source sensor
    double timestamp;                      ///< Measurement timestamp in seconds (epoch)
    std::vector<double> data;              ///< Raw measurement data (dimension varies by sensor)
    std::array<double, 9> covariance;      ///< 3×3 covariance matrix in row-major order
    std::string sensor_id;                 ///< Unique identifier for the sensor instance
};

/**
 * @brief Represents the fused output state from the sensor fusion filter.
 *
 * Contains the best estimate of the vehicle's position, velocity, and
 * orientation along with the full 9×9 state covariance matrix that
 * characterizes the uncertainty of the estimate.
 */
struct FusedState {
    std::array<double, 3> position;        ///< [x, y, z] position in world frame (meters)
    std::array<double, 3> velocity;        ///< [vx, vy, vz] velocity in world frame (m/s)
    std::array<double, 3> orientation;     ///< [roll, pitch, yaw] Euler angles (radians)
    Eigen::MatrixXd covariance;            ///< 9×9 state covariance matrix
};

/**
 * @brief Multi-sensor fusion engine using an Extended Kalman Filter.
 *
 * The SensorFusion class maintains a 9-dimensional state vector
 * [x, y, z, vx, vy, vz, roll, pitch, yaw] and performs predict-update
 * cycles as measurements arrive. It supports configurable fusion modes
 * (sequential vs. batch) and a synchronization threshold that determines
 * how tightly measurements from different sensors must be aligned in time
 * before they are fused together.
 *
 * Thread safety:
 *   All public methods are thread-safe. Measurements can be added from
 *   multiple threads concurrently. The predict/update cycle is protected
 *   by an internal mutex to prevent data races on the state vector.
 *
 * Usage example:
 * @code
 *   avcs::SensorFusion fusion("sequential", 0.05);
 *   fusion.addMeasurement(lidar_measurement);
 *   fusion.predict(0.01);
 *   fusion.update();
 *   auto state = fusion.getFusedState();
 * @endcode
 */
class SensorFusion {
public:
    /**
     * @brief Construct a SensorFusion instance with configuration parameters.
     *
     * @param fusion_mode      Fusion strategy: "sequential" processes measurements
     *                         one at a time; "batch" fuses synchronized groups.
     * @param sync_threshold   Maximum time difference (seconds) between measurements
     *                         to be considered synchronized for batch fusion.
     */
    SensorFusion(const std::string& fusion_mode = "sequential",
                 double sync_threshold = 0.05);

    /**
     * @brief Destructor.
     */
    ~SensorFusion() = default;

    // Copy and move semantics
    SensorFusion(const SensorFusion&) = delete;
    SensorFusion& operator=(const SensorFusion&) = delete;
    SensorFusion(SensorFusion&&) = default;
    SensorFusion& operator=(SensorFusion&&) = default;

    /**
     * @brief Add a new sensor measurement to the fusion pipeline.
     *
     * The measurement is buffered internally according to the configured
     * fusion mode. In sequential mode it is queued for the next update;
     * in batch mode it is grouped with temporally close measurements.
     *
     * @param measurement  The sensor measurement to ingest.
     */
    void addMeasurement(const SensorMeasurement& measurement);

    /**
     * @brief Propagate the state forward by the given time step.
     *
     * Applies the process model (constant-velocity with orientation
     * propagation) to advance the state estimate by dt seconds. The
     * state covariance is also propagated using the process noise model.
     *
     * @param dt  Time step in seconds for the prediction.
     */
    void predict(double dt);

    /**
     * @brief Fuse all buffered measurements into the current state.
     *
     * In sequential mode, each measurement is applied as a separate
     * EKF update step. In batch mode, synchronized groups are fused
     * together using a stacked measurement Jacobian.
     */
    void update();

    /**
     * @brief Retrieve the current fused state estimate.
     *
     * @return The FusedState containing position, velocity, orientation,
     *         and the full state covariance matrix.
     */
    FusedState getFusedState() const;

    /**
     * @brief Reset the filter to an uninitialized state.
     *
     * Clears all buffered measurements and resets the state vector
     * and covariance to their initial (high-uncertainty) values.
     */
    void reset();

private:
    /**
     * @brief 9-dimensional state vector: [x, y, z, vx, vy, vz, roll, pitch, yaw].
     */
    Eigen::VectorXd state_;

    /**
     * @brief 9×9 state covariance matrix.
     */
    Eigen::MatrixXd covariance_;

    /**
     * @brief Buffer of pending measurements awaiting fusion.
     *
     * Organized as a per-sensor-type map for efficient batch retrieval.
     */
    std::vector<SensorMeasurement> measurement_buffer_;

    /**
     * @brief Mutex protecting concurrent access to state and buffers.
     */
    mutable std::mutex mutex_;

    /**
     * @brief Fusion mode: "sequential" or "batch".
     */
    std::string fusion_mode_;

    /**
     * @brief Synchronization threshold in seconds for batch fusion.
     */
    double sync_threshold_;
};

}  // namespace avcs

#endif  // AVCS_SENSOR_FUSION_HPP_
