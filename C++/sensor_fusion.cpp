/**
 * @file sensor_fusion.cpp
 * @brief Implementation of the multi-sensor Extended Kalman Filter fusion engine.
 *
 * This file implements all methods of the avcs::SensorFusion class declared in
 * sensor_fusion.hpp. The filter maintains a 15-dimensional state vector:
 *
 *   [x, y, z, vx, vy, vz, roll, pitch, yaw, bx, by, bz, gbx, gby, gbz]
 *
 * where bx,by,bz are accelerometer biases and gbx,gby,gbz are gyroscope biases.
 * The publicly exposed FusedState only contains the 9 observable states
 * (position, velocity, orientation) and the corresponding 9×9 covariance block.
 *
 * The prediction step uses a constant-velocity motion model with orientation
 * propagation. The update step supports both sequential (per-measurement) and
 * batch (synchronized-group) fusion, using the numerically stable Joseph form
 * for covariance update to guarantee symmetry and positive-definiteness.
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#include "sensor_fusion.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace avcs {

// ─── Constants ────────────────────────────────────────────────────────────────

/** @brief Internal state dimension including bias states. */
static constexpr int STATE_DIM = 15;

/** @brief Observable state dimension (position + velocity + orientation). */
static constexpr int OBSERVABLE_DIM = 9;

/** @brief Initial position uncertainty standard deviation (meters). */
static constexpr double INIT_POS_STD = 10.0;

/** @brief Initial velocity uncertainty standard deviation (m/s). */
static constexpr double INIT_VEL_STD = 5.0;

/** @brief Initial orientation uncertainty standard deviation (radians). */
static constexpr double INIT_ORI_STD = 0.5;

/** @brief Initial accelerometer bias uncertainty standard deviation (m/s²). */
static constexpr double INIT_ACCEL_BIAS_STD = 0.1;

/** @brief Initial gyroscope bias uncertainty standard deviation (rad/s). */
static constexpr double INIT_GYRO_BIAS_STD = 0.01;

/** @brief Process noise position standard deviation per second (meters). */
static constexpr double PROC_NOISE_POS = 0.1;

/** @brief Process noise velocity standard deviation per second (m/s). */
static constexpr double PROC_NOISE_VEL = 0.5;

/** @brief Process noise orientation standard deviation per second (radians). */
static constexpr double PROC_NOISE_ORI = 0.05;

/** @brief Process noise accelerometer bias standard deviation per second (m/s²). */
static constexpr double PROC_NOISE_ACCEL_BIAS = 0.001;

/** @brief Process noise gyroscope bias standard deviation per second (rad/s). */
static constexpr double PROC_NOISE_GYRO_BIAS = 0.0001;

/** @brief Maximum number of measurements retained per sensor buffer. */
static constexpr size_t MAX_BUFFER_SIZE = 200;

/** @brief Default GPS position measurement noise (meters). */
static constexpr double GPS_POS_NOISE = 2.0;

/** @brief Default IMU orientation measurement noise (radians). */
static constexpr double IMU_ORI_NOISE = 0.05;

/** @brief Default LIDAR position measurement noise (meters). */
static constexpr double LIDAR_POS_NOISE = 0.1;

/** @brief Default LIDAR orientation measurement noise (radians). */
static constexpr double LIDAR_ORI_NOISE = 0.02;

/** @brief Default RADAR range measurement noise (meters). */
static constexpr double RADAR_RANGE_NOISE = 0.5;

// ─── Helper: map SensorType to string for logging ─────────────────────────────

/**
 * @brief Convert a SensorType enum to its string representation.
 * @param type  The sensor type.
 * @return Human-readable sensor type name.
 */
static const char* sensorTypeToString(SensorType type) {
    switch (type) {
        case SensorType::LIDAR:      return "LIDAR";
        case SensorType::RADAR:      return "RADAR";
        case SensorType::CAMERA:     return "CAMERA";
        case SensorType::GPS:        return "GPS";
        case SensorType::IMU:        return "IMU";
        case SensorType::ULTRASONIC: return "ULTRASONIC";
        default:                     return "UNKNOWN";
    }
}

// ─── Constructor ──────────────────────────────────────────────────────────────

SensorFusion::SensorFusion(const std::string& fusion_mode, double sync_threshold)
    : state_(STATE_DIM)
    , covariance_(STATE_DIM, STATE_DIM)
    , fusion_mode_(fusion_mode)
    , sync_threshold_(sync_threshold)
{
    // Initialize state vector to zero — no prior knowledge of position,
    // velocity, orientation, or sensor biases.
    state_.setZero();

    // Initialize covariance with high uncertainty on the diagonal.
    // Off-diagonal elements remain zero (no initial cross-correlations).
    covariance_.setZero();
    for (int i = 0; i < 3; ++i) {
        covariance_(i, i)       = INIT_POS_STD * INIT_POS_STD;       // position
        covariance_(i + 3, i + 3) = INIT_VEL_STD * INIT_VEL_STD;    // velocity
        covariance_(i + 6, i + 6) = INIT_ORI_STD * INIT_ORI_STD;    // orientation
        covariance_(i + 9, i + 9)  = INIT_ACCEL_BIAS_STD * INIT_ACCEL_BIAS_STD;  // accel bias
        covariance_(i + 12, i + 12) = INIT_GYRO_BIAS_STD * INIT_GYRO_BIAS_STD;   // gyro bias
    }

    measurement_buffer_.reserve(MAX_BUFFER_SIZE);

    std::cout << "[SensorFusion] Initialized with mode='" << fusion_mode_
              << "', sync_threshold=" << sync_threshold_ << "s, "
              << "state_dim=" << STATE_DIM << std::endl;
}

// ─── addMeasurement ───────────────────────────────────────────────────────────

void SensorFusion::addMeasurement(const SensorMeasurement& measurement) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Discard measurements that are significantly older than the newest
    // measurement already in the buffer (out-of-sequence rejection).
    // A measurement is considered too old if its timestamp is more than
    // 1.0 seconds behind the latest buffered measurement.
    if (!measurement_buffer_.empty()) {
        double latest_ts = 0.0;
        for (const auto& m : measurement_buffer_) {
            latest_ts = std::max(latest_ts, m.timestamp);
        }
        if (measurement.timestamp < latest_ts - 1.0) {
            std::cerr << "[SensorFusion] Discarding stale "
                      << sensorTypeToString(measurement.sensor_type)
                      << " measurement (ts=" << measurement.timestamp
                      << ", latest=" << latest_ts << ")" << std::endl;
            return;
        }
    }

    // Enforce buffer size limit by removing the oldest measurements
    if (measurement_buffer_.size() >= MAX_BUFFER_SIZE) {
        // Find and remove the oldest measurement
        auto oldest_it = std::min_element(
            measurement_buffer_.begin(), measurement_buffer_.end(),
            [](const SensorMeasurement& a, const SensorMeasurement& b) {
                return a.timestamp < b.timestamp;
            });
        measurement_buffer_.erase(oldest_it);
    }

    measurement_buffer_.push_back(measurement);
}

// ─── predict ──────────────────────────────────────────────────────────────────

void SensorFusion::predict(double dt) {
    if (dt <= 0.0) {
        std::cerr << "[SensorFusion] Warning: non-positive dt=" << dt
                  << " in predict(). Skipping." << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // ── Build state transition matrix F (15×15) ────────────────────────
    // Constant-velocity motion model:
    //   position_new     = position + velocity * dt
    //   velocity_new     = velocity
    //   orientation_new  = orientation  (no angular rate in state)
    //   bias_new         = bias         (biases are assumed constant)
    //
    // F has identity on the diagonal and dt in the position-velocity
    // cross-block.

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

    // Position depends on velocity: pos_new = pos + vel * dt
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

    // ── Apply state transition ─────────────────────────────────────────
    state_ = F * state_;

    // ── Build process noise covariance Q (15×15) ──────────────────────
    // Modeled as piecewise-white-noise jerk and angular acceleration.
    // For constant-velocity model, the noise on position accumulates as
    //   Q_pos = (dt^4 / 4) * sigma_acc^2
    //   Q_vel = (dt^2) * sigma_acc^2
    //   Q_cross = (dt^3 / 2) * sigma_acc^2
    // Simplified to diagonal approximation for robustness:

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;

    // Position process noise (from acceleration uncertainty)
    double sigma_acc = PROC_NOISE_VEL;
    for (int i = 0; i < 3; ++i) {
        Q(i, i)       = (dt4 / 4.0) * sigma_acc * sigma_acc;         // position
        Q(i + 3, i + 3) = dt2 * sigma_acc * sigma_acc;               // velocity
        Q(i, i + 3)   = (dt3 / 2.0) * sigma_acc * sigma_acc;        // cross-term
        Q(i + 3, i)   = (dt3 / 2.0) * sigma_acc * sigma_acc;        // cross-term
    }

    // Orientation process noise
    double sigma_ori = PROC_NOISE_ORI;
    for (int i = 0; i < 3; ++i) {
        Q(i + 6, i + 6) = dt2 * sigma_ori * sigma_ori;
    }

    // Accelerometer bias random walk
    double sigma_ab = PROC_NOISE_ACCEL_BIAS;
    for (int i = 0; i < 3; ++i) {
        Q(i + 9, i + 9) = dt * sigma_ab * sigma_ab;
    }

    // Gyroscope bias random walk
    double sigma_gb = PROC_NOISE_GYRO_BIAS;
    for (int i = 0; i < 3; ++i) {
        Q(i + 12, i + 12) = dt * sigma_gb * sigma_gb;
    }

    // ── Propagate covariance ───────────────────────────────────────────
    covariance_ = F * covariance_ * F.transpose() + Q;

    // Ensure symmetry (floating-point drift prevention)
    covariance_ = (covariance_ + covariance_.transpose()) / 2.0;
}

// ─── Helper: buildMeasurementMatrix ───────────────────────────────────────────

/**
 * @brief Construct the measurement matrix H and measurement vector z
 *        for a given sensor measurement.
 *
 * The measurement matrix H maps the 15-dim state into the measurement
 * space. Different sensor types observe different subsets of the state:
 *
 *   - GPS: observes position [x, y, z] → 3 rows
 *   - IMU: observes orientation [roll, pitch, yaw] → 3 rows
 *   - LIDAR: observes position [x, y, z] and orientation [roll, pitch, yaw] → 6 rows
 *   - RADAR: observes position [x, y, z] (range + bearing mapped) → 3 rows
 *   - CAMERA: observes orientation [roll, pitch, yaw] → 3 rows
 *   - ULTRASONIC: observes position [x, y] → 2 rows
 *
 * @param measurement  The sensor measurement.
 * @param H           [out] The measurement Jacobian matrix (m × STATE_DIM).
 * @param z           [out] The measurement vector (m × 1).
 * @param R           [out] The measurement noise covariance (m × m).
 * @return true if the measurement was successfully parsed, false otherwise.
 */
static bool buildMeasurementMatrix(const SensorMeasurement& measurement,
                                   Eigen::MatrixXd& H,
                                   Eigen::VectorXd& z,
                                   Eigen::MatrixXd& R)
{
    int meas_dim = static_cast<int>(measurement.data.size());

    switch (measurement.sensor_type) {
        case SensorType::GPS: {
            // GPS observes position [x, y, z]
            meas_dim = 3;
            H = Eigen::MatrixXd::Zero(meas_dim, STATE_DIM);
            H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

            z = Eigen::VectorXd(meas_dim);
            z(0) = measurement.data.size() > 0 ? measurement.data[0] : 0.0;
            z(1) = measurement.data.size() > 1 ? measurement.data[1] : 0.0;
            z(2) = measurement.data.size() > 2 ? measurement.data[2] : 0.0;

            R = Eigen::MatrixXd::Identity(meas_dim, meas_dim) * GPS_POS_NOISE * GPS_POS_NOISE;
            // Use covariance from measurement if available (non-zero)
            for (int i = 0; i < meas_dim; ++i) {
                if (measurement.covariance[i * 3 + i] > 1e-12) {
                    R(i, i) = measurement.covariance[i * 3 + i];
                }
            }
            return true;
        }

        case SensorType::IMU: {
            // IMU observes orientation [roll, pitch, yaw]
            meas_dim = 3;
            H = Eigen::MatrixXd::Zero(meas_dim, STATE_DIM);
            H.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();

            z = Eigen::VectorXd(meas_dim);
            z(0) = measurement.data.size() > 0 ? measurement.data[0] : 0.0;
            z(1) = measurement.data.size() > 1 ? measurement.data[1] : 0.0;
            z(2) = measurement.data.size() > 2 ? measurement.data[2] : 0.0;

            R = Eigen::MatrixXd::Identity(meas_dim, meas_dim) * IMU_ORI_NOISE * IMU_ORI_NOISE;
            for (int i = 0; i < meas_dim; ++i) {
                if (measurement.covariance[i * 3 + i] > 1e-12) {
                    R(i, i) = measurement.covariance[i * 3 + i];
                }
            }
            return true;
        }

        case SensorType::LIDAR: {
            // LIDAR observes position [x, y, z] and orientation [roll, pitch, yaw]
            meas_dim = 6;
            H = Eigen::MatrixXd::Zero(meas_dim, STATE_DIM);
            H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // position
            H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();  // orientation

            z = Eigen::VectorXd(meas_dim);
            z(0) = measurement.data.size() > 0 ? measurement.data[0] : 0.0;
            z(1) = measurement.data.size() > 1 ? measurement.data[1] : 0.0;
            z(2) = measurement.data.size() > 2 ? measurement.data[2] : 0.0;
            z(3) = measurement.data.size() > 3 ? measurement.data[3] : 0.0;
            z(4) = measurement.data.size() > 4 ? measurement.data[4] : 0.0;
            z(5) = measurement.data.size() > 5 ? measurement.data[5] : 0.0;

            R = Eigen::MatrixXd::Identity(meas_dim, meas_dim);
            for (int i = 0; i < 3; ++i) {
                R(i, i) = LIDAR_POS_NOISE * LIDAR_POS_NOISE;
            }
            for (int i = 3; i < 6; ++i) {
                R(i, i) = LIDAR_ORI_NOISE * LIDAR_ORI_NOISE;
            }
            return true;
        }

        case SensorType::RADAR: {
            // RADAR observes position [x, y, z] (converted from range/bearing)
            meas_dim = 3;
            H = Eigen::MatrixXd::Zero(meas_dim, STATE_DIM);
            H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

            z = Eigen::VectorXd(meas_dim);
            z(0) = measurement.data.size() > 0 ? measurement.data[0] : 0.0;
            z(1) = measurement.data.size() > 1 ? measurement.data[1] : 0.0;
            z(2) = measurement.data.size() > 2 ? measurement.data[2] : 0.0;

            R = Eigen::MatrixXd::Identity(meas_dim, meas_dim) * RADAR_RANGE_NOISE * RADAR_RANGE_NOISE;
            for (int i = 0; i < meas_dim; ++i) {
                if (measurement.covariance[i * 3 + i] > 1e-12) {
                    R(i, i) = measurement.covariance[i * 3 + i];
                }
            }
            return true;
        }

        case SensorType::CAMERA: {
            // Camera observes orientation [roll, pitch, yaw]
            meas_dim = 3;
            H = Eigen::MatrixXd::Zero(meas_dim, STATE_DIM);
            H.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();

            z = Eigen::VectorXd(meas_dim);
            z(0) = measurement.data.size() > 0 ? measurement.data[0] : 0.0;
            z(1) = measurement.data.size() > 1 ? measurement.data[1] : 0.0;
            z(2) = measurement.data.size() > 2 ? measurement.data[2] : 0.0;

            R = Eigen::MatrixXd::Identity(meas_dim, meas_dim) * IMU_ORI_NOISE * IMU_ORI_NOISE;
            for (int i = 0; i < meas_dim; ++i) {
                if (measurement.covariance[i * 3 + i] > 1e-12) {
                    R(i, i) = measurement.covariance[i * 3 + i];
                }
            }
            return true;
        }

        case SensorType::ULTRASONIC: {
            // Ultrasonic observes 2D position [x, y] only
            meas_dim = 2;
            H = Eigen::MatrixXd::Zero(meas_dim, STATE_DIM);
            H(0, 0) = 1.0;
            H(1, 1) = 1.0;

            z = Eigen::VectorXd(meas_dim);
            z(0) = measurement.data.size() > 0 ? measurement.data[0] : 0.0;
            z(1) = measurement.data.size() > 1 ? measurement.data[1] : 0.0;

            R = Eigen::MatrixXd::Identity(meas_dim, meas_dim) * 0.5 * 0.5;
            for (int i = 0; i < meas_dim; ++i) {
                if (measurement.covariance[i * 3 + i] > 1e-12) {
                    R(i, i) = measurement.covariance[i * 3 + i];
                }
            }
            return true;
        }

        default: {
            std::cerr << "[SensorFusion] Unknown sensor type in buildMeasurementMatrix."
                      << std::endl;
            return false;
        }
    }
}

// ─── Helper: synchronizeMeasurements ──────────────────────────────────────────

/**
 * @brief Group measurements that are temporally close within the sync threshold.
 *
 * Scans the measurement buffer and clusters measurements whose timestamps
 * fall within sync_threshold of each other. Each cluster represents a set
 * of measurements that can be fused together in batch mode.
 *
 * @param buffer         The full measurement buffer.
 * @param sync_threshold Maximum time difference for grouping (seconds).
 * @return Vector of measurement groups, where each group contains measurements
 *         that are synchronized in time.
 */
static std::vector<std::vector<SensorMeasurement>> synchronizeMeasurements(
    const std::vector<SensorMeasurement>& buffer,
    double sync_threshold)
{
    if (buffer.empty()) {
        return {};
    }

    // Sort measurements by timestamp
    std::vector<SensorMeasurement> sorted_buf = buffer;
    std::sort(sorted_buf.begin(), sorted_buf.end(),
              [](const SensorMeasurement& a, const SensorMeasurement& b) {
                  return a.timestamp < b.timestamp;
              });

    std::vector<std::vector<SensorMeasurement>> groups;
    std::vector<SensorMeasurement> current_group;
    current_group.push_back(sorted_buf[0]);

    for (size_t i = 1; i < sorted_buf.size(); ++i) {
        double time_diff = sorted_buf[i].timestamp - current_group[0].timestamp;
        if (time_diff <= sync_threshold) {
            current_group.push_back(sorted_buf[i]);
        } else {
            // Close the current group and start a new one
            groups.push_back(std::move(current_group));
            current_group.clear();
            current_group.push_back(sorted_buf[i]);
        }
    }

    // Push the last group
    if (!current_group.empty()) {
        groups.push_back(std::move(current_group));
    }

    return groups;
}

// ─── update ───────────────────────────────────────────────────────────────────

void SensorFusion::update() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (measurement_buffer_.empty()) {
        return;
    }

    if (fusion_mode_ == "sequential") {
        // ── Sequential fusion: process each measurement independently ──
        for (const auto& measurement : measurement_buffer_) {
            Eigen::MatrixXd H;
            Eigen::VectorXd z;
            Eigen::MatrixXd R;

            if (!buildMeasurementMatrix(measurement, H, z, R)) {
                continue;
            }

            // Innovation: y = z - H * x
            Eigen::VectorXd y = z - H * state_;

            // Innovation covariance: S = H * P * H^T + R
            Eigen::MatrixXd S = H * covariance_ * H.transpose() + R;

            // Kalman gain: K = P * H^T * S^-1
            Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

            // State update: x = x + K * y
            state_ = state_ + K * y;

            // Covariance update using Joseph form for numerical stability:
            // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
            Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H;
            covariance_ = I_KH * covariance_ * I_KH.transpose() + K * R * K.transpose();

            // Enforce symmetry
            covariance_ = (covariance_ + covariance_.transpose()) / 2.0;
        }
    } else {
        // ── Batch fusion: process synchronized groups together ──────────
        auto groups = synchronizeMeasurements(measurement_buffer_, sync_threshold_);

        for (const auto& group : groups) {
            // Determine total measurement dimension for the stacked update
            int total_meas_dim = 0;
            for (const auto& meas : group) {
                switch (meas.sensor_type) {
                    case SensorType::GPS:        total_meas_dim += 3; break;
                    case SensorType::IMU:        total_meas_dim += 3; break;
                    case SensorType::LIDAR:      total_meas_dim += 6; break;
                    case SensorType::RADAR:      total_meas_dim += 3; break;
                    case SensorType::CAMERA:     total_meas_dim += 3; break;
                    case SensorType::ULTRASONIC: total_meas_dim += 2; break;
                }
            }

            if (total_meas_dim == 0) continue;

            // Stack all measurement matrices, vectors, and covariances
            Eigen::MatrixXd H_stacked = Eigen::MatrixXd::Zero(total_meas_dim, STATE_DIM);
            Eigen::VectorXd z_stacked = Eigen::VectorXd::Zero(total_meas_dim);
            Eigen::MatrixXd R_stacked = Eigen::MatrixXd::Zero(total_meas_dim, total_meas_dim);

            int row_offset = 0;
            for (const auto& meas : group) {
                Eigen::MatrixXd H_i;
                Eigen::VectorXd z_i;
                Eigen::MatrixXd R_i;

                if (!buildMeasurementMatrix(meas, H_i, z_i, R_i)) {
                    continue;
                }

                int m = H_i.rows();
                H_stacked.block(row_offset, 0, m, STATE_DIM) = H_i;
                z_stacked.segment(row_offset, m) = z_i;
                R_stacked.block(row_offset, row_offset, m, m) = R_i;

                row_offset += m;
            }

            // Apply single batch Kalman update with stacked measurements
            Eigen::VectorXd y = z_stacked - H_stacked * state_;
            Eigen::MatrixXd S = H_stacked * covariance_ * H_stacked.transpose() + R_stacked;
            Eigen::MatrixXd K = covariance_ * H_stacked.transpose() * S.inverse();

            state_ = state_ + K * y;

            Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H_stacked;
            covariance_ = I_KH * covariance_ * I_KH.transpose() + K * R_stacked * K.transpose();

            covariance_ = (covariance_ + covariance_.transpose()) / 2.0;
        }
    }

    // Clear the measurement buffer after processing
    measurement_buffer_.clear();
}

// ─── getFusedState ────────────────────────────────────────────────────────────

FusedState SensorFusion::getFusedState() const {
    std::lock_guard<std::mutex> lock(mutex_);

    FusedState fused;

    // Extract position (indices 0-2)
    fused.position = {
        state_(0),  // x
        state_(1),  // y
        state_(2)   // z
    };

    // Extract velocity (indices 3-5)
    fused.velocity = {
        state_(3),  // vx
        state_(4),  // vy
        state_(5)   // vz
    };

    // Extract orientation (indices 6-8)
    // Normalize angles to [-pi, pi]
    auto normalizeAngle = [](double angle) -> double {
        while (angle > M_PI)  angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    };

    fused.orientation = {
        normalizeAngle(state_(6)),  // roll
        normalizeAngle(state_(7)),  // pitch
        normalizeAngle(state_(8))   // yaw
    };

    // Extract the 9×9 observable covariance sub-block from the 15×15 matrix.
    // The observable states are: position [0-2], velocity [3-5], orientation [6-8].
    fused.covariance = covariance_.block<9, 9>(0, 0);

    return fused;
}

// ─── reset ────────────────────────────────────────────────────────────────────

void SensorFusion::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Reset state vector to zero
    state_.setZero();

    // Reset covariance to initial high uncertainty
    covariance_.setZero();
    for (int i = 0; i < 3; ++i) {
        covariance_(i, i)       = INIT_POS_STD * INIT_POS_STD;
        covariance_(i + 3, i + 3) = INIT_VEL_STD * INIT_VEL_STD;
        covariance_(i + 6, i + 6) = INIT_ORI_STD * INIT_ORI_STD;
        covariance_(i + 9, i + 9)  = INIT_ACCEL_BIAS_STD * INIT_ACCEL_BIAS_STD;
        covariance_(i + 12, i + 12) = INIT_GYRO_BIAS_STD * INIT_GYRO_BIAS_STD;
    }

    // Clear all buffered measurements
    measurement_buffer_.clear();

    std::cout << "[SensorFusion] Filter reset to initial state." << std::endl;
}

}  // namespace avcs
