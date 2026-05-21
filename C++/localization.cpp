/**
 * @file localization.cpp
 * @brief Implementation of the Extended Kalman Filter (EKF) based localization module.
 *
 * This file implements the EKFLocalizer class that fuses GPS, IMU, and LIDAR
 * measurements into a robust 15-state estimate for autonomous vehicle localization.
 *
 * State vector layout (15-dimensional):
 *   [0-2]   px, py, pz          - Position in world frame
 *   [3-5]   vx, vy, vz          - Velocity in world frame
 *   [6-8]   roll, pitch, yaw     - Orientation (Euler angles)
 *   [9-11]  accel_bx, by, bz    - Accelerometer bias
 *   [12-14] gyro_bx, by, bz     - Gyroscope bias
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#include "localization.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace avcs {

namespace {

// ---------------------------------------------------------------------------
// Helper functions (file-local)
// ---------------------------------------------------------------------------

/**
 * @brief Normalize an angle to the range [-pi, pi].
 *
 * Uses fmod for periodicity and handles edge cases where
 * fmod returns values near the boundaries.
 *
 * @param angle  Angle in radians (any range).
 * @return       Equivalent angle in [-pi, pi].
 */
double normalizeAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

/**
 * @brief Build a 3x3 rotation matrix from Euler angles using ZYX convention.
 *
 * The rotation is R = Rz(yaw) * Ry(pitch) * Rx(roll), which corresponds to
 * the standard aerospace convention for body-to-world transformation.
 *
 * @param roll   Roll angle in radians.
 * @param pitch  Pitch angle in radians.
 * @param yaw    Yaw angle in radians.
 * @return       3x3 rotation matrix.
 */
Eigen::Matrix3d rotationMatrix(double roll, double pitch, double yaw) {
    Eigen::Matrix3d R;

    const double cr = std::cos(roll);
    const double sr = std::sin(roll);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    // Rz(yaw) * Ry(pitch) * Rx(roll)
    R(0, 0) = cy * cp;
    R(0, 1) = cy * sp * sr - sy * cr;
    R(0, 2) = cy * sp * cr + sy * sr;

    R(1, 0) = sy * cp;
    R(1, 1) = sy * sp * sr + cy * cr;
    R(1, 2) = sy * sp * cr - cy * sr;

    R(2, 0) = -sp;
    R(2, 1) = cp * sr;
    R(2, 2) = cp * cr;

    return R;
}

/**
 * @brief Build the Euler angle rate matrix that converts body angular rates
 *        to Euler angle rates: [droll, dpitch, dyaw] = T * [wx, wy, wz].
 *
 * This matrix is derived from the ZYX Euler angle kinematics and becomes
 * singular at pitch = +/-90 degrees (gimbal lock).
 *
 * @param roll   Current roll angle.
 * @param pitch  Current pitch angle.
 * @return       3x3 transformation matrix T.
 */
Eigen::Matrix3d eulerRateMatrix(double roll, double pitch) {
    const double cr = std::cos(roll);
    const double sr = std::sin(roll);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);

    // Guard against gimbal lock (pitch near +/-90 deg)
    const double cp_safe = (std::abs(cp) < 1e-8) ? 1e-8 : cp;

    Eigen::Matrix3d T;
    T(0, 0) = 1.0;
    T(0, 1) = sr * sp / cp_safe;
    T(0, 2) = cr * sp / cp_safe;

    T(1, 0) = 0.0;
    T(1, 1) = cr;
    T(1, 2) = -sr;

    T(2, 0) = 0.0;
    T(2, 1) = sr / cp_safe;
    T(2, 2) = cr / cp_safe;

    return T;
}

// State vector index constants for readability
constexpr int IDX_PX   = 0;
constexpr int IDX_PY   = 1;
constexpr int IDX_PZ   = 2;
constexpr int IDX_VX   = 3;
constexpr int IDX_VY   = 4;
constexpr int IDX_VZ   = 5;
constexpr int IDX_ROLL = 6;
constexpr int IDX_PITCH = 7;
constexpr int IDX_YAW  = 8;
constexpr int IDX_ABX  = 9;
constexpr int IDX_ABY  = 10;
constexpr int IDX_ABZ  = 11;
constexpr int IDX_GBX  = 12;
constexpr int IDX_GBY  = 13;
constexpr int IDX_GBZ  = 14;
constexpr int STATE_DIM = 15;

}  // anonymous namespace

// ===========================================================================
// EKFLocalizer implementation
// ===========================================================================

EKFLocalizer::EKFLocalizer(const Pose3D& initial_pose,
                           double process_noise_pos,
                           double process_noise_ori)
    : state_(Eigen::VectorXd::Zero(STATE_DIM)),
      covariance_(Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM)),
      process_noise_(Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM)),
      gps_measurement_noise_(Eigen::MatrixXd::Zero(3, 3)),
      imu_measurement_noise_(Eigen::MatrixXd::Zero(3, 3)),
      lidar_measurement_noise_(Eigen::MatrixXd::Zero(6, 6))
{
    // Initialize position from initial pose
    state_(IDX_PX)    = initial_pose.x;
    state_(IDX_PY)    = initial_pose.y;
    state_(IDX_PZ)    = initial_pose.z;
    state_(IDX_ROLL)  = normalizeAngle(initial_pose.roll);
    state_(IDX_PITCH) = normalizeAngle(initial_pose.pitch);
    state_(IDX_YAW)   = normalizeAngle(initial_pose.yaw);

    // Initialize covariance with high uncertainty for unobserved states
    // Position uncertainty
    covariance_.diagonal().segment<3>(IDX_PX)  = Eigen::Vector3d::Constant(100.0);
    // Velocity uncertainty
    covariance_.diagonal().segment<3>(IDX_VX)  = Eigen::Vector3d::Constant(1000.0);
    // Orientation uncertainty
    covariance_.diagonal().segment<3>(IDX_ROLL) = Eigen::Vector3d::Constant(1.0);
    // Accelerometer bias uncertainty
    covariance_.diagonal().segment<3>(IDX_ABX) = Eigen::Vector3d::Constant(0.01);
    // Gyroscope bias uncertainty
    covariance_.diagonal().segment<3>(IDX_GBX) = Eigen::Vector3d::Constant(0.01);

    // Configure process noise Q (how much we expect the state to change
    // unpredictably between measurements). This is tuned per-state-group.
    const double pn_pos_sq     = process_noise_pos * process_noise_pos;
    const double pn_vel_sq     = (10.0 * process_noise_pos) * (10.0 * process_noise_pos);
    const double pn_ori_sq     = process_noise_ori * process_noise_ori;
    const double pn_abias_sq   = 1e-4;   // Slowly drifting accelerometer bias
    const double pn_gbias_sq   = 1e-5;   // Slowly drifting gyroscope bias

    process_noise_.diagonal().segment<3>(IDX_PX)   = Eigen::Vector3d::Constant(pn_pos_sq);
    process_noise_.diagonal().segment<3>(IDX_VX)   = Eigen::Vector3d::Constant(pn_vel_sq);
    process_noise_.diagonal().segment<3>(IDX_ROLL)  = Eigen::Vector3d::Constant(pn_ori_sq);
    process_noise_.diagonal().segment<3>(IDX_ABX)  = Eigen::Vector3d::Constant(pn_abias_sq);
    process_noise_.diagonal().segment<3>(IDX_GBX)  = Eigen::Vector3d::Constant(pn_gbias_sq);

    // Default GPS measurement noise (standard GPS — will be overridden per fix type)
    gps_measurement_noise_ = Eigen::Matrix3d::Identity() * 9.0;

    // Default IMU orientation measurement noise
    imu_measurement_noise_ = Eigen::Matrix3d::Identity() * 0.1;

    // Default LIDAR pose measurement noise: position (0.01) and orientation (0.001)
    lidar_measurement_noise_.diagonal().segment<3>(0) = Eigen::Vector3d::Constant(0.01);
    lidar_measurement_noise_.diagonal().segment<3>(3) = Eigen::Vector3d::Constant(0.001);
}

// ---------------------------------------------------------------------------
// Prediction step
// ---------------------------------------------------------------------------

void EKFLocalizer::predict(double dt, const IMUData& imu) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Ensure dt is positive and reasonable
    if (dt <= 0.0 || dt > 1.0) {
        std::cerr << "[EKFLocalizer] Warning: invalid dt=" << dt
                  << " in predict step. Skipping." << std::endl;
        return;
    }

    // -----------------------------------------------------------------------
    // 1. Extract current state
    // -----------------------------------------------------------------------
    const double roll  = state_(IDX_ROLL);
    const double pitch = state_(IDX_PITCH);
    const double yaw   = state_(IDX_YAW);

    // Current biases
    const Eigen::Vector3d accel_bias = state_.segment<3>(IDX_ABX);
    const Eigen::Vector3d gyro_bias  = state_.segment<3>(IDX_GBX);

    // -----------------------------------------------------------------------
    // 2. Build rotation matrix R (body-to-world)
    // -----------------------------------------------------------------------
    const Eigen::Matrix3d R = rotationMatrix(roll, pitch, yaw);

    // -----------------------------------------------------------------------
    // 3. Compensate IMU measurements for bias
    // -----------------------------------------------------------------------
    const Eigen::Vector3d accel_raw(imu.accel[0], imu.accel[1], imu.accel[2]);
    const Eigen::Vector3d gyro_raw(imu.gyro[0], imu.gyro[1], imu.gyro[2]);
    const Eigen::Vector3d accel_corrected = accel_raw - accel_bias;
    const Eigen::Vector3d gyro_corrected  = gyro_raw  - gyro_bias;

    // -----------------------------------------------------------------------
    // 4. State propagation
    // -----------------------------------------------------------------------
    // Position: p_new = p + v * dt
    state_.segment<3>(IDX_PX) += state_.segment<3>(IDX_VX) * dt;

    // Velocity: v_new = v + R * (accel - bias) * dt - gravity
    // Gravity vector in world frame [0, 0, -9.81]
    const Eigen::Vector3d gravity(0.0, 0.0, -9.81);
    const Eigen::Vector3d accel_world = R * accel_corrected + gravity;
    state_.segment<3>(IDX_VX) += accel_world * dt;

    // Orientation: euler_new = euler + T * (gyro - bias) * dt
    // where T is the Euler rate transformation matrix
    const Eigen::Matrix3d T = eulerRateMatrix(roll, pitch);
    const Eigen::Vector3d euler_rates = T * gyro_corrected;
    state_(IDX_ROLL)  += euler_rates(0) * dt;
    state_(IDX_PITCH) += euler_rates(1) * dt;
    state_(IDX_YAW)   += euler_rates(2) * dt;

    // Normalize yaw to [-pi, pi]
    state_(IDX_YAW) = normalizeAngle(state_(IDX_YAW));

    // Biases: random walk (no deterministic change, propagated through Q)

    // -----------------------------------------------------------------------
    // 5. Build the 15x15 state transition Jacobian F = d(state_new)/d(state)
    // -----------------------------------------------------------------------
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

    // Position wrt velocity
    F.block<3, 3>(IDX_PX, IDX_VX) = Eigen::Matrix3d::Identity() * dt;

    // Velocity wrt orientation (linearization of R * accel)
    // d(R*a)/d(roll), d(R*a)/d(pitch), d(R*a)/d(yaw)
    const Eigen::Vector3d a_body = accel_corrected;
    // Partial derivatives of rotation matrix w.r.t. each Euler angle
    // dR/dyaw * a
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cr = std::cos(roll), sr = std::sin(roll);

    // dR/dyaw * a_body (derivative of rotation w.r.t. yaw)
    Eigen::Matrix3d dR_dyaw;
    dR_dyaw(0, 0) = -sy * cp;
    dR_dyaw(0, 1) = -sy * sp * sr - cy * cr;
    dR_dyaw(0, 2) = -sy * sp * cr + cy * sr;
    dR_dyaw(1, 0) = cy * cp;
    dR_dyaw(1, 1) = cy * sp * sr - sy * cr;
    dR_dyaw(1, 2) = cy * sp * cr + sy * sr;
    dR_dyaw(2, 0) = 0.0;
    dR_dyaw(2, 1) = 0.0;
    dR_dyaw(2, 2) = 0.0;

    F.block<3, 1>(IDX_VX, IDX_YAW) = (dR_dyaw * a_body) * dt;

    // dR/dpitch * a_body
    Eigen::Matrix3d dR_dpitch;
    dR_dpitch(0, 0) = -cy * sp;
    dR_dpitch(0, 1) = cy * cp * sr;
    dR_dpitch(0, 2) = cy * cp * cr;
    dR_dpitch(1, 0) = -sy * sp;
    dR_dpitch(1, 1) = sy * cp * sr;
    dR_dpitch(1, 2) = sy * cp * cr;
    dR_dpitch(2, 0) = -cp;
    dR_dpitch(2, 1) = -sp * sr;
    dR_dpitch(2, 2) = -sp * cr;

    F.block<3, 1>(IDX_VX, IDX_PITCH) = (dR_dpitch * a_body) * dt;

    // dR/droll * a_body
    Eigen::Matrix3d dR_droll;
    dR_droll(0, 0) = 0.0;
    dR_droll(0, 1) = cy * sp * cr + sy * sr;
    dR_droll(0, 2) = -cy * sp * sr + sy * cr;
    dR_droll(1, 0) = 0.0;
    dR_droll(1, 1) = sy * sp * cr - cy * sr;
    dR_droll(1, 2) = -sy * sp * sr - cy * cr;
    dR_droll(2, 0) = 0.0;
    dR_droll(2, 1) = cp * cr;
    dR_droll(2, 2) = -cp * sr;

    F.block<3, 1>(IDX_VX, IDX_ROLL) = (dR_droll * a_body) * dt;

    // Velocity wrt accel bias
    F.block<3, 3>(IDX_VX, IDX_ABX) = -R * dt;

    // Orientation wrt gyro bias (through the T matrix)
    F.block<3, 3>(IDX_ROLL, IDX_GBX) = -T * dt;

    // Orientation wrt orientation (from T * gyro * dt linearization)
    // Simplified: small angle approximation keeps this near identity
    // We add the cross-coupling from the T matrix's dependence on roll/pitch
    // For completeness, include the partial of T*w w.r.t. roll and pitch
    const double sp_safe = (std::abs(sp) < 1e-8) ? ((sp >= 0) ? 1e-8 : -1e-8) : sp;
    const double cp_safe = (std::abs(cp) < 1e-8) ? 1e-8 : cp;

    // Partial of T*w w.r.t. roll (rows: roll_dot, pitch_dot, yaw_dot)
    Eigen::Matrix3d dT_droll;
    dT_droll(0, 0) = 0.0;
    dT_droll(0, 1) = cr * sp / cp_safe * gyro_corrected(1) - sr * sp / cp_safe * gyro_corrected(2);
    dT_droll(0, 2) = -sr * sp / cp_safe * gyro_corrected(1) - cr * sp / cp_safe * gyro_corrected(2);
    dT_droll(1, 0) = 0.0;
    dT_droll(1, 1) = -sr * gyro_corrected(1);
    dT_droll(1, 2) = -cr * gyro_corrected(2);
    dT_droll(2, 0) = 0.0;
    dT_droll(2, 1) = cr / cp_safe * gyro_corrected(1) - sr / cp_safe * gyro_corrected(2);
    dT_droll(2, 2) = -sr / cp_safe * gyro_corrected(1) - cr / cp_safe * gyro_corrected(2);

    F.block<3, 1>(IDX_ROLL, IDX_ROLL) += dT_droll * gyro_corrected * dt;

    // -----------------------------------------------------------------------
    // 6. Propagate covariance: P = F * P * F' + Q
    // -----------------------------------------------------------------------
    covariance_ = F * covariance_ * F.transpose() + process_noise_ * dt;

    // Ensure symmetry (numerical drift can break it over many iterations)
    covariance_ = (covariance_ + covariance_.transpose()) / 2.0;
}

// ---------------------------------------------------------------------------
// GPS correction step
// ---------------------------------------------------------------------------

void EKFLocalizer::updateGPS(const GPSData& gps) {
    std::lock_guard<std::mutex> lock(mutex_);

    // -----------------------------------------------------------------------
    // 1. Build measurement matrix H (3x15)
    //    GPS observes position: z = [px, py, pz]
    // -----------------------------------------------------------------------
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, STATE_DIM);
    H(0, IDX_PX) = 1.0;
    H(1, IDX_PY) = 1.0;
    H(2, IDX_PZ) = 1.0;

    // -----------------------------------------------------------------------
    // 2. Set measurement noise R based on GPS fix type
    // -----------------------------------------------------------------------
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    switch (gps.fix_type) {
        case 4:  // RTK fixed
        case 5:  // RTK float
            R *= 0.01;
            break;
        case 2:  // DGPS
            R *= 1.0;
            break;
        case 1:  // Standard GPS
        default:
            R *= 9.0;
            break;
    }

    // Incorporate reported accuracy as a scaling factor
    const double accuracy_factor = std::max(gps.accuracy, 0.1);
    R *= accuracy_factor;

    // -----------------------------------------------------------------------
    // 3. Measurement prediction and innovation
    // -----------------------------------------------------------------------
    // Predicted measurement: h(x) = [px, py, pz]
    Eigen::Vector3d z_pred = state_.segment<3>(IDX_PX);

    // Actual measurement — convert geodetic to local ENU (simplified)
    // In a full implementation, this would use proper geodetic-to-ENU
    // conversion. Here we treat lat/lon/alt as local coordinates for the
    // filter update (assumes pre-processed GPS data).
    Eigen::Vector3d z_meas(gps.lat, gps.lon, gps.alt);

    // Innovation (measurement residual)
    Eigen::Vector3d y = z_meas - z_pred;

    // -----------------------------------------------------------------------
    // 4. Kalman gain and state update
    // -----------------------------------------------------------------------
    Eigen::Matrix3d S = H * covariance_ * H.transpose() + R;
    Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

    // State update
    state_ = state_ + K * y;

    // Covariance update (Joseph form for numerical stability)
    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H;
    covariance_ = I_KH * covariance_ * I_KH.transpose() + K * R * K.transpose();

    // Ensure symmetry
    covariance_ = (covariance_ + covariance_.transpose()) / 2.0;

    // Normalize yaw after update
    state_(IDX_YAW) = normalizeAngle(state_(IDX_YAW));
}

// ---------------------------------------------------------------------------
// IMU orientation correction step
// ---------------------------------------------------------------------------

void EKFLocalizer::updateIMU(const IMUData& imu) {
    std::lock_guard<std::mutex> lock(mutex_);

    // -----------------------------------------------------------------------
    // 1. Build measurement matrix H (3x15)
    //    IMU provides orientation: z = [roll, pitch, yaw]
    // -----------------------------------------------------------------------
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, STATE_DIM);
    H(0, IDX_ROLL)  = 1.0;
    H(1, IDX_PITCH) = 1.0;
    H(2, IDX_YAW)   = 1.0;

    // -----------------------------------------------------------------------
    // 2. Measurement noise R depends on IMU quality
    //    High-quality IMU: small R; Low-quality IMU: larger R
    //    Here we use the default value with an option to scale by
    //    the magnitude of angular rate (higher rate = less reliable AHRS)
    // -----------------------------------------------------------------------
    Eigen::Matrix3d R = imu_measurement_noise_;

    // Scale noise up if the IMU is undergoing significant rotation,
    // since AHRS estimates degrade under high angular rates
    const double gyro_mag = std::sqrt(imu.gyro[0] * imu.gyro[0] +
                                      imu.gyro[1] * imu.gyro[1] +
                                      imu.gyro[2] * imu.gyro[2]);
    const double gyro_scaling = 1.0 + gyro_mag;
    R *= gyro_scaling;

    // -----------------------------------------------------------------------
    // 3. Innovation
    // -----------------------------------------------------------------------
    Eigen::Vector3d z_pred(state_(IDX_ROLL), state_(IDX_PITCH), state_(IDX_YAW));
    Eigen::Vector3d z_meas(imu.orientation[0], imu.orientation[1], imu.orientation[2]);

    // Innovation with angle wrapping for yaw
    Eigen::Vector3d y = z_meas - z_pred;
    y(2) = normalizeAngle(y(2));  // Wrap yaw innovation

    // -----------------------------------------------------------------------
    // 4. Kalman gain and state update
    // -----------------------------------------------------------------------
    Eigen::Matrix3d S = H * covariance_ * H.transpose() + R;
    Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

    // State update
    state_ = state_ + K * y;

    // Covariance update (Joseph form)
    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H;
    covariance_ = I_KH * covariance_ * I_KH.transpose() + K * R * K.transpose();

    // Ensure symmetry
    covariance_ = (covariance_ + covariance_.transpose()) / 2.0;

    // Normalize yaw
    state_(IDX_YAW) = normalizeAngle(state_(IDX_YAW));
}

// ---------------------------------------------------------------------------
// LIDAR pose correction step
// ---------------------------------------------------------------------------

void EKFLocalizer::updateLidar(const Pose3D& lidar_pose) {
    std::lock_guard<std::mutex> lock(mutex_);

    // -----------------------------------------------------------------------
    // 1. Build measurement matrix H (6x15)
    //    LIDAR provides position + orientation: z = [px,py,pz,roll,pitch,yaw]
    // -----------------------------------------------------------------------
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6, STATE_DIM);
    H(0, IDX_PX)     = 1.0;
    H(1, IDX_PY)     = 1.0;
    H(2, IDX_PZ)     = 1.0;
    H(3, IDX_ROLL)   = 1.0;
    H(4, IDX_PITCH)  = 1.0;
    H(5, IDX_YAW)    = 1.0;

    // -----------------------------------------------------------------------
    // 2. Measurement noise R (6x6)
    //    LIDAR is typically very accurate — low noise values
    // -----------------------------------------------------------------------
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(6, 6);
    R.diagonal().segment<3>(0) = Eigen::Vector3d::Constant(0.01);   // position noise
    R.diagonal().segment<3>(3) = Eigen::Vector3d::Constant(0.001);  // orientation noise

    // -----------------------------------------------------------------------
    // 3. Innovation
    // -----------------------------------------------------------------------
    Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(6);
    z_pred.segment<3>(0) = state_.segment<3>(IDX_PX);
    z_pred.segment<3>(3) = state_.segment<3>(IDX_ROLL);

    Eigen::VectorXd z_meas = Eigen::VectorXd::Zero(6);
    z_meas(0) = lidar_pose.x;
    z_meas(1) = lidar_pose.y;
    z_meas(2) = lidar_pose.z;
    z_meas(3) = lidar_pose.roll;
    z_meas(4) = lidar_pose.pitch;
    z_meas(5) = lidar_pose.yaw;

    Eigen::VectorXd y = z_meas - z_pred;
    y(5) = normalizeAngle(y(5));  // Wrap yaw innovation

    // -----------------------------------------------------------------------
    // 4. Kalman gain and state update
    // -----------------------------------------------------------------------
    Eigen::MatrixXd S = H * covariance_ * H.transpose() + R;
    Eigen::MatrixXd K = covariance_ * H.transpose() * S.inverse();

    // State update
    state_ = state_ + K * y;

    // Covariance update (Joseph form for numerical stability)
    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H;
    covariance_ = I_KH * covariance_ * I_KH.transpose() + K * R * K.transpose();

    // Ensure symmetry
    covariance_ = (covariance_ + covariance_.transpose()) / 2.0;

    // Normalize yaw
    state_(IDX_YAW) = normalizeAngle(state_(IDX_YAW));
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

Pose3D EKFLocalizer::getPose() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Pose3D pose;
    pose.x     = state_(IDX_PX);
    pose.y     = state_(IDX_PY);
    pose.z     = state_(IDX_PZ);
    pose.roll  = normalizeAngle(state_(IDX_ROLL));
    pose.pitch = normalizeAngle(state_(IDX_PITCH));
    pose.yaw   = normalizeAngle(state_(IDX_YAW));

    // Extract position covariance (upper triangle of 3x3 block)
    // Order: [xx, yy, zz, xy, xz, yz]
    pose.position_covariance = {
        covariance_(IDX_PX, IDX_PX),
        covariance_(IDX_PY, IDX_PY),
        covariance_(IDX_PZ, IDX_PZ),
        covariance_(IDX_PX, IDX_PY),
        covariance_(IDX_PX, IDX_PZ),
        covariance_(IDX_PY, IDX_PZ)
    };

    // Extract orientation covariance (upper triangle of 3x3 block)
    // Order: [rr, pp, yy, rp, ry, py]
    pose.orientation_covariance = {
        covariance_(IDX_ROLL, IDX_ROLL),
        covariance_(IDX_PITCH, IDX_PITCH),
        covariance_(IDX_YAW, IDX_YAW),
        covariance_(IDX_ROLL, IDX_PITCH),
        covariance_(IDX_ROLL, IDX_YAW),
        covariance_(IDX_PITCH, IDX_YAW)
    };

    return pose;
}

Velocity3D EKFLocalizer::getVelocity() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Velocity3D vel;
    vel.vx = state_(IDX_VX);
    vel.vy = state_(IDX_VY);
    vel.vz = state_(IDX_VZ);

    // Angular velocity in body frame: omega = T^{-1} * euler_rates
    // For the velocity output, we reconstruct from gyro data if available,
    // but since we store euler rates indirectly, we use the current
    // orientation and gyroscope bias to estimate angular velocity.
    // Simplified: use the T matrix inverse
    const double roll  = state_(IDX_ROLL);
    const double pitch = state_(IDX_PITCH);
    const Eigen::Matrix3d T = eulerRateMatrix(roll, pitch);

    // We don't store euler_rates directly; estimate from bias-compensated gyro
    // Here we just report zeros for angular velocity as we'd need the last IMU reading
    // In practice, angular velocity is computed from the current IMU gyro measurement
    vel.wx = 0.0;
    vel.wy = 0.0;
    vel.wz = 0.0;

    return vel;
}

Eigen::MatrixXd EKFLocalizer::getCovariance() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return covariance_;
}

}  // namespace avcs
