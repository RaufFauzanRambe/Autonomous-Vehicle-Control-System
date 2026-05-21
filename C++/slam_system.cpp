/**
 * @file slam_system.cpp
 * @brief Implementation of the graph-based SLAM system with ICP scan matching.
 *
 * This file implements the SLAMSystem class that provides:
 *   - Iterative Closest Point (ICP) scan matching for incremental pose estimation
 *   - Keyframe management based on motion thresholds
 *   - Loop closure detection through geometric consistency checks
 *   - Pose graph optimization using Gauss-Newton on SE(3) constraints
 *
 * The ICP implementation uses brute-force nearest neighbor search (a KD-tree
 * would be preferred for production use) and SVD-based rigid transform
 * estimation. The pose graph optimizer is a simplified version; real systems
 * typically use g2o or GTSAM for robust SE(3) optimization.
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#include "slam_system.hpp"
#include "localization.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace avcs {

namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Loop closure search radius in meters
constexpr double LOOP_CLOSURE_RADIUS = 15.0;

/// Minimum translation before considering a candidate for loop closure
constexpr double LOOP_CLOSURE_MIN_TRANSLATION = 5.0;

/// Maximum ICP error for a valid loop closure match
constexpr double LOOP_CLOSURE_MAX_ICP_ERROR = 0.5;

/// Minimum confidence threshold for accepting a loop closure
constexpr double LOOP_CLOSURE_MIN_CONFIDENCE = 0.3;

/// Minimum number of ICP correspondences for a valid alignment
constexpr int ICP_MIN_CORRESPONDENCES = 10;

/// ICP convergence tolerance (change in mean squared error)
constexpr double ICP_TOLERANCE = 1e-6;

/// Number of Gauss-Newton iterations for pose graph optimization
constexpr int POSE_GRAPH_MAX_ITERATIONS = 20;

/// Damping factor for pose graph optimization (Levenberg-Marquardt style)
constexpr double POSE_GRAPH_DAMPING = 1e-3;

// ---------------------------------------------------------------------------
// Helper: Convert Pose3D to a 4x4 homogeneous transformation matrix
// ---------------------------------------------------------------------------

/**
 * @brief Convert a Pose3D (position + Euler angles) to a 4x4 homogeneous
 *        transformation matrix using ZYX Euler convention.
 *
 * @param pose  The pose to convert.
 * @return      4x4 homogeneous transformation matrix.
 */
Eigen::Matrix4d poseToMatrix(const Pose3D& pose) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    // Build rotation from ZYX Euler angles
    const double cr = std::cos(pose.roll),  sr = std::sin(pose.roll);
    const double cp = std::cos(pose.pitch), sp = std::sin(pose.pitch);
    const double cy = std::cos(pose.yaw),   sy = std::sin(pose.yaw);

    T(0, 0) = cy * cp;
    T(0, 1) = cy * sp * sr - sy * cr;
    T(0, 2) = cy * sp * cr + sy * sr;
    T(1, 0) = sy * cp;
    T(1, 1) = sy * sp * sr + cy * cr;
    T(1, 2) = sy * sp * cr - cy * sr;
    T(2, 0) = -sp;
    T(2, 1) = cp * sr;
    T(2, 2) = cp * cr;

    // Translation
    T(0, 3) = pose.x;
    T(1, 3) = pose.y;
    T(2, 3) = pose.z;

    return T;
}

// ---------------------------------------------------------------------------
// Helper: Convert a 4x4 homogeneous transformation matrix to Pose3D
// ---------------------------------------------------------------------------

/**
 * @brief Extract a Pose3D from a 4x4 homogeneous transformation matrix.
 *
 * Recovers Euler angles using the ZYX convention. Handles the gimbal lock
 * case by setting yaw to zero when pitch is near +/-90 degrees.
 *
 * @param T  4x4 homogeneous transformation matrix.
 * @return   Corresponding Pose3D.
 */
Pose3D matrixToPose(const Eigen::Matrix4d& T) {
    Pose3D pose;
    pose.x = T(0, 3);
    pose.y = T(1, 3);
    pose.z = T(2, 3);

    // Extract Euler angles from rotation matrix (ZYX convention)
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);

    // pitch = -asin(R(2,0))
    const double pitch = -std::asin(std::clamp(R(2, 0), -1.0, 1.0));

    if (std::abs(std::cos(pitch)) > 1e-6) {
        // Normal case
        pose.roll  = std::atan2(R(2, 1), R(2, 2));
        pose.pitch = pitch;
        pose.yaw   = std::atan2(R(1, 0), R(0, 0));
    } else {
        // Gimbal lock: set yaw = 0 and solve for roll
        pose.roll  = std::atan2(-R(0, 1), R(1, 1));
        pose.pitch = pitch;
        pose.yaw   = 0.0;
    }

    return pose;
}

// ---------------------------------------------------------------------------
// Helper: Extract rotation angle from a 4x4 transform
// ---------------------------------------------------------------------------

/**
 * @brief Compute the rotation angle (in radians) represented by the rotation
 *        part of a 4x4 homogeneous transform.
 *
 * Uses the formula: angle = arccos((trace(R) - 1) / 2), clamped for
 * numerical safety.
 *
 * @param T  4x4 homogeneous transformation matrix.
 * @return   Rotation angle in radians [0, pi].
 */
double rotationAngle(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const double trace = R.trace();
    const double cos_angle = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(cos_angle);
}

// ---------------------------------------------------------------------------
// Helper: Transform a 3D point by a 4x4 homogeneous matrix
// ---------------------------------------------------------------------------

/**
 * @brief Apply a 4x4 homogeneous transform to a 3D point.
 *
 * @param point  Input 3D point.
 * @param T      4x4 homogeneous transformation matrix.
 * @return       Transformed 3D point.
 */
Point3D transformPoint(const Point3D& point, const Eigen::Matrix4d& T) {
    Eigen::Vector4d p(point.x, point.y, point.z, 1.0);
    Eigen::Vector4d p_out = T * p;
    return {p_out(0), p_out(1), p_out(2), point.intensity};
}

// ---------------------------------------------------------------------------
// Brute-force nearest neighbor search
// ---------------------------------------------------------------------------

/**
 * @brief For each point in the source cloud, find the closest point in the
 *        target cloud using brute-force search.
 *
 * This is O(N*M) and is intended for small point clouds or as a reference
 * implementation. Production systems should use a KD-tree (e.g., FLANN).
 *
 * @param source  Vector of source points.
 * @param target  Vector of target points.
 * @return        Pair of (distances, target_indices) for each source point.
 */
std::pair<std::vector<double>, std::vector<int>> findNearestNeighbors(
    const std::vector<Point3D>& source,
    const std::vector<Point3D>& target)
{
    const size_t n = source.size();
    std::vector<double> distances(n, std::numeric_limits<double>::max());
    std::vector<int> indices(n, -1);

    for (size_t i = 0; i < n; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_idx = -1;

        for (size_t j = 0; j < target.size(); ++j) {
            const double dx = source[i].x - target[j].x;
            const double dy = source[i].y - target[j].y;
            const double dz = source[i].z - target[j].z;
            const double dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_idx = static_cast<int>(j);
            }
        }

        distances[i] = std::sqrt(best_dist);
        indices[i] = best_idx;
    }

    return {distances, indices};
}

// ---------------------------------------------------------------------------
// Compute optimal rigid transform using SVD
// ---------------------------------------------------------------------------

/**
 * @brief Compute the optimal rigid body transformation (rotation + translation)
 *        that aligns source points to target points using SVD decomposition.
 *
 * Algorithm:
 *   1. Compute centroids of both point sets.
 *   2. Center both point sets by subtracting centroids.
 *   3. Compute cross-covariance matrix H = src_centered' * tgt_centered.
 *   4. SVD of H: H = U * S * V'
 *   5. R = V * U' (handle reflection case: if det(R) < 0, flip sign of last
 *      column of V).
 *   6. t = centroid_tgt - R * centroid_src.
 *
 * @param source  Source point cloud (3xN matrix, each column is a point).
 * @param target  Target point cloud (3xN matrix, same column ordering as source).
 * @return        4x4 homogeneous transformation matrix.
 */
Eigen::Matrix4d computeRigidTransform(const Eigen::MatrixXd& source,
                                      const Eigen::MatrixXd& target)
{
    const int n = static_cast<int>(source.cols());

    // Need at least 3 non-degenerate point pairs
    if (n < 3) {
        return Eigen::Matrix4d::Identity();
    }

    // Step 1: Compute centroids
    const Eigen::Vector3d centroid_src = source.rowwise().mean();
    const Eigen::Vector3d centroid_tgt = target.rowwise().mean();

    // Step 2: Center the point clouds
    const Eigen::MatrixXd src_centered = source.colwise() - centroid_src;
    const Eigen::MatrixXd tgt_centered = target.colwise() - centroid_tgt;

    // Step 3: Cross-covariance matrix H = src' * tgt
    const Eigen::Matrix3d H = src_centered * tgt_centered.transpose();

    // Step 4: SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();

    // Step 5: Compute rotation R = V * U'
    Eigen::Matrix3d R = V * U.transpose();

    // Handle reflection case: if determinant is negative, flip the last column of V
    if (R.determinant() < 0.0) {
        Eigen::Matrix3d V_corrected = V;
        V_corrected.col(2) *= -1.0;
        R = V_corrected * U.transpose();
    }

    // Step 6: Compute translation
    const Eigen::Vector3d t = centroid_tgt - R * centroid_src;

    // Build 4x4 homogeneous transformation matrix
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;

    return T;
}

// ---------------------------------------------------------------------------
// ICP scan matching
// ---------------------------------------------------------------------------

/**
 * @brief Iterative Closest Point (ICP) algorithm for aligning two point clouds.
 *
 * Iteratively:
 *   1. Find closest point correspondences between source and target.
 *   2. Filter correspondences by maximum distance threshold.
 *   3. Compute optimal rigid transform using SVD.
 *   4. Apply transform to the source cloud.
 *   5. Check convergence based on MSE change.
 *
 * @param source        Source point cloud (will be modified in-place as aligned).
 * @param target        Target point cloud (reference).
 * @param max_dist      Maximum correspondence distance.
 * @param max_iter      Maximum number of iterations.
 * @param tolerance     Convergence tolerance on MSE change.
 * @return              4x4 cumulative transformation from source to target frame.
 */
Eigen::Matrix4d icpMatch(std::vector<Point3D>& source,
                         const std::vector<Point3D>& target,
                         double max_dist,
                         double max_iter,
                         double tolerance)
{
    Eigen::Matrix4d cumulative_transform = Eigen::Matrix4d::Identity();
    double prev_mse = std::numeric_limits<double>::max();

    for (int iter = 0; iter < static_cast<int>(max_iter); ++iter) {
        // Step 1: Find nearest neighbors
        auto [distances, indices] = findNearestNeighbors(source, target);

        // Step 2: Filter correspondences by max distance
        std::vector<int> valid_src;
        std::vector<int> valid_tgt;
        for (size_t i = 0; i < distances.size(); ++i) {
            if (indices[i] >= 0 && distances[i] < max_dist) {
                valid_src.push_back(static_cast<int>(i));
                valid_tgt.push_back(indices[i]);
            }
        }

        // Need at least 3 valid correspondences for a rigid transform
        if (static_cast<int>(valid_src.size()) < ICP_MIN_CORRESPONDENCES) {
            std::cerr << "[ICP] Warning: only " << valid_src.size()
                      << " valid correspondences at iteration " << iter << std::endl;
            break;
        }

        // Build source and target matrices for SVD
        const int n_valid = static_cast<int>(valid_src.size());
        Eigen::MatrixXd src_pts(3, n_valid);
        Eigen::MatrixXd tgt_pts(3, n_valid);

        for (int i = 0; i < n_valid; ++i) {
            src_pts(0, i) = source[valid_src[i]].x;
            src_pts(1, i) = source[valid_src[i]].y;
            src_pts(2, i) = source[valid_src[i]].z;

            tgt_pts(0, i) = target[valid_tgt[i]].x;
            tgt_pts(1, i) = target[valid_tgt[i]].y;
            tgt_pts(2, i) = target[valid_tgt[i]].z;
        }

        // Step 3: Compute incremental rigid transform
        const Eigen::Matrix4d delta_T = computeRigidTransform(src_pts, tgt_pts);

        // Step 4: Apply transform to the source cloud
        for (auto& pt : source) {
            pt = transformPoint(pt, delta_T);
        }

        // Update cumulative transform
        cumulative_transform = delta_T * cumulative_transform;

        // Step 5: Compute mean squared error for convergence check
        double mse = 0.0;
        for (int i = 0; i < n_valid; ++i) {
            const double dx = src_pts(0, i) - tgt_pts(0, i);
            const double dy = src_pts(1, i) - tgt_pts(1, i);
            const double dz = src_pts(2, i) - tgt_pts(2, i);
            mse += dx * dx + dy * dy + dz * dz;
        }
        mse /= static_cast<double>(n_valid);

        // Check convergence
        if (std::abs(prev_mse - mse) < tolerance) {
            break;
        }
        prev_mse = mse;
    }

    return cumulative_transform;
}

// ---------------------------------------------------------------------------
// Downsample a point cloud (random subsampling)
// ---------------------------------------------------------------------------

/**
 * @brief Randomly downsample a point cloud to a target size.
 *
 * If the point cloud is already at or below the target size, returns a copy.
 * Otherwise, randomly selects target_size points without replacement.
 *
 * @param cloud        Input point cloud.
 * @param target_size  Desired number of points after downsampling.
 * @return             Downsampled point cloud.
 */
std::vector<Point3D> downsampleCloud(const std::vector<Point3D>& cloud,
                                     size_t target_size)
{
    if (cloud.size() <= target_size) {
        return cloud;
    }

    std::vector<size_t> indices(cloud.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(42));

    std::vector<Point3D> downsampled;
    downsampled.reserve(target_size);
    for (size_t i = 0; i < target_size; ++i) {
        downsampled.push_back(cloud[indices[i]]);
    }
    return downsampled;
}

// ---------------------------------------------------------------------------
// Compute 6x6 information matrix for a pose graph edge
// ---------------------------------------------------------------------------

/**
 * @brief Build a 6x6 information matrix (inverse of covariance) for a pose
 *        graph edge based on the relative transform and ICP fitness.
 *
 * @param icp_mse      Mean squared error from ICP alignment.
 * @param n_correspondences  Number of valid ICP correspondences.
 * @return             6x6 information matrix.
 */
Eigen::MatrixXd computeInformationMatrix(double icp_mse, int n_correspondences) {
    Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6);

    // Position information: inversely proportional to MSE
    const double pos_info = (icp_mse > 1e-10) ? (1.0 / icp_mse) : 1e6;
    info.diagonal().segment<3>(0) = Eigen::Vector3d::Constant(
        std::min(pos_info, 1e6));

    // Orientation information: typically higher than position
    info.diagonal().segment<3>(3) = Eigen::Vector3d::Constant(1e3);

    // Scale by number of correspondences (more points = more reliable)
    const double scale = std::min(static_cast<double>(n_correspondences) / 100.0, 1.0);
    info *= std::max(scale, 0.1);

    return info;
}

// ---------------------------------------------------------------------------
// SE(3) logarithm map (simplified for small perturbations)
// ---------------------------------------------------------------------------

/**
 * @brief Compute a 6-vector tangent representation of the difference between
 *        two SE(3) transforms: delta = log(T_meas^{-1} * T_est).
 *
 * For small perturbations, this is approximated as:
 *   delta.head<3>() = translation error
 *   delta.tail<3>() = rotation error (axis-angle, approximated from R_err)
 *
 * @param T_est   Estimated transform.
 * @param T_meas  Measured (constraint) transform.
 * @return        6-vector error [tx, ty, tz, rx, ry, rz].
 */
Eigen::VectorXd se3Error(const Eigen::Matrix4d& T_est,
                         const Eigen::Matrix4d& T_meas)
{
    // Relative transform: T_meas^{-1} * T_est
    const Eigen::Matrix4d T_rel = T_meas.inverse() * T_est;

    Eigen::VectorXd error(6);

    // Translation error
    error.head<3>() = T_rel.block<3, 1>(0, 3);

    // Rotation error: extract axis-angle from the rotation part
    const Eigen::Matrix3d R_err = T_rel.block<3, 3>(0, 0);
    const double cos_angle = std::clamp((R_err.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);

    if (std::abs(angle) < 1e-8) {
        error.tail<3>() = Eigen::Vector3d::Zero();
    } else {
        // Use the skew-symmetric part to extract the axis
        // ln(R) = (angle / (2 * sin(angle))) * (R - R')
        const double sin_angle = std::sin(angle);
        if (std::abs(sin_angle) < 1e-10) {
            error.tail<3>() = Eigen::Vector3d::Zero();
        } else {
            const Eigen::Matrix3d ln_R = (angle / (2.0 * sin_angle)) * (R_err - R_err.transpose());
            error(3) = ln_R(2, 1);
            error(4) = ln_R(0, 2);
            error(5) = ln_R(1, 0);
        }
    }

    return error;
}

}  // anonymous namespace

// ===========================================================================
// SLAMSystem implementation
// ===========================================================================

SLAMSystem::SLAMSystem(double icp_max_correspondence_dist,
                       double icp_max_iterations,
                       double keyframe_translation_thresh,
                       double keyframe_rotation_thresh)
    : icp_max_correspondence_dist_(icp_max_correspondence_dist),
      icp_max_iterations_(icp_max_iterations),
      keyframe_translation_thresh_(keyframe_translation_thresh),
      keyframe_rotation_thresh_(keyframe_rotation_thresh),
      current_pose_{0, 0, 0, 0, 0, 0, {}, {}},
      next_frame_id_(0)
{
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void SLAMSystem::initialize(const Pose3D& initial_pose) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Set current pose from the provided initial pose
    current_pose_.x     = initial_pose.x;
    current_pose_.y     = initial_pose.y;
    current_pose_.z     = initial_pose.z;
    current_pose_.roll  = initial_pose.roll;
    current_pose_.pitch = initial_pose.pitch;
    current_pose_.yaw   = initial_pose.yaw;

    // Clear all existing state
    keyframes_.clear();
    pose_graph_edges_.clear();
    local_map_.clear();
    next_frame_id_ = 0;

    // Create the first keyframe at the initial pose
    KeyFrame kf;
    kf.frame_id = next_frame_id_++;
    kf.pose = poseToMatrix(current_pose_);
    kf.timestamp = 0.0;
    kf.point_cloud.clear();  // No scan yet
    keyframes_.push_back(kf);

    std::cout << "[SLAM] Initialized at pose ("
              << current_pose_.x << ", "
              << current_pose_.y << ", "
              << current_pose_.z << ") with keyframe "
              << kf.frame_id << std::endl;
}

// ---------------------------------------------------------------------------
// Update with new scan
// ---------------------------------------------------------------------------

Pose3D SLAMSystem::update(const std::vector<Point3D>& scan, double timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);

    // -----------------------------------------------------------------------
    // Case 1: No local map yet — this is the first scan
    // -----------------------------------------------------------------------
    if (local_map_.empty()) {
        // Transform scan to world frame using current pose and add to local map
        const Eigen::Matrix4d T_world = poseToMatrix(current_pose_);
        for (const auto& pt : scan) {
            local_map_.push_back(transformPoint(pt, T_world));
        }

        // Update the first keyframe's point cloud
        if (!keyframes_.empty()) {
            keyframes_.front().point_cloud = scan;
            keyframes_.front().timestamp = timestamp;
        }

        std::cout << "[SLAM] First scan added to local map ("
                  << scan.size() << " points)" << std::endl;
        return current_pose_;
    }

    // -----------------------------------------------------------------------
    // Case 2: Run ICP scan matching against the local map
    // -----------------------------------------------------------------------

    // Transform current scan to the estimated world frame for ICP
    const Eigen::Matrix4d T_current_est = poseToMatrix(current_pose_);
    std::vector<Point3D> scan_world;
    scan_world.reserve(scan.size());
    for (const auto& pt : scan) {
        scan_world.push_back(transformPoint(pt, T_current_est));
    }

    // Downsample both clouds for efficient ICP
    const size_t max_icp_points = 2000;
    std::vector<Point3D> scan_ds = downsampleCloud(scan_world, max_icp_points);
    std::vector<Point3D> map_ds = downsampleCloud(local_map_, max_icp_points);

    // Run ICP: align the (downsampled, world-frame) scan to the local map
    const Eigen::Matrix4d icp_transform = icpMatch(
        scan_ds, map_ds,
        icp_max_correspondence_dist_,
        icp_max_iterations_,
        ICP_TOLERANCE);

    // Update current pose by composing the ICP correction with the prior estimate
    const Eigen::Matrix4d T_corrected = icp_transform * T_current_est;
    current_pose_ = matrixToPose(T_corrected);

    // -----------------------------------------------------------------------
    // Case 3: Check if a new keyframe should be added
    // -----------------------------------------------------------------------
    if (shouldAddKeyframe()) {
        // Create new keyframe
        KeyFrame kf;
        kf.frame_id = next_frame_id_++;
        kf.pose = T_corrected;
        kf.timestamp = timestamp;
        kf.point_cloud = scan;

        // Add odometry edge between previous keyframe and new one
        if (!keyframes_.empty()) {
            const KeyFrame& prev_kf = keyframes_.back();
            const Eigen::Matrix4d relative_transform =
                prev_kf.pose.inverse() * T_corrected;

            // Compute information matrix based on ICP quality
            const Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6) * 100.0;

            pose_graph_edges_.push_back(
                std::make_tuple(prev_kf.frame_id, kf.frame_id,
                                relative_transform, info));
        }

        // Store keyframe
        keyframes_.push_back(kf);

        // Update local map: accumulate the new scan in world frame
        for (const auto& pt : scan_world) {
            local_map_.push_back(pt);
        }

        // Keep the local map size bounded (remove oldest points if too large)
        const size_t max_local_map_size = 100000;
        if (local_map_.size() > max_local_map_size) {
            const size_t excess = local_map_.size() - max_local_map_size;
            local_map_.erase(local_map_.begin(),
                             local_map_.begin() + static_cast<ptrdiff_t>(excess));
        }

        std::cout << "[SLAM] Keyframe " << kf.frame_id
                  << " added at ("
                  << current_pose_.x << ", "
                  << current_pose_.y << ", "
                  << current_pose_.z << "). Total keyframes: "
                  << keyframes_.size() << std::endl;
    }

    return current_pose_;
}

// ---------------------------------------------------------------------------
// Check if a new keyframe should be added
// ---------------------------------------------------------------------------

bool SLAMSystem::shouldAddKeyframe() const {
    if (keyframes_.empty()) {
        return true;  // Always add the first keyframe
    }

    const KeyFrame& last_kf = keyframes_.back();

    // Compute relative transform from last keyframe to current pose
    const Eigen::Matrix4d T_current = poseToMatrix(current_pose_);
    const Eigen::Matrix4d T_relative = last_kf.pose.inverse() * T_current;

    // Translation distance
    const double dx = T_relative(0, 3);
    const double dy = T_relative(1, 3);
    const double dz = T_relative(2, 3);
    const double translation = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Rotation angle
    const double angle = rotationAngle(T_relative);

    return (translation > keyframe_translation_thresh_ ||
            angle > keyframe_rotation_thresh_);
}

// ---------------------------------------------------------------------------
// Loop closure detection
// ---------------------------------------------------------------------------

std::optional<LoopClosureResult> SLAMSystem::detectLoopClosure() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (keyframes_.size() < 3) {
        return std::nullopt;  // Not enough keyframes for loop closure
    }

    const KeyFrame& current_kf = keyframes_.back();
    const Eigen::Vector3d current_pos(current_kf.pose(0, 3),
                                      current_kf.pose(1, 3),
                                      current_kf.pose(2, 3));

    // Iterate over all previous keyframes (skip the most recent ones to
    // avoid matching against keyframes that are too close in time)
    const size_t min_gap = 10;  // Minimum number of keyframes between matches

    for (size_t i = 0; i + min_gap < keyframes_.size(); ++i) {
        const KeyFrame& candidate_kf = keyframes_[i];
        const Eigen::Vector3d candidate_pos(candidate_kf.pose(0, 3),
                                            candidate_kf.pose(1, 3),
                                            candidate_kf.pose(2, 3));

        // Check distance between keyframes
        const double dist = (current_pos - candidate_pos).norm();
        if (dist > LOOP_CLOSURE_RADIUS || dist < LOOP_CLOSURE_MIN_TRANSLATION) {
            continue;
        }

        // Try ICP matching between the two keyframes' point clouds
        if (candidate_kf.point_cloud.empty() || current_kf.point_cloud.empty()) {
            continue;
        }

        // Transform current keyframe's cloud into candidate keyframe's frame
        const Eigen::Matrix4d T_relative =
            candidate_kf.pose.inverse() * current_kf.pose;

        std::vector<Point3D> scan_aligned;
        scan_aligned.reserve(current_kf.point_cloud.size());
        for (const auto& pt : current_kf.point_cloud) {
            scan_aligned.push_back(transformPoint(pt, T_relative));
        }

        // Downsample for ICP
        const size_t max_lc_points = 1000;
        std::vector<Point3D> scan_ds = downsampleCloud(scan_aligned, max_lc_points);
        std::vector<Point3D> tgt_ds = downsampleCloud(candidate_kf.point_cloud,
                                                       max_lc_points);

        // Run ICP
        const Eigen::Matrix4d icp_result = icpMatch(
            scan_ds, tgt_ds,
            icp_max_correspondence_dist_ * 2.0,  // Relaxed for loop closure
            icp_max_iterations_,
            ICP_TOLERANCE);

        // Evaluate ICP quality
        auto [distances, indices] = findNearestNeighbors(scan_ds, tgt_ds);
        int valid_count = 0;
        double mean_error = 0.0;
        for (size_t j = 0; j < distances.size(); ++j) {
            if (indices[j] >= 0 && distances[j] < icp_max_correspondence_dist_ * 2.0) {
                valid_count++;
                mean_error += distances[j];
            }
        }

        if (valid_count < ICP_MIN_CORRESPONDENCES) {
            continue;
        }
        mean_error /= static_cast<double>(valid_count);

        // Check if ICP converged with low error
        if (mean_error < LOOP_CLOSURE_MAX_ICP_ERROR) {
            // Compute the relative transform in world frame
            const Eigen::Matrix4d relative_transform_world =
                icp_result * T_relative;

            // Confidence is based on ICP fitness (low error + many correspondences = high confidence)
            const double fitness = static_cast<double>(valid_count) /
                                   static_cast<double>(scan_ds.size());
            const double confidence = std::min(
                fitness * (1.0 - mean_error / LOOP_CLOSURE_MAX_ICP_ERROR), 1.0);

            if (confidence > LOOP_CLOSURE_MIN_CONFIDENCE) {
                LoopClosureResult result;
                result.frame_id_a = candidate_kf.frame_id;
                result.frame_id_b = current_kf.frame_id;
                result.relative_transform = relative_transform_world;
                result.confidence = confidence;

                // Add loop closure edge to the pose graph
                const Eigen::MatrixXd info = computeInformationMatrix(
                    mean_error * mean_error, valid_count) * confidence;
                pose_graph_edges_.push_back(
                    std::make_tuple(result.frame_id_a, result.frame_id_b,
                                    result.relative_transform, info));

                std::cout << "[SLAM] Loop closure detected between keyframes "
                          << result.frame_id_a << " and " << result.frame_id_b
                          << " (confidence=" << confidence
                          << ", mean_error=" << mean_error << ")" << std::endl;

                return result;
            }
        }
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Pose graph optimization
// ---------------------------------------------------------------------------

void SLAMSystem::optimizePoseGraph() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (keyframes_.size() < 2 || pose_graph_edges_.empty()) {
        return;
    }

    std::cout << "[SLAM] Optimizing pose graph with "
              << keyframes_.size() << " keyframes and "
              << pose_graph_edges_.size() << " edges" << std::endl;

    // Build a map from frame_id to index in keyframes_ vector
    std::unordered_map<uint64_t, size_t> id_to_index;
    for (size_t i = 0; i < keyframes_.size(); ++i) {
        id_to_index[keyframes_[i].frame_id] = i;
    }

    // Number of pose variables: each keyframe has 6 DOF (tx, ty, tz, rx, ry, rz)
    const int n_poses = static_cast<int>(keyframes_.size());
    const int dim = 6;
    const int total_dim = n_poses * dim;

    // Current pose estimates as 6-vectors
    // Initialize from the keyframe poses
    std::vector<Eigen::VectorXd> pose_estimates(n_poses);
    for (int i = 0; i < n_poses; ++i) {
        const Pose3D p = matrixToPose(keyframes_[i].pose);
        pose_estimates[i] = Eigen::VectorXd(6);
        pose_estimates[i] << p.x, p.y, p.z, p.roll, p.pitch, p.yaw;
    }

    // Gauss-Newton iterations
    for (int iter = 0; iter < POSE_GRAPH_MAX_ITERATIONS; ++iter) {
        // Build the linear system: H * dx = b
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(total_dim, total_dim);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(total_dim);

        double total_error = 0.0;

        // Process each edge in the pose graph
        for (const auto& [id_a, id_b, rel_transform, info_matrix] : pose_graph_edges_) {
            auto it_a = id_to_index.find(id_a);
            auto it_b = id_to_index.find(id_b);
            if (it_a == id_to_index.end() || it_b == id_to_index.end()) {
                continue;
            }

            const int idx_a = static_cast<int>(it_a->second);
            const int idx_b = static_cast<int>(it_b->second);

            // Current estimates as SE(3) transforms
            const Eigen::Matrix4d T_a = poseToMatrix({
                pose_estimates[idx_a](0), pose_estimates[idx_a](1),
                pose_estimates[idx_a](2), pose_estimates[idx_a](3),
                pose_estimates[idx_a](4), pose_estimates[idx_a](5),
                {}, {}
            });
            const Eigen::Matrix4d T_b = poseToMatrix({
                pose_estimates[idx_b](0), pose_estimates[idx_b](1),
                pose_estimates[idx_b](2), pose_estimates[idx_b](3),
                pose_estimates[idx_b](4), pose_estimates[idx_b](5),
                {}, {}
            });

            // Measured relative transform: T_a^{-1} * T_b should equal rel_transform
            // Error: e = log(rel_transform^{-1} * T_a^{-1} * T_b)
            const Eigen::Matrix4d T_measured = rel_transform;
            const Eigen::Matrix4d T_predicted = T_a.inverse() * T_b;
            const Eigen::VectorXd error = se3Error(T_predicted, T_measured);

            total_error += error.squaredNorm();

            // Jacobians (simplified): for small perturbations,
            // J_a = -I (derivative of error w.r.t. pose_a)
            // J_b = +I (derivative of error w.r.t. pose_b)
            // In a full implementation, these would be the adjoint-based Jacobians
            const Eigen::MatrixXd J_a = -Eigen::MatrixXd::Identity(dim, dim);
            const Eigen::MatrixXd J_b = Eigen::MatrixXd::Identity(dim, dim);

            // Weighted contributions to the normal equations
            const Eigen::MatrixXd info_J_a = info_matrix * J_a;
            const Eigen::MatrixXd info_J_b = info_matrix * J_b;
            const Eigen::VectorXd info_error = info_matrix * error;

            // H += J' * info * J
            H.block(idx_a * dim, idx_a * dim, dim, dim) += J_a.transpose() * info_J_a;
            H.block(idx_a * dim, idx_b * dim, dim, dim) += J_a.transpose() * info_J_b;
            H.block(idx_b * dim, idx_a * dim, dim, dim) += J_b.transpose() * info_J_a;
            H.block(idx_b * dim, idx_b * dim, dim, dim) += J_b.transpose() * info_J_b;

            // b += J' * info * e
            b.segment(idx_a * dim, dim) += J_a.transpose() * info_error;
            b.segment(idx_b * dim, dim) += J_b.transpose() * info_error;
        }

        // Fix the first pose (anchor) to eliminate gauge freedom
        // Set the corresponding block of H to identity and b to zero
        H.block(0, 0, dim, dim) = Eigen::MatrixXd::Identity(dim, dim);
        b.segment(0, dim) = Eigen::VectorXd::Zero(dim);

        // Add damping for numerical stability (Levenberg-Marquardt style)
        H += Eigen::MatrixXd::Identity(total_dim, total_dim) * POSE_GRAPH_DAMPING;

        // Solve the linear system
        Eigen::VectorXd dx;
        if (total_dim > 0) {
            // Use LDLT decomposition for symmetric positive semi-definite system
            Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
            if (ldlt.info() == Eigen::Success) {
                dx = ldlt.solve(b);
            } else {
                // Fallback to QR decomposition
                dx = H.colPivHouseholderQr().solve(b);
            }
        } else {
            dx = Eigen::VectorXd::Zero(total_dim);
        }

        // Apply updates to pose estimates
        for (int i = 0; i < n_poses; ++i) {
            pose_estimates[i] += dx.segment(i * dim, dim);
        }

        // Check convergence
        const double update_norm = dx.norm();
        if (update_norm < 1e-8) {
            std::cout << "[SLAM] Pose graph optimization converged at iteration "
                      << iter << " (update_norm=" << update_norm << ")" << std::endl;
            break;
        }
    }

    // Update keyframe poses from the optimized estimates
    for (int i = 0; i < n_poses; ++i) {
        keyframes_[i].pose = poseToMatrix({
            pose_estimates[i](0), pose_estimates[i](1),
            pose_estimates[i](2), pose_estimates[i](3),
            pose_estimates[i](4), pose_estimates[i](5),
            {}, {}
        });
    }

    // Update current pose from the last keyframe
    if (!keyframes_.empty()) {
        current_pose_ = matrixToPose(keyframes_.back().pose);
    }

    // Rebuild local map from keyframe point clouds with updated poses
    local_map_.clear();
    for (const auto& kf : keyframes_) {
        for (const auto& pt : kf.point_cloud) {
            local_map_.push_back(transformPoint(pt, kf.pose));
        }
    }

    std::cout << "[SLAM] Pose graph optimization complete. "
              << keyframes_.size() << " keyframes updated." << std::endl;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::vector<KeyFrame> SLAMSystem::getKeyFrames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_;
}

Pose3D SLAMSystem::getCurrentPose() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_pose_;
}

}  // namespace avcs
