/**
 * @file slam_system.hpp
 * @brief Simultaneous Localization and Mapping (SLAM) system module.
 *
 * This module implements a graph-based SLAM system for autonomous vehicles
 * that incrementally builds a map of the environment while simultaneously
 * estimating the vehicle's pose within it. Key components include:
 *
 *   - **ICP Scan Matching**: Aligns incoming LIDAR scans against the local
 *     map to estimate incremental motion.
 *   - **Keyframe Management**: Selectively stores keyframes based on motion
 *     thresholds to build a sparse but informative pose graph.
 *   - **Loop Closure Detection**: Identifies previously visited locations
 *     to correct accumulated drift.
 *   - **Pose Graph Optimization**: Uses nonlinear least-squares optimization
 *     (Gauss-Newton / Levenberg-Marquardt) to globally refine all keyframe
 *     poses once a loop closure is detected.
 *
 * The system is designed for real-time operation with LIDAR sensors
 * producing 3D point clouds at 10–20 Hz.
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#ifndef AVCS_SLAM_SYSTEM_HPP_
#define AVCS_SLAM_SYSTEM_HPP_

#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace avcs {

// Forward declaration to avoid circular dependency; Pose3D is defined in localization.hpp
struct Pose3D;

/**
 * @brief 3D point with intensity for LIDAR point cloud representation.
 *
 * Each point stores its Cartesian coordinates (x, y, z) in the sensor
 * frame and an intensity value that reflects the return signal strength.
 * Intensity can be useful for feature matching and loop closure detection.
 */
struct Point3D {
    double x;           ///< X coordinate (meters)
    double y;           ///< Y coordinate (meters)
    double z;           ///< Z coordinate (meters)
    double intensity;   ///< Return signal intensity (0.0–1.0 normalized)
};

/**
 * @brief A keyframe in the SLAM pose graph.
 *
 * A keyframe is a snapshot of the vehicle's state at a significant pose
 * change. It stores the 4×4 homogeneous transform representing the
 * keyframe's pose in the world frame, the associated point cloud, and
 * a unique frame identifier for indexing in the pose graph.
 */
struct KeyFrame {
    uint64_t frame_id;                          ///< Unique identifier for this keyframe
    Eigen::Matrix4d pose;                       ///< 4×4 homogeneous pose in world frame
    double timestamp;                           ///< Timestamp when the keyframe was created (seconds)
    std::vector<Point3D> point_cloud;           ///< Associated 3D point cloud in sensor frame
};

/**
 * @brief Result of a loop closure detection attempt.
 *
 * Contains the pair of keyframe IDs that were matched, the relative
 * transformation between them, and a confidence score indicating the
 * quality of the match. A high confidence (> 0.8) typically indicates
 * a reliable loop closure that can be added to the pose graph.
 */
struct LoopClosureResult {
    uint64_t frame_id_a;                ///< ID of the first keyframe
    uint64_t frame_id_b;                ///< ID of the second keyframe
    Eigen::Matrix4d relative_transform; ///< Relative transform from frame A to frame B
    double confidence;                  ///< Match confidence score [0.0, 1.0]
};

/**
 * @brief Graph-based SLAM system with ICP scan matching and loop closure.
 *
 * The SLAMSystem class provides a complete SLAM pipeline:
 *   1. **Initialization**: Sets the origin pose and builds the first keyframe.
 *   2. **Update**: Receives a new point cloud, performs ICP scan matching
 *      against the local map, estimates the current pose, and decides
 *      whether to create a new keyframe.
 *   3. **Loop Closure**: Periodically searches for loop closures between
 *      the current keyframe and past keyframes using feature descriptors.
 *   4. **Optimization**: When a loop closure is confirmed, optimizes the
 *      entire pose graph to distribute the correction globally.
 *
 * Thread safety:
 *   All public methods are thread-safe. The update and optimization
 *   operations are protected by an internal mutex.
 *
 * Usage example:
 * @code
 *   avcs::SLAMSystem slam(0.1, 0.5, 5.0);
 *   slam.initialize(origin_pose);
 *   auto pose = slam.update(scan_points, timestamp);
 *   if (auto loop = slam.detectLoopClosure()) {
 *       slam.optimizePoseGraph();
 *   }
 * @endcode
 */
class SLAMSystem {
public:
    /**
     * @brief Construct the SLAM system with ICP and keyframe parameters.
     *
     * @param icp_max_correspondence_dist  Maximum point-to-point distance (meters)
     *                                     for ICP correspondence matching.
     * @param icp_max_iterations           Maximum number of ICP iterations per scan.
     * @param keyframe_translation_thresh  Minimum translation (meters) between keyframes.
     * @param keyframe_rotation_thresh     Minimum rotation (radians) between keyframes.
     */
    SLAMSystem(double icp_max_correspondence_dist = 0.1,
               double icp_max_iterations = 50.0,
               double keyframe_translation_thresh = 0.5,
               double keyframe_rotation_thresh = 0.1);

    /**
     * @brief Destructor.
     */
    ~SLAMSystem() = default;

    // Non-copyable
    SLAMSystem(const SLAMSystem&) = delete;
    SLAMSystem& operator=(const SLAMSystem&) = delete;

    // Movable
    SLAMSystem(SLAMSystem&&) = default;
    SLAMSystem& operator=(SLAMSystem&&) = default;

    /**
     * @brief Initialize the SLAM system with a known starting pose.
     *
     * Creates the first keyframe at the given pose and initializes the
     * local map. Must be called before update().
     *
     * @param initial_pose  The starting pose of the vehicle.
     */
    void initialize(const Pose3D& initial_pose);

    /**
     * @brief Process a new LIDAR scan and estimate the current pose.
     *
     * Performs ICP scan matching between the incoming scan and the local
     * map to compute the incremental motion. The current pose is updated
     * and a new keyframe is created if the motion thresholds are exceeded.
     *
     * @param scan       Vector of 3D points from the LIDAR scan (sensor frame).
     * @param timestamp  Timestamp of the scan in seconds (epoch).
     * @return The estimated current pose after scan matching.
     */
    Pose3D update(const std::vector<Point3D>& scan, double timestamp);

    /**
     * @brief Attempt to detect a loop closure.
     *
     * Compares the most recent keyframe against all previous keyframes
     * (beyond a minimum temporal gap) using feature-based matching.
     * If a match is found, returns the loop closure result; otherwise
     * returns std::nullopt.
     *
     * @return LoopClosureResult if a loop closure is detected, std::nullopt otherwise.
     */
    std::optional<LoopClosureResult> detectLoopClosure();

    /**
     * @brief Optimize the pose graph using all current constraints.
     *
     * Runs a global nonlinear least-squares optimization over the entire
     * pose graph, using odometry edges and loop closure edges as constraints.
     * After optimization, all keyframe poses and the local map are updated.
     */
    void optimizePoseGraph();

    /**
     * @brief Retrieve all keyframes in the pose graph.
     *
     * @return Const reference to the vector of keyframes.
     */
    std::vector<KeyFrame> getKeyFrames() const;

    /**
     * @brief Get the current estimated pose of the vehicle.
     *
     * @return The current pose estimate.
     */
    Pose3D getCurrentPose() const;

private:
    /**
     * @brief List of all keyframes in the pose graph.
     */
    std::vector<KeyFrame> keyframes_;

    /**
     * @brief Edges in the pose graph representing relative transforms
     *        between consecutive keyframes and loop closures.
     *
     * Each edge is a tuple: (frame_id_a, frame_id_b, relative_transform, information_matrix).
     */
    std::vector<std::tuple<uint64_t, uint64_t, Eigen::Matrix4d, Eigen::MatrixXd>> pose_graph_edges_;

    /**
     * @brief Accumulated local map as a point cloud in world frame.
     */
    std::vector<Point3D> local_map_;

    /**
     * @brief ICP scan matcher parameters.
     */
    double icp_max_correspondence_dist_;
    double icp_max_iterations_;

    /**
     * @brief Keyframe creation thresholds.
     */
    double keyframe_translation_thresh_;
    double keyframe_rotation_thresh_;

    /**
     * @brief Current estimated vehicle pose.
     */
    Pose3D current_pose_;

    /**
     * @brief Next unique keyframe ID.
     */
    uint64_t next_frame_id_;

    /**
     * @brief Mutex for thread-safe access to SLAM state.
     */
    mutable std::mutex mutex_;
};

}  // namespace avcs

#endif  // AVCS_SLAM_SYSTEM_HPP_
