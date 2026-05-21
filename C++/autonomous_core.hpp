/**
 * @file autonomous_core.hpp
 * @brief Core orchestrator for the Autonomous Vehicle Control System (AVCS).
 *
 * This module defines the top-level AutonomousCore class that coordinates
 * all subsystems of the autonomous vehicle stack: sensor fusion, localization,
 * SLAM, trajectory tracking, and vehicle state estimation. It manages the
 * vehicle's operational mode, system lifecycle, and inter-subsystem
 * communication through callback-based update interfaces.
 *
 * Architecture overview:
 *
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │                    AutonomousCore                           │
 *   │                                                             │
 *   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
 *   │  │ SensorFusion │  │ EKFLocalizer │  │   SLAMSystem     │ │
 *   │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
 *   │         │                 │                    │            │
 *   │  ┌──────┴─────────────────┴────────────────────┴─────────┐ │
 *   │  │              Internal Message Bus                     │ │
 *   │  └──────┬─────────────────┬────────────────────┬─────────┘ │
 *   │         │                 │                    │            │
 *   │  ┌──────┴───────┐  ┌─────┴──────────┐                      │
 *   │  │ Trajectory   │  │ VehicleState   │                      │
 *   │  │ Tracker      │  │ Estimator      │                      │
 *   │  └──────────────┘  └────────────────┘                      │
 *   └─────────────────────────────────────────────────────────────┘
 *
 * The core runs a main loop that:
 *   1. Collects sensor data and feeds it to SensorFusion
 *   2. Uses fused data for localization (EKFLocalizer) and SLAM
 *   3. Feeds localization output to the trajectory tracker
 *   4. Produces control commands via TrajectoryTracker
 *   5. Updates the vehicle state estimator for monitoring
 *
 * Safety features:
 *   - Emergency stop capability that transitions to EMERGENCY mode
 *   - Fault-tolerant mode management with state machine enforcement
 *   - Thread-safe callbacks for asynchronous subsystem updates
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#ifndef AVCS_AUTONOMOUS_CORE_HPP_
#define AVCS_AUTONOMOUS_CORE_HPP_

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace avcs {

// Forward declarations of all subsystem classes and structs
class SensorFusion;
class EKFLocalizer;
class SLAMSystem;
class TrajectoryTracker;
class VehicleStateEstimator;

struct FusedState;
struct Pose3D;
struct TrajectoryPoint;

/**
 * @brief Autonomous driving mode enumeration.
 *
 * Defines the operational modes of the autonomous vehicle. Mode
 * transitions are enforced by the AutonomousCore state machine to
 * prevent unsafe transitions (e.g., switching to AUTONOMOUS before
 * the system is READY).
 */
enum class AutonomousMode {
    MANUAL,           ///< Human driver has full control; system monitors only
    AUTONOMOUS,       ///< Full autonomous driving; system has complete control
    SEMI_AUTONOMOUS,  ///< Shared control; system assists the human driver
    EMERGENCY,        ///< Emergency override; system brings vehicle to safe stop
    PARKING           ///< Autonomous parking mode; low-speed maneuvering
};

/**
 * @brief System state enumeration for the AVCS lifecycle.
 *
 * Represents the high-level state of the system. The state machine
 * enforces a strict transition order: INITIALIZING → READY → DRIVING,
 * with emergency transitions to EMERGENCY_STOP or FAULT from any state.
 */
enum class SystemState {
    INITIALIZING,    ///< System is booting up; subsystems being configured
    READY,           ///< All subsystems initialized; vehicle ready to drive
    DRIVING,         ///< Vehicle is actively driving in autonomous mode
    EMERGENCY_STOP,  ///< Emergency stop activated; vehicle decelerating to halt
    FAULT,           ///< Critical fault detected; system in fail-safe mode
    SHUTDOWN         ///< Graceful shutdown in progress
};

/**
 * @brief System-wide configuration parameters.
 *
 * Aggregates all configuration parameters needed by the AutonomousCore
 * and its subsystems. Parameters are typically loaded from a JSON or
 * YAML configuration file at startup.
 */
struct SystemConfig {
    // Sensor fusion parameters
    std::string fusion_mode;              ///< "sequential" or "batch"
    double sync_threshold;                ///< Sensor sync threshold (seconds)

    // Localization parameters
    double process_noise_pos;             ///< Position process noise (meters)
    double process_noise_ori;             ///< Orientation process noise (radians)

    // SLAM parameters
    double icp_max_correspondence_dist;   ///< ICP correspondence distance (meters)
    double icp_max_iterations;            ///< Max ICP iterations
    double keyframe_translation_thresh;   ///< Keyframe translation threshold (meters)
    double keyframe_rotation_thresh;      ///< Keyframe rotation threshold (radians)

    // Trajectory tracking parameters
    double kp_lateral;                    ///< Lateral proportional gain
    double kd_lateral;                    ///< Lateral derivative gain
    double kp_longitudinal;               ///< Longitudinal proportional gain
    double ki_longitudinal;               ///< Longitudinal integral gain

    // Vehicle parameters
    double wheelbase;                     ///< Vehicle wheelbase (meters)
    double mass;                          ///< Vehicle mass (kilograms)
    double inertia;                       ///< Yaw moment of inertia (kg·m²)

    // System parameters
    double control_loop_rate;             ///< Main control loop frequency (Hz)
    double max_speed;                     ///< Maximum allowed speed (m/s)
    double emergency_decel;               ///< Emergency deceleration (m/s²)
    std::string log_level;                ///< Logging verbosity: "debug", "info", "warn", "error"
};

/**
 * @brief Core orchestrator for the Autonomous Vehicle Control System.
 *
 * The AutonomousCore class is the top-level manager that:
 *   - Owns and configures all subsystem instances
 *   - Manages the system lifecycle (initialize → run → shutdown)
 *   - Enforces mode and state transitions
 *   - Routes data between subsystems via callback interfaces
 *   - Runs the main control loop in a dedicated thread
 *   - Handles emergency stops and fault conditions
 *
 * Thread safety:
 *   All public methods are thread-safe. The main control loop runs
 *   on an internal thread. Subsystem updates may arrive from external
 *   threads (e.g., sensor drivers) and are synchronized via mutex
 *   and condition variables.
 *
 * Usage example:
 * @code
 *   avcs::AutonomousCore core("/etc/avcs/config.json");
 *   if (!core.initialize()) {
 *       // handle initialization failure
 *   }
 *   core.setMode(avcs::AutonomousMode::AUTONOMOUS);
 *   core.run();
 *   // ... later ...
 *   core.shutdown();
 * @endcode
 */
class AutonomousCore {
public:
    /**
     * @brief Construct the AutonomousCore with a configuration file path.
     *
     * The configuration file is loaded during initialize(). The constructor
     * only stores the path and sets the initial system state to INITIALIZING.
     *
     * @param config_path  Path to the JSON/YAML configuration file.
     */
    explicit AutonomousCore(const std::string& config_path);

    /**
     * @brief Destructor — ensures graceful shutdown if not already done.
     */
    ~AutonomousCore();

    // Non-copyable
    AutonomousCore(const AutonomousCore&) = delete;
    AutonomousCore& operator=(const AutonomousCore&) = delete;

    // Non-movable (owns threads and mutexes)
    AutonomousCore(AutonomousCore&&) = delete;
    AutonomousCore& operator=(AutonomousCore&&) = delete;

    /**
     * @brief Initialize all subsystems and prepare for operation.
     *
     * Loads the configuration file, creates subsystem instances with
     * the configured parameters, and performs self-tests. Must be
     * called before run().
     *
     * @return true if initialization succeeded, false on any failure.
     */
    bool initialize();

    /**
     * @brief Start the main control loop.
     *
     * Spawns the control loop thread and begins autonomous operation.
     * The method blocks until shutdown() is called from another thread
     * or an unrecoverable fault occurs.
     */
    void run();

    /**
     * @brief Gracefully shut down the system.
     *
     * Signals the control loop to stop, waits for the thread to join,
     * and releases all subsystem resources. After this call, the
     * system state is SHUTDOWN.
     */
    void shutdown();

    /**
     * @brief Set the autonomous driving mode.
     *
     * Requests a mode transition. The transition may be denied if it
     * violates the state machine constraints (e.g., switching to
     * AUTONOMOUS while in FAULT state).
     *
     * @param mode  The requested autonomous mode.
     */
    void setMode(AutonomousMode mode);

    /**
     * @brief Get the current autonomous driving mode.
     *
     * @return The current AutonomousMode.
     */
    AutonomousMode getMode() const;

    /**
     * @brief Get the current system state.
     *
     * @return The current SystemState.
     */
    SystemState getSystemState() const;

    /**
     * @brief Trigger an emergency stop.
     *
     * Immediately transitions the system to EMERGENCY mode and
     * EMERGENCY_STOP state. The vehicle will decelerate to a halt
     * using maximum braking. This operation cannot be overridden
     * by setMode() — only a full reset can clear the emergency state.
     */
    void emergencyStop();

    /**
     * @brief Callback for perception subsystem updates.
     *
     * Called by the perception pipeline when new fused sensor data
     * (detected objects, obstacles, etc.) is available. The data is
     * forwarded to the trajectory tracker and SLAM system as needed.
     *
     * @param objects  The fused perception state.
     */
    void onPerceptionUpdate(const FusedState& objects);

    /**
     * @brief Callback for localization subsystem updates.
     *
     * Called when the localizer produces a new pose estimate. The
     * pose is forwarded to the trajectory tracker for control
     * computation and to the SLAM system for map alignment.
     *
     * @param pose  The updated vehicle pose.
     */
    void onLocalizationUpdate(const Pose3D& pose);

    /**
     * @brief Callback for planning subsystem updates.
     *
     * Called when the planning module generates a new reference
     * trajectory. The trajectory is stored and used by the
     * trajectory tracker for control computation.
     *
     * @param trajectory  The planned trajectory as a sequence of TrajectoryPoints.
     */
    void onPlanningUpdate(const std::vector<TrajectoryPoint>& trajectory);

private:
    /**
     * @brief Main control loop executed on a dedicated thread.
     *
     * At each iteration:
     *   1. Checks for emergency conditions
     *   2. Collects latest sensor data from fusion
     *   3. Updates localization and SLAM
     *   4. Runs trajectory tracking to compute control commands
     *   5. Sends commands to the vehicle interface
     *   6. Updates the vehicle state estimator
     */
    void controlLoop();

    // ─── Subsystem smart pointers ──────────────────────────────────

    /**
     * @brief Multi-sensor fusion engine.
     */
    std::unique_ptr<SensorFusion> sensor_fusion_;

    /**
     * @brief EKF-based localizer.
     */
    std::unique_ptr<EKFLocalizer> localizer_;

    /**
     * @brief Graph-based SLAM system.
     */
    std::unique_ptr<SLAMSystem> slam_system_;

    /**
     * @brief Trajectory tracking controller.
     */
    std::unique_ptr<TrajectoryTracker> trajectory_tracker_;

    /**
     * @brief Vehicle state estimator.
     */
    std::unique_ptr<VehicleStateEstimator> state_estimator_;

    // ─── State and mode ────────────────────────────────────────────

    /**
     * @brief Current autonomous driving mode.
     */
    AutonomousMode mode_;

    /**
     * @brief Current system lifecycle state.
     */
    SystemState system_state_;

    /**
     * @brief System configuration parameters.
     */
    SystemConfig config_;

    /**
     * @brief Path to the configuration file.
     */
    std::string config_path_;

    // ─── Threading primitives ──────────────────────────────────────

    /**
     * @brief Main control loop thread.
     */
    std::thread control_thread_;

    /**
     * @brief Flag indicating whether the control loop should continue running.
     */
    bool running_;

    /**
     * @brief Mutex protecting all shared state (mode, system_state, etc.).
     */
    mutable std::mutex mutex_;

    /**
     * @brief Condition variable for waking the control loop on new data.
     */
    std::condition_variable cv_;

    // ─── Cached data from callbacks ────────────────────────────────

    /**
     * @brief Latest perception update data.
     */
    FusedState latest_perception_;

    /**
     * @brief Latest localization pose.
     */
    Pose3D latest_pose_;

    /**
     * @brief Latest planned trajectory.
     */
    std::vector<TrajectoryPoint> latest_trajectory_;
};

}  // namespace avcs

#endif  // AVCS_AUTONOMOUS_CORE_HPP_
