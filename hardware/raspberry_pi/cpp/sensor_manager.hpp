/**
 * File:        cpp/sensor_manager.hpp
 * Brief:       Declaration of SensorManager — owns the camera and
 *              arbitrary GPIO-based sensors, runs a poller thread,
 *              and exposes thread-safe accessors.
 * Author:      Autonomous Vehicle Team
 * Date:        2025-01-30
 * License:     MIT
 */

#ifndef AV_SENSOR_MANAGER_HPP
#define AV_SENSOR_MANAGER_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "camera.hpp"

namespace av {

// ----------------------------------------------------------------------
// Generic sensor interface
// ----------------------------------------------------------------------
class ISensor {
 public:
    virtual ~ISensor() = default;

    /// Human-readable name (e.g. "ultrasonic.front").
    virtual const std::string& name() const = 0;

    /// Initialize hardware and open handles.
    virtual void initialize() = 0;

    /// Poll the sensor once (called from the manager thread).
    /// Should be non-blocking or have a short timeout.
    virtual void poll() = 0;

    /// Release hardware resources.
    virtual void shutdown() = 0;

    /// True if the sensor is healthy and producing data.
    virtual bool healthy() const = 0;
};

// ----------------------------------------------------------------------
// Manager configuration
// ----------------------------------------------------------------------
struct SensorManagerConfig {
    int         rate_hz     = 30;
    std::string config_path = "config/av-01.yaml";
    bool        verbose     = false;
    bool        enable_camera = true;
};

// ----------------------------------------------------------------------
// Manager statistics
// ----------------------------------------------------------------------
struct SensorManagerStats {
    int      sensor_count       = 0;
    uint64_t total_samples      = 0;
    uint64_t total_errors       = 0;
    double   last_loop_dt_ms    = 0.0;
    double   actual_rate_hz     = 0.0;
};

// ----------------------------------------------------------------------
// SensorManager
// ----------------------------------------------------------------------
class SensorManager {
 public:
    explicit SensorManager(const SensorManagerConfig& config);
    ~SensorManager();

    SensorManager(const SensorManager&)            = delete;
    SensorManager& operator=(const SensorManager&) = delete;
    SensorManager(SensorManager&&)                 = delete;
    SensorManager& operator=(SensorManager&&)      = delete;

    /// Register an external sensor (e.g. an IMU driver). Manager takes
    /// ownership of the pointer.
    void register_sensor(std::unique_ptr<ISensor> sensor);

    /// Initialize the built-in camera (if enabled) and all registered
    /// sensors. Throws on failure.
    void initialize();

    /// Start the poller thread.
    void start();

    /// Stop the poller thread and release all resources.
    void shutdown();

    /// Snapshot of statistics (thread-safe).
    SensorManagerStats snapshot() const;

    /// Get a pointer to the camera (may be nullptr if disabled).
    Camera* camera() { return camera_.get(); }

    /// Subscribe to per-loop completion events.
    using LoopCallback = std::function<void(const SensorManagerStats&)>;
    void on_loop(LoopCallback cb);

 private:
    void poll_loop();
    void notify_loop(const SensorManagerStats& stats);

    SensorManagerConfig                    config_;
    std::unique_ptr<Camera>                camera_;
    std::vector<std::unique_ptr<ISensor>>  sensors_;
    std::map<std::string, ISensor*>        by_name_;

    // Threading
    std::thread                            poll_thread_;
    std::atomic<bool>                      running_{false};
    std::mutex                             stats_mutex_;
    SensorManagerStats                     stats_{};
    mutable std::mutex                     cb_mutex_;
    LoopCallback                           loop_callback_;

    // Notification (not strictly required, but useful for blocking
    // consumers waiting on the next poll cycle).
    std::mutex                             notify_mutex_;
    std::condition_variable                notify_cv_;
};

}  // namespace av

#endif  // AV_SENSOR_MANAGER_HPP
