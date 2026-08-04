/**
 * File:        cpp/sensor_manager.cpp
 * Brief:       Implementation of SensorManager. Owns the camera plus
 *              an arbitrary number of ISensor implementations, runs a
 *              fixed-rate poller thread, and exposes thread-safe
 *              stats and a notification CV.
 * Author:      Autonomous Vehicle Team
 * Date:        2025-01-30
 * License:     MIT
 */

#include "sensor_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace av {

namespace {
using namespace std::chrono;
using Clock = steady_clock;

double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return duration<double, std::milli>(t1 - t0).count();
}
}  // namespace

// ----------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------
SensorManager::SensorManager(const SensorManagerConfig& config)
    : config_(config) {}

SensorManager::~SensorManager() {
    shutdown();
}

void SensorManager::register_sensor(std::unique_ptr<ISensor> sensor) {
    if (!sensor) return;
    std::string name = sensor->name();
    if (by_name_.count(name)) {
        throw std::runtime_error("Duplicate sensor name: " + name);
    }
    if (config_.verbose) {
        std::cout << "[mgr] Registered sensor '" << name << "'\n";
    }
    by_name_[name] = sensor.get();
    sensors_.push_back(std::move(sensor));
}

void SensorManager::initialize() {
    if (config_.enable_camera) {
        CameraConfig cam_cfg;
        cam_cfg.device     = "/dev/video0";
        cam_cfg.width      = 640;
        cam_cfg.height     = 480;
        cam_cfg.fps        = config_.rate_hz > 0 ? config_.rate_hz : 30;
        cam_cfg.verbose    = config_.verbose;
        try {
            camera_ = std::make_unique<Camera>(cam_cfg);
            camera_->start();
            if (config_.verbose) {
                std::cout << "[mgr] Camera started\n";
            }
        } catch (const std::exception& exc) {
            std::cerr << "[mgr] Camera init failed: " << exc.what() << "\n";
            camera_.reset();
        }
    }

    for (auto& sensor : sensors_) {
        try {
            sensor->initialize();
        } catch (const std::exception& exc) {
            std::cerr << "[mgr] Sensor '" << sensor->name()
                      << "' init failed: " << exc.what() << "\n";
        }
    }
}

void SensorManager::start() {
    if (running_.load()) return;
    running_.store(true);
    poll_thread_ = std::thread(&SensorManager::poll_loop, this);
    if (config_.verbose) {
        std::cout << "[mgr] Poller thread started at "
                  << config_.rate_hz << " Hz\n";
    }
}

void SensorManager::shutdown() {
    if (!running_.load()) return;
    running_.store(false);
    notify_cv_.notify_all();
    if (poll_thread_.joinable()) {
        poll_thread_.join();
    }
    for (auto& sensor : sensors_) {
        try {
            sensor->shutdown();
        } catch (const std::exception& exc) {
            std::cerr << "[mgr] Sensor '" << sensor->name()
                      << "' shutdown failed: " << exc.what() << "\n";
        }
    }
    if (camera_) {
        camera_->stop();
        camera_.reset();
    }
    if (config_.verbose) {
        std::cout << "[mgr] Shutdown complete\n";
    }
}

// ----------------------------------------------------------------------
// Stats
// ----------------------------------------------------------------------
SensorManagerStats SensorManager::snapshot() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SensorManager::on_loop(LoopCallback cb) {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    loop_callback_ = std::move(cb);
}

void SensorManager::notify_loop(const SensorManagerStats& s) {
    {
        std::lock_guard<std::mutex> lock(cb_mutex_);
        if (loop_callback_) {
            loop_callback_(s);
        }
    }
    notify_cv_.notify_all();
}

// ----------------------------------------------------------------------
// Poller thread
// ----------------------------------------------------------------------
void SensorManager::poll_loop() {
    const double period_s = 1.0 / std::max(1, config_.rate_hz);
    const auto period_us = static_cast<int64_t>(period_s * 1'000'000);

    auto last_loop = Clock::now();
    uint64_t loop_count = 0;
    uint64_t error_count = 0;

    // Sliding window for actual rate computation
    std::vector<Clock::time_point> timestamps;
    timestamps.reserve(64);

    while (running_.load()) {
        auto loop_start = Clock::now();

        // Poll each sensor
        for (auto& sensor : sensors_) {
            if (!running_.load()) break;
            try {
                sensor->poll();
            } catch (const std::exception& exc) {
                ++error_count;
                std::cerr << "[mgr] Sensor '" << sensor->name()
                          << "' poll error: " << exc.what() << "\n";
            }
        }

        ++loop_count;

        // Update stats
        auto loop_end = Clock::now();
        double dt_ms = elapsed_ms(loop_start, loop_end);
        timestamps.push_back(loop_end);
        if (timestamps.size() > 60) timestamps.erase(timestamps.begin());

        double actual_hz = 0.0;
        if (timestamps.size() >= 2) {
            double window_s = duration<double>(
                timestamps.back() - timestamps.front()).count();
            if (window_s > 0) {
                actual_hz = (timestamps.size() - 1) / window_s;
            }
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.sensor_count    = static_cast<int>(sensors_.size());
            stats_.total_samples   = loop_count;
            stats_.total_errors    = error_count;
            stats_.last_loop_dt_ms = dt_ms;
            stats_.actual_rate_hz  = actual_hz;
        }

        notify_loop(stats_);

        // Sleep until next period
        auto elapsed_us = duration_cast<microseconds>(
            Clock::now() - loop_start).count();
        auto sleep_us = period_us - elapsed_us;
        if (sleep_us > 0) {
            std::this_thread::sleep_for(microseconds(sleep_us));
        } else if (config_.verbose && (loop_count % 100 == 0)) {
            std::cerr << "[mgr] Loop overrun by " << (-sleep_us) << " us\n";
        }
    }

    if (config_.verbose) {
        std::cout << "[mgr] Poller exiting after " << loop_count
                  << " loops (" << error_count << " errors)\n";
    }
}

}  // namespace av
