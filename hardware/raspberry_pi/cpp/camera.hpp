/**
 * File:        cpp/camera.hpp
 * Brief:       Declaration of the Camera class — V4L2/libcamera-based
 *              frame grabber with mmap buffers that exposes frames as
 *              cv::Mat (or a simple struct if OpenCV is unavailable).
 * Author:      Autonomous Vehicle Team
 * Date:        2025-01-30
 * License:     MIT
 */

#ifndef AV_CAMERA_HPP
#define AV_CAMERA_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward-declare OpenCV's Mat so the header compiles with or without it.
namespace cv { class Mat; }

namespace av {

// ----------------------------------------------------------------------
// Camera configuration
// ----------------------------------------------------------------------
struct CameraConfig {
    std::string device     = "/dev/video0";
    int         width      = 640;
    int         height     = 480;
    int         fps        = 30;
    int         buffer_count = 4;        // mmap buffer count
    bool        use_libcamera = false;   // true on Pi OS Bookworm
    std::string pixel_format = "BGR3";   // V4L2 fourcc
    bool        verbose     = false;
};

// ----------------------------------------------------------------------
// Frame data
// ----------------------------------------------------------------------
struct Frame {
    std::vector<uint8_t> data;          // raw bytes (BGR or YUYV)
    int      width   = 0;
    int      height  = 0;
    int      channels = 3;
    uint64_t frame_id = 0;
    double   timestamp_s = 0.0;          // monotonic
    int      exposure_us = 0;
    float    gain = 0.0f;
};

// ----------------------------------------------------------------------
// Camera class (RAII)
// ----------------------------------------------------------------------
class Camera {
 public:
    /// Open the camera with the given configuration.
    explicit Camera(const CameraConfig& config);
    /// Release all resources.
    ~Camera();

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&)                 = delete;
    Camera& operator=(Camera&&)      = delete;

    /// Start the background capture thread.
    void start();
    /// Stop the background capture thread.
    void stop();

    /// Synchronously grab one frame (blocking up to ``timeout_ms``).
    /// Returns ``true`` on success.
    bool capture(Frame& out, int timeout_ms = 100);

    /// Return the most recent frame. Thread-safe.
    bool latest(Frame& out) const;

    /// Set V4L2 exposure time (microseconds). 0 = auto.
    void set_exposure(int us);
    /// Set analogue gain. 0 = auto.
    void set_gain(float gain);
    /// Set target frame rate.
    void set_fps(int fps);

    /// Save the latest frame as a PNG/JPG (uses OpenCV if available).
    bool save_snapshot(const std::string& path) const;

    /// Convert the latest frame to an OpenCV Mat (shallow copy).
    /// Returns ``nullptr`` if OpenCV is unavailable or no frame exists.
    std::shared_ptr<cv::Mat> latest_as_mat() const;

    // Statistics
    struct Stats {
        uint64_t frames_captured = 0;
        uint64_t frames_dropped  = 0;
        int      width           = 0;
        int      height          = 0;
        int      fps_target      = 0;
    };
    Stats stats() const;

 private:
    // V4L2 helpers
    bool open_v4l2();
    bool open_libcamera();
    void close_device();
    bool init_mmap();
    bool start_streaming();
    bool stop_streaming();
    bool dequeue_frame(Frame& out);
    bool enqueue_buffer(uint32_t index);

    void capture_loop();

    // Configuration
    CameraConfig config_;

    // V4L2 state
    int      fd_             = -1;
    std::vector<uint8_t>     buffer_data_;   // mmap'd
    std::vector<uint32_t>    buffer_lengths_;
    std::vector<void*>       buffer_ptrs_;
    uint32_t current_buffer_ = 0;

    // Latest frame (protected by mutex)
    mutable std::mutex       frame_mutex_;
    Frame                    latest_frame_;
    std::atomic<uint64_t>    frame_counter_{0};
    std::atomic<uint64_t>    drop_counter_{0};

    // Capture thread
    std::thread              capture_thread_;
    std::atomic<bool>        running_{false};
};

}  // namespace av

#endif  // AV_CAMERA_HPP
