/**
 * File:        cpp/camera.cpp
 * Brief:       Implementation of the Camera class. Uses V4L2 mmap
 *              buffers for capture and supports the Raspberry Pi
 *              Camera Module via the standard V4L2 driver.
 * Author:      Autonomous Vehicle Team
 * Date:        2025-01-30
 * License:     MIT
 */

#include "camera.hpp"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Linux V4L2 headers (compile on Pi with: apt install libv4l-dev)
#include <linux/videodev2.h>

// OpenCV (optional)
#ifdef AV_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

namespace av {

namespace {
constexpr int kDefaultTimeoutMs = 100;

/// Convert a four-character code string ("BGR3") to a V4L2 FourCC.
uint32_t fourcc(const std::string& s) {
    if (s.size() < 4) return V4L2_PIX_FMT_BGR24;
    return v4l2_fourcc(s[0], s[1], s[2], s[3]);
}

/// Get monotonic time in seconds.
double monotonic_now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/// Call ioctl() and retry on EINTR.
int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ::ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

}  // namespace

// ----------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------
Camera::Camera(const CameraConfig& config) : config_(config) {
    if (config_.use_libcamera) {
        if (!open_libcamera()) {
            std::cerr << "[camera] libcamera open failed; falling back to V4L2\n";
            config_.use_libcamera = false;
        }
    }
    if (fd_ < 0) {
        if (!open_v4l2()) {
            throw std::runtime_error("Failed to open camera device: "
                                     + config_.device);
        }
    }
}

Camera::~Camera() {
    stop();
    close_device();
}

// ----------------------------------------------------------------------
// Open V4L2 device
// ----------------------------------------------------------------------
bool Camera::open_v4l2() {
    fd_ = ::open(config_.device.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        std::cerr << "[camera] Cannot open " << config_.device
                  << ": " << std::strerror(errno) << "\n";
        return false;
    }

    // Check capabilities
    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "[camera] VIDIOC_QUERYCAP failed\n";
        return false;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        std::cerr << "[camera] Device does not support video capture\n";
        return false;
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        std::cerr << "[camera] Device does not support streaming\n";
        return false;
    }

    // Set format
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = config_.width;
    fmt.fmt.pix.height      = config_.height;
    fmt.fmt.pix.pixelformat = fourcc(config_.pixel_format);
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "[camera] VIDIOC_S_FMT failed\n";
        return false;
    }
    config_.width  = fmt.fmt.pix.width;
    config_.height = fmt.fmt.pix.height;

    // Set frame rate
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = config_.fps;
    if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        std::perror("[camera] VIDIOC_S_PARM (non-fatal)");
    }

    if (!init_mmap()) return false;
    if (config_.verbose) {
        std::cout << "[camera] Opened " << config_.device
                  << " " << config_.width << "x" << config_.height
                  << " @ " << config_.fps << " fps\n";
    }
    return true;
}

// ----------------------------------------------------------------------
// Stub for libcamera (full implementation would link libcamera)
// ----------------------------------------------------------------------
bool Camera::open_libcamera() {
    // TODO: integrate with libcamera C++ API.
    // For now, return false so we fall back to V4L2.
    return false;
}

// ----------------------------------------------------------------------
// mmap buffers
// ----------------------------------------------------------------------
bool Camera::init_mmap() {
    v4l2_requestbuffers req{};
    req.count  = config_.buffer_count;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "[camera] VIDIOC_REQBUFS failed\n";
        return false;
    }
    if (req.count < 2) {
        std::cerr << "[camera] Insufficient buffers (" << req.count << ")\n";
        return false;
    }

    buffer_ptrs_.resize(req.count);
    buffer_lengths_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            std::cerr << "[camera] VIDIOC_QUERYBUF failed for " << i << "\n";
            return false;
        }
        buffer_lengths_[i] = buf.length;
        buffer_ptrs_[i] = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_, buf.m.offset);
        if (buffer_ptrs_[i] == MAP_FAILED) {
            std::cerr << "[camera] mmap failed for " << i << "\n";
            return false;
        }
    }
    return true;
}

void Camera::close_device() {
    for (size_t i = 0; i < buffer_ptrs_.size(); ++i) {
        if (buffer_ptrs_[i] && buffer_ptrs_[i] != MAP_FAILED) {
            ::munmap(buffer_ptrs_[i], buffer_lengths_[i]);
        }
    }
    buffer_ptrs_.clear();
    buffer_lengths_.clear();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

// ----------------------------------------------------------------------
// Streaming
// ----------------------------------------------------------------------
bool Camera::start_streaming() {
    for (uint32_t i = 0; i < buffer_ptrs_.size(); ++i) {
        if (!enqueue_buffer(i)) return false;
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        std::perror("[camera] VIDIOC_STREAMON");
        return false;
    }
    return true;
}

bool Camera::stop_streaming() {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
        std::perror("[camera] VIDIOC_STREAMOFF (non-fatal)");
        return false;
    }
    return true;
}

bool Camera::enqueue_buffer(uint32_t index) {
    v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = index;
    return xioctl(fd_, VIDIOC_QBUF, &buf) >= 0;
}

bool Camera::dequeue_frame(Frame& out) {
    v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        return false;
    }
    out.frame_id     = frame_counter_.fetch_add(1) + 1;
    out.timestamp_s  = monotonic_now_s();
    out.width        = config_.width;
    out.height       = config_.height;
    out.channels     = 3;
    out.exposure_us  = 0;
    out.gain         = 0.0f;
    out.data.assign(
        static_cast<uint8_t*>(buffer_ptrs_[buf.index]),
        static_cast<uint8_t*>(buffer_ptrs_[buf.index]) + buf.bytesused
    );
    // Re-queue the buffer for the next capture
    if (!enqueue_buffer(buf.index)) {
        std::cerr << "[camera] Failed to re-queue buffer " << buf.index << "\n";
    }
    return true;
}

// ----------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------
void Camera::start() {
    if (running_.load()) return;
    if (!start_streaming()) {
        throw std::runtime_error("Failed to start camera streaming");
    }
    running_.store(true);
    capture_thread_ = std::thread(&Camera::capture_loop, this);
}

void Camera::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    stop_streaming();
}

bool Camera::capture(Frame& out, int timeout_ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);
    timeval tv{};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int r = ::select(fd_ + 1, &fds, nullptr, nullptr, &tv);
    if (r <= 0) {
        drop_counter_.fetch_add(1);
        return false;
    }
    return dequeue_frame(out);
}

bool Camera::latest(Frame& out) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (latest_frame_.data.empty()) return false;
    out = latest_frame_;
    return true;
}

void Camera::set_exposure(int us) {
    v4l2_control ctrl{};
    ctrl.id   = V4L2_CID_EXPOSURE_ABSOLUTE;
    ctrl.value = us;
    if (xioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
        std::perror("[camera] set_exposure (non-fatal)");
    }
}

void Camera::set_gain(float gain) {
    v4l2_control ctrl{};
    ctrl.id   = V4L2_CID_GAIN;
    ctrl.value = static_cast<int32_t>(gain * 1000);
    if (xioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
        std::perror("[camera] set_gain (non-fatal)");
    }
}

void Camera::set_fps(int fps) {
    config_.fps = fps;
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        std::perror("[camera] set_fps (non-fatal)");
    }
}

bool Camera::save_snapshot(const std::string& path) const {
#ifdef AV_HAVE_OPENCV
    auto mat = latest_as_mat();
    if (!mat) return false;
    return cv::imwrite(path, *mat);
#else
    (void)path;
    std::cerr << "[camera] save_snapshot requires OpenCV\n";
    return false;
#endif
}

std::shared_ptr<cv::Mat> Camera::latest_as_mat() const {
#ifdef AV_HAVE_OPENCV
    Frame f;
    if (!latest(f)) return nullptr;
    cv::Mat mat(f.height, f.width, CV_8UC(f.channels),
                const_cast<uint8_t*>(f.data.data()));
    // Make a deep copy so the caller owns the memory
    return std::make_shared<cv::Mat>(mat.clone());
#else
    return nullptr;
#endif
}

Camera::Stats Camera::stats() const {
    Stats s;
    s.frames_captured = frame_counter_.load();
    s.frames_dropped  = drop_counter_.load();
    s.width           = config_.width;
    s.height          = config_.height;
    s.fps_target      = config_.fps;
    return s;
}

// ----------------------------------------------------------------------
// Background capture loop
// ----------------------------------------------------------------------
void Camera::capture_loop() {
    while (running_.load()) {
        Frame f;
        if (capture(f, kDefaultTimeoutMs)) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = std::move(f);
        }
    }
}

}  // namespace av
