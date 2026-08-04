/**
 * File:        cpp/gpio_controller.cpp
 * Brief:       Implementation of GPIOController using libgpiod (v1 API).
 *              Provides RAII pin management, edge interrupts, and
 *              software PWM on any pin.
 * Author:      Autonomous Vehicle Team
 * Date:        2025-01-30
 * License:     MIT
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread gpio_controller.cpp -lgpiod
 *
 * On a Pi:
 *   sudo apt install libgpiod-dev gpiod
 */

#include "gpio_controller.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <gpiod.h>
#include <poll.h>
#include <unistd.h>

namespace av {

namespace {
using namespace std::chrono;
using Clock = steady_clock;

/// Look up the default chip name based on the board revision.
std::string default_chip_name() {
    // Pi 5 uses gpiochip4 (RP1); older boards use gpiochip0.
    for (const char* candidate : {"gpiochip0", "gpiochip4"}) {
        gpiod_chip* chip = gpiod_chip_open_by_name(candidate);
        if (chip) {
            gpiod_chip_close(chip);
            return candidate;
        }
    }
    return "gpiochip0";
}

}  // namespace

// ----------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------
GPIOController::GPIOController(const std::string& chip_name)
    : chip_name_(chip_name.empty() ? default_chip_name() : chip_name) {
    chip_ = gpiod_chip_open_by_name(chip_name_.c_str());
    if (!chip_) {
        throw std::runtime_error("Cannot open GPIO chip '" + chip_name_
                                 + "': " + std::strerror(errno));
    }
    // Start the IRQ listener thread (it will wait on lines once they
    // have interrupts configured).
    irq_running_.store(true);
    irq_thread_ = std::thread(&GPIOController::irq_loop, this);
}

GPIOController::~GPIOController() {
    // Stop PWM
    {
        std::lock_guard<std::mutex> lock(pwm_mutex_);
        for (auto& [pin, state] : pwms_) {
            state->running.store(false);
            if (state->thread.joinable()) state->thread.join();
        }
        pwms_.clear();
    }

    // Stop IRQ listener
    irq_running_.store(false);
    if (irq_thread_.joinable()) irq_thread_.join();

    // Release all claimed lines
    std::vector<int> pins;
    {
        std::lock_guard<std::mutex> lock(lines_mutex_);
        for (auto& [pin, _] : lines_) pins.push_back(pin);
    }
    for (int pin : pins) release(pin);

    if (chip_) {
        gpiod_chip_close(chip_);
        chip_ = nullptr;
    }
}

// ----------------------------------------------------------------------
// Claim / release
// ----------------------------------------------------------------------
bool GPIOController::request_line(int pin, PinDirection dir, PinBias bias,
                                   bool initial_value) {
    gpiod_line* line = gpiod_chip_get_line(chip_, pin);
    if (!line) {
        std::cerr << "[gpio] Cannot get line " << pin << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }
    int rv = 0;
    if (dir == PinDirection::Output) {
        rv = gpiod_line_request_output(line, "av_gpio",
                                        initial_value ? 1 : 0);
    } else {
        int flags = 0;
        if (bias == PinBias::PullUp)   flags |= GPIOD_LINE_REQUEST_FLAG_BIAS_PULL_UP;
        if (bias == PinBias::PullDown) flags |= GPIOD_LINE_REQUEST_FLAG_BIAS_PULL_DOWN;
        gpiod_line_request_config cfg{};
        cfg.consumer = "av_gpio";
        cfg.request_type = GPIOD_LINE_REQUEST_DIRECTION_INPUT;
        cfg.flags = flags;
        rv = gpiod_line_request(line, &cfg, 0);
    }
    if (rv < 0) {
        std::cerr << "[gpio] Request failed for line " << pin << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }
    std::lock_guard<std::mutex> lock(lines_mutex_);
    lines_[pin] = LineState{line, dir, false};
    return true;
}

bool GPIOController::claim(int pin, PinDirection dir, PinBias bias,
                           bool initial_value) {
    std::lock_guard<std::mutex> lock(lines_mutex_);
    if (lines_.count(pin)) {
        std::cerr << "[gpio] Pin " << pin << " already claimed\n";
        return false;
    }
    return request_line(pin, dir, bias, initial_value);
}

void GPIOController::release_line(int pin) {
    auto it = lines_.find(pin);
    if (it == lines_.end()) return;
    gpiod_line_release(it->second.line);
    gpiod_line_close(it->second.line);  // noop in libgpiod v1
    lines_.erase(it);
}

void GPIOController::release(int pin) {
    // Stop PWM if running
    stop_pwm(pin);

    // Disable interrupts
    set_interrupt(pin, EdgeTrigger::None, nullptr);

    std::lock_guard<std::mutex> lock(lines_mutex_);
    release_line(pin);
}

// ----------------------------------------------------------------------
// Read / write
// ----------------------------------------------------------------------
void GPIOController::write(int pin, bool value) {
    std::lock_guard<std::mutex> lock(lines_mutex_);
    auto it = lines_.find(pin);
    if (it == lines_.end()) {
        throw std::runtime_error("write() on unclaimed pin " + std::to_string(pin));
    }
    if (it->second.direction != PinDirection::Output) {
        throw std::runtime_error("write() on input pin " + std::to_string(pin));
    }
    if (gpiod_line_set_value(it->second.line, value ? 1 : 0) < 0) {
        std::cerr << "[gpio] write failed on " << pin << ": "
                  << std::strerror(errno) << "\n";
    }
}

bool GPIOController::read(int pin) const {
    std::lock_guard<std::mutex> lock(lines_mutex_);
    auto it = lines_.find(pin);
    if (it == lines_.end()) {
        throw std::runtime_error("read() on unclaimed pin " + std::to_string(pin));
    }
    int v = gpiod_line_get_value(it->second.line);
    if (v < 0) {
        std::cerr << "[gpio] read failed on " << pin << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }
    return v != 0;
}

// ----------------------------------------------------------------------
// Interrupts
// ----------------------------------------------------------------------
bool GPIOController::set_interrupt(int pin, EdgeTrigger edge,
                                    InterruptCallback cb) {
    std::lock_guard<std::mutex> lock(lines_mutex_);
    auto it = lines_.find(pin);
    if (it == lines_.end()) {
        std::cerr << "[gpio] set_interrupt on unclaimed pin " << pin << "\n";
        return false;
    }
    if (it->second.has_interrupt) {
        gpiod_line_release(it->second.line);
        it->second.has_interrupt = false;
    }
    if (edge == EdgeTrigger::None) {
        std::lock_guard<std::mutex> cblock(callbacks_mutex_);
        callbacks_.erase(pin);
        return true;
    }
    // Re-request as input with edge events
    int flags = 0;
    int req_type = GPIOD_LINE_REQUEST_EVENT_BOTH_EDGES;
    switch (edge) {
        case EdgeTrigger::Rising:  req_type = GPIOD_LINE_REQUEST_EVENT_RISING_EDGE;  break;
        case EdgeTrigger::Falling: req_type = GPIOD_LINE_REQUEST_EVENT_FALLING_EDGE; break;
        case EdgeTrigger::Both:    req_type = GPIOD_LINE_REQUEST_EVENT_BOTH_EDGES;   break;
        default: break;
    }
    gpiod_line_request_config cfg{};
    cfg.consumer = "av_gpio_irq";
    cfg.request_type = req_type;
    cfg.flags = flags;
    if (gpiod_line_request(it->second.line, &cfg, 0) < 0) {
        std::cerr << "[gpio] IRQ request failed on " << pin << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }
    it->second.has_interrupt = true;
    {
        std::lock_guard<std::mutex> cblock(callbacks_mutex_);
        if (cb) callbacks_[pin] = std::move(cb);
        else     callbacks_.erase(pin);
    }
    return true;
}

void GPIOController::irq_loop() {
    // Poll all registered event lines every 5 ms
    while (irq_running_.load()) {
        std::vector<gpiod_line*> ev_lines;
        std::vector<int>         pins;
        {
            std::lock_guard<std::mutex> lock(lines_mutex_);
            for (auto& [pin, state] : lines_) {
                if (state.has_interrupt) {
                    ev_lines.push_back(state.line);
                    pins.push_back(pin);
                }
            }
        }
        if (ev_lines.empty()) {
            std::this_thread::sleep_for(milliseconds(5));
            continue;
        }
        // Use gpiod_line_event_wait with a 50 ms timeout
        timespec ts{};
        ts.tv_sec  = 0;
        ts.tv_nsec = 50 * 1000 * 1000;
        for (size_t i = 0; i < ev_lines.size(); ++i) {
            gpiod_line_event event{};
            if (gpiod_line_event_read(ev_lines[i], &event) == 0) {
                bool value = (event.event_type == GPIOD_LINE_EVENT_RISING_EDGE);
                InterruptCallback cb;
                {
                    std::lock_guard<std::mutex> cblock(callbacks_mutex_);
                    auto it = callbacks_.find(pins[i]);
                    if (it != callbacks_.end()) cb = it->second;
                }
                if (cb) {
                    try {
                        cb(pins[i], value);
                    } catch (const std::exception& exc) {
                        std::cerr << "[gpio] IRQ callback error on "
                                  << pins[i] << ": " << exc.what() << "\n";
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// Software PWM
// ----------------------------------------------------------------------
void GPIOController::start_pwm(int pin, int freq_hz, double duty) {
    {
        std::lock_guard<std::mutex> lock(lines_mutex_);
        auto it = lines_.find(pin);
        if (it == lines_.end() ||
            it->second.direction != PinDirection::Output) {
            throw std::runtime_error("start_pwm: pin " + std::to_string(pin)
                                     + " not claimed as output");
        }
    }
    stop_pwm(pin);
    auto state = std::make_unique<PwmState>();
    state->freq_hz.store(freq_hz > 0 ? freq_hz : 1000);
    state->duty.store(duty < 0 ? 0.0 : (duty > 1 ? 1.0 : duty));
    state->running.store(true);
    auto* raw = state.get();
    std::thread t(&GPIOController::pwm_loop, this, pin);
    state->thread = std::move(t);
    std::lock_guard<std::mutex> lock(pwm_mutex_);
    pwms_[pin] = std::move(state);
    (void)raw;
}

void GPIOController::set_pwm_duty(int pin, double duty) {
    std::lock_guard<std::mutex> lock(pwm_mutex_);
    auto it = pwms_.find(pin);
    if (it == pwms_.end()) return;
    it->second->duty.store(duty < 0 ? 0.0 : (duty > 1 ? 1.0 : duty));
}

void GPIOController::set_pwm_freq(int pin, int freq_hz) {
    std::lock_guard<std::mutex> lock(pwm_mutex_);
    auto it = pwms_.find(pin);
    if (it == pwms_.end()) return;
    it->second->freq_hz.store(freq_hz > 0 ? freq_hz : 1000);
}

void GPIOController::stop_pwm(int pin) {
    std::unique_ptr<PwmState> state;
    {
        std::lock_guard<std::mutex> lock(pwm_mutex_);
        auto it = pwms_.find(pin);
        if (it == pwms_.end()) return;
        state = std::move(it->second);
        pwms_.erase(it);
    }
    state->running.store(false);
    if (state->thread.joinable()) state->thread.join();
    // Leave the pin low
    try { write(pin, false); } catch (...) { /* ignore */ }
}

void GPIOController::pwm_loop(int pin) {
    std::unique_lock<std::mutex> lock(pwm_mutex_, std::defer_lock);
    while (true) {
        int   freq;
        double duty;
        {
            std::lock_guard<std::mutex> g(pwm_mutex_);
            auto it = pwms_.find(pin);
            if (it == pwms_.end() || !it->second->running.load()) break;
            freq = it->second->freq_hz.load();
            duty = it->second->duty.load();
        }
        if (freq <= 0) freq = 1000;
        if (duty < 0)  duty = 0;
        if (duty > 1)  duty = 1;

        auto period_us = static_cast<int>(1'000'000 / freq);
        auto on_us  = static_cast<int>(period_us * duty);
        auto off_us = period_us - on_us;

        if (on_us > 0) {
            write(pin, true);
            std::this_thread::sleep_for(microseconds(on_us));
        }
        if (off_us > 0) {
            write(pin, false);
            std::this_thread::sleep_for(microseconds(off_us));
        }
    }
    // Pin left in last state; release()/stop_pwm() will reset to low.
}

// ----------------------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------------------
std::vector<int> GPIOController::claimed_pins() const {
    std::lock_guard<std::mutex> lock(lines_mutex_);
    std::vector<int> out;
    out.reserve(lines_.size());
    for (auto& [pin, _] : lines_) out.push_back(pin);
    return out;
}

}  // namespace av
