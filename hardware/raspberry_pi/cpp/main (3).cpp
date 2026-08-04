/**
 * File:        cpp/main.cpp
 * Brief:       Entry point for the C++ autonomous vehicle controller on
 *              Raspberry Pi. Initializes the SensorManager, runs the main
 *              control loop, and handles POSIX signals for clean shutdown.
 * Author:      Autonomous Vehicle Team
 * Date:        2025-01-30
 * License:     MIT
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -I. main.cpp sensor_manager.cpp \
 *       camera.cpp gpio_controller.cpp -lavcodec -lavutil -lstdc++fs \
 *       -o vehicle_controller
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "sensor_manager.hpp"
#include "camera.hpp"

// ----------------------------------------------------------------------
// Globals (visible to the signal handler)
// ----------------------------------------------------------------------
static std::atomic<bool> g_running{true};

// ----------------------------------------------------------------------
// Signal handler
// ----------------------------------------------------------------------
extern "C" void signal_handler(int signum) {
    std::cerr << "\n[main] Received signal " << signum
              << " — shutting down\n";
    g_running.store(false, std::memory_order_release);
}

// ----------------------------------------------------------------------
// CLI parsing (very small — use cxxopts in production)
// ----------------------------------------------------------------------
struct CliArgs {
    int         rate_hz      = 30;
    std::string config_path  = "config/av-01.yaml";
    bool        verbose      = false;
    bool        dry_run      = false;
};

static CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&](const std::string& opt) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << opt << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "-r" || arg == "--rate") {
            args.rate_hz = std::stoi(value(arg));
        } else if (arg == "-c" || arg == "--config") {
            args.config_path = value(arg);
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--dry-run") {
            args.dry_run = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout <<
                "Usage: vehicle_controller [-r RATE] [-c CONFIG] [-v] [--dry-run]\n"
                "  -r, --rate       Loop rate in Hz (default 30)\n"
                "  -c, --config     Path to YAML config (default config/av-01.yaml)\n"
                "  -v, --verbose    Verbose logging\n"
                "      --dry-run    Initialize then exit\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    return args;
}

// ----------------------------------------------------------------------
// Main loop
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);

    // Install signal handlers
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "[main] Autonomous Vehicle Controller starting\n"
              << "[main]   rate_hz: " << args.rate_hz << "\n"
              << "[main]   config:  " << args.config_path << "\n";

    // Create sensor manager
    av::SensorManagerConfig cfg;
    cfg.rate_hz = args.rate_hz;
    cfg.config_path = args.config_path;
    cfg.verbose = args.verbose;

    auto manager = std::make_unique<av::SensorManager>(cfg);

    // Initialize all registered sensors
    try {
        manager->initialize();
    } catch (const std::exception& exc) {
        std::cerr << "[main] Initialization failed: " << exc.what() << "\n";
        return 3;
    }

    if (args.dry_run) {
        std::cout << "[main] Dry-run: initialization succeeded\n";
        manager->shutdown();
        return 0;
    }

    // Start the background poller thread
    manager->start();

    // Main loop: just wait for shutdown; SensorManager runs its own
    // internal threads. We print stats every second.
    using namespace std::chrono_literals;
    const auto period = 1s;
    while (g_running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(period);
        auto stats = manager->snapshot();
        std::cout << "[main] sensors=" << stats.sensor_count
                  << " samples=" << stats.total_samples
                  << " errors=" << stats.total_errors
                  << " dt_ms=" << stats.last_loop_dt_ms << "\n";
    }

    std::cout << "[main] Shutting down\n";
    manager->shutdown();
    std::cout << "[main] Bye\n";
    return 0;
}
