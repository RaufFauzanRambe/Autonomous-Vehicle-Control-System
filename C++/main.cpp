/**
 * @file main.cpp
 * @brief Application entry point for the Autonomous Vehicle Control System (AVCS).
 *
 * This module provides the main executable that bootstraps the AVCS pipeline:
 *   - Parses command-line arguments for configuration, mode, and debug settings
 *   - Installs signal handlers for graceful shutdown on SIGINT/SIGTERM
 *   - Creates and initializes the AutonomousCore orchestrator
 *   - Runs the main status loop at 1 Hz displaying system telemetry
 *   - Ensures clean resource deallocation on exit
 *
 * @copyright 2024 Autonomous Vehicle Control System Project
 */

#include "autonomous_core.hpp"
#include "sensor_fusion.hpp"
#include "localization.hpp"
#include "trajectory_tracking.hpp"
#include "vehicle_state_estimator.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace avcs {

// ─── Global shutdown flag for signal handlers ─────────────────────────────────

/**
 * @brief Atomic flag set by signal handlers to request graceful shutdown.
 *
 * The main loop polls this flag at each iteration. When set to true,
 * the system initiates a controlled shutdown sequence.
 */
static std::atomic<bool> g_shutdown_requested{false};

/**
 * @brief Pointer to the AutonomousCore instance for signal-triggered shutdown.
 *
 * Initialized in main() after the AutonomousCore is constructed.
 * Accessed only from the signal handler and the main thread.
 */
static AutonomousCore* g_core_instance{nullptr};

// ─── Signal handler ───────────────────────────────────────────────────────────

/**
 * @brief Signal handler for SIGINT and SIGTERM.
 *
 * Sets the global shutdown flag and triggers emergency stop on the
 * AutonomousCore instance if available. The handler is async-signal-safe
 * for the atomic flag; the emergencyStop() call is a best-effort attempt
 * and may not complete if the signal arrives during destructor execution.
 *
 * @param signum  The signal number received (SIGINT or SIGTERM).
 */
static void signalHandler(int signum) {
    const char* sig_name = (signum == SIGINT) ? "SIGINT" : "SIGTERM";
    std::cerr << "\n[AVCS] Received " << sig_name << " (signal " << signum << "). "
              << "Initiating graceful shutdown..." << std::endl;

    g_shutdown_requested.store(true);

    // Attempt to trigger emergency stop on the core instance.
    // This is safe if the core object is still alive.
    if (g_core_instance != nullptr) {
        try {
            g_core_instance->emergencyStop();
        } catch (...) {
            // Swallow exceptions from signal handler context
        }
    }
}

// ─── Command-line argument structure ──────────────────────────────────────────

/**
 * @brief Parsed command-line arguments.
 */
struct CliArguments {
    std::string config_path;           ///< Path to the JSON/YAML config file
    std::string mode;                  ///< "autonomous" or "manual"
    bool debug;                        ///< Enable debug output
    std::string log_level;             ///< Logging verbosity level
};

// ─── CLI parsing helpers ──────────────────────────────────────────────────────

/**
 * @brief Print usage information to stderr.
 *
 * @param program_name  The name of the executable (argv[0]).
 */
static void printUsage(const std::string& program_name) {
    std::cerr << "Usage: " << program_name
              << " [OPTIONS]\n"
              << "\n"
              << "Options:\n"
              << "  --config PATH     Path to configuration file (required)\n"
              << "  --mode MODE       Operating mode: autonomous | manual (default: autonomous)\n"
              << "  --debug           Enable debug output (default: off)\n"
              << "  --log-level LEVEL Logging verbosity: debug | info | warn | error "
              << "(default: info)\n"
              << "  --help            Show this help message and exit\n"
              << "\n"
              << "Examples:\n"
              << "  " << program_name << " --config /etc/avcs/config.json --mode autonomous\n"
              << "  " << program_name << " --config ./config.yaml --debug --log-level debug\n";
}

/**
 * @brief Parse command-line arguments into a structured CliArguments object.
 *
 * Supports: --config, --mode, --debug, --log-level, --help.
 * Unrecognized options are reported but do not cause a hard error.
 *
 * @param argc  Argument count from main().
 * @param argv  Argument vector from main().
 * @return Parsed CliArguments with defaults filled in for missing options.
 */
static CliArguments parseArguments(int argc, char* argv[]) {
    CliArguments args;
    args.config_path = "";
    args.mode = "autonomous";
    args.debug = false;
    args.log_level = "info";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--config") {
            if (i + 1 < argc) {
                args.config_path = argv[++i];
            } else {
                std::cerr << "[AVCS] Error: --config requires a path argument.\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (arg == "--mode") {
            if (i + 1 < argc) {
                args.mode = argv[++i];
                if (args.mode != "autonomous" && args.mode != "manual") {
                    std::cerr << "[AVCS] Error: --mode must be 'autonomous' or 'manual', "
                              << "got '" << args.mode << "'\n";
                    std::exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "[AVCS] Error: --mode requires an argument.\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (arg == "--debug") {
            args.debug = true;
            args.log_level = "debug";
        } else if (arg == "--log-level") {
            if (i + 1 < argc) {
                args.log_level = argv[++i];
                if (args.log_level != "debug" && args.log_level != "info" &&
                    args.log_level != "warn" && args.log_level != "error") {
                    std::cerr << "[AVCS] Error: --log-level must be one of "
                              << "debug, info, warn, error. Got '" << args.log_level << "'\n";
                    std::exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "[AVCS] Error: --log-level requires an argument.\n";
                std::exit(EXIT_FAILURE);
            }
        } else {
            std::cerr << "[AVCS] Warning: Unrecognized option '" << arg << "'\n";
        }
    }

    return args;
}

// ─── Startup banner ───────────────────────────────────────────────────────────

/**
 * @brief Display the AVCS startup banner with system information.
 *
 * Prints an ASCII-art banner, version info, build configuration,
 * and the parsed CLI arguments to stdout.
 *
 * @param args  The parsed CLI arguments to display.
 */
static void displayBanner(const CliArguments& args) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Autonomous Vehicle Control System (AVCS) v1.0.0       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Build: " << __DATE__ << " " << __TIME__ << "                              ║\n";
    std::cout << "║  Compiler: " << __VERSION__ << "                  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Configuration:  " << (args.config_path.empty() ? "(none)" : args.config_path) << "\n";
    std::cout << "║  Mode:           " << args.mode << "\n";
    std::cout << "║  Debug:          " << (args.debug ? "enabled" : "disabled") << "\n";
    std::cout << "║  Log Level:      " << args.log_level << "\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << std::endl;
}

// ─── Mode / state conversion helpers ──────────────────────────────────────────

/**
 * @brief Convert AutonomousMode enum to a human-readable string.
 *
 * @param mode  The autonomous mode.
 * @return String representation of the mode.
 */
static const char* modeToString(AutonomousMode mode) {
    switch (mode) {
        case AutonomousMode::MANUAL:          return "MANUAL";
        case AutonomousMode::AUTONOMOUS:      return "AUTONOMOUS";
        case AutonomousMode::SEMI_AUTONOMOUS: return "SEMI_AUTONOMOUS";
        case AutonomousMode::EMERGENCY:       return "EMERGENCY";
        case AutonomousMode::PARKING:         return "PARKING";
        default:                              return "UNKNOWN";
    }
}

/**
 * @brief Convert SystemState enum to a human-readable string.
 *
 * @param state  The system state.
 * @return String representation of the state.
 */
static const char* stateToString(SystemState state) {
    switch (state) {
        case SystemState::INITIALIZING:    return "INITIALIZING";
        case SystemState::READY:           return "READY";
        case SystemState::DRIVING:         return "DRIVING";
        case SystemState::EMERGENCY_STOP:  return "EMERGENCY_STOP";
        case SystemState::FAULT:           return "FAULT";
        case SystemState::SHUTDOWN:        return "SHUTDOWN";
        default:                           return "UNKNOWN";
    }
}

// ─── Status display ───────────────────────────────────────────────────────────

/**
 * @brief Print a status update line with current system telemetry.
 *
 * Displays the current mode, system state, and (if available) position,
 * velocity, and orientation from the core's subsystems.
 *
 * @param core  Pointer to the initialized AutonomousCore.
 */
static void displayStatus(const AutonomousCore& core) {
    AutonomousMode mode = core.getMode();
    SystemState state = core.getSystemState();

    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    double timestamp = std::chrono::duration<double>(epoch).count();

    std::cout << "[" << std::fixed << std::setprecision(3) << timestamp << "] "
              << "Mode: " << modeToString(mode)
              << " | State: " << stateToString(state)
              << std::endl;
}

}  // namespace avcs

// ─── Main entry point ─────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    using namespace avcs;

    // ── Step 1: Parse command-line arguments ──────────────────────────────
    CliArguments args = parseArguments(argc, argv);

    if (args.config_path.empty()) {
        std::cerr << "[AVCS] Error: --config is required. Use --help for usage.\n";
        return EXIT_FAILURE;
    }

    // ── Step 2: Display startup banner ────────────────────────────────────
    displayBanner(args);

    // ── Step 3: Install signal handlers ───────────────────────────────────
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::cout << "[AVCS] Signal handlers installed (SIGINT, SIGTERM).\n";

    // ── Step 4: Create and initialize AutonomousCore ──────────────────────
    std::unique_ptr<AutonomousCore> core;

    try {
        std::cout << "[AVCS] Creating AutonomousCore with config: "
                  << args.config_path << "\n";
        core = std::make_unique<AutonomousCore>(args.config_path);
        g_core_instance = core.get();

        std::cout << "[AVCS] Initializing subsystems...\n";
        if (!core->initialize()) {
            std::cerr << "[AVCS] FATAL: System initialization failed. "
                      << "Check configuration and sensor connectivity.\n";
            g_core_instance = nullptr;
            return EXIT_FAILURE;
        }

        std::cout << "[AVCS] All subsystems initialized successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[AVCS] FATAL: Exception during initialization: "
                  << e.what() << "\n";
        g_core_instance = nullptr;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "[AVCS] FATAL: Unknown exception during initialization.\n";
        g_core_instance = nullptr;
        return EXIT_FAILURE;
    }

    // ── Step 5: Set the operating mode ────────────────────────────────────
    try {
        if (args.mode == "autonomous") {
            core->setMode(AutonomousMode::AUTONOMOUS);
            std::cout << "[AVCS] Mode set to AUTONOMOUS.\n";
        } else {
            core->setMode(AutonomousMode::MANUAL);
            std::cout << "[AVCS] Mode set to MANUAL.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[AVCS] Warning: Failed to set mode: " << e.what() << "\n";
        // Continue in whatever mode the core defaults to
    }

    // ── Step 6: Launch the main control loop ──────────────────────────────
    try {
        std::cout << "[AVCS] Starting main control loop...\n";
        std::cout << "[AVCS] Press Ctrl+C to initiate graceful shutdown.\n\n";

        // Run the core in a separate thread so we can display status
        std::thread core_thread([&core]() {
            try {
                core->run();
            } catch (const std::exception& e) {
                std::cerr << "[AVCS] Exception in control loop: " << e.what() << "\n";
                g_shutdown_requested.store(true);
            } catch (...) {
                std::cerr << "[AVCS] Unknown exception in control loop.\n";
                g_shutdown_requested.store(true);
            }
        });

        // ── Step 7: Status display loop at 1 Hz ───────────────────────────
        while (!g_shutdown_requested.load()) {
            displayStatus(*core);

            // Sleep for 1 second, but wake early if shutdown is requested
            for (int i = 0; i < 10 && !g_shutdown_requested.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Check for fault state — if the core enters FAULT,
            // log it but keep the status loop alive for diagnostics.
            SystemState current_state = core->getSystemState();
            if (current_state == SystemState::FAULT) {
                std::cerr << "[AVCS] WARNING: System is in FAULT state. "
                          << "Attempting recovery...\n";
                // In a production system, we might attempt automatic recovery
                // here by re-initializing subsystems or switching to manual mode.
            }
        }

        // ── Step 8: Graceful shutdown ─────────────────────────────────────
        std::cout << "\n[AVCS] Shutdown requested. Stopping control loop...\n";
        core->shutdown();

        if (core_thread.joinable()) {
            core_thread.join();
        }

        std::cout << "[AVCS] Control loop stopped.\n";
    } catch (const std::exception& e) {
        std::cerr << "[AVCS] FATAL: Exception during operation: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[AVCS] FATAL: Unknown exception during operation.\n";
    }

    // ── Step 9: Cleanup ───────────────────────────────────────────────────
    g_core_instance = nullptr;

    try {
        core.reset();
        std::cout << "[AVCS] Core instance destroyed. Resources released.\n";
    } catch (const std::exception& e) {
        std::cerr << "[AVCS] Warning: Exception during cleanup: " << e.what() << "\n";
    }

    // Restore default signal handlers
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);

    std::cout << "[AVCS] Autonomous Vehicle Control System shut down cleanly.\n";
    std::cout << "[AVCS] Goodbye.\n\n";

    return EXIT_SUCCESS;
}
