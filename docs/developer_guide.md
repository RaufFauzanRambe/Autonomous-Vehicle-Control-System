# Autonomous Vehicle Control System — Developer Guide

**Version:** 2.0.0
**Last Updated:** 2026-03-04
**Applies To:** AVCS v0.1.0+

---

## Table of Contents

1. [Development Setup](#1-development-setup)
2. [Code Style](#2-code-style)
3. [Project Structure](#3-project-structure)
4. [Adding a New Module](#4-adding-a-new-module)
5. [Testing Guidelines](#5-testing-guidelines)
6. [Git Workflow](#6-git-workflow)
7. [Debugging](#7-debugging)
8. [Performance Profiling](#8-performance-profiling)

---

## 1. Development Setup

### 1.1 IDE Setup

#### Visual Studio Code (Recommended)

VSCode is the recommended IDE for AVCS development because it provides excellent C++, Python, and ROS2 support with a unified interface.

**Install the following extensions:**

| Extension                     | ID                              | Purpose                                  |
|-------------------------------|---------------------------------|------------------------------------------|
| C/C++                         | `ms-vscode.cpptools`           | IntelliSense, debugging, code navigation |
| C/C++ Extension Pack          | `ms-vscode.cpptools-extension-pack` | CMake Tools, Catch2 test adapter    |
| CMake Tools                   | `ms-vscode.cmake-tools`        | CMake configure, build, debug            |
| Python                        | `ms-python.python`             | Linting, debugging, IntelliSense         |
| Pylance                       | `ms-python.vscode-pylance`     | Fast Python language server              |
| ROS                           | `ms-iot.vscode-ros`            | ROS2 launch, topic monitoring            |
| clangd                        | `llvm-vs-code-extensions.vscode-clangd` | Alternative C++ language server   |
| YAML                          | `redhat.vscode-yaml`           | YAML validation and autocomplete         |
| Docker                        | `ms-azuretools.vscode-docker`  | Dockerfile and compose support           |
| GitLens                       | `eamodio.gitlens`              | Git blame, history, file annotations     |
| clang-format                  | `xaver.clang-format`           | Auto-format C++ on save                  |
| Black Formatter               | `ms-python.black-formatter`    | Auto-format Python on save               |
| Error Lens                    | `usernamehw.errorlens`         | Inline error highlighting                |

**VSCode `settings.json` (add to `.vscode/settings.json`):**

```json
{
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureArgs": [
        "-DUSE_CUDA=ON",
        "-DBUILD_PYTHON_BINDINGS=ON",
        "-DBUILD_TESTING=ON"
    ],
    "python.analysis.extraPaths": [
        "${workspaceFolder}/src"
    ],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=120"],
    "[cpp]": {
        "editor.defaultFormatter": "xaver.clang-format",
        "editor.formatOnSave": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true
    },
    "files.associations": {
        "*.hpp": "cpp",
        "*.yaml": "yaml"
    }
}
```

#### CLion

CLion by JetBrains provides deep CMake integration and is an excellent choice for C++-heavy development.

1. Open the project directory in CLion.
2. Go to **File → Settings → Build, Execution, Deployment → CMake**.
3. Add the following CMake options:
   ```
   -DUSE_CUDA=ON
   -DBUILD_PYTHON_BINDINGS=ON
   -DBUILD_TESTING=ON
   ```
4. Set the build type to **Debug** for development.
5. Under **File → Settings → Build, Execution, Deployment → Toolchains**, ensure GCC 11+ is selected.
6. Install the Python and ROS2 plugins from the JetBrains marketplace.

### 1.2 Debugging Configuration

#### VSCode Launch Configuration (`.vscode/launch.json`)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug AVCS Core (C++)",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/avcs_core",
            "args": ["--config", "${workspaceFolder}/config/system.yaml", "--mode", "autonomous"],
            "cwd": "${workspaceFolder}",
            "environment": [
                {"name": "LD_LIBRARY_PATH", "value": "${workspaceFolder}/install/lib"}
            ],
            "MIMode": "gdb",
            "setupCommands": [
                {"description": "Enable pretty-printing", "text": "-enable-pretty-printing", "ignoreFailures": true}
            ]
        },
        {
            "name": "Debug C++ Tests",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/avcs_tests",
            "args": ["--gtest_filter=*"],
            "cwd": "${workspaceFolder}",
            "MIMode": "gdb"
        },
        {
            "name": "Debug Python Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["test/", "-v", "--cov=src"],
            "cwd": "${workspaceFolder}",
            "env": {"PYTHONPATH": "${workspaceFolder}/src"}
        },
        {
            "name": "Attach to ROS2 Node",
            "type": "cppdbg",
            "request": "attach",
            "program": "${workspaceFolder}/build/avcs_ros2_node",
            "processId": "${command:pickProcess}",
            "MIMode": "gdb"
        }
    ]
}
```

#### CLion Run/Debug Configuration

1. Go to **Run → Edit Configurations**.
2. Add a **CMake Application** configuration:
   - Target: `avcs_core_bin`
   - Program arguments: `--config config/system.yaml --mode autonomous`
   - Environment: `LD_LIBRARY_PATH=install/lib`
3. Add a **Google Test** configuration for C++ unit tests.
4. Add a **pytest** configuration for Python tests.

---

## 2. Code Style

### 2.1 C++ Style Guide

The AVCS C++ codebase follows a style guide inspired by the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with modifications for real-time embedded systems development.

#### Naming Conventions

| Element                | Convention          | Example                          |
|------------------------|---------------------|----------------------------------|
| Class/Struct names     | PascalCase          | `SensorFusion`, `EKFLocalizer`   |
| Function names         | snake_case          | `process_point_cloud()`, `update_state()` |
| Variable names         | snake_case          | `num_tracks`, `lateral_error`    |
| Member variables       | snake_case with trailing underscore | `covariance_`, `is_initialized_` |
| Constants              | kPascalCase         | `kMaxSpeed`, `kDefaultRateHz`    |
| Enum values            | kPascalCase         | `LaneKeeping`, `EmergencyStop`   |
| Macros                 | UPPER_SNAKE_CASE    | `AVCS_USE_CUDA`, `AVCS_VERSION`  |
| Namespace names        | snake_case          | `avcs::perception`, `avcs::control` |
| Template parameters    | PascalCase          | `typename Scalar`, `typename StateType` |
| File names             | snake_case          | `sensor_fusion.hpp`, `ekf_localizer.cpp` |

#### Formatting with clang-format

All C++ code must be formatted with `clang-format` using the project's `.clang-format` configuration file. The project uses the following key settings:

```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 120
AllowShortFunctionsOnASingleLine: Inline
AllowShortIfStatementsOnASingleLine: Never
AllowShortLoopsOnASingleLine: false
BreakBeforeBraces: Attach
PointerAlignment: Left
SortIncludes: CaseInsensitive
```

Format your code before committing:

```bash
# Format all C++ files
find src/ -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i -style=file

# Or use the Makefile target
make format-cpp
```

#### Include Order

Headers must be included in the following order, separated by blank lines:

1. **Corresponding header** (for `.cpp` files, include their own `.hpp` first)
2. **C system headers** (`<cstdio>`, `<cmath>`, `<cstring>`)
3. **C++ standard library headers** (`<vector>`, `<string>`, `<memory>`)
4. **Third-party library headers** (`<Eigen/Dense>`, `<opencv2/opencv.hpp>`)
5. **ROS2 headers** (`<rclcpp/rclcpp.hpp>`, `<sensor_msgs/msg/point_cloud2.hpp>`)
6. **AVCS project headers** (`"sensor_fusion.hpp"`, `"localization.hpp"`)

Example:

```cpp
// 1. Corresponding header
#include "sensor_fusion.hpp"

// 2. C system headers
#include <cstdio>
#include <cstring>

// 3. C++ standard library headers
#include <algorithm>
#include <memory>
#include <vector>

// 4. Third-party library headers
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// 5. ROS2 headers
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// 6. AVCS project headers
#include "object_tracker.hpp"
#include "lidar_processor.hpp"
```

#### C++ Best Practices for Real-Time Code

- **Avoid dynamic memory allocation** in the hot path (control loop, sensor callbacks). Pre-allocate buffers at initialization.
- **Use `std::array`** instead of `std::vector` for fixed-size containers in the real-time loop.
- **Prefer stack allocation** over heap allocation for small, short-lived objects.
- **Avoid exceptions** in the real-time path; use return codes or `std::optional` for error handling.
- **Mark functions `noexcept`** when they cannot throw (especially in the control loop).
- **Use `[[nodiscard]]`** for functions whose return value must not be ignored.
- **Prefer `constexpr`** for compile-time computation.

### 2.2 Python Style Guide

The AVCS Python codebase follows [PEP 8](https://peps.python.org/pep-0008/) with a line length of 120 characters.

#### Formatting Tools

- **Black** for code formatting (line length: 120)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for static type checking

```bash
# Format Python code
python3 -m black src/ test/ --line-length=120
python3 -m isort src/ test/

# Lint Python code
python3 -m flake8 src/ test/ --max-line-length=120
python3 -m mypy src/ --ignore-missing-imports

# Or use the Makefile targets
make format-python
make lint-python
```

#### Python Import Order (enforced by isort)

1. **Standard library** (`os`, `sys`, `math`, `dataclasses`)
2. **Third-party** (`numpy`, `torch`, `cv2`, `rclpy`)
3. **Local application** (`avcs.perception`, `avcs.control`)

#### Type Hints

All public Python functions must include type hints:

```python
from typing import Optional, List

def process_point_cloud(
    points: np.ndarray,
    ground_removal: bool = True,
    resolution: float = 0.1,
) -> Optional[np.ndarray]:
    """Process a LiDAR point cloud.

    Args:
        points: Nx3 array of XYZ point coordinates.
        ground_removal: Whether to remove ground points first.
        resolution: Voxel grid resolution in meters.

    Returns:
        Processed point cloud as an Mx3 ndarray, or None if input is empty.
    """
    ...
```

### 2.3 Documentation Standards

#### C++ Documentation (Doxygen)

All public C++ classes, functions, and member variables must be documented with Doxygen-style comments:

```cpp
/**
 * @brief Extended Kalman Filter for vehicle state estimation.
 *
 * The EKFLocalizer fuses GPS, IMU, wheel odometry, and LiDAR
 * measurements to produce a 15-state vehicle pose estimate at 10 Hz.
 *
 * @note This class is NOT thread-safe. External synchronization is required
 *       if accessed from multiple threads.
 */
class EKFLocalizer {
public:
    /**
     * @brief Construct a new EKFLocalizer.
     *
     * @param process_noise Position process noise standard deviation (m).
     * @param gps_noise_xy GPS XY measurement noise (m).
     * @param gps_noise_z GPS Z measurement noise (m).
     */
    EKFLocalizer(double process_noise, double gps_noise_xy, double gps_noise_z);

    /**
     * @brief Update the state with a new GPS measurement.
     *
     * @param position GPS position in ENU frame [east, north, up].
     * @param timestamp Measurement timestamp in seconds.
     * @return true if the update was accepted, false if rejected (e.g., high DOP).
     */
    bool updateGPS(const Eigen::Vector3d& position, double timestamp);

private:
    Eigen::Matrix15d covariance_;  ///< State covariance matrix (15x15)
    bool is_initialized_ = false;  ///< Whether the filter has been initialized
};
```

#### Python Documentation (docstrings)

All public Python functions, classes, and modules must have Google-style docstrings:

```python
"""Perception module for object detection and tracking."""

class ObjectTracker:
    """Multi-object tracker using Kalman filter and Hungarian data association.

    The tracker maintains a persistent state for each detected object across
    frames, producing smooth position and velocity estimates with stable
    tracking IDs.

    Attributes:
        max_coast_frames: Maximum number of frames to coast without measurement.
        tracks: List of currently active Track objects.
    """

    def __init__(self, max_coast_frames: int = 5) -> None:
        """Initialize the ObjectTracker.

        Args:
            max_coast_frames: Maximum frames to keep a track without updates.
        """
        ...
```

---

## 3. Project Structure

### 3.1 Directory Layout

```
Autonomous-Vehicle-Control-System/
├── src/                            # All source code
│   ├── perception/                 # Perception module (Python)
│   │   ├── __init__.py             # Module entry point, exports public API
│   │   ├── camera_processor.py     # Camera undistortion, BEV, detection
│   │   ├── lidar_processor.py      # Point cloud processing, ground removal
│   │   ├── object_detector.py      # 2D/3D detection wrapper
│   │   ├── object_tracker.py       # Multi-object Kalman tracker
│   │   └── sensor_fusion.py        # Late/hybrid multi-sensor fusion
│   │
│   ├── localization/               # Localization module (Python)
│   │   ├── __init__.py
│   │   ├── ekf_localizer.py        # 15-state EKF localizer
│   │   ├── gps_processor.py        # GPS LLA→ENU conversion
│   │   ├── imu_processor.py        # IMU bias and gravity compensation
│   │   ├── slam_node.py            # SLAM front-end and loop closure
│   │   └── map_manager.py          # HD map querying (Lanelet2)
│   │
│   ├── planning/                   # Planning module (Python)
│   │   ├── __init__.py
│   │   ├── path_planner.py         # A* and lattice-based planners
│   │   ├── behavior_planner.py     # FSM-based behavior planner
│   │   ├── trajectory_optimizer.py # Frenet-frame QP trajectory optimization
│   │   └── cost_calculator.py      # Multi-objective cost function
│   │
│   ├── control/                    # Control module (Python)
│   │   ├── __init__.py
│   │   ├── pid_controller.py       # Anti-windup PID
│   │   ├── mpc_controller.py       # Model Predictive Control
│   │   ├── stanley_controller.py   # Stanley lateral controller
│   │   ├── pure_pursuit_controller.py  # Pure pursuit lateral controller
│   │   └── vehicle_model.py        # Bicycle vehicle dynamics model
│   │
│   ├── communication/              # V2X communication module (Python)
│   │   ├── __init__.py
│   │   ├── dsrc_interface.py       # DSRC 5.9 GHz radio interface
│   │   ├── v2x_node.py             # V2X ROS2 node
│   │   ├── camp_processor.py       # Cooperative Awareness Message
│   │   └── message_parser.py       # J2735 BSM/MAP/SPAT parsing
│   │
│   ├── simulation/                 # Simulation interfaces (Python)
│   │   ├── __init__.py
│   │   ├── carla_simulator.py      # CARLA Python API wrapper
│   │   ├── sumo_interface.py       # SUMO TraCI interface
│   │   ├── scenario_runner.py      # Parameterized scenario execution
│   │   └── data_recorder.py        # ROS2 bag recording
│   │
│   ├── main.cpp                    # C++ application entry point
│   ├── sensor_fusion.cpp/hpp       # C++ EKF sensor fusion core
│   ├── localization.cpp/hpp        # C++ EKF localizer core
│   ├── slam_system.cpp/hpp         # C++ SLAM implementation
│   ├── lidar_processing.cpp        # C++ point cloud algorithms
│   ├── radar_processing.cpp        # C++ radar processing
│   ├── real_time_engine.cpp        # C++ deterministic control loop
│   ├── trajectory_tracking.cpp/hpp # C++ path tracking controller
│   ├── vehicle_state_estimator.cpp/hpp  # C++ vehicle state EKF
│   ├── autonomous_core.cpp/hpp     # C++ system orchestrator
│   └── python_bindings.cpp         # pybind11 Python bindings
│
├── include/                        # Public C++ header files
│   └── avcs/                       # Installed headers
│
├── config/                         # Configuration YAML files
│   ├── system.yaml                 # System-wide settings
│   ├── perception.yaml             # Perception pipeline parameters
│   ├── localization.yaml           # EKF/SLAM parameters
│   ├── planning.yaml               # Planner parameters
│   ├── control.yaml                # Controller gains and limits
│   ├── vehicle.yaml                # Vehicle dimensions and sensor positions
│   └── simulation.yaml             # CARLA/SUMO configuration
│
├── test/                           # Test code
│   ├── cpp/                        # C++ GTest unit tests
│   ├── unit/                       # Python pytest unit tests
│   ├── integration/                # Integration tests
│   └── benchmarks/                 # Google Benchmark files
│
├── launch/                         # ROS2 launch files
├── scripts/                        # Utility and automation scripts
├── dashboard/                      # Web dashboard (React/Node.js)
├── docs/                           # Documentation
│   ├── installation_guide.md
│   ├── developer_guide.md
│   ├── architecture/
│   └── api_documentation/
│
├── CMakeLists.txt                  # CMake build configuration
├── package.xml                     # ROS2 package manifest
├── setup.py                        # Python package setup
├── requirements.txt                # Python pip dependencies
├── environment.yml                 # Conda environment definition
├── Makefile                        # Build automation
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Multi-container orchestration
├── package.json                    # Node.js dashboard dependencies
├── LICENSE                         # Apache 2.0
└── README.md                       # Project overview
```

### 3.2 Where to Add New Code

| What you want to add          | Where to put it                                         |
|-------------------------------|---------------------------------------------------------|
| New perception algorithm      | `src/perception/` (Python) or `src/*.cpp` (C++ core)   |
| New control algorithm         | `src/control/` (Python)                                 |
| New C++ utility function      | `src/*.hpp` (header) + `src/*.cpp` (implementation)     |
| Python bindings for C++       | `src/python_bindings.cpp`                               |
| Unit test (C++)               | `test/cpp/`                                             |
| Unit test (Python)            | `test/unit/`                                            |
| Integration test              | `test/integration/`                                     |
| ROS2 launch file              | `launch/`                                               |
| Configuration parameter       | `config/<appropriate>.yaml`                             |
| Build system change           | `CMakeLists.txt`                                        |
| New ROS2 message type         | `msg/` or `srv/` directory + `package.xml` update       |

---

## 4. Adding a New Module

This section walks through adding a complete new module to the AVCS, from directory creation through testing and configuration. We will use a **Risk Assessment** module as a running example.

### 4.1 Create the Directory Structure

```bash
# Create the Python module directory
mkdir -p src/risk_assessment

# Create the C++ core files (if high-performance computation is needed)
touch src/risk_assessment_core.hpp
touch src/risk_assessment_core.cpp

# Create the test directory
mkdir -p test/unit/test_risk_assessment
touch test/unit/test_risk_assessment/__init__.py
touch test/unit/test_risk_assessment/test_risk_evaluator.py

# Create the C++ test directory
touch test/cpp/risk_assessment_test.cpp

# Create configuration
touch config/risk_assessment.yaml
```

### 4.2 Write the Header File

```cpp
// src/risk_assessment_core.hpp
#ifndef AVCS_RISK_ASSESSMENT_CORE_HPP
#define AVCS_RISK_ASSESSMENT_CORE_HPP

#include <vector>
#include <Eigen/Dense>

namespace avcs {
namespace risk {

/**
 * @brief Risk level for a traffic scenario.
 */
enum class RiskLevel {
    kSafe = 0,
    kCaution = 1,
    kWarning = 2,
    kCritical = 3
};

/**
 * @brief Represents a potential collision risk between the ego vehicle
 *        and another traffic participant.
 */
struct CollisionRisk {
    int object_id;              ///< Tracked object ID
    double ttc;                 ///< Time-to-collision (s)
    double distance;            ///< Distance to object (m)
    RiskLevel risk_level;       ///< Computed risk level
    Eigen::Vector2d relative_velocity;  ///< Relative velocity (m/s)
};

/**
 * @brief Assesses collision risk for the ego vehicle based on
 *        tracked objects and the planned trajectory.
 */
class RiskAssessor {
public:
    /**
     * @brief Construct a RiskAssessor with configurable thresholds.
     *
     * @param ttc_critical  Time-to-collision for critical risk (s).
     * @param ttc_warning   Time-to-collision for warning risk (s).
     * @param ttc_caution   Time-to-collision for caution risk (s).
     */
    explicit RiskAssessor(double ttc_critical = 1.5,
                          double ttc_warning = 3.0,
                          double ttc_caution = 5.0);

    /**
     * @brief Evaluate collision risks for all tracked objects.
     *
     * @param ego_position  Ego vehicle position (2D).
     * @param ego_velocity  Ego vehicle velocity (2D).
     * @param object_positions Tracked object positions (Nx2).
     * @param object_velocities Tracked object velocities (Nx2).
     * @return Vector of CollisionRisk for each object with TTC > 0.
     */
    std::vector<CollisionRisk> evaluate(
        const Eigen::Vector2d& ego_position,
        const Eigen::Vector2d& ego_velocity,
        const Eigen::MatrixXd& object_positions,
        const Eigen::MatrixXd& object_velocities) const;

private:
    double ttc_critical_;
    double ttc_warning_;
    double ttc_caution_;

    RiskLevel classifyRisk(double ttc) const;
};

}  // namespace risk
}  // namespace avcs

#endif  // AVCS_RISK_ASSESSMENT_CORE_HPP
```

### 4.3 Write the Source File

```cpp
// src/risk_assessment_core.cpp
#include "risk_assessment_core.hpp"

#include <algorithm>
#include <cmath>

namespace avcs {
namespace risk {

RiskAssessor::RiskAssessor(double ttc_critical, double ttc_warning, double ttc_caution)
    : ttc_critical_(ttc_critical),
      ttc_warning_(ttc_warning),
      ttc_caution_(ttc_caution) {}

std::vector<CollisionRisk> RiskAssessor::evaluate(
    const Eigen::Vector2d& ego_position,
    const Eigen::Vector2d& ego_velocity,
    const Eigen::MatrixXd& object_positions,
    const Eigen::MatrixXd& object_velocities) const {

    std::vector<CollisionRisk> risks;
    const int num_objects = object_positions.rows();

    for (int i = 0; i < num_objects; ++i) {
        Eigen::Vector2d rel_pos = object_positions.row(i).transpose() - ego_position;
        Eigen::Vector2d rel_vel = object_velocities.row(i).transpose() - ego_velocity;

        double distance = rel_pos.norm();
        double closing_speed = -rel_pos.normalized().dot(rel_vel);

        double ttc = (closing_speed > 1e-6) ? (distance / closing_speed) : -1.0;

        if (ttc > 0 && ttc < ttc_caution_ * 2.0) {
            CollisionRisk risk;
            risk.object_id = i;
            risk.ttc = ttc;
            risk.distance = distance;
            risk.risk_level = classifyRisk(ttc);
            risk.relative_velocity = rel_vel;
            risks.push_back(risk);
        }
    }

    std::sort(risks.begin(), risks.end(),
              [](const CollisionRisk& a, const CollisionRisk& b) {
                  return a.ttc < b.ttc;
              });

    return risks;
}

RiskLevel RiskAssessor::classifyRisk(double ttc) const {
    if (ttc <= ttc_critical_) return RiskLevel::kCritical;
    if (ttc <= ttc_warning_)  return RiskLevel::kWarning;
    if (ttc <= ttc_caution_)  return RiskLevel::kCaution;
    return RiskLevel::kSafe;
}

}  // namespace risk
}  // namespace avcs
```

### 4.4 Update CMakeLists.txt

Add the new source files to the `AVCS_CORE_SOURCES` and `AVCS_CORE_HEADERS` lists:

```cmake
set(AVCS_CORE_SOURCES
    src/main.cpp
    src/sensor_fusion.cpp
    src/localization.cpp
    src/slam_system.cpp
    src/lidar_processing.cpp
    src/radar_processing.cpp
    src/gps_processing.cpp
    src/imu_processing.cpp
    src/real_time_engine.cpp
    src/trajectory_tracking.cpp
    src/vehicle_state_estimator.cpp
    src/autonomous_core.cpp
    src/risk_assessment_core.cpp        # <-- NEW
)

set(AVCS_CORE_HEADERS
    src/sensor_fusion.hpp
    src/localization.hpp
    src/slam_system.hpp
    src/trajectory_tracking.hpp
    src/vehicle_state_estimator.hpp
    src/autonomous_core.hpp
    src/risk_assessment_core.hpp        # <-- NEW
)
```

### 4.5 Add Python Bindings

Add the new module to `src/python_bindings.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "risk_assessment_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_core, m) {
    // ... existing bindings ...

    // Risk Assessment bindings
    py::module risk = m.def_submodule("risk", "Risk assessment algorithms");

    py::enum_<avcs::risk::RiskLevel>(risk, "RiskLevel")
        .value("SAFE", avcs::risk::RiskLevel::kSafe)
        .value("CAUTION", avcs::risk::RiskLevel::kCaution)
        .value("WARNING", avcs::risk::RiskLevel::kWarning)
        .value("CRITICAL", avcs::risk::RiskLevel::kCritical);

    py::class_<avcs::risk::CollisionRisk>(risk, "CollisionRisk")
        .def_readonly("object_id", &avcs::risk::CollisionRisk::object_id)
        .def_readonly("ttc", &avcs::risk::CollisionRisk::ttc)
        .def_readonly("distance", &avcs::risk::CollisionRisk::distance)
        .def_readonly("risk_level", &avcs::risk::CollisionRisk::risk_level);

    py::class_<avcs::risk::RiskAssessor>(risk, "RiskAssessor")
        .def(py::init<double, double, double>(),
             py::arg("ttc_critical") = 1.5,
             py::arg("ttc_warning") = 3.0,
             py::arg("ttc_caution") = 5.0)
        .def("evaluate", &avcs::risk::RiskAssessor::evaluate,
             py::arg("ego_position"),
             py::arg("ego_velocity"),
             py::arg("object_positions"),
             py::arg("object_velocities"));
}
```

### 4.6 Add Tests

**C++ Test (test/cpp/risk_assessment_test.cpp):**

```cpp
#include <gtest/gtest.h>
#include "risk_assessment_core.hpp"

using namespace avcs::risk;

TEST(RiskAssessorTest, NoRiskWhenFarAway) {
    RiskAssessor assessor;
    Eigen::Vector2d ego_pos(0.0, 0.0);
    Eigen::Vector2d ego_vel(10.0, 0.0);
    Eigen::MatrixXd obj_pos(1, 2);
    obj_pos << 100.0, 0.0;
    Eigen::MatrixXd obj_vel(1, 2);
    obj_vel << 0.0, 0.0;

    auto risks = assessor.evaluate(ego_pos, ego_vel, obj_pos, obj_vel);
    EXPECT_TRUE(risks.empty());
}

TEST(RiskAssessorTest, CriticalRiskDetected) {
    RiskAssessor assessor(1.5, 3.0, 5.0);
    Eigen::Vector2d ego_pos(0.0, 0.0);
    Eigen::Vector2d ego_vel(20.0, 0.0);
    Eigen::MatrixXd obj_pos(1, 2);
    obj_pos << 10.0, 0.0;
    Eigen::MatrixXd obj_vel(1, 2);
    obj_vel << 0.0, 0.0;

    auto risks = assessor.evaluate(ego_pos, ego_vel, obj_pos, obj_vel);
    ASSERT_EQ(risks.size(), 1u);
    EXPECT_EQ(risks[0].risk_level, RiskLevel::kCritical);
    EXPECT_NEAR(risks[0].ttc, 0.5, 0.01);
}

TEST(RiskAssessorTest, RisksSortedByTTC) {
    RiskAssessor assessor;
    Eigen::Vector2d ego_pos(0.0, 0.0);
    Eigen::Vector2d ego_vel(20.0, 0.0);
    Eigen::MatrixXd obj_pos(2, 2);
    obj_pos << 30.0, 0.0,
               10.0, 0.0;
    Eigen::MatrixXd obj_vel(2, 2);
    obj_vel << 0.0, 0.0,
               0.0, 0.0;

    auto risks = assessor.evaluate(ego_pos, ego_vel, obj_pos, obj_vel);
    ASSERT_GE(risks.size(), 2u);
    EXPECT_LT(risks[0].ttc, risks[1].ttc);
}
```

**Python Test (test/unit/test_risk_assessment/test_risk_evaluator.py):**

```python
"""Unit tests for the risk assessment module."""

import numpy as np
import pytest

from avcs.risk_assessment.risk_evaluator import RiskEvaluator, RiskLevel


class TestRiskEvaluator:
    """Tests for the RiskEvaluator class."""

    def test_no_risk_when_far_away(self) -> None:
        evaluator = RiskEvaluator()
        ego_pos = np.array([0.0, 0.0])
        ego_vel = np.array([10.0, 0.0])
        obj_pos = np.array([[100.0, 0.0]])
        obj_vel = np.array([[0.0, 0.0]])

        risks = evaluator.evaluate(ego_pos, ego_vel, obj_pos, obj_vel)
        assert len(risks) == 0

    def test_critical_risk_detected(self) -> None:
        evaluator = RiskEvaluator(ttc_critical=1.5)
        ego_pos = np.array([0.0, 0.0])
        ego_vel = np.array([20.0, 0.0])
        obj_pos = np.array([[10.0, 0.0]])
        obj_vel = np.array([[0.0, 0.0]])

        risks = evaluator.evaluate(ego_pos, ego_vel, obj_pos, obj_vel)
        assert len(risks) == 1
        assert risks[0].risk_level == RiskLevel.CRITICAL

    def test_risks_sorted_by_ttc(self) -> None:
        evaluator = RiskEvaluator()
        ego_pos = np.array([0.0, 0.0])
        ego_vel = np.array([20.0, 0.0])
        obj_pos = np.array([[30.0, 0.0], [10.0, 0.0]])
        obj_vel = np.array([[0.0, 0.0], [0.0, 0.0]])

        risks = evaluator.evaluate(ego_pos, ego_vel, obj_pos, obj_vel)
        assert len(risks) >= 2
        assert risks[0].ttc < risks[1].ttc
```

### 4.7 Add Configuration

Create `config/risk_assessment.yaml`:

```yaml
# Risk Assessment Configuration
ttc_critical: 1.5       # Time-to-collision for CRITICAL risk level (s)
ttc_warning: 3.0        # Time-to-collision for WARNING risk level (s)
ttc_caution: 5.0        # Time-to-collision for CAUTION risk level (s)
max_evaluation_distance: 100.0  # Max distance to consider objects (m)
```

And reference it from `config/system.yaml` by adding:

```yaml
risk_assessment:
  config_file: "risk_assessment.yaml"
  rate_hz: 20
  topics:
    input: "/avcs/perception"
    output: "/avcs/risk_assessment"
```

---

## 5. Testing Guidelines

### 5.1 C++ Unit Tests (GTest)

The AVCS uses [Google Test](https://github.com/google/googletest) (GTest) for C++ unit testing and [Google Mock](https://github.com/google/googletest/tree/main/googlemock) (GMock) for mocking.

#### Test Naming Convention

```
<ClassName>Test.<MethodName>_<Scenario>_<ExpectedResult>
```

Examples:
- `EKFLocalizerTest.UpdateGPS_ValidMeasurement_ReturnsTrue`
- `EKFLocalizerTest.UpdateGPS_HighDOP_ReturnsFalse`
- `SensorFusionTest.FuseTracks_SingleObject_CorrectPosition`
- `PIDControllerTest.ComputeControl_SaturatedOutput_ClampsToMax`

#### Writing a GTest

```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "sensor_fusion.hpp"

using namespace avcs;
using namespace testing;

// Fixture for common setup
class SensorFusionTest : public ::testing::Test {
protected:
    void SetUp() override {
        fusion_ = std::make_unique<SensorFusion>(fusion_config_);
    }

    SensorFusionConfig fusion_config_{.max_tracks = 100, .fusion_mode = "late"};
    std::unique_ptr<SensorFusion> fusion_;
};

TEST_F(SensorFusionTest, Initialize_StartsWithNoTracks) {
    EXPECT_EQ(fusion_->numTracks(), 0);
}

TEST_F(SensorFusionTest, AddDetection_CreatesNewTrack) {
    Eigen::Vector3d position(10.0, 5.0, 0.0);
    fusion_->addDetection(position, SensorType::LIDAR, 0.95);
    EXPECT_EQ(fusion_->numTracks(), 1);
}

// Parameterized test example
class PIDGainTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {};

TEST_P(PIDGainTest, OutputBoundedForAllGains) {
    auto [kp, ki, kd] = GetParam();
    PIDController pid(kp, ki, kd);
    double output = pid.compute(5.0, 0.0, 0.01);
    EXPECT_GE(output, -pid.maxOutput());
    EXPECT_LE(output, pid.maxOutput());
}

INSTANTIATE_TEST_SUITE_P(
    PIDGains,
    PIDGainTest,
    ::testing::Combine(
        ::testing::Values(0.5, 1.0, 2.0),
        ::testing::Values(0.0, 0.1, 0.5),
        ::testing::Values(0.0, 0.05, 0.1)
    )
);
```

#### Running C++ Tests

```bash
# Run all tests
cd build && ctest --output-on-failure -j$(nproc)

# Run a specific test binary
./build/avcs_tests

# Run tests matching a pattern
./build/avcs_tests --gtest_filter="EKFLocalizerTest.*"

# Run with verbose output
./build/avcs_tests --gtest_print_time=1 --gtest_output=xml:test_results.xml
```

### 5.2 Python Unit Tests (pytest)

The AVCS uses [pytest](https://docs.pytest.org/) for Python testing with the `pytest-cov` plugin for coverage reporting.

#### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<method>_<scenario>_<expected_result>`

#### Writing a pytest Test

```python
"""Unit tests for the PID controller module."""

import pytest
import numpy as np

from avcs.control.pid_controller import PIDController


class TestPIDController:
    """Tests for the PIDController class."""

    @pytest.fixture
    def controller(self) -> PIDController:
        """Create a PID controller with default gains."""
        return PIDController(kp=1.0, ki=0.1, kd=0.05)

    def test_compute_proportional_only(self, controller: PIDController) -> None:
        """Proportional term should be kp * error."""
        controller.ki = 0.0
        controller.kd = 0.0
        output = controller.compute(setpoint=10.0, measurement=8.0, dt=0.01)
        assert abs(output - 2.0) < 1e-6

    def test_anti_windup_clamps_integral(self, controller: PIDController) -> None:
        """Integral should not exceed the anti-windup limit."""
        controller.anti_windup = 5.0
        for _ in range(10000):
            controller.compute(setpoint=100.0, measurement=0.0, dt=0.01)
        assert abs(controller.integral) <= controller.anti_windup

    @pytest.mark.parametrize("setpoint,measurement", [
        (0.0, 0.0),
        (10.0, 10.0),
        (-5.0, -5.0),
    ])
    def test_zero_error_produces_zero_proportional(
        self, controller: PIDController, setpoint: float, measurement: float
    ) -> None:
        """When error is zero, only integral/derivative terms may be non-zero."""
        controller.reset()
        output = controller.compute(setpoint=setpoint, measurement=measurement, dt=0.01)
        assert abs(output) < 1e-6
```

#### Running Python Tests

```bash
# Run all Python tests
python3 -m pytest test/unit/ -v

# Run with coverage
python3 -m pytest test/unit/ -v --cov=src --cov-report=html --cov-report=term

# Run a specific test file
python3 -m pytest test/unit/test_control/test_pid_controller.py -v

# Run tests matching a keyword
python3 -m pytest test/unit/ -v -k "pid"

# Run with timeout (for potentially hanging tests)
python3 -m pytest test/integration/ -v --timeout=120
```

### 5.3 Integration Test Guidelines

Integration tests verify that multiple AVCS modules work together correctly. They are located in `test/integration/`.

- **Do not use real hardware or live sensors** in integration tests. Use recorded bag files or simulated sensor data.
- **Each test should be self-contained** and not depend on the state left by a previous test.
- **Use `pytest` fixtures** to set up and tear down the ROS2 environment.
- **Tag slow tests** with `@pytest.mark.slow` so they can be skipped during quick CI runs.

```python
"""Integration test: perception → planning → control pipeline."""

import pytest
import numpy as np


@pytest.mark.slow
@pytest.mark.integration
class TestPerceptionToControlPipeline:
    """Test the full perception → planning → control data flow."""

    def test_static_object_triggers_braking(self) -> None:
        """A static obstacle ahead should cause the controller to decelerate."""
        # 1. Inject simulated object detection
        # 2. Run planning pipeline
        # 3. Verify control output is negative acceleration
        ...

    def test_lane_change_avoids_slow_vehicle(self) -> None:
        """A slow vehicle ahead should trigger a lane change."""
        ...
```

### 5.4 Coverage Requirements

| Module        | Minimum Line Coverage | Target Line Coverage |
|---------------|----------------------|----------------------|
| Control       | 90%                  | 95%                  |
| Localization  | 85%                  | 90%                  |
| Planning      | 80%                  | 90%                  |
| Perception    | 70%                  | 85%                  |
| Communication | 75%                  | 85%                  |
| Simulation    | 60%                  | 75%                  |

Coverage reports are generated automatically by the CI pipeline. To generate them locally:

```bash
# C++ coverage (requires gcov)
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"
make -j$(nproc)
ctest
gcovr -r .. --html-details coverage.html

# Python coverage
python3 -m pytest test/ --cov=src --cov-report=html --cov-report=term-missing
```

---

## 6. Git Workflow

### 6.1 Branch Naming

All branches must follow the naming convention:

```
<type>/<ticket-number>-<short-description>
```

| Type        | Purpose                                    | Example                                     |
|-------------|--------------------------------------------|---------------------------------------------|
| `feature/`  | New functionality                          | `feature/AVCS-123-risk-assessment-module`   |
| `bugfix/`   | Bug fix                                    | `bugfix/AVCS-456-ekf-divergence-on-gps-loss`|
| `hotfix/`   | Critical production fix                    | `hotfix/AVCS-789-emergency-stop-timeout`    |
| `refactor/` | Code restructuring (no behavior change)    | `refactor/AVCS-101-extract-fusion-interface`|
| `docs/`     | Documentation changes                      | `docs/AVCS-200-installation-guide-update`   |
| `test/`     | Adding or modifying tests                  | `test/AVCS-300-mpc-controller-tests`        |
| `perf/`     | Performance optimization                   | `perf/AVCS-400-lidar-processing-simd`       |

### 6.2 Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

**Examples:**

```
feat(control): add Stanley lateral controller

Implement the Stanley method for lateral vehicle control with
heading error and cross-track error feedback. Add configuration
parameters to control.yaml.

Refs: AVCS-123
```

```
fix(localization): prevent EKF covariance explosion during GPS outage

Clamp the covariance matrix diagonal to a maximum value when GPS
updates are suspended for more than 5 seconds. This prevents
numerical instability when GPS becomes available again.

Fixes: AVCS-456
```

### 6.3 Pull Request Process

1. **Create a feature branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/AVCS-123-risk-assessment-module
   ```

2. **Make your changes** and commit with descriptive messages.

3. **Ensure all tests pass:**
   ```bash
   make test
   make lint
   ```

4. **Push your branch and open a PR** against the `develop` branch.

5. **Fill out the PR template** with:
   - Description of changes
   - Related issue/ticket numbers
   - Testing performed
   - Configuration changes required
   - Breaking changes (if any)

6. **Request at least two reviewers**, including one domain expert (e.g., a control systems engineer for control module changes).

7. **Address all review comments** and push updates.

8. **Squash and merge** once approved (or use merge commits for large feature branches with meaningful intermediate commits).

### 6.4 Code Review Checklist

Before approving a PR, reviewers must verify:

- [ ] **Correctness:** The code does what it claims to do
- [ ] **Safety:** No regressions in safety-critical paths (control loop, emergency stop)
- [ ] **Tests:** Adequate unit and integration test coverage
- [ ] **Performance:** No unbounded memory allocations or O(n²) algorithms in the hot path
- [ ] **Thread safety:** Shared state is properly protected; no data races
- [ ] **Error handling:** All error paths are handled gracefully; no silent failures
- [ ] **Documentation:** Public APIs are documented; complex algorithms have inline comments
- [ ] **Naming:** Follows the project naming conventions
- [ ] **Formatting:** Code is auto-formatted (clang-format / black)
- [ ] **Configuration:** New parameters have sensible defaults and are documented in the appropriate YAML file
- [ ] **Build:** No compiler warnings on GCC 11+ and Clang 14+
- [ ] **ROS2 compatibility:** Changes don't break the ROS2 build or message contracts

---

## 7. Debugging

### 7.1 Debugging Tools

#### GDB (GNU Debugger)

The primary tool for debugging C++ code:

```bash
# Debug the standalone executable
gdb --args ./build/avcs_core --config config/system.yaml --mode autonomous

# Common GDB commands:
# break avcs::control::PIDController::compute
# break main.cpp:42
# run
# continue
# next
# step
# print variable_name
# backtrace
# frame 2
# watch covariance_(0,0)
```

#### Valgrind (Memory Debugging)

Use Valgrind to detect memory leaks, buffer overflows, and uninitialized memory reads:

```bash
# Run the test suite under Valgrind
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
    ./build/avcs_tests --gtest_filter="SensorFusionTest.*"

# Run with suppression file for ROS2/DDS false positives
valgrind --suppressions=scripts/valgrind_ros2.supp \
    --leak-check=full \
    ./build/avcs_core --config config/system.yaml
```

#### ROS2 rqt Tools

For debugging the ROS2 data flow:

```bash
# Launch the rqt GUI
rqt

# Useful plugins:
# - rqt_graph:    Visualize the node/topic graph
# - rqt_plot:     Plot topic values in real time
# - rqt_console:  View ROS2 log messages
# - rqt_top:      Monitor node CPU/memory usage

# Command-line topic inspection
ros2 topic list
ros2 topic echo /avcs/perception
ros2 topic hz /avcs/control
ros2 topic info /avcs/localization

# Inspect TF transforms
ros2 run tf2_tools view_frames
```

### 7.2 Common Debugging Scenarios

#### Scenario: Control Loop Timing Violation

**Symptom:** The control loop exceeds its 10 ms budget, causing jerky steering.

**Debugging steps:**

1. Enable deadline monitoring in `config/system.yaml`:
   ```yaml
   safety:
     watchdog_timeout_ms: 200
     monitor_control_timing: true
   ```

2. Check the timing statistics published on `/avcs/diagnostics`:
   ```bash
   ros2 topic echo /avcs/diagnostics
   ```

3. Profile the control loop with `perf`:
   ```bash
   perf record -g ./build/avcs_core --config config/system.yaml
   perf report
   ```

4. Common causes:
   - Memory allocation in the hot path (check for `new`, `malloc`, `push_back` on `std::vector`)
   - Lock contention (check for `std::mutex` in the control thread)
   - Cache misses from poor data layout (check struct sizes and alignment)

#### Scenario: EKF Divergence

**Symptom:** The localization covariance grows unboundedly; position estimates drift.

**Debugging steps:**

1. Plot the covariance diagonal over time:
   ```bash
   ros2 topic echo /avcs/localization --field pose.covariance
   ```

2. Check GPS measurement quality:
   ```bash
   ros2 topic echo /avcs/gps --field status
   ```

3. Verify sensor timestamps are synchronized:
   ```bash
   ros2 topic hz /avcs/gps
   ros2 topic hz /avcs/imu
   ros2 topic hz /avcs/lidar
   ```

4. Enable EKF debug logging:
   ```yaml
   localization:
     ekf_debug: true
     log_innovation: true
   ```

### 7.3 Logging Best Practices

The AVCS uses a hierarchical logging system. Use the appropriate level for each message:

| Level   | When to Use                                           | Example                                        |
|---------|-------------------------------------------------------|------------------------------------------------|
| TRACE   | Very detailed, per-iteration data                    | `RCLCPP_TRACE(logger, "EKF state: %s", state)` |
| DEBUG   | Diagnostic information useful during development      | `RCLCPP_DEBUG(logger, "Track %d updated", id)` |
| INFO    | Normal operational messages                           | `RCLCPP_INFO(logger, "Module initialized")`    |
| WARN    | Unexpected but recoverable situations                | `RCLCPP_WARN(logger, "GPS DOP exceeds threshold")` |
| ERROR   | Significant problems that may affect functionality   | `RCLCPP_ERROR(logger, "NDT scan matching failed")` |
| FATAL   | Critical failures requiring system shutdown           | `RCLCPP_FATAL(logger, "Watchdog timeout!")`    |

**Rules:**

1. **Never log at TRACE or DEBUG level in the hot path** (control loop). Use conditional compilation:
   ```cpp
   #ifdef AVCS_DEBUG_LOGGING
       RCLCPP_TRACE(logger_, "Control output: steer=%.4f, accel=%.4f", steer, accel);
   #endif
   ```

2. **Log the module name** in every message so logs can be filtered by subsystem.

3. **Use structured logging** (key-value pairs) for machine-parseable log entries:
   ```cpp
   RCLCPP_INFO(logger_, "object_detected id=%d class=%s confidence=%.2f distance=%.1f",
               id, class_name, confidence, distance);
   ```

4. **Rate-limit repeated log messages** to avoid flooding:
   ```cpp
   RCLCPP_INFO_THROTTLE(logger_, *clock_, 5000,  // Max once per 5 seconds
                        "GPS signal degraded, using dead-reckoning");
   ```

---

## 8. Performance Profiling

### 8.1 Profiling Tools

#### Linux perf

```bash
# Record a profile of the AVCS core
perf record -g -F 99 -- ./build/avcs_core --config config/system.yaml

# View the report
perf report

# Generate a flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

#### gprof

```bash
# Build with profiling support
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-pg"
make -j$(nproc)

# Run the program (generates gmon.out)
./build/avcs_core --config config/system.yaml --mode autonomous

# View the profile
gprof ./build/avcs_core gmon.out > profile.txt
```

#### Intel VTune

```bash
# Install VTune (requires Intel oneAPI toolkit)
sudo apt install intel-oneapi-vtune

# Profile hotspots
vtune -collect hotspots -- ./build/avcs_core --config config/system.yaml

# Profile threading behavior
vtune -collect threading -- ./build/avcs_core --config config/system.yaml

# View results
vtune-gui
```

### 8.2 Benchmarking with Google Benchmark

The AVCS includes Google Benchmark targets for performance-critical code. Benchmarks are located in `test/benchmarks/`.

#### Writing a Benchmark

```cpp
// test/benchmarks/benchmark_localization.cpp
#include <benchmark/benchmark.h>
#include "localization.hpp"

static void BM_EKFPredict(benchmark::State& state) {
    avcs::EKFLocalizer localizer(0.1, 0.5, 2.0);
    Eigen::Vector3d imu_accel(0.0, 0.0, 9.81);
    Eigen::Vector3d imu_gyro(0.0, 0.0, 0.01);

    for (auto _ : state) {
        localizer.predict(imu_accel, imu_gyro, 0.005);  // 200 Hz
        benchmark::DoNotOptimize(localizer.state());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EKFPredict);

static void BM_EKFUpdateGPS(benchmark::State& state) {
    avcs::EKFLocalizer localizer(0.1, 0.5, 2.0);
    Eigen::Vector3d gps_position(100.0, 200.0, 10.0);

    for (auto _ : state) {
        localizer.updateGPS(gps_position, 0.0);
        benchmark::DoNotOptimize(localizer.state());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EKFUpdateGPS);

BENCHMARK_MAIN();
```

#### Running Benchmarks

```bash
# Build with benchmark support
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)

# Run all benchmarks
./build/avcs_benchmarks

# Run a specific benchmark
./build/avcs_benchmarks --benchmark_filter="EKF*"

# Export results to JSON for comparison
./build/avcs_benchmarks --benchmark_out=results.json --benchmark_out_format=json
```

### 8.3 Optimization Tips for Real-Time Code

The following guidelines apply to code that runs in the AVCS real-time control loop (100 Hz) or any path with a hard deadline.

#### 1. Avoid Dynamic Memory Allocation

```cpp
// BAD: Allocates on every call
std::vector<double> computeGains(double speed) {
    std::vector<double> gains(3);
    gains[0] = speed * 0.1;
    // ...
    return gains;
}

// GOOD: Pre-allocated buffer
class GainComputer {
public:
    GainComputer() : gains_(3) {}
    const std::vector<double>& computeGains(double speed) {
        gains_[0] = speed * 0.1;
        // ...
        return gains_;
    }
private:
    std::vector<double> gains_;  // Allocated once at construction
};

// BEST: Fixed-size array
struct Gains {
    double data[3];
};
Gains computeGains(double speed) {
    Gains gains;
    gains.data[0] = speed * 0.1;
    // ...
    return gains;
}
```

#### 2. Use Lock-Free Data Structures

```cpp
// BAD: Mutex in the hot path
std::mutex mtx;
void controlLoop() {
    std::lock_guard<std::mutex> lock(mtx);  // Blocks if fusion is writing
    // ... control computation ...
}

// GOOD: Lock-free SPSC ring buffer
#include <boost/lockfree/spsc_queue.hpp>
boost::lockfree::spsc_queue<SensorData, boost::lockfree::capacity<16>> queue_;

void sensorCallback(const SensorData& data) {
    queue_.push(data);  // Non-blocking
}

void controlLoop() {
    SensorData data;
    while (queue_.pop(data)) {
        // ... control computation ...
    }
}
```

#### 3. Optimize Cache Usage

```cpp
// BAD: Array of structs (poor cache locality for iteration)
struct Object {
    double x, y, z;          // Position
    double confidence;        // Only needed for display
    std::string class_name;  // Heap-allocated!
};
std::vector<Object> objects;  // Iterating over x,y,z loads unnecessary data

// GOOD: Struct of arrays (cache-friendly iteration)
struct ObjectArray {
    std::vector<double> x, y, z;  // Contiguous memory for position data
    std::vector<double> confidence;
    std::vector<int> class_id;    // Use integer ID, look up name separately
};
```

#### 4. Use SIMD Where Appropriate

```cpp
// Use Eigen's vectorized operations (automatically uses SSE/AVX when available)
Eigen::VectorXd distances = (positions.rowwise() - ego_position).rowwise().norm();

// Or use compiler pragmas for critical loops
#pragma GCC optimize("O3,unroll-loops")
void computeDistances(const double* positions, const double* ego, double* output, int n) {
    for (int i = 0; i < n; ++i) {
        double dx = positions[i * 3] - ego[0];
        double dy = positions[i * 3 + 1] - ego[1];
        double dz = positions[i * 3 + 2] - ego[2];
        output[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
}
```

#### 5. Minimize Branching in the Hot Path

```cpp
// BAD: Branch prediction misses
for (int i = 0; i < num_objects; ++i) {
    if (objects[i].type == ObjectType::VEHICLE) {
        // vehicle-specific code
    } else if (objects[i].type == ObjectType::PEDESTRIAN) {
        // pedestrian-specific code
    }
}

// BETTER: Sort by type first, then process each group contiguously
auto vehicles = objects.getByType(ObjectType::VEHICLE);
auto pedestrians = objects.getByType(ObjectType::PEDESTRIAN);
processVehicles(vehicles);
processPedestrians(pedestrians);
```

#### 6. Profile Before Optimizing

Always measure before and after any optimization:

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
// ... code to benchmark ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
RCLCPP_INFO(logger_, "Operation took %ld us", duration.count());
```

Never optimize based on intuition alone. The profiler will reveal the actual bottlenecks, which are often surprising.

---

## Additional Resources

- **Installation Guide:** [docs/installation_guide.md](installation_guide.md)
- **System Architecture:** [docs/architecture/system_architecture.md](architecture/system_architecture.md)
- **API Documentation:** [docs/api_documentation/api_documentation.md](api_documentation/api_documentation.md)
- **Google C++ Style Guide:** https://google.github.io/styleguide/cppguide.html
- **PEP 8 Style Guide:** https://peps.python.org/pep-0008/
- **Google Test Documentation:** https://google.github.io/googletest/
- **pytest Documentation:** https://docs.pytest.org/
- **Google Benchmark:** https://github.com/google/benchmark
- **Conventional Commits:** https://www.conventionalcommits.org/
- **ROS2 Humble Documentation:** https://docs.ros.org/en/humble/
