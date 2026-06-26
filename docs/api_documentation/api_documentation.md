# Autonomous Vehicle Control System — API Documentation

> **Version:** 0.1.0
> **Namespace:** `avcs`
> **Language Standard:** C++17 (core), Python 3.8+ (modules)
> **Last Updated:** 2024

---

## Table of Contents

1. [API Overview](#1-api-overview)
2. [SensorFusion API](#2-sensorfusion-api)
3. [EKFLocalizer API](#3-ekflocalizer-api)
4. [SLAMSystem API](#4-slamsystem-api)
5. [TrajectoryTracker API](#5-trajectorytracker-api)
6. [VehicleStateEstimator API](#6-vehiclestateestimator-api)
7. [AutonomousCore API](#7-autonomouscore-api)
8. [Python API](#8-python-api)

---

## 1. API Overview

### 1.1 Namespace

All C++ types and classes reside in the `avcs` namespace:

```cpp
#include <sensor_fusion.hpp>
#include <localization.hpp>
#include <slam_system.hpp>
#include <trajectory_tracking.hpp>
#include <vehicle_state_estimator.hpp>
#include <autonomous_core.hpp>

using namespace avcs;  // or qualify explicitly: avcs::SensorFusion
```

### 1.2 General Conventions

| Convention | Description |
|---|---|
| **Naming** | Classes use `PascalCase`; methods use `camelCase`; member variables use `snake_case_` (trailing underscore); enums use `PascalCase` for enumerators. |
| **Parameter passing** | Input parameters are passed as `const T&` for non-trivial types and `T` for scalar types. Output is returned by value. |
| **Ownership** | Subsystems are owned via `std::unique_ptr` by `AutonomousCore`. Raw pointers or references are never transferred to callers. |
| **Error handling** | Methods that can fail return `bool` or `std::optional<T>`. No exceptions are thrown across the public API boundary. |
| **Time representation** | Timestamps are `double` values representing seconds since the Unix epoch. |
| **Linear algebra** | All matrix and vector operations use the **Eigen 3** library (`Eigen::VectorXd`, `Eigen::MatrixXd`, `Eigen::Matrix4d`). |

### 1.3 Error Handling

The AVCS API adopts a no-throw guarantee across all public methods:

- **Initialization failures** — `AutonomousCore::initialize()` returns `false`.
- **Missing data** — `SLAMSystem::detectLoopClosure()` returns `std::nullopt`.
- **Invalid state transitions** — `AutonomousCore::setMode()` silently rejects transitions that violate the state machine (e.g., switching to `AUTONOMOUS` from `FAULT`).
- **Filter divergence** — Internally detected; the filter resets to high-uncertainty initialization. Call `reset()` to explicitly reinitialize any filter.

### 1.4 Thread Safety

| Class | Thread Safety |
|---|---|
| `SensorFusion` | All public methods are thread-safe (internal `std::mutex`). |
| `EKFLocalizer` | All public methods are thread-safe (internal `std::mutex`). |
| `SLAMSystem` | All public methods are thread-safe (internal `std::mutex`). |
| `TrajectoryTracker` | **NOT thread-safe.** External synchronization required. |
| `VehicleStateEstimator` | All public methods are thread-safe (internal `std::mutex`). |
| `AutonomousCore` | All public methods are thread-safe (internal `std::mutex` + `std::condition_variable`). |

### 1.5 C++17 Requirements

The project requires a C++17-compliant compiler. Features used:

- `std::optional` — for optional return values
- `std::variant` / `std::any` — internal use
- Structured bindings and `if constexpr` — internal template logic
- `std::string_view` — internal string handling
- `<mutex>`, `<condition_variable>`, `<thread>` — concurrency primitives

Minimum supported compilers: GCC 9+, Clang 10+, MSVC 19.20+.

---

## 2. SensorFusion API

**Header:** `sensor_fusion.hpp`

The `SensorFusion` class implements a multi-sensor data fusion engine based on an Extended Kalman Filter (EKF). It maintains a 9-dimensional state vector and supports six sensor modalities with configurable sequential or batch fusion modes.

### 2.1 Data Structures

#### `enum class SensorType`

```cpp
enum class SensorType {
    LIDAR,      // Light Detection and Ranging — 3D point cloud range sensor
    RADAR,      // Radio Detection and Ranging — velocity-aware range sensor
    CAMERA,     // Visual camera — appearance-based perception sensor
    GPS,        // Global Positioning System — absolute position sensor
    IMU,        // Inertial Measurement Unit — acceleration and angular rate sensor
    ULTRASONIC  // Ultrasonic — short-range proximity sensor
};
```

#### `struct SensorMeasurement`

| Field | Type | Description |
|---|---|---|
| `sensor_type` | `SensorType` | Type of the source sensor |
| `timestamp` | `double` | Measurement timestamp in seconds (epoch) |
| `data` | `std::vector<double>` | Raw measurement data (dimension varies by sensor) |
| `covariance` | `std::array<double, 9>` | 3×3 covariance matrix in row-major order |
| `sensor_id` | `std::string` | Unique identifier for the sensor instance |

> **Note:** The `covariance` array stores the upper-triangular 3×3 noise covariance associated with the measurement. For sensors that produce higher-dimensional data (e.g., LIDAR with N points), the covariance represents the per-point uncertainty model.

#### `struct FusedState`

| Field | Type | Description |
|---|---|---|
| `position` | `std::array<double, 3>` | `[x, y, z]` position in world frame (meters) |
| `velocity` | `std::array<double, 3>` | `[vx, vy, vz]` velocity in world frame (m/s) |
| `orientation` | `std::array<double, 3>` | `[roll, pitch, yaw]` Euler angles (radians) |
| `covariance` | `Eigen::MatrixXd` | 9×9 full state covariance matrix |

### 2.2 Class Reference

```cpp
class SensorFusion {
public:
    SensorFusion(const std::string& fusion_mode = "sequential",
                 double sync_threshold = 0.05);
    ~SensorFusion() = default;

    // Non-copyable, movable
    SensorFusion(const SensorFusion&) = delete;
    SensorFusion& operator=(const SensorFusion&) = delete;
    SensorFusion(SensorFusion&&) = default;
    SensorFusion& operator=(SensorFusion&&) = default;

    void addMeasurement(const SensorMeasurement& measurement);
    void predict(double dt);
    void update();
    FusedState getFusedState() const;
    void reset();
};
```

### 2.3 Method Details

#### `SensorFusion(const std::string& fusion_mode, double sync_threshold)`

Constructs the fusion engine.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fusion_mode` | `const std::string&` | `"sequential"` | Fusion strategy: `"sequential"` processes measurements one at a time; `"batch"` fuses synchronized groups |
| `sync_threshold` | `double` | `0.05` | Maximum time difference (seconds) between measurements to be considered synchronized for batch fusion |

#### `void addMeasurement(const SensorMeasurement& measurement)`

Adds a new sensor measurement to the internal buffer. In sequential mode, the measurement is queued for the next `update()` call. In batch mode, it is grouped with temporally close measurements.

| Parameter | Type | Description |
|---|---|---|
| `measurement` | `const SensorMeasurement&` | The sensor measurement to ingest |

**Thread safety:** Safe to call from multiple threads concurrently.

#### `void predict(double dt)`

Propagates the 9-dimensional state vector forward by `dt` seconds using the constant-velocity process model with orientation propagation. The state covariance is also propagated using the process noise model.

| Parameter | Type | Description |
|---|---|---|
| `dt` | `double` | Time step in seconds for the prediction |

#### `void update()`

Fuses all buffered measurements into the current state. In sequential mode, each measurement is applied as a separate EKF update step. In batch mode, synchronized groups are fused together using a stacked measurement Jacobian.

#### `FusedState getFusedState() const`

Returns the current fused state estimate including position, velocity, orientation, and the full 9×9 covariance matrix.

**Returns:** `FusedState`

#### `void reset()`

Clears all buffered measurements and resets the state vector and covariance to their initial high-uncertainty values.

### 2.4 Usage Example

```cpp
#include <sensor_fusion.hpp>

int main() {
    // Create a sequential fusion engine with 50ms sync threshold
    avcs::SensorFusion fusion("sequential", 0.05);

    // Ingest a LIDAR measurement
    avcs::SensorMeasurement lidar_meas;
    lidar_meas.sensor_type = avcs::SensorType::LIDAR;
    lidar_meas.timestamp = 1700000000.0;
    lidar_meas.data = {10.5, 2.3, 0.8};
    lidar_meas.covariance = {0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01};
    lidar_meas.sensor_id = "lidar_front";

    fusion.addMeasurement(lidar_meas);

    // Predict forward by 10ms
    fusion.predict(0.01);

    // Fuse buffered measurements
    fusion.update();

    // Retrieve the fused estimate
    avcs::FusedState state = fusion.getFusedState();
    double x = state.position[0];
    double vx = state.velocity[0];
    double yaw = state.orientation[2];

    // Reset the filter
    fusion.reset();

    return 0;
}
```

---

## 3. EKFLocalizer API

**Header:** `localization.hpp`

The `EKFLocalizer` class implements a 15-state Extended Kalman Filter for vehicle localization, fusing GPS, IMU, and LIDAR pose measurements into a robust, high-frequency pose and velocity estimate.

### 3.1 Data Structures

#### `struct Pose3D`

| Field | Type | Description |
|---|---|---|
| `x` | `double` | X position in world frame (meters) |
| `y` | `double` | Y position in world frame (meters) |
| `z` | `double` | Z position in world frame (meters) |
| `roll` | `double` | Roll angle (radians) |
| `pitch` | `double` | Pitch angle (radians) |
| `yaw` | `double` | Yaw angle (radians) |
| `position_covariance` | `std::array<double, 6>` | `[xx, yy, zz, xy, xz, yz]` upper-triangular 3×3 |
| `orientation_covariance` | `std::array<double, 6>` | `[rr, pp, yy, rp, ry, py]` upper-triangular 3×3 |

#### `struct Velocity3D`

| Field | Type | Description |
|---|---|---|
| `vx` | `double` | Linear velocity along x-axis (m/s) |
| `vy` | `double` | Linear velocity along y-axis (m/s) |
| `vz` | `double` | Linear velocity along z-axis (m/s) |
| `wx` | `double` | Angular velocity about x-axis (rad/s) |
| `wy` | `double` | Angular velocity about y-axis (rad/s) |
| `wz` | `double` | Angular velocity about z-axis (rad/s) |

#### `struct GPSData`

| Field | Type | Description |
|---|---|---|
| `lat` | `double` | Latitude (degrees) |
| `lon` | `double` | Longitude (degrees) |
| `alt` | `double` | Altitude above WGS-84 ellipsoid (meters) |
| `accuracy` | `double` | Estimated horizontal accuracy (meters) |
| `num_satellites` | `int` | Number of satellites used in the fix |
| `fix_type` | `int` | Fix type: 0=none, 1=GPS, 2=DGPS, 4=RTK-fixed, 5=RTK-float |

#### `struct IMUData`

| Field | Type | Description |
|---|---|---|
| `accel` | `std::array<double, 3>` | Linear acceleration `[ax, ay, az]` (m/s²) |
| `gyro` | `std::array<double, 3>` | Angular velocity `[wx, wy, wz]` (rad/s) |
| `orientation` | `std::array<double, 3>` | Orientation estimate `[roll, pitch, yaw]` (rad) |
| `timestamp` | `double` | Measurement timestamp in seconds (epoch) |

### 3.2 State Vector Description (15 Dimensions)

The EKFLocalizer maintains a 15-dimensional error-state vector with the following layout:

| Index | Symbol | Description |
|---|---|---|
| 0–2 | `x, y, z` | Position in world frame (meters) |
| 3–5 | `vx, vy, vz` | Velocity in world frame (m/s) |
| 6–8 | `roll, pitch, yaw` | Orientation as Euler angles (radians) |
| 9–11 | `bx, by, bz` | Accelerometer bias (m/s²) |
| 12–14 | `gbx, gby, gbz` | Gyroscope bias (rad/s) |

The error-state formulation linearizes around the current best estimate and is numerically stable for large orientation changes between updates.

### 3.3 Class Reference

```cpp
class EKFLocalizer {
public:
    EKFLocalizer(const Pose3D& initial_pose,
                 double process_noise_pos = 0.1,
                 double process_noise_ori = 0.01);
    ~EKFLocalizer() = default;

    // Non-copyable, movable
    EKFLocalizer(const EKFLocalizer&) = delete;
    EKFLocalizer& operator=(const EKFLocalizer&) = delete;
    EKFLocalizer(EKFLocalizer&&) = default;
    EKFLocalizer& operator=(EKFLocalizer&&) = default;

    void predict(double dt, const IMUData& imu);
    void updateGPS(const GPSData& gps);
    void updateIMU(const IMUData& imu);
    void updateLidar(const Pose3D& lidar_pose);
    Pose3D getPose() const;
    Velocity3D getVelocity() const;
    Eigen::MatrixXd getCovariance() const;
};
```

### 3.4 Method Details

#### `EKFLocalizer(const Pose3D& initial_pose, double process_noise_pos, double process_noise_ori)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_pose` | `const Pose3D&` | — | The starting pose estimate |
| `process_noise_pos` | `double` | `0.1` | Process noise standard deviation for position (meters) |
| `process_noise_ori` | `double` | `0.01` | Process noise standard deviation for orientation (radians) |

#### `void predict(double dt, const IMUData& imu)`

Integrates IMU accelerometer and gyroscope measurements over `dt` to advance the state estimate. Accelerometer and gyroscope biases are estimated and compensated internally.

| Parameter | Type | Description |
|---|---|---|
| `dt` | `double` | Time step in seconds |
| `imu` | `const IMUData&` | IMU measurement data |

**Typical rate:** 100–500 Hz

#### `void updateGPS(const GPSData& gps)`

Applies a Kalman update using the GPS position fix. The measurement model maps the state position to the observation space and corrects the full 15-state vector.

| Parameter | Type | Description |
|---|---|---|
| `gps` | `const GPSData&` | GPS measurement data |

**Typical rate:** 1–10 Hz

#### `void updateIMU(const IMUData& imu)`

Uses the IMU's orientation estimate (typically from an internal AHRS/compass fusion) as an observation to correct the orientation and bias states.

| Parameter | Type | Description |
|---|---|---|
| `imu` | `const IMUData&` | IMU measurement containing orientation data |

#### `void updateLidar(const Pose3D& lidar_pose)`

Applies a Kalman update using a LIDAR scan-matching pose estimate. Both position and orientation components of the LIDAR pose are used.

| Parameter | Type | Description |
|---|---|---|
| `lidar_pose` | `const Pose3D&` | Pose estimate from LIDAR scan matching |

**Typical rate:** 10–20 Hz

#### `Pose3D getPose() const`

Returns the current best pose estimate with covariances.

**Returns:** `Pose3D`

#### `Velocity3D getVelocity() const`

Returns the current velocity estimate with linear and angular components.

**Returns:** `Velocity3D`

#### `Eigen::MatrixXd getCovariance() const`

Returns the full 15×15 state covariance matrix in the state ordering defined in [Section 3.2](#32-state-vector-description-15-dimensions).

**Returns:** 15×15 `Eigen::MatrixXd`

### 3.5 Usage Example

```cpp
#include <localization.hpp>

int main() {
    // Initialize localizer at the origin
    avcs::Pose3D initial_pose;
    initial_pose.x = 0.0;
    initial_pose.y = 0.0;
    initial_pose.z = 0.0;
    initial_pose.roll = 0.0;
    initial_pose.pitch = 0.0;
    initial_pose.yaw = 0.0;
    initial_pose.position_covariance = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    initial_pose.orientation_covariance = {0.1, 0.1, 0.1, 0.0, 0.0, 0.0};

    avcs::EKFLocalizer localizer(initial_pose, 0.1, 0.01);

    // IMU prediction at 100 Hz
    avcs::IMUData imu;
    imu.accel = {0.0, 0.0, 9.81};
    imu.gyro = {0.0, 0.0, 0.01};
    imu.orientation = {0.0, 0.0, 0.0};
    imu.timestamp = 1700000000.0;

    localizer.predict(0.01, imu);

    // GPS correction at 5 Hz
    avcs::GPSData gps;
    gps.lat = 37.7749;
    gps.lon = -122.4194;
    gps.alt = 10.0;
    gps.accuracy = 2.0;
    gps.num_satellites = 12;
    gps.fix_type = 4;  // RTK-fixed

    localizer.updateGPS(gps);

    // LIDAR correction at 10 Hz
    avcs::Pose3D lidar_pose;
    lidar_pose.x = 1.2;
    lidar_pose.y = 0.3;
    lidar_pose.z = 0.0;
    lidar_pose.roll = 0.0;
    lidar_pose.pitch = 0.0;
    lidar_pose.yaw = 0.05;
    lidar_pose.position_covariance = {0.05, 0.05, 0.05, 0.0, 0.0, 0.0};
    lidar_pose.orientation_covariance = {0.01, 0.01, 0.01, 0.0, 0.0, 0.0};

    localizer.updateLidar(lidar_pose);

    // Retrieve estimates
    avcs::Pose3D pose = localizer.getPose();
    avcs::Velocity3D vel = localizer.getVelocity();
    Eigen::MatrixXd cov = localizer.getCovariance();

    return 0;
}
```

---

## 4. SLAMSystem API

**Header:** `slam_system.hpp`

The `SLAMSystem` class implements a graph-based SLAM pipeline with ICP scan matching, keyframe management, loop closure detection, and pose graph optimization.

### 4.1 Data Structures

#### `struct Point3D`

| Field | Type | Description |
|---|---|---|
| `x` | `double` | X coordinate (meters) |
| `y` | `double` | Y coordinate (meters) |
| `z` | `double` | Z coordinate (meters) |
| `intensity` | `double` | Return signal intensity (0.0–1.0 normalized) |

#### `struct KeyFrame`

| Field | Type | Description |
|---|---|---|
| `frame_id` | `uint64_t` | Unique identifier for this keyframe |
| `pose` | `Eigen::Matrix4d` | 4×4 homogeneous pose in world frame |
| `timestamp` | `double` | Timestamp when the keyframe was created (seconds) |
| `point_cloud` | `std::vector<Point3D>` | Associated 3D point cloud in sensor frame |

#### `struct LoopClosureResult`

| Field | Type | Description |
|---|---|---|
| `frame_id_a` | `uint64_t` | ID of the first keyframe |
| `frame_id_b` | `uint64_t` | ID of the second keyframe |
| `relative_transform` | `Eigen::Matrix4d` | Relative transform from frame A to frame B |
| `confidence` | `double` | Match confidence score [0.0, 1.0]; values > 0.8 are reliable |

### 4.2 Class Reference

```cpp
class SLAMSystem {
public:
    SLAMSystem(double icp_max_correspondence_dist = 0.1,
               double icp_max_iterations = 50.0,
               double keyframe_translation_thresh = 0.5,
               double keyframe_rotation_thresh = 0.1);
    ~SLAMSystem() = default;

    // Non-copyable, movable
    SLAMSystem(const SLAMSystem&) = delete;
    SLAMSystem& operator=(const SLAMSystem&) = delete;
    SLAMSystem(SLAMSystem&&) = default;
    SLAMSystem& operator=(SLAMSystem&&) = default;

    void initialize(const Pose3D& initial_pose);
    Pose3D update(const std::vector<Point3D>& scan, double timestamp);
    std::optional<LoopClosureResult> detectLoopClosure();
    void optimizePoseGraph();
    std::vector<KeyFrame> getKeyFrames() const;
    Pose3D getCurrentPose() const;
};
```

### 4.3 Method Details

#### `SLAMSystem(double icp_max_correspondence_dist, double icp_max_iterations, double keyframe_translation_thresh, double keyframe_rotation_thresh)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `icp_max_correspondence_dist` | `double` | `0.1` | Maximum point-to-point distance (meters) for ICP correspondence matching |
| `icp_max_iterations` | `double` | `50.0` | Maximum number of ICP iterations per scan |
| `keyframe_translation_thresh` | `double` | `0.5` | Minimum translation (meters) between keyframes |
| `keyframe_rotation_thresh` | `double` | `0.1` | Minimum rotation (radians) between keyframes |

#### `void initialize(const Pose3D& initial_pose)`

Creates the first keyframe at the given pose and initializes the local map. **Must** be called before `update()`.

| Parameter | Type | Description |
|---|---|---|
| `initial_pose` | `const Pose3D&` | The starting pose of the vehicle |

#### `Pose3D update(const std::vector<Point3D>& scan, double timestamp)`

Processes a new LIDAR scan via ICP scan matching against the local map. Creates a new keyframe if the motion thresholds are exceeded.

| Parameter | Type | Description |
|---|---|---|
| `scan` | `const std::vector<Point3D>&` | 3D points from the LIDAR scan (sensor frame) |
| `timestamp` | `double` | Timestamp of the scan in seconds (epoch) |

**Returns:** `Pose3D` — the estimated current pose after scan matching.

#### `std::optional<LoopClosureResult> detectLoopClosure()`

Compares the most recent keyframe against all previous keyframes (beyond a minimum temporal gap) using feature-based matching.

**Returns:** `std::optional<LoopClosureResult>` — the loop closure result if detected, `std::nullopt` otherwise.

#### `void optimizePoseGraph()`

Runs a global nonlinear least-squares optimization (Gauss-Newton / Levenberg-Marquardt) over the entire pose graph using odometry and loop closure edges as constraints. After optimization, all keyframe poses and the local map are updated.

#### `std::vector<KeyFrame> getKeyFrames() const`

Returns all keyframes in the pose graph.

**Returns:** `std::vector<KeyFrame>`

#### `Pose3D getCurrentPose() const`

Returns the current estimated pose of the vehicle.

**Returns:** `Pose3D`

### 4.4 Usage Example

```cpp
#include <slam_system.hpp>
#include <localization.hpp>

int main() {
    avcs::Pose3D origin;
    origin.x = origin.y = origin.z = 0.0;
    origin.roll = origin.pitch = origin.yaw = 0.0;
    origin.position_covariance = {1, 1, 1, 0, 0, 0};
    origin.orientation_covariance = {0.1, 0.1, 0.1, 0, 0, 0};

    avcs::SLAMSystem slam(0.1, 50.0, 0.5, 0.1);
    slam.initialize(origin);

    // Simulate incoming LIDAR scans
    for (int i = 0; i < 1000; ++i) {
        std::vector<avcs::Point3D> scan = getLidarScan();  // user-defined
        double t = 1700000000.0 + i * 0.1;

        avcs::Pose3D pose = slam.update(scan, t);

        // Periodically check for loop closures
        if (i % 50 == 0) {
            if (auto loop = slam.detectLoopClosure()) {
                std::cout << "Loop closure: frame " << loop->frame_id_a
                          << " <-> " << loop->frame_id_b
                          << " (confidence=" << loop->confidence << ")\n";
                slam.optimizePoseGraph();
            }
        }
    }

    // Inspect keyframes
    auto keyframes = slam.getKeyFrames();
    std::cout << "Total keyframes: " << keyframes.size() << "\n";

    return 0;
}
```

---

## 5. TrajectoryTracker API

**Header:** `trajectory_tracking.hpp`

The `TrajectoryTracker` class computes steering, throttle, and brake commands to guide the vehicle along a planned reference trajectory using decoupled lateral (PD) and longitudinal (PI) control.

### 5.1 Data Structures

#### `struct TrajectoryPoint`

| Field | Type | Description |
|---|---|---|
| `x` | `double` | X position in world frame (meters) |
| `y` | `double` | Y position in world frame (meters) |
| `z` | `double` | Z position in world frame (meters) |
| `yaw` | `double` | Heading angle (radians) |
| `velocity` | `double` | Desired longitudinal speed (m/s) |
| `acceleration` | `double` | Desired longitudinal acceleration (m/s²) |
| `curvature` | `double` | Path curvature at this point (1/meters) |
| `timestamp` | `double` | Time at which the vehicle should reach this point (seconds) |

#### `struct TrackingError`

| Field | Type | Description |
|---|---|---|
| `lateral_error` | `double` | Cross-track error (meters); positive = left of path |
| `longitudinal_error` | `double` | Along-track error (meters); positive = ahead of target |
| `heading_error` | `double` | Heading deviation (radians); positive = pointing left |
| `curvature_error` | `double` | Curvature deviation (1/meters) |

#### `struct ControlCommand`

| Field | Type | Description |
|---|---|---|
| `steering_angle` | `double` | Front wheel steering angle (radians); positive = left |
| `throttle` | `double` | Throttle position [0.0, 1.0]; 0 = no throttle |
| `brake` | `double` | Brake pressure [0.0, 1.0]; 0 = no braking |
| `gear` | `int` | Gear selection: -1=reverse, 0=neutral, 1=drive |

### 5.2 Class Reference

```cpp
class TrajectoryTracker {
public:
    TrajectoryTracker(double kp_lateral = 1.5,
                      double kd_lateral = 0.3,
                      double kp_longitudinal = 1.0,
                      double ki_longitudinal = 0.1);
    ~TrajectoryTracker() = default;

    // Copyable and movable
    TrajectoryTracker(const TrajectoryTracker&) = default;
    TrajectoryTracker& operator=(const TrajectoryTracker&) = default;
    TrajectoryTracker(TrajectoryTracker&&) = default;
    TrajectoryTracker& operator=(TrajectoryTracker&&) = default;

    ControlCommand computeControl(const Pose3D& current_pose,
                                  const Velocity3D& current_vel,
                                  const std::vector<TrajectoryPoint>& trajectory,
                                  size_t target_idx);
    TrackingError computeError(const Pose3D& pose, const TrajectoryPoint& target);
    size_t findClosestPoint(const Pose3D& pose,
                            const std::vector<TrajectoryPoint>& trajectory);
    void setGains(double kp_lat, double kd_lat, double kp_lon, double ki_lon);
};
```

### 5.3 Method Details

#### `TrajectoryTracker(double kp_lateral, double kd_lateral, double kp_longitudinal, double ki_longitudinal)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `kp_lateral` | `double` | `1.5` | Proportional gain for lateral (steering) control |
| `kd_lateral` | `double` | `0.3` | Derivative gain for lateral (steering) control |
| `kp_longitudinal` | `double` | `1.0` | Proportional gain for longitudinal (speed) control |
| `ki_longitudinal` | `double` | `0.1` | Integral gain for longitudinal (speed) control |

#### `ControlCommand computeControl(const Pose3D& current_pose, const Velocity3D& current_vel, const std::vector<TrajectoryPoint>& trajectory, size_t target_idx)`

Determines the closest trajectory point, calculates tracking errors, and applies PD/PI control laws.

| Parameter | Type | Description |
|---|---|---|
| `current_pose` | `const Pose3D&` | Current estimated pose of the vehicle |
| `current_vel` | `const Velocity3D&` | Current velocity of the vehicle |
| `trajectory` | `const std::vector<TrajectoryPoint>&` | The reference trajectory |
| `target_idx` | `size_t` | Index of the target trajectory point (hint for search) |

**Returns:** `ControlCommand` with steering, throttle, brake, and gear.

#### `TrackingError computeError(const Pose3D& pose, const TrajectoryPoint& target)`

Decomposes the error into lateral, longitudinal, heading, and curvature components in the Frenet frame.

| Parameter | Type | Description |
|---|---|---|
| `pose` | `const Pose3D&` | Current vehicle pose |
| `target` | `const TrajectoryPoint&` | Target trajectory point |

**Returns:** `TrackingError`

#### `size_t findClosestPoint(const Pose3D& pose, const std::vector<TrajectoryPoint>& trajectory)`

Performs nearest-neighbor search over the trajectory to find the point with minimum Euclidean distance.

| Parameter | Type | Description |
|---|---|---|
| `pose` | `const Pose3D&` | Current vehicle pose |
| `trajectory` | `const std::vector<TrajectoryPoint>&` | The reference trajectory |

**Returns:** `size_t` — index of the closest trajectory point.

#### `void setGains(double kp_lat, double kd_lat, double kp_lon, double ki_lon)`

Updates the controller gains at runtime. Useful for adaptive gain scheduling based on speed or road conditions.

| Parameter | Type | Description |
|---|---|---|
| `kp_lat` | `double` | New proportional gain for lateral control |
| `kd_lat` | `double` | New derivative gain for lateral control |
| `kp_lon` | `double` | New proportional gain for longitudinal control |
| `ki_lon` | `double` | New integral gain for longitudinal control |

### 5.4 Usage Example

```cpp
#include <trajectory_tracking.hpp>
#include <localization.hpp>

int main() {
    avcs::TrajectoryTracker tracker(1.5, 0.3, 1.0, 0.1);

    // Build a simple reference trajectory (straight line, 10 m/s)
    std::vector<avcs::TrajectoryPoint> trajectory;
    for (int i = 0; i < 100; ++i) {
        avcs::TrajectoryPoint pt;
        pt.x = i * 1.0;
        pt.y = 0.0;
        pt.z = 0.0;
        pt.yaw = 0.0;
        pt.velocity = 10.0;
        pt.acceleration = 0.0;
        pt.curvature = 0.0;
        pt.timestamp = i * 0.1;
        trajectory.push_back(pt);
    }

    // Current vehicle state
    avcs::Pose3D pose;
    pose.x = 5.0; pose.y = 0.2; pose.z = 0.0;
    pose.roll = 0.0; pose.pitch = 0.0; pose.yaw = 0.02;

    avcs::Velocity3D vel;
    vel.vx = 9.5; vel.vy = 0.0; vel.vz = 0.0;
    vel.wx = 0.0; vel.wy = 0.0; vel.wz = 0.01;

    // Find closest point and compute control
    size_t idx = tracker.findClosestPoint(pose, trajectory);
    avcs::ControlCommand cmd = tracker.computeControl(pose, vel, trajectory, idx);

    std::cout << "Steering: " << cmd.steering_angle << " rad\n"
              << "Throttle: " << cmd.throttle << "\n"
              << "Brake:    " << cmd.brake << "\n"
              << "Gear:     " << cmd.gear << "\n";

    // Compute tracking error for monitoring
    avcs::TrackingError err = tracker.computeError(pose, trajectory[idx]);
    std::cout << "Lateral error: " << err.lateral_error << " m\n"
              << "Heading error: " << err.heading_error << " rad\n";

    // Adapt gains for high-speed driving
    tracker.setGains(0.8, 0.2, 1.2, 0.05);

    return 0;
}
```

---

## 6. VehicleStateEstimator API

**Header:** `vehicle_state_estimator.hpp`

The `VehicleStateEstimator` class fuses IMU, wheel odometry, and GPS data using an EKF with a bicycle kinematic model for the prediction step.

### 6.1 Data Structures

#### `struct VehicleState`

| Field | Type | Description |
|---|---|---|
| `position` | `std::array<double, 3>` | `[x, y, z]` position in world frame (meters) |
| `velocity` | `std::array<double, 3>` | `[vx, vy, vz]` velocity in world frame (m/s) |
| `acceleration` | `std::array<double, 3>` | `[ax, ay, az]` acceleration in world frame (m/s²) |
| `orientation` | `std::array<double, 3>` | `[roll, pitch, yaw]` Euler angles (radians) |
| `angular_velocity` | `std::array<double, 3>` | `[wx, wy, wz]` angular velocity in body frame (rad/s) |
| `steering_angle` | `double` | Current front wheel steering angle (radians) |
| `throttle` | `double` | Current throttle position [0.0, 1.0] |
| `brake` | `double` | Current brake pressure [0.0, 1.0] |
| `gear` | `int` | Current gear: -1=reverse, 0=neutral, 1=drive |
| `timestamp` | `double` | State timestamp in seconds (epoch) |

### 6.2 Class Reference

```cpp
class VehicleStateEstimator {
public:
    VehicleStateEstimator(double wheelbase = 2.8,
                          double mass = 1500.0,
                          double inertia = 2500.0);
    ~VehicleStateEstimator() = default;

    // Non-copyable, movable
    VehicleStateEstimator(const VehicleStateEstimator&) = delete;
    VehicleStateEstimator& operator=(const VehicleStateEstimator&) = delete;
    VehicleStateEstimator(VehicleStateEstimator&&) = default;
    VehicleStateEstimator& operator=(VehicleStateEstimator&&) = default;

    void updateIMU(const IMUData& imu, double timestamp);
    void updateOdometry(double wheel_speed_fl, double wheel_speed_fr,
                        double wheel_speed_rl, double wheel_speed_rr,
                        double steering_angle, double timestamp);
    void updateGPS(const GPSData& gps);
    VehicleState getState() const;
    Eigen::MatrixXd getStateCovariance() const;
    void reset(const VehicleState& initial_state);
};
```

### 6.3 Method Details

#### `VehicleStateEstimator(double wheelbase, double mass, double inertia)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `wheelbase` | `double` | `2.8` | Distance between front and rear axles (meters) |
| `mass` | `double` | `1500.0` | Total vehicle mass including payload (kilograms) |
| `inertia` | `double` | `2500.0` | Yaw moment of inertia (kg·m²) |

#### `void updateIMU(const IMUData& imu, double timestamp)`

Performs the EKF prediction step by integrating IMU accelerometer and gyroscope readings.

| Parameter | Type | Description |
|---|---|---|
| `imu` | `const IMUData&` | IMU measurement data |
| `timestamp` | `double` | Current timestamp in seconds (epoch) |

**Typical rate:** 100–500 Hz

#### `void updateOdometry(double wheel_speed_fl, double wheel_speed_fr, double wheel_speed_rl, double wheel_speed_rr, double steering_angle, double timestamp)`

Uses four wheel speed measurements and steering angle to compute a longitudinal velocity and yaw rate estimate, then applies an EKF correction step.

| Parameter | Type | Description |
|---|---|---|
| `wheel_speed_fl` | `double` | Front-left wheel speed (m/s) |
| `wheel_speed_fr` | `double` | Front-right wheel speed (m/s) |
| `wheel_speed_rl` | `double` | Rear-left wheel speed (m/s) |
| `wheel_speed_rr` | `double` | Rear-right wheel speed (m/s) |
| `steering_angle` | `double` | Current steering angle (radians) |
| `timestamp` | `double` | Current timestamp in seconds (epoch) |

**Typical rate:** 50–100 Hz

#### `void updateGPS(const GPSData& gps)`

Applies an EKF correction step using the GPS position fix to constrain position states and reduce drift.

| Parameter | Type | Description |
|---|---|---|
| `gps` | `const GPSData&` | GPS measurement data |

**Typical rate:** 1–10 Hz

#### `VehicleState getState() const`

Returns the current vehicle state estimate.

**Returns:** `VehicleState`

#### `Eigen::MatrixXd getStateCovariance() const`

Returns the full state covariance matrix.

**Returns:** `Eigen::MatrixXd`

#### `void reset(const VehicleState& initial_state)`

Clears the state history buffer and re-initializes the EKF state vector and covariance from the provided initial state.

| Parameter | Type | Description |
|---|---|---|
| `initial_state` | `const VehicleState&` | The state to initialize from |

### 6.4 Usage Example

```cpp
#include <vehicle_state_estimator.hpp>
#include <localization.hpp>

int main() {
    avcs::VehicleStateEstimator estimator(2.8, 1500.0, 2500.0);

    double t = 1700000000.0;

    // High-frequency IMU updates
    avcs::IMUData imu;
    imu.accel = {0.5, 0.0, 9.81};
    imu.gyro = {0.0, 0.0, 0.02};
    imu.orientation = {0.0, 0.0, 0.01};
    imu.timestamp = t;

    estimator.updateIMU(imu, t);

    // Medium-frequency odometry updates
    estimator.updateOdometry(8.5, 8.6, 8.4, 8.5, 0.02, t);

    // Low-frequency GPS updates
    avcs::GPSData gps;
    gps.lat = 37.7749;
    gps.lon = -122.4194;
    gps.alt = 10.0;
    gps.accuracy = 1.5;
    gps.num_satellites = 14;
    gps.fix_type = 4;

    estimator.updateGPS(gps);

    // Retrieve full state
    avcs::VehicleState state = estimator.getState();
    std::cout << "Position: [" << state.position[0] << ", "
              << state.position[1] << ", " << state.position[2] << "]\n"
              << "Velocity: [" << state.velocity[0] << ", "
              << state.velocity[1] << ", " << state.velocity[2] << "]\n"
              << "Heading:  " << state.orientation[2] << " rad\n";

    // Reset to a known state if needed
    avcs::VehicleState init;
    init.position = {0, 0, 0};
    init.velocity = {0, 0, 0};
    init.acceleration = {0, 0, 0};
    init.orientation = {0, 0, 0};
    init.angular_velocity = {0, 0, 0};
    init.steering_angle = 0;
    init.throttle = 0;
    init.brake = 0;
    init.gear = 0;
    init.timestamp = t;

    estimator.reset(init);

    return 0;
}
```

---

## 7. AutonomousCore API

**Header:** `autonomous_core.hpp`

The `AutonomousCore` class is the top-level orchestrator that coordinates all AVCS subsystems, manages the system lifecycle, enforces mode/state transitions, and runs the main control loop.

### 7.1 Data Structures

#### `enum class AutonomousMode`

```cpp
enum class AutonomousMode {
    MANUAL,           // Human driver has full control; system monitors only
    AUTONOMOUS,       // Full autonomous driving; system has complete control
    SEMI_AUTONOMOUS,  // Shared control; system assists the human driver
    EMERGENCY,        // Emergency override; system brings vehicle to safe stop
    PARKING           // Autonomous parking mode; low-speed maneuvering
};
```

#### `enum class SystemState`

```cpp
enum class SystemState {
    INITIALIZING,    // System is booting up; subsystems being configured
    READY,           // All subsystems initialized; vehicle ready to drive
    DRIVING,         // Vehicle is actively driving in autonomous mode
    EMERGENCY_STOP,  // Emergency stop activated; vehicle decelerating to halt
    FAULT,           // Critical fault detected; system in fail-safe mode
    SHUTDOWN         // Graceful shutdown in progress
};
```

**Valid state transitions:**

```
INITIALIZING → READY → DRIVING
DRIVING → READY
Any state → EMERGENCY_STOP
Any state → FAULT
Any state → SHUTDOWN
```

#### `struct SystemConfig`

| Field | Type | Description |
|---|---|---|
| `fusion_mode` | `std::string` | `"sequential"` or `"batch"` |
| `sync_threshold` | `double` | Sensor sync threshold (seconds) |
| `process_noise_pos` | `double` | Position process noise (meters) |
| `process_noise_ori` | `double` | Orientation process noise (radians) |
| `icp_max_correspondence_dist` | `double` | ICP correspondence distance (meters) |
| `icp_max_iterations` | `double` | Max ICP iterations |
| `keyframe_translation_thresh` | `double` | Keyframe translation threshold (meters) |
| `keyframe_rotation_thresh` | `double` | Keyframe rotation threshold (radians) |
| `kp_lateral` | `double` | Lateral proportional gain |
| `kd_lateral` | `double` | Lateral derivative gain |
| `kp_longitudinal` | `double` | Longitudinal proportional gain |
| `ki_longitudinal` | `double` | Longitudinal integral gain |
| `wheelbase` | `double` | Vehicle wheelbase (meters) |
| `mass` | `double` | Vehicle mass (kilograms) |
| `inertia` | `double` | Yaw moment of inertia (kg·m²) |
| `control_loop_rate` | `double` | Main control loop frequency (Hz) |
| `max_speed` | `double` | Maximum allowed speed (m/s) |
| `emergency_decel` | `double` | Emergency deceleration (m/s²) |
| `log_level` | `std::string` | Logging verbosity: `"debug"`, `"info"`, `"warn"`, `"error"` |

### 7.2 Class Reference

```cpp
class AutonomousCore {
public:
    explicit AutonomousCore(const std::string& config_path);
    ~AutonomousCore();

    // Non-copyable, non-movable
    AutonomousCore(const AutonomousCore&) = delete;
    AutonomousCore& operator=(const AutonomousCore&) = delete;
    AutonomousCore(AutonomousCore&&) = delete;
    AutonomousCore& operator=(AutonomousCore&&) = delete;

    bool initialize();
    void run();
    void shutdown();
    void setMode(AutonomousMode mode);
    AutonomousMode getMode() const;
    SystemState getSystemState() const;
    void emergencyStop();

    // Callbacks
    void onPerceptionUpdate(const FusedState& objects);
    void onLocalizationUpdate(const Pose3D& pose);
    void onPlanningUpdate(const std::vector<TrajectoryPoint>& trajectory);
};
```

### 7.3 Method Details

#### `explicit AutonomousCore(const std::string& config_path)`

Constructs the core orchestrator. The configuration file is loaded during `initialize()`. The constructor only stores the path and sets the initial system state to `INITIALIZING`.

| Parameter | Type | Description |
|---|---|---|
| `config_path` | `const std::string&` | Path to the JSON/YAML configuration file |

#### `bool initialize()`

Loads the configuration file, creates subsystem instances with the configured parameters, and performs self-tests. Must be called before `run()`.

**Returns:** `true` if initialization succeeded, `false` on any failure.

#### `void run()`

Spawns the control loop thread and begins autonomous operation. This method **blocks** until `shutdown()` is called from another thread or an unrecoverable fault occurs.

The control loop executes the following at each iteration:
1. Checks for emergency conditions
2. Collects latest sensor data from fusion
3. Updates localization and SLAM
4. Runs trajectory tracking to compute control commands
5. Sends commands to the vehicle interface
6. Updates the vehicle state estimator

#### `void shutdown()`

Signals the control loop to stop, waits for the thread to join, and releases all subsystem resources. After this call, the system state is `SHUTDOWN`.

#### `void setMode(AutonomousMode mode)`

Requests a mode transition. The transition may be silently rejected if it violates the state machine constraints (e.g., switching to `AUTONOMOUS` while in `FAULT` state).

| Parameter | Type | Description |
|---|---|---|
| `mode` | `AutonomousMode` | The requested autonomous mode |

#### `AutonomousMode getMode() const`

Returns the current autonomous driving mode.

**Returns:** `AutonomousMode`

#### `SystemState getSystemState() const`

Returns the current system lifecycle state.

**Returns:** `SystemState`

#### `void emergencyStop()`

Immediately transitions the system to `EMERGENCY` mode and `EMERGENCY_STOP` state. The vehicle will decelerate to a halt using maximum braking. This operation **cannot** be overridden by `setMode()` — only a full system reset can clear the emergency state.

#### `void onPerceptionUpdate(const FusedState& objects)`

Callback for the perception pipeline. Called when new fused sensor data (detected objects, obstacles, etc.) is available. The data is forwarded to the trajectory tracker and SLAM system.

| Parameter | Type | Description |
|---|---|---|
| `objects` | `const FusedState&` | The fused perception state |

#### `void onLocalizationUpdate(const Pose3D& pose)`

Callback for the localization subsystem. Called when the localizer produces a new pose estimate. The pose is forwarded to the trajectory tracker and SLAM system.

| Parameter | Type | Description |
|---|---|---|
| `pose` | `const Pose3D&` | The updated vehicle pose |

#### `void onPlanningUpdate(const std::vector<TrajectoryPoint>& trajectory)`

Callback for the planning module. Called when a new reference trajectory is generated. The trajectory is stored and used by the trajectory tracker.

| Parameter | Type | Description |
|---|---|---|
| `trajectory` | `const std::vector<TrajectoryPoint>&` | The planned trajectory |

### 7.4 Usage Example

```cpp
#include <autonomous_core.hpp>

int main() {
    // Construct with configuration file
    avcs::AutonomousCore core("/etc/avcs/config.json");

    // Initialize all subsystems
    if (!core.initialize()) {
        std::cerr << "Failed to initialize AVCS\n";
        return 1;
    }

    // Verify system is ready
    if (core.getSystemState() != avcs::SystemState::READY) {
        std::cerr << "System not ready\n";
        return 1;
    }

    // Switch to autonomous mode
    core.setMode(avcs::AutonomousMode::AUTONOMOUS);

    // Run the main control loop (blocks until shutdown)
    // In a real deployment, sensors would call the callbacks
    // from separate threads:
    //   core.onPerceptionUpdate(fused_state);
    //   core.onLocalizationUpdate(pose);
    //   core.onPlanningUpdate(trajectory);
    core.run();

    // Graceful shutdown
    core.shutdown();

    return 0;
}
```

### 7.5 Emergency Stop Flow

```cpp
// From any thread, at any time:
core.emergencyStop();

// This immediately:
// 1. Sets mode to AutonomousMode::EMERGENCY
// 2. Sets state to SystemState::EMERGENCY_STOP
// 3. Applies maximum braking deceleration
// 4. Ignores all subsequent setMode() calls
// 5. Only a full system restart can clear the emergency state
```

---

## 8. Python API

The AVCS project also provides Python modules for perception, localization, planning, control, communication, and simulation. All modules require Python 3.8+.

### 8.1 Perception (`perception`)

```python
from perception import ObjectDetector, ObjectTracker, SensorFusionNode, LidarProcessor, CameraProcessor
```

| Class | Description |
|---|---|
| `ObjectDetector` | Deep-learning-based object detection from camera and LIDAR data |
| `ObjectTracker` | Multi-object tracking with data association and track management |
| `SensorFusionNode` | ROS-style sensor fusion node combining detection outputs |
| `LidarProcessor` | Point cloud filtering, segmentation, and feature extraction |
| `CameraProcessor` | Image preprocessing, undistortion, and feature detection |

### 8.2 Localization (`localization`)

```python
from localization import EKFLocalizer, GPSProcessor, IMUProcessor, MapManager, SLAMNode
```

| Class | Description |
|---|---|
| `EKFLocalizer` | Python EKF localizer fusing GPS, IMU, and LIDAR (mirrors C++ API) |
| `GPSProcessor` | GPS NMEA parsing, coordinate transforms, and quality filtering |
| `IMUProcessor` | IMU data conditioning, bias estimation, and AHRS integration |
| `MapManager` | HD map loading, query, and lane-level routing |
| `SLAMNode` | Graph-SLAM wrapper with visualization and map storage |

### 8.3 Planning (`planning`)

```python
from planning import (
    PathPlanner, AStarPlanner, LatticePlanner, Path, HeuristicType, smooth_path,
    BehaviorPlanner, DrivingState, Maneuver, ManeuverType, ObstacleInfo, RoadContext,
    TrajectoryOptimizer, Trajectory, TrajectoryPoint, OptimizerConfig,
    CostCalculator, CostBreakdown, CostComponent, CostWeights, DrivingScenario,
)
```

| Class / Type | Description |
|---|---|
| `AStarPlanner` / `PathPlanner` | A* path planning on occupancy grids with configurable heuristics |
| `LatticePlanner` | Lattice-based motion planning for structured roads |
| `BehaviorPlanner` | FSM-based behavior planner for lane change, turn, and stop decisions |
| `TrajectoryOptimizer` | Quadratic programming trajectory optimization with kinematic constraints |
| `CostCalculator` | Weighted multi-component cost computation with scenario presets |

### 8.4 Control (`control`)

```python
from control import (
    PIDController, PIDGains,
    MPCController, MPCWeights, MPCConstraints,
    StanleyController, StanleyParams,
    PurePursuitController, PurePursuitParams,
    BicycleModel, VehicleParameters,
)
```

| Class | Description |
|---|---|
| `PIDController` | Versatile PID with anti-windup and derivative filtering |
| `MPCController` | Model Predictive Control with bicycle model and QP solver |
| `StanleyController` | Front-axle lateral controller with cross-track error correction |
| `PurePursuitController` | Classic lookahead lateral path follower |
| `BicycleModel` | Kinematic and dynamic bicycle vehicle model |

### 8.5 Communication (`communication`)

```python
from communication import V2XNode, MessageParser, DSRCInterface, CAMPProcessor
```

| Class | Description |
|---|---|
| `V2XNode` | Central V2X communication hub with pub/sub messaging and peer management |
| `MessageParser` | J2735 DSRC message parsing/serialization for BSM, MAP, and SPAT |
| `DSRCInterface` | Low-level 5.9 GHz DSRC radio interface with WAVE protocol support |
| `CAMPProcessor` | Cooperative Awareness Message processing and broadcasting |

### 8.6 Simulation (`simulation`)

```python
from simulation import CARLASimulator, SUMOInterface, ScenarioRunner, DataRecorder
```

| Class | Description |
|---|---|
| `CARLASimulator` | Interface to the CARLA autonomous driving simulator |
| `SUMOInterface` | Interface to SUMO microscopic traffic simulation |
| `ScenarioRunner` | Framework for executing and evaluating test scenarios |
| `DataRecorder` | Recording and playback of simulation data streams |

---

## Appendix A: Include Dependencies

All headers are self-contained and internally include the necessary Eigen headers. The dependency graph is:

```
autonomous_core.hpp
  ├── sensor_fusion.hpp       (Eigen, <array>, <vector>, <mutex>)
  ├── localization.hpp        (Eigen, <array>, <mutex>)
  ├── slam_system.hpp         (Eigen, <optional>, <vector>, <mutex>)
  ├── trajectory_tracking.hpp (<vector>, <cstddef>)
  └── vehicle_state_estimator.hpp (Eigen, <deque>, <mutex>)
```

> **Note:** `slam_system.hpp` and `trajectory_tracking.hpp` forward-declare types from `localization.hpp` (`Pose3D`, `Velocity3D`, `GPSData`, `IMUData`). You must include `localization.hpp` before using these headers if you need the full type definitions.

## Appendix B: CMake Integration

```cmake
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(avcs_node
    src/main.cpp
    src/sensor_fusion.cpp
    src/localization.cpp
    src/slam_system.cpp
    src/lidar_processing.cpp
    src/radar_processing.cpp
)

target_include_directories(avcs_node PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(avcs_node Eigen3::Eigen)
target_compile_features(avcs_node PUBLIC cxx_std_17)
```

## Appendix C: Quick-Start Minimal Program

```cpp
#include <autonomous_core.hpp>

int main() {
    avcs::AutonomousCore core("config.json");
    if (!core.initialize()) return 1;
    core.setMode(avcs::AutonomousMode::AUTONOMOUS);
    core.run();
    core.shutdown();
    return 0;
}
```
