# Autonomous Vehicle Control System — System Architecture Document

**Version:** 2.0.0  
**Date:** 2026-03-04  
**Status:** Approved  
**Classification:** Internal — Engineering  

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [Perception Module](#3-perception-module)
4. [Localization Module](#4-localization-module)
5. [Planning Module](#5-planning-module)
6. [Control Module](#6-control-module)
7. [Communication Module](#7-communication-module)
8. [Real-Time Engine](#8-real-time-engine)
9. [Data Flow](#9-data-flow)
10. [Safety Architecture](#10-safety-architecture)

---

## 1. Overview

### 1.1 System Purpose

The Autonomous Vehicle Control System (AVCS) is a fully integrated software platform designed to enable SAE Level 4 autonomous driving capabilities in structured and semi-structured environments. The system processes multi-modal sensor data in real time, builds and maintains a coherent world model, generates safe and efficient trajectory plans, and executes precise vehicle control commands — all while satisfying stringent safety, latency, and reliability requirements.

The AVCS operates as a closed-loop cyber-physical system: it continuously perceives the environment, localizes the ego vehicle within a high-definition map, reasons about the intentions and future states of other traffic participants, plans a collision-free trajectory, and actuates the vehicle's steering, throttle, and brake systems. The entire perception-to-actuator pipeline must complete within a bounded worst-case execution time of 100 ms under nominal operating conditions, and the control loop must execute at 100 Hz to maintain vehicle stability at highway speeds.

### 1.2 Design Philosophy

The AVCS architecture is guided by five foundational design principles, each of which shapes the module boundaries, interface contracts, threading model, and fault-tolerance strategy of the system.

#### 1.2.1 Modularity

The system is decomposed into six primary modules — Perception, Localization, Planning, Control, Communication, and Simulation — each with clearly defined responsibilities and well-specified inter-module interfaces. Modules communicate exclusively through typed message contracts published to a shared real-time bus, with no direct memory sharing or hidden coupling. This decomposition allows independent development, testing, and replacement of modules. For example, the Perception module can be retrained and redeployed without modifying the Planning or Control modules, provided the output message schema remains compatible. Each module exposes a versioned API and advertises its capabilities through a module descriptor that the system manager consults at startup.

#### 1.2.2 Real-Time Performance

Every processing stage in the AVCS operates under hard or firm real-time constraints. The control loop runs at 100 Hz with a deadline of 8 ms per iteration; the perception pipeline must produce updated object lists within 50 ms; the planning module must emit a new trajectory within 100 ms. The system employs priority-based preemptive scheduling, lock-free ring buffers for inter-thread communication, and memory pre-allocation to avoid garbage collection pauses or page faults during critical processing windows. Deadline monitoring is continuous, and violations trigger escalating alarms from logging through graceful degradation to emergency stop.

#### 1.2.3 Safety-First

Safety is the overriding design constraint. The system conforms to ISO 26262 Automotive Safety Integrity Level D (ASIL-D) for the control and safety monitoring subsystems, and ASIL-B for the perception and planning subsystems where diversity and redundancy provide independent safety arguments. Every control command passes through a safety gate that validates command bounds, rate-of-change limits, and consistency with the planned trajectory before the command is forwarded to the actuator interface. A hardware watchdog timer requires a periodic heartbeat from the safety monitor; failure to service the watchdog triggers a hardware-level safe state (brake application, steering lock to center).

#### 1.2.4 Sensor Redundancy

The AVCS employs a multi-sensor suite — cameras, LiDAR, radar, ultrasonic sensors, and GNSS/IMU — with overlapping fields of regard and complementary failure modes. The perception pipeline is designed so that the loss of any single sensor modality degrades system capability gracefully rather than causing a catastrophic failure. For example, the object detection pipeline fuses camera-based detections (excellent classification, poor range accuracy) with LiDAR-based detections (excellent range accuracy, limited classification) and radar-based detections (excellent velocity measurement, poor angular resolution) to produce a unified object list that is more reliable than any single modality alone. Cross-modal consistency checks detect sensor faults: if camera detections suddenly vanish while LiDAR detections persist, the system flags a potential camera fault and adjusts the fusion weights accordingly.

#### 1.2.5 Extensibility

The architecture supports the addition of new sensor types, new algorithms, and new vehicle platforms through well-defined extension points. Sensor drivers implement a common `SensorDriver` interface and are loaded as shared libraries at runtime. Planning algorithms implement a `PlannerBase` abstract class and are selected via configuration. Vehicle dynamics parameters are loaded from a vehicle description file, enabling the same software stack to drive different vehicle platforms. The simulation module provides a hardware-in-the-loop (HIL) and software-in-the-loop (SIL) testing framework that allows new algorithms to be validated against recorded scenarios before deployment on the vehicle.

---

## 2. System Architecture Diagram

### 2.1 Top-Level Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Autonomous Vehicle Control System                    │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │              │  │              │  │              │  │              │   │
│  │  PERCEPTION  │  │ LOCALIZATION │  │   PLANNING   │  │   CONTROL    │   │
│  │              │  │              │  │              │  │              │   │
│  │ • Detection  │  │ • EKF Loc.   │  │ • Path Plan  │  │ • PID        │   │
│  │ • Tracking   │  │ • SLAM       │  │ • Behavior   │  │ • MPC        │   │
│  │ • Fusion     │  │ • HD Map     │  │ • Trajectory │  │ • Stanley    │   │
│  │ • Segmentation│ │ • GPS/IMU    │  │ • Cost Calc  │  │ • Pure Purs. │   │
│  │              │  │              │  │              │  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                  │                  │           │
│         ▼                 ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Real-Time Message Bus (DDS/ROS2)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         ▲                 ▲                  ▲                              │
│         │                 │                  │                              │
│  ┌──────┴───────┐  ┌─────┴────────┐  ┌─────┴──────────┐                   │
│  │              │  │              │  │                │                    │
│  │COMMUNICATION │  │  SIMULATION  │  │ SYSTEM MANAGER │                    │
│  │              │  │              │  │                │                    │
│  │ • DSRC/V2X   │  │ • SIL/HIL    │  │ • Health Mon.  │                    │
│  │ • J2735 Msgs │  │ • Scenario   │  │ • Fault Mgr.   │                    │
│  │ • CAMP       │  │ • Replay     │  │ • Watchdog     │                    │
│  │              │  │              │  │                │                    │
│  └──────────────┘  └──────────────┘  └────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Between Modules

```
┌────────┐    Raw Sensor     ┌────────────┐  Fused Objects  ┌──────────┐
│        │ ──── Data ──────▶ │            │ ──────────────▶ │          │
│ Sensors│                   │ Perception │                  │ Planning │
│        │    Raw Sensor     │            │  Semantic Map   │          │
│        │ ──── Data ──────▶ │            │ ──────────────▶ │          │
└───┬────┘                   └─────┬──────┘                  └────┬─────┘
    │                              │                              │
    │ Raw IMU/GPS                  │ Ego Pose                    │ Trajectory
    ▼                              ▼                              ▼
┌────────┐                   ┌───────────┐                  ┌──────────┐
│        │                   │           │                  │          │
│   GPS  │──────────────────▶│Localizer  │─────────────────▶│ Control  │
│        │                   │           │  Vehicle Pose    │          │
└────────┘                   └───────────┘                  └────┬─────┘
                                                                │
                                                                │ Commands
                                                                ▼
                                                          ┌──────────┐
                                                          │          │
                                                          │Actuators │
                                                          │          │
                                                          └──────────┘

┌──────────────┐  V2X Messages  ┌──────────┐  Traffic Info  ┌──────────┐
│              │ ──────────────▶ │          │ ─────────────▶ │          │
│  V2X/DSRC   │                 │  Comm.   │                │ Planning │
│  Network    │                 │  Module  │                │          │
│              │ ◀────────────── │          │                │          │
└──────────────┘  Broadcast BSM └──────────┘                └──────────┘
```

### 2.3 Sensor-to-Actuator Pipeline

```
  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
  │  Camera │───▶│          │    │          │    │         │    │          │
  │  LiDAR  │───▶│Perception│───▶│Planning  │───▶│Control  │───▶│Actuator  │
  │  Radar  │───▶│Pipeline  │    │Pipeline  │    │Pipeline │    │Interface │
  │  USS    │───▶│          │    │          │    │         │    │          │
  └─────────┘    └──────────┘    └──────────┘    └─────────┘    └──────────┘
       │              │                │               │               │
    <5 ms         <45 ms          <80 ms          <95 ms         <100 ms
  Capture      Detection +      Path + Behav.    MPC Solve      CAN Frame
               Fusion +         + Trajectory     + Safety        Transmit
               Tracking         Optim.           Gate            to ECU
```

---

## 3. Perception Module

### 3.1 Overview

The Perception module is responsible for transforming raw sensor data into a structured, temporally consistent representation of the surrounding environment. Its primary outputs are a fused object list (with position, velocity, acceleration, classification, and tracking IDs), a drivable surface segmentation, a semantic occupancy grid, and traffic signal state information. The module must operate at 20 Hz with an end-to-end latency under 50 ms from sensor timestamp to published object list.

### 3.2 Object Detection Pipeline

The object detection pipeline processes camera and LiDAR data in parallel and fuses the results at the detection level before passing them to the multi-object tracker.

#### 3.2.1 Camera-Based Detection

The camera pipeline ingests synchronized images from six cameras providing a 360-degree surround view. Each image undergoes the following processing stages:

1. **Image Undistortion:** Lens distortion is corrected using pre-calibrated intrinsic parameters (focal length, principal point, distortion coefficients k1–k6). The undistortion map is pre-computed and stored as a GPU texture lookup table to enable real-time remapping at frame rate.

2. **Bird's-Eye View (BEV) Transform:** The undistorted front-view images are warped into a unified bird's-eye view coordinate frame using a homography derived from the camera extrinsics and the assumed ground plane. This BEV representation is critical for downstream fusion with LiDAR data, which is natively in a 3D world coordinate frame. The BEV transform uses a learned perspective transformer (LSS-style) that predicts depth distributions to handle the front-to-top mapping more robustly than a flat-ground homography.

3. **2D Detection:** A YOLOX-based detector processes each camera image and produces 2D bounding boxes with class labels (vehicle, pedestrian, cyclist, traffic cone, barrier, unknown) and confidence scores. The detector is optimized with TensorRT and runs at 30 FPS on the onboard GPU with FP16 inference.

4. **3D Lifting:** Each 2D detection is projected into 3D space using the predicted depth distribution and camera extrinsics, yielding a 3D bounding box proposal in the vehicle coordinate frame. The 3D box dimensions are refined by a small MLP head that predicts (dx, dy, dz, heading) residuals from the ROI-pooled features.

#### 3.2.2 LiDAR-Based Detection

The LiDAR pipeline processes point clouds from one or more spinning or solid-state LiDAR units. The processing stages are:

1. **Ground Removal:** A RANSAC-based plane fitting algorithm segments ground points from the point cloud. The algorithm fits a plane model to a random subset of points over multiple iterations, selecting the plane with the largest inlier count. Points within a distance threshold of the fitted plane are labeled as ground and removed from subsequent processing. This step typically removes 40–60% of the point cloud, significantly reducing the computational load for downstream stages.

2. **Voxelization:** The non-ground points are voxelized into a 3D grid with configurable resolution (default: 0.1 m along X/Y, 0.2 m along Z). Each voxel stores the mean position, intensity, and point count of its constituent points. Voxelization serves two purposes: it regularizes the irregular point cloud into a dense tensor suitable for 3D convolution, and it provides an implicit downsampling that controls computational cost regardless of the raw point density.

3. **3D Object Detection:** A VoxelNet/PointPillars-style network processes the voxelized features and produces 3D bounding box proposals with (x, y, z, l, w, h, heading, class, confidence). The network uses sparse convolution layers followed by a region proposal network (RPN) with anchor boxes sized for vehicle, pedestrian, and cyclist classes. Anchor boxes are oriented at 0° and 90° for vehicles and at 0° only for pedestrians.

4. **Non-Maximum Suppression (NMS):** Rotated NMS is applied per class to suppress overlapping detections. The IoU threshold is 0.1 for pedestrians (dense crowds) and 0.3 for vehicles (sparse). The NMS output constitutes the LiDAR detection list.

### 3.3 Multi-Object Tracking

The multi-object tracker maintains a persistent state for each detected object across frames, producing smooth position and velocity estimates with stable tracking IDs. The tracker operates in two stages: prediction and update.

#### 3.3.1 Prediction Step

Each tracked object is modeled as a Constant Turn Rate and Velocity (CTRV) state vector:

```
x = [px, py, v, theta, omega]
```

where `(px, py)` is the object center position, `v` is the scalar speed, `theta` is the heading angle, and `omega` is the yaw rate. The CTRV model predicts the state forward in time using the nonlinear process model:

```
px'     = px + (v / omega) * [sin(theta + omega*dt) - sin(theta)]
py'     = py + (v / omega) * [cos(theta) - cos(theta + omega*dt)]
v'      = v
theta'  = theta + omega * dt
omega'  = omega
```

When `omega` is near zero (straight-line motion), the model reduces to the constant velocity model to avoid division by zero. The prediction covariance is propagated using the Jacobian of the process model with additive process noise `Q = diag([0.1, 0.1, 0.5, 0.05, 0.01])`.

#### 3.3.2 Data Association

The assignment of new detections to existing tracks is solved using the Hungarian algorithm (Munkres assignment) on a cost matrix. The cost between track `i` and detection `j` is the Mahalanobis distance between the predicted measurement and the actual detection position, computed as:

```
d_ij = sqrt((z_j - h(x_i))^T * S_i^{-1} * (z_j - h(x_i)))
```

where `S_i = H * P_i * H^T + R` is the innovation covariance. Gating is applied before assignment: any track-detection pair with `d_ij > gating_threshold` (default: 9.49, corresponding to the 97.5% chi-squared quantile with 2 DOF) is excluded from the assignment problem. Unassigned detections initiate new tracks; unassigned tracks are coasted (predicted without update) for up to 5 frames before being deleted.

#### 3.3.3 Update Step

Assigned detections update their corresponding tracks using the standard Extended Kalman Filter update equations. The measurement model maps the CTRV state to the measurement space (position only):

```
z = [px, py] + noise
```

The innovation `y = z_meas - h(x_pred)` and Kalman gain `K = P_pred * H^T * S^{-1}` are computed, and the state and covariance are updated:

```
x_post = x_pred + K * y
P_post = (I - K * H) * P_pred
```

### 3.4 Sensor Fusion

The sensor fusion subsystem combines detection and tracking outputs from multiple sensor modalities into a single, unified object list. The AVCS uses a hybrid fusion architecture that combines late fusion (track-level) for the steady-state operation with detection-level fusion for closely spaced objects where track-level association is ambiguous.

#### 3.4.1 Late Fusion Architecture

In the default late fusion mode, each sensor modality maintains its own independent tracker. The fusion node receives track lists from the camera tracker, LiDAR tracker, and radar tracker and performs a second-level track-to-track association using a similar Hungarian algorithm on Mahalanobis distance. Fused tracks combine the position accuracy of LiDAR, the classification accuracy of cameras, and the velocity accuracy of radar through a covariance-weighted state estimate:

```
x_fused = (sum_k P_k^{-1})^{-1} * (sum_k P_k^{-1} * x_k)
P_fused = (sum_k P_k^{-1})^{-1}
```

#### 3.4.2 Extended Kalman Filter Hybrid Fusion

When objects are closely spaced (inter-object distance < 2.0 m) or in high-clutter environments, late fusion may produce incorrect associations. In these situations, the system switches to a hybrid fusion mode where raw detections from all modalities are projected into a common coordinate frame and associated jointly before tracking. This avoids the chicken-and-egg problem of needing tracks to associate detections when the detections themselves are ambiguous.

The hybrid fusion EKF uses a 9-state vector for each fused object:

```
x = [px, py, pz, vx, vy, vz, ax, ay, az]
```

with a constant-acceleration process model. Measurement updates are applied sequentially from each sensor modality in order of increasing measurement noise (radar first, then LiDAR, then camera), which improves numerical conditioning of the covariance update.

### 3.5 LiDAR Processing Pipeline (Detailed)

The complete LiDAR processing pipeline proceeds through the following stages:

```
Raw Point Cloud
      │
      ▼
┌──────────────┐
│  Deskewing   │  Compensate for ego-motion during scan acquisition
└──────┬───────┘
       ▼
┌──────────────┐
│Ground Removal│  RANSAC plane fitting, multi-region for slopes
└──────┬───────┘
       ▼
┌──────────────┐
│ Voxelization │  0.1m × 0.1m × 0.2m grid, mean+max features
└──────┬───────┘
       ▼
┌──────────────┐
│ 3D Detection │  PointPillars + RPN, anchor-based
└──────┬───────┘
       ▼
┌──────────────┐
│ Clustering   │  DBSCAN on non-ground, eps=0.8, min_pts=3
└──────┬───────┘
       ▼
┌──────────────┐
│  Free Space  │  Ray-casting to build occupancy grid
└──────────────┘
```

**Deskewing:** Spinning LiDAR sensors acquire points over a 100 ms scan period during which the ego vehicle moves. Without correction, this motion introduces geometric distortion (especially at high speed). The deskewing step uses the interpolated ego pose at each point's timestamp to transform all points into the coordinate frame at the scan start time.

**Multi-Region Ground Removal:** A single-plane RANSAC model fails on roads with significant grade changes or banked curves. The algorithm divides the point cloud into 4 m × 4 m cells, fits a local plane per cell, and then merges cells whose plane normals are consistent with a global ground surface. This approach handles slopes up to 15° and crown/bank angles up to 10°.

**DBSCAN Clustering:** After ground removal and detection, remaining non-associated points are clustered using DBSCAN with `eps = 0.8 m` and `min_pts = 3`. Clusters that are too small (< 5 points) are discarded as noise; clusters that are too large (> 5000 points) are re-clustered with a smaller eps. This provides a fallback detection mechanism for objects that the learning-based detector misses (e.g., debris, unusual obstacles).

### 3.6 Camera Processing Pipeline (Detailed)

```
Raw Image (6 cameras)
      │
      ▼
┌──────────────┐
│ Undistortion │  Pre-computed GPU LUT, barrel + tangential correction
└──────┬───────┘
       ▼
┌──────────────┐
│ BEV Transform│  Learned perspective (LSS) + flat-ground homography fusion
└──────┬───────┘
       ▼
┌──────────────┐
│ 2D Detection │  YOLOX-X, TensorRT FP16, 30 FPS per camera
└──────┬───────┘
       ▼
┌──────────────┐
│ 3D Lifting   │  Depth distribution prediction + MLP refinement
└──────┬───────┘
       ▼
┌──────────────┐
│ Lane Detect  │  Semantic segmentation (DeepLabV3+) → polynomial fitting
└──────┬───────┘
       ▼
┌──────────────┐
│ Traffic Light│  Color + state classification from front camera ROI
└──────────────┘
```

### 3.7 Thread Model and Latency Requirements

The Perception module employs a multi-threaded pipeline to meet its latency budget:

| Stage | Thread | CPU Affinity | Priority | Budget (ms) | Frequency |
|-------|--------|-------------|----------|-------------|-----------|
| Image Capture | Cam-Cap | Core 0 | RT-90 | 2 | 30 Hz |
| LiDAR Capture | Lidar-Cap | Core 1 | RT-90 | 2 | 20 Hz |
| Camera Detection | Cam-Det | GPU 0 | RT-80 | 15 | 30 Hz |
| LiDAR Detection | Lidar-Det | GPU 0 | RT-80 | 20 | 20 Hz |
| Tracking & Fusion | Track-Fuse | Core 2 | RT-85 | 10 | 20 Hz |
| Segmentation | Seg | GPU 1 | RT-70 | 15 | 10 Hz |

The total worst-case latency from sensor timestamp to published object list is 47 ms, leaving a 3 ms margin against the 50 ms budget. Lock-free single-producer single-consumer (SPSC) rings buffer data between stages, and a time-synchronizer node aligns camera and LiDAR frames to the nearest 10 ms boundary before passing them to the fusion stage.

---

## 4. Localization Module

### 4.1 Overview

The Localization module estimates the ego vehicle's pose (position, orientation, and velocity) in a global coordinate frame with centimeter-level accuracy. It fuses data from GNSS receivers, an inertial measurement unit (IMU), and LiDAR scan matching against a pre-built HD map. The module provides two complementary localization strategies: an EKF localizer for smooth, high-rate output, and a SLAM system for environments where the HD map is unavailable or unreliable.

### 4.2 EKF Localizer

The EKF localizer is the primary localization method for map-covered areas. It maintains a 15-state error-state vector that captures the deviation between the true vehicle state and the nominal (integrated) state:

```
δx = [δpx, δpy, δpz, δvx, δvy, δvz, δφ, δθ, δψ, δbx, δby, δbz, δbgx, δbgy, δbgz]
     ├── position ──┤ ├── velocity ──┤ ├── attitude ─┤ ├── accel bias ─┤ ├── gyro bias ───┤
```

**Prediction Step (IMU-driven):** The IMU provides angular rates and linear accelerations at 200 Hz. The nominal state is propagated forward using the IMU measurements, and the error-state covariance is propagated using the linearized error dynamics. The process model accounts for Earth's rotation rate and the Coriolis effect (significant at highway speeds), though these terms are small enough to be neglected in urban driving scenarios.

**Update Step 1 — GNSS Position:** GPS/GNSS measurements arrive at 10 Hz. The measurement model is a simple position observation `z_gps = [px, py, pz] + noise` with a measurement noise covariance that adapts based on the reported dilution of precision (DOP) and the number of satellites in view. When the DOP exceeds 5.0 or the satellite count drops below 6, the GPS update is suspended and the system relies on dead-reckoning alone.

**Update Step 2 — LiDAR Scan Matching:** At 10 Hz, the LiDAR point cloud is matched against the HD map using Normal Distributions Transform (NDT) scan matching. The NDT aligns the current scan to the map by maximizing the likelihood of the scan points under the map's Gaussian mixture representation. The NDT provides a 6-DOF pose correction `(δpx, δpy, δpz, δroll, δpitch, δyaw)` that is used as a measurement update in the EKF. The NDT convergence is monitored: if the transformation change between iterations exceeds a threshold or the fitness score is above 2.0, the scan match is rejected as unreliable.

**Update Step 3 — LiDAR Altitude:** In addition to the 6-DOF NDT output, a separate altitude measurement is derived from the ground plane estimation in the LiDAR scan. This provides a direct observation of the Z coordinate that is more accurate than the NDT vertical estimate (which suffers from the sparse vertical resolution of spinning LiDAR).

### 4.3 SLAM System

When the vehicle operates in areas without HD map coverage, or when the map is detected as outdated (e.g., due to construction), the SLAM system provides self-contained localization and mapping.

#### 4.3.1 ICP Scan Matching

The SLAM front-end registers consecutive LiDAR scans using Iterative Closest Point (ICP) with a point-to-plane metric. The point-to-plane ICP minimizes the distance from each source point to the tangent plane of the closest target point, which converges faster and is more accurate than point-to-point ICP for structured environments. The algorithm operates on downsampled voxel grids (0.2 m resolution) for computational efficiency. Convergence is declared when the change in transformation falls below 1e-6 or after 30 iterations.

#### 4.3.2 Keyframe Management

Not every scan is stored as a keyframe. A new keyframe is inserted when the estimated motion since the last keyframe exceeds any of the following thresholds:

- Translation > 1.0 m
- Rotation > 5.0°
- Time elapsed > 2.0 s

Keyframes store the downsampled point cloud, the estimated pose, and the IMU preintegration between the previous and current keyframe. Old keyframes are pruned when they fall outside a sliding window of 50 m from the current position, but keyframes near loop closure candidates are retained.

#### 4.3.3 Loop Closure

Loop closure detection identifies previously visited locations by matching the current scan's feature descriptor against a database of keyframe descriptors. The feature descriptor is a ScanContext — a 2D polar-coordinate representation of the point cloud that is rotation-invariant and efficient to compare. When a match is found with a distance below a threshold, a loop closure constraint is added to the pose graph.

#### 4.3.4 Pose Graph Optimization

The pose graph is optimized using iSAM2 (incremental Smoothing and Mapping using the Bayes tree), which provides efficient incremental updates as new keyframes and loop closures are added. The graph nodes represent keyframe poses, and the edges represent relative pose constraints from ICP scan matching and IMU preintegration. Loop closure edges have a higher information matrix scaling factor to ensure they are satisfied tightly. The optimization runs in a background thread and updates the keyframe poses asynchronously; the front-end always uses the latest available optimized poses for its initial guess.

### 4.4 GPS Processing

Raw GPS measurements from the GNSS receiver are reported in the LLA (Latitude, Longitude, Altitude) coordinate frame. Before they can be fused with the EKF, they must be converted to a local Cartesian frame.

#### 4.4.1 LLA to ENU Conversion

The conversion from LLA to the local East-North-Up (ENU) frame proceeds in two steps:

1. **LLA to ECEF:** The geodetic coordinates are converted to Earth-Centered Earth-Fixed Cartesian coordinates using the WGS-84 ellipsoid parameters:

```python
# WGS-84 constants
a = 6378137.0          # semi-major axis (m)
f = 1 / 298.257223563  # flattening
e2 = 2*f - f*f         # first eccentricity squared

N = a / sqrt(1 - e2 * sin(lat)**2)
x_ecef = (N + alt) * cos(lat) * cos(lon)
y_ecef = (N + alt) * cos(lat) * sin(lon)
z_ecef = (N * (1 - e2) + alt) * sin(lat)
```

2. **ECEF to ENU:** The ECEF coordinates are rotated into the local ENU frame centered at a reference point `(lat0, lon0, alt0)` chosen at system startup:

```python
dx = x_ecef - x_ref
dy = y_ecef - y_ref
dz = z_ecef - z_ref

east  = -sin(lon0)*dx + cos(lon0)*dy
north = -sin(lat0)*cos(lon0)*dx - sin(lat0)*sin(lon0)*dy + cos(lat0)*dz
up    =  cos(lat0)*cos(lon0)*dx + cos(lat0)*sin(lon0)*dy + sin(lat0)*dz
```

### 4.5 IMU Processing

The IMU provides angular velocity and linear acceleration measurements at 200 Hz. Before these measurements are used in the EKF prediction, they must be corrected for bias and gravity.

#### 4.5.1 Bias Estimation

Accelerometer and gyroscope biases are estimated as part of the 15-state EKF vector (states 9–14). During initialization, the vehicle is stationary for 5 seconds, and the average accelerometer and gyroscope readings are used to initialize the bias states. The gravity vector is computed from the mean accelerometer reading during this static period. During operation, the bias states are continuously updated by the EKF; the gyroscope bias typically converges within 30 seconds, while the accelerometer bias converges more slowly (several minutes) due to its weaker observability.

#### 4.5.2 Gravity Compensation

The raw accelerometer measurement includes the reaction to gravity. The gravity-compensated acceleration is computed as:

```
a_compensated = R_body_to_world * (a_raw - a_bias) - g_vector
```

where `R_body_to_world` is the rotation matrix from the body frame to the world frame (derived from the current attitude estimate), and `g_vector = [0, 0, -9.80665]` is the gravity vector in the ENU world frame. Incorrect gravity compensation is the dominant source of drift in dead-reckoning mode; even a 0.1° attitude error produces a 1.7 cm/s² acceleration error, which accumulates to 1 m of position error in 11 seconds.

### 4.6 HD Map Integration

The HD map provides a prior representation of the road environment including lane geometry, road boundaries, traffic sign positions, and semantic attributes. The map is stored in the Lanelet2 format and loaded at startup. The localizer queries the map for the region within 100 m of the estimated position and extracts:

- **Lane centerlines:** Polynomial curves (cubic splines) used for lateral error computation.
- **Road boundaries:** Used for drivable area validation in the planning module.
- **Traffic sign positions:** Used to initialize the traffic sign detection ROI in the perception module.
- **Elevation profile:** Used to constrain the altitude estimate and improve vertical accuracy.

The map is versioned, and the localizer periodically checks for map updates via the communication module. If a map update is available, it is downloaded and applied at the next system restart to avoid runtime consistency issues.

---

## 5. Planning Module

### 5.1 Overview

The Planning module is responsible for determining the vehicle's future path and behavior. It operates in three hierarchical layers: path planning (global route), behavior planning (tactical decisions), and trajectory optimization (smooth, executable trajectory). The planning module must produce a new trajectory at 10 Hz with a computation budget of 100 ms per cycle.

### 5.2 Path Planning

#### 5.2.1 A* Grid-Based Planner

For global route planning, the system uses an A* search algorithm on a discretized road network graph derived from the HD map. Each node in the graph represents a lanelet (a segment of a lane), and edges represent lanelet connectivity (successor, predecessor, left neighbor, right neighbor). The cost function for each edge is:

```
g(edge) = w_dist * length + w_time * (length / speed_limit) + w_lane * lane_change_cost + w_turn * turn_penalty
```

where `w_dist`, `w_time`, `w_lane`, and `w_turn` are tunable weights. The lane change cost penalizes routes that require many lane changes, and the turn penalty discourages routes with frequent turns at intersections. The heuristic function `h(node)` is the Euclidean distance to the goal divided by the maximum speed limit, ensuring admissibility and thus optimality of the A* solution.

The A* planner runs once when a new destination is set and whenever the vehicle deviates significantly from the planned route (> 10 m lateral offset or > 50 m longitudinal offset). Re-planning is throttled to at most once every 5 seconds to avoid oscillation.

#### 5.2.2 Lattice-Based Planner

For local path planning within the current maneuver, a lattice-based planner generates a set of candidate paths by connecting the current state to a set of terminal states via parametric curves. The lattice is defined in the Frenet frame (longitudinal s, lateral d) along the reference path. Terminal states are sampled at a fixed lookahead distance (typically 30–80 m) with lateral offsets from -3.5 m to +3.5 m in 0.5 m increments and heading offsets from -15° to +15° in 5° increments.

Each candidate path is generated by fitting a quartic polynomial in the longitudinal direction and a quintic polynomial in the lateral direction, ensuring continuity of position, velocity, and acceleration at the boundary. The quintic polynomial for the lateral direction:

```
d(s) = a0 + a1*s + a2*s^2 + a3*s^3 + a4*s^4 + a5*s^5
```

is solved from the boundary conditions: initial (d, d', d'') and terminal (d, d', d'').

### 5.3 Behavior Planning

The behavior planner manages high-level tactical decisions using a Finite State Machine (FSM). The FSM has five states, each with defined entry conditions, exit conditions, and associated actions:

```
                    ┌──────────────┐
          ┌────────│  LANE_KEEPING │◀───────┐
          │        └──────┬───────┘        │
          │               │                │
          │  lane_change  │  obstacle      │  clear
          │  _requested   │  ahead         │
          │               ▼                │
          │        ┌──────────────┐        │
          │        │ LANE_CHANGE  │────────┘
          │        └──────┬───────┘
          │               │
          │   approach    │   emergency
          │  intersection │
          │               ▼
          │        ┌──────────────┐     ┌───────────────┐
          │        │ INTERSECTION │     │ EMERGENCY_STOP│◀── any state
          │        └──────┬───────┘     └───────────────┘
          │               │                    │
          │    enter       │                    │ resolved
          │  parking zone  │                    ▼
          │               ▼              ┌──────────────┐
          │        ┌──────────────┐      │ LANE_KEEPING │
          └───────▶│   PARKING    │─────▶│  (recovery)  │
                   └──────────────┘      └──────────────┘
```

**LANE_KEEPING:** The default state. The vehicle follows the current lane centerline at the desired speed. Transitions to LANE_CHANGE when a slower vehicle ahead cannot be passed within the current lane and an adjacent lane is available. Transitions to INTERSECTION when approaching an intersection within 50 m. Transitions to EMERGENCY_STOP when a critical collision risk is detected (TTC < 1.5 s).

**LANE_CHANGE:** The vehicle executes a lane change maneuver. A target lane is selected based on a cost evaluation (lane clearance, speed advantage, route alignment). The lane change trajectory is generated with a minimum duration of 3.0 seconds and a maximum lateral acceleration of 1.5 m/s². The FSM returns to LANE_KEEPING when the lateral displacement exceeds 80% of the lane width and the lateral velocity drops below 0.2 m/s.

**INTERSECTION:** The vehicle approaches and navigates an intersection. This state handles traffic light compliance, right-of-way rules, and unprotected turns. A gap acceptance model evaluates whether sufficient gaps exist in cross traffic before the vehicle proceeds. The minimum acceptable gap is 5.0 seconds for left turns and 3.0 seconds for right turns at 30 km/h.

**EMERGENCY_STOP:** The vehicle applies maximum deceleration (up to -8.0 m/s²) to avoid a collision. This state is entered from any other state when the time-to-collision drops below 1.5 seconds or when a critical system fault is detected. The vehicle stops in the current lane unless a shoulder is detected within 1.0 m, in which case it steers toward the shoulder while decelerating.

**PARKING:** The vehicle executes a parking maneuver (parallel, perpendicular, or angled). A hybrid A* planner searches for a collision-free path in the tight space around the parking spot. The parking trajectory is executed at a maximum speed of 5 km/h with frequent re-planning (2 Hz) to account for small positional errors.

### 5.4 Trajectory Optimization

The trajectory optimizer refines the candidate paths from the lattice planner into smooth, dynamically feasible trajectories using a Quadratic Programming (QP) formulation in the Frenet frame.

#### 5.4.1 Frenet-Frame QP Formulation

The optimization is decoupled into longitudinal and lateral problems:

**Longitudinal QP:** Optimizes the speed profile along the path. The decision variables are the longitudinal jerk values at discretized time steps. The objective minimizes a weighted sum of jerk (comfort), speed deviation from the desired speed (efficiency), and acceleration (energy). Constraints enforce speed limits, acceleration limits (-6.0 to +3.0 m/s²), jerk limits (-4.0 to +4.0 m/s³), and safe following distance (1.5 s time gap).

**Lateral QP:** Optimizes the lateral offset from the reference path. The decision variables are the lateral jerk values at discretized arc-length steps. The objective minimizes lateral offset (lane centering), lateral acceleration (comfort), and lateral jerk (smoothness). Constraints enforce lane boundaries, lateral acceleration limits (±2.0 m/s²), and collision avoidance (minimum 0.3 m lateral clearance from static obstacles, 0.5 m from dynamic obstacles).

The QP problems are solved using the OSQP solver, which converges in under 5 ms for typical problem sizes (50–100 decision variables). If the QP is infeasible (e.g., no collision-free trajectory exists within the constraints), the system falls back to an emergency trajectory that applies increasing deceleration.

### 5.5 Cost Calculation Framework

Every candidate trajectory is evaluated against a comprehensive cost function that balances safety, comfort, efficiency, and compliance:

```
J_total = w_safety * J_safety + w_comfort * J_comfort + w_efficiency * J_efficiency + w_rule * J_rule

J_safety    = Σ min_distance_violation² + collision_probability * 1e6
J_comfort   = Σ (jerk_lat² + jerk_lon²) * dt + Σ (accel_lat² + accel_lon²) * dt
J_efficiency = Σ (v_desired - v_actual)² * dt + arrival_time_deviation²
J_rule      = speed_limit_violation² + lane_boundary_violation² + red_light_violation * 1e8
```

The safety cost includes a hard penalty for any trajectory that comes within the minimum safe distance of another object, and a scaled penalty for collision probability (computed from the object's position and velocity uncertainty). The rule cost includes extremely large penalties for red light violations and wrong-way driving, ensuring these are never selected regardless of other costs.

---

## 6. Control Module

### 6.1 Overview

The Control module translates the planned trajectory into steering, throttle, and brake commands that are sent to the vehicle actuators via the CAN bus. The module runs at 100 Hz and must produce a new command within 8 ms of each control tick. It supports multiple control algorithms that can be selected and tuned per vehicle platform and operating mode.

### 6.2 PID Controller

The PID controller provides a simple, well-understood baseline for both longitudinal and lateral control. It is used as the default controller during low-speed maneuvers (< 20 km/h) and as a fallback when more sophisticated controllers fail to produce valid output.

#### 6.2.1 Longitudinal PID

The longitudinal PID controls the vehicle's speed by adjusting the throttle and brake commands:

```
error_v = v_target - v_current
throttle_cmd = Kp_v * error_v + Ki_v * ∫error_v dt + Kd_v * d(error_v)/dt
```

Anti-windup is implemented by clamping the integral term to ±10% throttle equivalent and by resetting the integrator when the error sign changes. The throttle and brake outputs are mutually exclusive: if `throttle_cmd > 0`, the throttle is applied and the brake is released; if `throttle_cmd < 0`, the brake is applied proportionally and the throttle is zeroed. The gains are scheduled by speed:

| Speed Range (km/h) | Kp | Ki | Kd |
|---------------------|-----|----|----|
| 0–20 | 0.8 | 0.05 | 0.02 |
| 20–60 | 0.5 | 0.03 | 0.015 |
| 60–120 | 0.3 | 0.02 | 0.01 |

#### 6.2.2 Lateral PID

The lateral PID controls the vehicle's heading by adjusting the steering angle:

```
error_yaw = yaw_target - yaw_current  # normalized to [-π, π]
steer_cmd = Kp_y * error_yaw + Kd_y * d(error_yaw)/dt
```

The lateral PID does not include an integral term because steady-state lateral offset is handled by adding a cross-track error (CTE) feedforward term:

```
steer_cmd += Kp_cte * cross_track_error
```

### 6.3 MPC Controller

The Model Predictive Controller is the primary control algorithm for highway and urban driving. It solves a constrained optimization problem at each control step, predicting the vehicle's future states over a receding horizon and selecting the control sequence that minimizes a cost function while satisfying dynamic and safety constraints.

#### 6.3.1 Bicycle Model

The MPC uses a dynamic bicycle model with the following state and control vectors:

```
State:  x = [X, Y, ψ, v, β]     (position X/Y, heading, speed, sideslip)
Control: u = [δ, a]               (steering angle, longitudinal acceleration)
```

The dynamic bicycle model equations:

```
Ẋ = v * cos(ψ + β)
Ẏ = v * sin(ψ + β)
ψ̇ = v * cos(β) * tan(δ) / L
v̇ = a
β = arctan(lr / (lf + lr) * tan(δ))
```

where `lf` and `lr` are the distances from the center of gravity to the front and rear axles, and `L = lf + lr` is the wheelbase. The tire forces are modeled using a linear tire model: `Fyf = -Cf * αf` and `Fyr = -Cr * αr`, where `Cf` and `Cr` are the front and rear cornering stiffnesses and `αf`, `αr` are the tire slip angles.

#### 6.3.2 QP Formulation

The MPC problem is transcribed into a QP by linearizing the bicycle model around the current state and discretizing using the Euler method with time step `dt = 0.01 s`. The prediction horizon is `N = 30` steps (0.3 s lookahead at 100 Hz). The QP decision variables are the control increments `Δu_k = u_k - u_{k-1}`.

```
minimize  Σ_{k=0}^{N-1} (x_k - x_ref_k)^T Q (x_k - x_ref_k) + Δu_k^T R Δu_k
subject to  x_{k+1} = A_k x_k + B_k u_k + d_k    (linearized dynamics)
            u_min ≤ u_k ≤ u_max                     (control bounds)
            Δu_min ≤ Δu_k ≤ Δu_max                  (rate limits)
            y_min ≤ C x_k ≤ y_max                   (output constraints)
```

The weight matrices are:

```
Q = diag([1.0, 1.0, 10.0, 0.5, 0.1])    # heading error weighted most
R = diag([50.0, 10.0])                    # steering effort penalized heavily
```

The QP is solved using the qpOASES active-set solver, which warm-starts from the previous solution and typically converges in 1–3 iterations (under 2 ms).

### 6.4 Stanley Controller

The Stanley controller is a geometric lateral controller developed by Stanford's DARPA Grand Challenge team. It computes the steering angle as a function of the heading error and the cross-track error:

```
δ = ψ_e + arctan(k * e_ct / v_x)
```

where `ψ_e` is the heading error (difference between the vehicle heading and the path heading at the nearest point), `e_ct` is the cross-track error (signed lateral distance from the vehicle to the nearest path point), `k` is a gain parameter (default: 2.5), and `v_x` is the longitudinal velocity. The arctangent function naturally reduces the CTE correction at high speeds, preventing oscillatory behavior.

**Adaptive gain scheduling:** The gain `k` is adjusted based on the current speed and the magnitude of the cross-track error:

```python
k_adaptive = k_base * (1.0 + 0.5 * abs(e_ct))  # increase gain for large errors
k_adaptive = max(k_min, min(k_adaptive, k_max))  # clamp to [0.5, 5.0]
```

At very low speeds (< 2 m/s), the Stanley controller becomes numerically unstable due to the division by `v_x`. In this regime, the system switches to the Pure Pursuit controller.

### 6.5 Pure Pursuit Controller

The Pure Pursuit controller computes the steering angle by targeting a lookahead point on the planned path at a distance `L_d` ahead of the vehicle:

```
δ = arctan(2 * L * sin(α) / L_d)
```

where `α` is the angle between the vehicle heading and the line from the rear axle to the lookahead point, `L` is the wheelbase, and `L_d` is the lookahead distance.

**Adaptive Lookahead:** The lookahead distance adapts to the vehicle's speed to balance responsiveness (short lookahead) and stability (long lookahead):

```
L_d = k_dd * v_x + L_d_min
```

where `k_dd = 0.8 s` (a time-based scaling) and `L_d_min = 3.0 m`. At 10 m/s (36 km/h), `L_d = 11.0 m`; at 30 m/s (108 km/h), `L_d = 27.0 m`. This speed-adaptive lookahead ensures smooth, stable tracking at all speeds.

### 6.6 Vehicle Dynamics Model

The control module maintains two vehicle models for different purposes:

#### 6.6.1 Kinematic Bicycle Model

Used for low-speed maneuvers (v < 5 m/s) and for the MPC prediction model when tire forces are small:

```
Ẋ = v * cos(ψ)
Ẏ = v * sin(ψ)
ψ̇ = v * tan(δ) / L
v̇ = a
```

This model assumes no tire slip and is accurate to within 5% for lateral accelerations below 0.3g.

#### 6.6.2 Dynamic Bicycle Model

Used for high-speed driving and for the MPC prediction model at higher accelerations:

```
m * v̇ = Fx_f * cos(δ) - Fy_f * sin(δ) + Fx_r
m * v * β̇ = Fx_f * sin(δ) + Fy_f * cos(δ) + Fy_r - m * v * ψ̇
Iz * ψ̈ = lf * (Fx_f * sin(δ) + Fy_f * cos(δ)) - lr * Fy_r
```

Tire forces use a Pacejka magic formula for accurate force computation near the friction limit:

```
Fy = D * sin(C * arctan(B * α - E * (B * α - arctan(B * α))))
```

where `B`, `C`, `D`, `E` are tire-specific parameters identified from vehicle testing. The model parameters are loaded from a vehicle description YAML file that includes mass, inertia, wheelbase, tire coefficients, and actuator limits.

---

## 7. Communication Module

### 7.1 Overview

The Communication module handles all vehicle-to-everything (V2X) interactions, enabling the autonomous vehicle to receive and broadcast safety-critical information. The module processes DSRC (Dedicated Short-Range Communications) messages, parses SAE J2735 standard messages, and integrates CAMP (Crash Avoidance Metrics Partnership) safety applications. It operates as a bridge between the external V2X radio hardware and the internal AVCS message bus.

### 7.2 DSRC Interface

The DSRC interface communicates with the onboard V2X radio unit over Ethernet (IEEE 802.3) using the WAVE (Wireless Access in Vehicular Environments) protocol stack. The radio operates in the 5.850–5.925 GHz band (5.9 GHz DSRC band) with 10 MHz channels, supporting both safety channel 172 (for BSM) and service channels (for MAP/SPAT).

**Physical Layer:** The radio supports a data rate of 3–27 Mbps using OFDM with 52 subcarriers. The typical communication range is 300–1000 m depending on line-of-sight conditions, transmit power (up to 20 dBm), and channel load.

**Link Layer:** The WAVE Short Message Protocol (WSMP) provides low-latency, connectionless message delivery. WSMP messages bypass the IP stack and are delivered directly to the application layer, achieving end-to-end latencies under 10 ms. The Communication module listens on WSMP port numbers assigned to each J2735 message type.

**Security:** All outgoing messages are signed with the vehicle's digital certificate (SCMS — Security Credential Management System), and all incoming messages are verified against the sender's certificate. Certificate revocation lists are updated daily via the backend connectivity channel. Messages with invalid signatures are silently dropped.

### 7.3 Message Parsing (J2735)

The module parses SAE J2735 messages, which define the standard data frames for V2X communication. The three primary message types are:

#### 7.3.1 BSM (Basic Safety Message) — J2735 Part II

BSMs are broadcast by all DSRC-equipped vehicles at 10 Hz. Each BSM contains:

```
BSM Core Data:
  - msgCount:       Message sequence number (0–127)
  - id:             Temporary vehicle ID (4 bytes, rotated every 5 min)
  - secMark:        Minute-of-year timestamp (0–60999, ms resolution)
  - lat/lon/ele:    Position (1/10 microdegree, 0.1 m elevation)
  - elevation:      Elevation above sea level
  - heading:        Heading (0–65535, 0.0055° resolution)
  - speed:          Speed (0–16383, 0.02 m/s resolution)
  - accelSet:       4-axis acceleration (longitudinal, lateral, vertical, yaw rate)
  - brakeSystem:    Brake status (applied, abs, traction, stability, etc.)
  - vehicleSize:    Length and width (0.1 m resolution)

BSM Part II (optional):
  - VehicleSafetyExtensions: Event flags (hard braking, hazard lights, etc.)
  - SpecialVehicleExtensions: Emergency vehicle type, lights, siren
  - SupplementalVehicleExtensions: Weather data, road friction estimate
```

The parser validates each field against the J2735 range constraints and rejects malformed messages. Valid BSMs are published to the internal message bus as `V2X_BSM` messages, which the Planning module uses for cooperative maneuver planning and the Perception module uses as an additional object detection source.

#### 7.3.2 MAP (Map Data) — J2735 Part II

MAP messages describe the geometric and logical layout of intersections and road segments. They are broadcast by roadside units (RSUs) at 1 Hz. The parser extracts:

- **Intersection geometry:** Node positions, lane connectivity, and lane attributes (type, direction, restrictions).
- **Approach lanes:** Lane widths, speed limits, and allowed maneuvers (straight, left, right, U-turn).
- **Connects-to list:** Maps each ingress lane to its egress lanes and allowed maneuvers.

The MAP data is fused with the onboard HD map to update signalized intersection geometry in real time, enabling the Planning module to handle temporary lane closures or construction zones reported by the RSU.

#### 7.3.3 SPAT (Signal Phase and Timing) — J2735 Part II

SPAT messages provide real-time traffic signal status, including current phase, remaining time, and predicted phase changes. They are broadcast at 2 Hz. The parser extracts:

```
SPAT Data:
  - intersectionId:   Map reference for intersection
  - phase:            Current phase (red, yellow, green, flashing)
  - remainingTime:    Time until next phase change (0–3600 s)
  - predictedPhase:   Next phase
  - confidence:       Confidence in phase prediction (0–100%)
```

The SPAT data enables the Planning module to perform Green Wave Speed Advisory (GWSA) — adjusting the vehicle's speed to arrive at the intersection during the green phase — and to make informed decisions about whether to proceed through a yellow phase or initiate a stop.

### 7.4 CAMP Processing

The Crash Avoidance Metrics Partnership (CAMP) safety applications process V2X data to generate driver/vehicle alerts. The AVCS implements the following CAMP applications:

1. **Forward Collision Warning (FCW):** Monitors BSM data from vehicles ahead. Issues a warning when the time-to-collision (TTC) drops below 2.5 s. The TTC is computed using the relative range and range-rate derived from BSM position and speed data.

2. **Emergency Electronic Brake Light (EEBL):** Detects hard braking events (deceleration > -6.0 m/s²) in BSM messages from vehicles ahead, even if they are occluded by other vehicles. This provides a "see-through" capability for chain-reaction braking scenarios.

3. **Intersection Movement Assist (IMA):** Uses MAP and SPAT data to detect potential cross-path conflicts at signalized intersections. Warns if the vehicle's planned path intersects with another vehicle's predicted path within the next 5 seconds and the other vehicle is not expected to stop (e.g., running a red light).

4. **Blind Spot Warning (BSW) / Lane Change Warning (LCW):** Monitors BSM data from vehicles in adjacent lanes. Issues a warning when a vehicle is detected in the blind spot zone (0.5–3.0 m lateral, ±5.0 m longitudinal from the ego vehicle's rear bumper) and the ego vehicle signals a lane change.

The CAMP applications run in a dedicated thread at 20 Hz and publish alerts to the Planning module, which incorporates them into its behavior planning decisions. Critical alerts (FCW with TTC < 1.5 s, EEBL) also trigger an immediate message to the Control module's safety gate.

---

## 8. Real-Time Engine

### 8.1 Overview

The Real-Time Engine is the infrastructure layer that ensures all AVCS modules meet their timing constraints. It provides priority-based scheduling, deadline monitoring, watchdog management, and inter-module communication with bounded latency. The engine runs on a Linux kernel with the PREEMPT_RT patch to provide deterministic scheduling with worst-case latency under 50 μs for the highest-priority threads.

### 8.2 Control Loop Frequencies

Each module operates at a frequency determined by its timing requirements and the dynamics of the physical process it controls:

| Module | Frequency | Period (ms) | Deadline (ms) | Priority |
|--------|-----------|-------------|---------------|----------|
| Control | 100 Hz | 10 | 8 | RT-99 |
| Localization (IMU) | 200 Hz | 5 | 4 | RT-98 |
| Perception | 20 Hz | 50 | 50 | RT-85 |
| Planning | 10 Hz | 100 | 100 | RT-80 |
| Communication (V2X) | 10 Hz | 100 | 80 | RT-75 |
| Safety Monitor | 100 Hz | 10 | 8 | RT-99 |
| Health Monitor | 1 Hz | 1000 | 500 | RT-50 |

The control and safety monitor threads share the highest priority (RT-99) and are pinned to dedicated CPU cores (Core 3 and Core 4, respectively) to prevent preemption by any other thread. The localization IMU thread runs at RT-98 on Core 2. Perception and planning share the GPU and remaining CPU cores.

### 8.3 Priority Scheduling

The scheduling policy is SCHED_FIFO (POSIX real-time first-in-first-out) for all time-critical threads and SCHED_OTHER for non-critical background tasks (logging, map updates, diagnostics). The priority assignment follows a rate-monotonic principle: higher-frequency tasks receive higher priorities, with the exception that safety-critical tasks (safety monitor) are always assigned the highest priority regardless of frequency.

CPU affinity is enforced using `pthread_setaffinity_np` to bind each thread to a specific core, preventing cache pollution and scheduling jitter:

```
Core 0: Camera capture, Image processing
Core 1: LiDAR capture, Point cloud processing
Core 2: Localization (IMU, GPS, EKF)
Core 3: Control loop, Safety gate
Core 4: Safety monitor, Watchdog
Core 5: Planning, Communication
Core 6: Tracking, Fusion
Core 7: System manager, Logging, Diagnostics
```

### 8.4 Watchdog Timer

A hardware watchdog timer (watchdog device `/dev/watchdog`) provides a last-resort safety mechanism. The safety monitor thread services the watchdog every 10 ms by writing to the device. If the watchdog is not serviced within 100 ms (the hardware timeout), the watchdog triggers a hardware reset that:

1. Applies maximum braking force via a dedicated brake actuator circuit (independent of the main compute).
2. Disables the throttle and shifts the transmission to neutral.
3. Activates the hazard lights.
4. Sounds an audible alarm inside the vehicle.

The watchdog servicing code is the simplest possible — a single `write()` system call — to minimize the risk of a software bug preventing the watchdog from being serviced.

### 8.5 Deadline Monitoring

Each real-time thread instruments its execution with timing markers at the start and end of each iteration. The deadline monitor thread (running at 1 Hz on Core 7) collects these markers and computes:

- **Average execution time:** Mean time per iteration over the last 1000 cycles.
- **Worst-case execution time (WCET):** Maximum time per iteration over the last 60 seconds.
- **Deadline miss count:** Number of iterations that exceeded their deadline in the last 60 seconds.
- **Deadline miss rate:** Miss count / total iterations.

The deadline monitor publishes a `TimingReport` message at 1 Hz. If the deadline miss rate for any RT-99 or RT-98 thread exceeds 0.1%, an alarm is raised. If it exceeds 1.0%, the system transitions to a degraded mode where non-essential processing (simulation, logging) is suspended. If it exceeds 5.0%, an emergency stop is triggered.

```
Deadline Miss Rate Escalation:
  0.1%  →  ALARM (log + dashboard warning)
  1.0%  →  DEGRADED (suspend non-essential threads)
  5.0%  →  EMERGENCY_STOP (hardware watchdog-assisted safe state)
```

---

## 9. Data Flow

### 9.1 End-to-End Data Pipeline

The following diagram traces a single data item (a LiDAR detection of a vehicle ahead) from sensor acquisition to actuator command:

```
Time    Stage                              Data                           Latency
─────────────────────────────────────────────────────────────────────────────────
 0 ms   LiDAR Sensor Acquisition          Raw point cloud (100k pts)     —
 1 ms   Point Cloud Deskewing             Corrected point cloud          1 ms
 3 ms   Ground Removal                    Non-ground points (40k pts)    2 ms
 5 ms   Voxelization                      Voxel grid (5k voxels)         2 ms
12 ms   3D Detection (GPU)                3D boxes (15 objects)          7 ms
15 ms   Camera Detection (GPU)            2D+3D boxes (20 objects)       15 ms*
17 ms   Multi-Object Tracking             Tracked objects (25 tracks)    2 ms
20 ms   Sensor Fusion                     Fused object list (22 objects) 3 ms
25 ms   Object List Published             —                              —
─────────────────────────────────────────────────────────────────────────────────
30 ms   EKF Localization Update           Ego pose (6-DOF + cov)         5 ms†
35 ms   Pose Published                    —                              —
─────────────────────────────────────────────────────────────────────────────────
40 ms   Behavior Planning (FSM)           Maneuver decision              5 ms
55 ms   Lattice Path Generation           30 candidate paths             15 ms
65 ms   Trajectory Optimization (QP)      Optimal trajectory             10 ms
70 ms   Trajectory Published              —                              —
─────────────────────────────────────────────────────────────────────────────────
72 ms   MPC Solve                         Steering + accel commands      2 ms
74 ms   Safety Gate Validation            Validated commands             2 ms
76 ms   CAN Frame Transmission            Actuator commands              2 ms
─────────────────────────────────────────────────────────────────────────────────
                                            TOTAL LATENCY:             76 ms
```

*Camera detection runs in parallel with LiDAR detection; its latency does not add to the critical path.  
†Localization updates overlap with perception processing.

### 9.2 Sensor Data Acquisition

Each sensor driver runs in its own thread and manages the hardware interface:

| Sensor | Interface | Frequency | Data Size | Thread |
|--------|-----------|-----------|-----------|--------|
| LiDAR (64-ch) | Ethernet (UDP) | 20 Hz | ~2 MB/frame | Lidar-Cap |
| Camera (6×) | GMSL2 (MIPI CSI-2) | 30 Hz | ~18 MB/frame | Cam-Cap |
| Radar (4×) | CAN-FD | 20 Hz | ~2 KB/frame | Radar-Cap |
| GNSS/IMU | UART (115200 baud) | 10 Hz / 200 Hz | ~100 B / ~50 B | Gps-Cap |
| Ultrasonic (12×) | CAN | 20 Hz | ~200 B/frame | Uss-Cap |

All sensor data is timestamped with a common clock (PTP-synchronized to GPS time with <1 ms offset) at the point of physical measurement, not at the point of software reception. This ensures that downstream processing can accurately model the temporal relationship between sensor measurements.

### 9.3 Processing Pipeline

The processing pipeline is organized as a directed acyclic graph (DAG) of processing nodes, each of which subscribes to input topics and publishes to output topics. The message bus (DDS/ROS2) handles message serialization, transport, and delivery with configurable Quality of Service (QoS) policies:

- **Reliability:** RELIABLE for control and safety messages, BEST_EFFORT for sensor data.
- **History:** KEEP_LAST(1) with a depth of 1 (always the latest data), KEEP_ALL for recording.
- **Durability:** VOLATILE (no persistent storage) for real-time data, TRANSIENT_LOCAL for configuration.

### 9.4 Actuator Command Generation

The final stage of the data flow pipeline is the generation and transmission of actuator commands:

1. **Control Command Generation:** The selected control algorithm (MPC, PID, Stanley, or Pure Pursuit) computes a raw steering angle and longitudinal acceleration command.

2. **Coordinate Transform:** The raw commands are transformed from the control frame (center of rear axle) to the actuator frame (steering column, throttle body, brake caliper) using the vehicle kinematic model.

3. **Safety Gate:** Every command passes through the safety gate, which enforces:
   - **Absolute bounds:** Steering ±540°, throttle 0–100%, brake 0–100%.
   - **Rate limits:** Steering rate ≤ 500°/s, throttle rate ≤ 50%/s, brake rate ≤ 80%/s.
   - **Consistency checks:** Throttle and brake not simultaneously active; steering angle consistent with lateral acceleration; speed within ±10% of the planned speed.
   - **Heartbeat check:** If no new trajectory has been received from Planning within 200 ms, the safety gate commands a gradual deceleration to stop.

4. **CAN Frame Construction:** Validated commands are packed into CAN 2.0B frames according to the vehicle's DBC (Database Container) file. Each frame includes a rolling counter and checksum for integrity verification.

5. **CAN Transmission:** Frames are transmitted over the vehicle's high-speed CAN bus (500 kbps) via the SocketCAN interface. Transmission latency is under 1 ms.

### 9.5 Latency Budget Breakdown

The total latency budget for the perception-to-actuator pipeline is 100 ms, allocated as follows:

| Stage | Budget (ms) | Typical (ms) | Worst-Case (ms) |
|-------|-------------|--------------|-----------------|
| Sensor capture + deskew | 5 | 2 | 5 |
| Perception (detection + tracking + fusion) | 25 | 20 | 25 |
| Localization update | 10 | 5 | 10 |
| Planning (behavior + trajectory) | 40 | 30 | 40 |
| Control (solve + safety gate + CAN) | 15 | 8 | 15 |
| **Total** | **95** | **65** | **95** |
| **Margin** | **5** | **35** | **5** |

The 5 ms margin in the worst case is intentionally tight to motivate continuous performance monitoring and optimization. If the worst-case margin drops below 3 ms (measured over a 60-second window), the system raises a timing alarm.

---

## 10. Safety Architecture

### 10.1 Overview

The Safety Architecture defines the multi-layered defense strategy that ensures the autonomous vehicle operates safely even in the presence of software bugs, hardware faults, sensor failures, and unexpected environmental conditions. The architecture follows the ISO 26262 functional safety standard and implements the principles of defense-in-depth: no single fault, nor any reasonably foreseeable combination of faults, shall result in an unsafe vehicle state.

### 10.2 Functional Safety (ISO 26262)

The AVCS is developed according to ISO 26262, with ASIL allocations as follows:

| Subsystem | ASIL | Rationale |
|-----------|------|-----------|
| Control loop | ASIL-D | Directly controls actuators; failure can cause uncontrolled motion |
| Safety monitor | ASIL-D | Last line of defense; must never fail |
| Safety gate | ASIL-D | Prevents unsafe commands from reaching actuators |
| Localization | ASIL-B | Position error can cause route violations; mitigated by perception |
| Perception | ASIL-B | Detection failures mitigated by redundancy and conservative planning |
| Planning | ASIL-B | Bad plans mitigated by control safety gate and monitor |
| Communication | QM | V2X data is advisory; not safety-critical |
| Simulation | QM | Offline tool; not in the safety chain |

The ASIL-D subsystems (control, safety monitor, safety gate) are developed with the following measures:
- **Design:** Formal specification of safety requirements using temporal logic; design reviews with independent auditors.
- **Implementation:** MISRA-C:2012 compliant code, static analysis (Coverity, Polyspace), coding standard enforcement.
- **Verification:** Unit testing with 100% statement and branch coverage, integration testing, fault injection testing.
- **Validation:** Hardware-in-the-loop (HIL) testing with fault injection, proving ground tests, and structured on-road testing.

### 10.3 Fault Detection and Isolation

The system employs a hierarchical fault detection and isolation (FDI) strategy:

#### 10.3.1 Sensor Fault Detection

Each sensor is monitored for the following fault modes:

- **Signal loss:** No data received within 2× the expected period. Detected by the sensor driver's receive timeout check.
- **Out-of-range values:** Measurements outside physically plausible ranges (e.g., GPS latitude outside [-90°, 90°], IMU acceleration outside [-50g, 50g]). Detected by the input validation layer.
- **Stuck-at values:** The same value repeated for more than 10 consecutive samples (indicating a frozen ADC or communication fault). Detected by a change detector.
- **Cross-modal inconsistency:** Camera detects an object at a position where LiDAR sees free space, or vice versa. Detected by the fusion consistency checker.
- **GNSS spoofing/jamming:** GNSS position jumps by more than 50 m between consecutive updates, or the reported position velocity disagrees with the IMU by more than 5 m/s. Detected by the EKF innovation monitor.

When a sensor fault is detected, the sensor is marked as degraded, and its data is excluded from fusion. The fault is reported to the fault manager, which decides whether the remaining sensor suite provides sufficient capability to continue autonomous operation.

#### 10.3.2 Compute Fault Detection

- **Process health:** Each module process sends a heartbeat at 10 Hz. The system manager detects a crashed or hung process if 3 consecutive heartbeats are missed (300 ms).
- **Memory corruption:** Each critical data structure includes a CRC-32 checksum that is verified on read. A mismatch triggers an immediate process restart.
- **GPU health:** The GPU watchdog detects CUDA errors and GPU hangs. If the GPU becomes unresponsive for more than 200 ms, the perception module falls back to a CPU-only mode with reduced accuracy.
- **CAN bus health:** The CAN interface monitors bus-off conditions and error frame rates. A bus-off condition on the vehicle control CAN is a critical fault that triggers an emergency stop.

#### 10.3.3 Actuator Fault Detection

- **Steering actuator:** The steering angle sensor provides feedback. The control module compares the commanded and measured steering angles; a discrepancy exceeding 5° for more than 100 ms is flagged as a steering fault.
- **Throttle actuator:** The throttle position sensor provides feedback. A discrepancy exceeding 10% for more than 200 ms is flagged as a throttle fault.
- **Brake actuator:** The brake pressure sensor provides feedback. A discrepancy exceeding 15% for more than 100 ms is flagged as a brake fault.

### 10.4 Graceful Degradation Modes

The system defines four operational modes with progressively reduced capability:

```
┌─────────────────────────────────────────────────────────────────┐
│  Mode 0: FULL AUTONOMY                                         │
│  All sensors nominal. All modules operational.                  │
│  Capability: Full autonomous driving in all supported ODD.      │
│  Speed: Up to maximum operational speed (120 km/h).             │
├─────────────────────────────────────────────────────────────────┤
│  Mode 1: DEGRADED AUTONOMY                                     │
│  One sensor modality failed (e.g., camera or radar).           │
│  Capability: Autonomous driving with increased safety margins.  │
│  Speed: Limited to 80 km/h.                                    │
│  Restrictions: No lane changes, no unprotected turns.           │
├─────────────────────────────────────────────────────────────────┤
│  Mode 2: MINIMAL RISK CONDITION                                │
│  Multiple sensor failures or compute degradation.              │
│  Capability: Continue to next safe stop point only.             │
│  Speed: Limited to 30 km/h.                                    │
│  Action: Pull over to shoulder or park in safe location.        │
│  Timeout: Must reach safe stop within 60 seconds.               │
├─────────────────────────────────────────────────────────────────┤
│  Mode 3: EMERGENCY STOP                                        │
│  Critical fault (brake/steering fault, safety monitor trigger). │
│  Capability: None — vehicle is brought to immediate stop.       │
│  Action: Maximum deceleration in current lane.                  │
│  Hazard lights activated.                                       │
└─────────────────────────────────────────────────────────────────┘
```

The mode transitions are managed by the fault manager, which evaluates the current fault state against the transition criteria. Transitions to more restrictive modes (0→1, 1→2, 2→3) are immediate; transitions to less restrictive modes (1→0, 2→1) require the fault to be cleared for at least 10 seconds and a self-test to pass.

### 10.5 Emergency Stop Handling

The emergency stop procedure is the most safety-critical function of the AVCS. It is triggered by any of the following conditions:

1. Safety monitor detects an unresolvable safety violation.
2. Hardware watchdog timeout (software unresponsive).
3. Critical actuator fault (brake or steering).
4. Simultaneous failure of two or more sensor modalities.
5. Control command validation failure at the safety gate for 5 consecutive cycles.
6. Human operator trigger via the emergency stop button.

The emergency stop procedure executes the following sequence:

```
Step 1 (0–5 ms):    Disable throttle → zero throttle command
Step 2 (5–10 ms):   Apply maximum brake pressure → brake command 100%
Step 3 (10–20 ms):  Center steering → steer toward current lane center
Step 4 (20–50 ms):  Activate hazard lights → CAN command to body controller
Step 5 (50–100 ms): Transmit emergency BSM → V2X radio broadcast
Step 6 (ongoing):   Maintain brake pressure until vehicle speed = 0
Step 7 (at stop):   Engage parking brake, shift to PARK, activate horn
```

The emergency stop commands are hard-coded in the safety gate's firmware and do not depend on any software path beyond the safety monitor's heartbeat check. This ensures that even a complete software crash (kernel panic, power failure) results in a safe vehicle state via the hardware watchdog and the independently powered brake actuator.

---

## Appendix A: Acronyms

| Acronym | Definition |
|---------|------------|
| AVCS | Autonomous Vehicle Control System |
| BSM | Basic Safety Message |
| BEV | Bird's-Eye View |
| CAMP | Crash Avoidance Metrics Partnership |
| CTE | Cross-Track Error |
| CTRV | Constant Turn Rate and Velocity |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise |
| DDS | Data Distribution Service |
| DOP | Dilution of Precision |
| DSRC | Dedicated Short-Range Communications |
| ECEF | Earth-Centered Earth-Fixed |
| EEBL | Emergency Electronic Brake Light |
| EKF | Extended Kalman Filter |
| ENU | East-North-Up |
| FCW | Forward Collision Warning |
| FDI | Fault Detection and Isolation |
| FSM | Finite State Machine |
| GNSS | Global Navigation Satellite System |
| GPS | Global Positioning System |
| GWSA | Green Wave Speed Advisory |
| HD | High-Definition |
| HIL | Hardware-in-the-Loop |
| ICP | Iterative Closest Point |
| IMA | Intersection Movement Assist |
| IMU | Inertial Measurement Unit |
| ISO | International Organization for Standardization |
| LLA | Latitude, Longitude, Altitude |
| MAP | Map Data (J2735) |
| MPC | Model Predictive Controller |
| NDT | Normal Distributions Transform |
| NMS | Non-Maximum Suppression |
| ODD | Operational Design Domain |
| QP | Quadratic Programming |
| QoS | Quality of Service |
| RANSAC | Random Sample Consensus |
| RPN | Region Proposal Network |
| RSU | Roadside Unit |
| SCMS | Security Credential Management System |
| SIL | Software-in-the-Loop |
| SLAM | Simultaneous Localization and Mapping |
| SPAT | Signal Phase and Timing |
| TTC | Time-to-Collision |
| USS | Ultrasonic Sensor |
| V2X | Vehicle-to-Everything |
| WAVE | Wireless Access in Vehicular Environments |
| WCET | Worst-Case Execution Time |
| WSMP | WAVE Short Message Protocol |

---

*End of Document*
