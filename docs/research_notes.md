# Research Notes - Autonomous Vehicle Control System

This document compiles key research findings, algorithm references, and design decisions that inform the AVCS implementation.

---

## Table of Contents

1. [Sensor Fusion](#sensor-fusion)
2. [Localization & SLAM](#localization--slam)
3. [Perception](#perception)
4. [Planning & Decision Making](#planning--decision-making)
5. [Vehicle Control](#vehicle-control)
6. [V2X Communication](#v2x-communication)
7. [Safety & Functional Safety](#safety--functional-safety)
8. [Key References](#key-references)

---

## Sensor Fusion

### Extended Kalman Filter for Multi-Sensor Fusion

The choice of EKF over UKF or Particle Filter for the primary fusion algorithm was driven by the following considerations:

**EKF Advantages:**
- Computational efficiency: O(n²) per update step where n is state dimension, compared to O(n³) for UKF with 2n+1 sigma points
- Deterministic execution time: Critical for real-time control loops running at 100 Hz
- Well-understood convergence properties: Extensive literature on tuning and stability
- Sufficient accuracy for moderately nonlinear vehicle dynamics: The vehicle operates primarily in the linear regime with small angles

**Limitations Addressed:**
- Linearization error: Mitigated by frequent updates (20+ Hz) keeping prediction steps short
- Single-mode assumption: Addressed by maintaining separate EKF instances for multi-hypothesis tracking
- Gaussian assumption: Acceptable for position/velocity estimation where measurement noise is approximately Gaussian

### Fusion Architecture: Late Fusion vs. Early Fusion

We adopt a **hybrid fusion** strategy:

| Aspect | Early Fusion | Late Fusion | AVCS Choice |
|--------|-------------|-------------|-------------|
| Input | Raw sensor data | Detection outputs | Both |
| Alignment | Pixel/point level | Object level | Hybrid |
| Latency | Higher (wait for all sensors) | Lower | Late for time-critical |
| Accuracy | Higher (richer information) | Lower | Early for LiDAR+Camera |
| Robustness | Lower (sensor failure propagates) | Higher | Late for radar |

- **LiDAR + Camera**: Early fusion using point cloud painting (projecting image features onto 3D points)
- **Radar**: Late fusion due to complementary information (velocity from Doppler, poor angular resolution)
- **GPS/IMU**: Sequential EKF updates in the localization module

---

## Localization & SLAM

### State Vector Design

The 15-dimensional EKF state vector was chosen as:

```
x = [px, py, pz, vx, vy, vz, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
     ─────────────────  ─────────────────  ────────────────────  ──────────────────────
     Position (3)       Velocity (3)       Orientation (3)       IMU Biases (6)
```

**Rationale for bias states:**
- IMU biases are the dominant error source in dead-reckoning localization
- Online bias estimation enables extended GPS-denied operation
- Accelerometer and gyroscope biases exhibit random walk behavior, making them suitable for state augmentation
- Including bias states reduces the need for frequent IMU recalibration

### ICP Scan Matching

The Iterative Closest Point algorithm is used for LiDAR-based localization with the following enhancements over vanilla ICP:

1. **Point-to-Plane ICP**: Using surface normals for cost computation reduces convergence iterations by 30-50% compared to point-to-point ICP in structured environments (roads, buildings)

2. **Voxel Grid Pre-filtering**: Downsampling to 0.1m voxels reduces point count by 10x while preserving geometric structure, enabling <50ms scan matching on CPU

3. **Initial Guess from Motion Prediction**: Using EKF-predicted pose as the initial transform reduces ICP iterations from ~50 to ~10

4. **Robust Outlier Rejection**: Trimming correspondences beyond 2.0m distance eliminates dynamic objects (vehicles, pedestrians) from the matching

### Loop Closure Detection

Loop closure is essential for correcting accumulated drift in long-duration autonomous operation. Our approach:

1. **Location-based trigger**: Check all keyframes within 15m radius of current position
2. **Geometric verification**: ICP matching with tight convergence criteria (RMSE < 0.3m)
3. **Pose graph optimization**: Gauss-Newton on SE(3) manifold with information-weighted edges
4. **Map update**: Rebuild local map from optimized keyframes after pose graph adjustment

**Key insight from research**: Loop closure detection should be conservative (high precision, moderate recall) to avoid adding incorrect constraints that corrupt the map. A single incorrect loop closure can cause catastrophic map distortion.

---

## Perception

### Object Detection Model Selection

| Model | Modality | FPS (RTX 3080) | mAP (nuScenes) | Latency |
|-------|----------|-----------------|-----------------|---------|
| PointPillars | LiDAR | 42 | 58.1 | 24ms |
| CenterPoint | LiDAR | 16 | 65.5 | 62ms |
| YOLOv8 | Camera | 85 | 52.3 | 12ms |
| BEVFusion | LiDAR+Camera | 10 | 70.2 | 100ms |
| TransFusion | LiDAR+Camera | 8 | 68.9 | 125ms |

**Design decision**: Use CenterPoint as the primary LiDAR detector for the balance of accuracy and speed. Use YOLOv8 for camera-only backup. BEVFusion reserved for offline processing and high-accuracy scenarios.

### Multi-Object Tracking: Kalman + Hungarian

The tracking pipeline follows the Tracking-by-Detection paradigm:

1. **State representation**: Constant Turn Rate and Velocity (CTRV) model for vehicles, Constant Velocity (CV) for pedestrians
   - CTRV provides better prediction during turning maneuvers common in urban driving
   - CV is sufficient for pedestrians who change direction unpredictably (random walk)

2. **Data association**: Hungarian algorithm on a cost matrix combining:
   - Mahalanobis distance (weighted by track covariance)
   - Appearance similarity (when ReID features available)
   - Class consistency penalty

3. **Track lifecycle**:
   - Tentative → Confirmed after 3 consecutive hits (reduces false alarms)
   - Confirmed → Coasting after 3 missed detections (handles occlusion)
   - Coasting → Lost after 10 missed detections (removes stale tracks)

### Ground Plane Removal

RANSAC-based plane fitting is used for ground removal with the following considerations:

- **Inlier threshold**: 0.2m works well for flat roads; may need adaptive threshold for hilly terrain
- **Plane model**: ax + by + cz + d = 0 with constraint that normal points upward (reduces false ground removal on sloped surfaces)
- **Sequential RANSAC**: Multiple plane extraction handles multi-level ground (curbs, parking structures)
- **Alternative approach**: Cloth Simulation Filtering (CSF) provides better results on complex terrain but is 5x slower

---

## Planning & Decision Making

### Behavior Planning: FSM vs. POMDP

| Approach | Optimality | Computation | Interpretability | Robustness |
|----------|------------|-------------|-------------------|------------|
| FSM | Local | O(1) | High | High |
| Decision Tree | Local | O(depth) | High | Medium |
| POMDP | Global | Exponential | Low | Low |
| Reinforcement Learning | Learned | Variable | Low | Variable |

**Design decision**: FSM with cost-based transitions provides the best balance for production deployment:
- Deterministic behavior is easier to validate and certify
- Cost functions can be hand-tuned for specific operational design domains
- FSM transitions are auditable for regulatory compliance
- POMDPs remain an active research area for handling uncertainty in intersection scenarios

### Trajectory Optimization: Frenet Frame

The Frenet frame (s, d) representation decomposes the trajectory optimization problem into longitudinal and lateral components, which offers several advantages:

1. **Decoupled optimization**: Longitudinal (speed) and lateral (position) profiles can be optimized independently and then combined
2. **Road-aligned coordinates**: Naturally follows the road geometry, making constraint formulation intuitive
3. **Separate comfort criteria**: Longitudinal comfort (acceleration/jerk) and lateral comfort (curvature/centripetal acceleration) can be independently bounded
4. **Efficient replanning**: When only speed needs adjustment (e.g., following a slowing vehicle), only the longitudinal profile is re-optimized

**Constraint formulation:**
- Speed: 0 ≤ s_dot ≤ v_max
- Acceleration: a_min ≤ s_ddot ≤ a_max
- Jerk: j_min ≤ s_dddot ≤ j_max
- Lateral offset: |d| ≤ d_max (lane boundary)
- Lateral acceleration: |d_ddot| ≤ a_lat_max

---

## Vehicle Control

### PID vs. MPC Trade-offs

| Aspect | PID | MPC |
|--------|-----|-----|
| Computation | O(1) | O(N·n²) |
| Preview | None | Yes (horizon) |
| Constraints | Clipping only | Native handling |
| Tuning | 3 gains | Cost weights + constraints |
| Robustness | High (simple) | Medium (model-dependent) |
| Optimality | Local | Horizon-optimal |

**AVCS Strategy:**
- **Low speed (< 30 km/h)**: MPC for precise maneuvering in tight spaces with active constraint handling
- **Highway speed (> 30 km/h)**: PID for reliable, low-latency lane keeping with Stanley lateral control
- **Emergency**: Direct steering/throttle commands bypassing all controllers

### Bicycle Model Fidelity

The kinematic bicycle model is sufficient for most driving scenarios:

```
dx/dt = v * cos(ψ + β)
dy/dt = v * sin(ψ + β)
dψ/dt = v * cos(β) / (lr + lf) * tan(δ)
dv/dt = a
```

Where β = arctan(lr / (lr + lf) * tan(δ)) is the sideslip angle at the center of gravity.

**When to use the dynamic model:**
- High-speed driving (> 120 km/h) where tire forces approach the friction limit
- Emergency maneuvers requiring tire saturation modeling
- Winter driving on low-friction surfaces

The dynamic model adds Pacejka tire forces, lateral load transfer, and tire relaxation dynamics, increasing computational cost by approximately 5x.

---

## V2X Communication

### DSRC vs. C-V2X

| Feature | DSRC (802.11p) | C-V2X (3GPP Rel. 14+) |
|---------|----------------|------------------------|
| Spectrum | 5.9 GHz | 5.9 GHz (PC5) + Cellular |
| Latency | < 10 ms | < 20 ms (PC5), < 100 ms (Uu) |
| Range | ~300m | ~500m (PC5), Cellular range (Uu) |
| Deployment | Limited | Growing with 5G |
| Maturity | Proven | Evolving |
| Cost | Moderate | Higher |

**AVCS approach**: Implement DSRC as primary (mature standard, J2735 messages well-defined) with C-V2X as a future upgrade path. The message parser is designed to be transport-agnostic, allowing either radio technology.

### BSM (Basic Safety Message) Processing

The SAE J2735 BSM is transmitted at 10 Hz and contains:
- **Part I** (required): Position, speed, heading, brake status, vehicle size
- **Part II** (optional): Path history, path prediction, acceleration, steering angle, weather data

**Key processing considerations:**
1. **Message rate**: 10 Hz is the standard; adaptive rate (T.J. 2735) increases rate during hard braking
2. **Authentication**: Must verify digital signatures to prevent spoofing attacks
3. **Fusion with onboard sensors**: V2X detections have higher latency but provide 360° coverage beyond sensor FOV
4. **Staleness**: Messages older than 500ms should be discarded

---

## Safety & Functional Safety

### ISO 26262 ASIL Allocation

| Module | ASIL | Rationale |
|--------|------|-----------|
| Emergency Stop | ASIL D | Direct safety impact |
| Motion Control | ASIL D | Direct vehicle control |
| Perception | ASIL B | Indirect: object detection |
| Localization | ASIL B | Indirect: position knowledge |
| Planning | ASIL C | Semi-direct: decision making |
| V2X | QM | Supplementary information |
| Simulation | QM | Development only |

### Safety Monitor Architecture

The safety monitor operates as an independent watchdog with its own processing pipeline:

1. **Kinematic feasibility check**: Verify planned trajectory satisfies vehicle dynamics constraints
2. **Collision check**: Independent collision verification using raw sensor data (bypasses perception pipeline)
3. **Road boundary check**: Ensure vehicle remains within drivable area per HD map
4. **Timeout monitor**: Detect software hangs via heartbeat mechanism
5. **Sensor health monitor**: Check for sensor data staleness and inconsistency

If any safety check fails, the monitor triggers an escalation:
- Level 1: Warning (driver notification)
- Level 2: Safe stop (pull over to road edge)
- Level 3: Emergency stop (maximum deceleration in lane)

---

## Key References

### Sensor Fusion
1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Bar-Shalom, Y., Li, X.R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.

### Localization & SLAM
3. Zhang, J. & Singh, S. (2014). "LOAM: Lidar Odometry and Mapping in Real-time." *RSS*.
4. Shan, T. et al. (2020). "LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping." *IROS*.

### Perception
5. Shi, S. et al. (2019). "PointPillars: Fast Encoders for Object Detection from Point Clouds." *CVPR*.
6. Yin, J. et al. (2021). "Center-based 3D Object Detection and Tracking." *CVPR*.

### Planning
7. Werling, M. et al. (2010). "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame." *ICRA*.
8. Pivtoraiko, M. et al. (2009). "Differentially Constrained Mobile Robot Motion Planning in State Lattices." *JFR*.

### Control
9. Rajamani, R. (2012). *Vehicle Dynamics and Control* (2nd ed.). Springer.
10. Kong, J. et al. (2015). "Kinematic and dynamic vehicle models for autonomous driving control design." *IV*.

### V2X
11. SAE J2735 (2020). "Dedicated Short Range Communications (DSRC) Message Set Dictionary."
12. ETSI TS 102 894-2 (2018). "Intelligent Transport Systems; CAM Generation."

### Safety
13. ISO 26262 (2018). "Road vehicles - Functional safety."
14. Koopman, P. & Wagner, M. (2016). "Autonomous Vehicle Safety: An Interdisciplinary Challenge." *IEEE Intelligent Transportation Systems Magazine*.
