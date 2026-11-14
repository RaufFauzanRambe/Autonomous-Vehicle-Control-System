# 02_sensor_simulation

## Purpose

This module simulates the five canonical automotive sensor families
(camera, lidar, radar, GPS, IMU) as lightweight analytic models driven by a
simple straight-line ego ground truth. Each sensor emits readings in the
project-wide envelope format (see `shared/schemas/sensor_data.schema.json`),
so the output JSONL is directly usable as fixture data for fusion, detection
or calibration experiments.

## Key concepts and physical parameters

- Ground truth: constant-speed straight-line motion, `x = ego_speed * t` [m],
  `yaw = 0` [rad], `speed = ego_speed_mps` [m/s] (documented stand-in
  generator; `dt_s` [s] sets the update rate).
- Envelope contract: every reading dict has EXACTLY the keys
  `sensor_type` (`camera|lidar|radar|gps|imu`), `timestamp` [s],
  `frame_id` [-] and `data` (sensor-specific payload).
- Camera: pinhole projection with `focal_px = (image_width_px/2) / tan(fov_rad/2)`
  [px]; one virtual lead vehicle at a constant ego-relative gap
  (45 m ahead, 3 m left); bbox half-size 20 px with Gaussian jitter
  `detection_noise_std_px` [px]; detections dropped beyond
  `max_detection_range_m` [m].
- Lidar: synthetic front-semicircle scan, `points_per_scan` bearings in
  [-pi/2, pi/2] [rad] around the ego heading; ranges uniform in
  [10, `max_range_m`] [m] plus Gaussian noise `range_noise_std_m` [m],
  clamped to `[min_range_m, max_range_m]` [m]; points returned in the
  sensor frame (x forward, y left).
- Radar: two static virtual targets at fixed WORLD-frame coordinates
  (40.0, -2.0) m ("A") and (70.0, 3.0) m ("B"); reports `range_m` [m],
  `azimuth_rad` [rad] (wrapped to [-pi, pi], relative to ego yaw) and
  `range_rate_mps = -ego_speed * cos(azimuth)` [m/s] plus Gaussian noise.
- GPS: independent per-axis Gaussian position noise `position_noise_std_m` [m],
  `fix_quality` fixed at 0.9.
- IMU: longitudinal acceleration = ground-truth acceleration + constant bias
  `bias_ax_mpss` [m/s^2] + Gaussian noise `accel_noise_std_mpss` [m/s^2];
  yaw rate = finite difference of ego yaw [rad/s] + Gaussian noise
  `gyro_noise_std_radps` [rad/s].
- All randomness comes from `np.random.default_rng(seed)` per sensor;
  runs are deterministic for a fixed `--seed`.

## How to run

All commands work from the project root; paths are resolved relative to the
module directory, never the current working directory.

```bash
# Headless run (default): saves figure + JSONL + run_summary.json into 02_sensor_simulation/outputs/
python 02_sensor_simulation/main.py

# Open an interactive matplotlib window instead of headless-only rendering
python 02_sensor_simulation/main.py --show

# Custom config, output directory and random seed
python 02_sensor_simulation/main.py --config 02_sensor_simulation/config/sensor_config.yaml --output-dir /tmp/sim02_out --seed 7
```

## Inputs and outputs

Input files:

- `config/sensor_config.yaml` - simulation timing, ego speed, logging stride
  and per-sensor parameter sections (overridable via `--config`).
- `sample_data/telemetry/sample_telemetry.csv` - ready-made example of the
  reading envelope format in tabular form (one row per reading with
  `timestamp`, `sensor_type`, `frame_id` and three value columns).
- `sample_data/camera/` and `sample_data/lidar/` - empty frame folders where
  real exports (images, point clouds) would be placed; the current synthetic
  sensors do not read files from them.

Generated output files (written into `outputs/` or `--output-dir`):

- `outputs/sensor_readings.jsonl` - one JSON object per line, each a
  schema-conforming envelope reading (filtered by `record_every_n_steps`).
- `outputs/sensor_trace.png` - 2x3 panel figure (GPS vs ground truth, lidar
  last scan in sensor frame, radar range per target, IMU longitudinal
  acceleration, camera detection count, ego speed), dpi 150.
- `outputs/run_summary.json` - key metrics of the run.

## Metrics to observe

- `n_readings` [-]: number of envelope records written to
  `sensor_readings.jsonl` (5 sensors x logged steps).
- `lidar_points_last_scan` [-]: points in the final lidar scan; equals
  `points_per_scan` (120 by default) and should never drift.
- `radar_target_count` [-]: targets visible in the final radar reading
  (2 by default; drops to 1 if a target leaves `max_range_m`).
- `camera_detections_total` [-]: summed detection count over the run; equals
  the number of steps while the constant-gap lead vehicle stays within
  `max_detection_range_m`.
- JSONL payload sanity: GPS x grows linearly with noise around
  `ego_speed_mps * t`; radar `range_rate_mps` trends toward `-ego_speed`
  as targets align with the heading; IMU `ax_mpss` hovers around the
  configured `bias_ax_mpss`.

## Extension points

- Realistic ground truth: replay states from `01_basic_simulation`
  (curved trajectories expose the currently unused ego-yaw transforms).
- Physically grounded lidar: ray-cast beams against the road environment
  instead of drawing uniform ranges.
- Richer radar: per-target RCS/amplitude, azimuth noise, FOV filtering and
  multi-path ghosts.
- Temporally correlated noise: colored noise (Ornstein-Uhlenbeck) for GPS
  and IMU, plus Allan-variance-based bias calibration.
- Schema enforcement: validate every JSONL line against
  `shared/schemas/sensor_data.schema.json` before writing.
