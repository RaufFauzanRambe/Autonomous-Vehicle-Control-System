# 04 Localization Demo

Multi-sensor localization on a recorded log: GPS low-pass filtering,
bias-corrected IMU dead reckoning and a complementary filter that fuses
both, evaluated against ground truth.

## Purpose

This module demonstrates how absolute but noisy GPS fixes and smooth but
drifting IMU dead reckoning are combined into one position estimate. The
recorded log `sample_data/sample_position.csv` contains 60 samples at
`dt = 0.1 s` of a vehicle moving at a constant 5.0 m/s along +x, with
noisy raw GPS, ground truth positions and IMU signals. The IMU forward
acceleration channel `imu_ax` contains a known +0.1 m/s^2 bias, which
the config removes (`imu.accel_bias_mpss`) before integration. The demo
quantifies how much the complementary fusion improves on raw GPS in
terms of 2-D position RMSE.

## Key concepts and physical parameters

- GPS low-pass filter: `low_pass_alpha` [-] in [0, 1], `est = alpha * meas + (1 - alpha) * est`; higher alpha trusts the newest fix, lower alpha smooths more but adds lag.
- Known IMU biases: `accel_bias_mpss` [m/s^2] = 0.1 removed from `imu_ax` (forward axis) and `gyro_bias_radps` [rad/s] removed from `imu_yaw_rate` before integration; uncompensated accelerometer bias integrates into a linear position ramp (0.1 m/s^2 grows to ~18 m of drift over the 5.9 s log).
- Dead reckoning: semi-implicit Euler integration, `v += a * dt`, position increment `dx = v * dt` [m]; velocity is not observable from accelerations alone, hence the initial-speed seeding below.
- Initial-speed estimation trick: `v0 = hypot(dx, dy) / dt` from GPS displacement over a time baseline [m/s]. Applied to only the first two fixes its standard deviation is `sqrt(2) * sigma_gps / dt` (about 15 m/s for this log with sigma_gps ~ 1.2 m and dt = 0.1 s), which would wreck the dead reckoning. The demo therefore applies the same displacement-over-time trick over the full logged baseline (5.9 s), reducing the speed error to roughly 0.3 m/s; the estimated `v0` is printed by every run.
- Complementary filter: `fused = alpha * (fused + imu_dx_dy) + (1 - alpha) * gps` with `complementary_alpha` [-] = 0.95; the IMU provides the high-frequency content and the GPS term continuously pulls the estimate back toward the absolute anchor, bounding drift.
- Metrics: 2-D RMSE [m] = sqrt of the mean squared Euclidean error over all samples; improvement [-] in percent relative to raw GPS; final drift [m] = Euclidean distance between the last fused position and the last ground-truth position.

## How to run

Run from the repository root:

```bash
python 04_localization_demo/main.py
```

Interactive preview windows (figures are still saved):

```bash
python 04_localization_demo/main.py --show
```

Custom configuration, output directory and seed:

```bash
python 04_localization_demo/main.py --config 04_localization_demo/config/localization_config.yaml \
    --output-dir 04_localization_demo/outputs --seed 42 --show
```

All file paths are resolved against the module directory, so the
commands work from any working directory as long as the script path is
given.

## Inputs and outputs

Inputs:

- `sample_data/sample_position.csv` — columns: `timestamp` [s], `gps_x` [m], `gps_y` [m], `gps_fix_quality` [-] (reported but unused by the demo), `ground_truth_x` [m], `ground_truth_y` [m], `imu_ax` [m/s^2] (carries the known +0.1 m/s^2 bias), `imu_ay` [m/s^2], `imu_yaw_rate` [rad/s]; 60 rows at dt = 0.1 s.
- `config/localization_config.yaml` — filter parameters, known biases and fusion weight.

Outputs (in `outputs/` unless `--output-dir` is given):

- `localization_trace.png` — three-panel matplotlib figure: trajectories (raw GPS, filtered GPS, fused, ground truth), 2-D position error vs time for raw GPS and fused, estimated speed vs time with the true 5.0 m/s line.
- `run_summary.json` — `n_samples`, `gps_rmse_m`, `filtered_gps_rmse_m`, `fused_rmse_m`, `improvement_pct`, `final_drift_m`.

## Metrics to observe

- `gps_rmse_m` — raw GPS accuracy, about 1.9 m for this log (sigma ~ 1.1 m along x, ~ 1.5 m along y).
- `filtered_gps_rmse_m` — after the low-pass filter, about 1.4 m: noise shrinks but a lag bias appears because the vehicle moves.
- `fused_rmse_m` — about 0.5 m: the IMU dead reckoning supplies smooth short-term motion while the GPS term bounds drift, clearly better than raw GPS.
- `improvement_pct` — about 70 percent with the default config; decreasing `complementary_alpha` toward 0.5 makes the fused estimate follow GPS noise again and the improvement collapses.
- `final_drift_m` — distance between the last fused position and ground truth; watch it grow if you set `imu.accel_bias_mpss` to 0 (the known bias is no longer removed and dead reckoning ramps away).
- `estimated initial speed` (console) — about 4.7-5.3 m/s depending on the realized GPS noise; compare with the true 5.0 m/s.

## Extension points

- Fusion: replace the complementary filter with a 15-state error-state EKF (position, velocity, attitude, accelerometer and gyro biases) or a factor-graph smoother; use `gps_fix_quality` as a measurement-noise input.
- Inertial model: strapdown INS with quaternion attitude propagation and body-to-world rotation of accelerations; ZUPT updates when the vehicle is stationary.
- Calibration: estimate the sensor biases from data (Allan variance, static intervals) instead of hard-coding them in the config.
- Dataset: larger logs with turns, GPS outages (the fusion already handles `gps_xy = None` as pure dead reckoning) and variable sample rates.
