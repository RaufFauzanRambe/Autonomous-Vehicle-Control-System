# sample_data/telemetry — Synchronized vehicle telemetry

Seven files describing one consistent 10 s drive: accelerate to 8 m/s
(0-4 s), cruise (4-8 s), brake to 2 m/s (8-10 s), gentle left turn
(0.1 rad/s between 5 s and 8 s). Ground truth starts at rest at the
scenario origin (lat 37.422000, lon -122.084000); x = east, y = north.
All modalities derive from the same Euler-integrated motion grid, so
cross-sensor consistency is testable (see `tests/test_telemetry_data.py`).

## Files and columns

| File | Rate | Columns (units) |
|---|---|---|
| `vehicle_motion.csv` | 10 Hz, 101 rows | `timestamp_s` [s], `speed_mps` [m/s], `acceleration_mpss` [m/s^2], `yaw_rate_radps` [rad/s], `position_x_m` [m], `position_y_m` [m], `distance_m` [m] |
| `gps_coordinates.csv` | 10 Hz, 101 rows | `timestamp_s`, `latitude_deg`, `longitude_deg` [deg], `altitude_m` [m], `fix_quality` [-], `hdop` [-]; position noise sigma 1.5 m |
| `imu_readings.csv` | 50 Hz, 501 rows | `timestamp_s`, `accel_x_mpss` [m/s^2] (includes a KNOWN +0.1 bias), `accel_y_mpss` (centripetal v*yaw_rate + noise), `accel_z_mpss` (gravity ~9.81), `gyro_x/y/z_radps` [rad/s] |
| `vehicle_status.csv` | 1 Hz, 11 rows | `timestamp_s`, `gear`, `battery_voltage_v` [V], `battery_soc_pct` [%], `odometer_m` [m], `system_mode` |
| `sensor_timestamps.csv` | index, 670 rows | `global_timestamp_s`, `sensor_type`, `sample_id`, `sensor_timestamp_s` (includes latency), `latency_ms` [ms] |
| `sample_telemetry.csv` | 5 Hz, 50 rows | envelope format: `timestamp`, `sensor_type`, `frame_id`, `value_1..value_3` (semantics per modality, see below) |
| `telemetry_metadata.json` | - | Scenario, conventions, biases, noise, per-file row counts |

## Envelope value semantics

| sensor_type | value_1 | value_2 | value_3 |
|---|---|---|---|
| `gps` | x [m] (noisy) | y [m] (noisy) | fix quality [-] |
| `imu` | accel_x [m/s^2] (with bias) | accel_y [m/s^2] | gyro_z [rad/s] |
| `radar` | range to lead vehicle [m] | azimuth [rad] | range rate [m/s] |
| `lidar` | point x [m] (sensor frame) | point y [m] | intensity [-] |
| `camera` | bbox center u [px] | bbox center v [px] | confidence [-] |

Camera rows use the sentinel `(-1, -1, 0.0)` once the ego has passed all
obstacles and nothing is in view.

## Consistency notes

* IMU `accel_x` carries a documented +0.1 m/s^2 bias — subtract it
  before integrating speed (the tests do exactly that and expect the
  result to match `vehicle_motion.speed_mps`).
* GPS lat/lon map back to local meters via
  `x = (lon - lon0) * 111320 * cos(lat0)`, `y = (lat - lat0) * 111320`.
* `sensor_timestamps.csv` is the synchronization index: per-sensor
  latencies are camera 45 ms, lidar 80 ms, GPS 120 ms, IMU 5 ms,
  status 10 ms, envelope 20 ms (plus small jitter).

## Extension points

Add modalities by appending rows to the envelope and extending
`telemetry_metadata.json`; keep one ground-truth grid so consistency
tests remain meaningful.
