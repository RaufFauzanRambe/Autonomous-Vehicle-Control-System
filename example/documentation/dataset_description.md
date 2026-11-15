# Dataset Description

All datasets shipped with the example suite are synthetic, generated
with a fixed RNG seed (42) so they are bit-stable. Units are SI
everywhere; angles are radians unless a column name says otherwise.

## shared/sample_data/sample_vehicle_data.csv

A 25-step kinematic trace of a straight-line acceleration maneuver.

| Column | Unit | Description |
|---|---|---|
| `timestamp` | s | sample time, `dt = 0.5 s` |
| `x` | m | world-frame longitudinal position |
| `y` | m | world-frame lateral position (0: straight line) |
| `yaw` | rad | heading (0: aligned with +x) |
| `speed` | m/s | body-frame longitudinal velocity |
| `acceleration` | m/s^2 | longitudinal acceleration (1.5 during the first 6 s, then 0) |
| `steering_angle` | rad | front wheel angle (0: straight) |

The file matches the state fields in
`shared/schemas/vehicle_state.schema.json` and is the input for the
`09_visualization/vehicle_visualizer.py` demo.

## 02_sensor_simulation/sample_data/telemetry/sample_telemetry.csv

A 24-row example of the sensor envelope written as tabular data — one
row per reading, cycling through the five modalities at 0.2 s spacing.
Generic column names carry modality-dependent values:

| Column | Description |
|---|---|
| `timestamp` | acquisition time [s] |
| `sensor_type` | one of `camera`, `lidar`, `radar`, `gps`, `imu` |
| `frame_id` | device/coordinate frame, e.g. `gps_antenna`, `lidar_top` |
| `value_1` .. `value_3` | modality-specific, see below |

Per-modality semantics of the value columns (documented for analysts;
the runtime envelope in `sensor_readings.jsonl` uses explicit names):

* `gps`: `value_1` = x [m], `value_2` = y [m], `value_3` = fix quality [-].
* `imu`: `value_1` = ax [m/s^2], `value_2` = ay [m/s^2], `value_3` = yaw rate [rad/s].
* `radar`: `value_1` = range [m], `value_2` = azimuth [rad], `value_3` = range rate [m/s].
* `lidar`: `value_1` = x [m], `value_2` = y [m] (sensor frame), `value_3` = intensity proxy [-].
* `camera`: `value_1` = u [px], `value_2` = v [px], `value_3` = confidence [-].

The `sample_data/camera/` and `sample_data/lidar/` folders are empty
placeholders (`.gitkeep`) for real frame exports.

## 04_localization_demo/sample_data/sample_position.csv

60 rows at `dt = 0.1 s`: ground truth is a constant 5.0 m/s drive along
+x; GPS readings carry sigma = 1.5 m Gaussian noise.

| Column | Unit | Description |
|---|---|---|
| `timestamp` | s | sample time |
| `gps_x`, `gps_y` | m | noisy GPS position (sigma 1.5 m) |
| `gps_fix_quality` | - | constant 0.9 indicator |
| `ground_truth_x`, `ground_truth_y` | m | noise-free reference trajectory |
| `imu_ax` | m/s^2 | longitudinal acceleration including a KNOWN +0.1 m/s^2 bias (removed by `imu.accel_bias_mpss` in the module config) |
| `imu_ay` | m/s^2 | lateral acceleration (zero mean, sigma 0.05) |
| `imu_yaw_rate` | rad/s | gyro z-rate (zero mean, sigma 0.005) |

This is the evaluation input for module 04: raw-GPS RMSE on the shipped
data is about 1.9 m, the fused estimate about 0.5 m.

## 05_path_planning_demo/maps/*.json

Routable graphs with explicit units:

| Field | Unit | Description |
|---|---|---|
| `nodes[].x`, `nodes[].y` | m | node coordinates in the world frame |
| `links[].from_id`, `links[].to_id` | - | directed edge endpoints (adjacency built undirected by default) |
| `links[].speed_limit_mps` | m/s | legal speed on the edge |
| `links[].lanes` | - | lane count (informational) |
| `links[].length_m` | m | optional; derived from node coordinates when absent |

`simple_road_map.json`: 8 nodes along a 350 m avenue with a gentle
offset between n2 and n5. `intersection_map.json`: five nodes around a
four-way intersection with 100 m approaches.

## 07_ai_decision_demo/sample_data/driving_scenarios.json

Four benchmark scenarios (`free_flow_cruise`, `red_light_stop`,
`pedestrian_crossing`, `slow_lead_vehicle`). Each scenario carries:

| Field | Unit | Description |
|---|---|---|
| `ego_state.x`, `.y` | m | ego position |
| `ego_state.yaw` | rad | ego heading |
| `ego_state.speed` | m/s | ego speed |
| `actors[].type` | - | `vehicle` or `pedestrian` |
| `actors[].x`, `.y` | m | actor position |
| `actors[].speed` | m/s | actor speed |
| `actors[].heading` | rad | actor heading |
| `road_features[].type` | - | `traffic_light`, `pedestrian_crossing`, `speed_limit` |
| `road_features[].distance_m` | m | distance ahead of the ego |
| `road_features[].state` | - | e.g. `red`, `occupied`, `clear` |
| `expected_decision` | - | `cruise`, `stop` or `follow`; the benchmark target |

The file doubles as a regression suite: any decision policy is expected
to reproduce `expected_decision` on all four scenarios.

## Generated (non-shipped) artifacts

Each run also writes derived datasets into module outputs —
`sensor_readings.jsonl` (schema-validated envelopes), `run_summary.json`
metrics, and `results/<scenario>_results.json` tick records in module
08. These are reproducible from the inputs above via the documented
commands and are safe to delete.
