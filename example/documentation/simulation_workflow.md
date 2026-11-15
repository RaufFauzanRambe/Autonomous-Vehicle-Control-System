# Simulation Workflow

This document describes how a run moves through the stack, the file
conventions that make modules interoperable, and how to chain modules
into larger experiments.

## Stage pipeline

```
                 ┌──────────────────────────────────────────────┐
                 │            08_full_system_demo               │
                 │  sense → localize → plan → act (per tick)    │
                 └──────┬──────────┬──────────┬────────┬────────┘
                        │          │          │        │
   ground truth ──▶ 02 sensors  04 fusion  05 A*    06 PID /
   (01 model)         (noise)    (pose)   (route)   steering
                        │          │          │        │
                        ▼          ▼          ▼        ▼
                   03 perception (lanes, objects, signs)
```

* **01** produces ground-truth states with the kinematic bicycle model.
* **02** converts states into modality-specific noisy readings, each in
  the shared envelope `{sensor_type, timestamp, frame_id, data}`.
* **04** turns readings back into an estimated pose (low-pass GPS +
  IMU dead reckoning + complementary filter).
* **03** extracts structure (lanes, objects, sign classes) from frames.
* **05** turns a routable map and start/goal nodes into a resampled
  waypoint polyline.
* **06** tracks that polyline with a PID speed loop and heading law,
  reusing the 01 dynamics.
* **07** sits above control: ordered rules map an environment snapshot
  to an action (`stop` / `follow` / `cruise`).
* **08** wires simplified versions of every stage into one tick loop
  with logging and result files.

## File conventions

| Artifact type | Convention | Example |
|---|---|---|
| Module config | one YAML per module, every parameter commented with unit and typical range | `01_basic_simulation/simulation_config.yaml` |
| Maps / scenarios | JSON or YAML with SI units, path entries resolved module-relative | `05_path_planning_demo/maps/simple_road_map.json` |
| Sensor exchange | JSON envelope per `shared/schemas/sensor_data.schema.json` | `02_sensor_simulation/outputs/sensor_readings.jsonl` |
| State exchange | `VehicleState` fields per `shared/schemas/vehicle_state.schema.json` | `shared/sample_data/sample_vehicle_data.csv` |
| Figures | PNG, dpi 150, written into the module's `outputs/` (or `plots/`) | `outputs/localization_trace.png` |
| Metrics | one `run_summary.json` per run, written via `shared/utilities/data_loader.save_json` | `outputs/run_summary.json` |

## Running one experiment

```bash
# 1. choose parameters
$EDITOR 06_motion_control_simulation/scenarios/curved_road.yaml

# 2. run headless with a fixed seed
python 06_motion_control_simulation/main.py --seed 42

# 3. inspect outputs
ls 06_motion_control_simulation/outputs/
cat 06_motion_control_simulation/outputs/run_summary.json

# 4. optionally open plots interactively
python 06_motion_control_simulation/main.py --show
```

The console prints the headline metrics; the JSON keeps the full record
so batch scripts can compare seeds and configurations without parsing
figures.

## Chaining modules

The modules deliberately exchange plain data structures instead of
framework objects, so chaining is a matter of calling one module's API
from another script:

1. Plan a route (05) and pass `waypoints` to 06 as the reference.
2. Feed `VehicleState` histories (01) into 02's sensors, then play the
   resulting `sensor_readings.jsonl` through your own estimator and
   score it against ground truth the way 04 does.
3. Use 07's `driving_scenarios.json` as a regression benchmark: any new
   decision policy must reproduce `expected_decision` on all four
   scenarios before it is compared on harder cases.
4. Run `09_visualization/dashboard.py` after batch runs to see, at a
   glance, which modules produced artifacts and when.

## Timing and numerical notes

* Default integration step is `dt = 0.05 s` (01) or `dt = 0.1 s`
  (02/04); keep Euler steps at least an order of magnitude below the
  fastest dynamics you introduce.
* All angles are radians, counter-clockwise positive, 0 aligned with the
  +x axis; `wrap_angle` helpers map to [-pi, pi] where heading errors
  are computed.
* Speeds are body-frame longitudinal; the world frame is flat 2-D with
  x forward and y left, so leftward curvature is positive.

## Extending a scenario set

Copy an existing scenario YAML, adjust `target_speed_mps`, geometry and
duration, and add the filename to the module config's `scenarios.files`
list. Metrics are computed per file, so new scenarios appear in
`run_summary.json` automatically — no code changes required.
