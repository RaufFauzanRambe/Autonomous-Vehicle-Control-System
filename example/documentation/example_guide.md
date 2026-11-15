# Example Guide

This guide explains what each module demonstrates, the concepts behind it,
and how the modules connect. Implementations are deliberately simple —
every file can be read top-to-bottom — and every module documents its own
extension points for research-grade replacements.

## Module overview

| Module | Stage demonstrated | Core algorithm | Key inputs | Key outputs |
|---|---|---|---|---|
| `01_basic_simulation` | Vehicle dynamics | Kinematic bicycle model, Euler integration | `simulation_config.yaml`, `vehicle_parameters.yaml`, `road_map.json` | trajectory PNG, `run_summary.json` |
| `02_sensor_simulation` | Sensing | Synthetic noise models per modality | `sensor_config.yaml` | `sensor_readings.jsonl`, `sensor_trace.png` |
| `03_perception_demo` | Perception | Canny + Hough lanes, contour objects, color signs | synthetic frame + `perception_config.yaml` | 4 overlay PNGs, `run_summary.json` |
| `04_localization_demo` | Localization | Low-pass GPS, IMU dead reckoning, complementary filter | `sample_position.csv` | `localization_trace.png`, RMSE metrics |
| `05_path_planning_demo` | Planning | A* on a node/link graph + waypoint resampling | two JSON maps | `planned_route_*.png`, path metrics |
| `06_motion_control_simulation` | Control | PID speed loop + pure-pursuit-style steering | three scenario YAMLs | `control_*.png`, RMSE / cross-track metrics |
| `07_ai_decision_demo` | Behaviour | Ordered rule-based decisions | `driving_scenarios.json` | `decision_*.png`, decision/expectation match |
| `08_full_system_demo` | Integration | sense → localize → plan → act loop | three environment YAMLs | `logs/*.log`, `results/*.json` |
| `09_visualization` | Tooling | Reusable plotters + artifact dashboard | other modules' outputs, shared CSV | `plots/*.png` |
| `10_testing_examples` | Quality | pytest suite over modules 01–07 | module code | 19 test results |

## Suggested reading order

1. **`01_basic_simulation`** — start here for the shared state
   representation. `vehicle/vehicle_state.py` defines the canonical
   `VehicleState` (timestamp [s], x/y [m], yaw [rad], speed [m/s]) that
   modules 02, 07, 08 and 09 all consume. `vehicle_model.py` shows the
   simplest defensible dynamics: a kinematic bicycle model with linear
   drag, integrated with Euler steps.
2. **`02_sensor_simulation`** — how ground truth becomes noisy
   measurements. Each sensor returns the same JSON envelope
   (`sensor_type`, `timestamp`, `frame_id`, `data`) validated against
   `shared/schemas/sensor_data.schema.json`, which makes downstream
   parsing uniform.
3. **`04_localization_demo`** — the inverse problem: fusing noisy
   measurements back into a pose. Watch the complementary filter trade
   GPS noise against IMU drift (fused RMSE ≈ 0.5 m vs raw GPS ≈ 1.9 m on
   the shipped dataset).
4. **`03_perception_demo`** — extracting structure from camera frames.
   The synthetic generator stands in for a real camera; swap in frames
   from your dataset and the pipeline (edges → lanes, contours → objects,
   color → sign class) is unchanged.
5. **`05_path_planning_demo`** — graph search. The map format (nodes +
   links with `speed_limit_mps`) is intentionally close to routable
   road-network extracts.
6. **`06_motion_control_simulation`** — tracking a reference path. The
   speed PID and heading controller run against the same bicycle model
   as module 01, closing the loop.
7. **`07_ai_decision_demo`** — the behavioural layer: rule precedence
   (traffic light > pedestrian > follow > cruise) over a benchmark
   scenario file with expected outcomes.
8. **`08_full_system_demo`** — all stages in one loop, with per-tick
   logs and per-scenario result JSONs.

## Running the whole suite

```bash
python 01_basic_simulation/main.py
python 02_sensor_simulation/main.py
python 03_perception_demo/main.py
python 04_localization_demo/main.py
python 05_path_planning_demo/main.py
python 06_motion_control_simulation/main.py
python 07_ai_decision_demo/main.py
python 08_full_system_demo/main.py
python 09_visualization/dashboard.py
python -m pytest 10_testing_examples -v
```

Every `main.py` accepts `--config`, `--show`, `--output-dir`, `--seed`
and writes a `run_summary.json` with its headline metrics, so runs are
comparable across machines and seeds. `09_visualization/dashboard.py`
summarizes which artifacts exist after your runs.

## Where to intervene as a researcher

* Dynamics: replace `VehicleModel.step` with a dynamic bicycle model or
  tire models; keep the signature and everything downstream still works.
* Sensors: replace each synthetic sensor with dataset playback (the
  envelope format stays the same).
* Perception: the detector/lane/sign classes are interface-shaped (a
  `detect` / `classify` call returning plain data) — plug in deep models
  without touching `main.py`.
* Localization: the complementary filter in `sensor_fusion.py` is a
  drop-in point for an error-state EKF.
* Planning: `AStarPlanner.plan` can be swapped for weighted A*, hybrid
  A* or sampling planners; `waypoints_from_path` already produces the
  resampled reference the controller expects.
* Control: `PIDController` / the heading law are intentionally minimal;
  MPC or LQR implementations can reuse the same reference polyline.
* Decisions: the ordered rules in `scenario_evaluator.py` define the
  benchmark; utility-based or learned policies should reproduce the
  expected outcomes in `driving_scenarios.json` before being compared.

See `simulation_workflow.md` for data-flow details, `dataset_description.md`
for every sample dataset's columns and units, and `safety_notes.md` for
the scope and limits of these models.
