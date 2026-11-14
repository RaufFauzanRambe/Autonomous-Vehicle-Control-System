# Autonomous Vehicle Control System — Example Suite

A research-oriented collection of runnable example modules that walk
through the full perception–localization–planning–control stack of an
autonomous vehicle, one stage per folder. Implementations are
deliberately simple (kinematic models, rule-based logic, first-order
filters) so each file can be read top-to-bottom as a reference, while
every module exposes documented extension points for replacing the
simple algorithms with research-grade ones.

## Repository layout

```
.
├── 01_basic_simulation/         # Kinematic bicycle model + road environment
├── 02_sensor_simulation/        # Synthetic camera / lidar / radar / GPS / IMU
├── 03_perception_demo/          # OpenCV pipeline: lanes, objects, sign colors
├── 04_localization_demo/        # GPS + IMU dead reckoning + fusion
├── 05_path_planning_demo/       # JSON map graphs + A* route planning
├── 06_motion_control_simulation/# PID speed control + path tracking scenarios
├── 07_ai_decision_demo/         # Rule-based driving decisions on scenarios
├── 08_full_system_demo/         # All stages combined in one loop
├── 09_visualization/            # Reusable plotters + summary dashboard
├── 10_testing_examples/         # pytest suite covering modules 01–07
├── shared/                      # Utilities, JSON schemas, sample dataset
└── documentation/               # Guide, installation, workflow, datasets, safety
```

## Quick start

```bash
pip install numpy matplotlib PyYAML pandas opencv-python scipy pytest

# run any example from the repository root
python 01_basic_simulation/main.py
python 05_path_planning_demo/main.py

# show interactive plot windows instead of (or in addition to) saved files
python 06_motion_control_simulation/main.py --show

# run the test suite
python -m pytest 10_testing_examples -v
```

Every `main.py` accepts `--config`, `--show`, `--output-dir` and
`--seed`. By default each run writes PNG figures and a
`run_summary.json` with key metrics into the module's `outputs/` folder,
so experiments can be compared without any interactive session.

## Common conventions

* **Units** are SI and stated in every docstring and config comment:
  meters, seconds, rad (CCW positive, 0 rad = +x axis), m/s, m/s².
* **Coordinates**: flat 2-D world frame, x forward / y left.
* **Configs**: YAML, one file per module, each parameter documented
  inline with unit and typical range.
* **Data exchange**: JSON maps/scenarios in `maps/` and `sample_data/`,
  validated against the JSON Schemas in `shared/schemas/`.
* **Logging**: `shared/utilities/logging_utils.py` gives every module the
  same log format.

## Suggested reading order for researchers

1. `01_basic_simulation` — state representation and dynamics integration.
2. `02_sensor_simulation` — how ground truth becomes noisy measurements.
3. `04_localization_demo` — fusing noisy measurements back into a pose.
4. `03_perception_demo` — extracting structure from camera frames.
5. `05_path_planning_demo` — graph search on routable maps.
6. `06_motion_control_simulation` — tracking a planned path.
7. `07_ai_decision_demo` — behavioural layer above control.
8. `08_full_system_demo` — everything in one loop.

See `documentation/example_guide.md` for the full walkthrough and
`documentation/safety_notes.md` for the scope and limits of these
simulation-only models.
