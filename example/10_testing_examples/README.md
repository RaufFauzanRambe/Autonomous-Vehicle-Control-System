# 10_testing_examples

## Purpose

pytest suite that verifies the core algorithm contracts of modules
01-07 of the example suite. The tests double as executable
documentation of the cross-module interfaces: state dataclass roundtrip,
sensor reading envelope, filter convergence, graph-search validity and
rule-based decision agreement. Everything runs headless and writes no
artifacts outside the pytest temporary directory.

## Covered behavior

* `test_vehicle_model.py` - kinematic vehicle model (module 01):
  positive throttle raises speed, speed is clamped to the configured
  maximum, steering changes yaw while rolling, and
  `VehicleState.to_dict()/from_dict()` roundtrips every field.
* `test_sensor_data.py` - synthetic sensors (module 02): every sensor
  returns the shared reading envelope (`sensor_type`, `timestamp`,
  `frame_id`, `data`) with the expected sensor type, identical seeds
  produce identical GPS readings, lidar scans contain exactly
  `points_per_scan` 2-D points, and GPS noise stays inside a 6-sigma
  (9.0 m) bound over 50 samples.
* `test_localization.py` - localization filters (module 04): the GPS
  low-pass processor converges onto a constant measurement, and
  complementary GPS + IMU fusion achieves a lower RMSE than the raw
  GPS track on a straight 5 m/s drive with seeded noise.
* `test_planning_visualization.py` - planning stack (module 05): the
  sample JSON road map loads with nodes/links/adjacency, A* returns a
  valid `n0 -> n7` route following adjacency edges, and the map-aware
  route visualizer produces a non-empty PNG (in `tmp_path` only).
* `test_scenario_validation.py` - decision layer (module 07): the
  driving-scenario dataset follows the documented structure, and the
  rule-based evaluator reproduces the `expected_decision` annotation of
  every scenario.

## How to run

From the project root:

```bash
# full suite
python -m pytest 10_testing_examples -v

# single file
python -m pytest 10_testing_examples/test_localization.py -v

# single test
python -m pytest 10_testing_examples/test_sensor_data.py::test_gps_noise_bounded -v
```

## Test data strategy

* **Fixed seeds**: every stochastic input is drawn from
  `np.random.default_rng(42)` (sensor noise, IMU/GPS noise in the
  fusion test), so failures are exactly reproducible.
* **Headless rendering**: every test file forces
  `matplotlib.use("Agg")` at the top, before any plotting import, so
  the suite runs on display-less machines and in CI.
* **Temporary artifacts**: the only file-producing test
  (`test_route_png_created`) writes into the pytest `tmp_path`
  fixture, which pytest cleans up automatically. Module `outputs/`
  folders are never written by tests.
* **Numeric tolerances**: assertions use closed-form bounds derived
  from the documented filter equations (e.g. the 6-sigma GPS bound of
  9.0 m for sigma 1.5 m) rather than magic numbers.

## Inputs and outputs

Reads (project-relative, resolved from the test file location, never
from the current working directory):

* `05_path_planning_demo/maps/simple_road_map.json` (road graph),
* `07_ai_decision_demo/sample_data/driving_scenarios.json` (scenario
  dataset),
* module source packages `01_basic_simulation`, `02_sensor_simulation`,
  `04_localization_demo`, `05_path_planning_demo`,
  `07_ai_decision_demo` (imported through `sys.path` inserts).

Writes: only `tmp_path` artifacts such as `route.png`; the suite prints
no files into the repository itself.

## Extension points

* **CI wiring**: add `python -m pytest 10_testing_examples -v` as a
  GitHub Actions step (with `pip install numpy matplotlib PyYAML pandas
  scipy pytest`) to gate every pull request on the full contract set.
* **Coverage measurement**: run `pytest --cov` with `pytest-cov`,
  targeting the module packages (e.g.
  `--cov=01_basic_simulation/vehicle --cov=04_localization_demo/localization`)
  and enforce a coverage floor in CI.
* **Property-based tests**: add hypothesis strategies for vehicle
  states, sensor configs and map graphs to fuzz the invariants tested
  here (clamping bounds, envelope keys, route connectivity).
* **Figure regression tests**: hash the generated route PNG against
  stored goldens to detect unintended rendering changes.
