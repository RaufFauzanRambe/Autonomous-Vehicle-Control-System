# 01_basic_simulation

## Purpose

This module demonstrates the smallest runnable closed piece of an autonomous
vehicle stack: an open-loop kinematic bicycle plant simulated on a straight
test track. A researcher can use it as a baseline to sanity-check the vehicle
model, the road geometry queries (on-road test, speed limits) and the
plotting/summary conventions shared by all other modules in the suite.

## Key concepts and physical parameters

- Kinematic bicycle model with front-wheel steering and Euler integration
  (no tire slip, flat road).
- Vehicle parameters (`vehicle/vehicle_parameters.yaml`):

  | Parameter            | Value  | Unit    | Meaning                                   |
  |----------------------|--------|---------|-------------------------------------------|
  | `wheelbase_m`        | 2.7    | m       | distance between front and rear axle      |
  | `max_steering_rad`   | 0.6    | rad     | maximum front wheel angle, + = left       |
  | `max_speed_mps`      | 33.0   | m/s     | top vehicle speed                         |
  | `max_accel_mpss`     | 3.0    | m/s^2   | peak acceleration at throttle = 1         |
  | `max_decel_mpss`     | 8.0    | m/s^2   | peak deceleration at throttle = -1        |
  | `drag_coefficient`   | 0.02   | 1/s     | linear speed drag gain                    |
  | `mass_kg`            | 1500.0 | kg      | curb mass (unused by kinematics)          |

- Vehicle state (`vehicle/vehicle_state.py`): `timestamp` [s], `x`, `y` [m],
  `yaw` [rad] (CCW positive, 0 rad = +x axis), `speed` [m/s],
  `acceleration` [m/s^2], `steering_angle` [rad].
- Road map (`environment/road_map.json`): straight segments with
  `width_m` [m] and `speed_limit_mps` [m/s]; a point is on the road when its
  distance to the nearest centerline is at most half the segment width.
- Controls: `throttle` [-] in -1..1 (negative = braking) and
  `steering_angle` [rad], both constant (open loop).

## How to run

All commands work from the project root; paths are resolved relative to the
module directory, never the current working directory.

```bash
# Headless run (default): saves figure + run_summary.json into 01_basic_simulation/outputs/
python 01_basic_simulation/main.py

# Open an interactive matplotlib window instead of headless-only rendering
python 01_basic_simulation/main.py --show

# Custom config, output directory and random seed
python 01_basic_simulation/main.py --config 01_basic_simulation/simulation_config.yaml --output-dir /tmp/sim01_out --seed 7
```

## Inputs and outputs

Input files (all read from the module directory by default):

- `simulation_config.yaml` - simulation duration, step size, initial state,
  constant controls, logging stride (overridable via `--config`).
- `vehicle/vehicle_parameters.yaml` - the seven vehicle parameters listed above.
- `environment/road_map.json` - straight test track segments.

Generated output files (written into `outputs/` or `--output-dir`):

- `outputs/basic_simulation_trajectory.png` - two-panel figure (trajectory
  colored by speed over the gray road band; speed vs time with the speed
  limit as a horizontal dashed line), dpi 150.
- `outputs/run_summary.json` - key metrics of the run.

## Metrics to observe

- `distance_m`: total path length traveled [m]; cross-check against
  `mean_speed_mps * duration_s` to verify drag/clip behavior.
- `final_speed_mps` [m/s]: steady-state speed reached under constant throttle
  (thrust balanced by the linear drag term).
- `mean_speed_mps` [m/s]: average over all recorded states.
- `off_road_steps` [-]: number of integration steps whose position failed
  `RoadEnvironment.is_on_road`; 0 for the default straight-line run, nonzero
  for steering experiments (which indicates the vehicle left the 7 m band).

## Extension points

- Close the loop: add a PID longitudinal controller tracking
  `env.get_speed_limit(x, y)` and a pure-pursuit lateral controller tracking
  the nearest segment centerline, replacing the constant open-loop controls.
- Upgrade the plant: RK4 integration, dynamic bicycle model with tire
  cornering stiffness, first-order actuator lag on throttle and steering.
- Enrich the map: polyline/graph road networks in `road_map.json` with a
  spatial index in `RoadEnvironment` for large maps.
- Stochastic runs: inject process noise through the already-threaded
  `np.random.default_rng(seed)` generator for Monte-Carlo studies.
