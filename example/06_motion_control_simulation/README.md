# Module 06: Motion Control Simulation (Speed PID + Heading Controller)

## Purpose

This module demonstrates closed-loop longitudinal and lateral motion
control of the kinematic bicycle vehicle model from module 01 in a
research-oriented, fully runnable form. The speed PID outputs a normalized
throttle in [-1, 1] that is consumed directly by the module 01 plant model
(positive = acceleration at `max_accel_mpss` [m/s^2], negative = braking at
`max_decel_mpss` [m/s^2], plus linear drag). The lateral controller is a
simplified pure-pursuit-style heading controller: it computes a desired
heading toward a reference point one lookahead distance ahead on the
scenario path and steers proportionally to the wrapped heading error,
clipped to the steering limit. Three scenarios (straight road, constant-
curvature road, low-speed parking maneuver) exercise the controllers, and
each run produces a three-panel response figure plus machine-readable
metrics in `run_summary.json`.

## Key concepts and physical parameters

- Kinematic bicycle plant (module 01): `wheelbase_m` = 2.7 [m],
  `max_steering_rad` = 0.6 [rad], `max_speed_mps` = 33.0 [m/s],
  `max_accel_mpss` = 3.0 [m/s^2], `max_decel_mpss` = 8.0 [m/s^2],
  `drag_coefficient` = 0.02 [1/s], `mass_kg` = 1500.0 [kg] (kept inline in
  `main.py` because module 01 owns the canonical parameter file).
- Speed PID gains: `kp` = 1.2 [-], `ki` = 0.15 [-], `kd` = 0.05 [-]
  (mapping the speed error [m/s] and its integral/derivative to a
  normalized throttle [-] in `output_limits` = [-1.0, 1.0]); anti-windup by
  conditional integration while the output is saturated.
- `steering_gain` = 0.8 [-]: heading error [rad] to steering angle [rad]
  proportional gain, typical 0.3-1.5.
- `lookahead_m` = 8.0 [m]: pure-pursuit-style lookahead along the
  reference polyline, typical 4-15 m.
- `dt_s` = 0.05 [s]: integration and control step, typical 0.01-0.1.
- `target_speed_mps` [m/s]: per-scenario speed setpoint (12.0 straight,
  10.0 curved, 1.5 parking).
- Reference geometry: `heading_rad` [rad] for straight paths,
  `curvature_1_per_m` [1/m] for constant-curvature arcs, `length_m` [m]
  path length, sampled at 1.0 m.
- `cross_track_error` [m]: perpendicular distance from the ego position to
  the nearest reference segment.
- Episode length: the simulated duration is clamped to the scenario
  `duration_s` [s] and additionally terminates as soon as the ego reaches
  the end of the reference polyline.

## How to run

From the project root (`autonomous_vehicle_control_system/`):

```bash
python 06_motion_control_simulation/main.py
python 06_motion_control_simulation/main.py --show
python 06_motion_control_simulation/main.py --config 06_motion_control_simulation/config/simulation_config.yaml --output-dir 06_motion_control_simulation/outputs --seed 42
```

- `--show` opens interactive Matplotlib windows in addition to saving the
  PNG files (requires a display); without it the headless `Agg` backend is
  used and only files are written.
- `--config` defaults to `config/simulation_config.yaml` inside the module
  directory; relative paths resolve against the module directory.
- `--output-dir` defaults to `outputs/` inside the module directory.
- `--seed` (default 42) is accepted for CLI uniformity across the example
  suite; the simulation is deterministic and the seed does not change
  results.

## Inputs and outputs

Inputs (all paths module-relative):

- `config/simulation_config.yaml` - PID gains, steering gain, lookahead,
  integration step and the scenario file list.
- `scenarios/straight_road.yaml` - 300 m straight path, accelerate from
  standstill to 12.0 m/s.
- `scenarios/curved_road.yaml` - 250 m constant-curvature path
  (0.01 1/m, radius 100 m), 8.0 m/s initial speed, 10.0 m/s setpoint.
- `scenarios/parking_scenario.yaml` - 20 m straight maneuver at 1.5 m/s.

Outputs (written to the directory given by `--output-dir`):

- `outputs/control_straight_road.png` - speed, steering and path panels
  for the straight-road scenario.
- `outputs/control_curved_road.png` - same panels for the curved-road
  scenario.
- `outputs/control_parking.png` - same panels for the parking scenario.
- `outputs/run_summary.json` - `{"scenarios": [...]}` with one metric
  record per scenario.

## Metrics to observe

Per scenario, in `run_summary.json` under `scenarios`:

- `duration_s` [s] - simulated duration (clamped by the scenario duration
  and ended early when the reference path is fully traversed).
- `distance_m` [m] - trapezoidal time integral of the tracked speed.
- `speed_rmse_mps` [m/s] - root-mean-square error between tracked speed
  and `target_speed_mps`, evaluated after the speed first enters a
  +/-10 % band around the setpoint (the unavoidable cold-start ramp at
  3.0 m/s^2 is excluded so the metric reflects steady tracking quality).
- `max_cross_track_error_m` [m] - largest perpendicular distance to the
  reference path; should stay below roughly 1 m after the brief initial
  transient.
- `final_cross_track_error_m` [m] - cross-track error at the last sample;
  near zero indicates the maneuver terminates on the path.

## Extension points

- Geometric pure pursuit: replace the proportional heading law with
  `steering = atan(2 * wheelbase_m * sin(alpha) / lookahead_m)` [rad] and
  compare `max_cross_track_error_m` [m] on the curved scenario.
- Lateral controllers: Stanley, LQR on a linear bicycle error model, or
  MPC with curvature preview; benchmark against `speed_rmse_mps` [m/s]
  and `max_cross_track_error_m` [m].
- Longitudinal upgrades: feed-forward of the drag map
  (`drag_coefficient * speed` [m/s^2]), cascade position -> speed PID, or
  gain scheduling on speed.
- Robustness studies: sweep PID gains, lookahead distances [m] and
  initial speeds [m/s] in Monte Carlo runs and aggregate the metric
  distributions from `run_summary.json`.
- Planner-in-the-loop: feed waypoint sequences from module 05
  (`planned_route_*.png` routes) as reference paths instead of the
  analytic scenario geometries.
