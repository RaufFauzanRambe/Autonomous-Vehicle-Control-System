# 08_full_system_demo

End-to-end autonomy pipeline demo for the Autonomous Vehicle Control
System example suite. It wires stub perception, noisy localization,
goal-based planning and a proportional surrogate controller around the
module `01_basic_simulation` kinematic bicycle model and runs the full
**sense -> localize -> plan -> act** loop at a fixed tick rate.

## Purpose

This module demonstrates how the individual example modules compose into
one closed loop, using deliberately simple stand-ins where the real
components live elsewhere in the suite:

1. **sense (perception)** — `PerceptionInterface` injects
   `n_static_actors` synthetic static vehicles at fixed ego-relative
   offsets (30 m ahead / +2 m lateral and 60 m ahead / -2 m lateral) and
   reports those within `max_range_m` as distance/bearing detections.
2. **localize** — `LocalizationInterface` returns a copy of the
   ground-truth pose with independent zero-mean Gaussian noise on x and
   y (GNSS surrogate), storing the most recent estimate.
3. **plan** — `PlanningInterface` computes the wrapped heading and the
   target speed toward the goal from the noisy estimate (target speed
   capped to 2.0 m/s inside 10 m of the goal) plus a 5 m lookahead
   point.
4. **act** — a proportional surrogate for the module 06 PID controller:
   `throttle = clip(0.5 * (target_speed - speed), -1, 1)` [-] and
   `steering = clip(steering_gain * wrap(target_heading - yaw), -0.6, 0.6)`
   [rad]; the module 01 kinematic bicycle model then advances the world
   by one tick.

The loop terminates when the ground-truth position is within 5.0 m of
the goal or when the configured duration has elapsed.

## Key concepts and physical parameters

| Parameter | Unit | Meaning | Typical range |
| --- | --- | --- | --- |
| `tick_rate_hz` | Hz | pipeline tick rate; model step `dt = 1/tick_rate_hz` | 5-50 |
| `duration_s` | s | maximum simulated time per scenario | 10-60 |
| `max_range_m` | m | perception detection range | 50-200 |
| `n_static_actors` | - | synthetic static vehicles injected per tick | 0-4 |
| `gps_noise_std_m` | m | standard deviation of GNSS-style position noise on x and y | 0.1-5.0 |
| `target_speed_mps` | m/s | planning cruise target (capped to `min(target, 2.0)` within 10 m of the goal) | 8-30 |
| `steering_gain` | - | heading-error to steering factor | 0.5-1.5 |
| throttle | - | `0.5 * speed error`, clipped to [-1, 1] | [-1, 1] |
| steering | rad | `steering_gain * wrapped heading error`, clipped to [-0.6, 0.6] | [-0.6, 0.6] |
| goal tolerance | m | euclidean distance within which the goal counts as reached | 5.0 |

Vehicle model (module 01 kinematic bicycle, Euler integration, inlined
default parameters): wheelbase 2.7 m, max steering 0.6 rad, max speed
33.0 m/s, max accel 3.0 m/s^2 (throttle +1), max decel 8.0 m/s^2
(throttle -1), drag coefficient 0.02, mass 1500 kg.

RNG handling: each scenario uses
`np.random.default_rng(seed + crc32(scenario_name) % 1000)`; `crc32` is
used instead of the built-in `hash()` because the latter is salted per
Python process, which would break run-to-run reproducibility.

## How to run

All commands are executed from the project root; every runtime path is
resolved against the module directory, never the current working
directory.

```bash
# headless run, processes all three scenarios
python 08_full_system_demo/main.py

# explicit overrides (all optional)
python 08_full_system_demo/main.py --config 08_full_system_demo/config/system_config.yaml --output-dir 08_full_system_demo/results --seed 42
```

`--show` is accepted for CLI symmetry with the other modules; module 08
produces no plots, so the flag has no effect.

## Inputs and outputs

Inputs (read by `main.py`):

- `config/system_config.yaml` — tick rate, duration, per-module
  parameters, logging level and the scenario file list.
- `scenarios/urban_environment.yaml` — urban scenario (goal 180 m,
  start speed 5 m/s).
- `scenarios/highway_environment.yaml` — highway scenario (goal 500 m,
  start speed 20 m/s).
- `scenarios/parking_environment.yaml` — parking scenario (goal 15 m,
  start speed 1 m/s, parallel-parking maneuver label).

Generated outputs:

- `logs/urban.log` — one INFO line per tick (t, x, y, speed,
  detections) plus start/finish markers.
- `logs/highway.log` — same format.
- `logs/parking.log` — same format.
- `results/urban_results.json` — per-scenario tick records and metrics.
- `results/highway_results.json` — per-scenario tick records and
  metrics.
- `results/parking_results.json` — per-scenario tick records and
  metrics.
- `results/run_summary.json` — `{"scenarios": [per-scenario summary
  metrics]}`.

## Metrics to observe

- Per scenario (`metrics` block in each results JSON):
  `duration_s` [s], `distance_m` [m], `mean_speed_mps` [m/s],
  `final_position_xy` [m], `detections_total` [-], `goal_reached` [-].
- Expected qualitative outcome: `urban` reaches the goal inside the 20 s
  budget (roughly 175 m of travel at the 12 m/s target);
  `highway` is duration-limited (500 m at a 12 m/s surrogate target
  needs about 42 s, so `goal_reached` is False and the run stops at
  `duration_s`); `parking` reaches its 15 m goal quickly (the 5.0 m
  goal tolerance plus the 2.0 m/s slow-down zone make it stop shortly
  after passing the 10 m mark).
- `detections_total` equals `2 * n_ticks` for all scenarios because both
  synthetic actors (30 m and 60 m) always sit inside the 120 m range.
- Tick logs let you verify the control behaviour: speed converges to
  `target_speed_mps`, and steering wobbles slightly because the heading
  target is computed from the noisy localization estimate while the
  controller acts on the true yaw (a deliberate, documented
  characteristic of the surrogate pipeline).

## Extension points

- Perception: replace the synthetic actor injection with sensor readings
  from module 02 fed through module 03 (same detection dictionary
  schema, add range/bearing noise from the rng).
- Localization: plug the module 04 SensorFusion filter in; the
  interface already stores `last_estimate` for downstream analysis.
- Planning/control: replace `PlanningInterface` with the module 05 A*
  path planner plus the module 06 tracking controller; the
  `compute_target` return schema (speed, heading, lookahead point) is
  designed to survive that upgrade, and the proportional surrogate in
  `main.py` marks exactly where the PID controller plugs in.
- Simulation: own a configuration-driven vehicle parameter set instead
  of the inlined defaults, add multi-vehicle world state, and model
  weather-dependent sensor degradation using the scenario `weather`
  labels.
