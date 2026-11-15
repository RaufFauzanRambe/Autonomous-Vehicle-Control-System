# 07_ai_decision_demo

Rule-based driving decision benchmark for the Autonomous Vehicle Control
System example suite. A lightweight decision layer inspects a single
snapshot of the driving environment (ego state, actors, road features)
and commits to exactly one action from the vocabulary
`{stop, follow, cruise}`.

## Purpose

This module demonstrates the smallest research-grade decision component
that is still physically meaningful:

- It consumes an `EnvironmentState` built from a JSON scenario catalog;
  the ego pose reuses the `VehicleState` dataclass of module
  `01_basic_simulation`, so both modules share one physics vocabulary.
- A `ScenarioEvaluator` applies four ordered rules with pinned precedence
  (first hit wins):
  1. traffic light: red light within the stop distance -> `stop`
  2. pedestrian: occupied crossing or pedestrian actor within the stop
     distance -> `stop`
  3. follow: slower lead vehicle within the desired time headway ->
     `follow`
  4. cruise: unconditional fallback -> `cruise`

  Rule precedence is therefore: **traffic light > pedestrian > follow >
  cruise**, and the action vocabulary is **{stop, follow, cruise}**.
- Every decision carries a human-readable reason string, a target speed
  in m/s and a confidence value in [0, 1], so runs can be audited and
  scored automatically against the `expected_decision` labels.

## Key concepts and physical parameters

| Parameter | Unit | Meaning | Typical range |
| --- | --- | --- | --- |
| `stop_distance_m` | m | activation distance for both stop rules | 10-40 |
| `follow_time_s` | s | desired time headway; follow threshold = `ego.speed * follow_time_s` [m] | 1.5-3.0 |
| `brake_deceleration_mpss` | m/s^2 | comfortable deceleration quoted in stop-reason audit strings | 2.0-4.0 |
| `cruise_speed_mps` | m/s | fallback cruise target speed | 10-20 |
| `ttc_stop_s` | s | time-to-collision advisory threshold | 3-6 |
| `confidence` | - | rule confidence in [0, 1]: 0.95 traffic light, 0.9 pedestrian, 0.85 follow, 0.8 cruise | 0.8-0.95 |

Time-to-collision is computed as `ttc = distance / (ego.speed - lead.speed)`
in seconds and is `None` when the closing speed is non-positive (the ego
is not catching up). Actor and road-feature positions are metres in the
local planning frame; headings are radians counter-clockwise from +x.
The follow rule additionally requires the lead to be slower than
`0.9 * ego.speed` [m/s] to avoid speed chatter behind barely-slower
vehicles.

## How to run

All commands are executed from the project root; every runtime path is
resolved against the module directory, never the current working
directory.

```bash
# headless run (Agg backend), writes 4 decision PNGs and run_summary.json
python 07_ai_decision_demo/main.py

# interactive run, opens one figure window per scenario after saving
python 07_ai_decision_demo/main.py --show

# explicit overrides (all optional)
python 07_ai_decision_demo/main.py --config 07_ai_decision_demo/config/decision_config.yaml --output-dir 07_ai_decision_demo/outputs --seed 42
```

`--seed` is accepted for CLI symmetry with the other modules; the rule
evaluation is fully deterministic and consumes no randomness.

## Inputs and outputs

Inputs (read by `main.py`):

- `config/decision_config.yaml` — rule thresholds, TTC advisory
  threshold and the `require_reason_strings` evaluation flag.
- `sample_data/driving_scenarios.json` — four benchmark scenarios with
  `expected_decision` labels: `free_flow_cruise`, `red_light_stop`,
  `pedestrian_crossing`, `slow_lead_vehicle`.

Generated outputs (default `outputs/`):

- `outputs/decision_free_flow_cruise.png` — top-down schematic.
- `outputs/decision_red_light_stop.png` — top-down schematic.
- `outputs/decision_pedestrian_crossing.png` — top-down schematic.
- `outputs/decision_slow_lead_vehicle.png` — top-down schematic.
- `outputs/run_summary.json` — `{"decisions": [...], "n_matches": int,
  "n_scenarios": int}` with one record per scenario (id, action, reason,
  target_speed_mps, confidence, expected_decision, matches_expected).

## Metrics to observe

- `n_matches` / `n_scenarios`: the benchmark passes with `4/4`; a value
  below 4 means the rule precedence or thresholds drifted from the
  pinned configuration.
- `confidence` per decision: `stop` 0.95 (traffic light) or 0.9
  (pedestrian), `follow` 0.85, `cruise` 0.8.
- `target_speed_mps`: 0.0 for both stop rules, the lead vehicle speed
  (6.0 m/s) for `follow`, `cruise_speed_mps` (14.0 m/s) for `cruise`.
- Reason strings: every decision must carry a non-empty audit trail
  (`evaluation.require_reason_strings: true`); missing reasons raise a
  `ValueError`.
- Scenario sanity anchors: `pedestrian_crossing` stops first because the
  pedestrian rule outranks follow even though the follow threshold
  (8 m/s * 2.0 s = 16 m) also covers the pedestrian at 10 m;
  `slow_lead_vehicle` follows because 20 m <= 12 m/s * 2.0 s = 24 m and
  6.0 m/s < 0.9 * 12.0 m/s.

## Extension points

- Replace the ordered rule list with utility-based scoring: each rule
  returns a score and the argmax wins, with the pinned precedence as the
  tie-breaker.
- Train a learned policy (small MLP or gradient-boosted classifier) on
  logged `EnvironmentState` -> `Decision` pairs and keep this rule
  engine as a fallback guardrail.
- Enforce `thresholds.ttc_stop_s` as a hard-brake override ahead of the
  soft rules once scenarios with genuinely dangerous closing rates are
  added; in the current benchmark the TTC value is quoted in the follow
  reason string for advisory only (forcing a stop at ttc < 4.0 s would
  flip the pinned `slow_lead_vehicle` expectation to stop, since its
  ttc is 20 m / 6 m/s = 3.3 s).
- Feed predicted actor trajectories instead of the static snapshot so
  rules can reason about future conflict zones rather than the current
  positions.
