# Module 05: Path Planning Demo (A* Route Planning on Road Graphs)

## Purpose

This module demonstrates global route planning for autonomous vehicles in a
research-oriented, fully runnable form. Road networks are modelled as small
node/link graphs (JSON map documents), and a textbook A* graph search plans
the cost-optimal route from a start node to a goal node. The planned node
route is resampled into evenly spaced waypoints that a downstream motion
controller (see module 06) could follow, and one figure per map visualises
the network, the route and the start/goal markers. All metrics of a run are
collected in a machine-readable `run_summary.json` for reproducible
experiment tracking.

## Key concepts and physical parameters

- Road graph nodes: planar world-frame coordinates `x`, `y` in [m].
- Road links: geometric length `length_m` [m] (recomputed from node
  coordinates when absent), legal speed limit `speed_limit_mps` [m/s],
  lane count `lanes` [-].
- Search cost: accumulated link length [m]; the heuristic is the Euclidean
  straight-line distance [m] between node coordinates, which is admissible
  (it never overestimates the remaining road distance), so A* returns the
  optimal route for the given edge costs.
- `waypoint_spacing_m` [m]: resampling interval of the route polyline for
  downstream controllers, typical 2-10 m.
- `straightness_ratio` [-]: Euclidean start-goal distance divided by the
  planned path length; 1.0 means a perfectly straight route, typical road
  routes fall between 0.5 and 1.0.
- `nodes_expanded` [-]: number of nodes dequeued and expanded by A*; a
  compact measure of search effort.
- `vehicle.max_speed_mps` [m/s] (15.0 in the shipped config) is reserved
  for travel-time cost extensions; the shipped cost function is pure
  distance.
- `node_label_size` [pt]: font size of the node identifier labels in the
  figures.

## How to run

From the project root (`autonomous_vehicle_control_system/`):

```bash
python 05_path_planning_demo/main.py
python 05_path_planning_demo/main.py --show
python 05_path_planning_demo/main.py --config 05_path_planning_demo/config/planning_config.yaml --output-dir 05_path_planning_demo/outputs --seed 42
```

- `--show` opens interactive Matplotlib windows in addition to saving the
  PNG files (requires a display); without it the headless `Agg` backend is
  used and only files are written.
- `--config` defaults to `config/planning_config.yaml` inside the module
  directory; relative paths resolve against the module directory.
- `--output-dir` defaults to `outputs/` inside the module directory.
- `--seed` (default 42) is accepted for CLI uniformity across the example
  suite; the planner is deterministic and the seed does not change results.

## Inputs and outputs

Inputs (all paths module-relative):

- `config/planning_config.yaml` - planner selection, the list of maps with
  start/goal nodes, waypoint spacing and visualisation options.
- `maps/simple_road_map.json` - 8-node avenue map with a gentle dip
  (`n0` ... `n7`).
- `maps/intersection_map.json` - 5-node four-way intersection map
  (`n_center`, `n_north`, `n_south`, `n_east`, `n_west`).

Outputs (written to the directory given by `--output-dir`):

- `outputs/planned_route_simple_road_map.png` - map, route, start/goal
  markers for the avenue map.
- `outputs/planned_route_intersection_map.png` - map, route, start/goal
  markers for the intersection map.
- `outputs/run_summary.json` - `{"plans": [...]}` with one metric record
  per map.

## Metrics to observe

Per map, in `run_summary.json` under `plans`:

- `n_nodes_in_map` [-] and `n_nodes_in_path` [-] - graph size versus route
  size (start and goal included).
- `path_length_m` [m] - sum of Euclidean segment lengths along the route.
- `n_waypoints` [-] - number of resampled waypoints at
  `waypoint_spacing_m` [m].
- `straightness_ratio` [-] - 1.0 for a straight route; the avenue map
  (dip at `n3`/`n4`) yields a value below 1.0, the intersection route
  (90-degree turn) about 0.71.
- `nodes_expanded` [-] - search effort; compare the two maps to see how
  the admissible Euclidean heuristic prunes the search.
- `unreachable` [-] - `true` signals an empty route (goal not connected);
  the pipeline records zero-length metrics instead of failing.

## Extension points

- Weighted A*: inflate the Euclidean heuristic by `epsilon > 1` for
  bounded-suboptimal but faster searches; benchmark with
  `nodes_expanded` [-] and `path_length_m` [m].
- Travel-time costs: replace accumulated link length [m] by link length
  divided by `speed_limit_mps` [m/s] to minimise travel time [s] instead
  of distance.
- Kinematically feasible planning: hybrid A* or RRT* in a continuous
  state space with steering primitives [rad] and curvature limits
  [1/m], replacing the graph search while keeping the same planner
  interface.
- Lane-level routing: extend the JSON schema with per-lane edges and turn
  manoeuvres (costs) at intersections, or import SUMO/OpenDRIVE networks.
- Waypoint smoothing: resample routes with clothoid or spline fits
  (`scipy.interpolate`) to provide continuous curvature references for
  module 06 controllers.
