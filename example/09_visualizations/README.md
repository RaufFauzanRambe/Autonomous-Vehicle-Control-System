# 09_visualization

## Purpose

Reusable, dependency-light plotting helpers for the example suite. This
module collects every figure-level concern in one place so the
simulation, sensor, localization, planning and decision modules can stay
focused on algorithms:

* `vehicle_visualizer.py` - `VehicleVisualizer.plot_trajectory`: renders
  2-D vehicle trajectories (position polyline, start/end markers,
  heading arrows) from `VehicleState` objects or plain dictionaries.
* `sensor_visualizer.py` - `SensorVisualizer` with three static plotters
  for lidar point clouds, noisy GPS tracks and radar range histories.
* `route_visualizer.py` - `RouteVisualizer.plot_route`: renders bare
  waypoint polylines without any road-map context.
* `dashboard.py` - `VisualizationDashboard`: scans the output folders of
  modules 01-08 and renders a single summary PNG of run artifacts.

Every script runs headless by default (matplotlib `Agg` backend), saves
its figures as PNG files and prints the saved paths.

## Key concepts and physical parameters

* **Coordinates**: flat 2-D world frame, x forward [m], y left [m].
* **Heading**: yaw in [rad], counter-clockwise positive, 0 rad = +x
  axis; heading arrows are drawn as unit vectors along the yaw direction.
* **Units**: all axis labels carry SI units, `[m]` for positions and
  ranges, `[s]` for time.
* **Figure conventions**: PNG output at `dpi=150`, figures are closed
  after saving, and every saved path is printed. Marker conventions:
  green circle = trajectory start, red square = trajectory end, magenta
  star = route goal, black dashed line = ground truth, gray dots = noisy
  measurements.
* **Headless rendering**: unless `--show` is passed, the `Agg` backend
  is selected before `matplotlib.pyplot` is imported, so all scripts run
  on display-less machines and inside CI.

## How to run

All commands are executed from the project root:

```bash
# plot the shared sample trajectory -> 09_visualization/plots/sample_trajectory.png
python 09_visualization/vehicle_visualizer.py

# generate seeded synthetic lidar / GPS / radar figures
# -> plots/sensor_lidar_demo.png, plots/sensor_gps_demo.png, plots/sensor_radar_demo.png
python 09_visualization/sensor_visualizer.py

# plot a synthetic S-curve waypoint route -> 09_visualization/plots/route_demo.png
python 09_visualization/route_visualizer.py

# scan module outputs and render the cross-module dashboard
# -> 09_visualization/plots/dashboard.png
python 09_visualization/dashboard.py
```

Common flags: `--show` opens interactive windows (skipped under Agg),
`--output-dir DIR` redirects saved figures, `--seed N` controls the
synthetic sensor data (`sensor_visualizer.py` only), and the dashboard
accepts `--outputs-root DIR` to scan a different checkout.

## Inputs and outputs

| Script | Reads | Writes |
| --- | --- | --- |
| `vehicle_visualizer.py` | `shared/sample_data/sample_vehicle_data.csv` (via `shared/utilities/data_loader.load_csv_records`) and, per state, `01_basic_simulation/vehicle/vehicle_state.py` | `09_visualization/plots/sample_trajectory.png` |
| `sensor_visualizer.py` | nothing (generates seeded synthetic data with `np.random.default_rng`) | `09_visualization/plots/sensor_lidar_demo.png`, `sensor_gps_demo.png`, `sensor_radar_demo.png` |
| `route_visualizer.py` | nothing (deterministic synthetic S-curve waypoints) | `09_visualization/plots/route_demo.png` |
| `dashboard.py` | `outputs/` and `results/` folders of modules 01-08 (PNG counts, `run_summary.json` / `*_results.json` mtimes) | `09_visualization/plots/dashboard.png` |

All plots land in `09_visualization/plots/` unless `--output-dir`
overrides the destination. `dashboard.py` is read-only with respect to
the other modules and renders a valid summary (zero-count bars and
"no results yet" entries) even on a fresh checkout where no module has
ever produced outputs.

## Notes

This module intentionally has no "metrics to observe" section: it
produces figures, not numeric metrics. Numeric run summaries (final
speeds, RMSE values, decision logs) are written by modules 01-08 into
their own `outputs/` folders; this module only aggregates their artifact
presence into `dashboard.png` and visualizes trajectories, sensor data
and routes on demand.

## Extension points

* Interactive dashboards: replace the static dashboard PNG with a
  Plotly/Dash web application reading the same `collect()` dictionary.
* ROS bridge: adapt the plotters to publish
  `visualization_msgs/MarkerArray` messages for an rviz bridge instead
  of (or in addition to) saving PNG files.
* Live telemetry: wrap the trajectory/sensor plotters in a matplotlib
  `FuncAnimation` for real-time updating windows.
* Vector output: add a `--format pdf|svg` option for publication-quality
  figures; `savefig` calls already accept any matplotlib format.
