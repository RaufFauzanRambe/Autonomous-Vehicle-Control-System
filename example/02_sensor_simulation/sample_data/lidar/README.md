# sample_data/lidar — Synthetic 16-channel lidar dataset

Three ASCII PCD scans (t = 0.0, 0.1, 0.2 s) from a simulated top-mounted
lidar, plus calibration, metadata and the ground-removed processed CSVs.
Each scan samples the same static scene the camera sees: ground plane,
lead vehicle (cluster 1), barrier (cluster 2), sign pole (cluster 3) and
two unclustered vegetation blobs (cluster 0).

## Files

| File | Content |
|---|---|
| `lidar_metadata.json` | Device, rate, mount pose, per-scan point counts, scene obstacle table |
| `lidar_calibration.json` | Mount position z = 1.8 m, zero extrinsic error, range accuracy |
| `point_clouds/frame_00000{1..3}.pcd` | ASCII PCD v0.7, `FIELDS x y z intensity`, ~3058 points per scan |
| `processed/filtered_points.csv` | Points after ground removal: `height_above_ground_m > 0.15` |
| `processed/obstacle_points.csv` | Clustered points (`cluster_id >= 1`), regardless of height |

## Coordinate frame

Sensor frame: x forward, y left, z up, origin at the sensor mount — the
ground plane therefore sits at `z = -1.8 m`. `height_above_ground_m`
adds the mount height back, so it is directly comparable with the world
frame height.

## Processing rule

`filtered_points.csv` = ground removal by height threshold (keeps
obstacle faces above 0.15 m and clutter). `obstacle_points.csv` =
cluster membership only, so it also retains low hits on box bottoms that
the height filter removes. Point counts therefore differ between the two
files by design.

## PCD format

```
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4      TYPE F F F F      COUNT 1 1 1 1
WIDTH <n>         HEIGHT 1          POINTS <n>
DATA ascii
```

## Extension points

Swap in real PCD recordings (or binary PCD with a matching parser) and
keep the `cluster_id` semantics; `tests/test_lidar_sensor.py` validates
header consistency, coordinate ranges and the processed-CSV contract.
