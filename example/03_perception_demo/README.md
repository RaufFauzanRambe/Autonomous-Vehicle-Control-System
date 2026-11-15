# 03 Perception Demo

Camera-space perception on a single frame: classical preprocessing, lane
detection, contour-based object detection and color-rule traffic sign
classification.

## Purpose

This module demonstrates the perception stage of the stack on one image.
Because `sample_data/` ships no images (its `.gitkeep` marks where real
frames would go), the demo **generates a synthetic camera frame** with a
deterministic OpenCV renderer (`perception/image_processor.py`,
`synthetic_road_image`). The frame shows a road trapezoid with lane
lines, two parked vehicles and a red stop-sign stand-in, and comes with
ground-truth bounding boxes in `meta`. The pipeline then runs exactly as
it would on a real frame: gray -> Gaussian blur -> Canny edges, Hough
lane segments with slope-based left/right assignment, Otsu contour
object detection, and BGR channel-ratio sign classification. The goal is
a readable, runnable reference whose every stage can be swapped for a
research-grade equivalent.

## Key concepts and physical parameters

- Gaussian smoothing: kernel `gaussian_kernel_px` [px], odd size; larger kernels remove more noise but blur thin structures (typical 3-9).
- Canny hysteresis: `canny_low` / `canny_high` [-] on 0-255 gradient magnitude; edges survive when above `canny_high` or between the thresholds and connected to a strong edge.
- Trapezoidal ROI: `roi_top_ratio` [-] is the fraction of frame height where the region of interest starts (sky and dashboard rejection).
- Probabilistic Hough transform: `hough_threshold` [-] votes, `min_line_length_px` [px], `max_line_gap_px` [px]; segments are split by slope sign in image coordinates (left lane: slope < 0, right lane: slope > 0, y axis pointing down) and `|slope| < 0.3` is rejected as near-horizontal clutter.
- Lane parameters: `(slope, intercept)` per side in image coordinates [1, px], from `y = slope * x + intercept` fitted to Hough segment endpoints.
- Otsu thresholding: automatic global binarization [-] separating dark objects from the bright ground plane in the synthetic scene.
- Object confidence: fill ratio `contour_area / (2 * bbox_area)` [-]; a perfectly filled rectangle scores 0.5, sparse shapes score lower.
- Color channel dominance: `red_ratio` / `blue_ratio` [-] compare mean R or B intensity against the other channels of the ROI (mean intensities in [0, 255]).

## How to run

Run from the repository root:

```bash
python 03_perception_demo/main.py
```

Interactive preview windows (figures are still saved):

```bash
python 03_perception_demo/main.py --show
```

Custom configuration, output directory and seed:

```bash
python 03_perception_demo/main.py --config 03_perception_demo/config/perception_config.yaml \
    --output-dir 03_perception_demo/outputs --seed 42 --show
```

All file paths are resolved against the module directory, so the
commands work from any working directory as long as the script path is
given.

## Inputs and outputs

Inputs:

- `config/perception_config.yaml` — all thresholds and frame geometry.
- No image files: the frame is generated in code because `sample_data/` contains no images (only the `.gitkeep` placeholder where real frames would go).

Outputs (in `outputs/` unless `--output-dir` is given):

- `synthetic_input.png` — the generated camera frame (BGR as rendered).
- `detection_overlay.png` — frame with green rectangles and labels for each object detection and a blue box plus class label for the sign ROI.
- `lane_overlay.png` — frame with the averaged left lane line in green and right lane line in red, plus the number of Hough segments kept.
- `preprocessing_stages.png` — 2x2 matplotlib panel: original RGB, grayscale, Gaussian blur, Canny edges.
- `run_summary.json` — `image_size_px`, `n_detections`, `left_lane_found`, `right_lane_found`, `sign_class`, `sign_confidence`.

## Metrics to observe

- `n_detections` — with default thresholds exactly 2 (the two filled vehicle rectangles; the circle sign stand-in scores below the fill-ratio threshold, which is why the demo classifies the sign from its annotated ROI instead of detector output).
- `left_lane_found` / `right_lane_found` — both should be `True` for the default scene; set `roi_top_ratio` above 0.6 or shrink `hough_threshold` to 200 to see failures.
- `sign_class` / `sign_confidence` — expect `stop_sign` with confidence 0.99 (red channel dominance of the circle stand-in).
- Raising `canny_low`/`canny_high` above 300 removes lane edges; raising `min_area_px` above 10000 removes the vehicles — useful sensitivity exercises.

## Extension points

- Real input: replace `synthetic_road_image` with a camera driver, ROS bag or dataset (KITTI/Cityscapes) reader; keep the `meta` annotation contract.
- Lane detection: sliding-window polynomial fits in bird's-eye view, RANSAC, or learned lane models; temporal smoothing of lane parameters across frames.
- Object detection: swap `ObjectDetector` for YOLO/DETR while keeping the `ObjectDetection` dataclass interface; add class labels and calibrated confidences.
- Sign recognition: replace channel-ratio rules with a CNN classifier (e.g. trained on GTSRB) or template matching with shape verification.
- Camera model: add intrinsics plus lens undistortion and inverse-perspective mapping before lane detection.
