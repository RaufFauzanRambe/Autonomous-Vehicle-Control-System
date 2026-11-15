"""Perception demo: preprocessing, lane detection, object detection, sign classification.

The demo generates a deterministic synthetic camera frame (because
``sample_data/`` ships no images), runs the classical perception
pipeline on it and writes overlay figures plus a JSON run summary. All
paths are resolved relative to this module directory, never to the
current working directory.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[0]  # project root, holds shared/ utilities
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(ROOT))  # shared utilities: from shared.utilities.config_loader import load_yaml

import cv2  # noqa: E402

from perception.image_processor import ImageProcessor  # noqa: E402
from perception.lane_detection import LaneDetector  # noqa: E402
from perception.object_detection import ObjectDetection, ObjectDetector  # noqa: E402
from perception.traffic_sign_recognition import TrafficSignRecognizer  # noqa: E402
from shared.utilities.config_loader import load_yaml  # noqa: E402
from shared.utilities.data_loader import save_json  # noqa: E402

CONFIG_PATH = MODULE_DIR / "config" / "perception_config.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Parser with ``--config``, ``--show``, ``--output-dir`` and
        ``--seed`` options.
    """
    parser = argparse.ArgumentParser(
        description="Run the perception demo pipeline on a synthetic road scene."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to the module YAML configuration file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open matplotlib windows in addition to saving files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for PNG figures and run_summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to the scene generator (rendering itself is deterministic).",
    )
    return parser


def _draw_detection_overlay(
    image: np.ndarray,
    detections: list[ObjectDetection],
    sign_bbox_px: list[int],
    sign_class: str,
    sign_confidence: float,
) -> np.ndarray:
    """Render detection rectangles and labels onto a copy of the frame.

    Args:
        image: BGR frame ``(height, width, 3)`` [0-255].
        detections: Detections from :class:`ObjectDetector`.
        sign_bbox_px: Ground-truth sign ROI ``[x1, y1, x2, y2]`` [px].
        sign_class: Recognized sign class [-].
        sign_confidence: Recognized sign confidence [-].

    Returns:
        BGR overlay frame with green object boxes, a blue sign box and
        text labels.
    """
    overlay = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_px
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    sx1, sy1, sx2, sy2 = sign_bbox_px
    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
    cv2.putText(
        overlay,
        f"{sign_class} {sign_confidence:.2f}",
        (sx1, max(12, sy2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _draw_lane_overlay(
    image: np.ndarray,
    lanes: dict,
    roi_top_px: int,
) -> np.ndarray:
    """Render the averaged lane lines onto a copy of the frame.

    Args:
        image: BGR frame ``(height, width, 3)`` [0-255].
        lanes: Result of :meth:`LaneDetector.detect`.
        roi_top_px: y coordinate [px] where the lane ROI starts; lines
            are drawn from here to the image bottom.

    Returns:
        BGR overlay frame with the left line in green and the right
        line in red (when found), plus the segment count.
    """
    overlay = image.copy()
    height = image.shape[0]
    colors = {"left_line": ((0, 255, 0), "left"), "right_line": ((0, 0, 255), "right")}
    for key, (color, label) in colors.items():
        line = lanes[key]
        if line is None:
            continue
        slope, intercept = line
        x_bottom = (height - 1 - intercept) / slope
        x_top = (roi_top_px - intercept) / slope
        cv2.line(
            overlay,
            (int(round(x_bottom)), height - 1),
            (int(round(x_top)), roi_top_px),
            color,
            3,
        )
        cv2.putText(
            overlay,
            label,
            (int(round(x_top)) + 8, roi_top_px + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"Hough segments kept: {lanes['n_segments']}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _save_preprocessing_figure(
    image: np.ndarray,
    stages: dict[str, np.ndarray],
    output_path: Path,
    show: bool,
) -> None:
    """Write the 2x2 preprocessing panel figure (original, gray, blur, edges).

    Args:
        image: Original BGR frame ``(height, width, 3)``.
        stages: Preprocessing outputs with keys ``gray``, ``blurred``,
            ``edges`` (each ``(height, width)`` uint8).
        output_path: Destination PNG path.
        show: When True, open an interactive window after saving.
    """
    import matplotlib.pyplot as plt  # imported after the backend decision in main()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original (RGB)", None),
        (axes[0, 1], stages["gray"], "Grayscale", "gray"),
        (axes[1, 0], stages["blurred"], "Gaussian blur", "gray"),
        (axes[1, 1], stages["edges"], "Canny edges", "gray"),
    ]
    for ax, data, title, cmap in panels:
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Preprocessing stages")
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the full perception demo pipeline.

    Flow: parse arguments -> load config -> generate the synthetic road
    frame -> preprocess -> detect lanes -> detect objects -> classify
    the sign ROI -> write overlays, the preprocessing figure and
    run_summary.json -> print a console summary.

    Args:
        argv: Optional command-line arguments (defaults to ``sys.argv``).

    Returns:
        Process exit code, 0 on success.
    """
    args = build_parser().parse_args(argv)
    if not args.show:
        # Headless backend must be selected before pyplot is imported.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (backend decision precedes this import)

    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    width = int(config["camera"]["width_px"])
    height = int(config["camera"]["height_px"])

    # 1. Deterministic synthetic camera frame (no images in sample_data/).
    image, meta = ImageProcessor.synthetic_road_image(width, height, seed=args.seed)

    # 2. Preprocessing chain.
    proc = ImageProcessor(config["preprocessing"]).preprocess(image)

    # 3. Lane detection on the shared edge map.
    lanes = LaneDetector(config["lane_detection"]).detect(image, edges=proc["edges"])

    # 4. Object detection on the raw frame.
    detections = ObjectDetector(config["object_detection"]).detect(image)

    # 5. Traffic sign classification. The ROI comes from the scene meta
    #    (ground-truth association stand-in); a production system derives
    #    the ROI from the object detector output.
    sx1, sy1, sx2, sy2 = meta["sign_bbox_px"]
    sign = TrafficSignRecognizer(config["traffic_sign"]).classify(image[sy1:sy2, sx1:sx2])

    # 6. Outputs.
    if not cv2.imwrite(str(output_dir / "synthetic_input.png"), image):
        raise RuntimeError("Failed to write synthetic_input.png")
    detection_overlay = _draw_detection_overlay(
        image, detections, meta["sign_bbox_px"], sign["sign_class"], float(sign["confidence"])
    )
    if not cv2.imwrite(str(output_dir / "detection_overlay.png"), detection_overlay):
        raise RuntimeError("Failed to write detection_overlay.png")
    lane_overlay = _draw_lane_overlay(image, lanes, int(float(config["lane_detection"]["roi_top_ratio"]) * height))
    if not cv2.imwrite(str(output_dir / "lane_overlay.png"), lane_overlay):
        raise RuntimeError("Failed to write lane_overlay.png")
    _save_preprocessing_figure(image, proc, output_dir / "preprocessing_stages.png", show=args.show)

    run_summary = {
        "image_size_px": [width, height],
        "n_detections": len(detections),
        "left_lane_found": lanes["left_line"] is not None,
        "right_lane_found": lanes["right_line"] is not None,
        "sign_class": str(sign["sign_class"]),
        "sign_confidence": float(sign["confidence"]),
    }
    summary_path = save_json(run_summary, output_dir / "run_summary.json")

    # 7. Console summary.
    print("Perception demo summary")
    print(f"  image size: {width} x {height} px")
    print(f"  detections: {len(detections)}")
    for det in detections:
        print(f"    - {det.label} bbox={det.bbox_px} confidence={det.confidence:.3f}")
    for key in ("left_line", "right_line"):
        line = lanes[key]
        name = "left" if key == "left_line" else "right"
        if line is None:
            print(f"  {name} lane: not found")
        else:
            print(f"  {name} lane: slope={line[0]:+.3f} intercept={line[1]:+.1f} px")
    print(f"  Hough segments kept: {lanes['n_segments']}")
    print(f"  traffic sign: {sign['sign_class']} (confidence {sign['confidence']:.2f})")
    print(f"  run summary written to: {summary_path}")
    print(f"  outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
