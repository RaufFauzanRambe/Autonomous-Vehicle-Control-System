"""Color-rule traffic sign classification on a cropped ROI.

Extension point: replace the channel-ratio rules with a CNN or template
matching classifier over a sign taxonomy, and add shape checks (octagon
for stop, circle for mandatory signs) before color classification.
"""

from __future__ import annotations

import numpy as np


class TrafficSignRecognizer:
    """Classify a cropped sign ROI by mean BGR channel dominance.

    The rule set encodes the two colors that matter for the demo scene:
    strong red dominance indicates a stop sign and strong blue dominance
    indicates an information sign. Otherwise the ROI is reported as
    ``unknown`` with a fixed low confidence.

    Extension point: train a compact CNN (e.g. on GTSRB) or run
    multi-scale template matching over the ROI and return per-class
    calibrated probabilities instead of ratio heuristics.
    """

    def __init__(self, config: dict) -> None:
        """Configure the recognizer.

        Args:
            config: ``traffic_sign`` section of the module YAML with keys
                ``red_ratio`` [-] and ``blue_ratio`` [-] (required channel
                dominance factors, typical 1.2-2.0).
        """
        self._red_ratio = float(config["red_ratio"])
        self._blue_ratio = float(config["blue_ratio"])

    def classify(self, roi: np.ndarray) -> dict[str, float | str]:
        """Classify one cropped sign region.

        Args:
            roi: Cropped BGR region, ``uint8`` array ``(h, w, 3)``; in the
                demo this is ``image[sign_bbox_px]`` from the scene meta.
                This is a ground-truth association stand-in: production
                derives the ROI from the object detector output.

        Returns:
            Dictionary with keys:

            * ``"sign_class"``: ``"stop_sign"``, ``"info_sign"`` or
              ``"unknown"`` [-].
            * ``"confidence"``: score in (0, 1] [-]; for ``stop_sign`` /
              ``info_sign`` the dominant-channel ratio capped at 0.99,
              for ``unknown`` the fixed value 0.3.
        """
        mean_b, mean_g, mean_r = roi.reshape(-1, 3).mean(axis=0)

        if mean_r > self._red_ratio * mean_g and mean_r > self._red_ratio * mean_b:
            confidence = min(0.99, float(mean_r) / max(float(mean_g), float(mean_b), 1.0))
            return {"sign_class": "stop_sign", "confidence": float(confidence)}

        if mean_b > self._blue_ratio * mean_g and mean_b > self._blue_ratio * mean_r:
            confidence = min(0.99, float(mean_b) / max(float(mean_g), float(mean_r), 1.0))
            return {"sign_class": "info_sign", "confidence": float(confidence)}

        return {"sign_class": "unknown", "confidence": 0.3}
