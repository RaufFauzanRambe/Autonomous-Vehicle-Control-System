"""Contour-based object detection stand-in for a neural network detector.

Extension point: swap this classical detector for a learned detector
(YOLO, DETR) and carry class labels and calibrated confidences through
:class:`ObjectDetection` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ObjectDetection:
    """One detected object in a frame.

    Attributes:
        label: Semantic class name [-]; this stand-in always uses
            ``"object"`` because contours carry no semantics.
        bbox_px: Axis-aligned bounding box ``(x1, y1, x2, y2)`` [px] in
            image coordinates (y grows downward).
        confidence: Detection score in [0, 1] [-], derived here from
            contour fill ratio instead of a classifier output.
    """

    label: str
    bbox_px: tuple[int, int, int, int]
    confidence: float


class ObjectDetector:
    """Detect dark objects on a Otsu-thresholded binary mask.

    Pipeline: grayscale -> Otsu inverse-binary threshold -> external
    contour extraction -> area and size gating -> fill-ratio confidence
    -> sort by area (descending) -> confidence gate. The detector is a
    deliberately simple stand-in tuned for the synthetic scene in which
    dark vehicles and the sign appear on a bright ground plane.

    Extension point: replace with a YOLO/DETR detector; keep the
    ``ObjectDetection`` dataclass as the interface for downstream
    tracking and fusion stages.
    """

    def __init__(self, config: dict) -> None:
        """Configure the detector.

        Args:
            config: ``object_detection`` section of the module YAML with
                keys ``min_area_px`` [px^2] (minimum contour area) and
                ``confidence_threshold`` [-] (minimum fill-ratio score).
        """
        self._min_area_px = int(config["min_area_px"])
        self._confidence_threshold = float(config["confidence_threshold"])

    def detect(self, image: np.ndarray) -> list[ObjectDetection]:
        """Find objects in one BGR frame.

        Args:
            image: BGR frame, ``uint8`` array ``(height, width, 3)``.

        Returns:
            Detections sorted by object area [px^2] in descending order.
            A contour is kept when its area is at least ``min_area_px``,
            its bounding box is smaller than the full frame (this rejects
            the background ground plane), and its confidence reaches
            ``confidence_threshold``.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, ObjectDetection]] = []
        for contour in contours:
            # Pixel area of the filled contour region. Unlike
            # cv2.contourArea (which excludes the boundary pixel ring),
            # this counts every foreground pixel, so a perfectly filled
            # rectangle scores exactly 0.5 in the fill-ratio confidence.
            component = np.zeros_like(mask)
            cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
            area = float(cv2.countNonZero(component))
            if area < self._min_area_px:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w >= width or box_h >= height:
                # Rejects full-frame blobs such as the background plane.
                continue
            # Fill-ratio score: filled contours reach ~0.5, partial
            # shapes score lower; the factor 2 maps the fill ratio onto [0, 1].
            confidence = min(1.0, area / (float(box_w) * float(box_h) * 2.0))
            detection = ObjectDetection(
                label="object",
                bbox_px=(int(x), int(y), int(x + box_w), int(y + box_h)),
                confidence=float(confidence),
            )
            candidates.append((area, detection))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            detection
            for _, detection in candidates
            if detection.confidence >= self._confidence_threshold
        ]
