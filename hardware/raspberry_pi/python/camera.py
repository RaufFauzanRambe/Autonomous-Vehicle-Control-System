"""
File:        python/camera.py
Brief:       CameraModule — encapsulates the Raspberry Pi camera (v2/v3)
             via libcamera/picamera2. Provides threaded frame capture,
             FPS control, exposure/gain settings, and basic image
             preprocessing (resize + normalize).
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger

# picamera2 may not be importable on a dev workstation; import lazily
# so unit tests can stub it.
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    _PICAMERA2_AVAILABLE = True
except ImportError:  # pragma: no cover - hardware-only
    Picamera2 = None  # type: ignore
    _PICAMERA2_AVAILABLE = False


# ----------------------------------------------------------------------
# Frame container
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Frame:
    """A captured camera frame with metadata."""

    image: np.ndarray              # BGR uint8 HxWxC
    timestamp: float               # monotonic seconds
    frame_id: int
    exposure_us: int
    gain: float

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape  # type: ignore[return-value]


# ----------------------------------------------------------------------
# Camera module
# ----------------------------------------------------------------------
class CameraModule:
    """Threaded camera driver for the Raspberry Pi.

    A dedicated worker thread continuously captures frames at the desired
    FPS. The latest frame is stored in a single-slot buffer protected by
    a lock so consumers always get the freshest available image without
    blocking the producer.

    Attributes:
        width:   Output frame width in pixels.
        height:  Output frame height in pixels.
        fps:     Target capture rate in Hz.
    """

    DEFAULT_WIDTH: int = 640
    DEFAULT_HEIGHT: int = 480
    DEFAULT_FPS: int = 30

    def __init__(self, config: dict) -> None:
        """Initialize the camera module.

        Args:
            config: Dictionary with keys ``width``, ``height``, ``fps``,
                    ``exposure_us``, ``gain``, ``normalize``, ``sensor_mode``.
        """
        self.width: int = int(config.get("width", self.DEFAULT_WIDTH))
        self.height: int = int(config.get("height", self.DEFAULT_HEIGHT))
        self.fps: int = int(config.get("fps", self.DEFAULT_FPS))
        self.exposure_us: int = int(config.get("exposure_us", 0))     # 0 = auto
        self.gain: float = float(config.get("gain", 0.0))             # 0 = auto
        self.normalize: bool = bool(config.get("normalize", False))
        self.sensor_mode: Optional[int] = config.get("sensor_mode")

        self._picam2: Optional[Picamera2] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._frame_id: int = 0
        self._drop_count: int = 0

        if not _PICAMERA2_AVAILABLE:
            logger.warning("picamera2 not available — running in stub mode")
        else:
            self._open_camera()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _open_camera(self) -> None:
        """Open the camera and configure the stream."""
        logger.info("Opening Pi Camera ({}x{} @ {} fps)",
                    self.width, self.height, self.fps)
        try:
            self._picam2 = Picamera2()
            config = self._picam2.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "BGR888",
                },
                controls={
                    "FrameRate": float(self.fps),
                    "AeEnable": self.exposure_us == 0,
                    "AwbEnable": True,
                    "ExposureTime": self.exposure_us if self.exposure_us > 0 else 0,
                    "AnalogueGain": self.gain if self.gain > 0 else 0.0,
                },
            )
            if self.sensor_mode is not None:
                config["sensor"] = {"output_size": None,
                                    "bit_depth": None,
                                    "mode": self.sensor_mode}
            self._picam2.configure(config)
            self._picam2.start()
            logger.info("Pi Camera started successfully")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open Pi Camera: {}", exc)
            self._picam2 = None
            raise

    # ------------------------------------------------------------------
    # Threaded capture loop
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background capture thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Capture thread already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True
        )
        self._thread.start()
        logger.info("Camera capture thread started")

    def stop(self) -> None:
        """Signal the capture thread to stop and join it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
            self._thread = None

    def _capture_loop(self) -> None:
        """Worker: continuously grab frames and stash the latest."""
        period = 1.0 / self.fps if self.fps > 0 else 0.033
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                frame = self._grab_one()
                if frame is not None:
                    with self._lock:
                        self._latest = frame
                else:
                    self._drop_count += 1
                    if self._drop_count % 50 == 0:
                        logger.warning("Camera dropped {} frames", self._drop_count)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Capture error: {}", exc)
                time.sleep(0.05)

            elapsed = time.monotonic() - loop_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Single-frame capture
    # ------------------------------------------------------------------
    def _grab_one(self) -> Optional[Frame]:
        """Capture a single frame from the camera."""
        if self._picam2 is None:
            # Stub mode for tests / no hardware
            self._frame_id += 1
            return Frame(
                image=np.zeros((self.height, self.width, 3), dtype=np.uint8),
                timestamp=time.monotonic(),
                frame_id=self._frame_id,
                exposure_us=0,
                gain=0.0,
            )

        metadata = self._picam2.capture_metadata() or {}
        raw = self._picam2.capture_array()
        if raw is None:
            return None
        if raw.dtype != np.uint8:
            raw = raw.astype(np.uint8)
        if raw.ndim == 2:                   # grayscale → BGR
            raw = np.stack([raw, raw, raw], axis=-1)
        if raw.shape[1] != self.width or raw.shape[0] != self.height:
            # picamera2 already sized the main stream; if not, resize.
            raw = self._resize(raw, (self.width, self.height))
        if self.normalize:
            raw = self._normalize(raw)

        self._frame_id += 1
        return Frame(
            image=raw,
            timestamp=time.monotonic(),
            frame_id=self._frame_id,
            exposure_us=int(metadata.get("ExposureTime", 0)),
            gain=float(metadata.get("AnalogueGain", 0.0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_latest_frame(self) -> Optional[Frame]:
        """Return the most recent frame (or ``None`` if not started)."""
        with self._lock:
            return self._latest

    def capture(self) -> Optional[Frame]:
        """Synchronously capture a single frame."""
        return self._grab_one()

    def set_exposure(self, exposure_us: int) -> None:
        """Set sensor exposure time in microseconds (0 = auto)."""
        self.exposure_us = exposure_us
        if self._picam2:
            self._picam2.set_controls({
                "AeEnable": exposure_us == 0,
                "ExposureTime": exposure_us,
            })
        logger.debug("Exposure set to {} us", exposure_us)

    def set_gain(self, gain: float) -> None:
        """Set analogue gain (0 = auto)."""
        self.gain = gain
        if self._picam2:
            self._picam2.set_controls({
                "AnalogueGain": gain,
            })
        logger.debug("Analogue gain set to {:.2f}", gain)

    def set_fps(self, fps: int) -> None:
        """Update the target frame rate."""
        self.fps = max(1, fps)
        if self._picam2:
            self._picam2.set_controls({"FrameRate": float(self.fps)})

    def save_snapshot(self, path: str) -> bool:
        """Save the latest frame as a PNG/JPEG.

        Args:
            path: Output file path; extension determines format.

        Returns:
            ``True`` if the file was written.
        """
        frame = self.get_latest_frame()
        if frame is None:
            return False
        try:
            from PIL import Image
            Image.fromarray(frame.image[..., ::-1]).save(path)  # BGR → RGB
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save snapshot: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Image preprocessing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Bilinear resize using NumPy (no OpenCV dependency required)."""
        try:
            import cv2
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            h_src, w_src = img.shape[:2]
            w_dst, h_dst = size
            xs = np.linspace(0, w_src - 1, w_dst).astype(np.float32)
            ys = np.linspace(0, h_src - 1, h_dst).astype(np.float32)
            xi = np.clip(xs.astype(np.int32), 0, w_src - 2)
            yi = np.clip(ys.astype(np.int32), 0, h_src - 2)
            xfrac = (xs - xi)[:, None]
            yfrac = (ys - yi)[None, :]
            tl = img[yi[:, None], xi[None, :]]
            tr = img[yi[:, None], xi[None, :] + 1]
            bl = img[yi[:, None] + 1, xi[None, :]]
            br = img[yi[:, None] + 1, xi[None, :] + 1]
            top = tl * (1 - xfrac) + tr * xfrac
            bot = bl * (1 - xfrac) + br * xfrac
            return (top * (1 - yfrac) + bot * yfrac).astype(np.uint8)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Scale to [0, 1] float32."""
        return (img.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return internal stats for telemetry."""
        return {
            "frame_id": self._frame_id,
            "drops": self._drop_count,
            "width": self.width,
            "height": self.height,
            "fps_target": self.fps,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the capture thread and release the camera."""
        self.stop()
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing camera: {}", exc)
            self._picam2 = None
        logger.info("Camera closed")

    def __enter__(self) -> "CameraModule":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
