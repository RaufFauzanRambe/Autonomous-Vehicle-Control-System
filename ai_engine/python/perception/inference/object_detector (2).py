"""
Object Detector Module for Autonomous Vehicle Perception

Implements real-time object detection using deep learning models
for identifying vehicles, pedestrians, cyclists, and other road objects
from camera and LiDAR data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ObjectClass(Enum):
    """Supported object detection classes."""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox2D:
    """2D bounding box in image coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: 'BoundingBox2D') -> float:
        """Calculate Intersection over Union with another bounding box."""
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class BoundingBox3D:
    """3D bounding box in world coordinates."""
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float  # Rotation around z-axis in radians

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def get_corners(self) -> np.ndarray:
        """Get the 8 corners of the 3D bounding box."""
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        half_dims = np.array([
            [self.length / 2, self.width / 2, self.height / 2],
            [self.length / 2, self.width / 2, -self.height / 2],
            [self.length / 2, -self.width / 2, self.height / 2],
            [self.length / 2, -self.width / 2, -self.height / 2],
            [-self.length / 2, self.width / 2, self.height / 2],
            [-self.length / 2, self.width / 2, -self.height / 2],
            [-self.length / 2, -self.width / 2, self.height / 2],
            [-self.length / 2, -self.width / 2, -self.height / 2],
        ])

        corners = (rotation @ half_dims.T).T + np.array([self.x, self.y, self.z])
        return corners


@dataclass
class DetectedObject:
    """Represents a detected object with all its properties."""
    object_id: int
    object_class: ObjectClass
    confidence: float
    bbox_2d: Optional[BoundingBox2D] = None
    bbox_3d: Optional[BoundingBox3D] = None
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    timestamp: float = 0.0
    attributes: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert detected object to dictionary for serialization."""
        result = {
            'object_id': self.object_id,
            'object_class': self.object_class.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'attributes': self.attributes,
        }
        if self.bbox_2d is not None:
            result['bbox_2d'] = {
                'x_min': self.bbox_2d.x_min, 'y_min': self.bbox_2d.y_min,
                'x_max': self.bbox_2d.x_max, 'y_max': self.bbox_2d.y_max,
            }
        if self.bbox_3d is not None:
            result['bbox_3d'] = {
                'x': self.bbox_3d.x, 'y': self.bbox_3d.y, 'z': self.bbox_3d.z,
                'length': self.bbox_3d.length, 'width': self.bbox_3d.width,
                'height': self.bbox_3d.height, 'yaw': self.bbox_3d.yaw,
            }
        if self.velocity is not None:
            result['velocity'] = self.velocity.tolist()
        return result


class ObjectDetector:
    """
    Real-time object detector for autonomous driving.

    Supports multiple backbone architectures and provides both
    2D and 3D object detection capabilities. Designed to work
    with camera images and LiDAR point clouds.
    """

    # Default confidence thresholds per class
    DEFAULT_THRESHOLDS = {
        ObjectClass.VEHICLE: 0.7,
        ObjectClass.PEDESTRIAN: 0.6,
        ObjectClass.CYCLIST: 0.6,
        ObjectClass.TRAFFIC_SIGN: 0.5,
        ObjectClass.TRAFFIC_LIGHT: 0.5,
    }

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_detections: int = 100,
        input_size: Tuple[int, int] = (640, 480),
    ):
        """
        Initialize the object detector.

        Args:
            model_path: Path to the trained model weights file.
            device: Compute device ('cuda' or 'cpu').
            confidence_threshold: Minimum confidence for detections.
            nms_threshold: Non-maximum suppression IoU threshold.
            max_detections: Maximum number of detections per frame.
            input_size: Input image size (width, height).
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        self._model = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """
        Load the model weights and prepare for inference.

        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            # Model loading logic would go here
            # self._model = load_model(self.model_path, self.device)
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize object detector: {e}")
            return False

    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> List[DetectedObject]:
        """
        Detect objects in an input image.

        Args:
            image: Input image as a numpy array (H, W, C) in BGR format.
            timestamp: Current timestamp for the detection.

        Returns:
            List of DetectedObject instances.
        """
        if not self._is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize() first.")

        # Preprocess image
        preprocessed = self._preprocess(image)

        # Run inference (placeholder)
        raw_detections = self._inference(preprocessed)

        # Post-process with NMS
        detections = self._postprocess(raw_detections, timestamp)

        return detections[:self.max_detections]

    def detect_point_cloud(
        self,
        points: np.ndarray,
        timestamp: float = 0.0
    ) -> List[DetectedObject]:
        """
        Detect objects from LiDAR point cloud data.

        Args:
            points: Point cloud array (N, 4) with [x, y, z, intensity].
            timestamp: Current timestamp for the detection.

        Returns:
            List of DetectedObject instances with 3D bounding boxes.
        """
        if not self._is_initialized:
            raise RuntimeError("Detector not initialized. Call initialize() first.")

        # Preprocess point cloud
        voxelized = self._voxelize(points)

        # Run 3D inference (placeholder)
        raw_detections = self._inference_3d(voxelized)

        # Post-process with 3D NMS
        detections = self._postprocess_3d(raw_detections, timestamp)

        return detections[:self.max_detections]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for model inference."""
        resized = self._resize_image(image, self.input_size)
        normalized = resized.astype(np.float32) / 255.0
        # Apply model-specific normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        # Placeholder - would use cv2 or PIL in production
        return image

    def _voxelize(self, points: np.ndarray) -> Dict:
        """Convert point cloud to voxel representation for 3D detection."""
        voxel_size = [0.1, 0.1, 0.2]  # meters
        point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # x, y, z ranges

        # Compute voxel grid dimensions
        grid_size = [
            int((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])
            for i in range(3)
        ]

        return {
            'voxels': points,
            'voxel_num_points': np.array([len(points)]),
            'voxel_coords': np.zeros((1, 3), dtype=np.int32),
            'grid_size': grid_size,
            'point_cloud_range': point_cloud_range,
        }

    def _inference(self, preprocessed: np.ndarray) -> List[Dict]:
        """Run model inference on preprocessed input."""
        # Placeholder for actual model inference
        return []

    def _inference_3d(self, voxelized: Dict) -> List[Dict]:
        """Run 3D model inference on voxelized point cloud."""
        # Placeholder for actual 3D model inference
        return []

    def _postprocess(self, raw_detections: List[Dict], timestamp: float) -> List[DetectedObject]:
        """Post-process raw detections with NMS and threshold filtering."""
        filtered = []
        for det in raw_detections:
            if det.get('confidence', 0) >= self.confidence_threshold:
                obj = DetectedObject(
                    object_id=det.get('id', 0),
                    object_class=ObjectClass(det.get('class', 'unknown')),
                    confidence=det['confidence'],
                    timestamp=timestamp,
                )
                filtered.append(obj)

        # Apply NMS
        return self._nms(filtered)

    def _postprocess_3d(self, raw_detections: List[Dict], timestamp: float) -> List[DetectedObject]:
        """Post-process 3D detections with 3D NMS."""
        filtered = []
        for det in raw_detections:
            if det.get('confidence', 0) >= self.confidence_threshold:
                obj = DetectedObject(
                    object_id=det.get('id', 0),
                    object_class=ObjectClass(det.get('class', 'unknown')),
                    confidence=det['confidence'],
                    bbox_3d=BoundingBox3D(**det.get('bbox3d', {})),
                    timestamp=timestamp,
                )
                filtered.append(obj)
        return filtered

    def _nms(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Apply Non-Maximum Suppression to filter overlapping detections."""
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            sorted_dets = [
                d for d in sorted_dets
                if d.bbox_2d is None
                or best.bbox_2d is None
                or d.bbox_2d.iou(best.bbox_2d) < self.nms_threshold
            ]

        return keep

    def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        self._is_initialized = False
