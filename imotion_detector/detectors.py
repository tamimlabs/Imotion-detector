"""Face detectors.

The default detector is OpenCV's YuNet (ONNX, ~230 KB), a modern deep-learning
detector that handles angles, low light and partial occlusion far better than the
classic Haar cascade. The Haar cascade remains available as a fallback for
minimal installations and as a baseline in benchmarks.
"""

# Import future annotations for better type hints
from __future__ import annotations

# Logging is used for debug messages
import logging
# Any is used for type hints
from typing import Any

# OpenCV is used for face detection
import cv2
# NumPy is used for array operations
import numpy as np

# Import the model download utility
from .utils import ensure_model_downloaded

# Create a logger for this module
logger = logging.getLogger(__name__)

#: A face bounding box as (x, y, width, height) in pixels.
FaceBox = tuple[int, int, int, int]  # Type alias for face bounding box

# URL for the YuNet model - this is hosted on GitHub
YU_NET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
# Filename for the YuNet model - this is what it saves as locally
YU_NET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

# Path to the Haar cascade XML file - comes with OpenCV
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]


# Base class for all face detectors
class FaceDetector:
    """Abstract interface for a face detector.

    Implementations return the bounding boxes of every face found in a BGR frame.
    """

    #: Machine-readable identifier used by the CLI (--face-detector).
    name = "base"  # The name of the detector

    def detect(self, frame: np.ndarray[Any, np.dtype[Any]]) -> list[FaceBox]:
        # This method should be implemented by subclasses
        raise NotImplementedError  # Raise error if not implemented


# YuNet face detector - the default detector
class YuNetFaceDetector(FaceDetector):
    """OpenCV Zoo YuNet face detector (ONNX model, auto-downloaded on first use)."""

    name = "yunet"  # The name used in CLI

    def __init__(
        self,
        model_dir: str | None = None,  # Directory to store the model
        min_face_size: int = 100,  # Minimum face size in pixels
        score_threshold: float = 0.9,  # Score threshold for detection
        nms_threshold: float = 0.3,  # NMS threshold
        top_k: int = 5000,  # Top K detections to keep
    ) -> None:
        # Download the model if not already present
        model_path = ensure_model_downloaded(
            YU_NET_MODEL_URL, YU_NET_MODEL_FILENAME, model_dir=model_dir
        )
        # Create the face detector
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),  # Path to the model
            "",  # Config file (empty)
            (320, 320),  # Input size
            score_threshold,  # Score threshold
            nms_threshold,  # NMS threshold
            top_k,  # Top K
        )
        self.min_face_size = min_face_size  # Store min face size
        logger.debug("Initialised YuNet face detector")  # Log initialization

    def detect(self, frame: np.ndarray[Any, np.dtype[Any]]) -> list[FaceBox]:
        # Get frame dimensions
        h, w = frame.shape[:2]  # Height and width
        # Set input size for the detector
        self._detector.setInputSize((w, h))  # Set to frame size
        # Detect faces
        retval, faces = self._detector.detect(frame)  # Detect faces in frame
        # Check if any faces were found
        if faces is None:  # No faces found
            return []  # Return empty list
        # Process detected faces
        boxes = []  # List to store face boxes
        for face in np.asarray(faces):  # Iterate over detected faces
            # Extract bounding box coordinates
            x, y, bw, bh = (int(v) for v in face[:4])  # x, y, width, height
            # Filter by minimum face size
            if bw >= self.min_face_size and bh >= self.min_face_size:  # Check size
                boxes.append((x, y, bw, bh))  # Add to list
        return boxes  # Return list of face boxes


# Haar cascade face detector - the fallback detector
class HaarFaceDetector(FaceDetector):
    """Classic OpenCV Haar cascade detector (no downloads, less accurate)."""

    name = "haar"  # The name used in CLI

    def __init__(
        self,
        min_face_size: int = 100,  # Minimum face size in pixels
        scale_factor: float = 1.1,  # Scale factor for detection
        min_neighbors: int = 5,  # Minimum neighbors for detection
    ) -> None:
        # Load the Haar cascade
        self._cascade = cv2.CascadeClassifier(HAAR_CASCADE)  # Load cascade
        # Check if cascade loaded successfully
        if self._cascade.empty():  # Empty cascade
            raise RuntimeError("Failed to load the Haar cascade for face detection.")
        # Store parameters
        self.min_face_size = min_face_size  # Store min face size
        self.scale_factor = scale_factor  # Store scale factor
        self.min_neighbors = min_neighbors  # Store min neighbors
        logger.debug("Initialised Haar face detector")  # Log initialization

    def detect(self, frame: np.ndarray[Any, np.dtype[Any]]) -> list[FaceBox]:
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # Detect faces
        faces = self._cascade.detectMultiScale(  # Detect faces
            gray,  # Grayscale image
            scaleFactor=self.scale_factor,  # Scale factor
            minNeighbors=self.min_neighbors,  # Min neighbors
            minSize=(self.min_face_size, self.min_face_size),  # Min size
        )
        # Convert to list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# Factory function to build a face detector
def build_detector(
    name: str, model_dir: str | None = None, min_face_size: int = 100
) -> FaceDetector:
    """Build a face detector by its registered name."""
    # Check which detector to build
    if name == YuNetFaceDetector.name:  # YuNet detector
        return YuNetFaceDetector(model_dir=model_dir, min_face_size=min_face_size)
    if name == HaarFaceDetector.name:  # Haar detector
        return HaarFaceDetector(min_face_size=min_face_size)
    # Unknown detector
    raise ValueError(
        f"Unknown face detector {name!r}. Choose one of: "
        f"{', '.join(sorted(DETECTORS))}"
    )


# Registry of all available detectors
DETECTORS = {
    cls.name: cls for cls in (YuNetFaceDetector, HaarFaceDetector)
}


# Function to list all available detectors
def list_detectors() -> list[str]:
    # Return sorted list of detector names
    return sorted(DETECTORS)
