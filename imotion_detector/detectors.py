"""Face detectors.

The default detector is OpenCV's YuNet (ONNX, ~230 KB), a modern deep-learning
detector that handles angles, low light and partial occlusion far better than the
classic Haar cascade. The Haar cascade remains available as a fallback for
minimal installations and as a baseline in benchmarks.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .utils import ensure_model_downloaded

logger = logging.getLogger(__name__)

#: A face bounding box as (x, y, width, height) in pixels.
FaceBox = tuple[int, int, int, int]

YU_NET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YU_NET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]


class FaceDetector:
    """Abstract interface for a face detector.

    Implementations return the bounding boxes of every face found in a BGR frame.
    """

    #: Machine-readable identifier used by the CLI (--face-detector).
    name = "base"

    def detect(self, frame: np.ndarray[Any, np.dtype[Any]]) -> list[FaceBox]:
        raise NotImplementedError


class YuNetFaceDetector(FaceDetector):
    """OpenCV Zoo YuNet face detector (ONNX model, auto-downloaded on first use)."""

    name = "yunet"

    def __init__(
        self,
        model_dir: str | None = None,
        min_face_size: int = 100,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        model_path = ensure_model_downloaded(
            YU_NET_MODEL_URL, YU_NET_MODEL_FILENAME, model_dir=model_dir
        )
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.min_face_size = min_face_size
        logger.debug("Initialised YuNet face detector")

    def detect(self, frame: np.ndarray[Any, np.dtype[Any]]) -> list[FaceBox]:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        retval, faces = self._detector.detect(frame)
        if faces is None:
            return []
        boxes = []
        for face in np.asarray(faces):
            x, y, bw, bh = (int(v) for v in face[:4])
            if bw >= self.min_face_size and bh >= self.min_face_size:
                boxes.append((x, y, bw, bh))
        return boxes


class HaarFaceDetector(FaceDetector):
    """Classic OpenCV Haar cascade detector (no downloads, less accurate)."""

    name = "haar"

    def __init__(
        self,
        min_face_size: int = 100,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
    ) -> None:
        self._cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        if self._cascade.empty():
            raise RuntimeError("Failed to load the Haar cascade for face detection.")
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        logger.debug("Initialised Haar face detector")

    def detect(self, frame: np.ndarray[Any, np.dtype[Any]]) -> list[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def build_detector(
    name: str, model_dir: str | None = None, min_face_size: int = 100
) -> FaceDetector:
    """Build a face detector by its registered name."""
    if name == YuNetFaceDetector.name:
        return YuNetFaceDetector(model_dir=model_dir, min_face_size=min_face_size)
    if name == HaarFaceDetector.name:
        return HaarFaceDetector(min_face_size=min_face_size)
    raise ValueError(
        f"Unknown face detector {name!r}. Choose one of: "
        f"{', '.join(sorted(DETECTORS))}"
    )


DETECTORS = {
    cls.name: cls for cls in (YuNetFaceDetector, HaarFaceDetector)
}


def list_detectors() -> list[str]:
    return sorted(DETECTORS)
