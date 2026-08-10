"""OpenCV DNN engine — lightest option, uses a MobileFaceNet ONNX model
from the official OpenCV Zoo repository (downloaded on first use).
"""

import logging

import cv2
import numpy as np

from ..utils import ensure_model_downloaded
from .base import EmotionEngine

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/"
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)
MODEL_FILENAME = "facial_expression_recognition_mobilefacenet_2022july.onnx"

# Label order used by the OpenCV Zoo facial expression recognition model.
EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


class OpenCVEngine(EmotionEngine):
    name = "opencv"
    description = "OpenCV DNN MobileFaceNet model - lightest, only needs opencv-python"
    requirements_file = "requirements/opencv.txt"

    def __init__(self, model_dir=None):
        model_path = ensure_model_downloaded(MODEL_URL, MODEL_FILENAME, model_dir=model_dir)
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        logger.debug("Loaded OpenCV emotion model from %s", model_path)

    def analyze(self, face_bgr):
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=2.0 / 255.0,
            size=(112, 112),
            mean=(1.0, 1.0, 1.0),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()
        index = int(np.argmax(output))
        return EMOTIONS[index]
