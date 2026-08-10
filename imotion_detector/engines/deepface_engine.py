"""DeepFace-based emotion engine — highest accuracy."""

import logging

from .base import EmotionEngine, EngineDependencyError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "DeepFace is not installed. Install it with:\n"
    "  pip install -r requirements/deepface.txt"
)


class DeepFaceEngine(EmotionEngine):
    name = "deepface"
    description = "DeepFace deep-learning model - highest accuracy, heavier (TensorFlow/PyTorch)"
    requirements_file = "requirements/deepface.txt"

    BACKENDS = [
        "opencv",
        "ssd",
        "dlib",
        "mtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet",
        "fastmtcnn",
    ]

    def __init__(self, detector_backend="opencv"):
        if detector_backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown DeepFace detector backend {detector_backend!r}. "
                f"Choose one of: {', '.join(self.BACKENDS)}"
            )
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise EngineDependencyError(_INSTALL_HINT) from exc

        self._analyze = DeepFace.analyze
        self.detector_backend = detector_backend
        logger.debug("Initialised DeepFace engine (backend=%s)", detector_backend)

    def analyze(self, face_bgr):
        result = self._analyze(
            face_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=self.detector_backend,
        )
        if result:
            return result[0].get("dominant_emotion")
        return None
