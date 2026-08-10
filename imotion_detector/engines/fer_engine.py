"""FER (Facial Expression Recognition) based engine — lightweight."""

from __future__ import annotations

import logging

import numpy as np

from .base import EmotionEngine, EngineDependencyError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "FER is not installed. Install it with:\n"
    "  pip install -r requirements/fer.txt"
)


class FEREngine(EmotionEngine):
    name = "fer"
    description = "FER facial-expression library - lightweight, CPU-friendly"
    requirements_file = "requirements/fer.txt"

    def __init__(self, mtcnn: bool = False) -> None:
        try:
            from fer import FER
        except ImportError as exc:
            raise EngineDependencyError(_INSTALL_HINT) from exc

        self._detector = FER(mtcnn=mtcnn)
        self.mtcnn = mtcnn
        logger.debug("Initialised FER engine (mtcnn=%s)", mtcnn)

    def analyze(self, face_bgr: np.ndarray) -> str | None:
        results = self._detector.detect_emotions(face_bgr)
        if results:
            emotions = results[0].get("emotions")
            if emotions:
                return max(emotions, key=emotions.get)
        return None
