"""Engine registry — every supported emotion backend is registered here."""

from .base import EmotionEngine, EngineDependencyError
from .deepface_engine import DeepFaceEngine
from .fer_engine import FEREngine
from .opencv_engine import OpenCVEngine

ENGINES = {
    DeepFaceEngine.name: DeepFaceEngine,
    FEREngine.name: FEREngine,
    OpenCVEngine.name: OpenCVEngine,
}

__all__ = [
    "EmotionEngine",
    "EngineDependencyError",
    "DeepFaceEngine",
    "FEREngine",
    "OpenCVEngine",
    "ENGINES",
]
