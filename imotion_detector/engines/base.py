"""Abstract base class shared by every emotion engine."""

from abc import ABC, abstractmethod


class EngineDependencyError(RuntimeError):
    """Raised when an engine's optional dependencies are not installed."""


class EmotionEngine(ABC):
    """A pluggable emotion analysis backend.

    Subclasses analyse a single face crop (a BGR numpy array) and return
    the dominant emotion label as a string (or ``None`` when no emotion
    could be determined).
    """

    #: Machine-readable identifier used by the CLI (--engine).
    name = "base"

    #: Human readable summary shown in --list-engines and the README.
    description = ""

    @abstractmethod
    def analyze(self, face_bgr):
        """Return the dominant emotion label for a face crop (BGR image)."""
