"""Abstract base class shared by every emotion engine."""

# Importing future annotations for type hinting support
from __future__ import annotations

# ABC is used for abstract base classes
from abc import ABC, abstractmethod
# Any type is used for numpy array typing
from typing import Any

# NumPy is used for array operations
import numpy as np


# This error is raised when dependencies are missing
class EngineDependencyError(RuntimeError):
    """Raised when an engine's optional dependencies are not installed."""


# This is the main engine class that all engines inherit from
class EmotionEngine(ABC):
    """A pluggable emotion analysis backend.

    Subclasses analyse a single face crop (a BGR numpy array) and return
    the dominant emotion label as a string (or ``None`` when no emotion
    could be determined).
    """

    #: Machine-readable identifier used by the CLI (--engine).
    name = "base"  # The name of the engine

    #: Human readable summary shown in --list-engines and the README.
    description = ""  # The description of the engine

    # This is the abstract method that all engines must implement
    @abstractmethod
    def analyze(self, face_bgr: np.ndarray[Any, np.dtype[Any]]) -> str | None:
        """Return the dominant emotion label for a face crop (BGR image)."""
        # This method should be implemented by subclasses
        # It takes a face crop as input and returns the emotion
        pass  # Pass is used because this is an abstract method
