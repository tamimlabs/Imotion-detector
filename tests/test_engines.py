"""Tests for the emotion engine registry and backends."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from imotion_detector.engines import ENGINES, EmotionEngine, EngineDependencyError
from imotion_detector.engines.deepface_engine import DeepFaceEngine
from imotion_detector.engines.fer_engine import FEREngine
from imotion_detector.engines.opencv_engine import EMOTIONS, OpenCVEngine


class TestRegistry:
    def test_all_engines_registered(self):
        assert set(ENGINES) == {"deepface", "fer", "opencv"}

    def test_every_engine_is_an_emotion_engine(self):
        for cls in ENGINES.values():
            assert issubclass(cls, EmotionEngine)
            assert isinstance(cls.name, str) and cls.name
            assert cls.description


class TestDeepFaceEngine:
    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError):
            DeepFaceEngine(detector_backend="does-not-exist")

    def test_missing_dependency_raises_engine_dependency_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "deepface", None)
        with pytest.raises(EngineDependencyError):
            DeepFaceEngine()


class TestFEREngine:
    def test_missing_dependency_raises_engine_dependency_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "fer", None)
        with pytest.raises(EngineDependencyError):
            FEREngine()


class TestOpenCVEngine:
    @pytest.mark.models
    def test_analyze_returns_a_known_emotion(self, tmp_path):
        engine = OpenCVEngine(model_dir=str(tmp_path))
        rng = np.random.default_rng(0)
        face = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)
        assert engine.analyze(face) in EMOTIONS
