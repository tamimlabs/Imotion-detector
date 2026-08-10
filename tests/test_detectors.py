"""Tests for the face detector backends (YuNet / Haar)."""

from __future__ import annotations

import numpy as np
import pytest

from imotion_detector.detectors import (
    DETECTORS,
    HaarFaceDetector,
    YuNetFaceDetector,
    build_detector,
    list_detectors,
)


def _noise_frame():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


class TestRegistry:
    def test_all_detectors_registered(self):
        assert set(DETECTORS) == {"yunet", "haar"}
        assert list_detectors() == ["haar", "yunet"]

    def test_build_by_name(self, tmp_path):
        haar = build_detector("haar", min_face_size=64)
        assert isinstance(haar, HaarFaceDetector)
        assert haar.min_face_size == 64

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            build_detector("nope")


class TestHaar:
    def test_no_faces_in_noise(self):
        detector = HaarFaceDetector(min_face_size=64)
        assert detector.detect(_noise_frame()) == []


class TestYuNet:
    @pytest.mark.models
    def test_creates_and_detects_nothing_in_noise(self, tmp_path):
        detector = YuNetFaceDetector(model_dir=str(tmp_path), min_face_size=64)
        assert detector.detect(_noise_frame()) == []
