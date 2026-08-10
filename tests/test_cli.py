"""Tests for the CLI argument parsing, source resolution and runners."""

from __future__ import annotations

import pytest

from imotion_detector.cli import build_parser, resolve_source
from imotion_detector.detectors import build_detector


class TestParser:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.engine == "deepface"
        assert args.face_detector == "yunet"
        assert args.frame_skip == 10
        assert args.min_face_size == 100
        assert args.iou_threshold == 0.3
        assert args.serve is None
        assert args.benchmark_frames is None

    def test_engine_choices(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--engine", "wat"])

    def test_face_detector_choices(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--face-detector", "wat"])

    def test_benchmark_and_serve_flags(self):
        args = build_parser().parse_args(["--benchmark-frames", "50", "--serve", "8080"])
        assert args.benchmark_frames == 50
        assert args.serve == 8080


class TestResolveSource:
    def test_none_means_webcam(self):
        assert resolve_source(None) == ("webcam", None)

    def test_numeric_means_camera(self):
        assert resolve_source("1") == ("webcam", 1)

    def test_webcam_keyword(self):
        assert resolve_source("webcam") == ("webcam", None)

    def test_image_file(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"fake")
        assert resolve_source(str(img)) == ("image", img)

    def test_video_file(self, tmp_path):
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"fake")
        assert resolve_source(str(vid)) == ("video", vid)

    def test_unsupported_extension(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_bytes(b"fake")
        with pytest.raises(SystemExit, match="Unsupported file type"):
            resolve_source(str(bad))

    def test_missing_source(self):
        with pytest.raises(SystemExit, match="Source not found"):
            resolve_source("C:/definitely/not/here.jpg")


class TestBuildDetector:
    @pytest.mark.models
    def test_yunet_default(self, tmp_path):
        detector = build_detector("yunet", model_dir=str(tmp_path), min_face_size=64)
        assert detector.name == "yunet"

    def test_haar_needs_no_model(self):
        detector = build_detector("haar", min_face_size=64)
        assert detector.name == "haar"
