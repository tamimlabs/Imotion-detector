"""Tests for face tracking, IOU matching and the annotation pipeline."""

from __future__ import annotations

import numpy as np

from imotion_detector.detector import FaceEmotionPipeline, iou


def _frame(size=200):
    return np.zeros((size, size, 3), dtype=np.uint8)


class TestIoU:
    def test_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert iou(box, box) == 1.0

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    def test_partial_overlap(self):
        assert 0.0 < iou((0, 0, 10, 10), (5, 5, 10, 10)) < 1.0

    def test_contained_box(self):
        assert iou((0, 0, 100, 100), (25, 25, 50, 50)) == 0.25

    def test_zero_area(self):
        assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


class TestPipeline:
    def test_creates_tracks_and_updates_emotions(self, fake_engine, fake_detector):
        fake_detector.boxes = [(10, 10, 80, 80), (150, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(fake_engine, detector=fake_detector)
        pipeline.process(_frame())
        snapshot = pipeline.snapshot()

        assert len(snapshot) == 2
        assert {t["emotion"] for t in snapshot} == {"happy"}
        assert [t["id"] for t in snapshot] == [0, 1]

    def test_frame_skip_controls_analysis_frequency(self, fake_detector):
        class CountingEngine:
            name = "counting"
            calls = 0

            def analyze(self, face_bgr):
                type(self).calls += 1
                return "happy"

        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(CountingEngine(), detector=fake_detector, frame_skip=3)

        for _ in range(6):
            pipeline.process(_frame())

        assert CountingEngine.calls == 2  # frames 0 and 3 are analysed

    def test_matches_moving_face(self, fake_engine, fake_detector):
        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(fake_engine, detector=fake_detector)
        pipeline.process(_frame())
        pipeline.process(_frame())
        first_id = pipeline.snapshot()[0]["id"]

        fake_detector.boxes = [(20, 15, 80, 80)]
        pipeline.process(_frame())
        after = pipeline.snapshot()

        assert len(after) == 1
        assert after[0]["id"] == first_id  # same track, not a new face

    def test_unmatched_face_starts_new_track(self, fake_engine, fake_detector):
        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(
            fake_engine, detector=fake_detector, iou_threshold=0.9
        )
        pipeline.process(_frame())

        fake_detector.boxes = [(100, 100, 80, 80)]  # zero overlap with track
        pipeline.process(_frame())
        after = pipeline.snapshot()

        assert len(after) == 2
        assert after[1]["id"] == 1

    def test_tracks_pruned_after_misses(self, fake_engine, fake_detector):
        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(fake_engine, detector=fake_detector, max_misses=2)
        pipeline.process(_frame())

        fake_detector.boxes = []
        pipeline.process(_frame())
        pipeline.process(_frame())
        assert len(pipeline.snapshot()) == 1

        pipeline.process(_frame())
        pipeline.process(_frame())
        assert pipeline.snapshot() == []

    def test_snapshot_is_json_serialisable(self, fake_engine, fake_detector):
        import json

        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(fake_engine, detector=fake_detector)
        pipeline.process(_frame())
        json.dumps(pipeline.snapshot())  # must not raise

    def test_reset_clears_tracks(self, fake_engine, fake_detector):
        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(fake_engine, detector=fake_detector)
        pipeline.process(_frame())
        pipeline.reset()
        assert pipeline.snapshot() == []

    def test_crop_respects_frame_bounds(self, fake_engine, fake_detector):
        fake_detector.boxes = [(190, 190, 80, 80)]  # sticks out of a 200x200 frame
        pipeline = FaceEmotionPipeline(fake_engine, detector=fake_detector, pad_ratio=0.0)
        result = pipeline.process(_frame())
        assert result.shape == (200, 200, 3)

    def test_analyze_fallback_on_engine_error(self, fake_detector):
        class BrokenEngine:
            name = "broken"

            def analyze(self, face_bgr):
                raise RuntimeError("boom")

        fake_detector.boxes = [(10, 10, 80, 80)]
        pipeline = FaceEmotionPipeline(BrokenEngine(), detector=fake_detector)
        pipeline.process(_frame())
        pipeline.process(_frame())  # analysis frame
        assert pipeline.snapshot()[0]["emotion"] == "Analyzing..."
