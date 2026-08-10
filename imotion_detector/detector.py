"""Face detection, IOU tracking, emotion analysis and annotation pipeline."""

from __future__ import annotations

import logging
import threading
from typing import Any

import cv2
import numpy as np

from .detectors import FaceBox, FaceDetector, YuNetFaceDetector
from .engines.base import EmotionEngine

logger = logging.getLogger(__name__)


def iou(a: FaceBox, b: FaceBox) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes, in [0, 1]."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class _Track:
    """A face followed across frames with a cached emotion label."""

    __slots__ = ("id", "box", "emotion", "misses")

    def __init__(
        self, track_id: int, box: FaceBox, emotion: str = "Analyzing..."
    ) -> None:
        self.id = track_id
        self.box = box
        self.emotion = emotion
        self.misses = 0


class FaceEmotionPipeline:
    """Detects faces, tracks them across frames with greedy IOU matching,
    refreshes their emotion every ``frame_skip`` frames and annotates the
    frame with bounding boxes and labels.
    """

    def __init__(
        self,
        engine: EmotionEngine,
        detector: FaceDetector | None = None,
        frame_skip: int = 10,
        min_face_size: int = 100,
        iou_threshold: float = 0.3,
        max_misses: int = 30,
        pad_ratio: float = 0.2,
    ) -> None:
        self.engine = engine
        self.frame_skip = max(1, frame_skip)
        self.min_face_size = min_face_size
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.pad_ratio = pad_ratio
        self.detector = detector or YuNetFaceDetector(min_face_size=min_face_size)

        self._tracks: list[_Track] = []
        self._next_id = 0
        self._frame_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Emotion analysis                                                    #
    # ------------------------------------------------------------------ #
    def _crop_face(self, frame: np.ndarray, box: FaceBox) -> np.ndarray:
        """Extract and lightly pad a face crop for the emotion engine."""
        x, y, w, h = (int(v) for v in box)
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        crop = frame[y : y + h, x : x + w]
        if self.pad_ratio > 0 and crop.size:
            px, py = int(w * self.pad_ratio), int(h * self.pad_ratio)
            crop = cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_REPLICATE)
        return np.asarray(crop)

    def analyze(self, frame: np.ndarray, box: FaceBox) -> str:
        """Run the engine on a single face box and return its emotion label."""
        try:
            emotion = self.engine.analyze(self._crop_face(frame, box))
            return emotion if emotion else "Unknown"
        except Exception as exc:
            logger.debug("Emotion analysis failed for face %s: %s", box, exc)
            return "Analyzing..."

    # ------------------------------------------------------------------ #
    # Tracking                                                            #
    # ------------------------------------------------------------------ #
    def _match(self, boxes: list[FaceBox]) -> dict[int, int]:
        """Greedily associate detections to existing tracks by IOU.

        Returns {detection_index: track_index}; detections left out start new
        tracks and tracks left out are marked as missing.
        """
        matches: dict[int, int] = {}
        used_tracks = set()
        for det_idx, box in enumerate(boxes):
            best_track = -1
            best_score = self.iou_threshold
            for tr_idx, track in enumerate(self._tracks):
                if tr_idx in used_tracks:
                    continue
                score = iou(box, track.box)
                if score > best_score:
                    best_score = score
                    best_track = tr_idx
            if best_track >= 0:
                matches[det_idx] = best_track
                used_tracks.add(best_track)
        return matches

    def _update(self, boxes: list[FaceBox]) -> None:
        matches = self._match(boxes)
        matched = set(matches.values())
        for det_idx, tr_idx in matches.items():
            self._tracks[tr_idx].box = boxes[det_idx]
            self._tracks[tr_idx].misses = 0
        for tr_idx, track in enumerate(self._tracks):
            if tr_idx not in matched:
                track.misses += 1
        for det_idx in range(len(boxes)):
            if det_idx not in matches:
                self._tracks.append(_Track(self._next_id, boxes[det_idx]))
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

    # ------------------------------------------------------------------ #
    # Annotation                                                          #
    # ------------------------------------------------------------------ #
    def _annotate(self, frame: np.ndarray, tracks: list[_Track]) -> None:
        for track in tracks:
            x, y, w, h = track.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                track.emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

    # ------------------------------------------------------------------ #
    # Per-frame processing                                                #
    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Track faces, refresh emotions every ``frame_skip`` frames and
        annotate the frame in place. Returns the annotated frame.
        """
        boxes = self.detector.detect(frame)
        with self._lock:
            self._update(boxes)
            should_analyze = self._frame_count % self.frame_skip == 0
            self._frame_count += 1
            tracks = list(self._tracks)

        if should_analyze:
            results = {
                track.id: self.analyze(frame, track.box)
                for track in tracks
                if track.misses == 0
            }
            if results:
                with self._lock:
                    for track in self._tracks:
                        if track.id in results:
                            track.emotion = results[track.id]

        self._annotate(frame, tracks)
        return frame

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    def snapshot(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable copy of the currently tracked faces."""
        with self._lock:
            return [
                {
                    "id": track.id,
                    "box": list(track.box),
                    "emotion": track.emotion,
                    "misses": track.misses,
                }
                for track in self._tracks
            ]

    def reset(self) -> None:
        """Forget cached tracks and frame counter (e.g. before a new input)."""
        with self._lock:
            self._tracks = []
            self._next_id = 0
            self._frame_count = 0
