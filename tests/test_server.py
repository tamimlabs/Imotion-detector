"""End-to-end tests for the HTTP streaming server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from imotion_detector.server import EmotionStreamServer


class StubPipeline:
    def __init__(self):
        self.frames_processed = 0

    def process(self, frame):
        self.frames_processed += 1
        return frame

    def snapshot(self):
        return [{"id": 0, "box": [10, 10, 80, 80], "emotion": "happy", "misses": 0}]


@pytest.fixture
def live_server():
    stub = StubPipeline()
    frames = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(20)]

    def read_frame():
        if not frames:
            return None
        time.sleep(0.02)
        return frames.pop(0)

    server = EmotionStreamServer("127.0.0.1", 0, stub, read_frame)
    server.start()

    deadline = time.time() + 5
    while stub.frames_processed == 0 and time.time() < deadline:
        time.sleep(0.02)

    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, stub
    finally:
        server.stop()


class TestServer:
    def test_index_page(self, live_server):
        base, _ = live_server
        with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read()
        assert b"Imotion Detector" in body
        assert b"/stream" in body

    def test_emotions_json(self, live_server):
        base, _ = live_server
        with urllib.request.urlopen(f"{base}/api/emotions", timeout=5) as resp:
            assert resp.headers.get("Content-Type") == "application/json"
            data = json.loads(resp.read())
        assert data == [{"id": 0, "box": [10, 10, 80, 80], "emotion": "happy", "misses": 0}]

    def test_mjpeg_stream(self, live_server):
        base, _ = live_server
        with urllib.request.urlopen(f"{base}/stream", timeout=5) as resp:
            assert resp.headers.get("Content-Type").startswith(
                "multipart/x-mixed-replace"
            )
            chunk = resp.read(128)
        assert b"--frame" in chunk

    def test_processes_captured_frames(self, live_server):
        base, stub = live_server
        time.sleep(0.1)
        assert stub.frames_processed > 0

    def test_unknown_route_is_404(self, live_server):
        base, _ = live_server
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{base}/nope", timeout=5)
        assert excinfo.value.code == 404

    def test_server_closes_cleanly(self):
        stub = StubPipeline()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8), None]

        def read_frame():
            return frames.pop(0) if frames else None

        server = EmotionStreamServer("127.0.0.1", 0, stub, read_frame)
        server.start()
        server.stop()  # must not hang or raise
