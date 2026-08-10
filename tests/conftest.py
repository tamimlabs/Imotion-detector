"""Shared pytest fixtures and markers.

Tests marked ``network`` or ``models`` need internet access (model downloads).
When the machine is offline those tests are skipped automatically so the suite
still passes on a plane or in a firewalled CI runner.
"""

from __future__ import annotations

import socket

import pytest


def _has_network() -> bool:
    try:
        with socket.create_connection(("github.com", 443), timeout=3):
            return True
    except OSError:
        return False


HAS_NETWORK = _has_network()


def pytest_collection_modifyitems(items):
    if HAS_NETWORK:
        return
    skip = pytest.mark.skip(reason="no network access")
    for item in items:
        if any(marker.name in ("network", "models") for marker in item.iter_markers()):
            item.add_marker(skip)


@pytest.fixture
def fake_engine():
    class FakeEngine:
        name = "fake"

        def analyze(self, face_bgr):
            return "happy"

    return FakeEngine()


@pytest.fixture
def fake_detector():
    class FakeDetector:
        name = "fake"

        def __init__(self):
            self.boxes = []

        def detect(self, frame):
            return list(self.boxes)

    return FakeDetector()
