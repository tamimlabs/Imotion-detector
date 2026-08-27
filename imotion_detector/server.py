"""HTTP streaming server — serves a live MJPEG feed and a JSON emotion API.

Built entirely on the standard library so the server adds no extra
dependencies: the emotion pipeline runs on a background capture thread and
pushes JPEG frames into a bounded ring buffer that HTTP clients poll.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import cv2
import numpy as np

from .detector import FaceEmotionPipeline

logger = logging.getLogger(__name__)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Imotion Detector</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee;
         display: flex; flex-direction: column; align-items: center; gap: 16px; }
  img  { max-width: 100%; height: auto; border-radius: 8px; }
  a    { color: #4caf50; }
</style>
</head>
<body>
<h1>Imotion Detector</h1>
<img src="/stream" alt="Live emotion feed">
<p>Emotion labels appear on the live feed. See <a href="/api/emotions">/api/emotions</a>
for a machine-readable JSON snapshot.</p>
</body>
</html>
"""

_BOUNDARY = "frame"
_EOF = object()  # sentinel pushed when the capture source ends


class _StreamingHandler(BaseHTTPRequestHandler):
    server_version = "ImotionStream/1.0"

    # ------------------------------------------------------------------ #
    # Routing                                                             #
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/":
            self._serve_index()
        elif self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/api/emotions":
            self._serve_emotions()
        else:
            self.send_error(404, "Not Found")

    # ------------------------------------------------------------------ #
    # Handlers                                                            #
    # ------------------------------------------------------------------ #
    def _serve_index(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _server(self) -> EmotionStreamServer:
        return cast(EmotionStreamServer, self.server)

    def _serve_emotions(self) -> None:
        body = json.dumps(self._server().snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_id = -1
        try:
            while True:
                frame_id, payload = self._server().wait_frame(last_id)
                if payload is _EOF:
                    break
                last_id = frame_id
                jpeg = cast(bytes, payload)
                self.wfile.write(
                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client went away mid-stream

    def log_message(self, fmt: str, *args: object) -> None:  # keep the console clean
        logger.debug(fmt, *args)


class EmotionStreamServer(ThreadingHTTPServer):
    """Runs the pipeline on a background thread and serves the result over HTTP.

    ``read_frame`` is a zero-argument callable returning a BGR frame or
    ``None`` when the source ends (e.g. ``lambda: cap.read()[1]``).
    """

    daemon_threads = True

    def __init__(
        self,
        host: str,
        port: int,
        pipeline: FaceEmotionPipeline,
        read_frame: Callable[[], np.ndarray[Any, np.dtype[Any]] | None],
        jpeg_quality: int = 80,
        max_frames: int = 300,
    ) -> None:
        super().__init__((host, port), _StreamingHandler)
        self.pipeline = pipeline
        self.read_frame = read_frame
        self.jpeg_quality = jpeg_quality

        self._frames: deque[tuple[int, object]] = deque(maxlen=max_frames)
        self._frame_seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._serving = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background capture and HTTP threads."""
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="imotion-capture", daemon=True
        )
        self._capture_thread.start()

        self._http_thread = threading.Thread(
            target=self._serve_loop, name="imotion-http", daemon=True
        )
        self._http_thread.start()

    def serve(self) -> None:
        """Block the calling thread until :meth:`stop` is called."""
        self._serving.wait()

    def _serve_loop(self) -> None:
        try:
            with suppress(OSError):  # socket closed by stop()
                self.serve_forever()
        except Exception:
            logger.exception("HTTP server error")
        finally:
            self._serving.set()

    def stop(self) -> None:
        """Stop capture, flush clients and shut the server down (idempotent)."""
        if not self._stop.is_set():
            self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        with suppress(Exception):  # closing the socket makes the serve loop exit
            self.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2)
        self._serving.set()  # unblock serve() if the HTTP thread did not exit cleanly

    # ------------------------------------------------------------------ #
    # Capture loop                                                        #
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.read_frame()
            if frame is None:
                break
            self.pipeline.process(frame)
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                continue
            with self._cond:
                self._frame_seq += 1
                self._frames.append((self._frame_seq, buf.tobytes()))
                self._cond.notify_all()
        with self._cond:
            self._frames.append((self._frame_seq + 1, _EOF))
            self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # Shared state for handlers                                           #
    # ------------------------------------------------------------------ #
    def wait_frame(self, last_id: int) -> tuple[int, object]:
        """Block until a frame newer than ``last_id`` exists.

        Returns ``(frame_id, jpeg_bytes)``, or ``(id, _EOF)`` when the capture
        source has ended. Oldest frames are dropped for slow clients so the
        feed always stays close to real-time.
        """
        with self._cond:
            while True:
                if self._frames and self._frames[-1][0] > last_id:
                    for frame_id, payload in reversed(self._frames):
                        if payload is not _EOF and frame_id > last_id:
                            return frame_id, payload
                for frame_id, payload in self._frames:
                    if payload is _EOF and frame_id > last_id:
                        return frame_id, payload
                if self._stop.is_set():
                    return last_id + 1, _EOF
                self._cond.wait(timeout=0.1)

    def snapshot(self) -> list[dict[str, Any]]:
        return self.pipeline.snapshot()
