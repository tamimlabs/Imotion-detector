"""Command-line interface for the Imotion Detector."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from . import __version__
from .detector import FaceEmotionPipeline
from .detectors import build_detector, list_detectors
from .engines import ENGINES, EmotionEngine, EngineDependencyError
from .engines.deepface_engine import DeepFaceEngine
from .engines.opencv_engine import OpenCVEngine

logger = logging.getLogger("imotion_detector")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# --------------------------------------------------------------------- #
# Argument parsing                                                       #
# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imotion-detector",
        description=(
            "Real-time facial emotion detection with pluggable engines: "
            "DeepFace, FER and OpenCV DNN."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        choices=sorted(ENGINES),
        default="deepface",
        help="Emotion analysis engine to use",
    )
    parser.add_argument(
        "--face-detector",
        choices=list_detectors(),
        default="yunet",
        help="Face detector backend (yunet is more accurate, haar needs no model)",
    )
    parser.add_argument(
        "--source",
        default=None,
        metavar="PATH",
        help=(
            "Camera index, image path or video path. Defaults to the webcam. "
            "Examples: --source 0, --source photo.jpg, --source clip.mp4"
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index used when --source is not provided",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save the annotated result to an image or video file",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Run emotion analysis every N frames",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=100,
        help="Minimum face width/height (pixels) for face detection",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IOU required to match a face to a tracked face between frames",
    )
    parser.add_argument(
        "--detector-backend",
        default="opencv",
        choices=DeepFaceEngine.BACKENDS,
        help="DeepFace face-detector backend (engine=deepface only)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help="Directory to store the auto-downloaded ONNX models",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a display window (headless/server use)",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List the available engines and exit",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=None,
        metavar="N",
        help="Run in benchmark mode over N frames (every frame analysed) and exit",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=None,
        metavar="PORT",
        help="Serve a live MJPEG stream + JSON API over HTTP on the given port",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address used with --serve",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug-level logging"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #
def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_engines() -> None:
    width = max(len(cls.name) for cls in ENGINES.values())
    print("Available engines:\n")
    for cls in ENGINES.values():
        print(f"  {cls.name:<{width}}  {cls.description}")
    print(
        "\nInstall the dependencies you need, e.g.:\n"
        "  pip install -r requirements/deepface.txt\n"
        "  pip install -r requirements/fer.txt\n"
        "  pip install -r requirements/opencv.txt"
    )


def resolve_source(raw: str | None) -> tuple[str, int | Path | None]:
    """Return (kind, target) where kind is 'webcam', 'image' or 'video'."""
    if raw is None:
        return "webcam", None
    text = str(raw)
    if text.isdigit() or text.lower() == "webcam":
        return "webcam", (int(text) if text.isdigit() else None)
    path = Path(text)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image", path
        if suffix in VIDEO_EXTENSIONS:
            return "video", path
        raise SystemExit(f"Unsupported file type: {suffix} (use {args_hint()})")
    raise SystemExit(f"Source not found: {text}")


def args_hint() -> str:
    return "a camera index, image, or video file"


def make_writer(path: str | Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video: {path}")
    logger.info("Writing video to %s", path)
    return writer


def build_engine(args: argparse.Namespace) -> EmotionEngine:
    engine_cls = ENGINES[args.engine]
    if engine_cls is DeepFaceEngine:
        return engine_cls(detector_backend=args.detector_backend)
    if engine_cls is OpenCVEngine:
        return engine_cls(model_dir=args.model_dir)
    return engine_cls()


def build_pipeline(args: argparse.Namespace) -> FaceEmotionPipeline:
    engine = build_engine(args)
    detector = build_detector(
        args.face_detector, model_dir=args.model_dir, min_face_size=args.min_face_size
    )
    return FaceEmotionPipeline(
        engine,
        detector=detector,
        frame_skip=args.frame_skip,
        min_face_size=args.min_face_size,
        iou_threshold=args.iou_threshold,
    )


def open_capture(source: int | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open source: {source}.")
    return cap


def print_pipeline_info(pipeline: FaceEmotionPipeline) -> None:
    logger.info(
        "Engine: %s | detector: %s | frame-skip: %d | min-face-size: %d",
        pipeline.engine.name,
        pipeline.detector.name,
        pipeline.frame_skip,
        pipeline.min_face_size,
    )


# --------------------------------------------------------------------- #
# Runners                                                                #
# --------------------------------------------------------------------- #
def run_capture(
    pipeline: FaceEmotionPipeline,
    cap: cv2.VideoCapture,
    output: str | None,
    no_display: bool,
    total_frames: int | None = None,
) -> None:
    """Process frames from an open capture until it ends or 'q' is pressed."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_no = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = pipeline.process(frame)
            if output and writer is None:
                writer = make_writer(output, fps, (frame.shape[1], frame.shape[0]))
            if writer:
                writer.write(frame)
            if not no_display:
                cv2.imshow("Imotion Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_no += 1
            if no_display and frame_no % 30 == 0:
                progress = f"{frame_no}/{total_frames}" if total_frames else str(frame_no)
                logger.info("Processed %s frames", progress)
    finally:
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()


def run_webcam(
    pipeline: FaceEmotionPipeline,
    camera: int,
    output: str | None,
    no_display: bool,
) -> None:
    cap = open_capture(camera)
    logger.info("Press 'q' in the window to quit.")
    run_capture(pipeline, cap, output, no_display)


def run_video(
    pipeline: FaceEmotionPipeline,
    path: Path,
    output: str | None,
    no_display: bool,
) -> None:
    cap = open_capture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    run_capture(pipeline, cap, output, no_display, total_frames=total)


def run_image(
    pipeline: FaceEmotionPipeline,
    path: Path,
    output: str | None,
    no_display: bool,
) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Error: could not read image: {path}.")
    pipeline.reset()
    frame = pipeline.process(frame)

    if output:
        cv2.imwrite(str(output), frame)
        logger.info("Saved annotated image to %s", output)

    if not no_display:
        cv2.imshow("Imotion Detector", frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_benchmark(
    pipeline: FaceEmotionPipeline, kind: str, target: int | Path, frames: int
) -> int:
    """Measure worst-case throughput: every frame is fully analysed."""
    cap = open_capture(target)
    processed = 0
    start = time.perf_counter()
    try:
        while processed < frames:
            ret, frame = cap.read()
            if not ret:
                break
            pipeline.process(frame)
            processed += 1
    finally:
        cap.release()

    elapsed = time.perf_counter() - start
    if processed == 0:
        raise SystemExit("Benchmark failed: no frames were read from the source.")
    ms_per_frame = elapsed / processed * 1000
    print(
        f"Processed {processed} frames in {elapsed:.2f}s "
        f"({pipeline.engine.name} engine, frame-skip=1)\n"
        f"  {processed / elapsed:.1f} FPS  ({ms_per_frame:.1f} ms/frame)"
    )
    return 0


def run_serve(
    pipeline: FaceEmotionPipeline, kind: str, target: int | Path, host: str, port: int
) -> int:
    from .server import EmotionStreamServer

    if kind == "image":
        raise SystemExit("--serve works with a webcam or video file, not an image.")

    cap = open_capture(target)
    server = EmotionStreamServer(
        host, port, pipeline, read_frame=lambda: cap.read()[1]
    )
    server.start()
    try:
        logger.info("Streaming at http://%s:%d (Ctrl+C to stop)", host, server.server_address[1])
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        cap.release()
    return 0


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if args.list_engines:
        print_engines()
        return 0

    try:
        pipeline = build_pipeline(args)
    except EngineDependencyError as exc:
        logger.error(str(exc))
        return 1
    print_pipeline_info(pipeline)

    if args.no_display and not args.output and not args.serve and not args.benchmark_frames:
        logger.warning(
            "--no-display is set without --output; frames will be processed but not saved."
        )

    kind, target = resolve_source(args.source)
    if kind == "webcam" and target is None:
        target = args.camera

    if args.benchmark_frames:
        assert target is not None
        benchmark_pipeline = FaceEmotionPipeline(
            pipeline.engine,
            detector=pipeline.detector,
            frame_skip=1,
            min_face_size=pipeline.min_face_size,
            iou_threshold=pipeline.iou_threshold,
        )
        return run_benchmark(benchmark_pipeline, kind, target, args.benchmark_frames)

    if args.serve:
        assert target is not None
        return run_serve(pipeline, kind, target, args.host, args.serve)

    try:
        if kind == "webcam":
            assert isinstance(target, int)
            run_webcam(pipeline, target, args.output, args.no_display)
        elif kind == "video":
            assert isinstance(target, Path)
            run_video(pipeline, target, args.output, args.no_display)
        else:
            assert isinstance(target, Path)
            run_image(pipeline, target, args.output, args.no_display)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1
    return 0
