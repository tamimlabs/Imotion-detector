"""Command-line interface for the Imotion Detector."""

import argparse
import logging
from pathlib import Path

import cv2

from . import __version__
from .detector import FaceEmotionPipeline
from .engines import ENGINES, EngineDependencyError
from .engines.deepface_engine import DeepFaceEngine
from .engines.opencv_engine import OpenCVEngine

logger = logging.getLogger("imotion_detector")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# --------------------------------------------------------------------- #
# Argument parsing                                                       #
# --------------------------------------------------------------------- #
def build_parser():
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
        "--detector-backend",
        default="opencv",
        choices=DeepFaceEngine.BACKENDS,
        help="DeepFace face-detector backend (engine=deepface only)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help="Directory to store the auto-downloaded OpenCV emotion model",
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
        "--verbose", action="store_true", help="Enable debug-level logging"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #
def setup_logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_engines():
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


def resolve_source(raw):
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


def args_hint():
    return "a camera index, image, or video file"


def make_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video: {path}")
    logger.info("Writing video to %s", path)
    return writer


def build_engine(args):
    engine_cls = ENGINES[args.engine]
    if engine_cls is DeepFaceEngine:
        return engine_cls(detector_backend=args.detector_backend)
    if engine_cls is OpenCVEngine:
        return engine_cls(model_dir=args.model_dir)
    return engine_cls()


# --------------------------------------------------------------------- #
# Source runners                                                         #
# --------------------------------------------------------------------- #
def run_webcam(pipeline, camera, output, no_display):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open webcam {camera}.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    logger.info("Press 'q' in the window to quit.")

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
    finally:
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()


def run_video(pipeline, path, output, no_display):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open video: {path}.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
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
                progress = f"{frame_no}/{total}" if total else str(frame_no)
                logger.info("Processed %s frames", progress)
    finally:
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()


def run_image(pipeline, path, output, no_display):
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


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #
def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if args.list_engines:
        print_engines()
        return 0

    try:
        engine = build_engine(args)
    except EngineDependencyError as exc:
        logger.error(str(exc))
        return 1

    pipeline = FaceEmotionPipeline(
        engine, frame_skip=args.frame_skip, min_face_size=args.min_face_size
    )
    logger.info(
        "Engine: %s | frame-skip: %d | min-face-size: %d",
        engine.name,
        pipeline.frame_skip,
        pipeline.min_face_size,
    )

    if args.no_display and not args.output:
        logger.warning(
            "--no-display is set without --output; frames will be processed but not saved."
        )

    kind, target = resolve_source(args.source)
    if kind == "webcam" and target is None:
        target = args.camera

    try:
        if kind == "webcam":
            run_webcam(pipeline, target, args.output, args.no_display)
        elif kind == "video":
            run_video(pipeline, target, args.output, args.no_display)
        else:
            run_image(pipeline, target, args.output, args.no_display)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1
    return 0
