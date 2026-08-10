<div align="center">

# Imotion Detector

**Real-time facial emotion recognition** with three pluggable engines — pick the one that fits your hardware and accuracy needs.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Author: [Tamim Hasan](https://github.com/shahmdtamimhasan)

</div>

---

## Table of Contents

- [Features](#features)
- [Supported Engines](#supported-engines)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [CLI Reference](#cli-reference)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Use as a Library](#use-as-a-library)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Real-time webcam detection** — live emotion labels overlaid on each face
- **Image & video support** — analyse any photo or clip from disk
- **3 pluggable engines** — DeepFace, FER and OpenCV DNN
- **Multi-face support** — every face in the frame is tracked individually
- **Performance tuning** — configurable frame skipping keeps it responsive on CPU
- **Headless mode** — process files on servers with `--no-display`
- **Save results** — export annotated images or videos with `--output`
- **Installable** — run from source or `pip install` the package

---

## Supported Engines

Choose an engine with the `--engine` flag:

| Engine | Accuracy | Speed | Dependencies | Best for |
|--------|----------|-------|--------------|----------|
| [`deepface`](https://github.com/serengil/deepface) _(default)_ | Highest | Slower | DeepFace + TensorFlow/PyTorch | Maximum accuracy, GPU users |
| [`fer`](https://github.com/justinshenk/fer) | Good | Faster | FER + Keras/TensorFlow | Balanced performance on CPU |
| [`opencv`](https://github.com/opencv/opencv_zoo) | Good | Fastest | OpenCV only | Low-end PCs, minimal install |

All three engines recognise the same seven emotions:

| Emotion | Emotion |
|---------|---------|
| Angry | Happy |
| Disgust | Neutral |
| Fear | Sad |
| Surprise | |

> The OpenCV engine uses the [OpenCV Zoo MobileFaceNet model](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) (~6 MB), downloaded automatically on first use. It labels emotions as `fearful` and `surprised`; DeepFace and FER use `fear` and `surprise`.

---

## Requirements

- **Python 3.8+**
- A webcam (for live detection)
- At least the [base dependencies](requirements/base.txt): `opencv-python` and `numpy`

---

## Installation

```bash
git clone https://github.com/shahmdtamimhasan/Imotion-detector.git
cd Imotion-detector

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
```

Install the dependencies for your chosen engine:

| Engine | Command |
|--------|---------|
| DeepFace (default) | `pip install -r requirements/deepface.txt` |
| FER | `pip install -r requirements/fer.txt` |
| OpenCV | `pip install -r requirements/opencv.txt` |
| All engines | `pip install -r requirements/all.txt` |

Or install the package itself:

```bash
pip install .[deepface]   # or: pip install .[fer]  |  pip install .[all]
imotion-detector --engine deepface
```

> **Note:** DeepFace works with either [TensorFlow](https://www.tensorflow.org/install) or [PyTorch](https://pytorch.org/). The `deepface.txt` file installs TensorFlow — swap it for `torch` if you prefer (see the comments in the file).
>
> **First run:** model weights download automatically. An internet connection is needed only on the first run.

---

## Usage

### Live webcam detection

```bash
python main.py
```

### Analyse an image

```bash
python main.py --engine fer --source photo.jpg --output result.jpg
```

### Analyse a video

```bash
python main.py --engine opencv --source clip.mp4 --output annotated.mp4
```

### Headless batch processing (no display window)

```bash
python main.py --engine deepface --source clip.mp4 --output out.mp4 --no-display
```

### List the available engines

```bash
python main.py --list-engines
```

**Controls:** press `q` in the video window to quit, or `Ctrl + C` in the terminal.

---

## CLI Reference

Run `python main.py --help` for the complete list.

| Flag | Default | Description |
|------|---------|-------------|
| `--engine {deepface,fer,opencv}` | `deepface` | Emotion analysis engine |
| `--source PATH` | webcam | Camera index, image path, or video path |
| `--camera N` | `0` | Webcam index (used when `--source` is not given) |
| `--output PATH` | — | Save the annotated result to this file |
| `--frame-skip N` | `10` | Run emotion analysis every N frames |
| `--min-face-size N` | `100` | Minimum face width/height (px) for detection |
| `--detector-backend NAME` | `opencv` | DeepFace face-detector backend |
| `--model-dir PATH` | `models/` | Directory for the auto-downloaded OpenCV model |
| `--no-display` | off | Run without a display window (headless) |
| `--list-engines` | off | Print available engines and exit |
| `--verbose` | off | Debug-level logging |
| `--version` | — | Show the installed version |

Examples:

```bash
python main.py --engine deepface --detector-backend mtcnn   # stronger face detector
python main.py --engine opencv --frame-skip 5               # faster on low-end PCs
python main.py --camera 1                                   # use a second webcam
```

---

## How It Works

```
Frame (webcam / image / video)
        │
        ▼
Convert to grayscale
        │
        ▼
Face detection (OpenCV Haar cascade)
        │
        ▼
Every N frames: emotion analysis per face
   (DeepFace / FER / OpenCV engine)
        │
        ▼
Draw bounding boxes + labels
        │
        ▼
Display and/or save
```

1. **Face detection** — [OpenCV's Haar cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) locates faces in the grayscale frame.
2. **Emotion analysis** — each face crop is passed to the selected engine, which predicts one of the seven emotions.
3. **Frame skipping** — analysis runs every `N` frames (default `10`) so the app stays responsive; the last prediction is kept and shown between runs.
4. **Annotation** — green bounding boxes and emotion labels are drawn on the frame.

---

## Project Structure

```
Imotion-detector/
├── main.py                    # CLI entry point
├── imotion_detector/          # Core package
│   ├── __init__.py
│   ├── __main__.py            # python -m imotion_detector
│   ├── cli.py                 # Argument parsing and source runners
│   ├── detector.py            # Face detection + annotation pipeline
│   ├── utils.py               # Model download helper
│   └── engines/               # Pluggable emotion engines
│       ├── base.py            # Abstract EmotionEngine interface
│       ├── deepface_engine.py # DeepFace backend
│       ├── fer_engine.py      # FER backend
│       └── opencv_engine.py   # OpenCV DNN backend
├── requirements/              # Per-engine dependency files
│   ├── base.txt
│   ├── deepface.txt
│   ├── fer.txt
│   ├── opencv.txt
│   └── all.txt
├── pyproject.toml             # Package metadata and build config
├── models/                    # Auto-downloaded models (gitignored)
├── AGENTS.md                  # Guidelines for AI-assisted contributors
├── CONTRIBUTING.md            # Guidelines for contributors
├── LICENSE                    # MIT License
└── README.md
```

---

## Use as a Library

```python
import cv2
from imotion_detector.detector import FaceEmotionPipeline
from imotion_detector.engines import FEREngine

pipeline = FaceEmotionPipeline(
    engine=FEREngine(),
    frame_skip=5,          # analyse every 5 frames
    min_face_size=80,      # accept smaller faces
    scale_factor=1.1,      # cascade scale factor
)

frame = cv2.imread("me.jpg")
annotated = pipeline.process(frame)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engine` | required | `DeepFaceEngine()`, `FEREngine()` or `OpenCVEngine()` |
| `frame_skip` | `10` | Analyse every N frames (higher = faster) |
| `min_face_size` | `100` | Minimum face size (px) |
| `scale_factor` | `1.1` | Cascade detection scale factor |
| `min_neighbors` | `5` | Cascade detection neighbours |
| `pad_ratio` | `0.2` | Context padding around face crops (helps accuracy) |

To add a new engine, implement [`EmotionEngine`](imotion_detector/engines/base.py) (a single `analyze(face_bgr)` method) and register it in [`engines/__init__.py`](imotion_detector/engines/__init__.py).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not opening | Close other apps using the camera, or try `--camera 1` |
| No faces detected | Improve lighting, face the camera, lower `--min-face-size` |
| Slow performance | Use `--engine opencv` or raise `--frame-skip` |
| Model download fails | Check your internet connection and retry (first run only) |
| `DeepFace is not installed` | Run `pip install -r requirements/deepface.txt` |
| `FER is not installed` | Run `pip install -r requirements/fer.txt` |
| Out of memory | Use `--engine opencv`, or install a CPU-only TensorFlow build |

Still stuck? [Open an issue](https://github.com/shahmdtamimhasan/Imotion-detector/issues/new).

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE).

---

<div align="center">

Made with ❤️ by [Tamim Hasan](https://github.com/shahmdtamimhasan) using [OpenCV](https://opencv.org/), [DeepFace](https://github.com/serengil/deepface) and [FER](https://github.com/justinshenk/fer).

</div>
