<div align="center">

# 🎭 Imotion Detector

**Real-time facial emotion recognition** — pick the engine that fits your hardware and accuracy needs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.4%2B-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![CI](https://github.com/tamimlabs/Imotion-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/tamimlabs/Imotion-detector/actions)
[![DeepFace](https://img.shields.io/badge/DeepFace-powered-orange?logo=python&logoColor=white)](https://github.com/serengil/deepface)
[![FER](https://img.shields.io/badge/FER-supported-yellow?logo=python&logoColor=white)](https://github.com/justinshenk/fer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

```
┌──────────────────────────────────┐
│  ┌──────────┐   ┌──────────┐     │
│  │  Happy   │   │ Neutral  │     │
│  │  ┌────┐  │   │  ┌────┐  │     │
│  │  │ 🙂 │  │   │  │ 😐│  │     │
│  │  └────┘  │   │  └────┘  │     │
│  └──────────┘   └──────────┘     │
│     Imotion Detector             │
└──────────────────────────────────┘
```

Made with ❤️ by [Tamim Hasan](https://github.com/tamimlabs)

</div>

---

## 📖 Table of Contents

- [✨ Features](#features)
- [🛠️ Supported Engines](#supported-engines)
- [😊 Supported Emotions](#supported-emotions)
- [🚀 Quick Start](#quick-start)
- [📦 Installation](#installation)
- [💻 Usage](#usage)
- [🖥️ HTTP Streaming](#http-streaming)
- [📈 Benchmarking](#benchmarking)
- [⚙️ CLI Reference](#cli-reference)
- [🧠 How It Works](#how-it-works)
- [🧪 Development & Testing](#development--testing)
- [📂 Project Structure](#project-structure)
- [🔧 Customization](#customization)
- [🩺 Troubleshooting](#troubleshooting)
- [❓ FAQ](#faq)
- [🤝 Contributing](#contributing)
- [📄 License](#license)

---

## ✨ Features

- **🎥 Real-time webcam detection** with live emotion labels on every face
- **🖼️ Image & 🎞️ video file support** — analyse any photo or clip
- **🔌 3 pluggable engines** — DeepFace, FER and OpenCV DNN
- **👥 Multi-face support** with **IOU face tracking** — stable labels per person across frames
- **😶 Face detection upgrades** — YuNet deep-learning detector by default (Haar fallback)
- **⚡ Performance tuned** — configurable frame skipping for smooth CPU use
- **🖥️ Headless mode** — batch processing on servers with `--no-display`
- **🌐 HTTP streaming** — live MJPEG feed + JSON API with `--serve` (stdlib, zero extra deps)
- **📊 Benchmark mode** — measure FPS/ms-per-frame with `--benchmark-frames`
- **💾 Save results** — export annotated images or videos with `--output`
- **✅ Tested & typed** — pytest suite + CI, ruff linting and mypy type checks
- **📦 Installable** — run from source or `pip install` the package

---

## 🛠️ Supported Engines

Choose your engine with the `--engine` flag:

| Engine | Accuracy | Speed | Dependencies | Best for |
|--------|----------|-------|--------------|----------|
| [`deepface`](https://github.com/serengil/deepface) ✅ *default* | ⭐⭐⭐ Highest | 🐢 Slower | DeepFace + TensorFlow/PyTorch | Maximum accuracy, GPU users |
| [`fer`](https://github.com/justinshenk/fer) | ⭐⭐ Good | 🐇 Faster | FER + Keras/TensorFlow | Balanced performance on CPU |
| [`opencv`](https://github.com/opencv/opencv_zoo) | ⭐⭐ Good | 🚀 Fastest | OpenCV only | Low-end PCs, minimal install |

> 💡 The OpenCV engine uses the [OpenCV Zoo MobileFaceNet model](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) (~6 MB, auto-downloaded on first use). It labels emotions `fearful` and `surprised`; DeepFace and FER use `fear` and `surprise`.

---

## 😊 Supported Emotions

| Emotion | Meaning |
|---------|---------|
| 😠 Angry | Frowning, tense jaw |
| 🤢 Disgust | Wrinkled nose, disgusted look |
| 😨 Fear | Wide eyes, raised brows |
| 😊 Happy | Smiling, raised cheeks |
| 😐 Neutral | Calm, relaxed face |
| 😢 Sad | Downturned mouth, drooping eyes |
| 😲 Surprise | Wide-open eyes and mouth |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/tamimlabs/Imotion-detector.git
cd Imotion-detector

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install dependencies for your chosen engine
pip install -r requirements/deepface.txt   # highest accuracy (default)
# or
pip install -r requirements/fer.txt        # lightweight
# or
pip install -r requirements/opencv.txt     # lightest (great for low-end PCs)

# 4. Run it! 🎉
python main.py --engine deepface
```

> ⏬ **First run:** model weights download automatically — internet is needed only the first time.
>
> 💻 **Low-end PC?** Use the lightest combo — no model downloads, no big ML libraries:
> `python main.py --engine opencv --face-detector haar`

---

## 📦 Installation

### Option A — Run from source (recommended)

Follow the [Quick Start](#quick-start).

### Option B — Install as a package

```bash
git clone https://github.com/tamimlabs/Imotion-detector.git
cd Imotion-detector
pip install .[deepface]       # DeepFace (default)
# or
pip install .[fer]            # FER
# or
pip install .[all]            # everything

# Run from anywhere:
imotion-detector --engine deepface
```

| Engine | Requirements command |
|--------|----------------------|
| DeepFace (default) | `pip install -r requirements/deepface.txt` |
| FER | `pip install -r requirements/fer.txt` |
| OpenCV | `pip install -r requirements/opencv.txt` |
| Everything | `pip install -r requirements/all.txt` |

> 💡 **DeepFace backend:** DeepFace runs on either [TensorFlow](https://www.tensorflow.org/install) or [PyTorch](https://pytorch.org/). `deepface.txt` installs TensorFlow; swap for `torch` if you prefer (see file comments).

---

## 💻 Usage

### Detect emotions live from your webcam

```bash
python main.py
```

### Analyse a single image

```bash
python main.py --engine fer --source photo.jpg --output result.jpg
```

### Analyse a video file

```bash
python main.py --engine opencv --source clip.mp4 --output annotated.mp4
```

### Headless mode (servers / batch processing)

```bash
python main.py --engine deepface --source clip.mp4 --output out.mp4 --no-display
```

### Stream over HTTP (MJPEG + JSON API)

```bash
python main.py --engine opencv --serve 8000 --source 0
# Open http://localhost:8000 in a browser (live feed)
# GET http://localhost:8000/api/emotions  -> JSON snapshot of tracked faces
# GET http://localhost:8000/stream         -> raw MJPEG stream
```

### Benchmark throughput

```bash
python main.py --engine opencv --source clip.mp4 --benchmark-frames 60
```

### List all available engines

```bash
python main.py --list-engines
```

**Controls:** press `q` in the window to quit, or `Ctrl + C` in the terminal.

---

## 🖥️ HTTP Streaming

Serve the live annotated feed over HTTP so any browser or device on your network can watch it.
Built purely on the Python standard library — **no extra dependencies**.

```bash
python main.py --engine opencv --face-detector haar --serve 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /` | Simple HTML page showing the live `<img src="/stream">` feed |
| `GET /stream` | MJPEG stream (`multipart/x-mixed-replace`) — embed in `<img>` tags |
| `GET /api/emotions` | JSON snapshot of every tracked face: `id`, `box`, `emotion`, `misses` |

Example `curl`:

```bash
curl http://127.0.0.1:8000/api/emotions
# [{"id": 0, "box": [142, 98, 210, 210], "emotion": "happy", "misses": 0}]
```

Use `--host 0.0.0.0` to allow access from other devices. Streaming works with a webcam or a
video file (not a single image).

---

## 📈 Benchmarking

Measure worst-case throughput (every frame is fully analysed — `frame-skip=1`) on any
image/video source or webcam:

```bash
python main.py --engine opencv --face-detector haar --source clip.mp4 --benchmark-frames 60
# Processed 60 frames in 0.38s (opencv engine, frame-skip=1)
#   156.8 FPS  (6.4 ms/frame)
```

Useful for picking the right engine/detector for your hardware before committing to a config.

---

## ⚙️ CLI Reference

Run `python main.py --help` for the complete list.

| Flag | Default | Description |
|------|---------|-------------|
| `--engine {deepface,fer,opencv}` | `deepface` | Emotion analysis engine |
| `--face-detector {yunet,haar}` | `yunet` | Face detector (YuNet is more accurate; Haar needs no model) |
| `--source PATH` | webcam | Camera index, image path, or video path |
| `--camera N` | `0` | Webcam index (when `--source` is not given) |
| `--output PATH` | — | Save the annotated result |
| `--frame-skip N` | `10` | Analyse emotions every N frames |
| `--min-face-size N` | `100` | Minimum face size (px) |
| `--iou-threshold F` | `0.3` | IOU needed to match a face to an existing track |
| `--detector-backend NAME` | `opencv` | DeepFace face-detector backend |
| `--model-dir PATH` | `models/` | Cache directory for auto-downloaded ONNX models |
| `--no-display` | off | Run without a display window |
| `--list-engines` | off | Print available engines and exit |
| `--benchmark-frames N` | off | Benchmark mode: analyse every frame over N frames, print FPS |
| `--serve PORT` | off | Serve a live MJPEG stream + JSON API over HTTP |
| `--host ADDR` | `127.0.0.1` | Bind address used with `--serve` |
| `--verbose` | off | Debug-level logging |
| `--version` | — | Show the installed version |

**Examples:**

```bash
python main.py --engine deepface --detector-backend mtcnn   # stronger face detector
python main.py --engine opencv --frame-skip 5               # faster on low-end PCs
python main.py --engine opencv --face-detector haar         # zero model downloads
python main.py --camera 1                                   # use a second webcam
python main.py --serve 8000 --host 0.0.0.0                  # stream to your LAN
```

---

## 🧠 How It Works

```
Frame (webcam / image / video)
        │
        ▼
Face detection (YuNet DNN by default / Haar fallback)
        │
        ▼
IOU face tracking (greedy matching keeps a stable label per person)
        │
        ▼
Every N frames: emotion analysis per tracked face
   (DeepFace / FER / OpenCV engine)
        │
        ▼
Draw bounding boxes + labels
        │
        ▼
Display, save and/or stream over HTTP
```

1. **Face detection** — by default OpenCV's [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) deep-learning detector (ONNX, ~230 KB, auto-downloaded). Use `--face-detector haar` for the classic [Haar cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html), which needs no model download.
2. **Face tracking** — detections are matched to existing tracks by [Intersection-over-Union](https://en.wikipedia.org/wiki/Jaccard_index) (IOU), so each person keeps a stable label and identity while they move. Faces that disappear for too long are pruned automatically.
3. **Emotion analysis** — each face crop goes to the selected engine, which predicts one of the [7 emotions](#supported-emotions).
4. **Frame skipping** — analysis runs every `N` frames (default `10`), keeping the app responsive; the last prediction is displayed in between.
5. **Annotation** — green bounding boxes and emotion labels are drawn on the frame.

---

## 🧪 Development & Testing

```bash
# One-time dev setup
pip install -r requirements/dev.txt

# Run the test suite (model-download tests auto-skip when offline)
pytest

# Lint
ruff check .

# Type check
mypy imotion_detector main.py tests
```

Tests marked `models` (which download the small ONNX models) and `network` are skipped
automatically when there's no internet access, so the suite works anywhere. CI runs all of
the above on Linux + Windows and Python 3.10 / 3.11 / 3.13.

---

## 📂 Project Structure

```
Imotion-detector/
├── main.py                    # 🚪 CLI entry point
├── imotion_detector/          # 📦 Core package
│   ├── __init__.py
│   ├── __main__.py            # python -m imotion_detector
│   ├── cli.py                 # Argument parsing & runners
│   ├── detector.py            # IOU face tracking + annotation pipeline
│   ├── detectors.py           # YuNet / Haar face detectors
│   ├── server.py              # HTTP streaming server (MJPEG + JSON API)
│   ├── utils.py               # Model download helper
│   └── engines/               # 🔌 Pluggable emotion engines
│       ├── base.py            # Abstract EmotionEngine interface
│       ├── deepface_engine.py # DeepFace backend
│       ├── fer_engine.py      # FER backend
│       └── opencv_engine.py   # OpenCV DNN backend
├── tests/                     # ✅ pytest suite (tracking, CLI, server, utils, models)
├── requirements/              # Per-engine dependency files
│   ├── base.txt
│   ├── dev.txt                # pytest + ruff + mypy
│   ├── deepface.txt
│   ├── fer.txt
│   ├── opencv.txt
│   └── all.txt
├── .github/workflows/ci.yml   # 🚀 CI: lint + typecheck + tests
├── pyproject.toml             # 📦 Packaging, lint & type config
├── models/                    # ⬇️ Auto-downloaded models (gitignored)
├── CONTRIBUTING.md            # 🤝 Contributor guidelines
├── LICENSE                    # 📄 MIT License
└── README.md
```

> 🧩 **Adding your own engine:** implement [`EmotionEngine`](imotion_detector/engines/base.py) (just `analyze(face_bgr)`) and register it in [`engines/__init__.py`](imotion_detector/engines/__init__.py).
> 🧩 **Adding your own face detector:** subclass [`FaceDetector`](imotion_detector/detectors.py) (implement `detect(frame)`) and register it in the [`DETECTORS`](imotion_detector/detectors.py) registry.
> See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🔧 Customization

Use the package directly in your own Python code:

```python
import cv2
from imotion_detector.detector import FaceEmotionPipeline
from imotion_detector.detectors import YuNetFaceDetector
from imotion_detector.engines import FEREngine

pipeline = FaceEmotionPipeline(
    engine=FEREngine(),
    detector=YuNetFaceDetector(min_face_size=80),  # or HaarFaceDetector()
    frame_skip=5,          # analyse every 5 frames
    min_face_size=80,      # accept smaller faces
    iou_threshold=0.3,     # IOU to match faces between frames
    max_misses=30,         # prune tracks after 30 undetected frames
)

frame = cv2.imread("me.jpg")
annotated = pipeline.process(frame)

# JSON-serialisable snapshot of tracked faces (used by the HTTP API)
pipeline.snapshot()  # [{"id": 0, "box": [142, 98, 210, 210], "emotion": "happy", "misses": 0}]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engine` | required | `DeepFaceEngine()`, `FEREngine()` or `OpenCVEngine()` |
| `detector` | `YuNetFaceDetector()` | `YuNetFaceDetector()` or `HaarFaceDetector()` |
| `frame_skip` | `10` | Analyse every N frames (higher = faster) |
| `min_face_size` | `100` | Minimum face size (px) |
| `iou_threshold` | `0.3` | IOU needed to match a face to a tracked face |
| `max_misses` | `30` | Frames a track can be unseen before it is pruned |
| `pad_ratio` | `0.2` | Context padding around crops (helps accuracy) |

---

## 🩺 Troubleshooting

| Problem | Solution |
|---------|----------|
| ❌ Webcam not opening | Close other apps using the camera, or try `--camera 1` |
| ❌ No faces detected | Improve lighting, face the camera, lower `--min-face-size` |
| 🐢 Slow performance | Use `--engine opencv --face-detector haar` or raise `--frame-skip` |
| ⏬ Model download fails | Check your internet connection and retry (first run only) |
| 🔌 `DeepFace is not installed` | Run `pip install -r requirements/deepface.txt` |
| 🔌 `FER is not installed` | Run `pip install -r requirements/fer.txt` |
| 🔀 Labels flicker between faces | Lower `--iou-threshold` so faces keep their identity, or raise `--frame-skip` |
| 🌐 `--serve` shows no stream | Check `--host` is reachable; use `--host 0.0.0.0` for LAN access |
| 💾 Out of memory | Use `--engine opencv` or install a CPU-only TensorFlow build |

Still stuck? [Open an issue](https://github.com/tamimlabs/Imotion-detector/issues/new).

---

## ❓ FAQ

**Which engine should I use?**
- Maximum accuracy → **DeepFace**
- Balanced speed on CPU → **FER**
- Low-end PCs / minimal dependencies → **OpenCV** (+ `--face-detector haar`)

**What is the YuNet face detector?**
OpenCV's modern deep-learning face detector (~230 KB ONNX model). It handles angles, low
light and partial occlusion far better than the old Haar cascade. Both are selectable with
`--face-detector`; Haar needs no model download.

**How does multi-face tracking work?**
Faces are matched between frames by Intersection-over-Union, so each person keeps a stable
label and track id while they move across the frame.

**Do I need a GPU?** No — all engines run on CPU, though DeepFace benefits most from a GPU.

**Are models included in the repo?** No. They download automatically on first run to keep the repo small.

**Can I watch the feed in a browser?** Yes — `python main.py --engine opencv --serve 8000`
serves a live MJPEG stream plus a JSON API, with zero extra dependencies.

**Does it work on multiple faces?** Yes — all faces in the frame are detected, tracked and analysed.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. 🍴 Fork the [repository](https://github.com/tamimlabs/Imotion-detector)
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4. ✅ Open a [pull request](https://github.com/tamimlabs/Imotion-detector/pulls)

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE).

---

<div align="center">

Made with ❤️ using [OpenCV](https://opencv.org/), [DeepFace](https://github.com/serengil/deepface) & [FER](https://github.com/justinshenk/fer)

⭐ If you find this project useful, please give it a star! and support me

</div>
