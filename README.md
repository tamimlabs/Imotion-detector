<div align="center">

# 🎭 Imotion Detector

**Real-time facial emotion recognition** — pick the engine that fits your hardware and accuracy needs.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)](https://opencv.org/)
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
- [⚙️ CLI Reference](#cli-reference)
- [🧠 How It Works](#how-it-works)
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
- **👥 Multi-face support** — every face in the frame is tracked
- **⚡ Performance tuned** — configurable frame skipping for smooth CPU use
- **🖥️ Headless mode** — batch processing on servers with `--no-display`
- **💾 Save results** — export annotated images or videos with `--output`
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
pip install -r requirements/opencv.txt     # lightest

# 4. Run it! 🎉
python main.py --engine deepface
```

> ⏬ **First run:** model weights download automatically — internet is needed only the first time.

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

### List all available engines

```bash
python main.py --list-engines
```

**Controls:** press `q` in the window to quit, or `Ctrl + C` in the terminal.

---

## ⚙️ CLI Reference

Run `python main.py --help` for the complete list.

| Flag | Default | Description |
|------|---------|-------------|
| `--engine {deepface,fer,opencv}` | `deepface` | Emotion analysis engine |
| `--source PATH` | webcam | Camera index, image path, or video path |
| `--camera N` | `0` | Webcam index (when `--source` is not given) |
| `--output PATH` | — | Save the annotated result |
| `--frame-skip N` | `10` | Analyse emotions every N frames |
| `--min-face-size N` | `100` | Minimum face size (px) |
| `--detector-backend NAME` | `opencv` | DeepFace face-detector backend |
| `--model-dir PATH` | `models/` | Cache directory for the OpenCV model |
| `--no-display` | off | Run without a display window |
| `--list-engines` | off | Print available engines and exit |
| `--verbose` | off | Debug-level logging |
| `--version` | — | Show the installed version |

**Examples:**

```bash
python main.py --engine deepface --detector-backend mtcnn   # stronger face detector
python main.py --engine opencv --frame-skip 5               # faster on low-end PCs
python main.py --camera 1                                   # use a second webcam
```

---

## 🧠 How It Works

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
2. **Emotion analysis** — each face crop goes to the selected engine, which predicts one of the [7 emotions](#supported-emotions).
3. **Frame skipping** — analysis runs every `N` frames (default `10`), keeping the app responsive; the last prediction is displayed in between.
4. **Annotation** — green bounding boxes and emotion labels are drawn on the frame.

---

## 📂 Project Structure

```
Imotion-detector/
├── main.py                    # 🚪 CLI entry point
├── imotion_detector/          # 📦 Core package
│   ├── __init__.py
│   ├── __main__.py            # python -m imotion_detector
│   ├── cli.py                 # Argument parsing & runners
│   ├── detector.py            # Face detection + annotation pipeline
│   ├── utils.py               # Model download helper
│   └── engines/               # 🔌 Pluggable emotion engines
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
├── pyproject.toml             # 📦 Packaging config
├── models/                    # ⬇️ Auto-downloaded models (gitignored)
├── CONTRIBUTING.md            # 🤝 Contributor guidelines
├── LICENSE                    # 📄 MIT License
└── README.md
```

> 🧩 **Adding your own engine:** implement [`EmotionEngine`](imotion_detector/engines/base.py) (just `analyze(face_bgr)`) and register it in [`engines/__init__.py`](imotion_detector/engines/__init__.py). See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🔧 Customization

Use the package directly in your own Python code:

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
| `pad_ratio` | `0.2` | Context padding around crops (helps accuracy) |

---

## 🩺 Troubleshooting

| Problem | Solution |
|---------|----------|
| ❌ Webcam not opening | Close other apps using the camera, or try `--camera 1` |
| ❌ No faces detected | Improve lighting, face the camera, lower `--min-face-size` |
| 🐢 Slow performance | Use `--engine opencv` or raise `--frame-skip` |
| ⏬ Model download fails | Check your internet connection and retry (first run only) |
| 🔌 `DeepFace is not installed` | Run `pip install -r requirements/deepface.txt` |
| 🔌 `FER is not installed` | Run `pip install -r requirements/fer.txt` |
| 💾 Out of memory | Use `--engine opencv` or install a CPU-only TensorFlow build |

Still stuck? [Open an issue](https://github.com/tamimlabs/Imotion-detector/issues/new).

---

## ❓ FAQ

**Which engine should I use?**
- Maximum accuracy → **DeepFace**
- Balanced speed on CPU → **FER**
- Low-end PCs / minimal dependencies → **OpenCV**

**Do I need a GPU?** No — all engines run on CPU, though DeepFace benefits most from a GPU.

**Are models included in the repo?** No. They download automatically on first run to keep the repo small.

**Does it work on multiple faces?** Yes — all faces in the frame are detected and tracked.

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

⭐ If you find this project useful, please give it a star!

</div>
