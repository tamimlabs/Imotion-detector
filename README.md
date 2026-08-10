<div align="center">

# 🎭 Imotion Detector

**Real-time facial emotion recognition** with three pluggable deep-learning engines — pick the one that fits your hardware and accuracy needs.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/shahmdtamimhasan/Imotion-detector?style=for-the-badge)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Stars](https://img.shields.io/github/stars/shahmdtamimhasan/Imotion-detector?style=for-the-badge)](https://github.com/shahmdtamimhasan/Imotion-detector)
[![Forks](https://img.shields.io/github/forks/shahmdtamimhasan/Imotion-detector?style=for-the-badge)](https://github.com/shahmdtamimhasan/Imotion-detector)

```
┌──────────────────────────────────┐
│  ┌──────────┐   ┌──────────┐     │
│  │  Happy   │   │ Neutral  │     │
│  │  ┌────┐  │   │  ┌────┐  │     │
│  │  │ 🙂 │  │   │  │ 😐 │  │     │
│  │  └────┘  │   │  └────┘  │     │
│  └──────────┘   └──────────┘     │
│     Imotion Detector              │
└──────────────────────────────────┘
```

Made with ❤️ by [Tamim Hasan](https://github.com/shahmdtamimhasan)

</div>

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🛠️ Supported Engines](#️-supported-engines)
- [😊 Supported Emotions](#-supported-emotions)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [💻 Usage](#-usage)
- [⚙️ CLI Reference](#️-cli-reference)
- [🧠 How It Works](#-how-it-works)
- [📂 Project Structure](#-project-structure)
- [🔧 Customization](#-customization)
- [🩺 Troubleshooting](#-troubleshooting)
- [❓ FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- **🎥 Real-time webcam detection** with live emotion labels overlaid on each face
- **🖼️ Image & 🎞️ video file support** — analyse any photo or clip
- **🔌 3 pluggable engines** — DeepFace, FER, and OpenCV DNN (see [comparison](#️-supported-engines))
- **👥 Multi-face support** — every face in the frame is tracked individually
- **⚡ Performance tuned** — configurable frame skipping keeps it responsive even on CPU
- **🖥️ Headless mode** — process batches on servers with `--no-display`
- **📁 Save results** — export annotated images or videos with `--output`
- **📦 Installable** — `pip install` the package or just run it from source

---

## 🛠️ Supported Engines

Pick the engine that matches your needs with the `--engine` flag:

| Engine | Accuracy | Speed | Size / Deps | Best for |
|--------|----------|-------|-------------|----------|
| [`deepface`](https://github.com/serengil/deepface) ✅ *default* | ⭐⭐⭐ Highest | 🐢 Slower | ~700 MB + [TensorFlow](https://www.tensorflow.org/) | Best accuracy, GPU users |
| [`fer`](https://github.com/justinshenk/fer) | ⭐⭐ Good | 🐇 Faster | ~400 MB + Keras/TensorFlow | Balanced on CPU |
| [`opencv`](https://github.com/opencv/opencv_zoo) | ⭐⭐ Good | 🚀 Fastest | ~6 MB model, [OpenCV](https://opencv.org/) only | Lightweight / low-end PCs |

> **OpenCV engine note:** it uses the [OpenCV Zoo MobileFaceNet model](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) which is downloaded automatically on first use (~6 MB).

**Comparison details:**

| Engine | Backend | Emotion labels |
|--------|---------|----------------|
| DeepFace | Deep neural network | angry, disgust, fear, happy, sad, surprise, neutral |
| FER | Keras + OpenCV | angry, disgust, fear, happy, sad, surprise, neutral |
| OpenCV | MobileFaceNet (ONNX) | angry, disgust, fearful, happy, neutral, sad, surprised |

---

## 😊 Supported Emotions

| Emotion | Example |
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
# 1. Clone the repo
git clone https://github.com/shahmdtamimhasan/Imotion-detector.git
cd Imotion-detector

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install dependencies for your chosen engine
pip install -r requirements/deepface.txt    # highest accuracy (default)
# or
pip install -r requirements/fer.txt         # lightweight
# or
pip install -r requirements/opencv.txt      # lightest

# 4. Run it! 🎉
python main.py --engine deepface
```

> **First run:** DeepFace and FER download their model weights automatically. The OpenCV engine downloads its model to `models/`. An internet connection is required only on the first run.

---

## 📦 Installation

### Option A — Run from source (recommended)

Follow the [Quick Start](#-quick-start) above.

### Option B — Install as a package

```bash
git clone https://github.com/shahmdtamimhasan/Imotion-detector.git
cd Imotion-detector
pip install .                      # installs base + deepface extras
# or with extra engines:
pip install .[deepface]            # DeepFace
pip install .[fer]                 # FER
pip install .[all]                 # everything

# Now run from anywhere:
imotion-detector --engine deepface
```

### Which requirements do I need?

| Engine | Command |
|--------|---------|
| DeepFace (default) | `pip install -r requirements/deepface.txt` |
| FER | `pip install -r requirements/fer.txt` |
| OpenCV | `pip install -r requirements/opencv.txt` |
| Everything | `pip install -r requirements/all.txt` |

> 💡 **DeepFace backend:** DeepFace can run on either [TensorFlow](https://www.tensorflow.org/install) or [PyTorch](https://pytorch.org/). The `deepface.txt` requirements install TensorFlow. For PyTorch, install `torch` instead — see the comments in `requirements/deepface.txt`.

---

## 💻 Usage

### Detect emotions live from your webcam

```bash
python main.py
```

You can also be explicit:

```bash
python main.py --engine deepface --source 0
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

No display window is opened — results are still saved to `--output`:

```bash
python main.py --engine deepface --source clip.mp4 --output out.mp4 --no-display
```

### List all available engines

```bash
python main.py --list-engines
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `Ctrl + C` | Force exit from the terminal |

---

## ⚙️ CLI Reference

Run `python main.py --help` for the full list. Here's the summary:

| Flag | Default | Description |
|------|---------|-------------|
| `--engine {deepface,fer,opencv}` | `deepface` | Emotion analysis engine |
| `--source PATH` | webcam | Camera index, image, or video path |
| `--camera N` | `0` | Webcam index (when `--source` is not given) |
| `--output PATH` | `None` | Save the annotated result |
| `--frame-skip N` | `10` | Analyse emotions every N frames |
| `--min-face-size N` | `100` | Minimum face size for detection (px) |
| `--detector-backend BACKEND` | `opencv` | DeepFace face-detector backend |
| `--model-dir PATH` | `models/` | Where to cache the OpenCV model |
| `--no-display` | off | Run without a display window |
| `--list-engines` | off | Print available engines and exit |
| `--verbose` | off | Debug-level logging |
| `--version` | — | Show version |

**Examples:**

```bash
# Use a better DeepFace face detector
python main.py --engine deepface --detector-backend mtcnn

# Trade accuracy for speed on a low-end machine
python main.py --engine opencv --frame-skip 5 --min-face-size 80

# Use a secondary webcam
python main.py --camera 1
```

---

## 🧠 How It Works

```
Webcam / Video / Image
        │
        ▼
┌─────────────────────┐
│  Frame (BGR)        │
└─────────┬───────────┘
          ▼
   Convert to grayscale
          │
          ▼
┌──────────────────────────────┐
│  Face detection (Haar)       │
│  cv2.CascadeClassifier       │
└─────────┬────────────────────┘
          ▼
┌──────────────────────────────┐
│  Every N frames:             │
│  Emotion analysis per face   │
│  (DeepFace / FER / OpenCV)   │
└─────────┬────────────────────┘
          ▼
  Draw boxes + labels
          │
          ▼
   Display / Save output
```

1. **Face detection** — [OpenCV's Haar cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) finds faces in the grayscale frame.
2. **Emotion analysis** — each face crop is passed to your chosen engine, which predicts one of the [7 emotions](#-supported-emotions).
3. **Frame skipping** — analysis runs every `N` frames (default `10`) so the app stays responsive; the last prediction is kept and displayed in between.
4. **Annotation** — green bounding boxes and emotion labels are drawn over the frame.

---

## 📂 Project Structure

```
Imotion-detector/
├── main.py                     # 🚪 CLI entry point
├── imotion_detector/           # 📦 Core package
│   ├── __init__.py
│   ├── __main__.py             # python -m imotion_detector
│   ├── cli.py                  # Argument parsing & runners
│   ├── detector.py             # Face detection + annotation pipeline
│   ├── utils.py                # Model download helper
│   └── engines/                # 🔌 Pluggable emotion engines
│       ├── base.py             # Abstract EmotionEngine interface
│       ├── deepface_engine.py  # DeepFace backend
│       ├── fer_engine.py       # FER backend
│       └── opencv_engine.py    # OpenCV DNN backend
├── requirements/               # Per-engine dependency files
│   ├── base.txt
│   ├── deepface.txt
│   ├── fer.txt
│   ├── opencv.txt
│   └── all.txt
├── pyproject.toml              # Packaging config
├── models/                     # Auto-downloaded models (gitignored)
├── .gitignore
├── .gitattributes
├── LICENSE                     # MIT License
└── README.md
```

> 🧩 **Adding your own engine:** implement [`EmotionEngine`](imotion_detector/engines/base.py) (just the `analyze(face_bgr)` method) and register it in [`engines/__init__.py`](imotion_detector/engines/__init__.py). That's it!

---

## 🔧 Customization

All knobs are available on the CLI, but you can also use the package in your own Python code:

```python
import cv2
from imotion_detector.detector import FaceEmotionPipeline
from imotion_detector.engines import FEREngine

pipeline = FaceEmotionPipeline(
    engine=FEREngine(),
    frame_skip=5,          # analyse every 5 frames
    min_face_size=80,      # smaller faces allowed
    scale_factor=1.1,      # cascade scale factor
)

frame = cv2.imread("me.jpg")
annotated = pipeline.process(frame)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engine` | required | `DeepFaceEngine()`, `FEREngine()`, `OpenCVEngine()` |
| `frame_skip` | `10` | Analyse every N frames (higher = faster) |
| `min_face_size` | `100` | Minimum face size (px) |
| `scale_factor` | `1.1` | Cascade detection scale |
| `min_neighbors` | `5` | Cascade detection neighbors |
| `pad_ratio` | `0.2` | Context padding around crops (helps accuracy) |

---

## 🩺 Troubleshooting

| Problem | Solution |
|---------|----------|
| ❌ `Webcam not opening` | Close other apps using the camera, or try `--camera 1` |
| ❌ `No faces detected` | Improve lighting, face the camera, lower `--min-face-size` |
| 🐢 `Slow performance` | Use `--engine opencv` or raise `--frame-skip` |
| 📥 `Model download fails` | Check your internet connection and retry (first run only) |
| 🔌 `DeepFace is not installed` | Run `pip install -r requirements/deepface.txt` |
| 🔌 `FER is not installed` | Run `pip install -r requirements/fer.txt` |
| 💾 `Out of memory` | Install `tensorflow-cpu` or use `--engine opencv` |
| 🪟 `cv2.waitKey` error on Linux | Install `opencv-python`'s GUI-capable build (`pip install opencv-python`) |

Still stuck? [Open an issue](https://github.com/shahmdtamimhasan/Imotion-detector/issues/new).

---

## ❓ FAQ

**Which engine should I use?**
- For **maximum accuracy**: DeepFace
- For **balanced speed on CPU**: FER
- For **low-end PCs / minimal dependencies**: OpenCV

**Do I need a GPU?** No. All engines run on CPU, though DeepFace benefits most from a GPU.

**Are models included in the repo?** No. Models download automatically on first run (this keeps the repo small).

**Does it work on multiple faces?** Yes — all faces in the frame are detected and tracked.

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. 🍴 Fork the [repository](https://github.com/shahmdtamimhasan/Imotion-detector)
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4. ✅ Open a [pull request](https://github.com/shahmdtamimhasan/Imotion-detector/pulls)

Ideas: more engines, better face tracking, confidence scores, emotion history graphs, GUI support...

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

### 🙌 Built with

[![OpenCV](https://img.shields.io/badge/OpenCV-grey?logo=opencv)](https://opencv.org/) [![DeepFace](https://img.shields.io/badge/DeepFace-grey?logo=python)](https://github.com/serengil/deepface) [![FER](https://img.shields.io/badge/FER-grey?logo=python)](https://github.com/justinshenk/fer)

#### 👤 Author: [Tamim Hasan](https://github.com/shahmdtamimhasan)

[![GitHub](https://img.shields.io/badge/GitHub-shahmdtamimhasan-blue?logo=github)](https://github.com/shahmdtamimhasan) [![Issues](https://img.shields.io/github/issues/shahmdtamimhasan/Imotion-detector?color=red)](https://github.com/shahmdtamimhasan/Imotion-detector/issues)

⭐ If you find this project useful, please give it a star!

</div>
