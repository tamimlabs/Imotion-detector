# Contributing to Imotion Detector

Thank you for your interest in contributing! This guide explains how to set up the project, add a new feature, and submit your changes.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing, Linting & Type Checks](#testing-linting--type-checks)
- [How to Contribute](#how-to-contribute)
- [Adding a New Engine](#adding-a-new-engine)
- [Adding a New Face Detector](#adding-a-new-face-detector)
- [Code Style](#code-style)
- [Commit Guidelines](#commit-guidelines)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

This project is hosted on GitHub:

- Repository: <https://github.com/tamimlabs/Imotion-detector>
- Issue tracker: <https://github.com/tamimlabs/Imotion-detector/issues>
- Pull requests: <https://github.com/tamimlabs/Imotion-detector/pulls>

You'll need:

- **Git** installed and configured
- **Python 3.10+**
- A **GitHub account** (for forks and pull requests)

---

## Development Setup

1. **Fork the repository** on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/Imotion-detector.git
   cd Imotion-detector
   ```

2. **Add the original repo as an upstream remote:**

   ```bash
   git remote add upstream https://github.com/tamimlabs/Imotion-detector.git
   ```

3. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   ```

4. **Install dependencies.** For engine work, install at least one engine's requirements:

   ```bash
   pip install -r requirements/opencv.txt   # base + lightest engine
   # or
   pip install -r requirements/deepface.txt # highest accuracy
   # or
   pip install -r requirements/all.txt      # everything
   ```

5. **Install the dev tooling** (pytest, ruff, mypy) if you plan to write tests:

   ```bash
   pip install -r requirements/dev.txt
   ```

6. **Verify the app runs:**

   ```bash
   python main.py --list-engines
   ```

### Dependency version pins

`requirements/base.txt` pins `opencv-python>=4.5.4,<5` and `numpy>=1.21,<2.5`. These are
deliberate, not stale:

- **OpenCV 5.x** removed `cv2.CascadeClassifier`, which breaks the Haar face detector.
- **numpy 2.5+** type stubs use Python 3.12-only syntax, which fails `mypy` on the supported
  Python 3.10/3.11 floor (`python_version = "3.10"` in `pyproject.toml`).

If you change these pins, update `[project.dependencies]` in `pyproject.toml` too, and make
sure CI still passes on **all** matrix entries (Linux + Windows × Python 3.10 / 3.11 / 3.13) —
the oldest Python in the matrix resolves different dependency versions than the newest.

---

## Testing, Linting & Type Checks

Every PR is expected to pass the same checks CI runs:

```bash
pytest             # run the test suite
ruff check .       # lint
mypy imotion_detector main.py tests   # static type checks
```

Tips:

- Write a test alongside any new behaviour — the suite lives in `tests/`.
- Tests marked `models` or `network` (which download small ONNX models) are skipped
  automatically when there is no internet access, so the suite passes anywhere.
- Run `pytest tests/<file>` to target a single file while iterating.

---

## How to Contribute

1. **Create a branch** off the latest `main`:

   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — keep them focused and small.

3. **Write a test** covering your change (see [Testing, Linting & Type Checks](#testing-linting--type-checks)).

4. **Run the checks** — make sure `pytest`, `ruff check .` and
   `mypy imotion_detector main.py tests` all pass before submitting.

5. **Commit and push:**

   ```bash
   git add <files>
   git commit -m "Add <short summary of the change>"
   git push origin feature/your-feature-name
   ```

6. **Open a pull request** against the `main` branch of the upstream repo and describe your change.

---

## Adding a New Engine

Adding a new emotion backend is deliberately simple:

1. Create `imotion_detector/engines/<name>_engine.py` with a class that inherits from [`EmotionEngine`](imotion_detector/engines/base.py):

   ```python
   from .base import EmotionEngine, EngineDependencyError

   class MyEngine(EmotionEngine):
       name = "myengine"
       description = "Short human-readable description"

       def __init__(self):
           try:
               import my_dependency
           except ImportError as exc:
               raise EngineDependencyError(
                   "MyEngine is not installed. Install it with: pip install my-dependency"
               ) from exc
           self._api = my_dependency

       def analyze(self, face_bgr):
           """Return the dominant emotion label (str) for a BGR face crop."""
           return self._api.predict(face_bgr)
   ```

2. **Register it** in [`imotion_detector/engines/__init__.py`](imotion_detector/engines/__init__.py) by adding it to the `ENGINES` dict.

3. **Add its requirements** as a new file under `requirements/` and reference it in the README and `pyproject.toml` (optional extras).

4. **Update the README** engine table with your engine's accuracy, speed, dependencies and use case.

---

## Adding a New Face Detector

Face detectors live in [`imotion_detector/detectors.py`](imotion_detector/detectors.py) behind a tiny interface:

1. Create a class that inherits from [`FaceDetector`](imotion_detector/detectors.py):

   ```python
   class MyFaceDetector(FaceDetector):
       name = "myface"

       def detect(self, frame):
           """Return a list of (x, y, width, height) boxes for every face."""
           # ... your detection logic ...
           return [(0, 0, 200, 200)]
   ```

2. **Register it** in the `DETECTORS` dict at the bottom of [`imotion_detector/detectors.py`](imotion_detector/detectors.py) so the CLI's `--face-detector` accepts it automatically.

3. If the detector needs an ONNX/weights file, download it with
   [`ensure_model_downloaded`](imotion_detector/utils.py) (cached, atomic) instead of committing it.

4. **Add a test** in `tests/test_detectors.py` (mark model-downloading tests with `@pytest.mark.models`).

5. **Update the README** face-detector notes and CLI reference.

---

## Code Style

- Follow **PEP 8** — enforced by **ruff** (`ruff check .`).
- **Type hints on all function signatures** — enforced by **mypy** (`mypy imotion_detector main.py tests`).
- No comments unless they explain *why* — the code should be self-documenting.
- Keep functions small and single-purpose.
- Preserve the existing structure: engine logic in `engines/`, detector backends in `detectors.py`,
  tracking/annotation pipeline in `detector.py`, CLI/runners in `cli.py`, HTTP server in `server.py`.
- Add or update tests for any behaviour you change.

---

## Commit Guidelines

- Use clear, imperative commit messages: `Add MTCNN support to the DeepFace engine`.
- Commit **related changes together**; avoid mixing unrelated edits in one commit.
- Keep commits small enough to review easily.

---

## Reporting Issues

When opening an issue, include:

- Your **OS** and **Python version**
- The **command** you ran
- The **full error output** (or a screenshot)
- Whether the problem affects one engine or all of them

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
