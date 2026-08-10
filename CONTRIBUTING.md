# Contributing to Imotion Detector

Thank you for your interest in contributing! This guide explains how to set up the project, add a new feature, and submit your changes.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Adding a New Engine](#adding-a-new-engine)
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
- **Python 3.8+**
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

5. **Verify the app runs:**

   ```bash
   python main.py --list-engines
   ```

---

## How to Contribute

1. **Create a branch** off the latest `main`:

   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — keep them focused and small.

3. **Test** with the lightest engine (OpenCV) to keep iteration fast:

   ```bash
   python main.py --engine opencv --source sample.jpg --output out.jpg
   ```

4. **Verify syntax** of any Python files you changed:

   ```bash
   python -m py_compile <changed_file.py>
   ```

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

## Code Style

- Follow **PEP 8**.
- No comments unless they explain *why* — the code should be self-documenting.
- Use **type hints** on function signatures.
- Keep functions small and single-purpose.
- Preserve the existing structure: engine logic in `engines/`, CLI/runners in `cli.py`, shared pipeline in `detector.py`.

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
