# AGENTS.md

Guidelines for AI agents and automated tools working in this repository.

## Project Overview

Real-time facial emotion recognition CLI. Webcam, image, and video input; three pluggable analysis engines (DeepFace, FER, OpenCV DNN). Python 3.8+, depends on OpenCV and numpy.

## Commands

- Run: `python main.py --help`
- List engines: `python main.py --list-engines`
- Syntax check only: `python -m py_compile <file.py>`
- Tests: none (do not add; this project is verified manually)

## Constraints

- **Do not run the app** unless the user explicitly asks. The user's machine is low-configuration.
- Model weights are never committed; they auto-download to `models/` (gitignored) at runtime.
- Keep engines pluggable: engine logic lives in `imotion_detector/engines/`, never inside `cli.py` or `detector.py`.
- New engines must subclass `EmotionEngine` (implement `analyze(face_bgr)`) and register in `imotion_detector/engines/__init__.py`.

## Style

- PEP 8, type hints on signatures, no explanatory comments unless they clarify intent.
- Follow existing patterns in the codebase; do not introduce new frameworks without need.

## Docs

- User-facing docs live in `README.md`.
- Contributor guidelines live in `CONTRIBUTING.md`.
- Keep the README accurate to the actual CLI flags and project structure when changing either.
