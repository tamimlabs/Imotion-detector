"""Allow running the app via `python -m imotion_detector`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
