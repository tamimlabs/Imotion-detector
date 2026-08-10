"""Shared helpers: model downloading and misc utilities."""

import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_model_downloaded(url, filename, model_dir=None):
    """Download a model file on first use and cache it locally.

    Returns the path to the (existing or freshly downloaded) model file.
    """
    directory = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / filename

    if model_path.exists() and model_path.stat().st_size > 0:
        logger.debug("Model already cached: %s", model_path)
        return model_path

    logger.info("Downloading %s ...", filename)
    temp_path = directory / (filename + ".part")
    try:
        urlretrieve(url, temp_path)
        os.replace(temp_path, model_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download model {filename}: {exc}") from exc

    logger.info("Model downloaded to %s", model_path)
    return model_path
