"""Tests for the model-download helper."""

from __future__ import annotations

import pytest

from imotion_detector.utils import DEFAULT_MODEL_DIR, ensure_model_downloaded


@pytest.fixture
def local_file(tmp_path):
    src = tmp_path / "model.onnx"
    src.write_bytes(b"fake-model-bytes")
    return src


class TestEnsureModelDownloaded:
    def test_downloads_from_file_url(self, tmp_path, local_file):
        model_dir = tmp_path / "models"
        path = ensure_model_downloaded(
            local_file.as_uri(), local_file.name, model_dir=model_dir
        )
        assert path == model_dir / local_file.name
        assert path.read_bytes() == b"fake-model-bytes"

    def test_reuses_cache(self, tmp_path, local_file):
        model_dir = tmp_path / "models"
        first = ensure_model_downloaded(local_file.as_uri(), local_file.name, model_dir=model_dir)
        local_file.write_bytes(b"changed-bytes")  # cache must NOT re-download
        second = ensure_model_downloaded(local_file.as_uri(), local_file.name, model_dir=model_dir)
        assert first == second
        assert second.read_bytes() == b"fake-model-bytes"

    def test_failed_download_raises_and_cleans_partial(self, tmp_path):
        missing = tmp_path / "missing.onnx"
        with pytest.raises(RuntimeError):
            ensure_model_downloaded(missing.as_uri(), "missing.onnx", model_dir=tmp_path)
        assert not (tmp_path / "missing.onnx.part").exists()
        assert not (tmp_path / "missing.onnx").exists()

    def test_default_model_dir_is_project_models_folder(self):
        assert DEFAULT_MODEL_DIR.name == "models"
        assert DEFAULT_MODEL_DIR.is_absolute()
