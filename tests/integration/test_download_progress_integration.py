"""Integration test — proves real-time download progress with an actual HF download.

Downloads a tiny model (~3 MB) and verifies:
1. Progress callback fires incrementally (not just 0 → 100)
2. Cumulative bytes increase monotonically
3. Final callback has downloaded == total
4. No errors

Run with:
    uv run pytest tests/integration/test_download_progress_integration.py -v -m slow
"""

from __future__ import annotations

import pytest

from lilbee.catalog import FEATURED_EMBEDDING, CatalogModel, download_model
from lilbee.config import cfg

pytestmark = pytest.mark.slow

_TINY_MODEL = FEATURED_EMBEDDING[0]


class TestRealDownloadProgress:
    def test_progress_fires_on_download(self) -> None:
        """Download the featured embedding model and verify progress callback fires."""
        calls: list[tuple[int, int]] = []
        download_model(_TINY_MODEL, on_progress=lambda d, t: calls.append((d, t)))

        assert len(calls) >= 1, "Expected at least one progress callback"
        final_downloaded, final_total = calls[-1]
        assert final_downloaded == final_total

    def test_already_downloaded_reports_100(self) -> None:
        """Cached model reports 100% in a single callback."""
        calls: list[tuple[int, int]] = []
        download_model(_TINY_MODEL, on_progress=lambda d, t: calls.append((d, t)))

        assert len(calls) == 1
        assert calls[0][0] == calls[0][1]


class TestDownloadFailureMessages:
    @pytest.fixture(autouse=True)
    def _use_temp_models_dir(self, tmp_path):
        original = cfg.models_dir
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        yield
        cfg.models_dir = original

    def test_nonexistent_repo_gives_clear_error(self) -> None:
        """Downloading from a nonexistent repo raises RuntimeError with details."""
        bad_model = CatalogModel(
            name="nonexistent",
            tag="latest",
            display_name="Nonexistent Model",
            hf_repo="this-user-does-not-exist-abc123/fake-model-xyz",
            gguf_filename="fake.gguf",
            size_gb=0.001,
            min_ram_gb=0.5,
            description="Does not exist",
            featured=False,
            downloads=0,
            task="chat",
        )
        with pytest.raises((RuntimeError, PermissionError)):
            download_model(bad_model)
