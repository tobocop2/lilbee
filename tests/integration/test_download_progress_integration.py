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

import time

import pytest

from lilbee.catalog import CatalogModel, download_model
from lilbee.config import cfg

pytestmark = pytest.mark.slow

# A very small GGUF model (~3 MB) — TinyLlama stories for testing
_TINY_MODEL = CatalogModel(
    name="tinystories",
    tag="260k",
    display_name="TinyStories 260K",
    hf_repo="karpathy/tinyllamas",
    gguf_filename="stories260K.gguf",
    size_gb=0.003,
    min_ram_gb=0.5,
    description="Tiny model for testing download progress",
    featured=False,
    downloads=0,
    task="chat",
)


class TestRealDownloadProgress:
    @pytest.fixture(autouse=True)
    def _use_temp_models_dir(self, tmp_path):
        """Use a temp dir for models so we always download fresh."""
        original = cfg.models_dir
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        yield
        cfg.models_dir = original

    def test_progress_fires_incrementally(self) -> None:
        """Download a tiny real model and verify progress updates are incremental."""
        calls: list[tuple[int, int, float]] = []
        start = time.monotonic()

        def on_progress(downloaded: int, total: int) -> None:
            elapsed = time.monotonic() - start
            calls.append((downloaded, total, elapsed))

        download_model(_TINY_MODEL, on_progress=on_progress)

        # Must have received multiple progress callbacks (not just one 100% jump)
        assert len(calls) >= 2, (
            f"Expected multiple progress callbacks, got {len(calls)}. "
            "Progress may be jumping to 100% immediately."
        )

        # Verify monotonically increasing downloaded bytes
        downloaded_values = [d for d, _, _ in calls]
        for i in range(1, len(downloaded_values)):
            assert downloaded_values[i] >= downloaded_values[i - 1], (
                f"Downloaded bytes decreased at call {i}: "
                f"{downloaded_values[i - 1]} → {downloaded_values[i]}"
            )

        # Final callback should have downloaded == total (100%)
        final_downloaded, final_total, _ = calls[-1]
        assert final_downloaded == final_total, (
            f"Final callback not at 100%: {final_downloaded}/{final_total}"
        )

        # Total should be > 0 (we know the file size)
        assert final_total > 0

    def test_already_downloaded_reports_100(self) -> None:
        """Second download of same model reports 100% immediately."""
        # First download
        download_model(_TINY_MODEL)

        # Second download should report 100% in a single call
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
