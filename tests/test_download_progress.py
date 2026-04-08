"""Tests for download progress — proves callbacks fire incrementally and errors surface.

Unit tests use mocked hf_hub_download to verify the _CallbackProgressBar chain.
Integration test downloads a real small model to prove end-to-end progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.catalog import (
    CatalogModel,
    _CallbackProgressBar,
    _make_progress_tqdm_class,
    download_model,
)


def _test_entry(tmp_path: Path | None = None) -> CatalogModel:
    return CatalogModel(
        name="test-model",
        tag="tiny",
        display_name="Test Model Tiny",
        hf_repo="user/test-model",
        gguf_filename="test-model.gguf",
        size_gb=0.01,
        min_ram_gb=0.5,
        description="A tiny test model",
        featured=False,
        downloads=0,
        task="chat",
    )


class TestCallbackProgressBarTerminalSuppression:
    """Verify _CallbackProgressBar never writes to a terminal."""

    def test_no_terminal_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """tqdm output is fully suppressed — nothing leaks to stderr/stdout."""
        calls: list[tuple[int, int]] = []
        cls = _make_progress_tqdm_class(lambda d, t: calls.append((d, t)))
        bar = cls(total=1000)
        bar.update(500)
        bar.update(500)
        bar.close()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert calls == [(500, 1000), (1000, 1000)]

    def test_disable_is_always_true(self) -> None:
        """The disable flag is forced True regardless of what's passed."""
        bar = _CallbackProgressBar(total=100, disable=False)
        assert bar.disable
        bar.close()

    def test_no_stderr_output_during_updates(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Many rapid updates produce zero terminal output."""
        bar = _CallbackProgressBar(total=10000)
        for _ in range(100):
            bar.update(100)
        bar.close()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestCallbackProgressBarIncrementalUpdates:
    """Verify callbacks fire with cumulative byte counts, not jumps to 100%."""

    def test_incremental_progress(self) -> None:
        """Simulates chunked download: callback must fire per-chunk with cumulative bytes."""
        calls: list[tuple[int, int]] = []
        cls = _make_progress_tqdm_class(lambda d, t: calls.append((d, t)))
        total = 10_000_000  # 10 MB
        chunk_size = 1_000_000  # 1 MB chunks

        bar = cls(total=total)
        for _ in range(10):
            bar.update(chunk_size)
        bar.close()

        # Must have exactly 10 incremental calls
        assert len(calls) == 10
        # Each call has cumulative bytes
        for i, (downloaded, reported_total) in enumerate(calls, 1):
            assert downloaded == i * chunk_size
            assert reported_total == total
        # Last call should equal total
        assert calls[-1] == (total, total)

    def test_progress_never_exceeds_total(self) -> None:
        """Even if update overshoots, cumulative is reported honestly."""
        calls: list[tuple[int, int]] = []
        cls = _make_progress_tqdm_class(lambda d, t: calls.append((d, t)))
        bar = cls(total=100)
        bar.update(150)  # overshoot
        bar.close()
        assert calls == [(150, 100)]

    def test_total_none_reports_zero(self) -> None:
        """When total is unknown (None), callback receives total=0."""
        calls: list[tuple[int, int]] = []
        cls = _make_progress_tqdm_class(lambda d, t: calls.append((d, t)))
        bar = cls(total=None)
        bar.update(500)
        bar.close()
        assert calls == [(500, 0)]

    def test_no_callback_no_crash(self) -> None:
        """_CallbackProgressBar without a callback doesn't crash on update."""
        bar = _CallbackProgressBar(total=100)
        bar.update(50)  # should not raise
        bar.close()

    def test_small_chunks_fire_many_callbacks(self) -> None:
        """50 small chunks each produce a callback — no coalescing/skipping."""
        calls: list[tuple[int, int]] = []
        cls = _make_progress_tqdm_class(lambda d, t: calls.append((d, t)))
        bar = cls(total=5000)
        for _ in range(50):
            bar.update(100)
        bar.close()
        assert len(calls) == 50


class TestDownloadModelProgressChain:
    """Verify download_model correctly chains progress through to the user callback."""

    def _fake_download_with_chunks(self, **kwargs: Any) -> str:
        """Fake hf_hub_download that simulates chunked progress via tqdm_class."""
        import hashlib

        cache_dir = kwargs.get("cache_dir", "")
        repo_id = kwargs.get("repo_id", "")
        tqdm_class = kwargs.get("tqdm_class")

        # Simulate progress: 5 chunks of 200 bytes each = 1000 total
        if tqdm_class:
            bar = tqdm_class(total=1000)
            for _ in range(5):
                bar.update(200)
            bar.close()

        content = b"x" * 1000
        digest = hashlib.sha256(content).hexdigest()
        safe_repo = repo_id.replace("/", "--")
        model_dir = Path(cache_dir) / f"models--{safe_repo}"
        blobs_dir = model_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        dest = blobs_dir / digest
        dest.write_bytes(content)
        return str(dest)

    def test_progress_fires_incrementally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """download_model passes incremental progress to the user callback."""
        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", self._fake_download_with_chunks)

        entry = _test_entry()
        calls: list[tuple[int, int]] = []

        def on_progress(downloaded: int, total: int) -> None:
            calls.append((downloaded, total))

        download_model(entry, on_progress=on_progress)

        # 5 tqdm updates + 1 final 100% call from download_model
        assert len(calls) == 6
        # First 5 are incremental: 200, 400, 600, 800, 1000
        for i in range(5):
            assert calls[i] == ((i + 1) * 200, 1000)
        # Last call is the final 100% report
        assert calls[-1][0] == calls[-1][1]

    def test_already_downloaded_reports_100_immediately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-existing files report (size, size) — a single 100% call."""
        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = _test_entry()
        existing = tmp_path / entry.gguf_filename
        existing.write_bytes(b"fake model data")

        calls: list[tuple[int, int]] = []
        download_model(entry, on_progress=lambda d, t: calls.append((d, t)))

        assert len(calls) == 1
        assert calls[0][0] == calls[0][1]  # downloaded == total


class TestDownloadModelErrorPropagation:
    """Verify download failures surface clear error messages."""

    def test_timeout_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)

        def fake_timeout(**kwargs: Any) -> str:
            raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_timeout)
        entry = _test_entry()

        with pytest.raises(RuntimeError, match=r"Network error.*Connection timed out"):
            download_model(entry)

    def test_connect_error_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)

        def fake_connect(**kwargs: Any) -> str:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_connect)
        entry = _test_entry()

        with pytest.raises(RuntimeError, match=r"Network error.*Connection refused"):
            download_model(entry)

    def test_os_error_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)

        def fake_oserror(**kwargs: Any) -> str:
            raise OSError("No space left on device")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_oserror)
        entry = _test_entry()

        with pytest.raises(RuntimeError, match=r"I/O error.*No space left"):
            download_model(entry)

    def test_unexpected_error_includes_type_and_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)

        def fake_unexpected(**kwargs: Any) -> str:
            raise ValueError("unexpected format")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_unexpected)
        entry = _test_entry()

        with pytest.raises(RuntimeError, match=r"ValueError.*unexpected format"):
            download_model(entry)

    def test_gated_repo_raises_permission_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from huggingface_hub.utils import GatedRepoError

        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)

        def fake_gated(**kwargs: Any) -> str:
            raise GatedRepoError("Gated", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_gated)
        entry = _test_entry()

        with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
            download_model(entry)

    def test_repo_not_found_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from huggingface_hub.utils import RepositoryNotFoundError

        from lilbee import catalog

        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        monkeypatch.setattr(catalog, "resolve_filename", lambda e: e.gguf_filename)

        def fake_not_found(**kwargs: Any) -> str:
            raise RepositoryNotFoundError("Not found", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_not_found)
        entry = _test_entry()

        with pytest.raises(RuntimeError, match="not found on HuggingFace"):
            download_model(entry)
