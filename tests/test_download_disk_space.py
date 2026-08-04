"""A full disk must name itself, not surface as a backend's internal error.

huggingface_hub only logs a warning when the volume is too small, and hf_xet
raises OSError for auth and not-found but not for a failed write, so before
these guards an over-large pull ran to ~76% and then reported a reconstruction
error that named neither the disk nor the file.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask

_GB = 1024**3


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/big-GGUF",
        gguf_filename="big.gguf",
        size_gb=8.0,
        min_ram_gb=16.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


def _usage(free: int) -> Any:
    return shutil._ntuple_diskusage(total=free * 4, used=free * 3, free=free)


def test_refuses_a_download_larger_than_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(5 * _GB))

    with pytest.raises(RuntimeError, match="Not enough disk space"):
        dl._require_disk_space(_entry(), tmp_path, 8 * _GB)


def test_allows_a_download_that_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(20 * _GB))

    dl._require_disk_space(_entry(), tmp_path, 8 * _GB)


def test_resumed_download_counts_bytes_already_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted pull only needs the remainder; a naive check refuses it."""
    blobs = tmp_path / "models--acme--big-GGUF" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc123.incomplete").write_bytes(b"x" * (6 * 1024))
    monkeypatch.setattr(dl, "_BYTES_PER_GB", 1024)  # keep the message readable
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(3 * 1024))

    # 8KB wanted, 3KB free, but 6KB is already held by the partial blob.
    dl._require_disk_space(_entry(), tmp_path, 8 * 1024)


def test_unknown_size_does_not_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline or unresolvable sizes have nothing to compare against."""
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(0))

    dl._require_disk_space(_entry(), tmp_path, dl._SIZE_UNKNOWN)


def test_failure_on_a_full_volume_reports_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The volume filling mid-transfer is what the pre-flight cannot catch."""
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(1024))
    config = dl.DownloadConfig(
        repo_id="acme/big-GGUF", filename="big.gguf", token=None, cache_dir=str(tmp_path)
    )

    def _boom(**_kw: Any) -> str:
        raise RuntimeError("File reconstruction error: Internal Writer Error")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="Ran out of disk space"):
        dl._hf_download_or_translate(_entry(), config)


def test_failure_with_room_left_keeps_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk branch must not swallow unrelated failures."""
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: _usage(50 * _GB))
    config = dl.DownloadConfig(
        repo_id="acme/big-GGUF", filename="big.gguf", token=None, cache_dir=str(tmp_path)
    )

    def _boom(**_kw: Any) -> str:
        raise RuntimeError("something unrelated")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="something unrelated"):
        dl._hf_download_or_translate(_entry(), config)


@pytest.mark.parametrize(
    ("cache_dir", "usage"),
    [
        pytest.param(None, None, id="no-cache-dir-to-measure"),
        pytest.param("present", OSError("gone"), id="path-vanished-with-the-failure"),
    ],
)
def test_unmeasurable_volume_keeps_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_dir: str | None, usage: Any
) -> None:
    """Without a readable free-space figure there is nothing to attribute."""
    if usage is not None:

        def _raise(_p: Any) -> Any:
            raise usage

        monkeypatch.setattr(dl.shutil, "disk_usage", _raise)
    config = dl.DownloadConfig(
        repo_id="acme/big-GGUF",
        filename="big.gguf",
        token=None,
        cache_dir=str(tmp_path) if cache_dir else None,
    )

    def _boom(**_kw: Any) -> str:
        raise RuntimeError("original failure")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="original failure"):
        dl._hf_download_or_translate(_entry(), config)
