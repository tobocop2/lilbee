"""Cancelling a xet download has to stop the transfer, not just mark the row.

xet drives the progress callback from a thread it owns, so the TaskCancelledError
lilbee raises there is swallowed and the download runs to completion behind a row
that says cancelled. Measured: the HTTP path aborts in 1.8s after 39 callbacks;
the xet path reported the file fully written.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.runtime.cancellation import TaskCancelledError


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/x-GGUF",
        gguf_filename="x.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


def test_abort_calls_through_to_the_xet_session(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    module = type(sys)("huggingface_hub.utils._xet")
    module.abort_xet_session = lambda: called.append("abort")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._xet", module)

    dl.abort_active_download()

    assert called == ["abort"]


def test_abort_is_a_no_op_without_xet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the xet path needs it; HTTP cancels by raising through the callback."""
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._xet", None)

    dl.abort_active_download()  # must not raise


def test_an_aborted_transfer_reads_as_cancelled_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hf_xet reports the abort as a bare RuntimeError. Left untranslated the row
    says failed, which reads as a broken download rather than the user's own
    keypress."""
    config = dl.DownloadConfig(
        repo_id="acme/x-GGUF", filename="x.gguf", token=None, cache_dir=str(tmp_path)
    )

    def _aborted(**_kw: Any) -> str:
        raise RuntimeError("Operation cancelled: Task cancelled: task 19 was cancelled")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _aborted)

    with pytest.raises(TaskCancelledError):
        dl._hf_download_or_translate(_entry(), config)


def test_an_unrelated_runtime_error_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancellation branch must not swallow real failures."""
    config = dl.DownloadConfig(
        repo_id="acme/x-GGUF", filename="x.gguf", token=None, cache_dir=str(tmp_path)
    )

    def _boom(**_kw: Any) -> str:
        raise RuntimeError("something genuinely broken")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="something genuinely broken"):
        dl._hf_download_or_translate(_entry(), config)
