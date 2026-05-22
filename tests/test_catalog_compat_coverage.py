"""Edge-case coverage for catalog.compat and catalog.header_probe."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from lilbee.catalog import compat, header_probe


def test_resolve_blob_url_returns_empty_when_hf_hub_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed repo ids cause hf_hub_url to raise; resolver must swallow that."""

    def _raise(*args: object, **kwargs: object) -> str:
        raise ValueError("bad repo")

    monkeypatch.setattr(compat, "hf_hub_url", _raise)
    assert compat._resolve_blob_url("acme/foo-GGUF:model.gguf") == ""


def test_probe_returns_empty_when_blob_lacks_gguf_magic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response that doesn't start with the GGUF magic bytes returns empty."""
    blob = b"NOTGGUF" + b"\x00" * 32
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=blob))
    assert header_probe.probe_architecture("https://example.test/x.gguf") == ""


def test_download_target_translates_unsupported_arch_to_runtime_error() -> None:
    """The task-bar download target converts UnsupportedArchError into a friendly RuntimeError."""
    from lilbee.catalog.compat import UnsupportedArchError
    from lilbee.catalog.models import CatalogModel
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.widgets.task_bar_controller import _download_target

    class _Reporter:
        def update(self, percent: float, detail: str) -> None:
            pass

    model = CatalogModel(
        hf_repo="acme/foo-GGUF",
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )

    def _raise(*args: object, **kwargs: object) -> object:
        raise UnsupportedArchError("acme/foo-GGUF", "kimi_k2")

    with (
        patch("lilbee.app.models.pull_model_data", side_effect=_raise),
        pytest.raises(RuntimeError, match="kimi_k2"),
    ):
        _download_target(_Reporter(), model)
