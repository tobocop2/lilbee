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
    assert header_probe.probe_header("https://example.test/x.gguf").architecture == ""


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


class TestFileHeader:
    """The per-candidate header read the pull path gates on."""

    @staticmethod
    def _header(monkeypatch: pytest.MonkeyPatch, header: header_probe.GgufHeader) -> None:
        monkeypatch.setattr(compat, "probe_header", lambda _url: header)

    def test_reads_model_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._header(monkeypatch, header_probe.GgufHeader("llama", "model"))
        header = compat.file_header("acme/foo-GGUF", "foo-Q4_K_M.gguf")
        assert (header.architecture, header.is_model) == ("llama", True)

    def test_reads_projector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """clip is a supported architecture, so only general.type rules the projector out."""
        self._header(monkeypatch, header_probe.GgufHeader("clip", "mmproj"))
        assert compat.file_header("acme/foo-GGUF", "foo-mmproj-Q8_0.gguf").is_model is False

    def test_unreadable_header_reads_as_an_eligible_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline, the probe yields nothing; that must not block the pull."""
        self._header(monkeypatch, header_probe.GgufHeader())
        header = compat.file_header("acme/foo-GGUF", "foo-Q4_K_M.gguf")
        assert (header.architecture, header.is_model) == ("", True)

    def test_unresolvable_repo_id_yields_an_empty_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*args: object, **kwargs: object) -> str:
            raise ValueError("bad repo")

        monkeypatch.setattr(compat, "hf_hub_url", _raise)
        assert compat.file_header("not a repo", "foo.gguf") == header_probe.GgufHeader()
