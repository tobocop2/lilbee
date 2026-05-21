"""HfClient.fetch_models populates architecture + compat fields and writes arch cache."""

from __future__ import annotations

import httpx
import pytest

from lilbee.catalog.hf_client import HfClient
from lilbee.catalog.types import ModelCompat


def _hf_row(architecture: str | None) -> dict[str, object]:
    gguf_block: dict[str, object] = {"total": 4_000_000_000, "context_length": 4096}
    if architecture is not None:
        gguf_block["architecture"] = architecture
    return {
        "id": "acme/test-GGUF",
        "siblings": [{"rfilename": "model.gguf", "size": 4_000_000_000}],
        "downloads": 100,
        "pipeline_tag": "text-generation",
        "cardData": {"description": "x"},
        "gguf": gguf_block,
    }


def _mock_response(rows: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json=rows)


def test_supported_arch_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response([_hf_row("llama")]))
    client = HfClient()
    page = client.fetch_models()
    assert page.models[0].architecture == "llama"
    assert page.models[0].compat is ModelCompat.SUPPORTED


def test_unsupported_arch_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response([_hf_row("kimi_k2_made_up")]))
    client = HfClient()
    page = client.fetch_models()
    assert page.models[0].compat is ModelCompat.UNSUPPORTED


def test_missing_arch_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response([_hf_row(None)]))
    client = HfClient()
    page = client.fetch_models()
    assert page.models[0].architecture == ""
    assert page.models[0].compat is ModelCompat.UNKNOWN


def test_fetch_populates_arch_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response([_hf_row("qwen3")]))
    client = HfClient()
    client.fetch_models()
    assert client.get_cached_arch("acme/test-GGUF") == "qwen3"


def test_featured_catalog_models_default_to_unknown_compat() -> None:
    """TOML-loaded featured entries default to UNKNOWN; probe at pull catches mismatch."""
    from lilbee.catalog.featured import FEATURED_ALL

    for entry in FEATURED_ALL:
        assert entry.compat is ModelCompat.UNKNOWN
        assert entry.architecture == ""
