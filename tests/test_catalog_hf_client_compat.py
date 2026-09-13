"""HfClient.fetch_models populates arch + compat + trending fields and the arch cache."""

from __future__ import annotations

import httpx
import pytest

from lilbee.catalog.hf_client import HfClient
from lilbee.catalog.types import ModelCompat

pytestmark = pytest.mark.real_hf_client


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


def test_fetch_classifies_compat_from_the_declared_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows carry a real classify() verdict, not a blanket SUPPORTED."""
    rows = [_hf_row("qwen3"), _hf_row("no-such-arch-xyz")]
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response(rows))
    models = HfClient().fetch_models().models

    by_arch = {m.architecture: m.compat for m in models}
    assert by_arch["qwen3"] is ModelCompat.SUPPORTED
    assert by_arch["no-such-arch-xyz"] is not ModelCompat.SUPPORTED


def test_fetch_populates_trending_score(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _hf_row("llama")
    row["trendingScore"] = 7
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response([row]))
    page = HfClient().fetch_models()
    assert page.models[0].trending_score == 7


def test_fetch_marks_a_stripped_repo_from_its_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """The content flag comes from upstream tags, not the repo name."""
    clean, stripped = _hf_row("qwen3"), _hf_row("qwen3")
    stripped["id"] = "x/Qwen3-8B-Uncensored-GGUF"
    stripped["tags"] = ["gguf", "uncensored", "text-generation"]
    name_only = _hf_row("qwen3")
    name_only["id"] = "y/Qwen3-8B-Uncensored-GGUF"
    name_only["tags"] = ["gguf", "text-generation"]
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _mock_response([clean, stripped, name_only]))
    models = HfClient().fetch_models().models

    by_repo = {m.hf_repo: m.safety_stripped for m in models}
    assert by_repo["acme/test-GGUF"] is False
    assert by_repo["x/Qwen3-8B-Uncensored-GGUF"] is True
    assert by_repo["y/Qwen3-8B-Uncensored-GGUF"] is False


def test_fetch_requests_the_tags_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``tags`` in the expand the flag would always read clean."""
    seen: dict[str, object] = {}

    def fake_get(*args: object, **kwargs: object) -> httpx.Response:
        seen["params"] = kwargs["params"]
        return _mock_response([])

    monkeypatch.setattr(httpx, "get", fake_get)
    HfClient().fetch_models()

    assert "tags" in str(seen["params"])
