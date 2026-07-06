"""Unit tests for catalog.compat.resolve_arch_for_pull."""

from __future__ import annotations

import pytest

from lilbee.catalog import compat
from lilbee.catalog.compat import resolve_arch_for_pull
from lilbee.catalog.hf_client import HfClient


@pytest.fixture
def client() -> HfClient:
    return HfClient()


def test_resolve_hits_arch_cache(client: HfClient) -> None:
    client.cache_arch("acme/foo-GGUF", "llama")
    assert resolve_arch_for_pull("acme/foo-GGUF", client) == "llama"


def test_resolve_miss_returns_empty_when_no_probe_target(
    client: HfClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compat, "_resolve_blob_url", lambda _ref: "")
    assert resolve_arch_for_pull("acme/missing-GGUF", client) == ""


def test_resolve_miss_invokes_probe(client: HfClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "_resolve_blob_url", lambda ref: f"https://example.test/{ref}.gguf")
    monkeypatch.setattr(compat, "probe_architecture", lambda _url: "qwen3")
    assert resolve_arch_for_pull("acme/foo-GGUF", client) == "qwen3"


def test_resolve_writes_back_to_cache(client: HfClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "_resolve_blob_url", lambda _ref: "https://example.test/x.gguf")
    monkeypatch.setattr(compat, "probe_architecture", lambda _url: "gemma3")
    resolve_arch_for_pull("acme/bar-GGUF", client)
    assert client.get_cached_arch("acme/bar-GGUF") == "gemma3"


def test_resolve_does_not_cache_empty_arch_on_probe_failure(
    client: HfClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient probe failure ('') must not be cached, or the unsupported-arch
    guard would be permanently disabled for this ref. A later successful probe
    must still be able to resolve and cache the real arch."""
    monkeypatch.setattr(compat, "_resolve_blob_url", lambda _ref: "https://example.test/x.gguf")
    monkeypatch.setattr(compat, "probe_architecture", lambda _url: "")
    assert resolve_arch_for_pull("acme/flaky-GGUF", client) == ""
    assert client.get_cached_arch("acme/flaky-GGUF") is None

    monkeypatch.setattr(compat, "probe_architecture", lambda _url: "llama")
    assert resolve_arch_for_pull("acme/flaky-GGUF", client) == "llama"
    assert client.get_cached_arch("acme/flaky-GGUF") == "llama"


def test_resolve_blob_url_returns_empty_for_glob_filename() -> None:
    """`*.gguf` filenames aren't unique blobs; resolver bails to UNKNOWN."""
    assert compat._resolve_blob_url("acme/foo-GGUF:*.gguf") == ""


def test_resolve_blob_url_returns_empty_for_bare_repo() -> None:
    """Without an explicit filename the blob URL is ambiguous."""
    assert compat._resolve_blob_url("acme/foo-GGUF") == ""


def test_resolve_blob_url_returns_url_for_concrete_filename() -> None:
    """Concrete repo:file pair resolves to a usable HF blob URL."""
    url = compat._resolve_blob_url("acme/foo-GGUF:model-q4.gguf")
    assert "acme/foo-GGUF" in url
    assert url.endswith("model-q4.gguf")


def test_resolve_blob_url_resolves_native_slash_ref() -> None:
    """A canonical native ref ``<org>/<repo>/<file>.gguf`` resolves to its blob URL.
    The prior split-on-colon left every native ref with an empty filename, so the
    arch-compat guard never probed (bb-ziks.72)."""
    url = compat._resolve_blob_url("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf")
    assert "Qwen/Qwen3-0.6B-GGUF" in url
    assert url.endswith("Qwen3-0.6B-Q4_K_M.gguf")


def test_resolve_blob_url_native_ref_keeps_quant_subdir() -> None:
    """A quant stored under a repo subdir keeps the subdir in the resolved filename."""
    url = compat._resolve_blob_url("unsloth/M-GGUF/Q4_K_M/model.gguf")
    assert "unsloth/M-GGUF" in url
    assert url.endswith("Q4_K_M/model.gguf")
