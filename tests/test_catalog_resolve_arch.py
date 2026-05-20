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
    client._arch_cache["acme/foo-GGUF"] = "llama"
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
    assert client._arch_cache["acme/bar-GGUF"] == "gemma3"


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
