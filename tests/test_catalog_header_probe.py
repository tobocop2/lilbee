"""Unit tests for catalog.header_probe.probe_architecture."""

from __future__ import annotations

import struct

import httpx
import pytest

from lilbee.catalog import header_probe
from lilbee.catalog.header_probe import GGUF_HEADER_PROBE_BYTES, probe_architecture
from tests._gguf_fixture import make_minimal_gguf


def test_probe_reads_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = make_minimal_gguf("llama")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=blob))
    assert probe_architecture("https://example.test/model.gguf") == "llama"


def test_probe_handles_truncated_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=b"GGUF\x03"))
    assert probe_architecture("https://example.test/model.gguf") == ""


def test_probe_handles_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(404, content=b""))
    assert probe_architecture("https://example.test/model.gguf") == ""


def test_probe_handles_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: object, **kw: object) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", _raise)
    assert probe_architecture("https://example.test/model.gguf") == ""


def test_probe_sends_range_header(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = make_minimal_gguf("llama")
    captured: dict[str, object] = {}

    def _capture(url: str, headers: dict[str, str], timeout: float) -> httpx.Response:
        captured["headers"] = headers
        return httpx.Response(200, content=blob)

    monkeypatch.setattr(header_probe.httpx, "get", _capture)
    probe_architecture("https://example.test/model.gguf")
    assert captured["headers"]["Range"] == f"bytes=0-{GGUF_HEADER_PROBE_BYTES - 1}"


def test_probe_returns_empty_on_missing_arch_key(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=blob))
    assert probe_architecture("https://example.test/model.gguf") == ""


def test_probe_returns_empty_on_non_string_arch_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """If general.architecture exists but isn't a STRING type, probe returns empty."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", 1)
    key = b"general.architecture"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", 4)
    buf += struct.pack("<I", 99)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=bytes(buf)))
    assert probe_architecture("https://example.test/model.gguf") == ""


def test_probe_skips_unrelated_kv_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KV pair before architecture is parsed and skipped without error."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", 2)
    pre_key = b"general.quantization_version"
    buf += struct.pack("<Q", len(pre_key)) + pre_key
    buf += struct.pack("<I", 4)
    buf += struct.pack("<I", 2)
    arch_key = b"general.architecture"
    buf += struct.pack("<Q", len(arch_key)) + arch_key
    buf += struct.pack("<I", 8)
    val = b"qwen3"
    buf += struct.pack("<Q", len(val)) + val
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=bytes(buf)))
    assert probe_architecture("https://example.test/model.gguf") == "qwen3"


def test_probe_handles_array_value_in_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Array-valued KV pairs (e.g. tokenizer lists) are skipped without crashing."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", 2)
    list_key = b"tokenizer.ggml.tokens"
    buf += struct.pack("<Q", len(list_key)) + list_key
    buf += struct.pack("<I", 9)
    buf += struct.pack("<I", 8)
    buf += struct.pack("<Q", 2)
    for token in (b"a", b"bb"):
        buf += struct.pack("<Q", len(token)) + token
    arch_key = b"general.architecture"
    buf += struct.pack("<Q", len(arch_key)) + arch_key
    buf += struct.pack("<I", 8)
    val = b"gemma3"
    buf += struct.pack("<Q", len(val)) + val
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=bytes(buf)))
    assert probe_architecture("https://example.test/model.gguf") == "gemma3"


def test_probe_returns_empty_on_unknown_value_type_in_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown type tag in a skipped KV produces empty (not crash)."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", 2)
    unk_key = b"weird"
    buf += struct.pack("<Q", len(unk_key)) + unk_key
    buf += struct.pack("<I", 99)
    arch_key = b"general.architecture"
    buf += struct.pack("<Q", len(arch_key)) + arch_key
    buf += struct.pack("<I", 8)
    val = b"llama"
    buf += struct.pack("<Q", len(val)) + val
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=bytes(buf)))
    assert probe_architecture("https://example.test/model.gguf") == ""
