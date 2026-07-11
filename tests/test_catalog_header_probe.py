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


def _gguf_arch_then_truncated_tokenizer(arch: str) -> bytes:
    """A header with general.architecture first, then a tokenizer string-array whose
    declared count is huge but whose bytes are cut off (as a real model's tokenizer
    arrays are by the Range-GET probe window)."""

    def gstr(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    blob = b"GGUF" + struct.pack("<I", 3)  # magic + version
    blob += struct.pack("<Q", 0)  # tensor_count
    blob += struct.pack("<Q", 2)  # kv_count
    # KV0: general.architecture = <arch> (value type 8 = STRING)
    blob += gstr(b"general.architecture") + struct.pack("<I", 8) + gstr(arch.encode())
    # KV1: tokenizer.ggml.tokens = ARRAY(STRING) declaring 151936 entries, then cut off.
    blob += gstr(b"tokenizer.ggml.tokens") + struct.pack("<I", 9)
    blob += struct.pack("<I", 8) + struct.pack("<Q", 151936)
    blob += gstr(b"only-one-token-then-truncated")
    return blob


def _kv(key: bytes, vtype: int, value: bytes) -> bytes:
    return struct.pack("<Q", len(key)) + key + struct.pack("<I", vtype) + value


def _gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _hdr(entries: list[bytes], kv_count: int | None = None) -> bytes:
    count = kv_count if kv_count is not None else len(entries)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", count)
    return head + b"".join(entries)


class TestParseArchWalker:
    """The KV walker behind probe_architecture, exercised at the byte level."""

    def test_skips_scalar_string_and_array_values_before_arch(self) -> None:
        blob = _hdr(
            [
                _kv(b"general.quantization_version", 4, struct.pack("<I", 2)),  # uint32 scalar
                _kv(b"general.name", 8, _gstr(b"m")),  # string
                _kv(
                    b"x.list",
                    9,
                    struct.pack("<I", 4) + struct.pack("<Q", 2) + struct.pack("<II", 1, 2),
                ),
                _kv(b"general.architecture", 8, _gstr(b"qwen3")),
            ]
        )
        assert header_probe._parse_arch(blob) == "qwen3"

    def test_skips_string_array_before_arch(self) -> None:
        arr = struct.pack("<I", 8) + struct.pack("<Q", 2) + _gstr(b"a") + _gstr(b"bb")
        blob = _hdr(
            [
                _kv(b"x.strs", 9, arr),
                _kv(b"general.architecture", 8, _gstr(b"llama")),
            ]
        )
        assert header_probe._parse_arch(blob) == "llama"

    def test_truncated_mid_kv_returns_empty(self) -> None:
        blob = _hdr([_kv(b"general.name", 8, _gstr(b"m"))], kv_count=5)  # claims 5, only 1 present
        assert header_probe._parse_arch(blob) == ""

    def test_unknown_value_type_returns_empty(self) -> None:
        blob = _hdr([_kv(b"weird", 99, b"")], kv_count=1)  # arch never reached
        assert header_probe._parse_arch(blob) == ""

    def test_unknown_array_element_type_returns_empty(self) -> None:
        bad_arr = struct.pack("<I", 99) + struct.pack("<Q", 1)
        blob = _hdr([_kv(b"weird", 9, bad_arr)], kv_count=1)
        assert header_probe._parse_arch(blob) == ""

    def test_non_string_arch_returns_empty(self) -> None:
        blob = _hdr([_kv(b"general.architecture", 4, struct.pack("<I", 1))])
        assert header_probe._parse_arch(blob) == ""


def test_probe_reads_arch_before_truncated_tokenizer_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real GGUF emits general.architecture first, then multi-megabyte tokenizer
    arrays that run past the probe window. The KV walker returns the arch without
    parsing those arrays, where gguf-py's GGUFReader chokes on the truncation and
    the arch-compat guard never fired (bb-ziks.43 end-to-end)."""
    blob = _gguf_arch_then_truncated_tokenizer("qwen3")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, content=blob))
    assert probe_architecture("https://example.test/model.gguf") == "qwen3"


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

    def _capture(
        url: str, headers: dict[str, str], timeout: float, follow_redirects: bool = False
    ) -> httpx.Response:
        captured["headers"] = headers
        captured["follow_redirects"] = follow_redirects
        return httpx.Response(200, content=blob)

    monkeypatch.setattr(header_probe.httpx, "get", _capture)
    probe_architecture("https://example.test/model.gguf")
    assert captured["headers"]["Range"] == f"bytes=0-{GGUF_HEADER_PROBE_BYTES - 1}"
    # HF /resolve/ URLs 302 to the CDN; the probe must follow the redirect or it
    # reads the redirect notice instead of the GGUF header (bb-ziks.43).
    assert captured["follow_redirects"] is True


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
