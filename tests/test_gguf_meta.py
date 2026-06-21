"""read_gguf_metadata robustness against corrupt/truncated GGUF headers."""

from __future__ import annotations

from pathlib import Path

from lilbee.providers.gguf_meta import read_gguf_metadata
from tests._gguf_fixture import make_minimal_gguf


def test_returns_metadata_for_valid_gguf(tmp_path: Path) -> None:
    f = tmp_path / "ok.gguf"
    f.write_bytes(make_minimal_gguf("llama"))
    meta = read_gguf_metadata(f)
    assert meta is not None
    assert meta.get("architecture") == "llama"


def test_returns_none_for_truncated_header(tmp_path: Path) -> None:
    """A truncated/corrupt GGUF must yield None, not a raw parser error that would
    abort the whole fleet build (bb-7jg1.15)."""
    f = tmp_path / "bad.gguf"
    # GGUF magic + version, then the tensor/kv count fields cut off mid-read.
    f.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x01\x00")
    assert read_gguf_metadata(f) is None
