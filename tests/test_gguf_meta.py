"""read_gguf_metadata robustness against corrupt/truncated GGUF headers."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.gguf_meta import read_gguf_metadata
from tests._gguf_fixture import make_minimal_gguf


def test_returns_metadata_for_valid_gguf(tmp_path: Path) -> None:
    f = tmp_path / "ok.gguf"
    f.write_bytes(make_minimal_gguf("llama"))
    meta = read_gguf_metadata(f)
    assert meta is not None
    assert meta.get("architecture") == "llama"


@pytest.mark.parametrize(
    ("label", "data"),
    [
        # version + counts cut off mid-read -> ValueError from the gguf reader.
        ("truncated", b"GGUF" + b"\x03\x00\x00\x00" + b"\x01\x00"),
        # bad magic -> ValueError.
        ("bad_magic", b"XXXX" + b"\x00" * 40),
        # magic only, no count/field table -> IndexError from the gguf reader.
        ("magic_only", b"GGUF"),
    ],
)
def test_returns_none_for_corrupt_header(tmp_path: Path, label: str, data: bytes) -> None:
    """A truncated/corrupt GGUF must yield None across the parser's failure modes
    (ValueError and IndexError are both observed), not a raw error that would abort
    the whole fleet build (bb-7jg1.15)."""
    f = tmp_path / f"{label}.gguf"
    f.write_bytes(data)
    assert read_gguf_metadata(f) is None
