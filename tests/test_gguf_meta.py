"""read_gguf_metadata robustness against corrupt/truncated GGUF headers."""

from __future__ import annotations

import struct
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


def _gguf_with_duplicate_key() -> bytes:
    """A parseable GGUF header carrying the same metadata key twice; GGUFReader
    raises KeyError ('Duplicate ... already in list') on the second one."""

    def gstr(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
    for _ in range(2):
        blob += gstr(b"general.name") + struct.pack("<I", 8) + gstr(b"x")
    return blob


@pytest.mark.parametrize(
    ("label", "data"),
    [
        # version + counts cut off mid-read -> ValueError from the gguf reader.
        ("truncated", b"GGUF" + b"\x03\x00\x00\x00" + b"\x01\x00"),
        # bad magic -> ValueError.
        ("bad_magic", b"XXXX" + b"\x00" * 40),
        # magic only, no count/field table -> IndexError from the gguf reader.
        ("magic_only", b"GGUF"),
        # duplicate metadata key -> KeyError from GGUFReader._push_field.
        ("duplicate_key", _gguf_with_duplicate_key()),
    ],
)
def test_returns_none_for_corrupt_header(tmp_path: Path, label: str, data: bytes) -> None:
    """A corrupt-but-parseable GGUF must yield None across the parser's failure
    modes (ValueError, IndexError, and the duplicate-key KeyError are all
    observed), not a raw error that would abort the whole fleet build (bb-7jg1.15)."""
    f = tmp_path / f"{label}.gguf"
    f.write_bytes(data)
    assert read_gguf_metadata(f) is None
