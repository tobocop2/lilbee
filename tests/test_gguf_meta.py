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


class TestMetadataDiskCache:
    """The parse costs ~60s for a multi-GB model, so it must survive the process."""

    def _isolate(self, monkeypatch, tmp_path):
        from lilbee.providers import gguf_meta

        monkeypatch.setattr("lilbee.core.system.default_state_dir", lambda: tmp_path)
        gguf_meta._METADATA_CACHE.clear()
        return gguf_meta

    def test_second_process_reads_the_disk_cache_instead_of_reparsing(
        self, monkeypatch, tmp_path
    ) -> None:
        gguf_meta = self._isolate(monkeypatch, tmp_path)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 32)
        calls: list[str] = []

        def fake_parse(path):
            calls.append(str(path))
            return {"context_length": "4096"}

        monkeypatch.setattr(gguf_meta, "_read_gguf_metadata_uncached", fake_parse)
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        assert len(calls) == 1

        # A fresh process: in-memory cache gone, disk entry still there.
        gguf_meta._METADATA_CACHE.clear()
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        assert len(calls) == 1, "re-parsed despite a usable disk cache"

    def test_a_changed_file_is_re_read(self, monkeypatch, tmp_path) -> None:
        gguf_meta = self._isolate(monkeypatch, tmp_path)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 32)
        calls: list[str] = []
        monkeypatch.setattr(
            gguf_meta,
            "_read_gguf_metadata_uncached",
            lambda p: (calls.append(str(p)), {"context_length": str(len(calls))})[1],
        )
        gguf_meta.read_gguf_metadata(model)
        gguf_meta._METADATA_CACHE.clear()
        model.write_bytes(b"y" * 64)  # different size + mtime -> different key
        gguf_meta.read_gguf_metadata(model)
        assert len(calls) == 2

    def test_a_corrupt_cache_entry_falls_back_to_parsing(self, monkeypatch, tmp_path) -> None:
        gguf_meta = self._isolate(monkeypatch, tmp_path)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 32)
        calls: list[str] = []
        monkeypatch.setattr(
            gguf_meta,
            "_read_gguf_metadata_uncached",
            lambda p: (calls.append(str(p)), {"context_length": "4096"})[1],
        )
        gguf_meta.read_gguf_metadata(model)
        gguf_meta._METADATA_CACHE.clear()
        for entry in (tmp_path / "gguf-meta").glob("*.json"):
            entry.write_text("{not json", encoding="utf-8")
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        assert len(calls) == 2  # corrupt entry ignored, parsed again

    def test_a_file_with_no_metadata_is_cached_too(self, monkeypatch, tmp_path) -> None:
        gguf_meta = self._isolate(monkeypatch, tmp_path)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 32)
        calls: list[str] = []
        monkeypatch.setattr(
            gguf_meta, "_read_gguf_metadata_uncached", lambda p: (calls.append(str(p)), None)[1]
        )
        assert gguf_meta.read_gguf_metadata(model) is None
        gguf_meta._METADATA_CACHE.clear()
        assert gguf_meta.read_gguf_metadata(model) is None
        assert len(calls) == 1, "an unparseable file re-parsed instead of using the cached miss"
