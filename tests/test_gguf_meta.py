"""read_gguf_metadata robustness against corrupt/truncated GGUF headers."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from lilbee.providers.gguf_meta import read_gguf_metadata, read_mmproj_projector_type
from tests._gguf_fixture import has_gguf_parser, make_minimal_gguf

# These hand real GGUF bytes to the engine's parser. The unit lane installs no
# engine helpers, where every one of them would pass for the wrong reason: with
# no parser the read yields no metadata, which is the answer a corrupt header is
# meant to produce. The mapping and the failure branches are covered against a
# stubbed parser in test_providers.py.
_needs_gguf_parser = pytest.mark.skipif(
    not has_gguf_parser(), reason="no gguf-parser on PATH (installed in the integration lane)"
)


@pytest.mark.real_engine_resolution
@_needs_gguf_parser
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
    ],
)
@pytest.mark.real_engine_resolution
@_needs_gguf_parser
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

        monkeypatch.setattr("lilbee.core.system.default_cache_dir", lambda: tmp_path)
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

    def test_a_wrong_shaped_cache_entry_is_ignored(self, monkeypatch, tmp_path) -> None:
        """Valid JSON of the wrong shape must not be served as metadata."""
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
            entry.write_text('{"metadata": {"ctx": 4096}}', encoding="utf-8")  # int, not str
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        assert len(calls) == 2

    def test_a_payload_without_the_metadata_key_is_ignored(self, monkeypatch, tmp_path) -> None:
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
            entry.write_text('{"other": 1}', encoding="utf-8")
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        assert len(calls) == 2

    def test_no_usable_state_dir_falls_back_to_parsing(self, monkeypatch, tmp_path) -> None:
        """With nowhere to cache, reads still work -- they just re-parse."""
        gguf_meta = self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(gguf_meta, "_disk_cache_file", lambda _key: None)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 32)
        calls: list[str] = []
        monkeypatch.setattr(
            gguf_meta,
            "_read_gguf_metadata_uncached",
            lambda p: (calls.append(str(p)), {"context_length": "4096"})[1],
        )
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        gguf_meta._METADATA_CACHE.clear()
        assert gguf_meta.read_gguf_metadata(model) == {"context_length": "4096"}
        assert len(calls) == 2  # nothing persisted, so the second read re-parses


class TestParserAbsenceIsNotCached:
    """A read the parser never answered must not become "this file has no metadata"."""

    _PAYLOAD = json.dumps(
        {
            "header": {
                "metadataKV": [
                    {"key": "general.architecture", "valueType": 8, "value": "bert"},
                    {"key": "bert.embedding_length", "valueType": 4, "value": 1024},
                ]
            }
        }
    )

    def _model(self, monkeypatch, tmp_path):
        from lilbee.providers import gguf_meta

        monkeypatch.setattr("lilbee.core.system.default_cache_dir", lambda: tmp_path / "cache")
        self._new_process(gguf_meta)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 32)
        return gguf_meta, model

    @staticmethod
    def _new_process(gguf_meta) -> None:
        """Empty every in-memory record, as a fresh process starts with."""
        gguf_meta._METADATA_CACHE.clear()
        gguf_meta._UNANSWERED.clear()

    @staticmethod
    def _no_parser(monkeypatch) -> None:
        def missing():
            raise FileNotFoundError("no gguf-parser")

        monkeypatch.setattr("lilbee.providers.fleet.binary.resolve_gguf_parser", missing)

    def _parser(self, monkeypatch, runner) -> list[object]:
        spawns: list[object] = []

        def run(*args, **kwargs):
            spawns.append(args)
            return runner()

        monkeypatch.setattr(
            "lilbee.providers.fleet.binary.resolve_gguf_parser", lambda: Path("/fake/gguf-parser")
        )
        monkeypatch.setattr("lilbee.providers.fleet.proc.run_bounded", run)
        return spawns

    def test_a_new_process_reads_a_file_the_last_one_found_no_parser_for(
        self, monkeypatch, tmp_path
    ) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)
        self._no_parser(monkeypatch)
        assert gguf_meta.read_gguf_metadata(model) is None
        self._parser(monkeypatch, lambda: (self._PAYLOAD, 0))
        # A new process: nothing may have been written to disk.
        self._new_process(gguf_meta)
        assert gguf_meta.read_gguf_metadata(model) == {
            "architecture": "bert",
            "embedding_length": "1024",
        }

    def test_one_lookup_per_process_until_an_install_and_no_disk_entry(
        self, monkeypatch, tmp_path
    ) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)
        lookups: list[int] = []

        def missing():
            lookups.append(1)
            raise FileNotFoundError("no gguf-parser")

        monkeypatch.setattr("lilbee.providers.fleet.binary.resolve_gguf_parser", missing)
        assert gguf_meta.read_gguf_metadata(model) is None
        assert gguf_meta.read_gguf_metadata(model) is None
        assert len(lookups) == 1
        assert list((tmp_path / "cache").rglob("*.json")) == []

    def test_forgetting_unanswered_reads_keeps_the_parser_answers(
        self, monkeypatch, tmp_path
    ) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)
        other = tmp_path / "other.gguf"
        other.write_bytes(b"y" * 32)
        spawns = self._parser(monkeypatch, lambda: ("", 1))
        assert gguf_meta.read_gguf_metadata(model) is None  # the parser's own answer
        self._no_parser(monkeypatch)
        assert gguf_meta.read_gguf_metadata(other) is None  # no parser to ask
        gguf_meta.forget_unanswered_reads()
        self._parser(monkeypatch, lambda: (self._PAYLOAD, 0))
        assert gguf_meta.read_gguf_metadata(other) == {
            "architecture": "bert",
            "embedding_length": "1024",
        }
        assert gguf_meta.read_gguf_metadata(model) is None  # still the cached answer
        assert len(spawns) == 1

    def test_parser_that_times_out_is_asked_again(self, monkeypatch, tmp_path) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)

        def timeout():
            raise subprocess.TimeoutExpired("gguf-parser", 60)

        spawns = self._parser(monkeypatch, timeout)
        assert gguf_meta.read_gguf_metadata(model) is None
        self._new_process(gguf_meta)
        self._parser(monkeypatch, lambda: (self._PAYLOAD, 0))
        assert gguf_meta.read_gguf_metadata(model) == {
            "architecture": "bert",
            "embedding_length": "1024",
        }
        assert len(spawns) == 1

    def test_parser_exit_with_an_error_is_cached(self, monkeypatch, tmp_path) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)
        spawns = self._parser(monkeypatch, lambda: ("", 1))
        assert gguf_meta.read_gguf_metadata(model) is None
        gguf_meta._METADATA_CACHE.clear()
        assert gguf_meta.read_gguf_metadata(model) is None
        assert len(spawns) == 1

    def test_unreadable_parser_output_is_cached(self, monkeypatch, tmp_path) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)
        spawns = self._parser(monkeypatch, lambda: ("not json", 0))
        assert gguf_meta.read_gguf_metadata(model) is None
        assert gguf_meta.read_gguf_metadata(model) is None
        assert len(spawns) == 1

    def test_disk_entries_from_an_older_reader_are_ignored(self, monkeypatch, tmp_path) -> None:
        gguf_meta, model = self._model(monkeypatch, tmp_path)
        stat = model.stat()
        key = (str(model), stat.st_mtime_ns, stat.st_size)
        stale = gguf_meta._disk_cache_file(key)
        assert stale is not None
        # Version 2 is the last reader that cached a no-parser read as "no metadata".
        digest = hashlib.sha256("\0".join(str(p) for p in (2, *key)).encode()).hexdigest()
        stale.parent.mkdir(parents=True, exist_ok=True)
        (stale.parent / f"{digest}.json").write_text('{"metadata": null}', encoding="utf-8")
        self._parser(monkeypatch, lambda: (self._PAYLOAD, 0))
        assert gguf_meta.read_gguf_metadata(model) == {
            "architecture": "bert",
            "embedding_length": "1024",
        }


@pytest.mark.real_engine_resolution
@_needs_gguf_parser
def test_duplicate_metadata_key_is_read_rather_than_refused(tmp_path: Path) -> None:
    """A repeated key parses, because the engine's own reader accepts one.

    gguf-py raised KeyError on the second copy and the whole file read as
    unusable. gguf-parser takes a value and carries on, so a file llama-server
    would load no longer reads as having no metadata at all.
    """
    f = tmp_path / "dup.gguf"
    f.write_bytes(_gguf_with_duplicate_key())
    assert read_gguf_metadata(f) == {"name": "x"}


_PATH_KEYED_PARSER = """#!{python}
import sys

if sys.argv[1:3] != ["--path", {model!r}]:
    sys.exit(1)
print({payload!r})
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in parser is a shebang script")
def test_projector_type_is_read_from_the_file_asked_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser is asked about the mmproj file itself, not about nothing."""
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"GGUF")
    payload = json.dumps(
        {"header": {"metadataKV": [{"key": "clip.projector_type", "valueType": 8, "value": "ldp"}]}}
    )
    exe = tmp_path / "fake-gguf-parser"
    exe.write_text(
        _PATH_KEYED_PARSER.format(python=sys.executable, model=str(mmproj), payload=payload),
        encoding="utf-8",
    )
    exe.chmod(0o755)
    monkeypatch.setattr("lilbee.providers.fleet.binary.resolve_gguf_parser", lambda: exe)
    assert read_mmproj_projector_type(mmproj) == "ldp"
