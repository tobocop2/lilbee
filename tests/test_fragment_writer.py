"""Unit tests for the chunk fragment writer, with lance mocked.

These run in the standard (no-extras) suite where pylance is absent, so lance is
injected as a fake module. The real pylance round-trip lives in
``tests/integration/test_fragment_writer_integration.py``.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa

from lilbee.data.ingest import fragment_writer


def _install_fake_lance(monkeypatch, *, calls: dict) -> list:
    """Inject a fake ``lance`` / ``lance.fragment`` and record what the writer calls."""
    fragments = [object(), object()]

    fake_fragment = types.ModuleType("lance.fragment")

    def write_fragments(table, uri):
        calls["table_rows"] = table.num_rows
        calls["write_uri"] = uri
        return fragments

    fake_fragment.write_fragments = write_fragments

    fake_lance = types.ModuleType("lance")
    fake_lance.fragment = fake_fragment
    fake_lance.dataset = lambda uri: types.SimpleNamespace(version=7)
    fake_lance.LanceOperation = types.SimpleNamespace(Append=lambda frags: ("append", frags))

    def commit(uri, operation, *, read_version, max_retries):
        calls["commit"] = {
            "uri": uri,
            "operation": operation,
            "read_version": read_version,
            "max_retries": max_retries,
        }

    fake_lance.LanceDataset = types.SimpleNamespace(commit=commit)

    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "lance.fragment", fake_fragment)
    return fragments


def _schema() -> pa.Schema:
    return pa.schema([pa.field("chunk", pa.utf8()), pa.field("vector", pa.list_(pa.float32(), 2))])


def test_fragments_available_true_when_lance_imports(monkeypatch):
    _install_fake_lance(monkeypatch, calls={})
    assert fragment_writer.fragments_available() is True


def test_fragments_available_false_when_lance_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "lance", None)  # import lance -> ImportError
    assert fragment_writer.fragments_available() is False


def test_append_writes_fragment_and_commits_appended(monkeypatch):
    calls: dict = {}
    fragments = _install_fake_lance(monkeypatch, calls=calls)
    records = [{"chunk": "a", "vector": [1.0, 2.0]}, {"chunk": "b", "vector": [3.0, 4.0]}]

    written = fragment_writer.append_chunk_fragment("/db/chunks.lance", records, _schema())

    assert written == 2
    assert calls["table_rows"] == 2
    assert calls["write_uri"] == "/db/chunks.lance"
    assert calls["commit"]["operation"] == ("append", fragments)
    assert calls["commit"]["read_version"] == 7
    assert calls["commit"]["max_retries"] == fragment_writer._COMMIT_MAX_RETRIES


def test_append_empty_records_is_a_noop(monkeypatch):
    # Returns before importing lance, so no fake is needed and nothing is written.
    calls: dict = {}
    _install_fake_lance(monkeypatch, calls=calls)
    assert fragment_writer.append_chunk_fragment("/db/chunks.lance", [], _schema()) == 0
    assert "commit" not in calls
