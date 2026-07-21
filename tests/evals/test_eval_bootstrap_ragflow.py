"""Corpus discovery for the RAGFlow arm.

``lilbee add`` ingests a directory recursively, so the RAGFlow side has to walk
the same tree or the two arms index different corpora and the comparison is not
about retrieval at all.
"""

import pytest
from evals.benchmark.bootstrap_ragflow import iter_corpus_files, upload_corpus


def _corpus(root):
    (root / "a.txt").write_text("top level")
    nested = root / "nested" / "deeper"
    nested.mkdir(parents=True)
    (nested / "b.txt").write_text("buried")
    (root / "nested" / "c.txt").write_text("one down")
    return root


def test_corpus_walk_is_recursive(tmp_path):
    found = iter_corpus_files(_corpus(tmp_path))
    assert [p.name for p in found] == ["a.txt", "c.txt", "b.txt"]


def test_corpus_walk_skips_directories(tmp_path):
    # A bare iterdir() would yield the directory itself and blow up on read_bytes.
    assert all(path.is_file() for path in iter_corpus_files(_corpus(tmp_path)))


def test_corpus_walk_is_deterministic(tmp_path):
    corpus = _corpus(tmp_path)
    assert iter_corpus_files(corpus) == iter_corpus_files(corpus)


def test_empty_corpus_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="no files found"):
        upload_corpus(_RecordingClient([]), "ds1", tmp_path)


class _RecordingClient:
    """Captures the multipart batches upload_corpus posts."""

    def __init__(self, ids_per_batch):
        self._ids_per_batch = list(ids_per_batch)
        self.batches = []

    def post(self, route, files=None, json=None):
        self.batches.append([name for _field, (name, _body) in files])
        ids = self._ids_per_batch.pop(0)
        return _FakeResponse({"code": 0, "data": [{"id": i} for i in ids]})


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_upload_sends_nested_paths_relative_to_the_corpus_root(tmp_path):
    client = _RecordingClient([["1", "2", "3"]])
    upload_corpus(client, "ds1", _corpus(tmp_path), batch_size=10)
    # Nested files keep a path-qualified name so they stay distinct documents.
    assert client.batches == [["a.txt", "nested/c.txt", "nested/deeper/b.txt"]]


def test_uploaded_names_are_os_independent(tmp_path):
    # The document name is the identifier the run is scored against. Windows
    # would give backslashes from str(), so the same corpus would be indexed
    # under different names depending on who ran the upload.
    client = _RecordingClient([["1", "2", "3"]])
    upload_corpus(client, "ds1", _corpus(tmp_path), batch_size=10)
    names = client.batches[0]
    assert not any("\\" in name for name in names)
    assert all(name == name.replace("\\", "/") for name in names)


def test_upload_batches_rather_than_posting_one_giant_request(tmp_path):
    client = _RecordingClient([["1", "2"], ["3"]])
    ids = upload_corpus(client, "ds1", _corpus(tmp_path), batch_size=2)
    assert [len(batch) for batch in client.batches] == [2, 1]
    assert ids == ["1", "2", "3"]
