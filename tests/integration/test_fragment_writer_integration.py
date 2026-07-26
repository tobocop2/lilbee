"""Real pylance round-trip for the worker chunk fragment writer.

Runs only where pylance is installed (the ``bulk-ingest`` extra); the unit suite
in ``tests/test_fragment_writer.py`` covers the same code with lance mocked.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

pytest.importorskip("lance")

from lilbee.data.ingest.fragment_writer import append_chunk_fragment

_DIM = 16


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.utf8()),
            pa.field("chunk", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), _DIM)),
        ]
    )


def _records(source: str, n: int) -> list[dict]:
    return [
        {"source": source, "chunk": f"{source} {i}", "chunk_index": i, "vector": [float(i)] * _DIM}
        for i in range(n)
    ]


def _new_table(tmp_path):
    import lancedb

    db = lancedb.connect(str(tmp_path / "lancedb"))
    db.create_table("chunks", schema=_schema())
    return db, str(tmp_path / "lancedb" / "chunks.lance")


def test_append_preserves_rows_and_vectors(tmp_path):
    db, uri = _new_table(tmp_path)
    assert append_chunk_fragment(uri, _records("doc-a", 5), _schema()) == 5
    table = db.open_table("chunks")
    assert table.count_rows() == 5
    row = table.search().where("chunk_index = 3").limit(1).to_list()[0]
    assert row["vector"] == pytest.approx([3.0] * _DIM)
    assert row["source"] == "doc-a"


def test_concurrent_appends_lose_no_rows(tmp_path):
    """Eight workers appending at once must all land (Appends do not conflict)."""
    db, uri = _new_table(tmp_path)
    workers, per = 8, 250
    with ThreadPoolExecutor(max_workers=workers) as pool:
        totals = list(
            pool.map(
                lambda w: append_chunk_fragment(uri, _records(f"doc-{w}", per), _schema()),
                range(workers),
            )
        )
    assert sum(totals) == workers * per
    assert db.open_table("chunks").count_rows() == workers * per
