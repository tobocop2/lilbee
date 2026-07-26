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


def _ceiling(uri: str) -> int:
    """The first rowid no existing row reaches, from the live fragment list."""
    import lance

    max_fragment = max((f.fragment_id for f in lance.dataset(uri).get_fragments()), default=-1)
    return (max_fragment + 1) << 32


def test_worker_appends_then_parent_ceiling_cleanup_keeps_exactly_the_new_rows(tmp_path):
    """The fragment-ingest loop: a worker re-ingests a source and commits its
    fragment; the parent's cleanup, bounded by the pre-run ceiling, removes
    exactly the old rows and none of the ones the worker just appended."""
    db, uri = _new_table(tmp_path)
    table = db.open_table("chunks")
    table.add(_records("doc-a", 3))  # pre-run rows
    ceiling = _ceiling(uri)

    assert append_chunk_fragment(uri, _records("doc-a", 5), _schema()) == 5
    db.open_table("chunks").delete(f"source IN ('doc-a') AND _rowid < {ceiling}")

    table = db.open_table("chunks")
    assert table.count_rows() == 5
    assert sorted(r["chunk"] for r in table.search().to_list()) == [f"doc-a {i}" for i in range(5)]


def test_concurrent_appends_and_ceiling_deletes_lose_no_rows(tmp_path):
    """A parent cleanup delete racing worker appends touches only pre-run rows."""
    db, uri = _new_table(tmp_path)
    db.open_table("chunks").add(_records("doc-z", 2))  # pre-run rows
    ceiling = _ceiling(uri)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(append_chunk_fragment, uri, _records(f"doc-{w}", 50), _schema())
            for w in range(4)
        ]
        futures.append(
            pool.submit(
                lambda: db.open_table("chunks").delete(
                    f"source IN ('doc-z') AND _rowid < {ceiling}"
                )
            )
        )
        totals = [f.result() for f in futures]

    assert sum(t for t in totals if isinstance(t, int)) == 200
    # doc-z's 2 pre-run rows are gone; every appended row landed.
    assert db.open_table("chunks").count_rows() == 200
