"""Merging shard indexes: what the merge must refuse, and what it must not hold in RAM.

A partitioned ingest is only worth running if the merge cannot quietly produce a
wrong index. The refusal tests below each describe one way a shard set can be
wrong (a shard missing, a shard embedded by a different model, a shard truncated
after its ingest, concept clusters that were only ever computed per shard) and
assert the merge stops instead of writing. The last test asserts the property
that decides whether the merge can run at corpus scale at all: peak memory
tracks the batch size, not the corpus.
"""

import json
import os
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest
from evals.infra.merge_shards import (
    SINGLETON_TABLES,
    MergeRefusedError,
    _iter_batches,
    _table_names,
    merge,
    meta_identity,
    observed_counts,
    read_manifest,
    verify_shards,
)

DIM = 8
MODEL = "Qwen3-Embedding-8B-Q8_0"
SCHEMA_VERSION = 3


def build_shard(
    root: Path,
    *,
    shard_index: int,
    shard_count: int,
    doc_ids: list[str],
    dim: int = DIM,
    model: str = MODEL,
    schema_version: int = SCHEMA_VERSION,
    concept_rows: int = 0,
    manifest_overrides: dict | None = None,
    write_manifest: bool = True,
) -> Path:
    """A synthetic lilbee shard: the tables the merge touches, plus its manifest."""
    db_dir = root / "data" / "lancedb"
    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    n = len(doc_ids)

    db.create_table(
        "chunks",
        pa.table(
            {
                "id": pa.array([f"{d}#0" for d in doc_ids]),
                "source": pa.array([f"{i // 1000:05d}/{d}.txt" for i, d in enumerate(doc_ids)]),
                "chunk": pa.array([f"text of {d}" for d in doc_ids]),
                "vector": pa.array(
                    [[float(i % 7)] * dim for i in range(n)], pa.list_(pa.float32(), dim)
                ),
            }
        ),
    )
    db.create_table(
        "_sources",
        pa.table(
            {
                "source": pa.array([f"{i // 1000:05d}/{d}.txt" for i, d in enumerate(doc_ids)]),
                "hash": pa.array([f"h{d}" for d in doc_ids]),
            }
        ),
    )
    db.create_table(
        "_page_texts",
        pa.table(
            {
                "source": pa.array([f"{i // 1000:05d}/{d}.txt" for i, d in enumerate(doc_ids)]),
                "page": pa.array([0] * n, pa.int64()),
                "text": pa.array([f"text of {d}" for d in doc_ids]),
            }
        ),
    )
    db.create_table(
        "_meta",
        pa.table(
            {
                "embedding_model": pa.array([model]),
                "embedding_dim": pa.array([dim], pa.int64()),
                "schema_version": pa.array([schema_version], pa.int64()),
                "updated_at": pa.array([f"2026-07-28T0{shard_index}:00:00+00:00"]),
            }
        ),
    )
    if concept_rows:
        db.create_table(
            "concept_nodes",
            pa.table(
                {
                    "cluster_id": pa.array(list(range(concept_rows)), pa.int64()),
                    "label": pa.array([f"c{i}" for i in range(concept_rows)]),
                }
            ),
        )

    (root / "config.toml").write_text(f'embedding_model = "{model}"\nembedding_dim = {dim}\n')

    if write_manifest:
        manifest = {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "embedding_model": model,
            "embedding_dim": dim,
            "schema_version": schema_version,
            "table_rows": {name: db.open_table(name).count_rows() for name in _table_names(db)},
        }
        manifest.update(manifest_overrides or {})
        (root / "shard_manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def two_shards(tmp_path: Path, **kw) -> list[str]:
    """A well-formed two-shard set holding six disjoint docs."""
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0", "d2", "d4"], **kw)
    b = build_shard(tmp_path / "s1", shard_index=1, shard_count=2, doc_ids=["d1", "d3", "d5"], **kw)
    return [str(a), str(b)]


def check(roots: list[str]) -> None:
    """Run the guardrails over a shard set exactly as merge() does."""
    manifests = [read_manifest(r) for r in roots]
    dbs = [lancedb.connect(os.path.join(r, "data", "lancedb")) for r in roots]
    observed = {r: observed_counts(db) for r, db in zip(roots, dbs, strict=True)}
    meta = {r: meta_identity(db) for r, db in zip(roots, dbs, strict=True)}
    verify_shards(manifests, observed, meta)


# --------------------------------------------------------------- completeness


def test_a_shard_without_a_manifest_cannot_be_merged(tmp_path):
    # An unmanifested shard cannot state which slice it holds, so no set
    # containing it can be shown to be complete.
    roots = two_shards(tmp_path)
    (Path(roots[1]) / "shard_manifest.json").unlink()
    with pytest.raises(MergeRefusedError, match=r"no shard_manifest\.json"):
        check(roots)


def test_a_missing_shard_refuses_instead_of_merging_a_partial_corpus(tmp_path):
    # The defect this exists for: merging 2 of 3 shards used to succeed and
    # produce an index silently missing a third of the corpus.
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=3, doc_ids=["d0", "d3"])
    b = build_shard(tmp_path / "s1", shard_index=1, shard_count=3, doc_ids=["d1", "d4"])
    with pytest.raises(MergeRefusedError, match=r"incomplete shard set.*Missing indices: \[2\]"):
        check([str(a), str(b)])


def test_the_same_shard_supplied_twice_refuses(tmp_path):
    # Duplicating a root would double its rows and still leave a slice missing.
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(tmp_path / "s0b", shard_index=0, shard_count=2, doc_ids=["d0"])
    with pytest.raises(MergeRefusedError, match="shard_index 0 supplied twice"):
        check([str(a), str(b)])


def test_shards_that_disagree_on_the_set_size_refuse(tmp_path):
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(tmp_path / "s1", shard_index=1, shard_count=4, doc_ids=["d1"])
    with pytest.raises(MergeRefusedError, match="disagree on shard_count"):
        check([str(a), str(b)])


# ------------------------------------------------------------------ identity


def test_shards_embedded_by_different_models_refuse(tmp_path):
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(
        tmp_path / "s1", shard_index=1, shard_count=2, doc_ids=["d1"], model="some-other-embedder"
    )
    with pytest.raises(MergeRefusedError, match="embedded differently"):
        check([str(a), str(b)])


def test_shards_embedded_at_different_dimensions_refuse(tmp_path):
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(tmp_path / "s1", shard_index=1, shard_count=2, doc_ids=["d1"], dim=16)
    with pytest.raises(MergeRefusedError, match="embedded differently"):
        check([str(a), str(b)])


def test_a_manifest_that_contradicts_its_own_index_refuses(tmp_path):
    # The manifest is a claim about the shard; the shard's _meta row is the
    # fact. A shard whose claim was edited, or copied from another run, must
    # not be taken at its word.
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(
        tmp_path / "s1",
        shard_index=1,
        shard_count=2,
        doc_ids=["d1"],
        manifest_overrides={"embedding_model": MODEL},
        model=MODEL,
    )
    # Rewrite the shard's own _meta row to a different model, leaving the
    # manifest claiming the expected one.
    db = lancedb.connect(str(Path(b) / "data" / "lancedb"))
    db.drop_table("_meta")
    db.create_table(
        "_meta",
        pa.table(
            {
                "embedding_model": pa.array(["drifted-embedder"]),
                "embedding_dim": pa.array([DIM], pa.int64()),
                "schema_version": pa.array([SCHEMA_VERSION], pa.int64()),
                "updated_at": pa.array(["2026-07-28T01:00:00+00:00"]),
            }
        ),
    )
    with pytest.raises(MergeRefusedError, match="_meta row says drifted-embedder"):
        check([str(a), str(b)])


def test_shards_at_different_schema_versions_refuse(tmp_path):
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(
        tmp_path / "s1", shard_index=1, shard_count=2, doc_ids=["d1"], schema_version=99
    )
    with pytest.raises(MergeRefusedError, match="embedded differently"):
        check([str(a), str(b)])


# -------------------------------------------------------------- shard is intact


def test_a_shard_that_lost_rows_after_its_ingest_refuses(tmp_path):
    # A shard restored from a stale checkpoint, or truncated by a failed copy,
    # holds fewer rows than it reported. Merging it loses documents silently.
    roots = two_shards(tmp_path)
    manifest_path = Path(roots[1]) / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["table_rows"]["chunks"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(MergeRefusedError, match=r"table chunks holds 3 rows but its manifest"):
        check(roots)


# ------------------------------------------------------------- concept graph


def test_shards_carrying_concept_clusters_refuse(tmp_path):
    # Cluster ids mean nothing outside the shard that assigned them, and
    # corpus-wide re-clustering is not implemented, so this must not merge.
    roots = two_shards(tmp_path, concept_rows=4)
    with pytest.raises(MergeRefusedError, match=r"concept-graph rows.*concept_nodes"):
        check(roots)


def test_the_concept_refusal_names_the_way_out(tmp_path):
    roots = two_shards(tmp_path, concept_rows=2)
    with pytest.raises(MergeRefusedError, match="Disable the concept wiki for partitioned runs"):
        check(roots)


# --------------------------------------------------------------- happy path


def test_a_well_formed_shard_set_merges_every_row_exactly_once(tmp_path):
    roots = two_shards(tmp_path)
    totals = merge(roots, str(tmp_path / "merged"))
    assert totals["chunks"] == 6
    assert totals["_sources"] == 6
    assert totals["_page_texts"] == 6

    db = lancedb.connect(str(tmp_path / "merged" / "data" / "lancedb"))
    ids = sorted(db.open_table("chunks").to_arrow().column("id").to_pylist())
    assert ids == [f"d{i}#0" for i in range(6)]


def test_the_merged_meta_holds_exactly_one_row(tmp_path):
    # Concatenating _meta used to leave one row per shard, so the store's
    # identity depended on which row a reader happened to pick.
    roots = two_shards(tmp_path)
    totals = merge(roots, str(tmp_path / "merged"))
    assert totals["_meta"] == 1

    db = lancedb.connect(str(tmp_path / "merged" / "data" / "lancedb"))
    rows = db.open_table("_meta").search().limit(None).to_list()
    assert len(rows) == 1
    assert rows[0]["embedding_model"] == MODEL
    assert rows[0]["embedding_dim"] == DIM


def test_the_merge_is_rerunnable_over_a_partial_previous_attempt(tmp_path):
    roots = two_shards(tmp_path)
    merged = str(tmp_path / "merged")
    merge(roots, merged)
    totals = merge(roots, merged)
    assert totals["chunks"] == 6
    assert totals["_meta"] == 1


def test_a_refused_merge_writes_no_tables(tmp_path):
    # Refusing after writing half the tables would leave a plausible-looking
    # index behind for the next command to pick up.
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=3, doc_ids=["d0"])
    b = build_shard(tmp_path / "s1", shard_index=1, shard_count=3, doc_ids=["d1"])
    merged = tmp_path / "merged"
    with pytest.raises(MergeRefusedError):
        merge([str(a), str(b)], str(merged))
    assert not (merged / "data" / "lancedb").exists()


def test_shards_built_by_different_lilbee_versions_refuse(tmp_path):
    # Differing columns on the same table means the shards were produced by
    # different code, which the row-count check cannot see.
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=2, doc_ids=["d0"])
    b = build_shard(tmp_path / "s1", shard_index=1, shard_count=2, doc_ids=["d1"])
    db = lancedb.connect(str(Path(b) / "data" / "lancedb"))
    db.drop_table("_sources")
    db.create_table(
        "_sources",
        pa.table(
            {
                "source": pa.array(["00000/d1.txt"]),
                "hash": pa.array(["hd1"]),
                "title": pa.array(["t"]),
            }
        ),
    )
    manifest_path = Path(b) / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["table_rows"]["_sources"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(MergeRefusedError, match="different schemas across shards"):
        merge([str(a), str(b)], str(tmp_path / "merged"))


# ------------------------------------------------------------- memory bound
#
# These assert live Arrow bytes, not RSS. RSS was measured and rejected as the
# instrument: reading a 393MB shard set grows RSS ~279MB even when every batch
# is dropped without being written anywhere, because the pages of the Lance
# files being read stay resident. RSS therefore tracks bytes read, which
# streaming does not change and which the kernel can reclaim. What the merge
# controls, and what the defect was about, is how much of the corpus it holds
# as Arrow data at once: the old code built one table containing every row.


def build_vector_shards(tmp_path: Path, *, rows: int, dim: int) -> list[str]:
    """A two-shard set whose chunks tables are large enough to measure."""
    roots = []
    for s in range(2):
        root = tmp_path / f"s{s}"
        db_dir = root / "data" / "lancedb"
        db_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(db_dir))
        db.create_table(
            "chunks",
            pa.table(
                {
                    "id": pa.array([f"d{s}_{i}" for i in range(rows)]),
                    "vector": pa.array([[0.5] * dim] * rows, pa.list_(pa.float32(), dim)),
                }
            ),
        )
        db.create_table(
            "_meta",
            pa.table(
                {
                    "embedding_model": pa.array([MODEL]),
                    "embedding_dim": pa.array([dim], pa.int64()),
                    "schema_version": pa.array([SCHEMA_VERSION], pa.int64()),
                    "updated_at": pa.array([f"2026-07-28T0{s}:00:00+00:00"]),
                }
            ),
        )
        (root / "shard_manifest.json").write_text(
            json.dumps(
                {
                    "shard_index": s,
                    "shard_count": 2,
                    "embedding_model": MODEL,
                    "embedding_dim": dim,
                    "schema_version": SCHEMA_VERSION,
                    "table_rows": {
                        "chunks": rows,
                        "_meta": 1,
                    },
                }
            )
        )
        roots.append(str(root))
    return roots


@pytest.mark.parametrize("rows", [1500, 6000])
def test_the_batch_stream_holds_the_same_bytes_however_big_the_corpus_is(tmp_path, rows):
    # The property that decides whether 8.8M x 4096-dim can merge at all: what
    # the merge holds must depend on the batch size, not the corpus size. A 4x
    # corpus must not mean 4x held bytes.
    dim, batch_rows = 512, 256
    roots = build_vector_shards(tmp_path, rows=rows, dim=dim)
    dbs = [lancedb.connect(os.path.join(r, "data", "lancedb")) for r in roots]

    widest = 0
    counted = 0
    for b in _iter_batches(dbs, "chunks", batch_rows):
        assert b.num_rows <= batch_rows
        widest = max(widest, b.nbytes)
        counted += b.num_rows

    assert counted == 2 * rows
    # One batch of 256 rows x 512 f32 is ~0.5MB whatever the corpus is.
    assert widest <= batch_rows * dim * 4 * 1.5, f"a single batch held {widest / 1e6:.1f}MB"


def test_the_merge_never_materialises_a_whole_table(tmp_path, monkeypatch):
    # The defect concretely: the old merge called to_arrow() on every shard
    # table and concat_tables'd the results, so one Arrow table held the entire
    # corpus of vectors (~140GB at 8.8M rows). Only the single-row tables may
    # be read whole.
    materialised: list[tuple[str, int]] = []
    original = lancedb.table.LanceTable.to_arrow

    def spy(self):
        result = original(self)
        materialised.append((self.name, result.num_rows))
        return result

    monkeypatch.setattr(lancedb.table.LanceTable, "to_arrow", spy)

    roots = two_shards(tmp_path)
    merge(roots, str(tmp_path / "merged"))

    whole_tables = [name for name, _ in materialised if name not in SINGLETON_TABLES]
    assert not whole_tables, f"merge read these tables whole instead of streaming: {whole_tables}"
    assert all(n <= 1 for _, n in materialised)


# ------------------------------------------------- malformed shard descriptions


def test_a_corrupt_manifest_refuses_rather_than_being_half_read(tmp_path):
    # A truncated upload or an interrupted write leaves valid-looking JSON on
    # disk. Refusing beats guessing which half is trustworthy.
    roots = two_shards(tmp_path)
    (Path(roots[1]) / "shard_manifest.json").write_text('{"shard_index": 1, "shard_c')
    with pytest.raises(MergeRefusedError, match="unreadable manifest"):
        check(roots)


def test_a_manifest_missing_a_required_field_refuses(tmp_path):
    # An older ingest.sh wrote fewer fields. Merging on the ones present would
    # skip whichever guardrail the missing field feeds.
    roots = two_shards(tmp_path)
    path = Path(roots[1]) / "shard_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["embedding_dim"]
    del manifest["schema_version"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(MergeRefusedError, match="missing embedding_dim, schema_version"):
        check(roots)


def test_a_shard_index_outside_the_declared_set_refuses(tmp_path):
    # shard 3 of a 2-shard set cannot be a real slice of anything.
    roots = two_shards(tmp_path)
    path = Path(roots[1]) / "shard_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shard_index"] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(MergeRefusedError, match=r"shard_index 3 outside 0\.\.1"):
        check(roots)


def test_a_shard_with_no_meta_table_at_all_refuses(tmp_path):
    # Distinct from a _meta that disagrees: here there is nothing to check the
    # manifest's claim against, so the claim cannot be trusted.
    roots = two_shards(tmp_path)
    db = lancedb.connect(os.path.join(roots[1], "data", "lancedb"))
    db.drop_table("_meta")
    path = Path(roots[1]) / "shard_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["table_rows"].pop("_meta", None)
    path.write_text(json.dumps(manifest))
    with pytest.raises(MergeRefusedError, match="no _meta row"):
        check(roots)


def test_a_meta_table_that_exists_but_holds_no_row_refuses(tmp_path):
    roots = two_shards(tmp_path)
    db = lancedb.connect(os.path.join(roots[1], "data", "lancedb"))
    schema = db.open_table("_meta").schema
    db.drop_table("_meta")
    db.create_table("_meta", schema=schema)
    path = Path(roots[1]) / "shard_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["table_rows"]["_meta"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(MergeRefusedError, match="no _meta row"):
        check(roots)


def test_one_shard_is_not_a_merge(tmp_path):
    a = build_shard(tmp_path / "s0", shard_index=0, shard_count=1, doc_ids=["d0"])
    with pytest.raises(MergeRefusedError, match="at least 2 shards"):
        merge([str(a)], str(tmp_path / "merged"))


# ------------------------------------------------------- uneven table presence


def test_a_table_only_one_shard_has_still_merges(tmp_path):
    # Optional tables (citations, memories) exist only where something wrote
    # them. The shard without one must contribute no rows rather than abort.
    roots = two_shards(tmp_path)
    db = lancedb.connect(os.path.join(roots[0], "data", "lancedb"))
    db.create_table(
        "_citations",
        pa.table({"source": pa.array(["00000/d0.txt"]), "cite": pa.array(["ref"])}),
    )
    path = Path(roots[0]) / "shard_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["table_rows"]["_citations"] = 1
    path.write_text(json.dumps(manifest))

    totals = merge(roots, str(tmp_path / "merged"))
    assert totals["_citations"] == 1
    assert totals["chunks"] == 6


def test_a_singleton_table_no_shard_has_is_simply_absent(tmp_path):
    # _entity_schema only exists once entities have been induced. Its absence
    # must not fabricate an empty row.
    roots = two_shards(tmp_path)
    totals = merge(roots, str(tmp_path / "merged"))
    assert "_entity_schema" not in totals


def test_a_table_every_shard_has_but_none_filled_survives_as_an_empty_table(tmp_path):
    # A shard that created a table without writing rows still has to appear in
    # the merged index with its schema, or the next writer infers a new one.
    roots = two_shards(tmp_path)
    for root in roots:
        db = lancedb.connect(os.path.join(root, "data", "lancedb"))
        db.create_table(
            "_memories",
            schema=pa.schema([("source", pa.utf8()), ("note", pa.utf8())]),
        )
        path = Path(root) / "shard_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["table_rows"]["_memories"] = 0
        path.write_text(json.dumps(manifest))

    totals = merge(roots, str(tmp_path / "merged"))
    assert totals["_memories"] == 0
    db = lancedb.connect(str(tmp_path / "merged" / "data" / "lancedb"))
    assert db.open_table("_memories").schema.names == ["source", "note"]


def test_an_empty_singleton_table_contributes_no_row(tmp_path):
    # _entity_schema created but never written: the merge must not invent a row
    # for a schema that was never induced.
    roots = two_shards(tmp_path)
    for root in roots:
        db = lancedb.connect(os.path.join(root, "data", "lancedb"))
        db.create_table(
            "_entity_schema",
            schema=pa.schema([("schema_json", pa.utf8()), ("updated_at", pa.utf8())]),
        )
        path = Path(root) / "shard_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["table_rows"]["_entity_schema"] = 0
        path.write_text(json.dumps(manifest))

    totals = merge(roots, str(tmp_path / "merged"))
    assert totals["_entity_schema"] == 0
