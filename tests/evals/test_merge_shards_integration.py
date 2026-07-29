"""The merge against real lilbee stores, not hand-built Arrow tables.

The unit tests build shard tables directly, which proves the guardrails and the
streaming but not that the merge agrees with lilbee about what a store looks
like. These build the shards through lilbee's own Store, merge them, rebuild the
indexes the way merge_shards' main() does, and then query the result, so a
schema or table-name drift in lilbee shows up here instead of on a GPU pod.

Vectors are deterministic stand-ins rather than real embeddings: what is under
test is the merge, and embedding identity is already covered by the guardrails.
"""

import contextlib
import math
import os

import lancedb
import pytest
from evals.infra import shard_manifest
from evals.infra.merge_shards import merge, read_manifest

pytest.importorskip("lilbee.data.store")

from lilbee.core.config import Config
from lilbee.data.store import Store

DIM = 16
MODEL = "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
OTHER_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf"


def unit_vector(seed: int) -> list[float]:
    """A deterministic unit vector, so cosine distances are meaningful."""
    raw = [math.sin(seed * (i + 1) * 0.7) for i in range(DIM)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def doc_text(index: int) -> str:
    return f"document {index} about topic{index % 5} with filler words"


@contextlib.contextmanager
def data_root(root):
    """Point lilbee at `root`. data_dir is derived from LILBEE_DATA, not settable."""
    previous = os.environ.get("LILBEE_DATA")
    os.environ["LILBEE_DATA"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LILBEE_DATA", None)
        else:
            os.environ["LILBEE_DATA"] = previous


def open_store(root, model: str = MODEL) -> Store:
    """A Store rooted at `root`, never the developer's real index."""
    with data_root(root):
        store = Store(Config(embedding_model=model, embedding_dim=DIM))
    assert str(root) in store.get_db().uri, "store escaped the test root"
    return store


def build_store(root, doc_indices: list[int], model: str = MODEL) -> None:
    """A real lilbee store holding the given documents, one chunk each."""
    root.mkdir(parents=True, exist_ok=True)
    store = open_store(root, model)
    records = []
    for i in doc_indices:
        source = f"{i // 1000:05d}/p{i}.txt"
        records.append(
            {
                "source": source,
                "content_type": "text/plain",
                "chunk_type": "raw",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "chunk": doc_text(i),
                "chunk_index": 0,
                "title": f"p{i}",
                "vector": unit_vector(i),
            }
        )
    store.add_chunks(records)
    for i in doc_indices:
        store.upsert_source(f"{i // 1000:05d}/p{i}.txt", f"hash{i}", 1)
    store.close()


def write_manifest(root, *, shard_index: int, shard_count: int) -> None:
    """Exactly what ingest.sh runs, so producer and consumer stay in step."""
    shard_manifest.write_manifest(root, shard_index=shard_index, shard_count=shard_count)


@pytest.fixture
def sharded(tmp_path):
    """Two real shards over 0..399 split even/odd, plus the single-host store."""
    everything = list(range(400))
    build_store(tmp_path / "single", everything)

    roots = []
    for index in range(2):
        root = tmp_path / f"s{index}"
        build_store(root, [i for i in everything if i % 2 == index])
        write_manifest(root, shard_index=index, shard_count=2)
        roots.append(str(root))
    return roots, tmp_path


def test_the_merged_store_holds_what_the_single_host_store_holds(sharded):
    roots, tmp_path = sharded
    merged_root = tmp_path / "merged"
    totals = merge(roots, str(merged_root))

    single = lancedb.connect(str(tmp_path / "single" / "data" / "lancedb"))
    got = lancedb.connect(str(merged_root / "data" / "lancedb"))

    assert totals["chunks"] == single.open_table("chunks").count_rows() == 400
    assert got.open_table("_sources").count_rows() == single.open_table("_sources").count_rows()

    merged_sources = sorted(got.open_table("chunks").to_arrow().column("source").to_pylist())
    single_sources = sorted(single.open_table("chunks").to_arrow().column("source").to_pylist())
    assert merged_sources == single_sources


def test_the_merged_store_has_one_meta_row_lilbee_can_read(sharded):
    # get_meta() takes the newest row when there are several, so N rows does not
    # crash; it silently decides identity by timestamp. The merged store must
    # not leave that choice open.
    roots, tmp_path = sharded
    merged_root = tmp_path / "merged"
    merge(roots, str(merged_root))

    db = lancedb.connect(str(merged_root / "data" / "lancedb"))
    assert db.open_table("_meta").count_rows() == 1

    store = open_store(merged_root)
    meta = store.get_meta()
    assert meta is not None
    assert meta["embedding_model"] == MODEL
    assert meta["embedding_dim"] == DIM
    store.close()


def test_the_merged_store_answers_a_vector_query_from_both_shards(sharded, monkeypatch):
    # The step merge_shards' main() runs after the tables land, and the one that
    # can only fail against a real store: lilbee's own index builders over merged
    # fragments, then a query through them. The nearest neighbours of a document
    # from one shard must include documents from the other.
    roots, tmp_path = sharded
    merged_root = tmp_path / "merged"
    merge(roots, str(merged_root))

    monkeypatch.setenv("LILBEE_DATA", str(merged_root))
    store = open_store(merged_root)
    store.ensure_vector_index(force=True)

    hits = store.search(unit_vector(7), top_k=10)
    assert hits, "merged store returned nothing for a vector query"
    sources = {h.source for h in hits}
    evens = {s for s in sources if int(s.split("/p")[1].removesuffix(".txt")) % 2 == 0}
    odds = sources - evens
    assert evens and odds, f"results came from one shard only: {sorted(sources)}"
    store.close()


def test_the_merged_store_gets_a_full_corpus_bm25_index(sharded, monkeypatch):
    # merge_shards rebuilds FTS so the merged index carries corpus-wide BM25
    # statistics rather than per-shard ones. The rebuild is a no-op on lancedb
    # below 0.34, where the index-creation call raises inside a swallowed
    # except, so this also pins the floor the store needs.
    roots, tmp_path = sharded
    merged_root = tmp_path / "merged"
    merge(roots, str(merged_root))

    monkeypatch.setenv("LILBEE_DATA", str(merged_root))
    store = open_store(merged_root)
    store.ensure_fts_index()
    try:
        hits = store.bm25_probe("topic3", top_k=5)
        assert hits, "merged store returned nothing for a term every shard contributed"
        assert all("topic3" in h.chunk for h in hits)
    finally:
        store.close()


def test_a_shard_that_used_a_different_embedder_is_refused_against_real_stores(tmp_path):
    build_store(tmp_path / "s0", [0, 2, 4])
    write_manifest(tmp_path / "s0", shard_index=0, shard_count=2)

    other = tmp_path / "s1"
    other.mkdir(parents=True, exist_ok=True)
    store = open_store(other, OTHER_MODEL)
    store.add_chunks(
        [
            {
                "source": "00000/p1.txt",
                "content_type": "text/plain",
                "chunk_type": "raw",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "chunk": doc_text(1),
                "chunk_index": 0,
                "title": "p1",
                "vector": unit_vector(1),
            }
        ]
    )
    store.close()
    write_manifest(other, shard_index=1, shard_count=2)

    from evals.infra.merge_shards import MergeRefusedError

    with pytest.raises(MergeRefusedError, match="embedded differently"):
        merge([str(tmp_path / "s0"), str(other)], str(tmp_path / "merged"))
    assert not (tmp_path / "merged" / "data" / "lancedb").exists()


def test_the_merged_store_carries_the_shard_config(sharded):
    roots, tmp_path = sharded
    merged_root = tmp_path / "merged"
    merge(roots, str(merged_root))
    # config.toml is how lilbee learns the embedder on a fresh process.
    if os.path.exists(os.path.join(roots[0], "config.toml")):
        assert (merged_root / "config.toml").exists()


def test_the_manifest_writer_produces_what_the_merge_reader_expects(tmp_path):
    # The writer runs on the pod and the reader runs at merge time. A field
    # renamed on one side would only surface as a refused merge after the GPU
    # hours were already spent, so pin the round trip here.
    build_store(tmp_path / "s0", [0, 2, 4])
    written = shard_manifest.write_manifest(
        tmp_path / "s0", shard_index=0, shard_count=2, dataset_id="msmarco-passage", smoke_n=80000
    )
    parsed = read_manifest(str(tmp_path / "s0"))

    assert parsed.shard_index == 0
    assert parsed.shard_count == 2
    assert parsed.embedding_model == written["embedding_model"] == MODEL
    assert parsed.embedding_dim == written["embedding_dim"] == DIM
    assert parsed.table_rows == written["table_rows"]
    assert parsed.table_rows["chunks"] == 3


def test_a_shard_whose_ingest_never_wrote_meta_cannot_claim_an_embedder(tmp_path):
    # An empty _meta means the ingest did not finish. Emitting a manifest anyway
    # would hand the merge a shard with no verifiable identity.
    root = tmp_path / "empty"
    (root / "data" / "lancedb").mkdir(parents=True)
    lancedb.connect(str(root / "data" / "lancedb"))
    with pytest.raises(RuntimeError, match=r"did not complete"):
        shard_manifest.write_manifest(root, shard_index=0, shard_count=2)
