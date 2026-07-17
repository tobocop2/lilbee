"""TREC run-file round-trip and the chunk-to-document collapse."""

from evals.benchmark.runfile import (
    ChunkHit,
    RunEntry,
    collapse_hits,
    read_run,
    run_to_pytrec,
    write_run,
)


def test_collapse_keeps_best_score_per_document_and_reranks():
    hits = [
        ChunkHit("q1", "docA", 0.2),
        ChunkHit("q1", "docA", 0.9),  # same doc, higher score wins
        ChunkHit("q1", "docB", 0.5),
    ]
    entries = collapse_hits(hits, "tag")
    assert [(e.doc_id, e.rank, e.score) for e in entries] == [
        ("docA", 1, 0.9),
        ("docB", 2, 0.5),
    ]


def test_collapse_breaks_score_ties_on_doc_id_deterministically():
    hits = [ChunkHit("q1", "docB", 0.5), ChunkHit("q1", "docA", 0.5)]
    entries = collapse_hits(hits, "tag")
    assert [e.doc_id for e in entries] == ["docA", "docB"]


def test_collapse_groups_by_query():
    hits = [ChunkHit("q2", "d", 0.1), ChunkHit("q1", "d", 0.1)]
    entries = collapse_hits(hits, "tag")
    assert sorted({e.query_id for e in entries}) == ["q1", "q2"]


def test_run_file_round_trips(tmp_path):
    entries = collapse_hits([ChunkHit("q1", "docA", 0.9), ChunkHit("q1", "docB", 0.5)], "run7")
    path = tmp_path / "run.trec"
    write_run(path, entries)
    line = path.read_text().splitlines()[0]
    assert line == "q1 Q0 docA 1 0.900000 run7"
    assert read_run(path) == entries


def test_from_line_parses_all_columns():
    entry = RunEntry.from_line("q9 Q0 doc42 3 1.500000 mytag")
    assert entry == RunEntry("q9", "doc42", 3, 1.5, "mytag")


def test_run_to_pytrec_shapes_query_doc_score():
    entries = [RunEntry("q1", "d1", 1, 0.9, "t"), RunEntry("q1", "d2", 2, 0.4, "t")]
    assert run_to_pytrec(entries) == {"q1": {"d1": 0.9, "d2": 0.4}}
