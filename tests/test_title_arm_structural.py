"""The title arm skips table-of-contents and cover chunks as a document's representative."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from lilbee.core.config import CHUNKS_TABLE, Config, cfg
from lilbee.data.store import SearchChunk, Store, core
from lilbee.retrieval.query.searcher import Searcher

_TOC = (
    "Table of Contents\nIntroduction ........ 1\nHydraulic Systems ........ 4\n"
    "Gaskets and Seals ........ 9\nElectrical ........ 14\nAppendix ........ 22"
)
_BODY_1 = "Replace the hydraulic valve seal every 400 hours. Inspect the cover before refitting it."
_BODY_2 = "Torque the actuator bolts evenly. Record the reading in the log after each service."
_SOUP = "Simmer the onions in butter until soft, then add stock and cook slowly for an hour."
_QUERY = "maintenance manual"


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=cfg.embedding_dim).astype("float32").tolist()


def _record(source: str, index: int, text: str, title: str) -> dict[str, Any]:
    return {
        "source": source,
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": text,
        "chunk_index": index,
        "title": title,
        "vector": _vector(sum(map(ord, source)) + index),
    }


def _store(tmp_path: Path, docs: dict[str, tuple[str, list[str]]], **changes: Any) -> Store:
    config = cfg.model_copy(
        update={"lancedb_dir": tmp_path / "lancedb", "title_search": True, **changes}
    )
    store = Store(config)
    store.add_chunks(
        [
            _record(source, index, text, title)
            for source, (title, chunks) in docs.items()
            for index, text in enumerate(chunks)
        ]
    )
    store.ensure_fts_index()
    store.ensure_title_fts_index()
    return store


def _title_arm(store: Store) -> list[tuple[str, int]]:
    config: Config = store._config
    table = store.open_table(CHUNKS_TABLE)
    indices = core._index_registry(table, config.lancedb_dir)
    rows = store._title_arm(table, _QUERY, 5, None, indices)
    return [(Path(row.source).name, row.chunk_index) for row in store._with_paths(rows)]


_MANUAL = {
    "manual.pdf": ("Maintenance Manual", [_TOC, _BODY_1, _BODY_2]),
    "cooking.md": ("Cooking Notes", [_SOUP]),
}


def test_title_arm_represents_a_document_by_its_first_prose_chunk(tmp_path: Path) -> None:
    store = _store(tmp_path, _MANUAL, filter_structural_chunks=True)

    assert _title_arm(store) == [("manual.pdf", 1)]


def test_title_arm_keeps_the_first_chunk_when_the_filter_is_off(tmp_path: Path) -> None:
    store = _store(tmp_path, _MANUAL, filter_structural_chunks=False)

    assert _title_arm(store) == [("manual.pdf", 0)]


def test_title_arm_keeps_the_first_chunk_of_a_document_with_no_prose_chunk(
    tmp_path: Path,
) -> None:
    store = _store(
        tmp_path,
        {"toc.pdf": ("Maintenance Manual", [_TOC]), "cooking.md": ("Cooking Notes", [_SOUP])},
        filter_structural_chunks=True,
    )

    assert _title_arm(store) == [("toc.pdf", 0)]


def test_search_drops_a_title_matched_table_of_contents(tmp_path: Path) -> None:
    store = _store(tmp_path, _MANUAL, filter_structural_chunks=True)
    config = store._config.model_copy(
        update={
            "intent_routing": False,
            "query_expansion_count": 0,
            "concept_graph": False,
            "hyde": False,
            "reranker_model": "",
            "expansion_skip_threshold": 0.0,
            "temporal_filtering": False,
            "max_distance": 0.0,
            "min_relevance_score": 0.0,
            "diversity_max_per_source": 100,
        }
    )
    searcher = Searcher(
        config, mock.MagicMock(), store, mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    )
    vec = _vector(1)
    searcher._embed_query = lambda text: vec  # type: ignore[method-assign]

    rows = searcher.search_report(_QUERY, top_k=5).rows

    manual = [row.chunk_index for row in rows if row.source.endswith("manual.pdf")]
    assert manual
    assert 0 not in manual


def _chunk(source: str, index: int, text: str) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        chunk_type="raw",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=text,
        chunk_index=index,
        vector=[0.1],
    )


def test_representative_is_the_lowest_index_chunk_whatever_the_row_order() -> None:
    rows = [
        _chunk("a.pdf", 3, _BODY_2),
        _chunk("a.pdf", 2, _BODY_1),
        _chunk("a.pdf", 0, _TOC),
        _chunk("b.pdf", 4, _BODY_2),
        _chunk("b.pdf", 1, _TOC),
        _chunk("b.pdf", 5, _BODY_1),
    ]

    skipped = core._title_representatives(rows, skip_structural=True)
    kept = core._title_representatives(rows, skip_structural=False)

    assert {source: row.chunk_index for source, row in skipped.items()} == {"a.pdf": 2, "b.pdf": 4}
    assert {source: row.chunk_index for source, row in kept.items()} == {"a.pdf": 0, "b.pdf": 1}
