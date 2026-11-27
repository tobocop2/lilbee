"""A hybrid query reads the index registry once, and the next query reads it again."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from lilbee.core.config import CHUNKS_TABLE, cfg
from lilbee.data.store import Store, core
from lilbee.data.store.lance_helpers import _indices_dir
from lilbee.data.store.types import ChunkType

_QUERY = np.full(cfg.embedding_dim, 0.1, dtype=np.float32)


def _record(source: str, index: int, title: str) -> dict[str, Any]:
    return {
        "source": source,
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": f"plain body words {source} {index}",
        "chunk_index": index,
        "title": title,
        "vector": [0.1 + index / 100] * cfg.embedding_dim,
    }


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    built = Store(
        cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb", "title_search": True})
    )
    built.add_chunks(
        [_record("zebra.md", i, "zebra manifesto") for i in range(3)]
        + [_record("lion.md", 0, "lion notes")]
    )
    built.ensure_fts_index()
    built.search(_QUERY, top_k=3, query_text="zebra")
    return built


@contextmanager
def _registry_reads(store: Store, *, unusable: bool = False) -> Iterator[list[Any]]:
    """The value of each registry read a search makes; index upkeep is switched off."""
    seen: list[Any] = []
    real = core._index_registry

    def counted(table: Any, lancedb_dir: Path) -> Any:
        value = None if unusable else real(table, lancedb_dir)
        seen.append(value)
        return value

    with (
        mock.patch.object(store, "ensure_scalar_indexes"),
        mock.patch.object(core, "_index_registry", counted),
    ):
        yield seen


@pytest.mark.parametrize("scope", [None, ChunkType.RAW])
def test_hybrid_query_reads_the_index_registry_once(store: Store, scope: ChunkType | None) -> None:
    with _registry_reads(store) as reads:
        rows = store.search(_QUERY, top_k=3, query_text="zebra", chunk_type=scope)

    assert len(reads) == 1
    assert "zebra.md" in {row.source for row in rows}


def test_both_arms_see_the_one_registry_read(store: Store) -> None:
    with (
        _registry_reads(store) as reads,
        mock.patch.object(store, "_title_arm", wraps=store._title_arm) as title_arm,
        mock.patch.object(store, "_vector_arm", wraps=store._vector_arm) as vector_arm,
    ):
        store.search(_QUERY, top_k=3, query_text="zebra")

    assert title_arm.call_args.args[-1] is reads[0]
    assert vector_arm.call_args.args[-1] is reads[0]


def test_an_unusable_registry_is_read_once_and_the_query_is_served(store: Store) -> None:
    with (
        _registry_reads(store, unusable=True) as reads,
        mock.patch.object(core, "_ann_nprobes", wraps=core._ann_nprobes) as nprobes,
    ):
        rows = store.search(_QUERY, top_k=3, query_text="zebra")

    assert reads == [None]
    assert rows
    nprobes.assert_called_once()


def test_a_vector_only_query_reads_the_index_registry_once(store: Store) -> None:
    with _registry_reads(store) as reads:
        store.search(_QUERY, top_k=3)

    assert len(reads) == 1


def test_index_loss_between_queries_is_noticed(store: Store) -> None:
    table = store.open_table(CHUNKS_TABLE)
    with _registry_reads(store) as reads:
        store.search(_QUERY, top_k=3, query_text="zebra")
        shutil.rmtree(_indices_dir(table, store._config.lancedb_dir))
        rows = store.search(_QUERY, top_k=3, query_text="zebra")

    assert len(reads) == 2
    assert reads[0] is not None
    assert reads[1] is None
    assert rows
