"""Mock services under which the real sync indexes text files into an in-memory source table."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from lilbee.app.services import Services
from tests.conftest import make_mock_services


def ingesting_services() -> tuple[Services, dict[str, dict[str, Any]]]:
    """Services whose store records each indexed file; returns them with the source table."""
    sources: dict[str, dict[str, Any]] = {}
    store = MagicMock()
    store.get_sources.side_effect = lambda: list(sources.values())

    def _upsert(filename: str, file_hash: str, chunk_count: int, source_type: str = "document"):
        sources[filename] = {
            "filename": filename,
            "file_hash": file_hash,
            "chunk_count": chunk_count,
            "ingested_at": datetime.now(UTC).isoformat(),
            "source_type": source_type,
        }

    def _write_batch(items: list[Any]) -> int:
        for item in items:
            _upsert(item.source, item.file_hash, len(item.records))
        return sum(len(item.records) for item in items)

    store.upsert_source.side_effect = _upsert
    store.write_chunks_batch.side_effect = _write_batch
    store.index_mismatch.return_value = None
    store.get_meta.return_value = None
    store.search.return_value = []
    store.bm25_probe.return_value = []
    embedder = MagicMock()
    embedder.embed.side_effect = lambda _text, **_kw: np.full(768, 0.1, dtype=np.float32)
    embedder.embed_batch.side_effect = lambda texts, **_kw: [[0.1] * 768 for _ in texts]
    embedder.truncated_total = 0
    services = make_mock_services(store=store, embedder=embedder, searcher=MagicMock())
    return services, sources
