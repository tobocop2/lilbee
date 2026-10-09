"""A library on a real store that a test syncs with fan-out workers or in one process.

``ingest_processes`` is what ``lilbee sync --processes N`` sets, and the fan-out
threshold is lowered to one file so a small corpus reaches the workers through
the real gate. The embedder is the only stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import lancedb
import numpy as np

from lilbee.app.services import Services, set_services
from lilbee.core.config import cfg
from lilbee.core.config.model import Config
from lilbee.data.store import Store
from lilbee.data.types import SyncResult
from tests.conftest import make_mock_services

# Columns that differ between two correct runs of one history.
_VOLATILE = frozenset({"vector", "ingested_at", "mtime_ns", "stat_captured_ns", "updated_at"})


def embedder(dim: int) -> MagicMock:
    """An embedder that returns one fixed vector for every text."""
    stub = MagicMock()
    stub.embed.side_effect = lambda _text, **_kw: np.full(dim, 0.1, dtype=np.float32)
    stub.embed_batch.side_effect = lambda texts, **_kw: [
        np.full(dim, 0.1, dtype=np.float32) for _ in texts
    ]
    stub.truncated_total = 0
    return stub


def services_for(config: Config) -> Services:
    """Services over a real store at *config*'s index, with the stand-in embedder."""
    return make_mock_services(
        store=Store(config), embedder=embedder(config.embedding_dim), searcher=MagicMock()
    )


def dump_index(lancedb_dir: Path) -> dict[str, list[str]]:
    """Every row of every table of the store, without the volatile columns, sorted."""
    tables: dict[str, list[str]] = {}
    if not lancedb_dir.exists():
        return tables
    database = lancedb.connect(str(lancedb_dir))
    for name in sorted(database.table_names()):
        rows = database.open_table(name).to_arrow().to_pylist()
        kept = [{k: v for k, v in row.items() if k not in _VOLATILE} for row in rows]
        tables[name] = sorted(json.dumps(row, sort_keys=True, default=str) for row in kept)
    return tables


class Library:
    """One data root, synced by *processes* ingest processes."""

    def __init__(self, root: Path, processes: int) -> None:
        self.root = root
        self.processes = processes
        self.documents = root / "documents"
        self.documents.mkdir(parents=True)
        self.use()

    def use(self) -> None:
        """Point the process config and services at this library."""
        cfg.data_root = self.root
        cfg.documents_dir = self.documents
        cfg.data_dir = self.root / "data"
        cfg.lancedb_dir = self.root / "data" / "lancedb"
        cfg.linked_roots = {}
        cfg.concept_graph = False
        cfg.ingest_processes = self.processes
        set_services(services_for(cfg))

    def write(self, name: str, text: str) -> Path:
        path = self.documents / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_notes(self, prefix: str, count: int) -> list[str]:
        names = [f"{prefix}{index}.txt" for index in range(count)]
        for index, name in enumerate(names):
            self.write(name, f"{prefix} {index} records the result of experiment {index}.")
        return names

    async def sync(self, **kwargs: object) -> SyncResult:
        from lilbee.data.ingest import sync

        self.use()
        return await sync(quiet=True, **kwargs)  # type: ignore[arg-type]

    def remove(self, *names: str) -> None:
        from lilbee.app.ingest import remove_documents_durably

        self.use()
        remove_documents_durably(list(names))

    def sources(self) -> list[str]:
        """The filename of every source row, duplicates kept."""
        return sorted(row["filename"] for row in Store(cfg).get_sources())

    def chunk_sources(self) -> list[str]:
        """The source of every chunk row, duplicates kept."""
        table = Store(cfg).open_table("chunks")
        if table is None:
            return []
        return sorted(table.to_arrow().column("source").to_pylist())

    def dump(self) -> dict[str, list[str]]:
        return dump_index(self.root / "data" / "lancedb")

    def private_stores(self) -> list[Path]:
        """Every store a worker keeps for itself under this root."""
        return sorted((self.root / "shards").glob("w*/data"))
