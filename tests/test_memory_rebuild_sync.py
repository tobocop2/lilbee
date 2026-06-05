"""Integration test: force_rebuild re-embeds preserved memories under the new model."""

from unittest.mock import MagicMock

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.data.ingest import sync
from lilbee.data.store import (
    LOCAL_OWNER,
    MemoryKind,
    MemoryRow,
    MemorySource,
    Store,
    local_owner_predicate,
)
from tests.conftest import make_mock_services


async def test_force_rebuild_preserves_and_reembeds_memory(tmp_path):
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.documents_dir = tmp_path / "docs"
    store = Store(cfg)
    new_vector = [0.5] * cfg.embedding_dim
    embedder = MagicMock()
    embedder.embedding_available.return_value = True
    embedder.embed_batch.return_value = [new_vector]
    embedder.truncated_total = 0
    services = make_mock_services(
        store=store,
        embedder=embedder,
        worker_pool=MagicMock(),
        pool_runtime=MagicMock(),
        pool_health_ticker=MagicMock(),
    )
    set_services(services)
    try:
        store.add_memory(
            MemoryRow(
                id="m1",
                owner=LOCAL_OWNER,
                shared=False,
                kind=MemoryKind.FACT,
                source=MemorySource.MANUAL,
                text="kept",
                vector=[0.1] * cfg.embedding_dim,
                created_at="t",
                updated_at="t",
            )
        )
        await sync(force_rebuild=True, quiet=True)
        memories = store.get_memories(owner_predicate=local_owner_predicate())
        assert [m.text for m in memories] == ["kept"]
        assert memories[0].vector == new_vector
        embedder.embed_batch.assert_called_once_with(["kept"])
    finally:
        set_services(None)
