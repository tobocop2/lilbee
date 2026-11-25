"""The index records the chunk settings its chunks were cut with, and a rebuild verdict reads it.

A first sync chunks its file at the old size and has not yet written. The user then
changes chunk_size. The settings boundary sees no chunks and reports no rebuild. The
sync's write then lands with chunks cut at the old size. The index records the settings
the sync used, so every later check reports the drift.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from lilbee.app import settings as app_settings
from lilbee.app.memory import make_memory_row
from lilbee.app.services import get_services, set_services
from lilbee.app.settings import apply_settings_update, index_has_chunks, reindex_values
from lilbee.app.status import gather_status
from lilbee.cli.helpers import render_status_result
from lilbee.core.config import cfg
from lilbee.data.extract.chunk import chunk_text
from lilbee.data.ingest import pipeline, sync
from lilbee.data.store import Store
from lilbee.data.store.schema import _meta_schema
from lilbee.data.types import SyncResult
from lilbee.server.models import StatusResponse, SyncSummary
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
)
from tests.conftest import make_mock_services

TEXT = " ".join(f"word{i}" for i in range(3000))
OLD_SIZE = 512
NEW_SIZE = 128


@pytest.fixture(autouse=True)
def real_store():
    cfg.chunk_size = OLD_SIZE
    cfg.chunk_overlap = 0
    _use(Store(cfg))
    yield
    set_services(None)


def _use(store: Store, embedder: MagicMock | None = None) -> None:
    set_services(
        make_mock_services(
            store=store,
            embedder=embedder or MagicMock(),
            worker_pool=MagicMock(),
            pool_runtime=MagicMock(),
            pool_health_ticker=MagicMock(),
        )
    )


def _records(pieces: list[str], source: str = "doc.md") -> list[dict]:
    rows = []
    for index, piece in enumerate(pieces):
        vector = np.zeros(cfg.embedding_dim, dtype=np.float32)
        vector[index % cfg.embedding_dim] = 1.0
        rows.append(
            {
                "source": source,
                "content_type": "text",
                "chunk_type": "raw",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "chunk": piece,
                "chunk_index": index,
                "vector": vector,
            }
        )
    return rows


def _legacy_meta(store: Store) -> None:
    """Rewrite the meta row as a library built before the record existed holds it."""
    row = store._meta_row()
    assert row is not None
    row["chunk_settings"] = None
    row["schema_version"] = 4
    store.get_db().create_table("_meta", [row], schema=_meta_schema(), mode="overwrite")


def test_a_chunk_size_change_during_a_sync_is_still_reported_as_a_rebuild():
    assert not index_has_chunks()

    # The sync snapshots its settings and chunks its first file at the old size.
    snapshot = reindex_values()
    stale = chunk_text(TEXT)

    # The change lands now. The index holds no chunks, so no rebuild is reported.
    answer = apply_settings_update({"chunk_size": NEW_SIZE})
    assert cfg.chunk_size == NEW_SIZE
    assert answer.reindex_required is False

    # The sync's write lands: the index now holds chunks cut at the old size.
    get_services().store.add_chunks(_records(stale), chunk_settings=snapshot)
    assert index_has_chunks()
    assert len(stale) < len(chunk_text(TEXT))

    store = get_services().store
    stamp = store.chunk_stamp()
    assert stamp is not None, "the index records no chunk settings, so no check sees the write"
    assert stamp["chunk_size"] == OLD_SIZE
    assert app_settings.chunk_drift() == ["chunk_size"]


async def test_a_sync_records_the_settings_it_cut_with_and_reports_the_drift():
    (cfg.documents_dir).mkdir(parents=True, exist_ok=True)
    (cfg.documents_dir / "doc.md").write_text(TEXT, encoding="utf-8")
    store = get_services().store
    changed: list[bool] = []

    def embed(texts, **_kw):
        # The change lands after the sync chunked the file and before its write.
        if not changed:
            changed.append(True)
            apply_settings_update({"chunk_size": NEW_SIZE})
        return [[0.1] * cfg.embedding_dim for _ in texts]

    embedder = MagicMock()
    embedder.validate_model.return_value = True
    embedder.embedding_available.return_value = True
    embedder.embed_batch.side_effect = embed
    _use(store, embedder)

    result = await sync(quiet=True)

    assert result.added == ["doc.md"]
    stamp = getattr(store, "chunk_stamp", lambda: None)()
    assert stamp is not None, "the sync wrote chunks and recorded no chunk settings"
    assert stamp["chunk_size"] == OLD_SIZE
    assert result.chunk_drift == ["chunk_size"]
    status = gather_status()
    assert status.index is not None
    assert status.index.chunk_drift == ["chunk_size"]
    rendered = "".join(str(line) for line in render_status_result(status))
    assert "chunk_size changed since the chunks were cut" in rendered
    # REST carries the same field on the status and on the add-files sync summary.
    rest = StatusResponse(**status.model_dump(exclude_none=True))
    assert rest.index is not None
    assert rest.index.chunk_drift == ["chunk_size"]
    assert SyncSummary(**result.model_dump()).chunk_drift == ["chunk_size"]


def test_an_empty_library_records_nothing_and_reports_no_drift():
    store = get_services().store
    assert store.chunk_stamp() is None
    assert store.chunk_drift({"chunk_size": NEW_SIZE}) == []


def test_a_data_dir_with_no_library_reports_no_drift_and_creates_none(tmp_path):
    absent = tmp_path / "absent"
    store = Store(cfg.model_copy(update={"data_root": absent, "lancedb_dir": absent / "lancedb"}))
    assert store.chunk_drift({"chunk_size": NEW_SIZE}) == []
    assert not absent.exists()


def test_an_incremental_write_keeps_the_record_of_the_first():
    store = get_services().store
    store.add_chunks(_records(["a"]), chunk_settings=reindex_values())
    cfg.chunk_size = NEW_SIZE
    store.add_chunks(_records(["b"], source="other.md"), chunk_settings=reindex_values())
    assert store.chunk_stamp()["chunk_size"] == OLD_SIZE
    assert app_settings.chunk_drift() == ["chunk_size"]


def test_a_rebuild_replaces_the_record():
    store = get_services().store
    store.add_chunks(_records(["a"]), chunk_settings=reindex_values())
    cfg.chunk_size = NEW_SIZE
    store.restamp_meta(reindex_values())
    assert store.chunk_stamp()["chunk_size"] == NEW_SIZE
    assert app_settings.chunk_drift() == []


def test_a_meta_rewrite_without_settings_keeps_the_record():
    store = get_services().store
    store.add_chunks(_records(["a"]), chunk_settings=reindex_values())
    store.restamp_meta()
    assert store.chunk_stamp()["chunk_size"] == OLD_SIZE


def test_a_write_without_settings_leaves_the_library_unrecorded():
    store = get_services().store
    store.add_chunks(_records(["a"]))
    assert store.get_meta() is not None
    assert store.chunk_stamp() is None


def test_an_inert_setting_does_not_count_as_drift():
    store = get_services().store
    cfg.semantic_chunking = False
    store.add_chunks(_records(["a"]), chunk_settings=reindex_values())
    cfg.topic_threshold = 0.1 if cfg.topic_threshold != 0.1 else 0.2
    assert store.chunk_drift(reindex_values()) == ["topic_threshold"]
    assert app_settings.chunk_drift() == []


def test_a_library_built_before_the_record_pins_the_settings_in_force_once(caplog):
    store = get_services().store
    store.add_chunks(_records(["a"]))
    _legacy_meta(store)
    with caplog.at_level(logging.WARNING, logger="lilbee.data.store.core"):
        assert store.chunk_drift(reindex_values()) == []
        assert store.chunk_drift(reindex_values()) == []
    pins = [r for r in caplog.records if "records no chunk settings" in r.getMessage()]
    assert len(pins) == 1
    assert "lilbee rebuild" in pins[0].getMessage()
    assert store.chunk_stamp()["chunk_size"] == OLD_SIZE
    assert store.get_meta()["schema_version"] == 5


def test_a_change_to_a_library_built_before_the_record_shows_as_drift():
    store = get_services().store
    store.add_chunks(_records(["a"]))
    _legacy_meta(store)
    apply_settings_update({"chunk_size": NEW_SIZE})
    assert store.chunk_stamp()["chunk_size"] == OLD_SIZE
    assert app_settings.chunk_drift() == ["chunk_size"]


def test_a_sync_into_a_legacy_library_records_its_settings_and_warns_once(caplog):
    store = get_services().store
    store.add_chunks(_records(["a"]))
    _legacy_meta(store)
    with caplog.at_level(logging.WARNING, logger="lilbee.data.store.core"):
        store.add_chunks(_records(["b"], source="other.md"), chunk_settings=reindex_values())
        store.add_chunks(_records(["c"], source="third.md"), chunk_settings=reindex_values())
    pins = [r for r in caplog.records if "records no chunk settings" in r.getMessage()]
    assert len(pins) == 1
    assert "lilbee rebuild" in pins[0].getMessage()
    assert store.chunk_stamp()["chunk_size"] == OLD_SIZE


def test_a_first_sync_after_a_memory_records_its_settings_without_a_warning(caplog):
    store = get_services().store
    store.add_memory(make_memory_row("a note", lambda _text: [0.5] * cfg.embedding_dim))
    assert store.get_meta() is not None
    with caplog.at_level(logging.WARNING, logger="lilbee.data.store.core"):
        store.add_chunks(_records(["a"]), chunk_settings=reindex_values())
    assert not [r for r in caplog.records if "records no chunk settings" in r.getMessage()]
    assert store.chunk_stamp()["chunk_size"] == OLD_SIZE


async def test_a_sync_clears_its_settings_snapshot_when_it_ends(monkeypatch):
    async def fails(*_args, **_kwargs):
        assert pipeline._SYNC_CHUNK_SETTINGS.get() == reindex_values()
        raise RuntimeError("sync failed")

    monkeypatch.setattr(pipeline, "_sync", fails)
    with pytest.raises(RuntimeError, match="sync failed"):
        await sync(quiet=True)
    assert pipeline._SYNC_CHUNK_SETTINGS.get() is None


def test_a_library_with_a_meta_row_and_no_chunks_records_nothing_on_a_check():
    store = get_services().store
    store.add_chunks(_records(["a"]))
    assert store.clear_table("chunks", "true")
    assert store.chunk_drift(reindex_values()) == []
    assert store.chunk_stamp() is None


def _summary(result: SyncResult) -> list[str]:
    return ["".join(part for part, _style in line) for line in result._lines()]


def test_a_library_with_chunks_and_no_meta_row_is_left_to_its_identity_pin():
    store = get_services().store
    store.add_chunks(_records(["a"]))
    store.get_db().drop_table("_meta")
    assert store.chunk_drift(reindex_values()) == []
    assert store.get_meta() is None


def test_the_sync_summary_names_the_drift_and_the_remedy():
    text = _summary(SyncResult(chunk_drift=["chunk_size", "chunk_overlap"]))
    assert (
        "Rebuild advised: chunk_size, chunk_overlap changed since the chunks were cut; "
        "run `lilbee rebuild`"
    ) in text
    assert not any("Rebuild advised" in line for line in _summary(SyncResult()))
