"""Read and write the held-out records of the store a test runs against."""

from __future__ import annotations

from unittest import mock

from lilbee.app.services import get_services, set_services
from lilbee.core.config import cfg
from lilbee.data.store import REMOVED_REASON, HeldOut, HeldRecord, Store
from lilbee.data.types import ShardId, SyncResult
from lilbee.providers.base import ProviderError

ENGINE_ERROR = "the embed engine did not start in 2 minutes"
FAILED_NAMES = ["bees.txt", "ants.txt", "wasps.txt"]
# What a search answers after ``fail_every_ingest(FAILED_NAMES)``.
FAILED_ANSWER = (
    "Nothing is indexed: 3 documents failed to ingest. "
    f"The error was: {ENGINE_ERROR}. Run `lilbee sync` to try again."
)


async def fail_every_ingest(
    names: list[str], error: Exception | None = None, shard: ShardId | None = None
) -> SyncResult:
    """Sync *names* on a real store, as one *shard* when given, with every file failing.

    The failure is an embed engine error unless *error* says otherwise.
    """
    from lilbee.data.ingest import sync
    from tests.conftest import make_mock_services

    set_services(make_mock_services(store=Store(cfg)))
    for name in names:
        (cfg.documents_dir / name).write_text(f"{name} holds plenty of text", encoding="utf-8")
    failure = error if error is not None else ProviderError(ENGINE_ERROR)
    with mock.patch("lilbee.data.ingest.pipeline.produce_records", side_effect=failure):
        return await sync(quiet=True, shard=shard)


def _store(store: Store | None) -> Store:
    return store if store is not None else get_services().store


def held_hashes(store: Store | None = None) -> dict[str, str]:
    """The hash each held-out path is held out at."""
    return {path: record.file_hash for path, record in _store(store).held_records().items()}


def held_kinds(store: Store | None = None) -> dict[str, HeldOut]:
    """Why each held-out path is held out."""
    return {path: record.kind for path, record in _store(store).held_records().items()}


def held_reasons(store: Store | None = None) -> dict[str, str]:
    """The reason shown for each held-out path."""
    return {path: record.reason for path, record in _store(store).held_records().items()}


def hold_failed(
    hashes: dict[str, str], reason: str = "no text extracted (0 chunks)", store: Store | None = None
) -> None:
    """Hold each path of *hashes* out as a failure at its hash."""
    _store(store).hold_failed(
        {path: HeldRecord(HeldOut.FAILED, digest, reason) for path, digest in hashes.items()}
    )


def hold_removed(hashes: dict[str, str], store: Store | None = None) -> None:
    """Hold each path of *hashes* out as a removal at its hash."""
    held = {
        path: HeldRecord(HeldOut.REMOVED, digest, REMOVED_REASON) for path, digest in hashes.items()
    }
    _store(store).remove_documents([], held)
