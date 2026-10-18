"""Read and write the held-out records of the store a test runs against."""

from __future__ import annotations

from lilbee.app.services import get_services
from lilbee.data.store import REMOVED_REASON, HeldOut, HeldRecord, Store


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
