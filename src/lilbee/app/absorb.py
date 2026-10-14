"""Fold registered sources into a parent source: re-key what they indexed, then the registry."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import Config, active_config
from lilbee.data.ingest.skip_marker import (
    SkipKind,
    SkipRecords,
    load_skip_records,
    update_skip_records,
)
from lilbee.data.store import Store
from lilbee.data.types import is_under, rekeyed_source
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    AbsorbPhase,
    HeldOut,
    absorb_pending,
    delete_journal,
    read_journal,
    write_journal,
)
from lilbee.runtime.cancellation import CancelSignal
from lilbee.runtime.lock import LOCK_TIMEOUT, library_held

if TYPE_CHECKING:
    from lilbee.runtime.lock import LibraryHold

log = logging.getLogger(__name__)

SYNC_RUNNING_ADD_AGAIN = "A sync or a wiki build is running. Add {names} again when it ends."
UNFINISHED_ADD_HELD = (
    "An interrupted add is not finished and another lilbee process holds this library. "
    "Try again when it ends."
)
_JOURNAL_ID_BYTES = 4


def new_journal(
    data_root: Path, moves: dict[str, str], add: dict[str, str], drop: list[str]
) -> AbsorbJournal:
    """A journal for *moves* and the registry change, under an id no source key starts with.

    The caller holds the records lock of *data_root*: the journal takes the
    records of the sources that move as they are now.
    """
    records = _records_after(load_skip_records(data_root), moves)
    return AbsorbJournal(
        id=secrets.token_hex(_JOURNAL_ID_BYTES), moves=moves, add=add, drop=drop, records=records
    )


def _inner_olds(moves: dict[str, str], outer_old: str) -> list[str]:
    """The key, below the absorbed source *outer_old*, of each absorbed source inside it."""
    outer_new = moves[outer_old]
    return [
        outer_old + inner_new[len(outer_new) :]
        for inner_new in moves.values()
        if inner_new != outer_new and is_under(inner_new, outer_new)
    ]


def _records_after(records: SkipRecords, moves: dict[str, str]) -> dict[str, HeldOut]:
    """The record of each file the sources in *moves* hold out, under the key it takes.

    A source inside another absorbed source decides for its own files, so the
    outer source's records of them are left out.
    """
    after: dict[str, HeldOut] = {}
    for old, new in moves.items():
        shadows = _inner_olds(moves, old)
        for name, marker in records.markers.items():
            key = rekeyed_source(name, old, new)
            if key is None or any(is_under(name, shadow) for shadow in shadows):
                continue
            removed = records.kinds[name] is SkipKind.REMOVED
            after[key] = HeldOut(marker, records.reasons.get(name), removed)
    return after


def _rekey(config: Config, store: Store, old: str, new: str) -> None:
    """Move *old* and every key below it to *new* in the store and the wiki."""
    from lilbee.wiki.rekey import rekey_wiki_pages  # heavy: the wiki package loads spaCy

    store.rekey_sources_under(old, new)
    rekey_wiki_pages(config.data_root / config.wiki_dir, old, new)


def _clear_targets(store: Store, targets: list[str]) -> None:
    """Remove what the index holds at the keys the lifted sources land on."""
    occupied = [
        source["filename"]
        for source in store.get_sources()
        if any(is_under(source["filename"], target) for target in targets)
    ]
    if occupied:
        store.remove_documents(occupied)


def _shadows(journal: AbsorbJournal) -> dict[str, str]:
    """Each temporary key of an absorbed source at which a source inside it lands, with its own.

    The outer source can hold entries for the files of the inner one, from a
    walk before the inner source was registered: a folder that moves keeps its
    entries. The inner source's entries are the ones that land.
    """
    inner_old = {new: old for old, new in journal.moves.items()}
    return {
        journal.lifted(outer_old) + inner_new[len(outer_new) :]: journal.lifted(
            inner_old[inner_new]
        )
        for outer_old, outer_new in journal.moves.items()
        for inner_new in journal.moves.values()
        if inner_new != outer_new and is_under(inner_new, outer_new)
    }


def _drop_shadows(store: Store, journal: AbsorbJournal) -> None:
    """Remove what an outer source indexed for the files of a source inside it.

    A citation of the outer copy cites the same file and lands with it, unless
    the inner source holds the same citation.
    """
    shadows = _shadows(journal)
    if not shadows:
        return
    _clear_targets(store, list(shadows))
    for shadow, inner in shadows.items():
        store.drop_citations_repeated_under(shadow, inner)


def _lift(config: Config, store: Store, journal: AbsorbJournal) -> None:
    """Move each absorbed source to its temporary key, then clear the keys it lands on."""
    for old in journal.moves:
        _rekey(config, store, old, journal.lifted(old))
    _clear_targets(store, list(journal.moves.values()))
    _drop_shadows(store, journal)


def _land(config: Config, store: Store, journal: AbsorbJournal) -> None:
    """Move each absorbed source from its temporary key to its key below the parent.

    No temporary key is a final key, so the moves run in any order.
    """
    for old, new in journal.moves.items():
        _rekey(config, store, journal.lifted(old), new)


def _write_records(config: Config, journal: AbsorbJournal) -> None:
    """Give the skip record files the journal's records, in place of any at an old or new key."""
    prefixes = [*journal.moves, *journal.moves.values()]

    def _replace(records: SkipRecords) -> None:
        for name in list(records.markers):
            if any(is_under(name, prefix) for prefix in prefixes):
                records.markers.pop(name)
                records.reasons.pop(name, None)
        for key, record in journal.records.items():
            records.markers[key] = record.hash
            if record.reason is not None:
                records.reasons[key] = record.reason
            records.kinds[key] = SkipKind.REMOVED if record.removed else SkipKind.FAILED

    update_skip_records(config.data_root, _replace)


def _refresh_wiki_index(config: Config, journal: AbsorbJournal) -> None:
    """Give the wiki's subject index the new keys of the absorbed sources.

    A subject can list a key of the lift: a wiki build in a process that was
    running when the add died aggregates the index from the rows as they are then.
    """
    from lilbee.wiki.stubs import refresh_sources_under  # heavy: the wiki package loads spaCy

    lifted = [journal.lifted(old) for old in journal.moves]
    refresh_sources_under([*journal.moves, *lifted], config)


def _write_registry(config: Config, journal: AbsorbJournal) -> None:
    """Register what the journal adds and un-register what it drops, in one write."""

    def _apply(persisted: dict[str, str] | None) -> tuple[dict[str, str], None]:
        roots = {
            label: target
            for label, target in (persisted or {}).items()
            if label not in journal.drop
        }
        roots.update(journal.add)
        config.linked_roots = roots
        return roots, None

    settings.mutate_value(config.data_root, "linked_roots", _apply)


def _sweep_wiki_temp_files(config: Config) -> None:
    """Delete the temp files an absorb that died left beside the wiki files it was writing."""
    from lilbee.wiki.shared import remove_dead_temp_files  # heavy: the wiki package loads spaCy

    remove_dead_temp_files(config.data_root / config.wiki_dir)


def _roll_forward(store: Store, journal: AbsorbJournal, hold: LibraryHold) -> None:
    """Finish the absorb *journal* records from its phase, on the library of *hold*.

    Each step is safe to run again.
    """
    config = hold.config
    _sweep_wiki_temp_files(config)
    if journal.phase is AbsorbPhase.LIFT:
        _lift(config, store, journal)
        journal = journal.at(AbsorbPhase.LAND)
        write_journal(config.data_root, journal)
    _land(config, store, journal)
    _write_records(config, journal)
    _refresh_wiki_index(config, journal)
    _write_registry(config, journal)
    delete_journal(config.data_root)


def absorb(store: Store, journal: AbsorbJournal, hold: LibraryHold) -> None:
    """Write the absorb *journal* and run what it records, on the library of *hold*."""
    write_journal(hold.config.data_root, journal)
    _roll_forward(store, journal, hold)


def finish_absorb(store: Store, hold: LibraryHold) -> list[str]:
    """Roll the journal of the library of *hold* forward.

    Returns the keys the absorbed sources took, empty when no journal was left.
    """
    journal = read_journal(hold.config.data_root)
    if journal is None:  # another process finished it while this one waited
        return []
    log.warning("Finishing an interrupted add: %s", ", ".join(journal.add))
    _roll_forward(store, journal, hold)
    return list(journal.moves.values())


def keys_of_pending_absorb() -> list[str]:
    """The keys the sources of an absorb in progress on the active data root take; else empty.

    Read without a lock: an add that arrives during an absorb, and then waits
    for it, keeps what the absorb takes in.
    """
    config = active_config()
    if not absorb_pending(config.data_root):
        return []
    journal = read_journal(config.data_root)
    return [] if journal is None else list(journal.moves.values())


def _finish(config: Config, store: Store, wait: float, cancel: CancelSignal | None) -> list[str]:
    """Finish the journal of *config*'s data root; a sync that outlasts *wait* refuses it."""
    with library_held(config, UNFINISHED_ADD_HELD, wait, cancel) as hold:
        return finish_absorb(store, hold)


def finish_pending_absorb(cancel: CancelSignal | None = None) -> list[str]:
    """Finish an absorb that started on the active data root and did not end.

    Costs one existence check without one. Waits for a running sync, with no
    records lock held. Returns the keys the absorbed sources took, empty when it
    finished none. Setting *cancel* ends the wait for a running sync and the wait
    for the records lock with ``SkipRecordsWaitCancelledError``, with nothing moved.
    """
    config = active_config()
    if not absorb_pending(config.data_root):
        return []
    return _finish(config, get_services(cancel).store, LOCK_TIMEOUT, cancel)


def finish_pending_absorb_at_start(
    config: Config, store: Store, cancel: CancelSignal | None = None
) -> None:
    """Finish an interrupted absorb when a process builds its services.

    A sync or an add finishes what this leaves. Raises only when *cancel* ends
    the take of a lock: ``SkipRecordsWaitCancelledError``.
    """
    try:
        _finish(config, store, 0.0, cancel)
    except (RuntimeError, OSError, ValueError) as exc:
        log.warning("An interrupted add is not finished yet: %s", exc)
