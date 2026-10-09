"""Fold registered sources into a parent source: re-key what they indexed, then the registry."""

from __future__ import annotations

import logging
import secrets

from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import Config, active_config
from lilbee.data.ingest.skip_marker import (
    SkipKind,
    SkipRecords,
    load_skip_records,
    skip_records_lock,
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
from lilbee.runtime.lock import LOCK_TIMEOUT, syncs_held_off

log = logging.getLogger(__name__)

SYNC_RUNNING_ADD_AGAIN = "A sync or a wiki build is running. Add {names} again when it ends."
_UNFINISHED_ADD_HELD = (
    "An interrupted add is not finished and another lilbee process holds this library. "
    "Try again when it ends."
)
_JOURNAL_ID_BYTES = 4


def new_journal(moves: dict[str, str], add: dict[str, str], drop: list[str]) -> AbsorbJournal:
    """A journal for *moves* and the registry change, under an id no source key starts with.

    The caller holds the skip-records lock: the journal takes the records of the
    sources that move as they are now.
    """
    records = _records_after(load_skip_records(active_config().data_root), moves)
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

    Two registered sources that lie inside each other indexed the files of the
    inner one twice. The inner source's entries are the ones that land.
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

    No temporary key is a final key, so the moves run in any order. A file two
    sources indexed must be single before the first of them lands.
    """
    for old, new in journal.moves.items():
        if not is_under(new, old):
            # A lilbee without the absorb that synced since the lift wrote below the old key.
            _rekey(config, store, old, journal.lifted(old))
    _drop_shadows(store, journal)
    for old, new in journal.moves.items():
        _rekey(config, store, journal.lifted(old), new)


def _written_since(journal: AbsorbJournal, records: SkipRecords) -> dict[str, HeldOut]:
    """The record *records* holds of each file below an absorbed source, by the key it takes.

    A lilbee without the absorb that ran since the journal was written holds a
    file out at the key it saw: the old one, the temporary one or the final one.
    A source that keeps its label below its parent is left out, because its old
    keys cannot be told from its final ones.
    """
    moves = {old: new for old, new in journal.moves.items() if not is_under(new, old)}
    return {
        **_records_after(records, {new: new for new in moves.values()}),
        **_records_after(records, {journal.lifted(old): new for old, new in moves.items()}),
        **_records_after(records, moves),
    }


def _held_out(config: Config, journal: AbsorbJournal) -> dict[str, HeldOut]:
    """What the absorbed sources hold out: the journal's records, then any written since."""
    return {**journal.records, **_written_since(journal, load_skip_records(config.data_root))}


def _drop_removed_again(store: Store, journal: AbsorbJournal, held: dict[str, HeldOut]) -> None:
    """Remove each file *held* names as removed that is indexed at the removed hash.

    A lilbee without the absorb that synced before the journal ended can index
    it again. Another hash is an edit since the removal, which every sync indexes.
    A removal written since the journal can leave rows in the tables that had
    already moved, so its file goes unless the index holds it at another hash.
    """
    removed = {key: record.hash for key, record in held.items() if record.removed}
    if not removed:
        return
    indexed = {source["filename"]: source["file_hash"] for source in store.get_sources()}
    again = [
        key
        for key, removed_hash in removed.items()
        if indexed.get(key) == removed_hash or (key not in indexed and key not in journal.records)
    ]
    if again:
        store.remove_rows_of(again)


def _write_records(config: Config, journal: AbsorbJournal, held: dict[str, HeldOut]) -> None:
    """Give the skip record files *held*, in place of any at an old, temporary or new key."""
    lifted = [journal.lifted(old) for old in journal.moves]
    prefixes = [*journal.moves, *lifted, *journal.moves.values()]

    def _replace(records: SkipRecords) -> None:
        for name in list(records.markers):
            if any(is_under(name, prefix) for prefix in prefixes):
                records.markers.pop(name)
                records.reasons.pop(name, None)
        for key, record in held.items():
            records.markers[key] = record.hash
            if record.reason is not None:
                records.reasons[key] = record.reason
            records.kinds[key] = SkipKind.REMOVED if record.removed else SkipKind.FAILED

    update_skip_records(config.data_root, _replace)


def _refresh_wiki_index(config: Config, journal: AbsorbJournal) -> None:
    """Give the wiki's subject index the new keys of the absorbed sources.

    A subject can list a key of the lift: an older lilbee that removes a file
    between the phases aggregates the index from the rows as they are then.
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


def _roll_forward(config: Config, store: Store, journal: AbsorbJournal) -> None:
    """Finish the absorb *journal* records from its phase; each step is safe to run again."""
    _sweep_wiki_temp_files(config)
    if journal.phase is AbsorbPhase.LIFT:
        _lift(config, store, journal)
        journal = journal.at(AbsorbPhase.LAND)
        write_journal(config.data_root, journal)
    _land(config, store, journal)
    held = _held_out(config, journal)
    _drop_removed_again(store, journal, held)
    _write_records(config, journal, held)
    _refresh_wiki_index(config, journal)
    _write_registry(config, journal)
    delete_journal(config.data_root)


def absorb(journal: AbsorbJournal, names: list[str]) -> None:
    """Run the absorb *journal* records; a running sync raises ``SyncRunningError`` first.

    The caller holds the skip-records lock. *names* are the paths the user is adding.
    """
    config = active_config()
    running = SYNC_RUNNING_ADD_AGAIN.format(names=", ".join(names))
    with syncs_held_off(config.data_root, running):
        write_journal(config.data_root, journal)
        _roll_forward(config, get_services().store, journal)


def _finish(config: Config, store: Store, running: str, wait: float) -> list[str]:
    """Roll the journal of *config*'s data root forward under the locks an absorb holds.

    Returns the keys the absorbed sources took, empty when no journal was left.
    """
    with syncs_held_off(config.data_root, running, wait), skip_records_lock(config.data_root):
        journal = read_journal(config.data_root)
        if journal is None:  # another process finished it while this one waited
            return []
        log.warning("Finishing an interrupted add: %s", ", ".join(journal.add))
        _roll_forward(config, store, journal)
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


def finish_pending_absorb(names: list[str] | None = None) -> list[str]:
    """Finish an absorb that started on the active data root and did not end.

    Costs one existence check without one. With *names*, a running sync raises
    ``SyncRunningError`` at once and asks for them again; without, it waits.
    Returns the keys the absorbed sources took, empty when it finished none.
    """
    config = active_config()
    if not absorb_pending(config.data_root):
        return []
    if names is None:
        return _finish(config, get_services().store, _UNFINISHED_ADD_HELD, LOCK_TIMEOUT)
    running = SYNC_RUNNING_ADD_AGAIN.format(names=", ".join(names))
    return _finish(config, get_services().store, running, 0.0)


def finish_pending_absorb_at_start(config: Config, store: Store) -> None:
    """Finish an interrupted absorb when a process builds its services; never raises.

    A sync or an add finishes what this leaves.
    """
    try:
        _finish(config, store, _UNFINISHED_ADD_HELD, 0.0)
    except (RuntimeError, OSError, ValueError) as exc:
        log.warning("An interrupted add is not finished yet: %s", exc)
