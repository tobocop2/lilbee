"""Fold registered sources into a parent source: re-key what they indexed, then the registry."""

from __future__ import annotations

import logging
import secrets

from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import Config, active_config
from lilbee.data.ingest.skip_marker import rekey_skip_records, skip_records_lock
from lilbee.data.store import Store
from lilbee.data.types import is_under
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    AbsorbPhase,
    absorb_pending,
    delete_journal,
    read_journal,
    write_journal,
)
from lilbee.runtime.lock import LOCK_TIMEOUT, syncs_held_off

log = logging.getLogger(__name__)

SYNC_RUNNING_ADD_AGAIN = "A sync is running. Add {names} again when it ends."
_UNFINISHED_ADD_HELD = (
    "An interrupted add is not finished and another lilbee process holds this library. "
    "Try again when it ends."
)
_JOURNAL_ID_BYTES = 4


def new_journal(moves: dict[str, str], add: dict[str, str], drop: list[str]) -> AbsorbJournal:
    """A journal for *moves* and the registry change, under an id no source key starts with."""
    return AbsorbJournal(id=secrets.token_hex(_JOURNAL_ID_BYTES), moves=moves, add=add, drop=drop)


def _rekey(config: Config, store: Store, old: str, new: str) -> None:
    """Move *old* and every key below it to *new* in the store, the skip records and the wiki."""
    from lilbee.wiki.rekey import rekey_wiki_pages  # heavy: the wiki package loads spaCy

    store.rekey_sources_under(old, new)
    rekey_skip_records(config.data_root, old, new)
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


def _lift(config: Config, store: Store, journal: AbsorbJournal) -> None:
    """Move each absorbed source to its temporary key, then clear the keys it lands on."""
    for old in journal.moves:
        _rekey(config, store, old, journal.lifted(old))
    _clear_targets(store, list(journal.moves.values()))


def _land(config: Config, store: Store, journal: AbsorbJournal) -> None:
    """Move each absorbed source from its temporary key to its key below the parent."""
    for old, new in journal.moves.items():
        if not is_under(new, old):
            # An older build that synced since the lift moved these keys back.
            _rekey(config, store, old, journal.lifted(old))
        _rekey(config, store, journal.lifted(old), new)


def _refresh_wiki_index(config: Config, journal: AbsorbJournal) -> None:
    """Give the wiki's subject index the new keys of the absorbed sources."""
    from lilbee.wiki.stubs import refresh_sources_under  # heavy: the wiki package loads spaCy

    refresh_sources_under(journal.moves, config)


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


def _roll_forward(config: Config, store: Store, journal: AbsorbJournal) -> None:
    """Finish the absorb *journal* records from its phase; each step is safe to run again."""
    if journal.phase is AbsorbPhase.LIFT:
        _lift(config, store, journal)
        journal = journal.at(AbsorbPhase.LAND)
        write_journal(config.data_root, journal)
    _land(config, store, journal)
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


def _finish(config: Config, store: Store, running: str, wait: float) -> None:
    """Roll the journal of *config*'s data root forward under the locks an absorb holds."""
    with syncs_held_off(config.data_root, running, wait), skip_records_lock(config.data_root):
        journal = read_journal(config.data_root)
        if journal is not None:  # another process finished it while this one waited
            log.warning("Finishing an interrupted add: %s", ", ".join(journal.add))
            _roll_forward(config, store, journal)


def finish_pending_absorb(names: list[str] | None = None) -> None:
    """Finish an absorb that started on the active data root and did not end.

    Costs one existence check without one. With *names*, a running sync raises
    ``SyncRunningError`` at once and asks for them again; without, it waits.
    """
    config = active_config()
    if not absorb_pending(config.data_root):
        return
    if names is None:
        _finish(config, get_services().store, _UNFINISHED_ADD_HELD, LOCK_TIMEOUT)
    else:
        running = SYNC_RUNNING_ADD_AGAIN.format(names=", ".join(names))
        _finish(config, get_services().store, running, 0.0)


def finish_pending_absorb_at_start(config: Config, store: Store) -> None:
    """Finish an interrupted absorb when a process builds its services; never raises.

    A sync or an add finishes what this leaves.
    """
    try:
        _finish(config, store, _UNFINISHED_ADD_HELD, 0.0)
    except (RuntimeError, OSError, ValueError) as exc:
        log.warning("An interrupted add is not finished yet: %s", exc)
