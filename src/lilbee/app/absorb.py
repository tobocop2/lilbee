"""Fold registered sources into a parent source: re-key what they indexed, then the registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lilbee.core import settings
from lilbee.core.config import Config
from lilbee.data.ingest.skip_marker import SkipRecords, update_skip_records
from lilbee.data.store import Store
from lilbee.data.types import is_under, rekeyed_source

if TYPE_CHECKING:
    from lilbee.runtime.lock import LibraryHold

SYNC_RUNNING_ADD_AGAIN = "A sync or a wiki build is running. Add {names} again when it ends."


@dataclass(frozen=True)
class AbsorbPlan:
    """One absorb: the keys that move and the registry change that follows them."""

    moves: dict[str, str]
    """Each absorbed source's key, and the key it takes below its parent."""
    add: dict[str, str]
    """Registry entries the absorb writes, label to path."""
    drop: list[str]
    """Registry labels the absorb removes."""


def _shadows(moves: dict[str, str]) -> dict[str, str]:
    """Each key below an absorbed source at which a source inside it lands, with that source's key.

    The outer source can hold entries for the files of the inner one, from a
    walk before the inner source was registered. The inner source's entries are
    the ones that move.
    """
    return {
        outer_old + inner_new[len(outer_new) :]: inner_old
        for outer_old, outer_new in moves.items()
        for inner_old, inner_new in moves.items()
        if inner_new != outer_new and is_under(inner_new, outer_new)
    }


def _copy_records(config: Config, moves: dict[str, str]) -> None:
    """Give each skip record of a moving source a copy at the key it takes.

    A source inside another absorbed source decides for its own files, so the
    outer source's records of them are not copied.
    """
    shadows = _shadows(moves)

    def _copy(records: SkipRecords) -> None:
        for old, new in moves.items():
            for name in list(records.markers):
                key = rekeyed_source(name, old, new)
                if key is None or any(is_under(name, shadow) for shadow in shadows):
                    continue
                records.markers[key] = records.markers[name]
                records.kinds[key] = records.kinds[name]
                if name in records.reasons:
                    records.reasons[key] = records.reasons[name]

    update_skip_records(config.data_root, _copy)


def _drop_shadows(store: Store, moves: dict[str, str]) -> None:
    """Remove what an outer source indexed for the files of a source inside it.

    A citation of the outer copy cites the same file and moves to the inner
    source, unless the inner source holds the same citation.
    """
    for shadow, inner_old in _shadows(moves).items():
        occupied = [
            row["filename"] for row in store.get_sources() if is_under(row["filename"], shadow)
        ]
        if occupied:
            store.remove_documents(occupied)
        store.drop_citations_repeated_under(shadow, inner_old)
        store.rekey_sources_under(shadow, inner_old)


def _prune_records(config: Config, moves: dict[str, str]) -> None:
    """Delete the skip records at the keys the moved sources left."""

    def _prune(records: SkipRecords) -> None:
        for name in list(records.markers):
            if any(is_under(name, old) for old in moves):
                records.markers.pop(name)

    update_skip_records(config.data_root, _prune)


def _write_registry(config: Config, plan: AbsorbPlan) -> None:
    """Register what the plan adds and un-register what it drops, in one write."""

    def _apply(persisted: dict[str, str] | None) -> tuple[dict[str, str], None]:
        roots = {
            label: target for label, target in (persisted or {}).items() if label not in plan.drop
        }
        roots.update(plan.add)
        config.linked_roots = roots
        return roots, None

    settings.mutate_value(config.data_root, "linked_roots", _apply)


def absorb(store: Store, plan: AbsorbPlan, hold: LibraryHold) -> None:
    """Move the sources of *plan* below their parent, then write the registry, on *hold*'s library.

    Each step is safe to run again and the registry write is the last, so an add
    that died is finished by running it again.
    """
    # heavy: the wiki package loads spaCy
    from lilbee.wiki.rerender import rerender_pages
    from lilbee.wiki.stubs import drop_removed_sources

    config = hold.config
    _copy_records(config, plan.moves)
    _drop_shadows(store, plan.moves)
    for old, new in plan.moves.items():
        store.clear_rekey_target(old, new)
    # Outer source first: after a kill, the rerun skips its clear and spares the moved inner rows.
    for old, new in sorted(plan.moves.items(), key=lambda move: move[1]):
        store.rekey_sources_under(old, new)
        rerender_pages(store, config, store.wiki_pages_citing([new]))
    # A source that landed on a row outside its old path left the store.
    drop_removed_sources(store, config)
    _prune_records(config, plan.moves)
    _write_registry(config, plan)
