"""Register external source roots, and remove indexed documents durably."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import Counter
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.app.absorb import (
    absorb,
    finish_pending_absorb,
    keys_of_pending_absorb,
    new_journal,
)
from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import active_config
from lilbee.core.config.enums import OcrMode
from lilbee.data.ingest.discovery import (
    excluded_extension_reasons,
    file_hash,
    resolve_source_path,
    walk_reaches,
)
from lilbee.data.ingest.ignore import IgnoreRules
from lilbee.data.ingest.skip_marker import (
    SkipRecords,
    held_out_names,
    mark_removed,
    skip_records_lock,
    update_skip_records,
)
from lilbee.data.store.types import RemoveResult
from lilbee.data.types import is_under
from lilbee.runtime.absorb_journal import AbsorbJournal
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError

_ADD_CANCELLED = "Add cancelled."
_ADD_CANCELLED_ONE = "Add cancelled. {name} was not added."
_ADD_CANCELLED_MANY = "Add cancelled. {names} were not added."
_ALSO_HIT_ERROR = " It also hit an error: {error}."
_CANCEL_ERRORS = (asyncio.CancelledError, TaskCancelledError, KeyboardInterrupt)


@dataclass
class RegisterResult:
    """Result of registering source roots into the knowledge base."""

    registered: list[str] = field(default_factory=list)  # labels newly registered
    name_taken: list[str] = field(default_factory=list)
    """Labels held by a different live source or an owned entry; ``--force`` overwrites."""
    overlapping: list[str] = field(default_factory=list)
    """Paths nesting under or over ``documents_dir`` or a live root; none is registered."""
    containing: list[str] = field(default_factory=list)
    """The overlapping paths that contain the documents directory; none is registered."""
    absorbed_into: dict[str, list[str]] = field(default_factory=dict)
    """Each newly registered label that took in registered sources, with their labels."""
    refused: list[str] = field(default_factory=list)
    """Files whose format lilbee does not index, as ``name: reason``."""
    outside_corpus: list[str] = field(default_factory=list)
    """One name per distinct path that is refused, name-taken or around the documents directory."""
    tracked: list[str] = field(default_factory=list)
    """Named sources the knowledge base already tracks, so nothing was registered.

    Either the path already lives under ``documents_dir`` or this exact source is
    already registered under that label. Nothing is wrong and ``--force`` would
    change nothing: the sync that follows covers them.
    """

    @property
    def reached_corpus(self) -> bool:
        """Whether a named path is in the corpus, so a sync has something to index for it.

        A refused or missing path is not, and neither is one whose label another
        source holds; a sync after one is a whole-vault pass whose summary would
        read as the outcome of the add.
        """
        return bool(self.registered or self.tracked or self.overlapping)

    @property
    def absorbed(self) -> list[str]:
        """The labels of the sources a newly registered parent took in."""
        return [label for labels in self.absorbed_into.values() for label in labels]

    @property
    def revocable(self) -> list[str]:
        """The registered labels a cancelled add un-registers: each that took in no source."""
        return [label for label in self.registered if label not in self.absorbed_into]

    @property
    def overlapping_inside(self) -> list[str]:
        """The overlapping paths a registered root already walks."""
        return list((Counter(self.overlapping) - Counter(self.containing)).elements())


def _resolve_label(
    base: str, roots: dict[str, str], docs_resolved: Path, *, force: bool
) -> str | None:
    """Choose the source-key label for a new root, or None when the name is taken.

    An owned ``documents_dir`` top-level entry of the same name always wins and is
    never shadowed, even under ``force`` -- a label that shadows it would make
    resolve_source_path disagree with how discovery keyed the owned file. Reuses
    the label when a root of that name was registered before but its path has since
    vanished (the source moved: re-register in place, no ``--force`` needed) or
    when ``force`` overwrites a live registered root of the same name.
    """
    if (docs_resolved / base).exists():
        return None  # an owned entry holds this name; never shadow it
    existing = roots.get(base)
    if existing is not None and not Path(existing).exists():
        return base  # dangling root; the source moved, re-point it to the new path
    if force:
        return base
    if base in roots:
        return None
    return base


def _live_roots(roots: dict[str, str]) -> list[Path]:
    """The resolved path of each registered root that exists; a vanished one overlaps nothing."""
    return [Path(target).resolve() for target in roots.values() if Path(target).exists()]


def _inside_a_root(src: Path, live_roots: list[Path]) -> bool:
    """Whether a live root already walks *src*; a second root there would index it twice."""
    return any(src.is_relative_to(root) for root in live_roots)


def _reached_below(src: Path, roots: dict[str, str]) -> dict[str, Path]:
    """Each registered root below *src* that the walk of *src* reaches, by label.

    A root whose folder is gone counts by its stored path.
    """
    resolved = {label: Path(target).resolve() for label, target in roots.items()}
    below = {
        label: path for label, path in resolved.items() if path != src and path.is_relative_to(src)
    }
    if not below:
        return below
    rules = IgnoreRules.for_corpus(active_config().data_root)
    return {label: path for label, path in below.items() if walk_reaches(src, path, rules)}


def _classify(
    paths: list[Path], roots: dict[str, str], docs_resolved: Path, *, force: bool
) -> tuple[RegisterResult, dict[str, str]]:
    """Sort *paths* by what registering each one does, leaving *roots* as the registry after it.

    Also returns the key of each source a new parent takes in, with the key it gets.
    """
    result = RegisterResult()
    moves: dict[str, str] = {}
    by_target = {target: label for label, target in roots.items()}
    refused = excluded_extension_reasons()
    outside: dict[Path, str] = {}
    for p in paths:
        src = p.resolve()
        reason = refused.get(src.suffix.lower()) if src.is_file() else None
        if reason is not None:
            result.refused.append(f"{p.name}: {reason}")
            outside.setdefault(src, p.name)
            continue
        if src == docs_resolved or docs_resolved in src.parents:
            result.tracked.append(p.name)  # already owned by the knowledge base
            continue
        already = by_target.get(str(src))
        if already is not None:
            result.tracked.append(already)  # this exact source is already registered
            continue
        live_roots = _live_roots(roots)
        if _inside_a_root(src, live_roots):
            result.overlapping.append(p.name)  # would walk the same files twice
            continue
        reached = _reached_below(src, roots)
        if docs_resolved.is_relative_to(src):
            result.overlapping.append(p.name)
            result.containing.append(p.name)
            outside.setdefault(src, p.name)
            continue
        others = {name: target for name, target in roots.items() if name not in reached}
        label = _resolve_label(src.name, others, docs_resolved, force=force)
        if label is None:
            result.name_taken.append(src.name)
            outside.setdefault(src, src.name)
            continue
        for child, child_path in reached.items():
            by_target.pop(roots.pop(child), None)
            moves[child] = f"{label}/{child_path.relative_to(src).as_posix()}"
        roots[label] = str(src)
        by_target[str(src)] = label
        result.registered = [name for name in result.registered if name not in reached] + [label]
        if reached:
            result.absorbed_into[label] = sorted(reached)
    result.outside_corpus = list(outside.values())
    return result, moves


def names_outside_corpus(paths: list[Path]) -> list[str]:
    """The name of each of *paths* the corpus does not hold, by the rule registration applies."""
    if not paths:
        return []  # a stopped sync has no paths, and reads no registry to say so
    config = active_config()
    roots = dict(settings.load(config.data_root).get("linked_roots") or {})
    plan, _moves = _classify(paths, roots, config.documents_dir.resolve(), force=False)
    return [*plan.outside_corpus, *plan.registered]


def source_label_taken(name: str, target: Path | None = None) -> bool:
    """Whether registering *target* under *name* would collide with a different source.

    The confirm-before-overwrite affordance in the TUI reads this. The label
    rule is delegated to :func:`_resolve_label` (the authority register_sources
    itself applies) rather than mirrored, and a *target* whose exact path is
    already registered under *name* is not a collision: re-adding the same
    source is idempotent, matching register_sources' by-target no-op.
    """
    config = active_config()
    roots = dict(config.linked_roots)
    if target is not None and roots.get(name) == str(target.resolve()):
        return False
    return _resolve_label(name, roots, config.documents_dir.resolve(), force=False) is None


def register_sources(paths: list[Path], *, force: bool = False) -> RegisterResult:
    """Register each path as a root lilbee indexes where it already lives.

    A prepared corpus is already on local disk, so ``add`` records where it is
    rather than copying or linking it: discovery walks the registered root and
    keys its files under the root's label (its basename). A path already inside
    ``documents_dir`` is left to the owned-files walk; a path already registered
    under the same target is a no-op; a label already taken by a different live
    root or an owned entry is skipped unless ``force``. The registry is persisted
    so later processes index the same roots.

    A directory that contains registered sources takes each one its walk reaches:
    the source leaves the registry and its indexed files, skip records and wiki
    citations move below the new label, with nothing extracted or embedded again.
    That needs every sync and wiki write stopped; a running one raises
    ``SyncRunningError`` and a removal in progress raises ``SkipRecordsLockError``,
    with nothing changed.
    """
    config = active_config()
    documents_dir = config.documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)
    docs_resolved = documents_dir.resolve()
    if not paths:
        return RegisterResult()

    def _mutate(
        persisted: dict[str, str] | None,
    ) -> tuple[dict[str, str] | None, tuple[RegisterResult, AbsorbJournal | None]]:
        # Read the registry from config.toml INSIDE the lock (not the possibly
        # stale in-memory copy) so two processes registering roots concurrently
        # cannot lose each other's entry.
        before = dict(persisted or {})
        roots = dict(before)
        result, moves = _classify(paths, roots, docs_resolved, force=force)
        if not moves:
            config.linked_roots = roots  # refresh the in-process view (picks up merges)
            return roots, (result, None)
        # The absorb writes the registry after it moved the keys.
        config.linked_roots = before
        add = {label: target for label, target in roots.items() if before.get(label) != target}
        drop = [label for label in before if label not in roots]
        return persisted, (result, new_journal(moves, add, drop))

    names = [p.name for p in paths]
    in_progress = keys_of_pending_absorb()
    # Taken before the registry changes, so a held lock refuses the add with nothing done.
    with skip_records_lock(config.data_root):
        taken = [*in_progress, *finish_pending_absorb(names)]
        result, journal = settings.mutate_value(config.data_root, "linked_roots", _mutate)
        if journal is not None:
            absorb(journal, list(result.absorbed_into))
            taken = [*taken, *journal.moves.values()]
        unmark_sources_under(paths, spared=taken)
    return result


def unmark_sources_under(paths: list[Path], *, spared: Iterable[str] = ()) -> None:
    """Drop the skip records (marker, reason and kind) of every source *paths* covers.

    A record at or below a key in *spared* stays: a parent that takes in a
    source keeps what the user removed from it and what failed in it.

    A marker exists to stop *discovery* from resurrecting a source the user
    removed, or from re-paying the extract cost on a file that yielded nothing.
    Naming the path outranks it: ``add`` is the user asking for that source
    back, so the marker goes and the sync that follows ingests the file again.
    Without this a removal would be permanent, undoable only by ``rebuild``,
    which the user has no reason to reach for after typing the path they want.

    Each root in *paths* must be registered when this runs: marker keys resolve
    to files through the live registry.
    """

    kept = list(spared)

    def _drop_covered(records: SkipRecords) -> None:
        for name in _markers_covering(records.markers, paths):
            if not any(is_under(name, key) for key in kept):
                records.markers.pop(name)

    update_skip_records(active_config().data_root, _drop_covered)


def _markers_covering(markers: dict[str, str], paths: list[Path]) -> set[str]:
    """Marker keys whose file is one of *paths* or lives beneath one.

    Each key is resolved back to the file it tracks -- the same mapping
    discovery keyed it by -- so an owned ``documents_dir`` entry, a file under a
    registered root, and a single-file root are all matched by the one rule
    instead of three shape-specific ones.
    """
    named = [p.resolve() for p in paths]
    covered = set()
    for name in markers:
        tracked = resolve_source_path(name).resolve(strict=False)
        if any(tracked == path or path in tracked.parents for path in named):
            covered.add(name)
    return covered


_GLOB_CHARS = frozenset("*?[")


def _is_glob(name: str) -> bool:
    """Whether *name* should be matched as a glob rather than a literal source."""
    return any(char in _GLOB_CHARS for char in name)


def folder_members(name: str, known: Iterable[str]) -> list[str]:
    """Known sources under folder *name*, matched on whole path segments.

    ``myrepo`` covers ``myrepo/a.py`` but never ``myrepo-2/x``. Empty when *name*
    is not a parent directory of any known source.
    """
    prefix = name.rstrip("/") + "/"
    return [source for source in known if source.startswith(prefix)]


def removable_names(indexed: list[str] | None = None) -> list[str]:
    """What a remove can name: *indexed* (the store's sources when not given), then failures."""
    if indexed is None:
        indexed = [s["filename"] for s in get_services().store.get_sources()]
    seen = set(indexed)
    failed = held_out_names(active_config().data_root)
    return indexed + [name for name in failed if name not in seen]


def expand_remove_targets(names: list[str], known: list[str] | None = None) -> list[str]:
    """Expand folder names and glob patterns to the known sources they cover.

    An exact source name is kept. A folder name (a parent directory of known
    sources) expands to every source beneath it. A glob (a name containing
    ``* ? [``) expands to every source it fnmatches. A name matching none of
    these is kept unchanged so the caller reports it not-found. Order and
    de-duplication are preserved. *known* is ``removable_names()`` when not
    supplied; a caller that already has it passes it to avoid a second read.
    """
    if known is None:
        known = removable_names()
    known_set = set(known)
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        if candidate not in seen:
            seen.add(candidate)
            expanded.append(candidate)

    for name in names:
        if name in known_set:
            _add(name)
            continue
        if _is_glob(name):
            matches = [source for source in known if fnmatch.fnmatchcase(source, name)]
        else:
            matches = folder_members(name, known)
        if matches:
            for match in matches:
                _add(match)
        else:
            _add(name)  # not-found; reported by the store
    return expanded


def unregister_roots(names: Iterable[str]) -> list[str]:
    """Un-register any top-level source root named in *names*. Returns removed labels.

    ``add`` registers a source root; removing it by its label drops the registry
    entry so discovery stops finding its files, which then need no skip marker.
    The source bytes on disk are never touched. Nested names (``corpus/a.txt``)
    are not roots and are left alone.
    """
    config = active_config()
    labels = list(names)
    removed: list[str] = []
    if not labels:
        return removed

    def _mutate(persisted: dict[str, str] | None) -> tuple[dict[str, str], list[str]]:
        roots = dict(persisted or {})
        for name in labels:
            label = name.strip("/")
            if "/" in label or label not in roots:
                continue
            del roots[label]
            removed.append(label)
        config.linked_roots = roots  # refresh the in-process view
        return roots, removed

    return settings.mutate_value(config.data_root, "linked_roots", _mutate)


log = logging.getLogger(__name__)


def remove_documents_durably(names: list[str], targets: list[str] | None = None) -> RemoveResult:
    """Remove documents from the index (folders and globs expand) and make it stick.

    Never deletes source bytes. A name is an indexed source, a file an ingestion
    failure holds out, a folder or glob covering either, or a registered root.
    Each removed file is held out of every later sync as a removal, at its
    current hash, so a held-out failure stays out instead of being retried.
    Removing a registered root un-registers it and drops the skip records under
    it: discovery can no longer find its files. Editing the source (new hash),
    ``rebuild`` or adding the path again restores it; ``retry-skipped`` does not.
    *targets* (the expanded names) is computed when not supplied; a caller that
    already expanded for a confirmation prompt passes it to avoid re-expanding.
    """
    if targets is None:
        targets = expand_remove_targets(names)
    # Taken before the index changes, so a held lock refuses the removal with nothing done.
    with skip_records_lock(active_config().data_root):
        result = get_services().store.remove_documents(targets)
        failed = set(held_out_names(active_config().data_root))
        held = [name for name in result.not_found if name in failed]
        roots = forget_roots(names)
        _hold_out_removed([*result.removed, *held], roots)
    forget_removed_from_wiki_index(list(result.removed))
    missing = [name for name in result.not_found if name not in failed]
    emptied = [name for name in missing if name.strip("/") in roots]
    return RemoveResult(
        removed=[*result.removed, *held, *emptied],
        not_found=[name for name in missing if name not in emptied],
    )


def forget_roots(names: list[str]) -> list[str]:
    """Un-register every root named in *names* and drop the skip records under it.

    The roots are un-registered even when the records cannot be changed; the
    error is raised after.
    """
    roots = active_config().linked_roots
    named = [Path(roots[label]) for label in (name.strip("/") for name in names) if label in roots]
    try:
        # Records resolve through the registry, so they go before un-registering.
        unmark_sources_under(named)
    finally:
        unregistered = unregister_roots(names)
    return unregistered


@dataclass
class AddRollback:
    """What an interrupted add did not add, and any error it hit once stopped."""

    not_added: list[str] = field(default_factory=list)
    error: str | None = None
    paths: list[Path] = field(default_factory=list)
    """The paths the add was given."""
    at_sync: bool = True
    """Whether the add reached its sync."""
    absorbed_into: dict[str, list[str]] = field(default_factory=dict)
    """Each label the add registered that took in sources, with their labels; it stays."""
    _roots: list[str] = field(default_factory=list)
    _before: dict[str, str] = field(default_factory=dict)

    def message(self, nothing_dropped: str) -> str:
        """Why the add stopped, naming what it did not add and any error it also hit."""
        stopped = self._stopped(nothing_dropped)
        return stopped if self.error is None else stopped + _ALSO_HIT_ERROR.format(error=self.error)

    def note_error(self, exc: BaseException) -> None:
        """Log and name *exc*, raised once the add was stopped; a cancel names its cause."""
        if isinstance(exc, _CANCEL_ERRORS):
            # A cancel that took a failed write carries that error as its cause.
            carried = exc.__cause__
            if carried is not None and not isinstance(carried, _CANCEL_ERRORS):
                self.error = str(carried) or type(carried).__name__
            return
        log.warning("A stopped sync also hit an error", exc_info=exc)
        self.error = str(exc) or type(exc).__name__

    def registered(self, labels: list[str], cancel: CancelSignal) -> None:
        """Take *labels* as the roots the add registered; its sync starts here."""
        self._roots, self.at_sync = labels, True
        try:
            self._before = indexed_stamps(labels)
        except Exception as exc:
            if not cancel.is_set():
                raise
            self.note_error(exc)
            # A read that fails once stopped keeps every root with an indexed file.
            self._before = {}

    def forget_unfinished(self) -> None:
        """Un-register each root the sync indexed nothing under, then name what the corpus lacks.

        An error a step raises is named with the cancel, and the other step still runs.
        """
        for step in (self._forget_roots, self._name_missing):
            try:
                step()
            except Exception as exc:
                self.note_error(exc)

    def _forget_roots(self) -> None:
        """Un-register each root with no file indexed since the sync started."""
        forget_unfinished_roots(self._roots, self._before)

    def _name_missing(self) -> None:
        """Take the name of each given path the corpus does not hold."""
        self.not_added = names_outside_corpus(self.paths)

    def _stopped(self, nothing_dropped: str) -> str:
        match self.not_added:
            case [] if self.at_sync:
                return nothing_dropped
            case []:
                return _ADD_CANCELLED
            case [name]:
                return _ADD_CANCELLED_ONE.format(name=name)
            case _:
                return _ADD_CANCELLED_MANY.format(names=", ".join(self.not_added))


def indexed_stamps(labels: list[str]) -> dict[str, str]:
    """The ``ingested_at`` stamp of every indexed source under the roots *labels*."""
    if not labels:
        return {}
    return {
        source["filename"]: source["ingested_at"]
        for source in get_services().store.get_sources()
        if any(is_under(source["filename"], label) for label in labels)
    }


def forget_unfinished_roots(labels: list[str], before: dict[str, str]) -> list[str]:
    """Un-register each root in *labels* with no file indexed since *before*; returns them.

    A file is indexed since *before* when its source is new or its stamp changed.
    """
    finished = [name for name, stamp in indexed_stamps(labels).items() if before.get(name) != stamp]
    unfinished = [label for label in labels if not any(is_under(name, label) for name in finished)]
    return forget_roots(unfinished) if unfinished else []


@contextmanager
def leave_as_cancel(
    rollback: AddRollback, cancel: CancelSignal, user_cancelled: Callable[[], bool]
) -> Generator[None, None, None]:
    """Wrap an add; anything raised while *cancel* is set leaves as a cancel after *rollback*."""
    try:
        yield
    except BaseException as exc:
        if not cancel.is_set():
            raise
        rollback.note_error(exc)
        if user_cancelled():
            rollback.forget_unfinished()
        raise asyncio.CancelledError from exc


@contextmanager
def forget_unfinished_on_cancel(
    paths: list[Path], labels: list[str], cancel: CancelSignal, user_cancelled: Callable[[], bool]
) -> Generator[AddRollback, None, None]:
    """Wrap an add of *paths* after registration; a raise under a set *cancel* is a cancel."""
    rollback = AddRollback(paths=paths)
    rollback.registered(labels, cancel)
    with leave_as_cancel(rollback, cancel, user_cancelled):
        yield rollback


def _hold_out_removed(names: list[str], roots: list[str]) -> None:
    """Hold each of *names* out of later syncs as a removal, except under an un-registered root.

    The marker takes the file's current hash. An imported source has no file and
    needs no marker; a held-out file that is not reachable keeps its marker's hash.
    """
    hashes: dict[str, str] = {}
    unreachable: list[str] = []
    for name in names:
        if any(is_under(name, root) for root in roots):
            continue  # the root is gone; discovery won't resurrect these
        path = resolve_source_path(name)
        if path.exists():
            hashes[name] = file_hash(path)
        else:
            unreachable.append(name)
    mark_removed(active_config().data_root, hashes, unreachable)


def forget_removed_from_wiki_index(removed: list[str]) -> None:
    """Drop removed documents from the wiki's browse index.

    Their skip markers keep them out of later syncs, so no refresh would ever
    revisit their entries and the tree would keep offering pages the library
    can no longer support. Best effort: the removal itself already succeeded.
    """
    if not active_config().wiki or not removed:
        return
    from lilbee.wiki.stubs import drop_sources_from_index

    try:
        drop_sources_from_index(set(removed))
    except Exception:
        log.warning("Failed to drop removed documents from the wiki index", exc_info=True)


@contextmanager
def temporary_ocr_config(
    ocr: OcrMode | None = None,
    ocr_timeout: float | None = None,
) -> Generator[None, None, None]:
    """Override OCR config for the duration of the block, per request.

    Backed by a ContextVar rather than a global ``cfg`` mutation, so concurrent
    ingests on the shared HTTP daemon do not clobber one another's OCR settings.
    """
    from lilbee.data.extract.document import ocr_override

    with ocr_override(ocr, ocr_timeout):
        yield
