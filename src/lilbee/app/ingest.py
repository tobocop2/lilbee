"""Register external source roots, and remove indexed documents durably."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import Counter
from collections.abc import Callable, Collection, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import Config, active_config
from lilbee.core.config.enums import OcrMode
from lilbee.core.system import DiskSpelling, disk_path
from lilbee.data.ingest.discovery import (
    excluded_extension_reasons,
    file_hash,
    resolve_source_path,
    walk_reaches,
)
from lilbee.data.ingest.ignore import IgnoreRules
from lilbee.data.store import REMOVED_REASON, HeldOut, HeldRecord
from lilbee.data.store.locations import SourceLocations, root_location
from lilbee.data.store.types import RemoveResult
from lilbee.data.types import is_under
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError
from lilbee.runtime.lock import (
    LibraryUnlockableError,
    LockingUnsupportedError,
    refuse_where_a_sync_cannot_lock,
)

_ADD_CANCELLED = "Add cancelled."
_ADD_CANCELLED_ONE = "Add cancelled. {name} was not added."
_ADD_CANCELLED_MANY = "Add cancelled. {names} were not added."
_ALSO_HIT_ERROR = " It also hit an error: {error}."
_CANCEL_ERRORS = (asyncio.CancelledError, TaskCancelledError, KeyboardInterrupt)
# What an add, a remove or a sync raises when another lilbee process or a data
# root it cannot lock keeps it from starting.
LIBRARY_REFUSALS = (
    LibraryUnlockableError,
    LockingUnsupportedError,
)


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


def _live_roots(spelled: dict[str, Path]) -> list[Path]:
    """Each of the disk paths in *spelled* that exists; a vanished root overlaps nothing."""
    return [path for path in spelled.values() if path.exists()]


def _inside_a_root(src: Path, live_roots: list[Path]) -> bool:
    """Whether a live root already walks *src*; a second root there would index it twice."""
    return any(src.is_relative_to(root) for root in live_roots)


def _reached_below(src: Path, spelled: dict[str, Path]) -> dict[str, Path]:
    """Each root of *spelled* below *src* that the walk of *src* reaches, by label.

    *spelled* holds the disk path of each registered root. A root whose folder is
    gone counts by its stored path.
    """
    below = {
        label: path for label, path in spelled.items() if path != src and path.is_relative_to(src)
    }
    if not below:
        return below
    rules = IgnoreRules.for_corpus(active_config().data_root)
    return {label: path for label, path in below.items() if walk_reaches(src, path, rules)}


def _classify(
    paths: list[Path], roots: dict[str, str], docs_resolved: Path, *, force: bool
) -> RegisterResult:
    """Sort *paths* by what registering each one does, leaving *roots* as the registry after it.

    A path of this call that a later path of it contains was never a source: it
    leaves the registry and is not reported as taken in.
    """
    result = RegisterResult()
    sources = frozenset(roots)
    spelling = DiskSpelling()
    # The disk is asked one time for each root; every comparison below reads this.
    spelled = {label: spelling.of(Path(target)) for label, target in roots.items()}
    by_target = {path: label for label, path in spelled.items()}
    refused = excluded_extension_reasons()
    outside: dict[Path, str] = {}
    for p in paths:
        src = spelling.of(p)
        reason = refused.get(src.suffix.lower()) if src.is_file() else None
        if reason is not None:
            result.refused.append(f"{p.name}: {reason}")
            outside.setdefault(src, p.name)
            continue
        if src == docs_resolved or docs_resolved in src.parents:
            result.tracked.append(p.name)  # already owned by the knowledge base
            continue
        already = by_target.get(src)
        if already is not None:
            result.tracked.append(already)  # this exact source is already registered
            continue
        if _inside_a_root(src, _live_roots(spelled)):
            result.overlapping.append(p.name)  # would walk the same files twice
            continue
        reached = _reached_below(src, spelled)
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
        for child in reached:
            del roots[child]
            by_target.pop(spelled.pop(child), None)
        taken = sorted(child for child in reached if child in sources)
        roots[label] = str(src)
        spelled[label] = src
        by_target[src] = label
        result.registered = [name for name in result.registered if name not in reached] + [label]
        if taken:
            result.absorbed_into[label] = taken
    result.outside_corpus = list(outside.values())
    return result


def names_outside_corpus(paths: list[Path]) -> list[str]:
    """The name of each of *paths* the corpus does not hold, by the rule registration applies."""
    if not paths:
        return []  # a stopped sync has no paths, and reads no registry to say so
    config = active_config()
    roots = dict(settings.load(config.data_root).get("linked_roots") or {})
    plan = _classify(paths, roots, disk_path(config.documents_dir), force=False)
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
    held = roots.get(name)
    if target is not None and held is not None and disk_path(Path(held)) == disk_path(target):
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
    the source leaves the registry, and its files, held-out records and wiki
    citations show below the new label, with nothing moved, extracted or embedded
    again. A label that takes a new folder takes the sources of its old one along;
    an add that stops before that leaves them to the next sync, which pairs them
    with their files by content.
    A data root whose filesystem cannot lock files raises ``LockingUnsupportedError``
    for every add, and a sync lock that cannot be opened raises
    ``LibraryUnlockableError``, because the sync after the add refuses there.
    """
    config = active_config()
    if paths:
        refuse_where_a_sync_cannot_lock(config.data_root)
    documents_dir = config.documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)
    docs_resolved = disk_path(documents_dir)
    if not paths:
        return RegisterResult()

    def _mutate(
        persisted: dict[str, str] | None,
    ) -> tuple[dict[str, str], tuple[RegisterResult, dict[str, str]]]:
        # Read the registry from config.toml INSIDE the lock (not the possibly
        # stale in-memory copy) so two processes registering roots concurrently
        # cannot lose each other's entry.
        before = dict(persisted or {})
        roots = dict(before)
        result = _classify(paths, roots, docs_resolved, force=force)
        return roots, (result, before)

    result, before = settings.mutate_value(config.data_root, "linked_roots", _mutate)
    settings.sync_from_resolver(config, ["linked_roots"])
    if result.registered:
        _follow_registered(before, result)
    names = SourceLocations(config.linked_roots)
    taken = [names.name(root_location(before[label])) for label in result.absorbed]
    if taken:
        show_current_paths_in_wiki()
    unmark_sources_under(paths, spared=[*taken, *_kept_sources(result)])
    return result


def _follow_registered(before: dict[str, str], result: RegisterResult) -> None:
    """Move the sources of each label *result* gave a new folder there, then name by location.

    A source whose label is now registered is stored where it is. A parent that
    takes in the source of its own label leaves that source's files where they are.
    """
    store = get_services().store
    for label in result.registered:
        if label in before and label not in result.absorbed_into.get(label, []):
            store.relocate_root(before[label], active_config().linked_roots[label])
    store.fill_locations()


def _kept_sources(result: RegisterResult) -> list[str]:
    """The label of each registered source the add did not name, whose records stay."""
    named = {*result.registered, *result.tracked}
    return [label for label in active_config().linked_roots if label not in named]


def unmark_sources_under(paths: list[Path], *, spared: Iterable[str] = ()) -> None:
    """Hold out of a sync no more any source *paths* covers.

    A record at or below a key in *spared* stays, unless a path names a file
    inside that source: the add asks for that file back. A parent that takes in a
    source keeps what the user removed from it and what failed in it.

    A record exists to stop *discovery* from resurrecting a source the user
    removed, or from re-paying the extract cost on a file that yielded nothing.
    Naming the path outranks it: ``add`` is the user asking for that source
    back, so the record goes and the sync that follows ingests the file again.
    Without this a removal would be permanent, undoable only by ``rebuild``,
    which the user has no reason to reach for after typing the path they want.

    Each root in *paths* must be registered when this runs: held keys resolve
    to files through the live registry.
    """
    kept = list(spared)
    store = get_services().store
    covered, inside = _held_covering(store.held_records(), paths)
    store.release_held(
        name for name in covered if name in inside or not any(is_under(name, key) for key in kept)
    )


def _held_covering(held: Collection[str], paths: list[Path]) -> tuple[set[str], set[str]]:
    """Held keys whose file is one of *paths* or lives beneath one, and those a path below
    their own root covers.

    Each key is resolved back to the file it tracks -- the same mapping
    discovery keyed it by -- so an owned ``documents_dir`` entry, a file under a
    registered root, and a single-file root are all matched by the one rule
    instead of three shape-specific ones.
    """
    if not held:
        return set(), set()  # nothing to match, so the disk is not asked
    config = active_config()
    with settings.fresh_registry_pinned(config):
        return _covering_one_registry(config, held, paths)


def _covering_one_registry(
    config: Config, held: Collection[str], paths: list[Path]
) -> tuple[set[str], set[str]]:
    """Match *held* against *paths* while the registry in *config* stays fixed."""
    spelling = DiskSpelling()
    named = [spelling.of(p) for p in paths]
    bases = (config.documents_dir, *map(Path, config.linked_roots.values()))
    covered: set[str] = set()
    inside: set[str] = set()
    for name in held:
        stored = resolve_source_path(name)
        base = next(base for base in bases if stored.is_relative_to(base))
        # The disk spells the root, one time; the rest of a key is the walk's own spelling.
        root = spelling.of(base)
        tracked = (root / stored.relative_to(base)).resolve(strict=False)
        hits = [path for path in named if tracked == path or path in tracked.parents]
        if hits:
            covered.add(name)
        if any(path != root and path.is_relative_to(root) for path in hits):
            inside.add(name)
    return covered, inside


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


def held_failures() -> dict[str, HeldRecord]:
    """The record of each file an ingestion failure holds out, by path, sorted."""
    held = get_services().store.held_records()
    return {name: held[name] for name in sorted(held) if held[name].kind is HeldOut.FAILED}


def removable_names(indexed: list[str] | None = None) -> list[str]:
    """What a remove can name: *indexed* (the store's sources when not given), then failures."""
    if indexed is None:
        indexed = [s["filename"] for s in get_services().store.get_sources()]
    seen = set(indexed)
    return indexed + [name for name in held_failures() if name not in seen]


def expand_remove_targets(names: list[str], known: list[str]) -> list[str]:
    """Expand folder names and glob patterns to the known sources they cover.

    An exact source name is kept. A folder name (a parent directory of known
    sources) expands to every source beneath it. A glob (a name containing
    ``* ? [``) expands to every source it fnmatches. A name matching none of
    these is kept unchanged so the caller reports it not-found. Order and
    de-duplication are preserved. *known* is what a remove can name,
    ``removable_names()``.
    """
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
    entry so discovery stops finding its files, which then need no held-out record.
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
        return roots, removed

    result = settings.mutate_value(config.data_root, "linked_roots", _mutate)
    settings.sync_from_resolver(config, ["linked_roots"])
    return result


log = logging.getLogger(__name__)


def remove_documents_durably(
    names: list[str],
    targets: list[str] | None = None,
) -> RemoveResult:
    """Remove documents from the index (folders and globs expand) and make it stick.

    Never deletes source bytes. A name is an indexed source, a file an ingestion
    failure holds out, a folder or glob covering either, or a registered root.
    Each removed file is held out of every later sync as a removal, at its
    current hash, so a held-out failure stays out instead of being retried.
    Removing a registered root un-registers it and drops the held-out records under
    it: discovery can no longer find its files. Editing the source (new hash),
    ``rebuild`` or adding the path again restores it; ``retry-skipped`` does not.
    *targets* (the expanded names) is computed when not supplied; a caller that
    already expanded for a confirmation prompt passes it to avoid re-expanding.

    The removal and the records it holds out are one write of the index.
    """
    store = get_services().store
    known = removable_names()
    if targets is None:
        targets = expand_remove_targets(names, known)
    failed = held_failures()
    settings.refresh_registry(active_config())
    linked = active_config().linked_roots
    going = [label for label in (name.strip("/") for name in names) if label in linked]
    removable = set(known)
    holds = _removal_holds([name for name in targets if name in removable], going, failed)
    result = store.remove_documents(targets, holds)
    held = [name for name in result.not_found if name in failed]
    roots = forget_roots(names)
    forget_removed_from_wiki_index(list(result.removed))
    missing = [name for name in result.not_found if name not in failed]
    emptied = [name for name in missing if name.strip("/") in roots]
    return RemoveResult(
        removed=[*result.removed, *held, *emptied],
        not_found=[name for name in missing if name not in emptied],
    )


def forget_roots(names: list[str]) -> list[str]:
    """Un-register every root named in *names* and drop the held-out records under it.

    A source row a root keeps for a citation is stored by its shown name first.
    The roots are un-registered even when the records cannot be changed; the
    error is raised after.
    """
    settings.refresh_registry(active_config())
    roots = active_config().linked_roots
    labels = [label for label in (name.strip("/") for name in names) if label in roots]
    named = [Path(roots[label]) for label in labels]
    try:
        # Records resolve through the registry, so they go before un-registering.
        # A root that stays keeps its records, also when it names the folder of one that goes.
        unmark_sources_under(named, spared=roots.keys() - set(labels))
        get_services().store.name_by_label(labels)
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


def _removal_holds(
    names: list[str], roots: list[str], failed: dict[str, HeldRecord]
) -> dict[str, HeldRecord]:
    """The removal record each of *names* takes, except under a root that is un-registered.

    The record takes the file's current hash. A source with no file, an imported
    one or an archive member, takes none; a held-out failure that is not
    reachable keeps the hash it failed at.
    """
    holds: dict[str, HeldRecord] = {}
    for name in names:
        if any(is_under(name, root) for root in roots):
            continue  # the root is gone; discovery won't resurrect these
        path = resolve_source_path(name)
        if path.is_file():
            holds[name] = HeldRecord(HeldOut.REMOVED, file_hash(path), REMOVED_REASON)
        elif name in failed:
            holds[name] = HeldRecord(HeldOut.REMOVED, failed[name].file_hash, REMOVED_REASON)
    return holds


def remove_sources_and_forget_in_wiki(names: list[str]) -> RemoveResult:
    """Remove indexed sources by name and drop them from the wiki's browse index."""
    result = get_services().store.remove_documents(names)
    forget_removed_from_wiki_index(list(result.removed))
    return result


def forget_removed_from_wiki_index(removed: list[str]) -> None:
    """Drop removed documents from the wiki's browse index.

    Their removal records keep them out of later syncs, so no refresh would ever
    revisit their entries and the tree would keep offering pages the library
    can no longer support. Best effort: the removal itself already succeeded.
    """
    if removed:
        forget_missing_from_wiki_index()


def forget_missing_from_wiki_index() -> None:
    """Drop from the wiki's browse index each source it lists that holds no source row.

    For the parent of a fan-out that stopped: its workers remove sources and
    report nothing. Best effort, as the removal already succeeded.
    """
    config = active_config()
    if not config.wiki:
        return
    from lilbee.wiki.stubs import drop_removed_sources

    try:
        drop_removed_sources(get_services().store, config)
    except Exception:
        log.warning("Failed to drop missing documents from the wiki index", exc_info=True)


def show_current_paths_in_wiki() -> None:
    """Render again each wiki page that shows a path its cited source no longer has.

    Best effort, as the sync itself already succeeded.
    """
    store = get_services().store
    try:
        citations = store.wiki_citations_by_page()
        if not citations:
            return
        # heavy: the wiki package loads spaCy
        from lilbee.wiki.rerender import rerender_stale_pages

        rerender_stale_pages(store, active_config(), citations)
    except Exception:
        log.warning("Failed to show the current document paths in the wiki pages", exc_info=True)


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
