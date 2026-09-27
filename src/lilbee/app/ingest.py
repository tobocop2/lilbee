"""Register external source roots, and remove indexed documents durably."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import active_config
from lilbee.data.ingest.discovery import (
    excluded_extension_reasons,
    file_hash,
    resolve_source_path,
)
from lilbee.data.ingest.skip_marker import (
    SkipRecords,
    held_out_names,
    load_skip_markers,
    mark_removed,
    update_skip_records,
)
from lilbee.data.store.types import RemoveResult


@dataclass
class RegisterResult:
    """Result of registering source roots into the knowledge base."""

    registered: list[str] = field(default_factory=list)  # labels newly registered
    name_taken: list[str] = field(default_factory=list)
    """Labels held by a different live source or an owned entry; ``--force`` overwrites."""
    overlapping: list[str] = field(default_factory=list)
    """Paths nesting under or over ``documents_dir`` or a live root; that source covers them."""
    refused: list[str] = field(default_factory=list)
    """Files whose format lilbee does not index, as ``name: reason``."""
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


def _overlaps_existing(src: Path, docs_resolved: Path, roots: dict[str, str]) -> bool:
    """Whether *src* overlaps ``documents_dir`` or a live registered root.

    Two roots covering the same tree would walk the same file twice and index it
    under two keys (double-index). The caller already rejects *src* inside
    ``documents_dir``; this rejects *src* being an ANCESTOR of it, and *src*
    nesting under or over any live registered root. A vanished root cannot
    double-index, so it is ignored.
    """
    if docs_resolved.is_relative_to(src):
        return True
    for target in roots.values():
        root = Path(target)
        if not root.exists():
            continue
        root = root.resolve()
        if src.is_relative_to(root) or root.is_relative_to(src):
            return True
    return False


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
    """
    config = active_config()
    documents_dir = config.documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)
    docs_resolved = documents_dir.resolve()
    result = RegisterResult()
    if not paths:
        return result

    def _mutate(persisted: dict[str, str] | None) -> tuple[dict[str, str], RegisterResult]:
        # Read the registry from config.toml INSIDE the lock (not the possibly
        # stale in-memory copy) so two processes registering roots concurrently
        # cannot lose each other's entry.
        roots = dict(persisted or {})
        by_target = {target: label for label, target in roots.items()}
        refused = excluded_extension_reasons()
        for p in paths:
            src = p.resolve()
            reason = refused.get(src.suffix.lower()) if src.is_file() else None
            if reason is not None:
                result.refused.append(f"{p.name}: {reason}")
                continue
            if src == docs_resolved or docs_resolved in src.parents:
                result.tracked.append(p.name)  # already owned by the knowledge base
                continue
            already = by_target.get(str(src))
            if already is not None:
                result.tracked.append(already)  # this exact source is already registered
                continue
            if _overlaps_existing(src, docs_resolved, roots):
                result.overlapping.append(p.name)  # would walk the same files twice
                continue
            label = _resolve_label(src.name, roots, docs_resolved, force=force)
            if label is None:
                result.name_taken.append(src.name)
                continue
            roots[label] = str(src)
            by_target[str(src)] = label
            result.registered.append(label)
        config.linked_roots = roots  # refresh the in-process view (picks up merges)
        return roots, result

    result = settings.mutate_value(config.data_root, "linked_roots", _mutate)
    unmark_sources_under(paths)
    return result


def unmark_sources_under(paths: list[Path]) -> None:
    """Drop the skip records (marker, reason and kind) of every source *paths* covers.

    A marker exists to stop *discovery* from resurrecting a source the user
    removed, or from re-paying the extract cost on a file that yielded nothing.
    Naming the path outranks it: ``add`` is the user asking for that source
    back, so the marker goes and the sync that follows ingests the file again.
    Without this a removal would be permanent, undoable only by
    ``retry-skipped`` or ``rebuild``, neither of which the user has any reason
    to reach for after typing the path they want.

    Each root in *paths* must be registered when this runs: marker keys resolve
    to files through the live registry.
    """

    def _drop_covered(records: SkipRecords) -> None:
        for name in _markers_covering(records.markers, paths):
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
    ``retry-skipped``, ``rebuild`` or adding the path again restores it.
    *targets* (the expanded names) is computed when not supplied; a caller that
    already expanded for a confirmation prompt passes it to avoid re-expanding.
    """
    if targets is None:
        targets = expand_remove_targets(names)
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
    """Un-register every root named in *names* and drop the skip records under it."""
    roots = active_config().linked_roots
    named = [Path(roots[label]) for label in (name.strip("/") for name in names) if label in roots]
    unmark_sources_under(named)  # records resolve through the registry, so before un-registering
    return unregister_roots(names)


def _hold_out_removed(names: list[str], roots: list[str]) -> None:
    """Hold each of *names* out of later syncs as a removal, except under an un-registered root.

    The marker takes the file's current hash. An imported source has no file and
    needs no marker; a held-out file that is not reachable keeps its marker's hash.
    """
    data_root = active_config().data_root
    markers = load_skip_markers(data_root)
    hashes: dict[str, str] = {}
    for name in names:
        if any(name == root or name.startswith(root + "/") for root in roots):
            continue  # the root is gone; discovery won't resurrect these
        path = resolve_source_path(name)
        if path.exists():
            hashes[name] = file_hash(path)
        elif name in markers:
            hashes[name] = markers[name]
    mark_removed(data_root, hashes)


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
    enable_ocr: bool | None = None,
    ocr_timeout: float | None = None,
) -> Generator[None, None, None]:
    """Override OCR config for the duration of the block, per request.

    Backed by a ContextVar rather than a global ``cfg`` mutation, so concurrent
    ingests on the shared HTTP daemon do not clobber one another's OCR settings.
    """
    from lilbee.data.extract.document import ocr_override

    with ocr_override(enable_ocr, ocr_timeout):
        yield
