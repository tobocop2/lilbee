"""Register external source roots, and remove indexed documents durably."""

from __future__ import annotations

import fnmatch
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.app.services import get_services
from lilbee.core import settings
from lilbee.core.config import active_config, cfg
from lilbee.data.store.types import RemoveResult


@dataclass
class RegisterResult:
    """Result of registering source roots into the knowledge base."""

    registered: list[str] = field(default_factory=list)  # labels newly registered
    skipped: list[str] = field(default_factory=list)


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


def source_label_taken(name: str) -> bool:
    """Whether *name* is already a live registered root or an owned top-level entry.

    The confirm-before-overwrite affordance in the TUI reads this; the authority
    on what actually collides is :func:`_resolve_label`, which this mirrors.
    """
    config = active_config()
    existing = config.linked_roots.get(name)
    if existing is not None:
        return Path(existing).exists()
    return (config.documents_dir / name).exists()


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
        for p in paths:
            src = p.resolve()
            if src == docs_resolved or docs_resolved in src.parents:
                result.skipped.append(p.name)  # already owned by the knowledge base
                continue
            already = by_target.get(str(src))
            if already is not None:
                result.skipped.append(already)  # this exact source is already registered
                continue
            if _overlaps_existing(src, docs_resolved, roots):
                result.skipped.append(p.name)  # nests under/over another root; would double-index
                continue
            label = _resolve_label(src.name, roots, docs_resolved, force=force)
            if label is None:
                result.skipped.append(src.name)  # name taken; --force to overwrite
                continue
            roots[label] = str(src)
            by_target[str(src)] = label
            result.registered.append(label)
        config.linked_roots = roots  # refresh the in-process view (picks up merges)
        return roots, result

    return settings.mutate_value(config.data_root, "linked_roots", _mutate)


_REMOVED_SKIP_REASON = "removed via remove (re-add the source or run retry-skipped to restore)"

_GLOB_CHARS = frozenset("*?[")


def _is_glob(name: str) -> bool:
    """Whether *name* should be matched as a glob rather than a literal source."""
    return any(char in _GLOB_CHARS for char in name)


def folder_members(name: str, known: Iterable[str]) -> list[str]:
    """Indexed sources under folder *name*, matched on whole path segments.

    ``myrepo`` covers ``myrepo/a.py`` but never ``myrepo-2/x``. Empty when *name*
    is not a parent directory of any known source.
    """
    prefix = name.rstrip("/") + "/"
    return [source for source in known if source.startswith(prefix)]


def expand_remove_targets(names: list[str], known: list[str] | None = None) -> list[str]:
    """Expand folder names and glob patterns to the indexed sources they cover.

    An exact source name is kept. A folder name (a parent directory of indexed
    sources) expands to every source beneath it. A glob (a name containing
    ``* ? [``) expands to every source it fnmatches. A name matching none of
    these is kept unchanged so the caller reports it not-found. Order and
    de-duplication are preserved. *known* (the indexed source filenames) is read
    from the store when not supplied; a caller that already has it passes it to
    avoid a second read.
    """
    if known is None:
        known = [s["filename"] for s in get_services().store.get_sources()]
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


def remove_documents_durably(names: list[str], targets: list[str] | None = None) -> RemoveResult:
    """Remove documents from the index (folders and globs expand) and make it stick.

    Never deletes source bytes. A folder or glob argument expands to every
    indexed source it covers. Each removed source gets a skip-marker keyed on its
    current hash so the next sync treats it as unchanged-and-skipped instead of
    re-ingesting it. Removing a top-level registered root instead un-registers it
    (discovery then can't re-find its files, so no markers are needed for them).
    Editing the source (new hash), ``retry-skipped``, or ``rebuild`` restores it.
    *targets* (the expanded names) is computed when not supplied; a caller that
    already expanded for a confirmation prompt passes it to avoid re-expanding.
    """
    from lilbee.data.ingest.discovery import file_hash, resolve_source_path
    from lilbee.data.ingest.skip_marker import (
        load_skip_markers,
        load_skip_reasons,
        write_skip_markers,
        write_skip_reasons,
    )

    if targets is None:
        targets = expand_remove_targets(names)
    result = get_services().store.remove_documents(targets)
    if not result.removed:
        return result
    unregistered = unregister_roots(names)
    markers = load_skip_markers(cfg.data_root)
    reasons = load_skip_reasons(cfg.data_root)
    for name in result.removed:
        if any(name == root or name.startswith(root + "/") for root in unregistered):
            continue  # the root is gone; discovery won't resurrect these
        path = resolve_source_path(name)
        # Imported sources have no file on disk; sync never re-ingests them, so a
        # marker is only needed for a real file that would otherwise be re-found.
        if path.exists():
            markers[name] = file_hash(path)
            reasons[name] = _REMOVED_SKIP_REASON
    write_skip_markers(cfg.data_root, markers)
    write_skip_reasons(cfg.data_root, reasons)
    return result


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
