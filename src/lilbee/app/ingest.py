"""Link sources into the documents directory, and remove them durably."""

from __future__ import annotations

import fnmatch
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.core.config import active_config, cfg
from lilbee.data.store.types import RemoveResult


@dataclass
class LinkResult:
    """Result of linking sources into the documents directory."""

    linked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def link_files(paths: list[Path], *, force: bool = False) -> LinkResult:
    """Symlink each source into the documents dir; never copy its bytes.

    A prepared corpus already lives on local disk, so ``add`` links to it in
    place: a directory becomes one symlink ``documents_dir/<name> -> src`` and a
    file becomes a file symlink. Discovery follows these top-level links, so the
    source is indexed where it lives, with no second copy and no serial byte-copy
    preamble before the GPUs get work. A source already inside documents_dir is
    left as-is (discovery already sees it); an existing name is kept unless
    ``force``, and re-linking the same target is idempotent.
    """
    documents_dir = active_config().documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)
    docs_resolved = documents_dir.resolve()
    result = LinkResult()
    for p in paths:
        src = p.resolve()
        if src == docs_resolved or docs_resolved in src.parents:
            result.skipped.append(p.name)  # already inside the knowledge base
            continue
        # Guard the destination name stays a direct child of documents_dir. Check
        # the name rather than resolving dest: an existing link there would resolve
        # to its (external) target and falsely read as an escape.
        if "/" in p.name or "\\" in p.name or p.name in ("", ".", ".."):
            result.skipped.append(p.name)
            continue
        dest = documents_dir / p.name
        if dest.is_symlink() or dest.exists():
            already_linked = dest.is_symlink() and dest.exists() and dest.resolve() == src
            # A dangling link (its old target is gone) means the source moved:
            # relink to the new path without demanding --force, since there is
            # nothing behind the dead link to protect. Sync recognizes the move
            # by content hash and repoints the index in place.
            dangling = dest.is_symlink() and not dest.exists()
            if already_linked or (not force and not dangling):
                result.skipped.append(p.name)
                continue
            if dest.is_dir() and not dest.is_symlink():
                # A real copied directory holds this name; force won't clobber it.
                result.skipped.append(p.name)
                continue
            dest.unlink()  # replace a stale/dangling symlink or a file on force
        dest.symlink_to(src, target_is_directory=src.is_dir())
        result.linked.append(p.name)
    return result


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


def expand_remove_targets(names: list[str]) -> list[str]:
    """Expand folder names and glob patterns to the indexed sources they cover.

    An exact source name is kept. A folder name (a parent directory of indexed
    sources) expands to every source beneath it. A glob (a name containing
    ``* ? [``) expands to every source it fnmatches. A name matching none of
    these is kept unchanged so the caller reports it not-found. Order and
    de-duplication are preserved.
    """
    from lilbee.app.services import get_services

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


def _unlink_linked_roots(names: Iterable[str], documents_dir: Path) -> list[str]:
    """Unlink any top-level symlink named in *names*, returning those removed.

    ``add`` links a source into the knowledge base as a symlink; removing it by
    its top-level name unlinks the link (never the source bytes behind it) and
    thereby stops discovery from re-finding its files, so those need no skip
    marker. Nested names (``corpus/a.txt``) are left alone.
    """
    unlinked: list[str] = []
    for name in names:
        if "/" in name.strip("/"):
            continue
        entry = documents_dir / name
        if entry.is_symlink():
            entry.unlink()
            unlinked.append(name.strip("/"))
    return unlinked


def remove_documents_durably(names: list[str]) -> RemoveResult:
    """Remove documents from the index (folders and globs expand) and make it stick.

    Never deletes source bytes. A folder or glob argument expands to every
    indexed source it covers. Each removed source gets a skip-marker keyed on its
    current hash so the next sync treats it as unchanged-and-skipped instead of
    re-ingesting it. Removing a top-level linked root instead unlinks that symlink
    (discovery then can't re-find its files, so no markers are needed for them).
    Editing the source (new hash), ``retry-skipped``, or ``rebuild`` restores it.
    """
    from lilbee.app.services import get_services
    from lilbee.data.ingest.discovery import file_hash
    from lilbee.data.ingest.skip_marker import (
        load_skip_markers,
        load_skip_reasons,
        write_skip_markers,
        write_skip_reasons,
    )

    documents_dir = cfg.documents_dir
    targets = expand_remove_targets(names)
    result = get_services().store.remove_documents(targets)
    if not result.removed:
        return result
    unlinked_roots = _unlink_linked_roots(names, documents_dir)
    markers = load_skip_markers(cfg.data_root)
    reasons = load_skip_reasons(cfg.data_root)
    for name in result.removed:
        if any(name == root or name.startswith(root + "/") for root in unlinked_roots):
            continue  # the link is gone; discovery won't resurrect these
        path = documents_dir / name
        # Imported sources have no file on disk; sync never re-ingests them, so a
        # marker is only needed for real files that would otherwise be re-found.
        if path.exists():  # follows the symlink to the real source when linked
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
    from lilbee.data.ingest.extract import ocr_override

    with ocr_override(enable_ocr, ocr_timeout):
        yield
