"""Copy files into the documents directory and OCR config helpers."""

from __future__ import annotations

import shutil
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.core.config import active_config, cfg
from lilbee.core.security import validate_path_within
from lilbee.core.system import is_ignored_dir
from lilbee.data.store.types import RemoveResult


@dataclass
class CopyResult:
    """Result of copying files into the documents directory."""

    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _copytree_ignore(directory: str, contents: list[str]) -> set[str]:
    """Ignore callback for shutil.copytree that filters ignored directories."""
    ignore_dirs = active_config().ignore_dirs
    return {
        name
        for name in contents
        if (Path(directory) / name).is_dir() and is_ignored_dir(name, ignore_dirs)
    }


def copy_files(paths: list[Path], *, force: bool = False) -> CopyResult:
    """Copy paths into documents dir. Returns structured result (no console output)."""
    documents_dir = active_config().documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)
    result = CopyResult()
    for p in paths:
        dest = documents_dir / p.name
        validate_path_within(dest, documents_dir)
        if dest.exists() and not force:
            result.skipped.append(p.name)
            continue
        if p.is_dir():
            shutil.copytree(p, dest, dirs_exist_ok=True, ignore=_copytree_ignore, symlinks=False)
        else:
            shutil.copy2(p, dest)
        result.copied.append(p.name)
    return result


_REMOVED_SKIP_REASON = "removed via delete (re-add the file or run retry-skipped to restore)"


def folder_members(name: str, known: Iterable[str]) -> list[str]:
    """Indexed sources under folder *name*, matched on whole path segments.

    ``myrepo`` covers ``myrepo/a.py`` but never ``myrepo-2/x``. Empty when *name*
    is not a parent directory of any known source. This is the single definition
    of the folder-boundary rule, shared by expansion and the delete-guard.
    """
    prefix = name.rstrip("/") + "/"
    return [source for source in known if source.startswith(prefix)]


def expand_folder_names(names: list[str]) -> list[str]:
    """Expand any folder name to the indexed sources beneath it.

    A name that matches an indexed source exactly is kept. A name that matches no
    source but is a parent directory of one or more sources expands to all of
    them (see :func:`folder_members`). A name that matches neither is kept
    unchanged so the caller reports it not-found. Order and de-duplication are
    preserved.
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
        members = folder_members(name, known)
        if members:
            for member in members:
                _add(member)
        else:
            _add(name)
    return expanded


def remove_documents_durably(
    names: list[str],
    *,
    delete_files: bool = False,
    documents_dir: Path | None = None,
) -> RemoveResult:
    """Remove documents from the index and skip-mark them so sync won't re-ingest.

    A folder name (a parent directory of indexed sources) removes every source
    beneath it. Unless *delete_files* is set the source files stay on disk, and
    each removed file gets a skip-marker keyed on its current hash so the next
    sync treats it as unchanged-and-skipped instead of re-ingesting it and
    resurrecting the doc. Editing the file (new hash), ``retry-skipped``, or
    ``rebuild`` restores it.
    """
    from lilbee.app.services import get_services
    from lilbee.data.ingest.discovery import file_hash
    from lilbee.data.ingest.skip_marker import (
        load_skip_markers,
        load_skip_reasons,
        write_skip_markers,
        write_skip_reasons,
    )

    docs_dir = documents_dir or cfg.documents_dir
    targets = expand_folder_names(names)
    result = get_services().store.remove_documents(
        targets, delete_files=delete_files, documents_dir=docs_dir
    )
    if not result.removed:
        return result
    markers = load_skip_markers(cfg.data_root)
    reasons = load_skip_reasons(cfg.data_root)
    for name in result.removed:
        path = docs_dir / name
        # Imported sources and files removed with delete_files have nothing on
        # disk; sync never re-ingests those, so a marker is only needed for real
        # files that would otherwise be re-found.
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
    from lilbee.data.ingest.extract import ocr_override

    with ocr_override(enable_ocr, ocr_timeout):
        yield
