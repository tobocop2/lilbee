"""Copy files into the documents directory and OCR config helpers."""

from __future__ import annotations

import shutil
from collections.abc import Generator
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


def remove_documents_durably(names: list[str]) -> RemoveResult:
    """Remove documents from the index and skip-mark them so sync won't re-ingest.

    The source files stay on disk (non-destructive), but each removed file gets a
    skip-marker keyed on its current hash, so the next sync treats it as
    unchanged-and-skipped instead of re-ingesting it and resurrecting the doc.
    Editing the file (new hash), ``retry-skipped``, or ``rebuild`` restores it.
    """
    from lilbee.app.services import get_services
    from lilbee.data.ingest.discovery import file_hash
    from lilbee.data.ingest.skip_marker import (
        load_skip_markers,
        load_skip_reasons,
        write_skip_markers,
        write_skip_reasons,
    )

    result = get_services().store.remove_documents(names)
    if not result.removed:
        return result
    markers = load_skip_markers(cfg.data_root)
    reasons = load_skip_reasons(cfg.data_root)
    for name in result.removed:
        path = cfg.documents_dir / name
        # Imported sources have no file on disk; sync never re-ingests them, so a
        # marker is only needed for real files that would otherwise be re-found.
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
