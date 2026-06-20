"""Copy files into the documents directory and OCR config helpers."""

from __future__ import annotations

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.system import is_ignored_dir


@dataclass
class CopyResult:
    """Result of copying files into the documents directory."""

    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _copytree_ignore(directory: str, contents: list[str]) -> set[str]:
    """Ignore callback for shutil.copytree that filters ignored directories."""
    return {
        name
        for name in contents
        if (Path(directory) / name).is_dir() and is_ignored_dir(name, cfg.ignore_dirs)
    }


def copy_files(paths: list[Path], *, force: bool = False) -> CopyResult:
    """Copy paths into documents dir. Returns structured result (no console output)."""
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    result = CopyResult()
    for p in paths:
        dest = cfg.documents_dir / p.name
        validate_path_within(dest, cfg.documents_dir)
        if dest.exists() and not force:
            result.skipped.append(p.name)
            continue
        if p.is_dir():
            shutil.copytree(p, dest, dirs_exist_ok=True, ignore=_copytree_ignore, symlinks=False)
        else:
            shutil.copy2(p, dest)
        result.copied.append(p.name)
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
