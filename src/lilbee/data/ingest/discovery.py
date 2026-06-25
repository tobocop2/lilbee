"""File discovery, classification, and hashing."""

from __future__ import annotations

import hashlib
import logging
import os
from functools import cache
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.system import is_ignored_dir
from lilbee.data.code_chunker import is_code_file
from lilbee.data.ingest.types import IMAGE_CONTENT_TYPE, PDF_CONTENT_TYPE

log = logging.getLogger(__name__)

_PDF_MIME = "application/pdf"


def _content_type_for(ext: str, mime: str) -> str:
    """content_type for a xberg format: PDFs and images grouped, others keyed by extension."""
    if mime == _PDF_MIME:
        return PDF_CONTENT_TYPE
    if mime.startswith("image/"):
        return IMAGE_CONTENT_TYPE
    return ext.lstrip(".")


@cache
def supported_extension_map() -> dict[str, str]:
    """Extension -> content_type for every format xberg can extract.

    Built from ``xberg.list_supported_formats()`` so lilbee covers the full set
    without a hand-maintained list. Source-code files are routed separately (their
    extensions are absent here), so ``classify_file`` falls through to the code path.
    """
    from xberg import list_supported_formats

    out: dict[str, str] = {}
    for fmt in list_supported_formats():
        ext = (fmt.extension if fmt.extension.startswith(".") else f".{fmt.extension}").lower()
        out[ext] = _content_type_for(ext, fmt.mime_type)
    return out


def file_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _relative_name(path: Path) -> str:
    """Get path relative to documents dir as a forward-slash string (portable across OS)."""
    return path.relative_to(cfg.documents_dir).as_posix()


def classify_file(path: Path) -> str | None:
    """Classify a file by extension: a xberg content_type, "code", or None.

    xberg-extractable formats win; source code (not in xberg's set) routes
    to the code chunker; anything else is unsupported.
    """
    doc_type = supported_extension_map().get(path.suffix.lower())
    if doc_type is not None:
        return doc_type
    if is_code_file(path):
        return "code"
    return None


def discover_files() -> dict[str, Path]:
    """Scan documents/ recursively, return {relative_name: absolute_path}."""
    if not cfg.documents_dir.exists():
        return {}
    docs_resolved = cfg.documents_dir.resolve()
    files: dict[str, Path] = {}
    for root, dirs, filenames in os.walk(cfg.documents_dir, topdown=True):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d, cfg.ignore_dirs)]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = Path(root) / fname
            try:
                validate_path_within(path, docs_resolved)
            except ValueError:
                log.warning("Symlink escapes documents dir, skipping: %s", path)
                continue
            if classify_file(path) is not None:
                files[_relative_name(path)] = path
    return files
