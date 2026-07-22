"""File discovery, classification, hashing, and source-path resolution."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from lilbee.core.config import active_config
from lilbee.core.system import is_ignored_dir
from lilbee.data.code_chunker import is_code_file
from lilbee.data.ingest.types import DOCUMENT_EXTENSION_MAP

log = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def classify_file(path: Path) -> str | None:
    """Classify file by extension. Returns content_type or None if unsupported."""
    doc_type = DOCUMENT_EXTENSION_MAP.get(path.suffix.lower())
    if doc_type is not None:
        return doc_type
    if is_code_file(path):
        return "code"
    return None


def resolve_source_path(filename: str) -> Path:
    """Map a stored source key back to the file it tracks on disk.

    A key's first segment is a registered root label when ``add`` recorded that
    root; the file then lives at ``linked_roots[label]/<rest>`` (or at the root
    itself for a single-file root, where there is no rest). Every other key
    belongs to a file lilbee owns under ``documents_dir`` and resolves there.
    The path is returned whether or not it still exists: a source whose file was
    moved or deleted keeps its index entry, and the dead path surfaces only when
    something tries to open it.
    """
    config = active_config()
    first, _, rest = filename.partition("/")
    root = config.linked_roots.get(first)
    if root is not None:
        base = Path(root)
        return base / rest if rest else base
    return config.documents_dir / filename


def resolve_source_path_checked(filename: str) -> Path | None:
    """Resolve *filename*, returning None if it escapes its owning root.

    Guards a surface that resolves a caller-supplied source key (the HTTP
    document-serving endpoint): a key with ``..`` that would climb out of
    ``documents_dir`` or a registered root is rejected. Keys produced by
    discovery never contain ``..``; this defends against a crafted request, not
    stored data.
    """
    config = active_config()
    resolved = resolve_source_path(filename).resolve(strict=False)
    roots = [
        config.documents_dir.resolve(),
        *(Path(root).resolve() for root in config.linked_roots.values()),
    ]
    if any(resolved == root or root in resolved.parents for root in roots):
        return resolved
    return None


def _walk_root(
    files: dict[str, Path],
    base: Path,
    label: str | None,
    ignore_dirs: frozenset[str],
) -> None:
    """Record supported files under *base*, keyed relative to it (prefixed by *label*).

    Symlinks are not followed (``followlinks=False``): each root is walked as the
    real tree it names, so there is no traversal loop and no path can escape the
    root it was registered under.
    """
    for root, dirs, filenames in os.walk(base, topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d, ignore_dirs)]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = Path(root) / fname
            if classify_file(path) is None:
                continue
            rel = path.relative_to(base).as_posix()
            files[f"{label}/{rel}" if label else rel] = path


def discover_files() -> dict[str, Path]:
    """Scan the owned documents dir and every registered root, return {key: path}.

    Files lilbee owns under ``documents_dir`` (crawl and upload output) are keyed
    by their path relative to it. Each root ``add`` registered is indexed where it
    lives: a directory root contributes its files keyed under the root's label; a
    single-file root contributes one entry keyed by the label alone. A root whose
    path has since vanished contributes nothing this pass, and its already-indexed
    sources are left in place (a dead path-link, not a removal).
    """
    config = active_config()
    files: dict[str, Path] = {}
    if config.documents_dir.exists():
        _walk_root(files, config.documents_dir, None, config.ignore_dirs)
    for label, root in config.linked_roots.items():
        root_path = Path(root)
        if root_path.is_dir():
            _walk_root(files, root_path, label, config.ignore_dirs)
        elif root_path.is_file() and classify_file(root_path) is not None:
            files[label] = root_path
    return files
