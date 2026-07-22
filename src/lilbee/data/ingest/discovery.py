"""File discovery, classification, and hashing."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from lilbee.core.config import active_config
from lilbee.core.system import is_ignored_dir, is_link
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


def _linked_roots(documents_dir: Path) -> dict[str, Path]:
    """Top-level link entries under *documents_dir*, mapped label -> resolved target.

    ``add`` links a prepared source into the knowledge base (a symlink, or a
    junction on unprivileged Windows) rather than copying it. These are the roots
    discovery follows, and the only escapes the containment guard permits. A
    dangling link is skipped (its documents are then marked removed by the next
    sync).
    """
    roots: dict[str, Path] = {}
    try:
        entries = list(documents_dir.iterdir())
    except OSError:
        return roots
    for entry in entries:
        if is_link(entry):
            try:
                roots[entry.name] = entry.resolve(strict=True)
            except OSError:
                continue
    return roots


def _walk_into(
    files: dict[str, Path],
    base: Path,
    label: str | None,
    allowed: tuple[Path, ...],
    ignore_dirs: frozenset[str],
    skip_dirs: frozenset[str] = frozenset(),
) -> None:
    """Walk *base*, recording supported files under *label* that stay within *allowed*.

    A file whose real location resolves outside every allowed root is an escaping
    symlink and is skipped with a warning; this is the containment guard, applied
    per file so a sneaky link nested inside a linked corpus cannot smuggle an
    outside path into the index. Top-level directories named in *skip_dirs* are
    not descended (they are the linked roots, walked separately under their label,
    which also stops os.walk from following a junction into a linked tree twice).
    """
    for root, dirs, filenames in os.walk(base, topdown=True):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d, ignore_dirs)]
        if Path(root) == base:
            dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = Path(root) / fname
            resolved = path.resolve()
            if not any(resolved == r or r in resolved.parents for r in allowed):
                log.warning("Symlink escapes documents dir, skipping: %s", path)
                continue
            if classify_file(path) is not None:
                rel = path.relative_to(base).as_posix()
                files[f"{label}/{rel}" if label else rel] = path


def discover_files() -> dict[str, Path]:
    """Scan documents/ recursively, return {relative_name: absolute_path}.

    Real files under documents/ are keyed by their path relative to it. Top-level
    links that ``add`` created (a symlink, or a junction on Windows) are followed:
    each links to a source living elsewhere on disk, whose files are keyed under
    the link's label so the name stays documents_dir-relative and every downstream
    consumer is unchanged. A symlink that resolves outside documents/ and outside
    these linked roots is an escape and is skipped.
    """
    config = active_config()
    documents_dir = config.documents_dir
    if not documents_dir.exists():
        return {}
    linked = _linked_roots(documents_dir)
    allowed = (documents_dir.resolve(), *linked.values())
    files: dict[str, Path] = {}
    # skip_dirs prunes the linked roots so each is walked exactly once below.
    _walk_into(files, documents_dir, None, allowed, config.ignore_dirs, skip_dirs=frozenset(linked))
    for label, target in linked.items():
        if target.is_dir():
            _walk_into(files, target, label, allowed, config.ignore_dirs)
    return files
