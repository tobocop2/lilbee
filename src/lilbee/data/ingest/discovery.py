"""File discovery, classification, hashing, and source-path resolution."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from functools import cache
from pathlib import Path

from lilbee.core.config import active_config
from lilbee.core.system import is_ignored_dir
from lilbee.data.extract.code_chunker import is_code_file
from lilbee.data.types import IMAGE_CONTENT_TYPE, PDF_CONTENT_TYPE

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


# How often the discovery walk logs progress. The walk runs before the file count
# is known (it is what produces the count), so it cannot show an ETA; it just
# proves the run is alive. A large or NFS-backed tree can take minutes to walk,
# during which the plan pass has not started and nothing else logs.
_SCAN_LOG_INTERVAL_S = 10.0


class _ScanProgress:
    """Periodic progress for the pre-plan discovery walk.

    Emitted at warning level, not info: the default LILBEE_LOG_LEVEL is WARNING,
    so an info line would be filtered before any handler and a headless
    ``lilbee sync`` would show nothing while the tree is walked. Interval-gated,
    so a fast walk (the common case) stays silent -- the first line appears only
    once the walk has run longer than the interval.
    """

    def __init__(self) -> None:
        self._examined = 0
        self._matched = 0
        self._started = time.monotonic()
        self._last = self._started

    def tick(self, *, matched: bool) -> None:
        self._examined += 1
        if matched:
            self._matched += 1
        now = time.monotonic()
        if now - self._last < _SCAN_LOG_INTERVAL_S:
            return
        self._last = now
        elapsed = now - self._started
        rate = self._examined / elapsed if elapsed > 0 else 0.0
        log.warning(
            "Scanning for files: examined %d, matched %d (%.0f files/s, %.0fs elapsed)",
            self._examined,
            self._matched,
            rate,
            elapsed,
        )


def file_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


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


def resolve_source_path(filename: str) -> Path:
    """Map a stored source key back to the file it tracks on disk.

    A key's first segment is a registered root label when ``add`` recorded that
    root; the file then lives at ``linked_roots[label]/<rest>`` (or at the root
    itself for a single-file root, where there is no rest). Every other key
    belongs to a file lilbee owns under ``documents_dir`` and resolves there.
    The path is returned whether or not it still exists: a source whose file was
    moved or deleted keeps its index entry, and the dead path surfaces only when
    something tries to open it.

    A registered label owns its whole key namespace: if an owned ``documents_dir``
    subtree of the same top-level name is created after the root is registered,
    its files resolve to the root, not the owned copy. ``discover_files`` walks
    the root after the owned tree and so keys the same file identically, keeping
    resolution and discovery in agreement; ``add`` blocks the reverse collision
    (registering a label that shadows an existing owned entry).
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
    progress: _ScanProgress,
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
            content_type = classify_file(path)
            # tick per file visited, not per match: a skip-heavy tree still walks
            # slowly and must still show a heartbeat.
            progress.tick(matched=content_type is not None)
            if content_type is None:
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
    progress = _ScanProgress()
    if config.documents_dir.exists():
        _walk_root(files, config.documents_dir, None, config.ignore_dirs, progress)
    for label, root in config.linked_roots.items():
        root_path = Path(root)
        if root_path.is_dir():
            _walk_root(files, root_path, label, config.ignore_dirs, progress)
        elif root_path.is_file() and classify_file(root_path) is not None:
            files[label] = root_path
    return files
