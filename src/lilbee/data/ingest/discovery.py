"""File discovery, classification, hashing, and source-path resolution."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Iterator, Mapping
from enum import StrEnum
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from lilbee.core.config import active_config
from lilbee.core.system import is_ignored_dir
from lilbee.data.extract.code_chunker import is_code_file
from lilbee.data.ingest.ignore import IgnoreRules
from lilbee.data.types import IMAGE_CONTENT_TYPE, PDF_CONTENT_TYPE, ShardId

log = logging.getLogger(__name__)

_PDF_MIME = "application/pdf"

# Base MIME subtypes that name a container of other files.
_ARCHIVE_SUBTYPES = frozenset(
    {
        "7z-compressed",
        "bzip",
        "bzip2",
        "compress",
        "gtar",
        "gzip",
        "lzip",
        "lzma",
        "rar",
        "rar-compressed",
        "tar",
        "xz",
        "zip",
        "zip-compressed",
        "zstd",
    }
)
# Prefixes that mark a subtype as unregistered or vendor-specific, not part of its name.
_SUBTYPE_PREFIXES = ("x-", "vnd.", "prs.")


class ExclusionReason(StrEnum):
    """Why discovery refuses a file whose extension xberg could otherwise extract."""

    VECTOR_GRAPHIC = "vector graphic, not a document"
    NEEDS_TRANSCRIPTION = "audio or video, needs a transcription model lilbee does not run"


# Refusals the MIME type cannot express: image/svg+xml is a drawing, not a scan.
_DENIED_EXTENSIONS: dict[str, ExclusionReason] = {".svg": ExclusionReason.VECTOR_GRAPHIC}
# Refusals by MIME type: xberg errors on these without a transcription config.
_DENIED_MIME_PREFIXES: dict[str, ExclusionReason] = {
    "audio/": ExclusionReason.NEEDS_TRANSCRIPTION,
    "video/": ExclusionReason.NEEDS_TRANSCRIPTION,
}


def _content_type_for(ext: str, mime: str) -> str:
    """content_type for a xberg format: PDFs and images grouped, others keyed by extension."""
    if mime == _PDF_MIME:
        return PDF_CONTENT_TYPE
    if mime.startswith("image/"):
        return IMAGE_CONTENT_TYPE
    return ext.lstrip(".")


def _normalized_ext(extension: str) -> str:
    """A xberg format's extension as a lowercase suffix with its leading dot."""
    ext = extension.lower()
    return ext if ext.startswith(".") else f".{ext}"


def _is_archive_mime(mime: str) -> bool:
    """Whether the base subtype of *mime* names an archive.

    ``application/epub+zip`` reads as ``epub``; ``application/x-tar`` reads as ``tar``.
    """
    subtype = mime.partition("/")[2].partition("+")[0].strip().lower()
    for prefix in _SUBTYPE_PREFIXES:
        subtype = subtype.removeprefix(prefix)
    return subtype in _ARCHIVE_SUBTYPES


def _denied_mime_reason(mime: str) -> ExclusionReason | None:
    return next(
        (reason for prefix, reason in _DENIED_MIME_PREFIXES.items() if mime.startswith(prefix)),
        None,
    )


@cache
def excluded_extension_reasons() -> Mapping[str, ExclusionReason]:
    """Extension -> why discovery refuses it, for xberg formats lilbee will not ingest."""
    from xberg import list_supported_formats

    refused = dict(_DENIED_EXTENSIONS)
    for fmt in list_supported_formats():
        reason = _denied_mime_reason(fmt.mime_type)
        if reason is not None:
            refused[_normalized_ext(fmt.extension)] = reason
    return MappingProxyType(refused)


@cache
def archive_content_types() -> frozenset[str]:
    """content_types of the containers whose members ingest as their own sources.

    Found by the MIME type xberg reports, so ``application/epub+zip`` stays a book.
    """
    from xberg import list_supported_formats

    return frozenset(
        _content_type_for(_normalized_ext(fmt.extension), fmt.mime_type)
        for fmt in list_supported_formats()
        if _is_archive_mime(fmt.mime_type)
    )


def member_content_type(path: str, mime: str) -> str:
    """content_type of an archive member: by MIME type, then extension, then MIME subtype."""
    return _content_type_for(Path(path).suffix.lower(), mime) or mime.partition("/")[2]


@cache
def supported_extension_map() -> dict[str, str]:
    """Extension -> content_type for every format lilbee ingests.

    Built from ``xberg.list_supported_formats()`` so lilbee covers the full set
    without a hand-maintained list, minus the formats ``excluded_extension_reasons``
    refuses. Source-code files are routed separately (their extensions are absent
    here), so ``classify_file`` falls through to the code path.
    """
    from xberg import list_supported_formats

    excluded = excluded_extension_reasons()
    out: dict[str, str] = {}
    for fmt in list_supported_formats():
        ext = _normalized_ext(fmt.extension)
        if ext not in excluded:
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

    Ingestable xberg formats win; source code (not in xberg's set) routes to the
    code chunker; a refused container and anything else is unsupported.
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


def resolve_source_root(filename: str) -> tuple[Path, Path] | None:
    """The walked root and the resolved path for *filename*, or None if none walks it.

    Pairs a source key with the base its patterns are written relative to, so the
    index can be reconciled against ``.lilbeeignore`` without a second walk. A
    single-file root is the file the user named, never a tree, so nothing walks
    it and no pattern applies.
    """
    config = active_config()
    first, _, rest = filename.partition("/")
    root = config.linked_roots.get(first)
    if root is None:
        return config.documents_dir, config.documents_dir / filename
    if not rest:
        return None
    base = Path(root)
    return base, base / rest


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


class ScannedFile(NamedTuple):
    """One file the corpus walk kept: its source key, its path, and why it is refused.

    ``excluded`` is None for a file that will be ingested.
    """

    key: str
    path: Path
    excluded: ExclusionReason | None = None


class CorpusScan(NamedTuple):
    """One walk's outcome: the files to ingest, and the refused ones keyed to their reason."""

    files: dict[str, Path]
    excluded: dict[str, ExclusionReason]


def _scan_entry(path: Path, key: str) -> ScannedFile | None:
    """The walk's verdict for one file, or None when nothing here can be ingested."""
    reason = excluded_extension_reasons().get(path.suffix.lower())
    if reason is not None:
        return ScannedFile(key, path, reason)
    if classify_file(path) is None:
        return None
    return ScannedFile(key, path)


def _walk_root(
    base: Path,
    label: str | None,
    ignore_dirs: frozenset[str],
    progress: _ScanProgress,
    rules: IgnoreRules,
) -> Iterator[ScannedFile]:
    """Yield the files under *base* lilbee knows, keyed relative to it (prefixed by *label*).

    A refused format is yielded with its reason; an unknown format is left out.

    Symlinks are not followed (``followlinks=False``): each root is walked as the
    real tree it names, so there is no traversal loop and no path can escape the
    root it was registered under.

    A directory ``.lilbeeignore`` excludes is pruned rather than filtered per
    file, so an excluded tree costs nothing to skip and no pattern beneath it can
    re-include a file -- git's rule, holding here because the walk never descends.
    """
    for root, dirs, filenames in os.walk(base, topdown=True, followlinks=False):
        here = Path(root)
        dirs[:] = [
            d
            for d in dirs
            if not is_ignored_dir(d, ignore_dirs)
            and not rules.excludes_entry(here / d, base=base, is_dir=True)
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = here / fname
            if rules.excludes_entry(path, base=base, is_dir=False):
                progress.tick(matched=False)
                continue
            rel = path.relative_to(base).as_posix()
            entry = _scan_entry(path, f"{label}/{rel}" if label else rel)
            # tick per file visited, not per match: a skip-heavy tree still walks
            # slowly and must still show a heartbeat.
            progress.tick(matched=entry is not None and entry.excluded is None)
            if entry is not None:
                yield entry


def _walk_corpus(rules: IgnoreRules | None = None) -> Iterator[ScannedFile]:
    """Yield every file lilbee knows in the owned tree and in each registered root.

    A single-file root is the file the user named at ``add`` time, so no ignore
    pattern is consulted for it: naming a file is a stronger statement than a
    pattern that would have swept it up.
    """
    config = active_config()
    progress = _ScanProgress()
    rules = rules if rules is not None else IgnoreRules.for_corpus()
    if config.documents_dir.exists():
        yield from _walk_root(config.documents_dir, None, config.ignore_dirs, progress, rules)
    for label, root in config.linked_roots.items():
        root_path = Path(root)
        if root_path.is_dir():
            yield from _walk_root(root_path, label, config.ignore_dirs, progress, rules)
        elif root_path.is_file() and (entry := _scan_entry(root_path, label)) is not None:
            yield entry


def discover_corpus(shard: ShardId | None = None, rules: IgnoreRules | None = None) -> CorpusScan:
    """Scan the owned documents dir and every registered root into a :class:`CorpusScan`.

    A refused file (see ``excluded_extension_reasons``) lands in ``excluded``, not ``files``.
    """
    files: dict[str, Path] = {}
    excluded: dict[str, ExclusionReason] = {}
    for entry in _walk_corpus(rules):
        if shard is not None and not shard.owns(entry.key):
            continue
        if entry.excluded is not None:
            excluded[entry.key] = entry.excluded
        else:
            files[entry.key] = entry.path
    return CorpusScan(files, excluded)


def discover_files(
    shard: ShardId | None = None, rules: IgnoreRules | None = None
) -> dict[str, Path]:
    """Scan the owned documents dir and every registered root, return {key: path}.

    Files lilbee owns under ``documents_dir`` (crawl and upload output) are keyed
    by their path relative to it. Each root ``add`` registered is indexed where it
    lives: a directory root contributes its files keyed under the root's label; a
    single-file root contributes one entry keyed by the label alone. A root whose
    path has since vanished contributes nothing this pass, and its already-indexed
    sources are left in place (a dead path-link, not a removal).

    A *shard* keeps only the keys that slice owns, so one worker of a multi-GPU
    ingest holds the paths of its own slice and not the whole corpus.

    A caller that also reconciles the index passes the *rules* it will reconcile
    with, so the walk and that pass read one set of compiled patterns.
    """
    return discover_corpus(shard, rules).files


def corpus_has_at_least(count: int) -> bool:
    """Whether the corpus holds at least *count* ingestable files.

    Stops at the threshold. The answer gates the multi-GPU ingest fan-out, and
    walking a million-file tree to learn "yes, more than a few thousand" would
    cost minutes before any work starts.
    """
    ingestable = (entry for entry in _walk_corpus() if entry.excluded is None)
    return any(seen >= count for seen, _ in enumerate(ingestable, start=1))
