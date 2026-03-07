"""Document sync engine — keeps documents/ dir in sync with LanceDB."""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from rich.progress import Progress, SpinnerColumn, TextColumn

import lilbee.config as cfg
from lilbee import embedder, store
from lilbee.code_chunker import CodeChunk, chunk_code, supported_extensions

log = logging.getLogger(__name__)

# Approximate chars-per-token ratio (kreuzberg uses chars, not tokens)
_CHARS_PER_TOKEN = 4


class ChunkRecord(TypedDict):
    """A single store-ready chunk record matching store._CHUNKS_SCHEMA."""

    source: str
    content_type: str
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    chunk: str
    chunk_index: int
    vector: list[float]


# File extensions routed to the code chunker (tree-sitter)
_CODE_EXTENSIONS = supported_extensions()

# All document extensions handled by kreuzberg
_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".html",
        ".rst",
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".epub",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
        ".csv",
        ".tsv",
    }
)

# Extension → content_type string for metadata
_EXTENSION_MAP: dict[str, str] = {
    **{ext: "text" for ext in (".md", ".txt", ".html", ".rst")},
    ".pdf": "pdf",
    **{ext: "code" for ext in _CODE_EXTENSIONS},
    **{ext: ext.lstrip(".") for ext in (".docx", ".xlsx", ".pptx")},
    ".epub": "epub",
    **{ext: "image" for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")},
    **{ext: "data" for ext in (".csv", ".tsv")},
}


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _relative_name(path: Path) -> str:
    """Get path relative to documents dir as string."""
    return str(path.relative_to(cfg.DOCUMENTS_DIR))


def _discover_files() -> dict[str, Path]:
    """Scan documents/ recursively, return {relative_name: absolute_path}."""
    if not cfg.DOCUMENTS_DIR.exists():
        return {}
    files: dict[str, Path] = {}
    for root, dirs, filenames in os.walk(cfg.DOCUMENTS_DIR, topdown=True):
        dirs[:] = [d for d in dirs if not cfg.is_ignored_dir(d)]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = Path(root) / fname
            if path.suffix.lower() in _EXTENSION_MAP:
                files[_relative_name(path)] = path
    return files


def _classify_file(path: Path) -> str | None:
    """Classify file by extension. Returns content_type or None if unsupported."""
    return _EXTENSION_MAP.get(path.suffix.lower())


def _kreuzberg_config(content_type: str) -> object:
    """Build kreuzberg ExtractionConfig for a given content type."""
    from kreuzberg import ChunkingConfig, ExtractionConfig, PageConfig

    chunking = ChunkingConfig(
        max_chars=cfg.CHUNK_SIZE * _CHARS_PER_TOKEN,
        max_overlap=cfg.CHUNK_OVERLAP * _CHARS_PER_TOKEN,
    )

    if content_type == "pdf":
        return ExtractionConfig(
            chunking=chunking,
            pages=PageConfig(extract_pages=True, insert_page_markers=False),
        )
    return ExtractionConfig(chunking=chunking)


async def _ingest_document(path: Path, source_name: str, content_type: str) -> list[ChunkRecord]:
    """Extract and chunk a document via kreuzberg, embed, return records."""
    from kreuzberg import extract_file

    config = _kreuzberg_config(content_type)
    result = await extract_file(str(path), config=config)

    if not result.chunks:
        return []

    texts = [chunk.content for chunk in result.chunks]
    vectors = await asyncio.to_thread(embedder.embed_batch, texts)

    return [
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            page_start=chunk.metadata.get("first_page") or 0,
            page_end=chunk.metadata.get("last_page") or 0,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=chunk.metadata.get("chunk_index", idx),
            vector=vec,
        )
        for idx, (chunk, text, vec) in enumerate(zip(result.chunks, texts, vectors, strict=True))
    ]


def _ingest_code_sync(path: Path, source_name: str) -> list[ChunkRecord]:
    """Parse code with tree-sitter, chunk, embed, and return store-ready records."""
    code_chunks: list[CodeChunk] = chunk_code(path)
    if not code_chunks:
        return []

    texts = [cc.chunk for cc in code_chunks]
    vectors = embedder.embed_batch(texts)

    return [
        ChunkRecord(
            source=source_name,
            content_type="code",
            page_start=0,
            page_end=0,
            line_start=cc.line_start,
            line_end=cc.line_end,
            chunk=cc.chunk,
            chunk_index=cc.chunk_index,
            vector=vec,
        )
        for cc, vec in zip(code_chunks, vectors, strict=True)
    ]


async def _ingest_file(path: Path, source_name: str, content_type: str) -> int:
    """Ingest a single file. Returns chunk count."""
    records: list[ChunkRecord]
    if content_type == "code":
        records = await asyncio.to_thread(_ingest_code_sync, path, source_name)
    else:
        records = await _ingest_document(path, source_name, content_type)
    return await asyncio.to_thread(store.add_chunks, cast(list[dict], records))


async def sync(force_rebuild: bool = False, quiet: bool = False) -> dict:
    """Sync documents/ with the vector store.

    Returns summary dict with keys: added, updated, removed, unchanged, failed.
    When *quiet* is True, the Rich progress bar is suppressed (for JSON output).
    """
    if force_rebuild:
        store.drop_all()

    cfg.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    disk_files = _discover_files()
    existing_sources = {s["filename"]: s["file_hash"] for s in store.get_sources()}

    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    unchanged = 0
    failed: list[str] = []

    # Find files to remove (in DB but not on disk)
    for name in existing_sources:
        if name not in disk_files:
            store.delete_by_source(name)
            store.delete_source(name)
            removed.append(name)

    # Process files on disk
    files_to_process: list[tuple[str, Path, str]] = []  # (name, path, content_type)

    for name, path in sorted(disk_files.items()):
        content_type = _classify_file(path)
        assert content_type is not None, f"Unsupported file slipped through discovery: {name}"

        current_hash = _file_hash(path)
        old_hash = existing_sources.get(name)

        if old_hash == current_hash:
            unchanged += 1
            continue

        if old_hash is not None:
            # Modified — remove old data
            store.delete_by_source(name)
            store.delete_source(name)
            files_to_process.append((name, path, content_type))
            updated.append(name)
        else:
            files_to_process.append((name, path, content_type))
            added.append(name)

    # Ingest files (with optional progress bar)
    if files_to_process:
        embedder.validate_model()
        await _ingest_batch(files_to_process, added, updated, failed, quiet=quiet)

    return {
        "added": added,
        "updated": updated,
        "removed": removed,
        "unchanged": unchanged,
        "failed": failed,
    }


# Limit concurrent ingestion to avoid overwhelming I/O
_MAX_CONCURRENT = os.cpu_count() or 4


@dataclass
class _IngestResult:
    """Outcome of a single file ingestion attempt."""

    name: str
    path: Path
    chunk_count: int
    error: Exception | None


async def _ingest_batch(
    files_to_process: list[tuple[str, Path, str]],
    added: list[str],
    updated: list[str],
    failed: list[str],
    *,
    quiet: bool = False,
) -> None:
    """Ingest a batch of files, optionally showing a Rich progress bar."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _process_one(name: str, path: Path, content_type: str) -> _IngestResult:
        async with semaphore:
            try:
                chunk_count = await _ingest_file(path, name, content_type)
                return _IngestResult(name, path, chunk_count, error=None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return _IngestResult(name, path, 0, error=exc)

    tasks = [
        asyncio.ensure_future(_process_one(name, path, ct)) for name, path, ct in files_to_process
    ]

    if quiet:
        await _collect_results(tasks, added, updated, failed)
    else:
        await _collect_results_with_progress(tasks, added, updated, failed)


async def _collect_results(
    tasks: list[asyncio.Task[_IngestResult]],
    added: list[str],
    updated: list[str],
    failed: list[str],
) -> None:
    """Collect task results without progress display."""
    for fut in asyncio.as_completed(tasks):
        result = await fut
        _apply_result(result, added, updated, failed)


async def _collect_results_with_progress(
    tasks: list[asyncio.Task[_IngestResult]],
    added: list[str],
    updated: list[str],
    failed: list[str],
) -> None:
    """Collect task results with Rich progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
    ) as progress:
        ptask = progress.add_task("Ingesting documents...", total=len(tasks))
        for fut in asyncio.as_completed(tasks):
            result = await fut
            _apply_result(result, added, updated, failed)
            desc = f"Ingested {result.name}" if result.error is None else f"Failed {result.name}"
            progress.update(ptask, description=desc)
            progress.advance(ptask)


def _apply_result(
    result: _IngestResult,
    added: list[str],
    updated: list[str],
    failed: list[str],
) -> None:
    """Record an ingestion result — update store on success, track failure."""
    if result.error is not None:
        log.exception("Failed to ingest %s", result.name, exc_info=result.error)
        if result.name in added:
            added.remove(result.name)
        if result.name in updated:
            updated.remove(result.name)
        failed.append(result.name)
        return
    store.upsert_source(result.name, _file_hash(result.path), result.chunk_count)
