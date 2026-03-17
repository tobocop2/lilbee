"""Document sync engine — keeps documents/ dir in sync with LanceDB."""

import asyncio
import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lilbee import embedder, store
from lilbee.chunker import chunk_text
from lilbee.code_chunker import CodeChunk, chunk_code, supported_extensions
from lilbee.config import cfg
from lilbee.platform import is_ignored_dir
from lilbee.preprocessors import preprocess_csv, preprocess_json, preprocess_xml
from lilbee.progress import (
    BatchProgressEvent,
    DetailedProgressCallback,
    EventType,
    FileDoneEvent,
    FileStartEvent,
    SyncDoneEvent,
    noop_callback,
    shared_progress,
)
from lilbee.vision import extract_pdf_vision

log = logging.getLogger(__name__)

# Minimum total chars for kreuzberg text to be considered meaningful.
# 50 chars ≈ 12 words — if a PDF yields less, it's almost certainly a scanned
# document with no embedded text layer. Text PDFs with even just a title page
# easily exceed this threshold; blank/scan-only PDFs yield 0 chars.
_MIN_MEANINGFUL_CHARS = 50

# Approximate chars-per-token ratio (kreuzberg uses chars, not tokens)
_CHARS_PER_TOKEN = 4


def _has_meaningful_text(result: Any) -> bool:
    """Check if kreuzberg extraction produced meaningful text."""
    if hasattr(result, "chunks") and result.chunks:
        total = sum(len(c.content.strip()) for c in result.chunks)
        return total > _MIN_MEANINGFUL_CHARS
    return False


class ChunkRecord(TypedDict):
    """A single store-ready chunk record matching store.CHUNKS_SCHEMA."""

    source: str
    content_type: str
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    chunk: str
    chunk_index: int
    vector: list[float]


class SyncResult(BaseModel):
    """Summary of a sync operation."""

    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    unchanged: int = 0
    failed: list[str] = []

    def __str__(self) -> str:
        lines = [
            f"Added: {len(self.added)}",
            f"Updated: {len(self.updated)}",
            f"Removed: {len(self.removed)}",
            f"Unchanged: {self.unchanged}",
            f"Failed: {len(self.failed)}",
        ]
        for f in self.failed:
            lines.append(f"  [red]{f}[/red]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def __rich__(self) -> str:
        return self.__str__()


@dataclass
class _IngestResult:
    """Outcome of a single file ingestion attempt."""

    name: str
    path: Path
    chunk_count: int
    error: Exception | None


# File extensions routed to the code chunker (tree-sitter)
_CODE_EXTENSIONS = supported_extensions()

# All document extensions handled by kreuzberg or structured preprocessors
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
        ".xml",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
    }
)

# Extension → content_type string for metadata
_EXTENSION_MAP: dict[str, str] = {
    **{ext: "text" for ext in (".md", ".txt", ".html", ".rst", ".yaml", ".yml")},
    ".pdf": "pdf",
    **{ext: "code" for ext in _CODE_EXTENSIONS if ext not in _DOCUMENT_EXTENSIONS},
    **{ext: ext.lstrip(".") for ext in (".docx", ".xlsx", ".pptx")},
    ".epub": "epub",
    **{ext: "image" for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")},
    **{ext: "data" for ext in (".csv", ".tsv")},
    ".xml": "xml",
    **{ext: "json" for ext in (".json", ".jsonl")},
}


# Preprocessors for structured formats: content_type → callable(Path) → str
_PREPROCESSORS: dict[str, Callable[[Path], str]] = {
    "xml": preprocess_xml,
    "json": preprocess_json,
    "data": preprocess_csv,
}


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


def discover_files() -> dict[str, Path]:
    """Scan documents/ recursively, return {relative_name: absolute_path}."""
    if not cfg.documents_dir.exists():
        return {}
    files: dict[str, Path] = {}
    for root, dirs, filenames in os.walk(cfg.documents_dir, topdown=True):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d, cfg.ignore_dirs)]
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = Path(root) / fname
            if path.suffix.lower() in _EXTENSION_MAP:
                files[_relative_name(path)] = path
    return files


def classify_file(path: Path) -> str | None:
    """Classify file by extension. Returns content_type or None if unsupported."""
    return _EXTENSION_MAP.get(path.suffix.lower())


def kreuzberg_config(content_type: str) -> object:
    """Build kreuzberg ExtractionConfig for a given content type."""
    from kreuzberg import ChunkingConfig, ExtractionConfig, PageConfig

    chunking = ChunkingConfig(
        max_chars=cfg.chunk_size * _CHARS_PER_TOKEN,
        max_overlap=cfg.chunk_overlap * _CHARS_PER_TOKEN,
    )

    if content_type == "pdf":
        return ExtractionConfig(
            chunking=chunking,
            pages=PageConfig(extract_pages=True, insert_page_markers=False),
        )
    return ExtractionConfig(chunking=chunking, output_format="markdown")


def kreuzberg_ocr_config() -> object:
    """Build kreuzberg ExtractionConfig with Tesseract OCR enabled for scanned PDFs."""
    from kreuzberg import ChunkingConfig, ExtractionConfig, OcrConfig, PageConfig

    chunking = ChunkingConfig(
        max_chars=cfg.chunk_size * _CHARS_PER_TOKEN,
        max_overlap=cfg.chunk_overlap * _CHARS_PER_TOKEN,
    )
    return ExtractionConfig(
        chunking=chunking,
        pages=PageConfig(extract_pages=True, insert_page_markers=False),
        ocr=OcrConfig(backend="tesseract"),
    )


async def _try_tesseract_ocr(path: Path, source_name: str, fallback: object) -> object:
    """Attempt Tesseract OCR on a scanned PDF. Returns the OCR result or *fallback* on failure."""
    try:
        from kreuzberg import extract_file

        log.info("PDF text extraction empty, trying Tesseract OCR: %s", source_name)
        # Suppress Tesseract's "Detected N diacritics" stderr noise at the fd level
        # (contextlib.redirect_stderr only catches Python's sys.stderr, not subprocess output)
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            return await extract_file(str(path), config=kreuzberg_ocr_config())
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
    except Exception:
        log.debug("Tesseract OCR unavailable or failed for %s, skipping", source_name)
        return fallback


async def _vision_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    on_progress: DetailedProgressCallback = noop_callback,
    *,
    quiet: bool = False,
) -> list[ChunkRecord]:
    """OCR a scanned PDF via vision model, chunk, and embed."""

    page_texts = await asyncio.to_thread(
        extract_pdf_vision,
        path,
        cfg.vision_model,
        quiet=quiet,
        timeout=cfg.vision_timeout,
        on_progress=on_progress,
    )
    if not page_texts:
        return []

    all_chunks = [(page_num, chunk) for page_num, text in page_texts for chunk in chunk_text(text)]
    if not all_chunks:
        return []

    texts = [c for _, c in all_chunks]
    vectors = await asyncio.to_thread(
        embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            page_start=page_num,
            page_end=page_num,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=i,
            vector=vec,
        )
        for i, ((page_num, text), vec) in enumerate(zip(all_chunks, vectors, strict=True))
    ]


async def ingest_document(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    force_vision: bool = False,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Extract and chunk a document via kreuzberg, embed, return records.

    When *force_vision* is True (CLI ``--vision``) or a vision model is
    configured, Tesseract OCR is skipped and we go straight to the vision
    model for scanned PDFs.
    """
    from kreuzberg import extract_file

    use_vision = force_vision or bool(cfg.vision_model)

    config = kreuzberg_config(content_type)
    result = await extract_file(str(path), config=config)

    # Scanned PDF fallback chain: Tesseract OCR → vision model
    if content_type == "pdf" and not _has_meaningful_text(result):
        # When vision is explicitly enabled, skip Tesseract and go straight to vision
        if not use_vision:
            result = await _try_tesseract_ocr(path, source_name, result)

        if not _has_meaningful_text(result):
            if not cfg.vision_model:
                log.warning(
                    "Skipped %s: Tesseract OCR produced no usable text. "
                    "For better results on complex scans, set a vision model "
                    "with /vision or LILBEE_VISION_MODEL.",
                    source_name,
                )
                return []
            log.info("PDF text extraction empty, falling back to vision OCR: %s", source_name)
            return await _vision_fallback(path, source_name, content_type, on_progress, quiet=quiet)

        log.info(
            "Scanned PDF detected — extracted with Tesseract OCR: %s. "
            "For structured markdown output (tables, headings), re-add with --vision.",
            source_name,
        )

    if not result.chunks:
        return []

    texts = [chunk.content for chunk in result.chunks]
    vectors = await asyncio.to_thread(
        embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )

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


def ingest_code_sync(
    path: Path,
    source_name: str,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Parse code with tree-sitter, chunk, embed, and return store-ready records."""
    code_chunks: list[CodeChunk] = chunk_code(path)
    if not code_chunks:
        return []

    texts = [cc.chunk for cc in code_chunks]
    vectors = embedder.embed_batch(texts, source=source_name, on_progress=on_progress)

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


async def ingest_structured(
    path: Path,
    source_name: str,
    content_type: str,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Preprocess a structured file, chunk, embed, and return store-ready records."""
    preprocessor = _PREPROCESSORS[content_type]
    text = await asyncio.to_thread(preprocessor, path)
    if not text.strip():
        return []
    texts = chunk_text(text)
    if not texts:
        return []
    vectors = await asyncio.to_thread(
        embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=idx,
            vector=vec,
        )
        for idx, (text, vec) in enumerate(zip(texts, vectors, strict=True))
    ]


async def _ingest_file(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    force_vision: bool = False,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> int:
    """Ingest a single file. Returns chunk count."""
    records: list[ChunkRecord]
    if content_type == "code":
        records = await asyncio.to_thread(ingest_code_sync, path, source_name, on_progress)
    elif content_type in _PREPROCESSORS:
        records = await ingest_structured(path, source_name, content_type, on_progress)
    else:
        records = await ingest_document(
            path,
            source_name,
            content_type,
            force_vision=force_vision,
            quiet=quiet,
            on_progress=on_progress,
        )
    return await asyncio.to_thread(store.add_chunks, cast(list[dict], records))


async def sync(
    force_rebuild: bool = False,
    quiet: bool = False,
    *,
    force_vision: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> SyncResult:
    """Sync documents/ with the vector store.

    Returns summary dict with keys: added, updated, removed, unchanged, failed.
    When *quiet* is True, the Rich progress bar is suppressed (for JSON output).
    """
    if force_rebuild:
        store.drop_all()

    cfg.documents_dir.mkdir(parents=True, exist_ok=True)

    disk_files = discover_files()
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
        content_type = classify_file(path)
        assert content_type is not None, f"Unsupported file slipped through discovery: {name}"

        current_hash = file_hash(path)
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
        await ingest_batch(
            files_to_process,
            added,
            updated,
            failed,
            quiet=quiet,
            force_vision=force_vision,
            on_progress=on_progress,
        )

    if files_to_process or removed:
        store.ensure_fts_index()

    result = SyncResult(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        failed=failed,
    )
    on_progress(
        EventType.DONE,
        SyncDoneEvent(
            added=len(result.added),
            updated=len(result.updated),
            removed=len(result.removed),
            failed=len(result.failed),
        ).model_dump(),
    )
    return result


# Limit concurrent ingestion to avoid overwhelming I/O
_MAX_CONCURRENT = os.cpu_count() or 4


async def ingest_batch(
    files_to_process: list[tuple[str, Path, str]],
    added: list[str],
    updated: list[str],
    failed: list[str],
    *,
    quiet: bool = False,
    force_vision: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> None:
    """Ingest a batch of files, optionally showing a Rich progress bar."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    total_files = len(files_to_process)

    async def _process_one(
        name: str, path: Path, content_type: str, file_index: int
    ) -> _IngestResult:
        async with semaphore:
            on_progress(
                EventType.FILE_START,
                FileStartEvent(
                    file=name, total_files=total_files, current_file=file_index
                ).model_dump(),
            )
            try:
                chunk_count = await _ingest_file(
                    path,
                    name,
                    content_type,
                    force_vision=force_vision,
                    quiet=quiet,
                    on_progress=on_progress,
                )
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="ok", chunks=chunk_count).model_dump(),
                )
                return _IngestResult(name, path, chunk_count, error=None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(
                    exc, RuntimeError
                ) and "cannot schedule new futures after shutdown" in str(exc):
                    raise asyncio.CancelledError from exc
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="error", chunks=0).model_dump(),
                )
                return _IngestResult(name, path, 0, error=exc)

    if quiet:
        tasks = [
            asyncio.ensure_future(_process_one(name, path, ct, idx))
            for idx, (name, path, ct) in enumerate(files_to_process, 1)
        ]
        await _collect_results(tasks, added, updated, failed, on_progress=on_progress)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            ptask = progress.add_task("Ingesting documents...", total=total_files)
            token = shared_progress.set((progress, ptask))
            try:
                tasks = [
                    asyncio.ensure_future(_process_one(name, path, ct, idx))
                    for idx, (name, path, ct) in enumerate(files_to_process, 1)
                ]
                await _collect_results_with_progress(
                    progress, ptask, tasks, added, updated, failed, on_progress=on_progress
                )
            finally:
                shared_progress.reset(token)


async def _collect_results(
    tasks: list[asyncio.Task[_IngestResult]],
    added: list[str],
    updated: list[str],
    failed: list[str],
    *,
    on_progress: DetailedProgressCallback = noop_callback,
) -> None:
    """Collect task results without progress display."""
    for completed_count, fut in enumerate(asyncio.as_completed(tasks), 1):
        result = await fut
        _apply_result(result, added, updated, failed)
        progress_status = "failed" if result.error is not None else "ingested"
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(
                file=result.name,
                status=progress_status,
                current=completed_count,
                total=len(tasks),
            ).model_dump(),
        )


async def _collect_results_with_progress(
    progress: Progress,
    ptask: Any,
    tasks: list[asyncio.Task[_IngestResult]],
    added: list[str],
    updated: list[str],
    failed: list[str],
    *,
    on_progress: DetailedProgressCallback = noop_callback,
) -> None:
    """Collect task results, updating an existing Rich progress bar."""
    for completed_count, fut in enumerate(asyncio.as_completed(tasks), 1):
        result = await fut
        _apply_result(result, added, updated, failed)
        desc = f"Ingested {result.name}" if result.error is None else f"Failed {result.name}"
        progress.update(ptask, description=desc)
        progress.advance(ptask)
        progress_status = "failed" if result.error is not None else "ingested"
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(
                file=result.name,
                status=progress_status,
                current=completed_count,
                total=len(tasks),
            ).model_dump(),
        )


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
    if result.chunk_count == 0:
        # No chunks produced (e.g. scanned PDF without vision model).
        # Don't record as a source so it gets retried on next sync.
        if result.name in added:
            added.remove(result.name)
        if result.name in updated:
            updated.remove(result.name)
        return
    store.upsert_source(result.name, file_hash(result.path), result.chunk_count)
