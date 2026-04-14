"""Document sync engine — keeps documents/ dir in sync with LanceDB."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

if TYPE_CHECKING:
    from kreuzberg import ExtractionConfig, ExtractionResult

from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from lilbee.chunk import CHARS_PER_TOKEN, chunk_text
from lilbee.code_chunker import CodeChunk, chunk_code, is_code_file
from lilbee.config import cfg
from lilbee.platform import is_ignored_dir
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
from lilbee.security import validate_path_within
from lilbee.services import get_services
from lilbee.vision import extract_pdf_vision

log = logging.getLogger(__name__)


class FileToProcess(NamedTuple):
    """A file queued for ingestion with its metadata."""

    name: str
    path: Path
    content_type: str
    file_hash: str
    needs_cleanup: bool


# Minimum total chars for extracted text to be considered meaningful.
# 50 chars ≈ 12 words — if a PDF yields less, it's almost certainly a scanned
# document with no embedded text layer. Text PDFs with even just a title page
# easily exceed this threshold; blank/scan-only PDFs yield 0 chars.
_MIN_MEANINGFUL_CHARS = 50


def _strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by ``---`` at the start."""
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text


def _has_meaningful_text(result: Any) -> bool:
    """Check if extraction produced meaningful text."""
    chunks = getattr(result, "chunks", None)
    if chunks:
        total = sum(len(c.content.strip()) for c in chunks)
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
        return (
            f"SyncResult(added={len(self.added)}, updated={len(self.updated)}, "
            f"removed={len(self.removed)}, unchanged={self.unchanged}, "
            f"failed={len(self.failed)})"
        )

    def __rich__(self) -> str:
        return self.__str__()


@dataclass
class _IngestResult:
    """Outcome of a single file ingestion attempt."""

    name: str
    path: Path
    chunk_count: int
    error: Exception | None
    file_hash: str = ""


# Extension → content_type string for document formats handled by kreuzberg
_DOCUMENT_EXTENSION_MAP: dict[str, str] = {
    **{ext: "text" for ext in (".md", ".txt", ".html", ".rst", ".yaml", ".yml")},
    ".pdf": "pdf",
    **{ext: ext.lstrip(".") for ext in (".docx", ".xlsx", ".pptx")},
    ".epub": "epub",
    **{ext: "image" for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")},
    **{ext: "data" for ext in (".csv", ".tsv")},
    ".xml": "xml",
    **{ext: "json" for ext in (".json", ".jsonl")},
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


def classify_file(path: Path) -> str | None:
    """Classify file by extension. Returns content_type or None if unsupported."""
    doc_type = _DOCUMENT_EXTENSION_MAP.get(path.suffix.lower())
    if doc_type is not None:
        return doc_type
    if is_code_file(path):
        return "code"
    return None


def extraction_config(content_type: str) -> ExtractionConfig:
    """Build ExtractionConfig for a given content type."""
    from kreuzberg import ChunkingConfig, ExtractionConfig, PageConfig

    chunking = ChunkingConfig(
        max_chars=cfg.chunk_size * CHARS_PER_TOKEN,
        max_overlap=cfg.chunk_overlap * CHARS_PER_TOKEN,
    )

    if content_type == "pdf":
        return ExtractionConfig(
            chunking=chunking,
            pages=PageConfig(extract_pages=True, insert_page_markers=False),
        )
    return ExtractionConfig(chunking=chunking, output_format="markdown")


def ocr_extraction_config() -> ExtractionConfig:
    """Build ExtractionConfig with Tesseract OCR enabled for scanned PDFs."""
    from kreuzberg import ChunkingConfig, ExtractionConfig, OcrConfig, PageConfig

    chunking = ChunkingConfig(
        max_chars=cfg.chunk_size * CHARS_PER_TOKEN,
        max_overlap=cfg.chunk_overlap * CHARS_PER_TOKEN,
    )
    return ExtractionConfig(
        chunking=chunking,
        pages=PageConfig(extract_pages=True, insert_page_markers=False),
        ocr=OcrConfig(backend="tesseract"),
    )


@contextlib.contextmanager
def suppress_fd_stderr() -> Generator[None, None, None]:
    """Suppress stderr at the file-descriptor level.
    Catches subprocess output (e.g. Tesseract's "Detected N diacritics")
    that ``contextlib.redirect_stderr`` cannot intercept.
    """
    old_stderr = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
    finally:
        os.close(old_stderr)


async def _try_tesseract_ocr(
    path: Path, source_name: str, fallback: ExtractionResult
) -> ExtractionResult:
    """Attempt Tesseract OCR on a scanned PDF. Returns the OCR result or *fallback* on failure."""
    try:
        from kreuzberg import extract_file

        log.info("PDF text extraction empty, trying Tesseract OCR: %s", source_name)
        with suppress_fd_stderr():
            return await extract_file(str(path), config=ocr_extraction_config())
    except Exception:
        log.debug("Tesseract OCR unavailable or failed for %s, skipping", source_name)
        return fallback


def _should_run_ocr() -> bool:
    """Decide whether to attempt vision-based OCR on scanned PDFs.

    Uses ``cfg.enable_ocr``: True = force on, False = force off,
    None = auto-detect from chat model capabilities.
    """
    if cfg.enable_ocr is True:
        return True
    if cfg.enable_ocr is False:
        return False
    from lilbee.model_manager import is_vision_capable

    return is_vision_capable(cfg.chat_model)


async def _vision_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    on_progress: DetailedProgressCallback = noop_callback,
    *,
    quiet: bool = False,
) -> list[ChunkRecord]:
    """OCR a scanned PDF via vision model, chunk, and embed.

    Uses ``cfg.chat_model`` as the vision model. Wraps the OCR call in
    a try/except so that forced OCR on an incompatible model degrades
    gracefully (logs a warning and returns empty).
    """
    try:
        page_texts = await asyncio.to_thread(
            extract_pdf_vision,
            path,
            cfg.chat_model,
            quiet=quiet,
            timeout=cfg.ocr_timeout,
            on_progress=on_progress,
        )
    except Exception:
        log.warning(
            "Vision OCR failed for %s. The chat model (%s) may not support vision input.",
            source_name,
            cfg.chat_model,
            exc_info=True,
        )
        return []
    if not page_texts:
        return []

    all_chunks = [(page_num, chunk) for page_num, text in page_texts for chunk in chunk_text(text)]
    if not all_chunks:
        return []

    texts = [c for _, c in all_chunks]

    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
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


async def _handle_scanned_pdf_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    result: ExtractionResult,
    *,
    quiet: bool,
    on_progress: DetailedProgressCallback,
) -> list[ChunkRecord] | ExtractionResult:
    """Handle scanned PDF fallback chain: Tesseract OCR then vision model.

    Returns chunk records if a fallback produced final results, or an
    updated ExtractionResult when Tesseract OCR succeeded (so the
    caller can proceed with normal chunking/embedding).

    Vision OCR is attempted when ``_should_run_ocr()`` is True. When
    force-OCR is on, Tesseract is skipped and we go straight to vision.
    """
    use_ocr = _should_run_ocr()

    if cfg.enable_ocr is not True:
        result = await _try_tesseract_ocr(path, source_name, result)

    if not _has_meaningful_text(result):
        if not use_ocr:
            log.warning(
                "Skipped %s: text extraction produced no usable text. "
                "For better results on scanned PDFs, switch to a vision-capable "
                "chat model or set LILBEE_ENABLE_OCR=true.",
                source_name,
            )
            return []
        log.info("PDF text extraction empty, falling back to vision OCR: %s", source_name)
        return await _vision_fallback(path, source_name, content_type, on_progress, quiet=quiet)

    log.info(
        "Scanned PDF detected — extracted with Tesseract OCR: %s. "
        "For structured markdown output (tables, headings), use a vision-capable chat model.",
        source_name,
    )
    return result


async def ingest_document(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Extract and chunk a document, embed, return records.

    Vision OCR is controlled by ``cfg.enable_ocr`` (see ``_should_run_ocr``).
    """
    from kreuzberg import extract_file

    config = extraction_config(content_type)
    result = await extract_file(str(path), config=config)

    if content_type == "pdf" and not _has_meaningful_text(result):
        fallback = await _handle_scanned_pdf_fallback(
            path,
            source_name,
            content_type,
            result,
            quiet=quiet,
            on_progress=on_progress,
        )
        if isinstance(fallback, list):
            return fallback
        # Tesseract OCR succeeded — use the updated ExtractionResult
        result = fallback

    if not result.chunks:
        return []

    texts = [chunk.content for chunk in result.chunks]
    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
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
    embedder = get_services().embedder
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


async def ingest_markdown(
    path: Path,
    source_name: str,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Chunk a markdown file with heading context prepended to each chunk.
    Each chunk gets the heading hierarchy path (e.g. "# Setup > ## Install")
    prepended for better retrieval context.
    """
    raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
    if not raw_text.strip():
        return []

    # Strip YAML frontmatter so metadata isn't indexed as document content.
    body = _strip_yaml_frontmatter(raw_text)
    if not body.strip():
        return []

    texts = chunk_text(body, mime_type="text/markdown", heading_context=True)
    if not texts:
        return []

    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type="text",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk=t,
            chunk_index=idx,
            vector=vec,
        )
        for idx, (t, vec) in enumerate(zip(texts, vectors, strict=True))
    ]


async def _rebuild_concept_clusters() -> None:
    """Re-run Leiden clustering after sync. No-op if disabled."""
    if not cfg.concept_graph:
        return
    from lilbee.concepts import concepts_available

    if not concepts_available():
        return
    try:
        cg = get_services().concepts
        if not cg.get_graph():
            return
        await asyncio.to_thread(cg.rebuild_clusters)
    except Exception:
        log.warning("Concept cluster rebuild failed", exc_info=True)


async def _index_concepts(records: list[ChunkRecord], source_name: str) -> None:
    """Extract and index concepts for ingested chunks. No-op if disabled."""
    if not cfg.concept_graph or not records:
        return
    from lilbee.concepts import concepts_available

    if not concepts_available():
        return
    try:
        cg = get_services().concepts
        texts = [r["chunk"] for r in records]
        concept_lists = await asyncio.to_thread(cg.extract_concepts_batch, texts)
        chunk_ids = [(source_name, r["chunk_index"]) for r in records]
        await asyncio.to_thread(cg.build_from_chunks, chunk_ids, concept_lists)
    except Exception:
        log.warning("Concept indexing failed for %s", source_name, exc_info=True)


async def _ingest_file(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> int:
    """Ingest a single file. Returns chunk count."""
    records: list[ChunkRecord]
    if content_type == "code":
        records = await asyncio.to_thread(ingest_code_sync, path, source_name, on_progress)
    elif path.suffix.lower() == ".md":
        records = await ingest_markdown(path, source_name, on_progress)
    else:
        records = await ingest_document(
            path,
            source_name,
            content_type,
            quiet=quiet,
            on_progress=on_progress,
        )

    store = get_services().store
    chunk_count = await asyncio.to_thread(store.add_chunks, cast(list[dict], records))
    await _index_concepts(records, source_name)
    return chunk_count


async def sync(
    force_rebuild: bool = False,
    quiet: bool = False,
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> SyncResult:
    """Sync documents/ with the vector store.
    Returns summary dict with keys: added, updated, removed, unchanged, failed.
    When *quiet* is True, the Rich progress bar is suppressed (for JSON output).
    When *cancel* is set, processing stops between files without data loss.
    """
    _store = get_services().store

    if force_rebuild:
        _store.drop_all()

    cfg.documents_dir.mkdir(parents=True, exist_ok=True)

    disk_files = discover_files()
    existing_sources = {s["filename"]: s["file_hash"] for s in _store.get_sources()}

    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    unchanged = 0
    failed: list[str] = []

    # Find files to remove (in DB but not on disk)
    for name in existing_sources:
        if name not in disk_files:
            _store.delete_by_source(name)
            _store.delete_source(name)
            removed.append(name)

    files_to_process: list[FileToProcess] = []

    for name, path in sorted(disk_files.items()):
        if cancel and cancel.is_set():
            break

        content_type = classify_file(path)
        if content_type is None:
            raise ValueError(f"Unsupported file slipped through discovery: {name}")

        current_hash = file_hash(path)
        old_hash = existing_sources.get(name)

        if old_hash == current_hash:
            unchanged += 1
            continue

        if old_hash is not None:
            # Modified — defer old chunk deletion to ingest_batch so
            # delete + re-ingest are atomic per file (no data loss on cancel).
            files_to_process.append(
                FileToProcess(name, path, content_type, current_hash, needs_cleanup=True)
            )
            updated.append(name)
        else:
            files_to_process.append(
                FileToProcess(name, path, content_type, current_hash, needs_cleanup=False)
            )
            added.append(name)

    # Ingest files (with optional progress bar)
    if files_to_process:
        get_services().embedder.validate_model()
        await ingest_batch(
            files_to_process,
            added,
            updated,
            failed,
            quiet=quiet,
            on_progress=on_progress,
            cancel=cancel,
        )

    if files_to_process or removed:
        _store.ensure_fts_index()
        await _rebuild_concept_clusters()

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
        ),
    )
    return result


# Limit concurrent ingestion to avoid overwhelming I/O
_MAX_CONCURRENT = os.cpu_count() or 4

# Concurrent.futures raises this exact RuntimeError message when submitting to
# a shutdown executor (Python 3.11+). There is no dedicated exception class to
# catch, so callers have to string-match the message.
_EXECUTOR_SHUTDOWN_MSG = "cannot schedule new futures after shutdown"


def _is_executor_shutdown(exc: BaseException) -> bool:
    """True if ``exc`` is the concurrent.futures shutdown-race RuntimeError."""
    return isinstance(exc, RuntimeError) and _EXECUTOR_SHUTDOWN_MSG in str(exc)


async def ingest_batch(
    files_to_process: list[FileToProcess],
    added: list[str],
    updated: list[str],
    failed: list[str],
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: threading.Event | None = None,
) -> None:
    """Ingest a batch of files, optionally showing a Rich progress bar.
    When *needs_cleanup* is True, old chunks are deleted immediately before
    ingesting new ones so the two operations are atomic per file.
    When *cancel* is set, pending files raise CancelledError before starting.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    total_files = len(files_to_process)

    async def _process_one(
        name: str,
        path: Path,
        content_type: str,
        fhash: str,
        needs_cleanup: bool,
        file_index: int,
    ) -> _IngestResult:
        async with semaphore:
            if cancel and cancel.is_set():
                raise asyncio.CancelledError

            on_progress(
                EventType.FILE_START,
                FileStartEvent(file=name, total_files=total_files, current_file=file_index),
            )
            try:
                if needs_cleanup:
                    get_services().store.delete_by_source(name)
                chunk_count = await _ingest_file(
                    path,
                    name,
                    content_type,
                    quiet=quiet,
                    on_progress=on_progress,
                )
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="ok", chunks=chunk_count),
                )
                return _IngestResult(name, path, chunk_count, error=None, file_hash=fhash)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # During shutdown, worker pools raise RuntimeError from
                # submit(). Prefer to treat these as cancellation rather than
                # as ingest failures. Detect via the cancel flag (source of
                # truth) or the executor's well-known shutdown message as a
                # fallback when cancel was set after the submit race.
                if (cancel and cancel.is_set()) or _is_executor_shutdown(exc):
                    raise asyncio.CancelledError from exc
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file=name, status="error", chunks=0),
                )
                return _IngestResult(name, path, 0, error=exc)

    if quiet:
        tasks = [
            asyncio.ensure_future(_process_one(name, path, ct, fh, cleanup, idx))
            for idx, (name, path, ct, fh, cleanup) in enumerate(files_to_process, 1)
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
                    asyncio.ensure_future(_process_one(name, path, ct, fh, cleanup, idx))
                    for idx, (name, path, ct, fh, cleanup) in enumerate(files_to_process, 1)
                ]
                await _collect_results(
                    tasks,
                    added,
                    updated,
                    failed,
                    on_progress=on_progress,
                    progress=progress,
                    ptask=ptask,
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
    progress: Progress | None = None,
    ptask: Any = None,
) -> None:
    """Collect task results, optionally updating a Rich progress bar."""
    for completed_count, fut in enumerate(asyncio.as_completed(tasks), 1):
        result = await fut
        _apply_result(result, added, updated, failed)
        if progress is not None and ptask is not None:
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
            ),
        )


def _discard_from_list(lst: list[str], value: str) -> None:
    """Remove *value* from *lst* if present."""
    with contextlib.suppress(ValueError):
        lst.remove(value)


def _apply_result(
    result: _IngestResult,
    added: list[str],
    updated: list[str],
    failed: list[str],
) -> None:
    """Record an ingestion result — update store on success, track failure."""
    if result.error is not None:
        log.exception("Failed to ingest %s", result.name, exc_info=result.error)
        _discard_from_list(added, result.name)
        _discard_from_list(updated, result.name)
        failed.append(result.name)
        return
    if result.chunk_count == 0:
        # No chunks produced (e.g. scanned PDF without vision model).
        # Don't record as a source so it gets retried on next sync.
        _discard_from_list(added, result.name)
        _discard_from_list(updated, result.name)
        return

    fhash = result.file_hash or file_hash(result.path)
    get_services().store.upsert_source(result.name, fhash, result.chunk_count)
