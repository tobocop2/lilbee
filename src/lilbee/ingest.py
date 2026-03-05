"""Document sync engine — keeps documents/ dir in sync with LanceDB."""

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

import lilbee.config as cfg
from lilbee import embedder, store
from lilbee.chunker import chunk_pages, chunk_text
from lilbee.code_chunker import CodeChunk, chunk_code, supported_extensions

log = logging.getLogger(__name__)

# File extensions routed to the text chunker
_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".html", ".rst"})

# File extensions routed to the code chunker
_CODE_EXTENSIONS = supported_extensions()


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
    supported = _TEXT_EXTENSIONS | _CODE_EXTENSIONS | frozenset({".pdf"})
    for ext in supported:
        for path in cfg.DOCUMENTS_DIR.rglob(f"*{ext}"):
            if path.is_file() and not path.name.startswith("."):
                files[_relative_name(path)] = path
    return files


def _classify_file(path: Path) -> str | None:
    """Classify file by extension. Returns content_type or None if unsupported."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    if ext in _CODE_EXTENSIONS:
        return "code"
    return None


def _ingest_pdf(path: Path, source_name: str) -> list[dict]:
    """Extract text from PDF, chunk, embed, and return store-ready records."""
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    pages = [{"page": chunk["metadata"]["page"] + 1, "text": chunk["text"]} for chunk in md]

    page_chunks = chunk_pages(pages)
    if not page_chunks:
        return []

    texts = [pc.chunk for pc in page_chunks]
    vectors = embedder.embed_batch(texts)

    return [
        {
            "source": source_name,
            "content_type": "pdf",
            "page_start": pc.page_start,
            "page_end": pc.page_end,
            "line_start": 0,
            "line_end": 0,
            "chunk": pc.chunk,
            "chunk_index": pc.chunk_index,
            "vector": vec,
        }
        for pc, vec in zip(page_chunks, vectors, strict=True)
    ]


def _ingest_text(path: Path, source_name: str) -> list[dict]:
    """Read text file, chunk, embed, and return store-ready records."""
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text)
    if not chunks:
        return []

    vectors = embedder.embed_batch(chunks)

    return [
        {
            "source": source_name,
            "content_type": "text",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": chunk,
            "chunk_index": idx,
            "vector": vec,
        }
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]


def _ingest_code(path: Path, source_name: str) -> list[dict]:
    """Parse code with tree-sitter, chunk, embed, and return store-ready records."""
    code_chunks: list[CodeChunk] = chunk_code(path)
    if not code_chunks:
        return []

    texts = [cc.chunk for cc in code_chunks]
    vectors = embedder.embed_batch(texts)

    return [
        {
            "source": source_name,
            "content_type": "code",
            "page_start": 0,
            "page_end": 0,
            "line_start": cc.line_start,
            "line_end": cc.line_end,
            "chunk": cc.chunk,
            "chunk_index": cc.chunk_index,
            "vector": vec,
        }
        for cc, vec in zip(code_chunks, vectors, strict=True)
    ]


_INGEST_DISPATCH = {
    "pdf": _ingest_pdf,
    "text": _ingest_text,
    "code": _ingest_code,
}


def _ingest_file(path: Path, source_name: str, content_type: str) -> int:
    """Ingest a single file. Returns chunk count."""
    ingest_fn = _INGEST_DISPATCH[content_type]
    records = ingest_fn(path, source_name)
    return store.add_chunks(records)


def sync(force_rebuild: bool = False) -> dict:
    """Sync documents/ with the vector store.

    Returns summary dict with keys: added, updated, removed, unchanged, failed.
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

    # Ingest with progress bar
    if files_to_process:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Ingesting documents...", total=len(files_to_process))

            def _process_file(name: str, path: Path, content_type: str) -> tuple[str, int]:
                chunk_count = _ingest_file(path, name, content_type)
                return name, chunk_count

            workers = min(os.cpu_count() or 4, len(files_to_process))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_file, name, path, ct): (name, path)
                    for name, path, ct in files_to_process
                }
                for future in as_completed(futures):
                    name, path = futures[future]
                    try:
                        _, chunk_count = future.result()
                        store.upsert_source(name, _file_hash(path), chunk_count)
                    except Exception:
                        log.exception("Failed to ingest %s", name)
                        if name in added:
                            added.remove(name)
                        if name in updated:
                            updated.remove(name)
                        failed.append(name)
                        progress.update(task, description=f"Failed {name}")
                        progress.advance(task)
                        continue
                    progress.update(task, description=f"Ingested {name}")
                    progress.advance(task)

    return {
        "added": added,
        "updated": updated,
        "removed": removed,
        "unchanged": unchanged,
        "failed": failed,
    }
