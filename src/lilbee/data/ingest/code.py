"""Code-file ingestion via tree-sitter chunking."""

from __future__ import annotations

from pathlib import Path

from lilbee.data.code_chunker import CodeChunk, chunk_code
from lilbee.core.services import get_services
from lilbee.data.ingest.types import ChunkRecord
from lilbee.runtime.progress import DetailedProgressCallback, noop_callback
from lilbee.data.store import CHUNK_TYPE_RAW


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
            chunk_type=CHUNK_TYPE_RAW,
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
