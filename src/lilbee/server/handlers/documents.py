"""Document listing, deletion, and source-content handlers."""

from __future__ import annotations

import asyncio
import mimetypes

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.server.models import (
    DocumentInfo,
    DocumentListResponse,
    DocumentRemoveResponse,
    SourceContentResponse,
)

# Windows mimetypes reads from the registry, which may not define ``.md``
# as ``text/markdown``. Pin the mapping at import time; ``add_type`` is
# idempotent so repeated imports are safe.
mimetypes.add_type("text/markdown", ".md")


# Types that can carry script even within an "inline-rendered" category.
# Keep the deny narrow and explicit. Broadening this set is a security-relevant
# change: file an issue with the ``security`` label before adding entries.
_RAW_INLINE_RENDER_DENY: frozenset[str] = frozenset(
    {
        "text/html",
        "text/javascript",
        "application/javascript",
        "application/xhtml+xml",
        "text/css",
        "image/svg+xml",
    }
)


def _is_safe_for_inline_render(content_type: str) -> bool:
    """Whether ``raw=1`` may serve this Content-Type as-is.

    Trusted categories (``text/*``, ``image/*``, ``application/pdf``) pass
    through, with named exceptions for types that embed executable script.
    Everything else degrades to ``application/octet-stream`` so an attacker-
    renamed file (e.g. ``evil.html``) cannot trick a browser into rendering
    it inline within the plugin origin.
    """
    if content_type in _RAW_INLINE_RENDER_DENY:
        return False
    if content_type == "application/pdf":
        return True
    return content_type.startswith("text/") or content_type.startswith("image/")


def _imported_source_markdown(source: str) -> str | None:
    """Page texts joined in page order; ``None`` when the source has none."""
    rows = get_services().store.get_page_texts(source)
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: row["page"])
    return "\n\n".join(row["text"] for row in ordered)


async def delete_documents(
    names: list[str], *, delete_files: bool = False
) -> DocumentRemoveResponse:
    """Remove documents from the knowledge base by source name."""
    result = get_services().store.remove_documents(names, delete_files=delete_files)
    return DocumentRemoveResponse(removed=result.removed, not_found=result.not_found)


async def list_documents(
    search: str = "",
    limit: int = 50,
    offset: int = 0,
) -> DocumentListResponse:
    """Return indexed documents with metadata, paginated and filterable.

    Pagination and the filename filter are pushed into LanceDB via
    ``Store.get_sources(search=..., limit=..., offset=...)`` and the
    total comes from ``Store.count_sources(search=...)`` so neither
    call materializes the full SOURCES table per request.
    """
    store = get_services().store
    search_term = search or None
    page = store.get_sources(search=search_term, limit=limit, offset=offset)
    total = store.count_sources(search=search_term)
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                filename=s["filename"],
                chunk_count=s.get("chunk_count", 0),
                ingested_at=s.get("ingested_at", ""),
            )
            for s in page
        ],
        total=total,
        limit=limit,
        offset=offset,
        has_more=len(page) > 0 and (offset + len(page)) < total,
    )


async def get_source_content(
    source: str, raw: bool = False
) -> SourceContentResponse | tuple[bytes, str]:
    """Return a stored source file: JSON with markdown text for text types, or
    ``(bytes, content_type)`` when *raw* is True. Binary types return empty
    markdown so clients know to re-request with ``raw=1``.

    Reads the file off the event loop so a large source doesn't stall it.
    """
    return await asyncio.to_thread(_get_source_content_sync, source, raw)


def _get_source_content_sync(source: str, raw: bool) -> SourceContentResponse | tuple[bytes, str]:
    """Blocking body of :func:`get_source_content`: path validation + file read."""
    from lilbee.wiki.index import parse_title

    if not source or not source.strip():
        raise ValueError("source must not be empty")
    documents_dir = cfg.documents_dir
    resolved = validate_path_within(documents_dir / source, documents_dir)
    if not resolved.is_file():
        # Imported sources have no file on disk; their text lives in the page-text store.
        markdown = _imported_source_markdown(source)
        if markdown is None:
            raise FileNotFoundError(source)
        if raw:
            return markdown.encode("utf-8"), "text/markdown"
        return SourceContentResponse(
            markdown=markdown, content_type="text/markdown", title=parse_title(markdown) or None
        )

    content_type, _ = mimetypes.guess_type(resolved.name)
    if content_type is None:
        content_type = "application/octet-stream"

    if raw:
        # Cap raw responses to inline-render-safe categories; anything else
        # degrades to a binary download so attacker-renamed files (e.g.
        # evil.html) can't trick the embedding browser into running script
        # under our origin.
        served_type = (
            content_type if _is_safe_for_inline_render(content_type) else "application/octet-stream"
        )
        return resolved.read_bytes(), served_type

    if not content_type.startswith("text/"):
        return SourceContentResponse(markdown="", content_type=content_type, title=None)

    text = resolved.read_text(encoding="utf-8", errors="replace")
    title = parse_title(text) or None
    return SourceContentResponse(markdown=text, content_type=content_type, title=title)
