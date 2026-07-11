"""Source formatting, context templating, and LLM citation extraction."""

from __future__ import annotations

import re
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.data.store import ChunkType, CitationRecord, SearchChunk

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


_CITE_REF_RE = re.compile(r"\[(\d+)\]")

# Matches trailing LLM-generated citation blocks like "Key sources:", "Sources:",
# "References:", "Bibliography:", "Citations:" (with optional markdown heading).
_LLM_CITATION_BLOCK_RE = re.compile(
    r"\n{1,3}(?:#+\s*)?(?:(?:Key\s+)?Sources|References|Bibliography|Citations)\s*:?\s*\n.*",
    re.IGNORECASE | re.DOTALL,
)


def display_source_path(source: str) -> str:
    """Render a chunk's source as an absolute path with ``~`` expansion.

    Source values in the store are stored relative to ``documents_dir`` so the
    database is portable across machines. For display we resolve back to the
    user's filesystem and substitute ``~`` for the home directory so the path
    is unambiguous without being noisy.

    Falls back to the raw source string if the file no longer exists on disk
    (e.g. the user moved the documents directory since ingestion).
    """
    candidate = cfg.documents_dir / source
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return source
    home = Path.home()
    try:
        return f"~/{resolved.relative_to(home)}"
    except ValueError:
        return str(resolved)


def _format_citation(citation: CitationRecord) -> str:
    """Format a single citation record as an indented attribution line."""
    source_display = display_source_path(citation["source_filename"])
    if citation["page_start"] or citation["page_end"]:
        ps, pe = citation["page_start"], citation["page_end"]
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"    → {source_display}, {pages}"
    if citation["line_start"] or citation["line_end"]:
        ls, le = citation["line_start"], citation["line_end"]
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"    → {source_display}, {lines}"
    return f"    → {source_display}"


def format_source(result: SearchChunk, citations: list[CitationRecord] | None = None) -> str:
    """Format a search result as a source citation line.
    For wiki chunks, shows the wiki page path followed by indented transitive citations.
    """
    source_display = display_source_path(result.source)
    if result.chunk_type is ChunkType.WIKI and citations:
        parts = [f"  → {source_display}"]
        for cit in citations:
            parts.append(_format_citation(cit))
        return "\n".join(parts)

    if result.content_type == "pdf":
        ps, pe = result.page_start, result.page_end
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"  → {source_display}, {pages}"

    if result.content_type == "code":
        ls, le = result.line_start, result.line_end
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"  → {source_display}, {lines}"

    return f"  → {source_display}"


def build_context(results: list[SearchChunk]) -> str:
    """Build context block from search results."""
    return "\n\n".join(f"[{i}] {r.chunk}" for i, r in enumerate(results, 1))


def _extract_cited_indices(text: str) -> set[int]:
    """Extract [N] citation references from LLM answer text."""
    return {int(m.group(1)) for m in _CITE_REF_RE.finditer(text)}


def cited_subset(answer: str, sources: list[SearchChunk]) -> list[SearchChunk]:
    """The sources the answer actually cited via [n] markers, in order (empty if none)."""
    cited = _extract_cited_indices(answer)
    return [sources[i - 1] for i in sorted(cited) if 1 <= i <= len(sources)]


def strip_llm_citations(text: str) -> str:
    """Remove LLM-generated trailing citation blocks from answer text."""
    return _LLM_CITATION_BLOCK_RE.sub("", text).rstrip()
