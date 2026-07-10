"""Source formatting, context templating, and LLM citation extraction."""

from __future__ import annotations

import re
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.data.store import ChunkType, CitationRecord, SearchChunk

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


# Bracketed citation groups: [1], [1, 2], [1-3], [1, 3-5]. Models mix all of
# these despite being asked for single [n] markers; matching only [n] made
# cited_sources under-count and fed JSON consumers false-negative grounding.
_CITE_GROUP_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
_CITE_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
# Ranges wider than this are page spans or line numbers, not citation lists.
_MAX_CITE_RANGE = 32

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


def _location_suffix(result: SearchChunk) -> str:
    """The page or line span of a chunk, or empty when neither applies."""
    if result.content_type == "pdf":
        ps, pe = result.page_start, result.page_end
        return f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
    if result.content_type == "code":
        ls, le = result.line_start, result.line_end
        return f"line {ls}" if ls == le else f"lines {ls}-{le}"
    return ""


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

    location = _location_suffix(result)
    if location:
        return f"  → {source_display}, {location}"
    return f"  → {source_display}"


def _context_header(result: SearchChunk) -> str:
    """One-line provenance for a context block: source name plus location.

    Without it the answering model sees bare numbered text: it cannot
    attribute a claim to a named document, notice two chunks share a source,
    or confirm it is reading the document the user asked about.
    """
    location = _location_suffix(result)
    if location:
        return f"{result.source}, {location}"
    return result.source


def build_context(results: list[SearchChunk]) -> str:
    """Build context block from search results, one provenance header each."""
    return "\n\n".join(f"[{i}] ({_context_header(r)})\n{r.chunk}" for i, r in enumerate(results, 1))


def _extract_cited_indices(text: str) -> set[int]:
    """Extract citation references from LLM answer text: [1], [1, 2], [1-3]."""
    indices: set[int] = set()
    for m in _CITE_GROUP_RE.finditer(text):
        group = m.group(1)
        remainder = _CITE_RANGE_RE.sub("", group)
        for start, end in _CITE_RANGE_RE.findall(group):
            lo, hi = int(start), int(end)
            if lo <= hi <= lo + _MAX_CITE_RANGE:
                indices.update(range(lo, hi + 1))
        indices.update(int(n) for n in re.findall(r"\d+", remainder))
    return indices


def _identifier_shaped(stem: str) -> bool:
    """Whether a filename stem is distinctive enough to match in prose.

    A stem carrying a digit or a separator ("survey_report", "ARC-00000482")
    only appears in an answer when the model names the document; a bare word
    stem ("notes") collides with ordinary prose and cannot be trusted.
    """
    return any(c.isdigit() or c in "_-" for c in stem)


def cited_subset(answer: str, sources: list[SearchChunk]) -> list[SearchChunk]:
    """The sources the answer actually referenced, in order (empty if none).

    ``[n]`` markers are the primary signal. Name mentions count too: context
    blocks show the model each source's name, and models often attribute by
    name ("according to survey_report.pdf") instead of by marker, which
    previously read as an ungrounded answer to JSON consumers.
    """
    cited = _extract_cited_indices(answer)
    picked = {i - 1 for i in cited if 1 <= i <= len(sources)}
    lowered = answer.lower()
    for i, source in enumerate(sources):
        if i in picked:
            continue
        name = Path(source.source).name.lower()
        stem = Path(source.source).stem
        if name in lowered or (_identifier_shaped(stem) and stem.lower() in lowered):
            picked.add(i)
    return [sources[i] for i in sorted(picked)]


def strip_llm_citations(text: str) -> str:
    """Remove LLM-generated trailing citation blocks from answer text."""
    return _LLM_CITATION_BLOCK_RE.sub("", text).rstrip()
