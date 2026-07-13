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


_WEB_PREFIX = "_web"
_SOURCE_SEP = " · "


def _source_label(source: str) -> str:
    """Readable name for a source. Web-ingested docs collapse to ``host · slug``;
    local files keep their documents-dir-relative path."""
    prefix = f"{_WEB_PREFIX}/"
    if not source.startswith(prefix):
        return source
    segments = [p for p in source.removeprefix(prefix).split("/") if p != "index.md"]
    if not segments:
        return source
    host = segments[0].removeprefix("www.")
    slug = segments[-1].removesuffix(".md")
    return host if slug == host else f"{host}{_SOURCE_SEP}{slug}"


def _source_file_url(source: str) -> str | None:
    """A ``file://`` URL to the source on disk, so a reader can click it open, or
    None when the path can't be resolved to an absolute location."""
    try:
        return (cfg.documents_dir / source).resolve(strict=False).as_uri()
    except (OSError, ValueError):
        return None


def _source_locator(result: SearchChunk) -> str:
    """The ``, page N`` / ``, lines A-B`` suffix for a source line, or ''."""
    if result.content_type == "pdf" and (result.page_start or result.page_end):
        ps, pe = result.page_start, result.page_end
        return f", page {ps}" if ps == pe else f", pages {ps}-{pe}"
    if result.content_type == "code" and (result.line_start or result.line_end):
        ls, le = result.line_start, result.line_end
        return f", line {ls}" if ls == le else f", lines {ls}-{le}"
    return ""


def _format_citation(citation: CitationRecord) -> str:
    """Format a single wiki transitive citation as an indented attribution line."""
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
    """Format a source as a clickable, readable citation: a ``[label](file-url)``
    markdown link plus any page/line locator. Web docs render as ``host · slug``;
    wiki chunks append their indented transitive citations.
    """
    label = _source_label(result.source)
    url = _source_file_url(result.source)
    head = f"[{label}]({url})" if url else label
    if result.chunk_type is ChunkType.WIKI and citations:
        return "\n".join([head, *(_format_citation(c) for c in citations)])
    return f"{head}{_source_locator(result)}"


def unique_sources(results: list[SearchChunk]) -> list[SearchChunk]:
    """The first chunk of each distinct source, in retrieval order. A source's
    1-based position here is the citation number the model emits (see
    ``build_context``) and the number shown in the Sources block, so the answer's
    ``[n]`` markers and the source list always agree."""
    seen: set[str] = set()
    out: list[SearchChunk] = []
    for r in results:
        if r.source not in seen:
            seen.add(r.source)
            out.append(r)
    return out


def build_context(results: list[SearchChunk]) -> str:
    """Number each passage by its source file, not its position, so citation
    numbers are stable while streaming and map 1:1 to the Sources block. Passages
    from the same file share a number."""
    order: dict[str, int] = {}
    for r in results:
        order.setdefault(r.source, len(order) + 1)
    return "\n\n".join(f"[{order[r.source]}] {r.chunk}" for r in results)


def format_sources_block(
    results: list[SearchChunk],
    citations_map: dict[str, list[CitationRecord]] | None = None,
) -> str:
    """The authoritative numbered ``Sources:`` block appended to an answer. Each
    unique source is numbered to match the ``[n]`` markers the model emitted, so
    every inline citation resolves to a line here. '' when there are no sources."""
    sources = unique_sources(results)
    if not sources:
        return ""
    # A markdown ordered list, so a Markdown renderer puts each source on its own
    # line (plain "  → path" lines collapse into one soft-wrapped paragraph) and
    # the list number is the citation number the model cited inline.
    lines = [
        f"{i}. {format_source(r, citations=(citations_map or {}).get(r.source))}"
        for i, r in enumerate(sources, 1)
    ]
    return "\n\nSources:\n\n" + "\n".join(lines)


def _extract_cited_indices(text: str) -> set[int]:
    """Extract [N] citation references from LLM answer text."""
    return {int(m.group(1)) for m in _CITE_REF_RE.finditer(text)}


def cited_subset(answer: str, sources: list[SearchChunk]) -> list[SearchChunk]:
    """The sources the answer actually cited via [n] markers, where n indexes the
    unique sources (matching ``build_context``/``format_sources_block``). Empty if
    none cited."""
    uniq = unique_sources(sources)
    cited = _extract_cited_indices(answer)
    return [uniq[i - 1] for i in sorted(cited) if 1 <= i <= len(uniq)]


def _drop_citation_block(text: str) -> str:
    """Remove a trailing LLM-generated citation block without trimming whitespace.

    Kept rstrip-free so the streaming filter can track emitted length exactly;
    ``strip_llm_citations`` layers the final rstrip on top for one-shot answers.
    """
    return _LLM_CITATION_BLOCK_RE.sub("", text)


def strip_llm_citations(text: str) -> str:
    """Remove LLM-generated trailing citation blocks from answer text."""
    return _drop_citation_block(text).rstrip()


class StreamingCitationFilter:
    """Suppress a model-generated trailing ``Sources:``/``References:`` block as
    it streams, so only lilbee's authoritative source list reaches the reader.

    A one-shot answer can be stripped after the fact, but streamed tokens are
    already on screen by the time the block is recognizable. This feeds the
    answer in incrementally and only releases text once it is certain not to be
    the start of a citation block: everything up to the last newline is safe to
    show, while the final (possibly partial) line is held back until more text
    arrives or the stream ends. If a citation heading does appear, the block and
    everything after it is dropped for good.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> str:
        """Accept the next answer chunk; return the portion safe to show now."""
        self._buffer += text
        stripped = _drop_citation_block(self._buffer)
        cut = stripped.rfind("\n")
        committed = stripped if cut == -1 else stripped[:cut]
        if len(committed) > self._emitted:
            out = committed[self._emitted :]
            self._emitted = len(committed)
            return out
        return ""

    def flush(self) -> str:
        """Release any remaining safe text once the stream has ended."""
        final = _drop_citation_block(self._buffer)
        if len(final) > self._emitted:
            out = final[self._emitted :]
            self._emitted = len(final)
            return out
        return ""

    @property
    def answer(self) -> str:
        """The full answer shown so far, with any citation block removed."""
        return strip_llm_citations(self._buffer)
