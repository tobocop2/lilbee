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

# Heading shapes that open a model-authored citation block. Anchored to the
# start of the text as well as to a preceding newline, so an answer that is
# nothing but a fabricated block (heading at position 0) is still stripped.
_CITE_HEADING = (
    r"(?:\n{1,3}|\A)[ \t]*(?:#+\s*)?"
    r"(?:(?:Key\s+)?Sources|References|Bibliography|Citations)\s*:?\s*"
)
# One list line: a bullet, arrow, "[1]" or "1." marker and the rest of its line.
_CITE_LIST_LINE = r"[ \t]*(?:[-*•→\[]|\d+[.)])[^\n]*"

# A heading line followed by a list. The list is required so prose discussing
# such a heading is not clipped; the match ends with the list, not end-of-text,
# so an answer resuming after its citations keeps the continuation. Items may be
# blank-line separated, which markdown does routinely; stopping at the first
# blank line would leave the rest of a fabricated list in the answer.
_LLM_CITATION_BLOCK_RE = re.compile(
    _CITE_HEADING + r"\n\s*" + _CITE_LIST_LINE + r"(?:\n+" + _CITE_LIST_LINE + r")*",
    re.IGNORECASE,
)

# A citation-style heading at the very end of the text, nothing after it yet.
# Mid-stream this is ambiguous (the next line decides list vs prose), so it is
# held back rather than shown; at end of stream it is a dangling artifact of a
# citation block the model never finished, and is dropped either way.
_TRAILING_HEADING_RE = re.compile(_CITE_HEADING + r"$", re.IGNORECASE)


def display_source_path(source: str) -> str:
    """Render a chunk's source as an absolute path with ``~`` expansion.

    Source values in the store are stored relative to ``documents_dir`` so the
    database is portable across machines. For display we resolve back to the
    user's filesystem and substitute ``~`` for the home directory so the path
    is unambiguous without being noisy.

    Falls back to the raw source string only if resolution itself fails (an
    exotic OSError such as a symlink loop). A missing file is not a failure:
    ``resolve(strict=False)`` still returns the absolute path to where the
    file would be, so a moved documents directory renders its old location.
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
    location = _location_suffix(result)
    return f", {location}" if location else ""


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


def _location_suffix(result: SearchChunk) -> str:
    """The page or line span of a chunk, or empty when neither applies.

    Zero means "no location": PDF chunks whose page metadata was missing are
    stored with page 0, so a locator is only rendered when at least one end
    of the span is set. Sole owner of the rule -- ``_source_locator`` adds
    the leading separator and nothing else.
    """
    if result.content_type == "pdf" and (result.page_start or result.page_end):
        ps, pe = result.page_start, result.page_end
        return f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
    if result.content_type == "code" and (result.line_start or result.line_end):
        ls, le = result.line_start, result.line_end
        return f"line {ls}" if ls == le else f"lines {ls}-{le}"
    return ""


def source_markdown_link(source: str) -> str:
    """A bare source name as the same clickable ``[label](file-url)`` markdown a
    live answer's Sources block uses; the plain label when no path resolves.
    Public so restored transcripts render sources identically to live ones."""
    label = _source_label(source)
    url = _source_file_url(source)
    return f"[{label}]({url})" if url else label


def format_source(result: SearchChunk, citations: list[CitationRecord] | None = None) -> str:
    """Format a source as a clickable, readable citation: a ``[label](file-url)``
    markdown link plus any page/line locator. Web docs render as ``host · slug``;
    wiki chunks append their indented transitive citations.
    """
    head = source_markdown_link(result.source)
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
    """Number each passage by its source file, not its position, so citation
    numbers are stable while streaming and map 1:1 to the Sources block. Passages
    from the same file share a number.

    Each block carries a provenance header: without it the answering model sees
    bare numbered text and cannot attribute a claim to a named document, notice
    two passages share a source, or confirm it is reading the document asked for.
    """
    order: dict[str, int] = {}
    for r in results:
        order.setdefault(r.source, len(order) + 1)
    return "\n\n".join(f"[{order[r.source]}] ({_context_header(r)})\n{r.chunk}" for r in results)


# Grepped by consumers to know an answer carries its own Sources list (the
# no-results toast; the pill row, which must not stack a second list).
SOURCES_BLOCK_MARKER = "\n\nSources:\n"


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
    return SOURCES_BLOCK_MARKER + "\n" + "\n".join(lines)


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

    ``[n]`` markers are the primary signal, and ``n`` indexes the unique sources
    (matching ``build_context``/``format_sources_block``). Name mentions count
    too: context blocks show the model each source's name, and models often
    attribute by name ("according to survey_report.pdf") instead of by marker,
    which otherwise reads as an ungrounded answer to JSON consumers.
    """
    uniq = unique_sources(sources)
    cited = _extract_cited_indices(answer)
    picked = {i - 1 for i in cited if 1 <= i <= len(uniq)}
    lowered = answer.lower()
    for i, source in enumerate(uniq):
        if i in picked:
            continue
        name = Path(source.source).name.lower()
        stem = Path(source.source).stem
        if _mentions(lowered, name) or (
            _identifier_shaped(stem) and _mentions(lowered, stem.lower())
        ):
            picked.add(i)
    return [uniq[i] for i in sorted(picked)]


def _mentions(answer_lower: str, needle: str) -> bool:
    """Whether *answer_lower* names *needle* as a whole token.

    Plain containment marks a source cited whenever its name embeds in a
    longer one ("log-1" inside "catalog-10", "notes.md" inside
    "footnotes.md"), which inflates the grounding signal. Filename
    characters are what must not abut the match; surrounding punctuation
    and whitespace still count as a mention.
    """
    return re.search(rf"(?<![\w.-]){re.escape(needle)}(?![\w-])", answer_lower) is not None


def _stream_safe_prefix(text: str) -> str:
    """The prefix of *text* safe to show: a model-authored citation block
    (heading plus list) is removed, and a bare trailing citation heading is
    withheld until whatever follows disambiguates it.

    Kept rstrip-free so the streaming filter can track emitted length exactly;
    ``strip_llm_citations`` layers the final rstrip on top for one-shot answers.
    """
    cleaned = _LLM_CITATION_BLOCK_RE.sub("", text)
    return _TRAILING_HEADING_RE.sub("", cleaned)


def strip_llm_citations(text: str) -> str:
    """Remove an LLM-generated trailing citation block (or dangling citation
    heading) from answer text. Prose that merely mentions such a heading stays."""
    return _stream_safe_prefix(text).rstrip()


class StreamingCitationFilter:
    """Suppress a model-generated ``Sources:``/``References:`` citation block as
    it streams, so only lilbee's authoritative source list reaches the reader.

    A one-shot answer can be stripped after the fact, but streamed tokens are
    already on screen by the time the block is recognizable. This feeds the
    answer in incrementally and only releases text once it is certain not to be
    the start of a citation block: everything up to the last newline is safe to
    show, the final (possibly partial) line is held back until more text
    arrives, and a bare citation heading is held until the next line shows
    whether a list (a citation block, dropped) or prose (a legitimate mention,
    shown) follows. A heading left dangling when the stream ends is dropped.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> str:
        """Accept the next answer chunk; return the portion safe to show now."""
        self._buffer += text
        stripped = _stream_safe_prefix(self._buffer)
        cut = stripped.rfind("\n")
        committed = stripped if cut == -1 else stripped[:cut]
        if len(committed) > self._emitted:
            out = committed[self._emitted :]
            self._emitted = len(committed)
            return out
        return ""

    def flush(self) -> str:
        """Release any remaining safe text once the stream has ended."""
        final = _stream_safe_prefix(self._buffer)
        if len(final) > self._emitted:
            out = final[self._emitted :]
            self._emitted = len(final)
            return out
        return ""

    @property
    def answer(self) -> str:
        """The full answer shown so far, with any citation block removed."""
        return strip_llm_citations(self._buffer)
