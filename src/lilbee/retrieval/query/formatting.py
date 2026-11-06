"""Source formatting, context templating, and LLM citation extraction."""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from lilbee.core.text import collapse_whitespace, strip_control_characters
from lilbee.core.web_names import INDEX_STEM, WEB_PREFIX, is_web_source, page_stem
from lilbee.data.store import SearchChunk, is_memory_source
from lilbee.data.types import CODE_CONTENT_TYPE

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

# Heading shapes that open a model-authored citation block: anchored to the
# start of the text as well as a preceding newline (an answer that is nothing
# but a fabricated block still strips), optionally markdown-decorated (ATX
# hashes, emphasis around the word or the colon: "**Sources:**", "*References:*").
_HEADING_WORDS = r"(?:(?:Key\s+)?Sources|References|Bibliography|Citations)"
_CITE_HEADING = (
    r"(?:\n{1,3}|\A)[ \t]*(?:#+\s*)?[*_]{0,3}" + _HEADING_WORDS + r"[*_]{0,3}\s*:?\s*[*_]{0,3}"
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
_TRAILING_HEADING_RE = re.compile(_CITE_HEADING + r"\s*$", re.IGNORECASE)


_SOURCE_SEP = " · "
# A page controls its title, so a rendered label never opens a link, image, code
# span, tag, entity, emphasis or strikethrough of its own, and never starts a block.
_MARKDOWN_SPECIAL = "\\[]`<&*_~"
_MARKDOWN_SPECIAL_RE = re.compile(f"[{re.escape(_MARKDOWN_SPECIAL)}]")
_BLOCK_MARK_RE = re.compile(
    f"^[{re.escape(''.join(c for c in string.punctuation if c not in _MARKDOWN_SPECIAL))}]"
)
_LIST_NUMBER_RE = re.compile(r"^(\d+)([.)])(?=\s|$)")


def source_label(source: str, title: str = "") -> str:
    """The plain one-line name of a source: ``host · title`` for a crawled page with a
    stored *title*, else ``host · slug``; a local file's documents-dir-relative path."""
    if not is_web_source(source):
        return collapse_whitespace(source)
    *folders, file_name = source.removeprefix(WEB_PREFIX).split("/")
    stem = page_stem(file_name)
    segments = folders if stem == INDEX_STEM else [*folders, stem]
    if not segments:
        return collapse_whitespace(source)
    host = segments[0].removeprefix("www.")
    slug = collapse_whitespace(title) or segments[-1]
    return host if slug == segments[0] else f"{host}{_SOURCE_SEP}{slug}"


def markdown_text(text: str) -> str:
    """*text* on one line as markdown that shows exactly that text, with no markup, block
    or control character."""
    plain = strip_control_characters(collapse_whitespace(text))
    escaped = _MARKDOWN_SPECIAL_RE.sub(r"\\\g<0>", plain)
    escaped = _BLOCK_MARK_RE.sub(r"\\\g<0>", escaped)
    return _LIST_NUMBER_RE.sub(r"\1\\\2", escaped)


def _source_file_url(source: str) -> str | None:
    """A ``file://`` URL to the source on disk, so a reader can click it open, or
    None when the path can't be resolved to an absolute location."""
    from lilbee.data.ingest.discovery import resolve_source_path

    try:
        return resolve_source_path(source).resolve(strict=False).as_uri()
    except (OSError, ValueError):
        return None


def _location_suffix(result: SearchChunk) -> str:
    """The page or line span of a chunk, or empty when neither applies.

    Zero means "no location": PDF chunks whose page metadata was missing are
    stored with page 0, so a locator is only rendered when at least one end
    of the span is set. Sole owner of the rule: a Sources line shows the
    ``SourceRef.locator`` it computes.
    """
    if result.content_type == "pdf" and (result.page_start or result.page_end):
        ps, pe = result.page_start, result.page_end
        return f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
    if result.content_type == CODE_CONTENT_TYPE and (result.line_start or result.line_end):
        ls, le = result.line_start, result.line_end
        return f"line {ls}" if ls == le else f"lines {ls}-{le}"
    return ""


# Every locator ``_location_suffix`` writes. A client saves a ref's locator back, and
# only these forms are accepted.
_LOCATOR_RE = re.compile(r"(?:page|line) \d+|(?:pages|lines) \d+-\d+")


def is_locator(text: str) -> bool:
    """Whether *text* is a locator lilbee writes, or '' for none."""
    return not text or _LOCATOR_RE.fullmatch(text) is not None


def source_ref_link(ref: SourceRef) -> str:
    """*ref* as the ``[label](file-url)`` link of a Sources line, the label as plain text;
    the plain label when no path resolves."""
    return _linked_label(ref.source, ref.label)


def source_ref_line(ref: SourceRef) -> str:
    """*ref* as a Sources line shows it: its link and its page or line locator."""
    return located(source_ref_link(ref), ref)


def markdown_source_line(ref: SourceRef) -> str:
    """*ref* as a Markdown renderer shows its Sources line: its label and locator as
    written, linked to the file of its own source."""
    line = _linked_label(ref.source, markdown_text(ref.label))
    return f"{line}, {markdown_text(ref.locator)}" if ref.locator else line


def located(name: str, ref: SourceRef) -> str:
    """*name*, the way a Sources line names *ref*, followed by its locator when it has one."""
    return f"{name}, {ref.locator}" if ref.locator else name


def _linked_label(source: str, label: str) -> str:
    """*label* linked to the file of *source*, or plain when *source* has no file."""
    if not source or is_memory_source(source):
        return label
    url = _source_file_url(source)
    return f"[{label}]({url})" if url else label


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


@dataclass(frozen=True)
class SourceRef:
    """One source as a client names it: its ``label``, the ``[n]`` an answer cites it
    by (None when no answer numbers it) and the plain-text ``locator`` of its page or
    line span, such as ``page 3`` ('' when it has none)."""

    source: str
    label: str
    citation_number: int | None
    locator: str = ""


def _chunk_ref(result: SearchChunk, title: str, number: int | None) -> SourceRef:
    """The ref of *result*'s source, located by *result*. A memory row has no title or locator."""
    if result.memory_id is not None:
        return SourceRef(result.source, source_label(result.source), number)
    return SourceRef(
        result.source, source_label(result.source, title), number, _location_suffix(result)
    )


def source_refs(results: list[SearchChunk], titles: Mapping[str, str]) -> list[SourceRef]:
    """Each unique source of *results*, with its label from its title in *titles*, its
    locator from its first chunk and its citation number: its position in
    ``unique_sources``, the ``[n]`` the model cites."""
    return [
        _chunk_ref(r, titles.get(r.source, ""), i) for i, r in enumerate(unique_sources(results), 1)
    ]


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
# A Sources line's ``[label](file-url)`` link. One source per line, so the label
# runs to the line's last ``](file://``: a label holding that text cannot end it.
FILE_LINK_RE = re.compile(r"\[(?P<label>[^\n]+)\]\((?P<url>file://[^)\s]+)\)")
# Parsed after a text; if it lands inside a fence, the text left that fence open.
_FENCE_PROBE = "\n\n# probe"


def format_sources_block(results: list[SearchChunk], titles: Mapping[str, str]) -> str:
    """The authoritative numbered ``Sources:`` block appended to an answer. Each
    unique source is numbered to match the ``[n]`` markers the model emitted, so
    every inline citation resolves to a line here; *titles* holds the stored title
    of each crawled page. '' when there are no sources."""
    refs = source_refs(results, titles)
    return _sources_list(refs, source_ref_line) if refs else ""


@dataclass(frozen=True)
class AnswerSources:
    """Emitted once after a grounded answer's last token: the retrieved *results* the
    answer was written from and the stored *titles* of their crawled pages."""

    results: list[SearchChunk]
    titles: Mapping[str, str]

    def refs(self) -> list[SourceRef]:
        """The answer's Sources list, numbered as the answer cites it."""
        return source_refs(self.results, self.titles)

    def block(self, answer: str) -> str:
        """The ``Sources:`` block that follows *answer*, closing a code block it left open."""
        refs = self.refs()
        if not refs:
            return ""
        return close_open_fence(answer).removeprefix(answer) + _sources_list(refs, source_ref_line)


def append_sources_block(
    content: str,
    refs: Sequence[SourceRef],
    render: Callable[[SourceRef], str] = source_ref_line,
) -> str:
    """*content* ending in the numbered ``Sources:`` list of *refs*, each shown by
    *render*, whatever the content says. A code block the content leaves open is
    closed first, so the list is never inside it."""
    return close_open_fence(content.rstrip()) + _sources_list(refs, render)


def _sources_list(refs: Sequence[SourceRef], render: Callable[[SourceRef], str]) -> str:
    """The ``Sources:`` list of *refs*, each shown by *render* under its citation number
    (its position when it has none). The one renderer of the live and restored lists."""
    numbers = [i if r.citation_number is None else r.citation_number for i, r in enumerate(refs, 1)]
    if numbers == list(range(1, len(refs) + 1)):
        # A markdown ordered list, so a Markdown renderer puts each source on its own
        # line and the list number is the citation number the model cited inline.
        lines = "\n".join(f"{n}. {render(r)}" for n, r in zip(numbers, refs, strict=True))
    else:
        # A markdown list shows its items numbered on from the first, so a cited
        # subset [1], [3] would show 1. and 2.: escaped numbers on hard-broken
        # lines show each number as written.
        lines = "\\\n".join(f"{n}\\. {render(r)}" for n, r in zip(numbers, refs, strict=True))
    return SOURCES_BLOCK_MARKER + "\n" + lines


def split_sources_list(content: str) -> tuple[str, str]:
    """*content* split before its last Sources list, unless that list is quoted in a code block.

    A list after a code block the answer never closed (a cut-off answer) is lilbee's;
    a list inside a code block that closes after it is pasted text.
    """
    text, marker, stored_sources = content.rpartition(SOURCES_BLOCK_MARKER)
    if not marker:
        return content, ""
    fence = open_code_fence(text)
    if fence is None or _never_closed(fence, content):
        return text, marker + stored_sources
    return content, ""


def _never_closed(fence: Token, content: str) -> bool:
    """Whether *fence*, opened in a prefix of *content*, is still open at its end."""
    end = open_code_fence(content)
    return (
        end is not None
        and end.map is not None
        and fence.map is not None
        and end.map[0] == fence.map[0]
    )


def terminal_sources_text(sources: str) -> str:
    """The Sources list *sources* with every control character and escape sequence removed.

    The one place a stored title or source label is made safe for a terminal. A page controls
    its title, and a terminal reads an escape sequence in it as a command. Line breaks stay.
    """
    return "\n".join(strip_control_characters(line) for line in sources.split("\n"))


def terminal_sources_block(content: str) -> str:
    """*content* with the Sources list it ends in made safe for a terminal or a TUI widget."""
    text, sources = split_sources_list(content)
    return text + terminal_sources_text(sources)


@cache
def commonmark() -> MarkdownIt:
    """The shared CommonMark parser."""
    return MarkdownIt("commonmark")


def open_code_fence(text: str) -> Token | None:
    """The top-level code fence *text* leaves open, or None when every fence is closed."""
    last = commonmark().parse(text + _FENCE_PROBE)[-1]
    return last if last.type == "fence" else None


def close_open_fence(text: str) -> str:
    """*text* with a code fence it leaves open closed, so it cannot swallow what follows.

    The parser decides: when a heading placed after *text* ends up inside a fence,
    that fence is closed. A fence in a list item never does, because the heading
    ends the item, so only a top-level fence is closed, and at column 0.
    """
    fence = open_code_fence(text)
    return f"{text}\n{fence.markup}" if fence else text


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
