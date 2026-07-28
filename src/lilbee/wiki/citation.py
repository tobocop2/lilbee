"""Parse, render, and verify wiki citations.

Pure functions: no LLM dependency. Operates on markdown text and citation records.
"""

from dataclasses import dataclass
from enum import Enum

from lilbee.data.store import CitationRecord
from lilbee.wiki.grammar import (
    CITATION_BLOCK_COMMENT,
    CITATION_BLOCK_SEP,
    FOOTNOTE_RE,
)


class CitationStatus(Enum):
    """Result of verifying a citation against its source."""

    VALID = "valid"
    EXCERPT_MISSING = "excerpt_missing"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class ParsedCitation:
    """A citation anchor extracted from wiki markdown."""

    citation_key: str  # e.g. "src1"
    source_ref: str  # human-readable ref, e.g. "python-docs/typing.md, lines 12-45"
    line_number: int  # 1-based line number in the markdown


def parse_wiki_citations(markdown: str) -> list[ParsedCitation]:
    """Extract citation footnote definitions from wiki markdown.

    When the auto-generated block comment is present, scans from that
    line onward. When a looser model leaves the comment out, falls back
    to scanning the whole document for ``[^srcN]: ...`` definition lines.
    That pattern unambiguously identifies a citation footnote and only
    appears at the block level.
    """
    lines = markdown.splitlines()
    block_start = _find_citation_block_start(lines)
    start = block_start if block_start is not None else 0

    citations: list[ParsedCitation] = []
    for line_idx in range(start, len(lines)):
        match = FOOTNOTE_RE.match(lines[line_idx])
        if match:
            citations.append(
                ParsedCitation(
                    citation_key=match.group(1),
                    source_ref=match.group(2).strip(),
                    line_number=line_idx + 1,  # 1-based
                )
            )
    return citations


def render_citation_block(citations: list[CitationRecord]) -> str:
    """Generate the markdown footnote footer from CitationRecord objects.
    Returns the full citation block including separator and comment,
    or an empty string when there are no citations.
    """
    if not citations:
        return ""
    lines = [CITATION_BLOCK_SEP, CITATION_BLOCK_COMMENT]
    for rec in citations:
        lines.append(f"[^{rec['citation_key']}]: {_format_source_ref(rec)}")
    return "\n".join(lines) + "\n"


def excerpt_in_chunks(excerpt: str, chunk_texts: list[str]) -> bool:
    """Single rule for excerpt presence, shared by generation and lint.

    The excerpt must be present in ONE chunk: a quote stitched across a chunk
    boundary belongs to no source passage. An empty excerpt is never present.
    """
    from xberg import verify_excerpt

    return bool(excerpt) and any(verify_excerpt(excerpt, text) for text in chunk_texts)


def verify_citation(citation: CitationRecord, chunk_texts: list[str]) -> CitationStatus:
    """Check whether a citation's excerpt exists in the source's extracted chunks.

    Returns ``UNVERIFIABLE`` when the source has no extracted text to check
    against. Does not check hash staleness or source existence: caller handles
    those by comparing ``citation.source_hash`` against the current file hash
    and checking file presence.
    """
    if not chunk_texts:
        return CitationStatus.UNVERIFIABLE
    if excerpt_in_chunks(citation["excerpt"], chunk_texts):
        return CitationStatus.VALID
    return CitationStatus.EXCERPT_MISSING


def find_unmarked_claims(markdown: str) -> list[str]:
    """Find body statements that are neither cited ``[^srcN]`` nor marked ``[*inference*]``.

    Delegates to xberg's footnote/citation API over the body (frontmatter and the
    citation block stripped).
    """
    from xberg import find_unmarked_claims as _find_unmarked_claims

    return _find_unmarked_claims(extract_body(markdown))


def strip_citation_block(markdown: str) -> str:
    """Remove the citation block (separator + comment + footnotes) from markdown."""
    lines = markdown.splitlines()
    body_end = _body_end(lines)
    if body_end == len(lines):
        return markdown
    return "\n".join(lines[:body_end]).rstrip() + "\n"


def _find_citation_block_start(lines: list[str]) -> int | None:
    """Return the 0-based line index where the citation block begins, or None."""
    for i, line in enumerate(lines):
        if line.strip() == CITATION_BLOCK_COMMENT:
            return i
    return None


def _body_end(lines: list[str]) -> int:
    """Return the line index the body ends at, before any citation block.

    Footnote definitions without the auto-generated comment are a citation
    block too: ``parse_wiki_citations`` reads them, so stripping removes them.
    """
    block_start = _find_citation_block_start(lines)
    if block_start is None:
        return _body_end_before_trailing_footnotes(lines)
    return _drop_separator(lines, block_start)


def _body_end_before_trailing_footnotes(lines: list[str]) -> int:
    """Return the line index before the document's trailing run of ``[^srcN]:`` definitions."""
    body_end = len(lines)
    found = False
    while body_end > 0:
        line = lines[body_end - 1]
        if FOOTNOTE_RE.match(line):
            found = True
        elif line.strip():
            break
        body_end -= 1
    if not found:
        return len(lines)
    return _drop_separator(lines, body_end)


def _drop_separator(lines: list[str], body_end: int) -> int:
    """Return *body_end* less a preceding ``---`` separator line, if there is one."""
    if body_end > 0 and lines[body_end - 1].strip() == CITATION_BLOCK_SEP:
        return body_end - 1
    return body_end


def extract_body(markdown: str) -> str:
    """Return markdown body: strip YAML frontmatter and citation block."""
    text = _strip_frontmatter(markdown)
    lines = text.splitlines()
    body_end = _body_end(lines)
    if body_end == len(lines):
        return text
    return "\n".join(lines[:body_end])


def _strip_frontmatter(markdown: str) -> str:
    """Remove YAML frontmatter delimited by ``---`` at the start."""
    if not markdown.startswith("---"):
        return markdown
    lines = markdown.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :])
    return markdown


def _format_source_ref(rec: CitationRecord) -> str:
    """Format a CitationRecord into a human-readable footnote reference."""
    ref = rec["source_filename"]
    has_page = rec["page_start"] is not None and rec["page_start"] > 0
    has_page_end = rec["page_end"] is not None and rec["page_end"] > 0
    has_line = rec["line_start"] is not None and rec["line_start"] > 0
    has_line_end = rec["line_end"] is not None and rec["line_end"] > 0
    if has_page or has_page_end:
        if rec["page_start"] == rec["page_end"]:
            ref += f", page {rec['page_start']}"
        else:
            ref += f", pages {rec['page_start']}-{rec['page_end']}"
    elif has_line or has_line_end:
        ref += f", lines {rec['line_start']}-{rec['line_end']}"
    if rec["excerpt"]:
        ref += f', excerpt: "{rec["excerpt"]}"'
    return ref
