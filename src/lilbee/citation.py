"""Parse, render, and verify wiki citations.

Pure functions — no LLM dependency. Operates on markdown text and citation records.
"""

import re
from dataclasses import dataclass
from enum import Enum

# Pattern for inline citation anchors: [^src1], [^src2], etc.
_CITE_RE = re.compile(r"\[\^(src\d+)\]")

# Pattern for footnote definitions in the citation block:
#   [^src1]: python-docs/typing.md, lines 12-45
_FOOTNOTE_RE = re.compile(r"^\[\^(src\d+)\]:\s*(.+)$", re.MULTILINE)

# Pattern for inference markers: [*inference*]
_INFERENCE_RE = re.compile(r"\[\*inference\*\]")

# Separator line that precedes the auto-generated citation block
_CITATION_BLOCK_SEP = "---"
_CITATION_BLOCK_COMMENT = "<!-- citations (auto-generated from _citations table -- do not edit) -->"


class CitationStatus(Enum):
    """Result of verifying a citation against its source."""

    VALID = "valid"
    STALE_HASH = "stale_hash"
    SOURCE_DELETED = "source_deleted"
    EXCERPT_MISSING = "excerpt_missing"


@dataclass(frozen=True)
class ParsedCitation:
    """A citation anchor extracted from wiki markdown."""

    citation_key: str  # e.g. "src1"
    source_ref: str  # human-readable ref, e.g. "python-docs/typing.md, lines 12-45"
    line_number: int  # 1-based line number in the markdown


@dataclass(frozen=True)
class CitationRecord:
    """A row from the _citations table.

    Defined here until store.py gains the _citations table in a later phase.
    """

    wiki_source: str
    citation_key: str
    source_filename: str
    source_hash: str
    excerpt: str
    page_start: int = 0
    page_end: int = 0
    line_start: int = 0
    line_end: int = 0


def parse_wiki_citations(markdown: str) -> list[ParsedCitation]:
    """Extract citation footnote definitions from the auto-generated citation block.

    Returns a ParsedCitation for each ``[^srcN]: ...`` line found after the
    citation block separator.
    """
    block_start = _find_citation_block_start(markdown)
    if block_start is None:
        return []

    lines = markdown.splitlines()
    citations: list[ParsedCitation] = []
    for line_idx in range(block_start, len(lines)):
        match = _FOOTNOTE_RE.match(lines[line_idx])
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

    Returns the full citation block including separator and comment.
    """
    lines = [_CITATION_BLOCK_SEP, _CITATION_BLOCK_COMMENT]
    for rec in citations:
        lines.append(f"[^{rec.citation_key}]: {_format_source_ref(rec)}")
    return "\n".join(lines) + "\n"


def verify_citation(citation: CitationRecord, source_text: str) -> CitationStatus:
    """Check whether a citation's excerpt exists in the source text.

    Does not check hash staleness or source existence — caller handles those
    by comparing ``citation.source_hash`` against the current file hash and
    checking file presence.
    """
    if not citation.excerpt:
        return CitationStatus.EXCERPT_MISSING
    if _normalize(citation.excerpt) in _normalize(source_text):
        return CitationStatus.VALID
    return CitationStatus.EXCERPT_MISSING


def find_unmarked_claims(markdown: str) -> list[str]:
    """Find statements that are neither cited ``[^srcN]`` nor marked ``[*inference*]``.

    Scans non-empty, non-metadata lines in the body (before the citation block).
    Returns the text of each unmarked line.
    """
    body = _extract_body(markdown)
    lines = body.splitlines()
    unmarked: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not _is_content_line(stripped):
            continue
        if _CITE_RE.search(stripped) or _INFERENCE_RE.search(stripped):
            continue
        unmarked.append(stripped)
    return unmarked


def _find_citation_block_start(markdown: str) -> int | None:
    """Return the 0-based line index where the citation block begins, or None."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == _CITATION_BLOCK_COMMENT:
            return i
    return None


def _extract_body(markdown: str) -> str:
    """Return markdown body: strip YAML frontmatter and citation block."""
    text = _strip_frontmatter(markdown)
    block_start = _find_citation_block_start(text)
    if block_start is None:
        return text
    lines = text.splitlines()
    # Also strip the --- separator line immediately before the comment
    body_end = block_start
    if body_end > 0 and lines[body_end - 1].strip() == _CITATION_BLOCK_SEP:
        body_end -= 1
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


def _is_content_line(stripped: str) -> bool:
    """Return True if a line contains a substantive claim (not heading/blank/marker)."""
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return stripped != _CITATION_BLOCK_SEP


def _format_source_ref(rec: CitationRecord) -> str:
    """Format a CitationRecord into a human-readable footnote reference."""
    ref = rec.source_filename
    if rec.page_start or rec.page_end:
        if rec.page_start == rec.page_end:
            ref += f", page {rec.page_start}"
        else:
            ref += f", pages {rec.page_start}-{rec.page_end}"
    elif rec.line_start or rec.line_end:
        ref += f", lines {rec.line_start}-{rec.line_end}"
    return ref


def _normalize(text: str) -> str:
    """Normalize whitespace for fuzzy excerpt matching."""
    return " ".join(text.split()).lower()
