"""Citation grammar and resolution for wiki pages.

The grammar half parses ``[^srcN]`` footnote definitions out of wiki
markdown, renders them back, strips the block from a body, and checks a
single record's excerpt against the text it claims. The resolution half
builds :class:`CitationRecord` rows from parsed markers (single-source
and multi-source variants), matches each excerpt back to the source
chunk it came from, and renders the YAML provenance block written into a
wiki page's frontmatter.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import yaml

from lilbee.core.config import Config
from lilbee.data.store import CitationRecord, SearchChunk
from lilbee.wiki.entity_extractor.factory import effective_entity_mode
from lilbee.wiki.grammar import (
    CITATION_BLOCK_COMMENT,
    CITATION_BLOCK_SEP,
    CITE_RE,
    CODE_FENCE_RE,
    FOOTNOTE_RE,
)

log = logging.getLogger(__name__)

# JSON-style escape sequences that may appear inside quoted excerpts the
# model emits. Any backslash-prefixed character not in this map stays
# verbatim (e.g. ``\\x`` passes through unchanged).
_EXCERPT_ESCAPES: dict[str, str] = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}

# Encoding side of the same table. A rendered footnote definition must stay on
# one line and must re-parse to the excerpt it was rendered from.
_EXCERPT_ESCAPE_INVERSE: dict[str, str] = {
    char: f"\\{escape}" for escape, char in _EXCERPT_ESCAPES.items()
}


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


def _fence_flags(lines: list[str]) -> list[bool]:
    """Per-line ``inside a code fence`` flags, fence delimiters included.

    Footnote definitions and ``[^srcN]`` markers inside a fence are example
    syntax on a page documenting the citation grammar, not citations.
    """
    flags: list[bool] = []
    in_fence = False
    for line in lines:
        if CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            flags.append(True)
            continue
        flags.append(in_fence)
    return flags


def parse_wiki_citations(markdown: str) -> list[ParsedCitation]:
    """Extract citation footnote definitions from wiki markdown.

    Scans the whole document outside code fences: the ``[^srcN]: ...`` pattern
    unambiguously identifies a citation footnote wherever a model put it, so a
    mid-body definition is a citation too, while a fenced one is example syntax.
    A key is taken once: a mid-body definition repeated in the trailing block is
    a single citation.
    """
    lines = markdown.splitlines()

    citations: list[ParsedCitation] = []
    seen: set[str] = set()
    for line_idx, fenced in enumerate(_fence_flags(lines)):
        if fenced:
            continue
        match = FOOTNOTE_RE.match(lines[line_idx])
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
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


def footnote_marker_keys(body: str) -> set[str]:
    """Citation keys the body's ``[^srcN]`` markers reference, fences excluded.

    Shares :func:`_fence_flags` with the parser and the scrubber: a marker the
    other two treat as example syntax must not count as a section's citation.
    """
    lines = body.splitlines()
    return {
        key
        for line, fenced in zip(lines, _fence_flags(lines), strict=True)
        if not fenced
        for key in CITE_RE.findall(line)
    }


def scrub_unverified_markers(body: str, verified: list[CitationRecord]) -> str:
    """Drop in-body footnote definition lines and unverified ``[^srcN]`` markers.

    The citation block is re-rendered from the verified records, so any
    definition line still inside the body would either duplicate a verified
    definition or publish an unverified excerpt as prose. A marker whose
    definition was dropped renders as literal ``[^srcN]`` text and hides the
    claim from ``find_unmarked_claims``. Fenced lines are example syntax and
    stay verbatim.
    """
    keys = {rec["citation_key"] for rec in verified}
    # keepends so removing a definition line does not also rewrite the body's
    # own line endings or drop its trailing newline.
    lines = body.splitlines(keepends=True)
    kept = [
        line if fenced else CITE_RE.sub(lambda m: m.group(0) if m.group(1) in keys else "", line)
        for line, fenced in zip(lines, _fence_flags(lines), strict=True)
        if fenced or not FOOTNOTE_RE.match(line)
    ]
    return "".join(kept)


def wiki_sourced_count(records: list[CitationRecord], config: Config) -> int:
    """Number of *records* citing a wiki page rather than a raw source.

    :func:`verify_citations` skips these, so they are neither rendered nor
    dropped as unverified.
    """
    return sum(1 for rec in records if _is_wiki_sourced(rec, config))


def _is_wiki_sourced(record: CitationRecord, config: Config) -> bool:
    """Whether a citation names a wiki page as its source."""
    return record["source_filename"].startswith(config.wiki_dir + "/")


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

    Without the auto-generated comment only the document's trailing run of
    ``[^srcN]:`` definitions is a block; definitions followed by prose stay
    in the body.
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
        ref += f', excerpt: "{_encode_excerpt_escapes(rec["excerpt"])}"'
    return ref


def _extract_excerpt(source_ref: str) -> str:
    """Extract the quoted excerpt from a citation source_ref string.
    e.g. 'doc.md, excerpt: "Python supports typing."' → 'Python supports typing.'

    Common JSON-style escape sequences inside the quoted span (``\\n``,
    ``\\t``, ``\\"``, ``\\\\``) are decoded to their literal characters so
    they round-trip against the source text. Some models "helpfully"
    encode real newlines as ``\\n`` when emitting a quoted excerpt; the
    source chunk they came from has real newlines, so skipping this
    step leaves otherwise-faithful citations unverifiable.
    """
    marker = 'excerpt: "'
    idx = source_ref.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = _find_closing_quote(source_ref, start)
    raw = source_ref[start:].strip() if end == -1 else source_ref[start:end].strip()
    return _decode_excerpt_escapes(raw)


def _find_closing_quote(text: str, start: int) -> int:
    """Index of the first unescaped ``"`` at or after *start*, or -1.

    An escaped quote belongs to the excerpt, so the scan steps over it rather
    than ending the quoted span there.
    """
    i = start
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i
        i += 1
    return -1


def _encode_excerpt_escapes(text: str) -> str:
    """Escape *text* so :func:`_decode_excerpt_escapes` returns it unchanged."""
    return "".join(_EXCERPT_ESCAPE_INVERSE.get(char, char) for char in text)


def _decode_excerpt_escapes(raw: str) -> str:
    """Decode the JSON-style escapes models commonly emit inside quoted strings."""
    if "\\" not in raw:
        return raw
    result: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        mapped = _EXCERPT_ESCAPES.get(raw[i + 1]) if ch == "\\" and i + 1 < len(raw) else None
        if mapped is not None:
            result.append(mapped)
            i += 2
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def _find_excerpt_location(
    excerpt: str,
    chunks: list[SearchChunk],
) -> tuple[int, int, int, int]:
    """Find page/line location of an excerpt within chunks.

    Matches with the verification rule, so a citation that verifies keeps
    its location.
    """
    for chunk in chunks:
        if excerpt_in_chunks(excerpt, [chunk.chunk]):
            return chunk.page_start, chunk.page_end, chunk.line_start, chunk.line_end
    return 0, 0, 0, 0


def _build_citation_record(
    citation_key: str,
    excerpt: str,
    source_filename: str,
    source_hash: str,
    page_start: int,
    page_end: int,
    line_start: int,
    line_end: int,
    created_at: str,
) -> CitationRecord:
    """Build a single CitationRecord with consistent defaults."""
    return CitationRecord(
        wiki_source="",  # filled by caller
        wiki_chunk_index=0,
        citation_key=citation_key,
        # A footnote definition is always a fact claim; an inference carries none.
        claim_type="fact",
        source_filename=source_filename,
        source_hash=source_hash,
        page_start=page_start,
        page_end=page_end,
        line_start=line_start,
        line_end=line_end,
        excerpt=excerpt,
        created_at=created_at,
    )


def verify_citations(
    citation_records: list[CitationRecord],
    chunks: list[SearchChunk],
    label: str,
    config: Config,
) -> list[CitationRecord]:
    """Filter citation records, keeping only those whose excerpts are in their own source.

    Each record is checked against the chunks of the source it names, the rule
    lint and draft-accept apply. Checking against the whole pool would pass a
    footnote that attributes source B a quote only source A carries, and publish
    it with B's hash and no location. A footnote left without a quotable excerpt
    is unverified and dropped.
    """
    chunk_texts_by_source: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunk_texts_by_source[chunk.source].append(chunk.chunk)
    verified: list[CitationRecord] = []
    for rec in citation_records:
        if _is_wiki_sourced(rec, config):
            log.debug("Skipping wiki-sourced citation %s", rec["citation_key"])
            continue
        if excerpt_in_chunks(rec["excerpt"], chunk_texts_by_source[rec["source_filename"]]):
            verified.append(rec)
        else:
            log.debug(
                "Citation %s excerpt not found in %s (%s), dropping",
                rec["citation_key"],
                rec["source_filename"],
                label,
            )
    return verified


def render_provenance(config: Config, chunks: list[SearchChunk]) -> str:
    """Render the provenance block: chunk references + extraction method.

    Uses ``yaml.safe_dump`` so a chunk source containing a quote, backslash,
    colon, or newline cannot produce invalid YAML that ``parse_frontmatter``
    would silently drop on read.
    """
    block = {
        "provenance": {
            # Record the extractor that actually runs (config mode may fall back),
            # so the audit reflects reality, not the requested setting.
            "extraction_method": effective_entity_mode(config.wiki_entity_mode).value,
            "chunks": [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        }
    }
    return yaml.safe_dump(block, sort_keys=False)


def resolve_multi_source_citations(
    parsed_citations: list[ParsedCitation],
    source_names: list[str],
    source_hashes: dict[str, str],
    chunks_by_source: dict[str, list[SearchChunk]],
) -> list[CitationRecord]:
    """Resolve citations from a synthesis page that cites multiple sources.

    Each citation's source_ref is matched against the source list to determine
    which source document it references. A citation that names no listed source
    and whose excerpt is in none of them is dropped, not attributed to an
    arbitrary source.
    """
    records: list[CitationRecord] = []
    now = datetime.now(UTC).isoformat()

    for parsed in parsed_citations:
        excerpt = _extract_excerpt(parsed.source_ref)

        matched_source = _match_citation_source(
            parsed.source_ref, source_names
        ) or _find_excerpt_source(excerpt, chunks_by_source)
        if not matched_source:
            log.warning(
                "Dropping citation %s: no source matches %r",
                parsed.citation_key,
                parsed.source_ref,
            )
            continue

        search_chunks = chunks_by_source.get(matched_source, [])
        page_start, page_end, line_start, line_end = _find_excerpt_location(excerpt, search_chunks)
        records.append(
            _build_citation_record(
                parsed.citation_key,
                excerpt,
                matched_source,
                source_hashes.get(matched_source, ""),
                page_start,
                page_end,
                line_start,
                line_end,
                now,
            )
        )
    return records


def _match_citation_source(source_ref: str, source_names: list[str]) -> str:
    """Find which source a citation references by matching filenames in the ref.

    Checks longest names first so a filename that is a substring of another
    (e.g. ``doc.md`` within ``mydoc.md``) can't shadow the more specific match.
    """
    for name in sorted(source_names, key=len, reverse=True):
        if name in source_ref:
            return name
    return ""


def _find_excerpt_source(excerpt: str, chunks_by_source: dict[str, list[SearchChunk]]) -> str:
    """Find which source contains a given excerpt, matching as verification does."""
    for source, chunks in chunks_by_source.items():
        if excerpt_in_chunks(excerpt, [c.chunk for c in chunks]):
            return source
    return ""
