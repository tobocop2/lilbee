"""Citation resolution and verification for wiki pages.

Builds :class:`CitationRecord` rows from parsed ``[^srcN]`` markers,
matches citation excerpts back to the source chunks they came from
(single-source and multi-source variants), verifies the excerpts are
substring-present in the chunk pool, and renders the YAML provenance
block written into a wiki page's frontmatter.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import yaml

from lilbee.core.config import Config
from lilbee.store import CitationRecord, SearchChunk
from lilbee.wiki.cache import _normalize_whitespace
from lilbee.wiki.citation import ParsedCitation

log = logging.getLogger(__name__)

# JSON-style escape sequences that may appear inside quoted excerpts the
# model emits. Any backslash-prefixed character not in this map stays
# verbatim (e.g. ``\\x`` passes through unchanged).
_EXCERPT_ESCAPES: dict[str, str] = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


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
    end = source_ref.find('"', start)
    raw = source_ref[start:].strip() if end == -1 else source_ref[start:end].strip()
    return _decode_excerpt_escapes(raw)


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
    """Find page/line location of an excerpt within chunks."""
    if excerpt:
        for chunk in chunks:
            if excerpt in chunk.chunk:
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
        claim_type="fact" if excerpt else "inference",
        source_filename=source_filename,
        source_hash=source_hash,
        page_start=page_start,
        page_end=page_end,
        line_start=line_start,
        line_end=line_end,
        excerpt=excerpt,
        created_at=created_at,
    )


def _resolve_citations(
    parsed_citations: list[ParsedCitation],
    source_name: str,
    source_hash: str,
    chunks: list[SearchChunk],
) -> list[CitationRecord]:
    """Resolve parsed citation refs to CitationRecord objects.
    Searches for each citation's excerpt in the source chunks to find
    the best matching location (page/line numbers).
    """
    records: list[CitationRecord] = []
    now = datetime.now(UTC).isoformat()

    for parsed in parsed_citations:
        excerpt = _extract_excerpt(parsed.source_ref)
        page_start, page_end, line_start, line_end = _find_excerpt_location(excerpt, chunks)
        records.append(
            _build_citation_record(
                parsed.citation_key,
                excerpt,
                source_name,
                source_hash,
                page_start,
                page_end,
                line_start,
                line_end,
                now,
            )
        )
    return records


def _verify_citations(
    citation_records: list[CitationRecord],
    chunks: list[SearchChunk],
    label: str,
    config: Config,
) -> list[CitationRecord]:
    """Filter citation records, keeping only those whose excerpts are in the chunks."""
    wiki_prefix = config.wiki_dir + "/"
    all_chunk_text = _normalize_whitespace(" ".join(c.chunk for c in chunks))
    verified: list[CitationRecord] = []
    for rec in citation_records:
        if rec["source_filename"].startswith(wiki_prefix):
            log.debug("Skipping wiki-sourced citation %s", rec["citation_key"])
            continue
        if rec["claim_type"] == "inference" or not rec["excerpt"]:
            verified.append(rec)
            continue
        if _normalize_whitespace(rec["excerpt"]) in all_chunk_text:
            verified.append(rec)
        else:
            log.debug("Citation %s excerpt not found in %s, dropping", rec["citation_key"], label)
    return verified


def _render_provenance(config: Config, chunks: list[SearchChunk]) -> str:
    """Render the provenance block: chunk references + extraction method.

    Routes through ``yaml.safe_dump`` rather than hand-rolled string
    formatting so a chunk source containing a quote, backslash,
    colon, or newline does not produce invalid YAML that
    ``parse_frontmatter`` would silently drop on read.
    """
    block = {
        "provenance": {
            "extraction_method": config.wiki_entity_mode.value,
            "chunks": [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        }
    }
    return yaml.safe_dump(block, sort_keys=False)


def _resolve_multi_source_citations(
    parsed_citations: list[ParsedCitation],
    source_names: list[str],
    source_hashes: dict[str, str],
    chunks_by_source: dict[str, list[SearchChunk]],
) -> list[CitationRecord]:
    """Resolve citations from a synthesis page that cites multiple sources.
    Each citation's source_ref is matched against the source list to
    determine which source document it references.
    """
    records: list[CitationRecord] = []
    now = datetime.now(UTC).isoformat()

    all_chunks = [c for cs in chunks_by_source.values() for c in cs]

    for parsed in parsed_citations:
        excerpt = _extract_excerpt(parsed.source_ref)

        matched_source = _match_citation_source(parsed.source_ref, source_names)
        if not matched_source:
            matched_source = _find_excerpt_source(excerpt, chunks_by_source)
        if not matched_source and source_names:
            # No citation match found; default to first listed source
            log.warning(
                "No citation match for chunk — defaulting to first source: %s",
                source_names[0],
            )
            matched_source = source_names[0]

        search_chunks = chunks_by_source.get(matched_source, all_chunks)
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
    """Find which source a citation references by matching filenames in the ref."""
    for name in source_names:
        if name in source_ref:
            return name
    return ""


def _find_excerpt_source(excerpt: str, chunks_by_source: dict[str, list[SearchChunk]]) -> str:
    """Find which source contains a given excerpt by searching chunks."""
    if not excerpt:
        return ""
    for source, chunks in chunks_by_source.items():
        for chunk in chunks:
            if excerpt in chunk.chunk:
                return source
    return ""
