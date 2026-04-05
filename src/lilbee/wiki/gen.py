"""Wiki page generation — LLM-driven synthesis with citation provenance.

Generates summary pages (1:1 with sources) and synthesis pages (cross-source,
concept-graph-driven) from raw chunks. Each page carries inline citations
([^srcN]) for facts and [*inference*] markers for LLM synthesis. The
_citations table is the source of truth; markdown footnotes are rendered from it.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from lilbee.config import Config, cfg
from lilbee.providers.base import LLMProvider
from lilbee.store import CitationRecord, SearchChunk, Store
from lilbee.wiki.citation import ParsedCitation, parse_wiki_citations, render_citation_block

log = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are a knowledge compiler. Given the source chunks below from a single document, \
write a concise wiki summary page in markdown.

Rules:
1. Every factual claim MUST have an inline citation [^src1], [^src2], etc.
2. Cite the EXACT text from the source that supports each claim by quoting it.
3. For interpretations or connections not directly stated in the source, mark with [*inference*].
4. Use blockquotes (>) for directly cited facts.
5. End with a citation block in this format:

---
<!-- citations (auto-generated from _citations table -- do not edit) -->
[^src1]: {source_name}, excerpt: "exact quoted text"
[^src2]: {source_name}, excerpt: "exact quoted text"

Source document: {source_name}

Chunks:
{chunks_text}

Write the wiki summary page now. Start with a heading."""

_FAITHFULNESS_PROMPT = """\
You are a fact-checker. Given source chunks and a wiki summary page generated from them, \
score the summary's faithfulness to the sources on a scale of 0.0 to 1.0.

Criteria:
- 1.0 = every claim is directly supported by the source chunks
- 0.5 = some claims are supported, some are unsupported extrapolations
- 0.0 = the summary contains fabricated information

Source chunks:
{chunks_text}

Wiki summary:
{wiki_text}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

_SYNTHESIS_PROMPT = """\
You are a knowledge compiler. Given source chunks from MULTIPLE documents about \
related concepts, write a synthesis wiki page in markdown that connects ideas \
across sources.

Rules:
1. Every factual claim MUST have an inline citation [^src1], [^src2], etc.
2. Cite the EXACT text from the source that supports each claim by quoting it.
3. For connections, interpretations, or patterns you identify across sources, \
mark with [*inference*].
4. Use blockquotes (>) for directly cited facts.
5. Reference each source by its filename when drawing connections.
6. End with a citation block in this format:

---
<!-- citations (auto-generated from _citations table -- do not edit) -->
[^src1]: {{source_name}}, excerpt: "exact quoted text"
[^src2]: {{source_name}}, excerpt: "exact quoted text"

Topic: {topic}

Sources:
{source_list}

Chunks:
{chunks_text}

Write the synthesis page now. Start with a heading."""


def _chunks_to_text(chunks: list[SearchChunk]) -> str:
    """Format chunks as numbered text blocks for the LLM prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        location = ""
        if chunk.page_start:
            location = f" (page {chunk.page_start})"
        elif chunk.line_start:
            location = f" (lines {chunk.line_start}-{chunk.line_end})"
        parts.append(f"[Chunk {i + 1}]{location}:\n{chunk.chunk}")
    return "\n\n".join(parts)


def _parse_faithfulness_score(response: str) -> float:
    """Extract a float score from the LLM's faithfulness response."""
    text = response.strip()
    for line in text.splitlines():
        line = line.strip()
        try:
            score = float(line)
            return max(0.0, min(1.0, score))
        except ValueError:
            continue
    log.warning("Could not parse faithfulness score from: %r", text[:100])
    return 0.0


def _extract_excerpt(source_ref: str) -> str:
    """Extract the quoted excerpt from a citation source_ref string.

    e.g. 'doc.md, excerpt: "Python supports typing."' → 'Python supports typing.'
    """
    marker = 'excerpt: "'
    idx = source_ref.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = source_ref.find('"', start)
    if end == -1:
        return source_ref[start:].strip()
    return source_ref[start:end].strip()


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
        citation_key = parsed.citation_key
        excerpt = _extract_excerpt(parsed.source_ref)

        page_start, page_end, line_start, line_end = 0, 0, 0, 0
        if excerpt:
            for chunk in chunks:
                if excerpt in chunk.chunk:
                    page_start = chunk.page_start
                    page_end = chunk.page_end
                    line_start = chunk.line_start
                    line_end = chunk.line_end
                    break

        records.append(
            CitationRecord(
                wiki_source="",  # filled by caller
                wiki_chunk_index=0,
                citation_key=citation_key,
                claim_type="fact" if excerpt else "inference",
                source_filename=source_name,
                source_hash=source_hash,
                page_start=page_start,
                page_end=page_end,
                line_start=line_start,
                line_end=line_end,
                excerpt=excerpt,
                created_at=now,
            )
        )
    return records


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _content_change_ratio(old_text: str, new_text: str) -> float:
    """Fraction of lines that changed between two texts (0.0 = identical, 1.0 = total rewrite)."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    if not old_lines and not new_lines:
        return 0.0
    total = max(len(old_lines), len(new_lines))
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed = total - sum(block.size for block in matcher.get_matching_blocks())
    return changed / total


def _diff_summary(old_text: str, new_text: str) -> str:
    """Human-readable unified diff summary (first 20 diff lines)."""
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        lineterm="",
        fromfile="old",
        tofile="new",
    )
    lines = list(diff)
    if len(lines) > 20:
        return "\n".join(lines[:20]) + f"\n... ({len(lines) - 20} more lines)"
    return "\n".join(lines)


def _divert_to_drafts(
    new_content: str,
    drafts_dir: Path,
    slug: str,
    change_ratio: float,
    diff_text: str,
) -> Path:
    """Write new content to wiki/drafts/ with a drift note instead of overwriting."""
    drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = drafts_dir / f"{slug}.md"
    note = f"<!-- DRIFT: {change_ratio:.0%} content changed — flagged for human review -->\n\n"
    draft_path.write_text(note + new_content, encoding="utf-8")
    log.warning(
        "Drift detected for %s (%.0f%% changed), diverted to drafts. Diff:\n%s",
        slug,
        change_ratio * 100,
        diff_text,
    )
    return draft_path


def generate_summary_page(
    source_name: str,
    chunks: list[SearchChunk],
    provider: LLMProvider,
    store: Store,
    config: Config | None = None,
) -> Path | None:
    """Generate a wiki summary page for a source document.

    Returns the path to the generated page, or None on failure.
    The page goes to wiki/summaries/ if it passes the quality gate,
    or wiki/drafts/ if it fails.
    """
    if config is None:
        config = cfg
    if not chunks:
        log.warning("No chunks for source %s, skipping wiki generation", source_name)
        return None

    wiki_root = config.data_root / config.wiki_dir
    chunks_text = _chunks_to_text(chunks)

    # Step 1: Generate the wiki page
    prompt = _SUMMARY_PROMPT.format(source_name=source_name, chunks_text=chunks_text)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        response = provider.chat(messages, stream=False)
        wiki_text = cast(str, response).strip()
    except Exception:
        log.warning("LLM failed to generate wiki page for %s", source_name, exc_info=True)
        return None

    if not wiki_text:
        return None

    # Step 2: Parse citations from the generated page
    parsed_citations = parse_wiki_citations(wiki_text)

    # Step 3: Resolve citations to source locations
    source_path = config.documents_dir / source_name
    source_hash = _file_hash(source_path) if source_path.exists() else ""
    citation_records = _resolve_citations(parsed_citations, source_name, source_hash, chunks)

    # Step 4: Verify excerpts exist in source chunks
    all_chunk_text = " ".join(c.chunk for c in chunks)
    verified = []
    for rec in citation_records:
        if rec["claim_type"] == "inference" or not rec["excerpt"]:
            verified.append(rec)
            continue
        if rec["excerpt"] in all_chunk_text:
            verified.append(rec)
        else:
            log.debug("Citation %s excerpt not found in source, dropping", rec["citation_key"])

    if not verified:
        log.warning("No valid citations for %s, skipping", source_name)
        return None

    # Step 5: Faithfulness check
    faith_prompt = _FAITHFULNESS_PROMPT.format(chunks_text=chunks_text, wiki_text=wiki_text)
    faith_messages: list[dict[str, Any]] = [{"role": "user", "content": faith_prompt}]
    try:
        faith_response = provider.chat(faith_messages, stream=False)
        score = _parse_faithfulness_score(cast(str, faith_response))
    except Exception:
        log.warning("Faithfulness check failed for %s, using 0.0", source_name, exc_info=True)
        score = 0.0

    # Step 6: Determine output directory
    threshold = config.wiki_faithfulness_threshold
    if score >= threshold:
        subdir = "summaries"
    else:
        subdir = "drafts"
        log.info(
            "Wiki page for %s scored %.2f (< %.2f), sending to drafts",
            source_name,
            score,
            threshold,
        )

    # Step 7: Build frontmatter and write page
    slug = source_name.replace("/", "--").rsplit(".", 1)[0]
    page_dir = wiki_root / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    frontmatter = (
        f"---\n"
        f"generated_by: {config.chat_model}\n"
        f"generated_at: {datetime.now(UTC).isoformat()}\n"
        f"sources: [{source_name}]\n"
        f"faithfulness_score: {score:.2f}\n"
        f"---\n\n"
    )

    # Render citation footnotes from records
    citation_block = render_citation_block(verified)
    full_content = frontmatter + wiki_text
    if citation_block:
        full_content += "\n\n" + citation_block

    # Step 7b: Drift detection — if existing page changed too much, divert to drafts
    if page_path.exists():
        old_content = page_path.read_text(encoding="utf-8")
        ratio = _content_change_ratio(old_content, full_content)
        if ratio > config.wiki_drift_threshold:
            drafts_dir = wiki_root / "drafts"
            diff_text = _diff_summary(old_content, full_content)
            return _divert_to_drafts(full_content, drafts_dir, slug, ratio, diff_text)

    page_path.write_text(full_content, encoding="utf-8")

    # Step 8: Write citation records to store
    wiki_source = f"{config.wiki_dir}/{subdir}/{slug}.md"
    for rec in verified:
        rec["wiki_source"] = wiki_source
    store.add_citations(verified)

    log.info(
        "Generated wiki page for %s -> %s (score=%.2f, citations=%d)",
        source_name,
        subdir,
        score,
        len(verified),
    )

    from lilbee.wiki.index import append_wiki_log, update_wiki_index

    update_wiki_index(config)
    append_wiki_log("generated", f"summary page for {source_name} -> {subdir}/{slug}.md", config)

    return page_path


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
            matched_source = source_names[0]

        page_start, page_end, line_start, line_end = 0, 0, 0, 0
        if excerpt:
            search_chunks = chunks_by_source.get(matched_source, all_chunks)
            for chunk in search_chunks:
                if excerpt in chunk.chunk:
                    page_start = chunk.page_start
                    page_end = chunk.page_end
                    line_start = chunk.line_start
                    line_end = chunk.line_end
                    break

        records.append(
            CitationRecord(
                wiki_source="",
                wiki_chunk_index=0,
                citation_key=parsed.citation_key,
                claim_type="fact" if excerpt else "inference",
                source_filename=matched_source,
                source_hash=source_hashes.get(matched_source, ""),
                page_start=page_start,
                page_end=page_end,
                line_start=line_start,
                line_end=line_end,
                excerpt=excerpt,
                created_at=now,
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


def _make_slug(label: str) -> str:
    """Turn a concept label into a filesystem-safe slug."""
    return label.lower().replace(" ", "-").replace("/", "--")


def _generate_synthesis_page(
    topic: str,
    source_names: list[str],
    chunks_by_source: dict[str, list[SearchChunk]],
    provider: LLMProvider,
    store: Store,
    config: Config,
) -> Path | None:
    """Generate a single synthesis page for a concept cluster.

    Returns the path to the generated page, or None on failure.
    """
    wiki_root = config.data_root / config.wiki_dir

    all_chunks = [c for cs in chunks_by_source.values() for c in cs]
    if not all_chunks:
        log.warning("No chunks for synthesis topic %r, skipping", topic)
        return None

    chunks_text = _chunks_to_text(all_chunks)
    source_list = "\n".join(f"- {name}" for name in sorted(source_names))

    prompt = _SYNTHESIS_PROMPT.format(topic=topic, source_list=source_list, chunks_text=chunks_text)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        response = provider.chat(messages, stream=False)
        wiki_text = cast(str, response).strip()
    except Exception:
        log.warning("LLM failed to generate synthesis page for %r", topic, exc_info=True)
        return None

    if not wiki_text:
        return None

    parsed_citations = parse_wiki_citations(wiki_text)

    source_hashes: dict[str, str] = {}
    for name in source_names:
        source_path = config.documents_dir / name
        if source_path.exists():
            source_hashes[name] = _file_hash(source_path)

    citation_records = _resolve_multi_source_citations(
        parsed_citations, source_names, source_hashes, chunks_by_source
    )

    all_chunk_text = " ".join(c.chunk for c in all_chunks)
    verified = []
    for rec in citation_records:
        if rec["claim_type"] == "inference" or not rec["excerpt"]:
            verified.append(rec)
            continue
        if rec["excerpt"] in all_chunk_text:
            verified.append(rec)
        else:
            log.debug("Citation %s excerpt not found in sources, dropping", rec["citation_key"])

    if not verified:
        log.warning("No valid citations for synthesis topic %r, skipping", topic)
        return None

    faith_prompt = _FAITHFULNESS_PROMPT.format(chunks_text=chunks_text, wiki_text=wiki_text)
    faith_messages: list[dict[str, Any]] = [{"role": "user", "content": faith_prompt}]
    try:
        faith_response = provider.chat(faith_messages, stream=False)
        score = _parse_faithfulness_score(cast(str, faith_response))
    except Exception:
        log.warning("Faithfulness check failed for %r, using 0.0", topic, exc_info=True)
        score = 0.0

    threshold = config.wiki_faithfulness_threshold
    if score >= threshold:
        subdir = "concepts"
    else:
        subdir = "drafts"
        log.info(
            "Synthesis page %r scored %.2f (< %.2f), sending to drafts",
            topic,
            score,
            threshold,
        )

    slug = _make_slug(topic)
    page_dir = wiki_root / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    sources_yaml = ", ".join(sorted(source_names))
    frontmatter = (
        f"---\n"
        f"generated_by: {config.chat_model}\n"
        f"generated_at: {datetime.now(UTC).isoformat()}\n"
        f"sources: [{sources_yaml}]\n"
        f"faithfulness_score: {score:.2f}\n"
        f"---\n\n"
    )

    citation_block = render_citation_block(verified)
    full_content = frontmatter + wiki_text
    if citation_block:
        full_content += "\n\n" + citation_block

    # Drift detection — if existing page changed too much, divert to drafts
    if page_path.exists():
        old_content = page_path.read_text(encoding="utf-8")
        ratio = _content_change_ratio(old_content, full_content)
        if ratio > config.wiki_drift_threshold:
            drafts_dir = wiki_root / "drafts"
            diff_text = _diff_summary(old_content, full_content)
            return _divert_to_drafts(full_content, drafts_dir, slug, ratio, diff_text)

    page_path.write_text(full_content, encoding="utf-8")

    wiki_source = f"{config.wiki_dir}/{subdir}/{slug}.md"
    for rec in verified:
        rec["wiki_source"] = wiki_source
    store.add_citations(verified)

    log.info(
        "Generated synthesis page %r -> %s (score=%.2f, citations=%d, sources=%d)",
        topic,
        subdir,
        score,
        len(verified),
        len(source_names),
    )

    from lilbee.wiki.index import append_wiki_log, update_wiki_index

    update_wiki_index(config)
    append_wiki_log("generated", f"synthesis page for {topic!r} -> {subdir}/{slug}.md", config)

    return page_path


def generate_synthesis_pages(
    provider: LLMProvider,
    store: Store,
    config: Config | None = None,
) -> list[Path]:
    """Generate synthesis pages for concept clusters spanning 3+ sources.

    Reads the concept graph to find qualifying clusters, gathers chunks
    from all cluster sources, and generates a synthesis page for each.
    Returns paths to all generated pages (concepts/ or drafts/).
    """
    from lilbee.concepts import ConceptGraph

    if config is None:
        config = cfg

    graph = ConceptGraph(config, store)
    cluster_sources = graph.get_cluster_sources(min_sources=3)
    if not cluster_sources:
        log.info("No concept clusters span 3+ sources, skipping synthesis")
        return []

    pages: list[Path] = []
    for cluster_id, sources in cluster_sources.items():
        topic = graph.get_cluster_label(cluster_id)
        source_names = sorted(sources)

        chunks_by_source: dict[str, list[SearchChunk]] = {}
        for name in source_names:
            chunks = store.get_chunks_by_source(name)
            if chunks:
                chunks_by_source[name] = chunks

        if len(chunks_by_source) < 3:
            continue

        page = _generate_synthesis_page(
            topic, source_names, chunks_by_source, provider, store, config
        )
        if page is not None:
            pages.append(page)

    log.info("Generated %d synthesis pages", len(pages))
    return pages
