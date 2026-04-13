"""Wiki page generation — LLM-driven synthesis with citation provenance.

Generates summary pages (1:1 with sources) and synthesis pages (cross-source,
concept-graph-driven) from raw chunks. Each page carries inline citations
([^srcN]) for facts and [*inference*] markers for LLM synthesis. The
_citations table is the source of truth; markdown footnotes are rendered from it.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from lilbee.clustering import SourceClusterer
from lilbee.config import Config, cfg
from lilbee.ingest import file_hash
from lilbee.providers.base import LLMProvider
from lilbee.reasoning import strip_reasoning
from lilbee.store import CitationRecord, SearchChunk, Store
from lilbee.wiki.citation import ParsedCitation, parse_wiki_citations, render_citation_block
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.shared import (
    DRAFTS_SUBDIR,
    MIN_CLUSTER_SOURCES,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    PageTarget,
    make_slug,
)

log = logging.getLogger(__name__)

WikiProgressCallback = Callable[[str, dict[str, object]], None]
"""Callback for wiki generation progress: (stage, data) -> None."""

_MAX_DIFF_PREVIEW_LINES = 20  # lines of unified diff shown in drift warnings

# Conservative default context window when num_ctx is not configured.
# Most modern models support at least 8192 tokens.
_DEFAULT_CONTEXT_WINDOW = 8192

# Fraction of context window reserved for chunks. The remainder leaves
# room for the system/user prompt template and generation output.
_CONTEXT_BUDGET_FRACTION = 0.75

# Approximate characters per token for budget estimation. 4 chars/token
# is a widely used heuristic for English text.
_CHARS_PER_TOKEN = 4


def _truncate_chunks_to_budget(
    chunks: list[SearchChunk],
    config: Config,
) -> list[SearchChunk]:
    """Drop trailing chunks so the total text fits within the model's context budget.

    Uses a chars/4 heuristic for token estimation. Returns the original list
    unchanged when all chunks fit.
    """
    context_window = config.num_ctx or _DEFAULT_CONTEXT_WINDOW
    budget_tokens = int(context_window * _CONTEXT_BUDGET_FRACTION)
    budget_chars = budget_tokens * _CHARS_PER_TOKEN

    total_chars = 0
    kept: list[SearchChunk] = []
    for chunk in chunks:
        chunk_chars = len(chunk.chunk)
        if total_chars + chunk_chars > budget_chars and kept:
            break
        kept.append(chunk)
        total_chars += chunk_chars

    if len(kept) < len(chunks):
        log.warning(
            "Truncated chunks from %d to %d to fit context window (%d tokens)",
            len(chunks),
            len(kept),
            context_window,
        )
    return kept


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
    if len(lines) > _MAX_DIFF_PREVIEW_LINES:
        extra = len(lines) - _MAX_DIFF_PREVIEW_LINES
        return "\n".join(lines[:_MAX_DIFF_PREVIEW_LINES]) + f"\n... ({extra} more lines)"
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


def _verify_citations(
    citation_records: list[CitationRecord],
    chunks: list[SearchChunk],
    label: str,
    config: Config,
) -> list[CitationRecord]:
    """Filter citation records, keeping only those whose excerpts are in the chunks."""
    wiki_prefix = config.wiki_dir + "/"
    all_chunk_text = " ".join(c.chunk for c in chunks)
    verified: list[CitationRecord] = []
    for rec in citation_records:
        if rec["source_filename"].startswith(wiki_prefix):
            log.debug("Skipping wiki-sourced citation %s", rec["citation_key"])
            continue
        if rec["claim_type"] == "inference" or not rec["excerpt"]:
            verified.append(rec)
            continue
        if rec["excerpt"] in all_chunk_text:
            verified.append(rec)
        else:
            log.debug("Citation %s excerpt not found in %s, dropping", rec["citation_key"], label)
    return verified


def _check_faithfulness(
    chunks_text: str,
    wiki_text: str,
    provider: LLMProvider,
    label: str,
    config: Config | None = None,
) -> float:
    """Run the faithfulness check and return the score (0.0 on failure)."""
    if config is None:
        config = cfg
    template = config.wiki_faithfulness_prompt
    prompt = template.format(chunks_text=chunks_text, wiki_text=wiki_text)
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    try:
        response = provider.chat(messages, stream=False)
        return _parse_faithfulness_score(strip_reasoning(cast(str, response)))
    except Exception as exc:
        log.warning("Faithfulness check failed for %s: %s", label, exc)
        return 0.0


def _build_frontmatter(
    config: Config,
    source_names: list[str],
    score: float,
) -> str:
    """Build YAML frontmatter for a wiki page."""
    sources_yaml = ", ".join(f'"{s}"' for s in sorted(source_names))
    return (
        f"---\n"
        f"generated_by: {config.chat_model}\n"
        f"generated_at: {datetime.now(UTC).isoformat()}\n"
        f"sources: [{sources_yaml}]\n"
        f"faithfulness_score: {score:.2f}\n"
        f"---\n\n"
    )


def _write_page(
    wiki_root: Path,
    subdir: str,
    slug: str,
    full_content: str,
    drift_threshold: float,
) -> Path:
    """Write page to disk with drift detection. Returns path written to."""
    page_dir = wiki_root / subdir
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    if page_path.exists():
        old_content = page_path.read_text(encoding="utf-8")
        ratio = _content_change_ratio(old_content, full_content)
        if ratio > drift_threshold:
            drafts_dir = wiki_root / DRAFTS_SUBDIR
            diff_text = _diff_summary(old_content, full_content)
            return _divert_to_drafts(full_content, drafts_dir, slug, ratio, diff_text)

    page_path.write_text(full_content, encoding="utf-8")
    return page_path


def _assemble_content(
    frontmatter: str,
    wiki_text: str,
    citation_block: str,
) -> str:
    """Combine frontmatter, body, and citations into the full page content."""
    full = frontmatter + wiki_text
    if citation_block:
        full += "\n\n" + citation_block
    return full


def _persist_and_finalize(
    content: str,
    target: PageTarget,
    verified: list[CitationRecord],
    source_names: list[str],
    store: Store,
    config: Config,
) -> Path:
    """Write page to disk, persist citations, update index and log."""
    page_path = _write_page(
        target.wiki_root, target.subdir, target.slug, content, config.wiki_drift_threshold
    )
    for rec in verified:
        rec["wiki_source"] = target.wiki_source
    store.delete_citations_for_wiki(target.wiki_source)
    store.add_citations(verified)

    if config.wiki_prune_raw:
        for name in source_names:
            store.delete_by_source(name)

    update_wiki_index(config)
    append_wiki_log(
        "generated",
        f"{target.page_type} page for {target.label} -> {target.subdir}/{target.slug}.md",
        config,
    )
    return page_path


def _generate_page(
    label: str,
    prompt: str,
    chunks: list[SearchChunk],
    chunks_text: str,
    citation_resolver: Callable[[list[ParsedCitation]], list[CitationRecord]],
    page_type: str,
    slug: str,
    source_names: list[str],
    provider: LLMProvider,
    store: Store,
    config: Config,
    on_progress: WikiProgressCallback | None = None,
) -> Path | None:
    """Core generation pipeline shared by summary and synthesis pages."""

    def _emit(stage: str, **data: object) -> None:
        if on_progress is not None:
            on_progress(stage, data)

    _emit("preparing", chunks=len(chunks), source=label)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    _emit("generating", source=label)
    try:
        response = provider.chat(messages, stream=False)
        wiki_text = strip_reasoning(cast(str, response)).strip()
    except Exception as exc:
        log.warning("LLM failed to generate wiki page for %s: %s", label, exc)
        _emit("failed", error=str(exc))
        return None

    if not wiki_text:
        log.warning("LLM returned empty response for wiki page %s", label)
        _emit("failed", error="Model returned empty response")
        return None

    parsed_citations = parse_wiki_citations(wiki_text)
    verified = _verify_citations(citation_resolver(parsed_citations), chunks, label, config)
    if not verified:
        log.warning("No valid citations for %s, skipping", label)
        return None

    _emit("faithfulness_check")
    score = _check_faithfulness(chunks_text, wiki_text, provider, label, config)
    threshold = config.wiki_faithfulness_threshold
    subdir = page_type if score >= threshold else DRAFTS_SUBDIR
    if subdir == DRAFTS_SUBDIR:
        log.info("Wiki page %s scored %.2f (< %.2f), sending to drafts", label, score, threshold)

    frontmatter = _build_frontmatter(config, source_names, score)
    citation_block = render_citation_block(verified)
    full_content = _assemble_content(frontmatter, wiki_text, citation_block)

    wiki_root = config.data_root / config.wiki_dir
    target = PageTarget(
        wiki_root=wiki_root,
        subdir=subdir,
        slug=slug,
        wiki_source=f"{config.wiki_dir}/{subdir}/{slug}.md",
        page_type=page_type,
        label=label,
    )
    page_path = _persist_and_finalize(full_content, target, verified, source_names, store, config)

    log.info(
        "Generated wiki page for %s -> %s (score=%.2f, citations=%d)",
        label,
        target.subdir,
        score,
        len(verified),
    )
    return page_path


def generate_summary_page(
    source_name: str,
    chunks: list[SearchChunk],
    provider: LLMProvider,
    store: Store,
    config: Config | None = None,
    on_progress: WikiProgressCallback | None = None,
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

    chunks = _truncate_chunks_to_budget(chunks, config)
    chunks_text = _chunks_to_text(chunks)
    template = config.wiki_summary_prompt
    prompt = template.format(source_name=source_name, chunks_text=chunks_text)
    slug = source_name.replace("/", "--").rsplit(".", 1)[0]

    source_path = config.documents_dir / source_name
    source_hash = file_hash(source_path) if source_path.exists() else ""

    def resolver(parsed: list[ParsedCitation]) -> list[CitationRecord]:
        return _resolve_citations(parsed, source_name, source_hash, chunks)

    return _generate_page(
        label=source_name,
        prompt=prompt,
        chunks=chunks,
        chunks_text=chunks_text,
        citation_resolver=resolver,
        page_type=SUMMARIES_SUBDIR,
        slug=slug,
        source_names=[source_name],
        provider=provider,
        store=store,
        config=config,
        on_progress=on_progress,
    )


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
    all_chunks = [c for cs in chunks_by_source.values() for c in cs]
    if not all_chunks:
        log.warning("No chunks for synthesis topic %r, skipping", topic)
        return None

    all_chunks = _truncate_chunks_to_budget(all_chunks, config)
    chunks_text = _chunks_to_text(all_chunks)
    source_list = "\n".join(f"- {name}" for name in sorted(source_names))
    template = config.wiki_synthesis_prompt
    prompt = template.format(topic=topic, source_list=source_list, chunks_text=chunks_text)
    slug = make_slug(topic)

    source_hashes: dict[str, str] = {}
    for name in source_names:
        source_path = config.documents_dir / name
        if source_path.exists():
            source_hashes[name] = file_hash(source_path)

    def resolver(parsed: list[ParsedCitation]) -> list[CitationRecord]:
        return _resolve_multi_source_citations(
            parsed, source_names, source_hashes, chunks_by_source
        )

    return _generate_page(
        label=repr(topic),
        prompt=prompt,
        chunks=all_chunks,
        chunks_text=chunks_text,
        citation_resolver=resolver,
        page_type=SYNTHESIS_SUBDIR,
        slug=slug,
        source_names=source_names,
        provider=provider,
        store=store,
        config=config,
    )


def _generate_for_cluster(
    label: str,
    sources: frozenset[str],
    provider: LLMProvider,
    store: Store,
    config: Config,
) -> Path | None:
    """Gather chunks for a cluster and generate a synthesis page."""
    source_names = sorted(sources)
    chunks_by_source: dict[str, list[SearchChunk]] = {}
    for name in source_names:
        chunks = store.get_chunks_by_source(name)
        if chunks:
            chunks_by_source[name] = chunks

    if len(chunks_by_source) < MIN_CLUSTER_SOURCES:
        return None

    return _generate_synthesis_page(label, source_names, chunks_by_source, provider, store, config)


def generate_synthesis_pages(
    provider: LLMProvider,
    store: Store,
    clusterer: SourceClusterer,
    config: Config | None = None,
) -> list[Path]:
    """Generate synthesis pages for source clusters spanning 3+ documents."""
    if config is None:
        config = cfg

    clusters = clusterer.get_clusters(min_sources=MIN_CLUSTER_SOURCES)
    if not clusters:
        log.info("No source clusters span %d+ sources, skipping synthesis", MIN_CLUSTER_SOURCES)
        return []

    pages: list[Path] = []
    for cluster in clusters:
        page = _generate_for_cluster(cluster.label, cluster.sources, provider, store, config)
        if page is not None:
            pages.append(page)

    log.info("Generated %d synthesis pages", len(pages))
    return pages
