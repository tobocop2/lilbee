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
import re
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
from lilbee.wiki.citation import (
    ParsedCitation,
    parse_wiki_citations,
    render_citation_block,
    strip_citation_block,
)
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.shared import (
    DRAFTS_SUBDIR,
    MIN_CLUSTER_SOURCES,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    WIKI_LOG_ACTION_GENERATED,
    WIKI_LOG_ACTION_REDUCED,
    PageTarget,
    make_slug,
    parse_frontmatter,
)
from lilbee.wiki.structure import (
    WikiNode,
    deserialize_document,
    walk_structure_to_wiki_nodes,
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

# Width (in characters) of the zero-padded page number inside a per-page slug.
_PAGE_SLUG_WIDTH = 4

# Directive recognized by chat templates that support a reasoning mode
# (Qwen3, DeepSeek-R1, etc.). Wiki generation is a summarization task
# where chain-of-thought adds wall-clock cost without improving output,
# so we suppress it whenever the provider reports the capability.
_NO_THINK_DIRECTIVE = "/no_think"
_THINKING_CAPABILITY = "thinking"


def _build_wiki_messages(
    prompt: str, provider: LLMProvider, config: Config
) -> list[dict[str, str]]:
    """Build the chat messages list for a wiki-gen call.

    When the provider reports the ``thinking`` capability for the active
    chat model, prepends ``/no_think`` so the chat template disables the
    reasoning mode. Otherwise the prompt passes through unchanged.
    """
    capabilities = provider.get_capabilities(config.chat_model)
    if _THINKING_CAPABILITY in capabilities:
        prompt = f"{_NO_THINK_DIRECTIVE}\n\n{prompt}"
    return [{"role": "user", "content": prompt}]


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


def _group_chunks_by_page(
    chunks: list[SearchChunk],
) -> list[tuple[int, list[SearchChunk]]]:
    """Group chunks by ``page_start``, preserving in-document order within a page.

    Returns ``(page_start, chunks)`` tuples sorted ascending by page number.
    Chunks with ``page_start=0`` (non-paginated sources) collapse to a single
    entry keyed at 0, so a markdown or code source still emits exactly one
    summary file until structure detection arrives in a later stage.
    """
    grouped: dict[int, list[SearchChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.page_start, []).append(chunk)
    return sorted(grouped.items())


def _page_slug(source_slug: str, page_num: int) -> str:
    """Build a per-page slug of the form ``<source>/page-NNNN``."""
    return f"{source_slug}/page-{page_num:0{_PAGE_SLUG_WIDTH}d}"


def _leaf_hash(chunks: list[SearchChunk]) -> str:
    """SHA-256 over concatenated chunk content (null-separated, in given order).

    Acts as the cache key for incremental rebuild: an existing page whose
    frontmatter ``leaf_hash`` matches this value has already summarized the
    exact same input and can be reused without a new LLM call.
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.chunk.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _find_cached_leaf(wiki_root: Path, slug: str, leaf_hash: str) -> Path | None:
    """Return an existing page whose ``leaf_hash`` frontmatter matches, or ``None``.

    Checks both ``summaries/`` and ``drafts/`` so an unchanged draft stays in
    drafts rather than triggering a speculative regeneration.
    """
    for subdir in (SUMMARIES_SUBDIR, DRAFTS_SUBDIR):
        candidate = wiki_root / subdir / f"{slug}.md"
        if not candidate.is_file():
            continue
        fm = parse_frontmatter(candidate.read_text(encoding="utf-8"))
        if fm.get("leaf_hash") == leaf_hash:
            return candidate
    return None


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
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            mapped = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt)
            if mapped is not None:
                result.append(mapped)
                i += 2
                continue
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
    draft_path = drafts_dir / f"{slug}.md"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    note = f"<!-- DRIFT: {change_ratio:.0%} content changed - flagged for human review -->\n\n"
    draft_path.write_text(note + new_content, encoding="utf-8")
    log.warning(
        "Drift detected for %s (%.0f%% changed), diverted to drafts. Diff:\n%s",
        slug,
        change_ratio * 100,
        diff_text,
    )
    return draft_path


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip the edges.

    PDF extractors preserve line breaks mid-sentence (``vehicle,\\nthe greater``)
    while LLMs paraphrase the same quote as a single-spaced string
    (``vehicle, the greater``). A strict substring check rejects a faithful
    citation on whitespace alone, so both sides are normalized before
    comparison.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


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
    messages = _build_wiki_messages(prompt, provider, config)
    options = config.generation_options(
        temperature=config.wiki_temperature,
        max_tokens=config.wiki_faithfulness_max_tokens,
    )
    try:
        response = provider.chat(messages, stream=False, options=options)
        return _parse_faithfulness_score(strip_reasoning(cast(str, response)))
    except Exception as exc:
        log.warning("Faithfulness check failed for %s: %s", label, exc)
        return 0.0


def _build_frontmatter(
    config: Config,
    source_names: list[str],
    score: float,
    leaf_hash: str = "",
) -> str:
    """Build YAML frontmatter for a wiki page.

    When ``leaf_hash`` is non-empty it is written so incremental rebuild can
    skip regeneration on a subsequent sync whose chunks produce the same hash.
    """
    sources_yaml = ", ".join(f'"{s}"' for s in sorted(source_names))
    hash_line = f"leaf_hash: {leaf_hash}\n" if leaf_hash else ""
    return (
        f"---\n"
        f"generated_by: {config.chat_model}\n"
        f"generated_at: {datetime.now(UTC).isoformat()}\n"
        f"sources: [{sources_yaml}]\n"
        f"faithfulness_score: {score:.2f}\n"
        f"{hash_line}"
        f"---\n\n"
    )


def _write_page(
    wiki_root: Path,
    subdir: str,
    slug: str,
    full_content: str,
    drift_threshold: float,
) -> Path:
    """Write page to disk with drift detection. Returns path written to.

    ``slug`` may contain forward slashes (e.g. ``cv-manual/page-0042``);
    any intermediate directories are created before writing.
    """
    page_path = wiki_root / subdir / f"{slug}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)

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
        WIKI_LOG_ACTION_GENERATED,
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
    leaf_hash: str = "",
) -> Path | None:
    """Core generation pipeline shared by summary and synthesis pages."""

    def _emit(stage: str, **data: object) -> None:
        if on_progress is not None:
            on_progress(stage, data)

    _emit("preparing", chunks=len(chunks), source=label)

    messages = _build_wiki_messages(prompt, provider, config)
    _emit("generating", source=label)
    options = config.generation_options(
        temperature=config.wiki_temperature,
        max_tokens=config.wiki_summary_max_tokens,
    )
    try:
        response = provider.chat(messages, stream=False, options=options)
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
        _emit("failed", error="No valid citations found")
        return None

    _emit("faithfulness_check")
    score = _check_faithfulness(chunks_text, wiki_text, provider, label, config)
    threshold = config.wiki_faithfulness_threshold
    subdir = page_type if score >= threshold else DRAFTS_SUBDIR
    if subdir == DRAFTS_SUBDIR:
        log.info("Wiki page %s scored %.2f (< %.2f), sending to drafts", label, score, threshold)

    wiki_text = strip_citation_block(wiki_text)
    frontmatter = _build_frontmatter(config, source_names, score, leaf_hash)
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


def _source_slug(source_name: str) -> str:
    """Filesystem-safe slug for a source: drop extension, replace ``/`` with ``--``."""
    return source_name.replace("/", "--").rsplit(".", 1)[0]


def _generate_page_for_leaf(
    source_name: str,
    source_hash: str,
    page_num: int,
    page_chunks: list[SearchChunk],
    source_slug: str,
    provider: LLMProvider,
    store: Store,
    config: Config,
    on_progress: WikiProgressCallback | None,
) -> Path | None:
    """Generate a single per-page leaf summary. Returns ``None`` if the page fails."""
    slug = _page_slug(source_slug, page_num)
    leaf_hash = _leaf_hash(page_chunks)

    wiki_root = config.data_root / config.wiki_dir
    cached = _find_cached_leaf(wiki_root, slug, leaf_hash)
    if cached is not None:
        log.info("Wiki leaf %s unchanged (hash match), skipping", slug)
        if on_progress is not None:
            on_progress("cached", {"source": f"{source_name} p{page_num}", "slug": slug})
        return cached

    trimmed = _truncate_chunks_to_budget(page_chunks, config)
    chunks_text = _chunks_to_text(trimmed)
    prompt = config.wiki_summary_prompt.format(source_name=source_name, chunks_text=chunks_text)

    def resolver(
        parsed: list[ParsedCitation],
        _chunks: list[SearchChunk] = trimmed,
        _hash: str = source_hash,
    ) -> list[CitationRecord]:
        return _resolve_citations(parsed, source_name, _hash, _chunks)

    return _generate_page(
        label=f"{source_name} p{page_num}",
        prompt=prompt,
        chunks=trimmed,
        chunks_text=chunks_text,
        citation_resolver=resolver,
        page_type=SUMMARIES_SUBDIR,
        slug=slug,
        source_names=[source_name],
        provider=provider,
        store=store,
        config=config,
        on_progress=on_progress,
        leaf_hash=leaf_hash,
    )


def generate_summary_page(
    source_name: str,
    chunks: list[SearchChunk],
    provider: LLMProvider,
    store: Store,
    config: Config | None = None,
    on_progress: WikiProgressCallback | None = None,
) -> list[Path]:
    """Generate per-page wiki summary leaves for a source document.

    Groups chunks by ``page_start`` and writes one summary per page at
    ``wiki/summaries/<source>/page-NNNN.md``. Pages that miss the
    faithfulness threshold are diverted to ``wiki/drafts/<source>/`` with
    the same filename. Returns the list of generated paths (empty if the
    source has no chunks or every page fails).
    """
    if config is None:
        config = cfg
    if not chunks:
        log.warning("No chunks for source %s, skipping wiki generation", source_name)
        return []

    source_slug = _source_slug(source_name)
    source_path = config.documents_dir / source_name
    source_hash = file_hash(source_path) if source_path.exists() else ""

    pages: list[Path] = []
    leaf_paths_by_page: dict[int, Path] = {}
    for page_num, page_chunks in _group_chunks_by_page(chunks):
        page_path = _generate_page_for_leaf(
            source_name=source_name,
            source_hash=source_hash,
            page_num=page_num,
            page_chunks=page_chunks,
            source_slug=source_slug,
            provider=provider,
            store=store,
            config=config,
            on_progress=on_progress,
        )
        if page_path is not None:
            pages.append(page_path)
            leaf_paths_by_page[page_num] = page_path

    inner_paths = _generate_inner_nodes(
        source_name=source_name,
        source_slug=source_slug,
        chunks=chunks,
        leaf_paths_by_page=leaf_paths_by_page,
        provider=provider,
        store=store,
        config=config,
        on_progress=on_progress,
    )
    pages.extend(inner_paths)

    return pages


def _chunks_in_page_range(
    chunks: list[SearchChunk], page_start: int, page_end: int
) -> list[SearchChunk]:
    """Return chunks whose page_start falls within [page_start, page_end] inclusive.

    Page ranges on WikiNodes are inclusive at both ends; a chunk with page_start
    matching either endpoint is considered inside the range.
    """
    if page_end < page_start:
        return []
    return [c for c in chunks if page_start <= c.page_start <= page_end]


def _leaves_in_range(
    leaf_paths_by_page: dict[int, Path], page_start: int, page_end: int
) -> list[Path]:
    """Return leaf page paths whose page number falls within [page_start, page_end]."""
    if page_end < page_start:
        return []
    return [
        leaf_paths_by_page[p] for p in sorted(leaf_paths_by_page) if page_start <= p <= page_end
    ]


def _read_page_body(path: Path) -> str:
    """Return the page body with any leading YAML frontmatter stripped."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return text


def _format_children_for_reduce(children_paths: list[Path]) -> str:
    """Concatenate child page bodies for inclusion in the reduce prompt."""
    blocks: list[str] = []
    for path in children_paths:
        body = _read_page_body(path).strip()
        if not body:
            continue
        blocks.append(f"[From {path.stem}]\n{body}")
    return "\n\n---\n\n".join(blocks)


def _load_document_structure(store: Store, source_name: str) -> list[WikiNode]:
    """Load and walk the persisted DocumentStructure tree for *source_name*.

    Returns an empty list when no structure is stored or the JSON blob is
    unparseable, so callers can treat "no structure" and "empty structure"
    identically.
    """
    record = store.get_document_structure(source_name)
    # isinstance guard handles mocked stores in tests that don't stage a
    # structure fixture; they return MagicMock sentinels instead of None.
    if not isinstance(record, dict):
        return []
    payload = record.get("document_json")
    if not isinstance(payload, str):
        return []
    document = deserialize_document(payload)
    if document is None:
        return []
    return walk_structure_to_wiki_nodes(document)


def _wiki_nodes_bottom_up(nodes: list[WikiNode]) -> list[WikiNode]:
    """Order wiki nodes so children come before parents (deepest first)."""
    return sorted(nodes, key=lambda n: -n.depth)


def _is_draft_path(path: Path, wiki_root: Path) -> bool:
    """True if *path* lives under wiki_root/drafts/."""
    try:
        relative = path.relative_to(wiki_root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == DRAFTS_SUBDIR


def _inner_node_target(wiki_root: Path, subdir: str, source_slug: str, node_slug: str) -> Path:
    """Return the ``index.md`` path for an inner wiki node."""
    return wiki_root / subdir / source_slug / node_slug / "index.md"


def _build_inner_frontmatter(
    config: Config,
    source_name: str,
    kind: str,
    score: float,
    partial: bool,
    partial_paths: list[str],
) -> str:
    """YAML frontmatter for a section/chapter/root index page."""
    partial_yaml = ", ".join(f'"{p}"' for p in partial_paths)
    lines = [
        "---",
        f"generated_by: {config.chat_model}",
        f"generated_at: {datetime.now(UTC).isoformat()}",
        f'sources: ["{source_name}"]',
        f"faithfulness_score: {score:.2f}",
        f"kind: {kind}",
    ]
    if partial:
        lines.append("partial: true")
        lines.append(f"partial_paths: [{partial_yaml}]")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n"


def _generate_inner_node(
    node: WikiNode,
    source_name: str,
    source_slug: str,
    children_paths: list[Path],
    raw_chunks: list[SearchChunk],
    partial_paths: list[str],
    provider: LLMProvider,
    config: Config,
    on_progress: WikiProgressCallback | None,
) -> Path | None:
    """Reduce child summaries into a single node summary, grounded against raw chunks.

    Writes ``wiki/summaries/<source>/<node.slug>/index.md`` on success or routes
    to ``wiki/drafts/<source>/<node.slug>/index.md`` when the faithfulness score
    falls below ``cfg.wiki_faithfulness_threshold``. Returns ``None`` when there
    is nothing to summarize (no children, no chunks) or the LLM call fails.
    """
    if not children_paths:
        return None

    def _emit(stage: str, **data: object) -> None:
        if on_progress is not None:
            on_progress(stage, data)

    label = f"{source_name} :: {node.slug}"
    _emit("preparing", source=label, kind=node.kind, children=len(children_paths))

    children_text = _format_children_for_reduce(children_paths)
    if not children_text:
        return None

    prompt = config.wiki_reduce_prompt.format(
        source_name=source_name,
        section_title=node.title,
        children_text=children_text,
    )

    _emit("generating", source=label)
    options = config.generation_options(
        temperature=config.wiki_temperature,
        max_tokens=config.wiki_summary_max_tokens,
    )
    try:
        response = provider.chat(
            _build_wiki_messages(prompt, provider, config), stream=False, options=options
        )
        summary_text = strip_reasoning(cast(str, response)).strip()
    except Exception as exc:
        log.warning("LLM failed to reduce %s: %s", label, exc)
        _emit("failed", error=str(exc))
        return None
    if not summary_text:
        _emit("failed", error="Model returned empty response")
        return None

    _emit("faithfulness_check")
    range_chunks = _chunks_in_page_range(raw_chunks, node.page_start, node.page_end)
    if range_chunks:
        trimmed = _truncate_chunks_to_budget(range_chunks, config)
        chunks_text = _chunks_to_text(trimmed)
        score = _check_faithfulness(chunks_text, summary_text, provider, label, config)
    else:
        # No raw chunks in this range (e.g. all descendants were drafted).
        # Fall back to grounding against children's summaries rather than
        # scoring 0 and routing every partial parent to drafts.
        score = _check_faithfulness(children_text, summary_text, provider, label, config)

    threshold = config.wiki_faithfulness_threshold
    subdir = SUMMARIES_SUBDIR if score >= threshold else DRAFTS_SUBDIR
    if subdir == DRAFTS_SUBDIR:
        log.info("Inner node %s scored %.2f (< %.2f), sending to drafts", label, score, threshold)

    wiki_root = config.data_root / config.wiki_dir
    partial = bool(partial_paths)
    frontmatter = _build_inner_frontmatter(
        config, source_name, node.kind, score, partial, partial_paths
    )
    target = _inner_node_target(wiki_root, subdir, source_slug, node.slug)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary_text = strip_citation_block(summary_text)
    target.write_text(frontmatter + summary_text + "\n", encoding="utf-8")

    append_wiki_log(
        WIKI_LOG_ACTION_REDUCED,
        f"{node.kind} {source_name}/{node.slug} -> {subdir}/...",
        config,
    )
    _emit("done", source=label, score=score, partial=partial)
    return target


def _collect_inner_children(
    node: WikiNode,
    all_nodes: list[WikiNode],
    node_paths: dict[str, Path],
    leaf_paths_by_page: dict[int, Path],
) -> tuple[list[Path], list[str]]:
    """Find the immediate child summaries for *node* and any drafted descendants.

    For a node with nested wiki children, returns those children's generated
    paths. For a node with no wiki children of its own (typical for the deepest
    heading that directly contains pages), returns the leaf paths whose page
    number falls in the node's page range.

    The second return value is the list of drafted descendant slugs, used to
    mark the parent as partial in its frontmatter.
    """
    wiki_children = [n for n in all_nodes if n.parent_slug == node.slug]
    partial_paths: list[str] = []

    if wiki_children:
        ordered = sorted(wiki_children, key=lambda n: n.ordinal)
        paths: list[Path] = []
        for child in ordered:
            p = node_paths.get(child.slug)
            if p is None:
                partial_paths.append(child.slug)
                continue
            paths.append(p)
        return paths, partial_paths

    leaf_paths = _leaves_in_range(leaf_paths_by_page, node.page_start, node.page_end)
    return leaf_paths, partial_paths


def _generate_inner_nodes(
    source_name: str,
    source_slug: str,
    chunks: list[SearchChunk],
    leaf_paths_by_page: dict[int, Path],
    provider: LLMProvider,
    store: Store,
    config: Config,
    on_progress: WikiProgressCallback | None,
) -> list[Path]:
    """Reduce per-page leaves into section/chapter index.md files for *source_name*.

    Walks the persisted kreuzberg DocumentStructure (populated at ingest time).
    Nodes are processed bottom-up so every parent's children already exist on
    disk when it's reduced. Returns the list of inner-node paths written.
    Returns an empty list when no structure was persisted, no leaves exist,
    or every inner node failed to generate.
    """
    wiki_root = config.data_root / config.wiki_dir
    nodes = _load_document_structure(store, source_name)
    if not nodes or not leaf_paths_by_page:
        return []

    node_paths: dict[str, Path] = {}
    written: list[Path] = []
    for node in _wiki_nodes_bottom_up(nodes):
        children_paths, missing_children = _collect_inner_children(
            node, nodes, node_paths, leaf_paths_by_page
        )
        drafted_children = [p for p in children_paths if _is_draft_path(p, wiki_root)]
        partial_paths = missing_children + [str(p.relative_to(wiki_root)) for p in drafted_children]
        path = _generate_inner_node(
            node=node,
            source_name=source_name,
            source_slug=source_slug,
            children_paths=children_paths,
            raw_chunks=chunks,
            partial_paths=partial_paths,
            provider=provider,
            config=config,
            on_progress=on_progress,
        )
        if path is not None:
            node_paths[node.slug] = path
            written.append(path)

    return written


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
