"""Wiki page generation — LLM-driven synthesis with citation provenance.

Generates summary pages (1:1 with sources) and synthesis pages (cross-source,
concept-graph-driven) from raw chunks. Each page carries inline citations
([^srcN]) for facts and [*inference*] markers for LLM synthesis. The
_citations table is the source of truth; markdown footnotes are rendered from it.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import yaml

from lilbee.chunk import chunk_text
from lilbee.clustering import SourceClusterer
from lilbee.config import CHUNKS_TABLE, DEFAULT_NUM_CTX, Config, cfg
from lilbee.ingest import file_hash
from lilbee.providers.base import LLMProvider
from lilbee.reasoning import strip_reasoning
from lilbee.services import get_services
from lilbee.store import (
    CHUNK_TYPE_WIKI,
    CitationRecord,
    SearchChunk,
    Store,
    escape_sql_string,
)
from lilbee.wiki.citation import (
    ParsedCitation,
    extract_body,
    parse_wiki_citations,
    render_citation_block,
    strip_citation_block,
)
from lilbee.wiki.entity_extractor import EntityKind, ExtractedEntity
from lilbee.wiki.index import append_wiki_log, update_wiki_index
from lilbee.wiki.links import apply_rewriter, compile_rewriter
from lilbee.wiki.shared import (
    ARCHIVE_SUBDIR,
    CONCEPTS_SUBDIR,
    DRAFTS_SUBDIR,
    ENTITIES_SUBDIR,
    MIN_CLUSTER_SOURCES,
    PENDING_KIND_PARSE,
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    WIKI_CONTENT_SUBDIRS,
    WIKI_LOG_ACTION_GENERATED,
    PageTarget,
    clean_label_for_display,
    is_valid_label,
    make_slug,
    parse_frontmatter,
)

log = logging.getLogger(__name__)

WikiProgressCallback = Callable[[str, dict[str, object]], None]
"""Callback for wiki generation progress: (stage, data) -> None."""

_MAX_DIFF_PREVIEW_LINES = 20  # lines of unified diff shown in drift warnings


# Fraction of context window reserved for chunks. The remainder leaves
# room for the system/user prompt template and generation output.
_CONTEXT_BUDGET_FRACTION = 0.75

# Approximate characters per token for budget estimation. 4 chars/token
# is a widely used heuristic for English text.
_CHARS_PER_TOKEN = 4

# Directive recognized by chat templates that support a reasoning mode
# (Qwen3, DeepSeek-R1, etc.). Wiki generation is a summarization task
# where chain-of-thought adds wall-clock cost without improving output,
# so we suppress it whenever the provider reports the capability.
_NO_THINK_DIRECTIVE = "/no_think"

# Capability string returned by llama-cpp providers for reasoning models
# (Qwen3, DeepSeek-R1). Defined locally so gen.py doesn't depend on a
# specific provider-layer constant name.
_CAPABILITY_THINKING = "thinking"

# JSON-style escape sequences that may appear inside quoted excerpts the
# model emits. Any backslash-prefixed character not in this map stays
# verbatim (e.g. ``\\x`` passes through unchanged).
_EXCERPT_ESCAPES: dict[str, str] = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def _build_wiki_messages(
    prompt: str, provider: LLMProvider, config: Config
) -> list[dict[str, str]]:
    """Build the chat messages list for a wiki-gen call.

    When the provider reports the ``thinking`` capability for the active
    chat model, prepends ``/no_think`` so the chat template disables the
    reasoning mode. Otherwise the prompt passes through unchanged.
    """
    capabilities = provider.get_capabilities(config.chat_model)
    if _CAPABILITY_THINKING in capabilities:
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
    context_window = config.num_ctx or DEFAULT_NUM_CTX
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


def _title_content_coherence(wiki_text: str, label: str) -> bool:
    """Deterministic pre-check: title and body must reference the concept.

    The LLM faithfulness score evaluates whether the prose reflects
    the source chunks but does not penalize structural noise in the
    title (bb-8b7s: ``| | designer`` passed at 0.90 because the body
    was coherent). This pre-check asserts three invariants:

    1. The first ``# `` heading must be a sanity-valid label per
       :func:`is_valid_label`. A heading like ``| | designer`` fails
       the structural-char gate even though it contains the cleaned
       display name as a substring.
    2. The cleaned display name must appear in the heading as a
       case-insensitive substring. Covers LLM drift where the
       heading names a different concept than requested.
    3. The body must mention the display name at least once outside
       the heading. Covers the "LLM talked about something adjacent
       but never named the concept" regression.

    Returns True when all three hold, False otherwise.
    """
    display = clean_label_for_display(label).lower()
    if not display:
        return False
    heading: str | None = None
    body_parts: list[str] = []
    for line in wiki_text.splitlines():
        if heading is None and line.startswith("# "):
            heading = line[2:].strip()
            continue
        body_parts.append(line)
    if heading is None:
        return False
    if not is_valid_label(heading):
        return False
    if display not in heading.lower():
        return False
    body = "\n".join(body_parts).lower()
    return display in body


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a non-empty vector list.

    Empty input returns an empty list; callers must check before any
    downstream dot-product so we do not leak a shape mismatch.

    Routes through numpy so the inner loop runs in C: for the typical
    ``D=768``, ``N=10`` case this cuts per-call cost from ~8k Python
    ops to a single SIMD-backed reduction.
    """
    if not vectors:
        return []
    result: list[float] = np.asarray(vectors, dtype=np.float32).mean(axis=0).tolist()
    return result


def _embedding_faithfulness_score(
    body_vec: list[float],
    source_vectors: list[list[float]],
) -> float:
    """Cosine-similarity score between the body and the mean source vector.

    Assumes L2-normalized vectors (both the embedder and the store
    return normalized vectors); cosine reduces to a dot product.
    Falls through to :func:`cosine_sim` so a non-normalized vector
    does not silently produce an out-of-range value. Result is
    clamped at zero because a negative cosine means the body vector
    points the other way from the mean of the sources — treat that
    the same as uncorrelated for threshold purposes.

    Returns 0.0 on a dimension mismatch between the body vector and
    the source-vector mean. That is not expected in production (the
    embedder and the chunk vectors come from the same model), but a
    stub-driven test may hand in off-shape vectors and crashing the
    whole pipeline on the shape-check hides the real assertion.
    """
    from lilbee.store import cosine_sim

    mean_vec = _mean_vector(source_vectors)
    if not mean_vec or not body_vec:
        return 0.0
    if len(mean_vec) != len(body_vec):
        log.warning(
            "Body vector dim %d does not match source vector dim %d; scoring 0.0",
            len(body_vec),
            len(mean_vec),
        )
        return 0.0
    return max(0.0, cosine_sim(body_vec, mean_vec))


def _check_faithfulness(
    chunks: list[SearchChunk],
    wiki_text: str,
    label: str,
    config: Config | None = None,
) -> float:
    """Score the wiki body's similarity to its source chunks, 0.0 on failure.

    Phase D: replaces the LLM-based faithfulness call with a
    deterministic cosine-similarity score between the page body and
    the mean of its source chunk vectors. The B3 title/body coherence
    pre-check still runs first as a hard gate: a garbage H1 returns
    0.0 regardless of embedding similarity, so structurally broken
    pages route to drafts even when the prose happens to be coherent.

    ``chunks`` carries ``.vector`` populated by LanceDB (see
    ``SearchChunk`` in ``store.py``), so no extra embedder call is
    needed for the source side. The body is embedded once via the
    shared services embedder. Any exception in the embedder (model
    missing, network issue, invalid config) is caught and reported as
    0.0 so a single faulty page drops to drafts instead of aborting
    the whole build.
    """
    if not _title_content_coherence(wiki_text, label):
        log.info(
            "Faithfulness title/body coherence failed for %r; scoring 0.0",
            label,
        )
        return 0.0
    source_vectors = [c.vector for c in chunks if c.vector]
    if not source_vectors:
        log.warning("No source vectors for %s; scoring 0.0", label)
        return 0.0

    # Strip the frontmatter + citation block so we embed only the body
    # prose. render_citation_block may not have run yet when the score
    # is computed (it is appended later), but strip_citation_block is
    # idempotent on missing trailers.
    body_text = strip_citation_block(wiki_text).strip()
    if not body_text:
        log.warning("Empty body for %s; scoring 0.0", label)
        return 0.0

    try:
        body_vectors = get_services().embedder.embed_batch([body_text])
    except Exception as exc:
        log.warning("Body embedding failed for %s: %s", label, exc)
        return 0.0
    if not body_vectors:
        return 0.0
    return _embedding_faithfulness_score(body_vectors[0], source_vectors)


def _build_frontmatter(
    config: Config,
    source_names: list[str],
    score: float,
    leaf_hash: str = "",
    chunks: list[SearchChunk] | None = None,
) -> str:
    """Build YAML frontmatter for a wiki page.

    When ``leaf_hash`` is non-empty it is written so incremental rebuild
    can skip regeneration on a subsequent sync whose chunks produce the
    same hash. When ``chunks`` is provided the frontmatter carries a
    ``provenance`` block naming the source/chunk-index pairs that fed
    the generator and the extraction method from config, so a bad page
    is auditable without re-running the pipeline.
    """
    sources_yaml = ", ".join(f'"{s}"' for s in sorted(source_names))
    hash_line = f"leaf_hash: {leaf_hash}\n" if leaf_hash else ""
    provenance_block = _render_provenance(config, chunks) if chunks is not None else ""
    return (
        f"---\n"
        f"generated_by: {config.chat_model}\n"
        f"generated_at: {datetime.now(UTC).isoformat()}\n"
        f"sources: [{sources_yaml}]\n"
        f"faithfulness_score: {score:.2f}\n"
        f"{hash_line}"
        f"{provenance_block}"
        f"---\n\n"
    )


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


def index_wiki_page(content: str, wiki_source: str, store: Store) -> int:
    """Chunk a wiki page body, embed it, and write rows with ``chunk_type="wiki"``.

    ``wiki_source`` must follow the ``<wiki_dir>/<subdir>/<slug>.md``
    shape (see :attr:`PageTarget.wiki_source`). Three branches:

    - subdir in :data:`WIKI_CONTENT_SUBDIRS`: clear stale rows, chunk,
      embed, write. Returns the row count.
    - subdir is ``drafts/`` or ``archive/``: skip without touching the
      store. Returns 0.
    - malformed ``wiki_source`` (no subdir component): log.warning and
      return 0. Does not raise because the caller set is narrow (only
      internal wiki paths reach here) and surfacing the bad input in
      the log is sufficient triage.

    Record shape matches the markdown-ingest convention in
    ``ingest.py``: ``content_type="text"``, all four page/line
    positions ``0`` (wiki pages are not paginated).
    """
    subdir = _subdir_from_wiki_source(wiki_source)
    if subdir is None:
        log.warning("index_wiki_page: malformed wiki_source %r (no subdir)", wiki_source)
        return 0
    if subdir not in WIKI_CONTENT_SUBDIRS:
        return 0

    body = extract_body(content).strip()
    store.clear_table(
        CHUNKS_TABLE,
        f"source = '{escape_sql_string(wiki_source)}' AND chunk_type = '{CHUNK_TYPE_WIKI}'",
    )
    if not body:
        return 0

    chunks = chunk_text(body, mime_type="text/markdown", use_semantic=True)
    if not chunks:
        return 0

    vectors = get_services().embedder.embed_batch(chunks)
    records = [
        {
            "source": wiki_source,
            "content_type": "text",
            "chunk_type": CHUNK_TYPE_WIKI,
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": text,
            "chunk_index": idx,
            "vector": vector,
        }
        for idx, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    store.add_chunks(records)
    return len(records)


def _subdir_from_wiki_source(wiki_source: str) -> str | None:
    """Return the subdir component (``summaries``, ``concepts``, ...) of *wiki_source*.

    ``wiki_source`` is the ``<wiki_dir>/<subdir>/<slug>.md`` path
    stored in citations and chunks. Returns None when the path has
    fewer than two components.
    """
    parts = wiki_source.split("/")
    return parts[1] if len(parts) >= 2 else None


def _persist_and_finalize(
    content: str,
    target: PageTarget,
    verified: list[CitationRecord],
    source_names: list[str],
    store: Store,
    config: Config,
) -> Path:
    """Write page to disk, persist citations, index body chunks, update index and log."""
    page_path = _write_page(
        target.wiki_root, target.subdir, target.slug, content, config.wiki_drift_threshold
    )
    for rec in verified:
        rec["wiki_source"] = target.wiki_source
    store.delete_citations_for_wiki(target.wiki_source)
    store.add_citations(verified)

    index_wiki_page(content, target.wiki_source, store)

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
    score = _check_faithfulness(chunks, wiki_text, label, config)
    threshold = config.wiki_embedding_faithfulness_threshold
    subdir = page_type if score >= threshold else DRAFTS_SUBDIR
    if subdir == DRAFTS_SUBDIR:
        log.info("Wiki page %s scored %.2f (< %.2f), sending to drafts", label, score, threshold)

    wiki_text = strip_citation_block(wiki_text)
    frontmatter = _build_frontmatter(config, source_names, score, leaf_hash, chunks=chunks)
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
    display_topic = clean_label_for_display(topic)
    prompt = template.format(topic=display_topic, source_list=source_list, chunks_text=chunks_text)
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
        label=topic,
        prompt=prompt,
        chunks=all_chunks,
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


def _hash_existing_sources(source_names: list[str], documents_dir: Path) -> dict[str, str]:
    """Hash each source file that still exists on disk (used for citation staleness)."""
    out: dict[str, str] = {}
    for name in source_names:
        source_path = documents_dir / name
        if source_path.exists():
            out[name] = file_hash(source_path)
    return out


# Phase D: archive-migration sentinel and helpers. The sentinel lives
# under data_dir (NOT inside wiki/) so Obsidian sync and wiki
# tree-walkers never surface it.
_PHASE_D_SENTINEL_NAME = ".phase-d-migrated"

# Pre-Phase-D wiki concepts that we move to archive/ as part of the
# one-time migration. Matches wiki/<CONCEPTS_SUBDIR>/*.md recursively.
_ARCHIVE_CONCEPTS_SUBPATH = Path(ARCHIVE_SUBDIR) / CONCEPTS_SUBDIR


def _maybe_run_phase_d_migration(wiki_root: Path, data_dir: Path) -> None:
    """One-time migration: archive pre-Phase-D concept pages.

    Runs idempotently, gated by ``{data_dir}/.phase-d-migrated``:

    1. Move every ``wiki/concepts/*.md`` to ``wiki/archive/concepts/``
       preserving relative subpaths. Older concept pages stay
       readable but drop out of the active wiki browse surface.
    2. Unwrap stale ``[[archived-slug]]`` references across the
       remaining pages so a reader clicking a link does not hit a
       404. Archived slugs become plain text.
    3. Write the sentinel so future builds skip this path.

    D3's freshly LLM-curated concept pages written AFTER the sentinel
    exists are never touched.
    """
    sentinel = data_dir / _PHASE_D_SENTINEL_NAME
    if sentinel.exists():
        return
    concepts_dir = wiki_root / CONCEPTS_SUBDIR
    archive_dir = wiki_root / _ARCHIVE_CONCEPTS_SUBPATH
    archived_slugs: list[str] = []
    if concepts_dir.is_dir():
        for src in sorted(concepts_dir.rglob("*.md")):
            rel = src.relative_to(concepts_dir)
            dest = archive_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dest)
            archived_slugs.append(str(rel.with_suffix("")).replace("\\", "/"))

    if archived_slugs:
        _unwrap_archived_links(wiki_root, archived_slugs)

    data_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    if archived_slugs:
        log.info(
            "Phase D migration: archived %d concept pages, sentinel written at %s",
            len(archived_slugs),
            sentinel,
        )


def _unwrap_archived_links(wiki_root: Path, archived_slugs: list[str]) -> None:
    """Rewrite ``[[slug]]`` → ``slug`` (plain text) across remaining wiki pages.

    The existing ``_rewrite_links_across_wiki`` path is the wrong
    tool here: it compiles an *additive* surface map, not a
    removal pass. Walk the active wiki content subdirs once per
    archived slug is acceptable because the archive count is
    bounded (concepts that existed pre-migration). Pages whose body
    did not change are not rewritten.
    """
    if not archived_slugs:
        return
    patterns = [(re.compile(r"\[\[" + re.escape(slug) + r"\]\]"), slug) for slug in archived_slugs]
    for subdir in WIKI_CONTENT_SUBDIRS:
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        for md_path in subdir_path.rglob("*.md"):
            original = md_path.read_text(encoding="utf-8")
            rewritten = original
            for pattern, replacement in patterns:
                rewritten = pattern.sub(replacement, rewritten)
            if rewritten != original:
                md_path.write_text(rewritten, encoding="utf-8")


# Pending-marker conventions: the drafts listing surface
# (``lilbee.wiki.drafts``) scans for these prefixes to classify a
# draft as PARSE or COLLISION instead of a drift-routed regen. The
# keyword phrases live in ``wiki.shared`` so writer (gen) and reader
# (drafts) stay in sync on the exact wording.
_PENDING_PARSE_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_PARSE}"
_PENDING_COLLISION_MARKER_PREFIX = f"<!-- {PENDING_MARKER_KEYWORD_COLLISION}"


def _write_pending_marker(
    drafts_dir: Path,
    slug: str,
    marker_line: str,
    frontmatter: str = "",
) -> Path:
    """Write a PENDING marker page under ``drafts/<slug>.md``.

    ``marker_line`` is the leading HTML comment that both identifies
    the marker kind and carries the context (source, label). The
    optional ``frontmatter`` preserves minimal metadata for the
    drafts surface to round-trip (e.g. ``bad_title``-style fields).
    """
    drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = drafts_dir / f"{slug}.md"
    body = marker_line + "\n"
    if frontmatter:
        body += "\n" + frontmatter
    draft_path.write_text(body, encoding="utf-8")
    return draft_path


def _delete_pending_marker_if_present(drafts_dir: Path, slug: str) -> bool:
    """Delete an existing PENDING marker for *slug*; return whether one was removed.

    Match is slug-equality (not fuzzy): an LLM that rephrases a
    label on retry (``brake system`` → ``braking system``) leaves
    the old marker behind for the user to drain via ``wiki drafts
    reject``. Documented limitation; follow-up if the pattern
    matters.
    """
    draft_path = drafts_dir / f"{slug}.md"
    if not draft_path.is_file():
        return False
    try:
        body = draft_path.read_text(encoding="utf-8")
    except OSError:
        return False
    first_line = body.splitlines()[0] if body else ""
    is_pending = first_line.startswith(_PENDING_PARSE_MARKER_PREFIX) or first_line.startswith(
        _PENDING_COLLISION_MARKER_PREFIX
    )
    if not is_pending:
        return False
    draft_path.unlink()
    return True


def _group_entities_by_primary_source(
    entities: list[ExtractedEntity],
) -> dict[str, list[ExtractedEntity]]:
    """Group entities under the source that mentions them most.

    Primary source = source with the highest chunk-ref count;
    lexicographic tiebreak. An entity with no refs is dropped
    silently (defensive: extractor always attaches refs, but a
    future extractor might not).
    """
    grouped: dict[str, list[ExtractedEntity]] = {}
    for entity in entities:
        if not entity.chunk_refs:
            continue
        counts: dict[str, int] = {}
        for ref in entity.chunk_refs:
            counts[ref.source] = counts.get(ref.source, 0) + 1
        primary = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        grouped.setdefault(primary, []).append(entity)
    return grouped


# Regex that matches section headers the batch parser recognizes:
# H1 (``# Name``), H2 (``## Name``), or a bold-line heading
# (``**Name**``) at line start. The name capture is anchored to the
# rest of the line (stripped of trailing whitespace) so labels like
# ``## Brake System (hydraulic)`` still parse.
_SECTION_HEADER_RE = re.compile(
    r"^(?:(?:##?)\s+(?P<hashname>[^\n]+)|\*\*(?P<boldname>[^\*\n]+)\*\*)\s*$",
    re.MULTILINE,
)

# In-body ``[^keyN]`` footnote-marker pattern. Module-scope so the
# batched-generation hot path (`_finalize_section`) does not recompile
# it on every recovered section.
_FOOTNOTE_MARKER_RE = re.compile(r"\[\^([a-zA-Z0-9_\-]+)\]")


def _split_batched_output(
    text: str,
    expected_entity_labels: set[str],
    expected_concept_labels: set[str] | None = None,
) -> dict[str, tuple[EntityKind, str]]:
    """Best-effort parse of the batched LLM response into per-label bodies.

    Splits on H1/H2/bold-line headers, then matches each header
    against the expected entity and concept label sets via
    case-insensitive substring. Known labels are tagged with the
    right ``EntityKind``; unknown headers are dropped. Labels whose
    section could not be recovered at all are surfaced to the caller
    (they show up as *missing from the return dict* rather than a
    separate list — caller loops over the expected sets to write
    PENDING markers).
    """
    concepts = expected_concept_labels or set()
    recovered: dict[str, tuple[EntityKind, str]] = {}
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return recovered
    for i, match in enumerate(matches):
        name = match.group("hashname") or match.group("boldname") or ""
        name = name.strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        lowered = name.lower()
        kind_label = _match_label(lowered, expected_entity_labels, EntityKind.ENTITY)
        if kind_label is None:
            kind_label = _match_label(lowered, concepts, EntityKind.CONCEPT)
        if kind_label is None:
            # Concept labels come from the LLM itself — tag any
            # unmatched section as CONCEPT only when the caller is
            # expecting concept curation; otherwise drop it as
            # noise.
            if concepts is not None and expected_concept_labels is not None:
                recovered.setdefault(name, (EntityKind.CONCEPT, _prefix_heading(name, body)))
            continue
        kind, label = kind_label
        recovered[label] = (kind, _prefix_heading(name, body))
    return recovered


def _match_label(
    lowered_name: str,
    expected: set[str],
    kind: EntityKind,
) -> tuple[EntityKind, str] | None:
    """Case-insensitive substring match of *lowered_name* against *expected*.

    Returns ``(kind, original_label)`` on hit, ``None`` otherwise.
    A substring match (not equality) accommodates the LLM adding
    qualifiers ("Brake System (hydraulic)" vs "brake system").
    """
    for label in expected:
        low = label.lower()
        if low and (low in lowered_name or lowered_name in low):
            return (kind, label)
    return None


def _prefix_heading(name: str, body: str) -> str:
    """Ensure the extracted body starts with a ``# Name`` H1.

    The batched prompt instructs the model to emit ``## Name`` per
    section. After splitting, the per-section body has lost its
    header. Rebuild an H1 so the B3 title/body coherence gate still
    has a heading to match.
    """
    stripped = body.lstrip()
    if stripped.startswith("# "):
        return body
    return f"# {name}\n\n{body}"


def _chunks_for_source(chunks: list[SearchChunk], source: str) -> list[SearchChunk]:
    """Return the subset of *chunks* whose ``source`` matches, preserving order."""
    return [c for c in chunks if c.source == source]


def _build_batch_prompt(
    source: str,
    entities: list[ExtractedEntity],
    chunks_text: str,
    extract_concepts: bool,
    config: Config,
) -> str:
    """Render :attr:`Config.wiki_entity_batch_prompt` for one source call.

    ``extract_concepts`` controls whether the concept-curation
    paragraph is injected: True adds a "identify 3-5 concepts" block;
    False leaves ``{concept_instruction}`` empty so the LLM writes
    entity sections only. Keeps the per-source batched call the
    single entry point whether or not concepts are requested.
    """
    entity_labels = ", ".join(clean_label_for_display(e.label) for e in entities) or "(none)"
    if extract_concepts:
        concept_instruction = (
            "First, identify 3-5 CONCEPTS — abstract topics or domain terms "
            "from the source that deserve a standalone wiki page. Do NOT include "
            "pronouns, articles, or generic nouns.\n\n"
            "Then write a wiki section for each of the concepts you identified, "
            "PLUS one section for each NER ENTITY listed below.\n\n"
        )
    else:
        concept_instruction = ""
    return config.wiki_entity_batch_prompt.format(
        source=source,
        entity_list=entity_labels,
        chunks_text=chunks_text,
        concept_instruction=concept_instruction,
    )


def _short_source_hash(source: str) -> str:
    """8-char sha256 digest of *source* (stable collision-marker suffix)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def _generate_source_batch(
    source: str,
    entities: list[ExtractedEntity],
    chunks: list[SearchChunk],
    provider: LLMProvider,
    store: Store,
    config: Config,
    *,
    extract_concepts: bool,
    written_concept_slugs: dict[str, str],
) -> list[Path]:
    """Issue one LLM call for *source* and finalize every recovered section.

    Returns the list of page paths written (entities + concepts
    combined). Labels not recovered by the parser become PENDING
    markers under ``wiki/drafts/`` so the next build can retry.
    Concept slugs already written by an earlier source produce a
    PENDING-COLLISION marker on the losing side (see
    :func:`_handle_concept_write`).

    ``written_concept_slugs`` is the per-build ledger of
    slug → first_source. Callers share one dict across the per-source
    loop. The second source to propose a slug is the one that gets
    diverted to a collision marker.
    """
    if not chunks:
        return []
    budgeted = _truncate_chunks_to_budget(chunks, config)
    chunks_text = _chunks_to_text(budgeted)
    prompt = _build_batch_prompt(source, entities, chunks_text, extract_concepts, config)
    messages = _build_wiki_messages(prompt, provider, config)
    options = config.generation_options(
        temperature=config.wiki_temperature,
        max_tokens=config.wiki_summary_max_tokens,
    )
    try:
        response = provider.chat(messages, stream=False, options=options)
        text = strip_reasoning(cast(str, response)).strip()
    except Exception as exc:
        log.warning("Batched LLM call failed for source %s: %s", source, exc)
        return []

    if not text:
        log.warning("Batched LLM call returned empty response for source %s", source)
        return []

    expected_entity_labels = {e.label for e in entities}
    expected_concepts: set[str] | None = set() if extract_concepts else None
    parsed = _split_batched_output(text, expected_entity_labels, expected_concepts)

    wiki_root = config.data_root / config.wiki_dir
    drafts_dir = wiki_root / DRAFTS_SUBDIR
    source_names = [source]
    source_hashes = _hash_existing_sources(source_names, config.documents_dir)
    chunks_by_source = {source: budgeted}

    # Citation definitions live in the trailing block of the WHOLE
    # response, not inside any one section body. Parse once over the
    # full text and replay the same list for every section, so each
    # page sees its own citations even when only the last section
    # carries the definition trailer.
    shared_parsed_citations = parse_wiki_citations(text)

    pages: list[Path] = []
    seen_labels: set[str] = set()
    for header_label, (kind, body) in parsed.items():
        seen_labels.add(header_label)
        resolver = functools.partial(
            _resolve_multi_source_citations,
            source_names=source_names,
            source_hashes=source_hashes,
            chunks_by_source=chunks_by_source,
        )
        page = _finalize_section(
            header_label=header_label,
            kind=kind,
            body=body,
            chunks=budgeted,
            citation_resolver=resolver,
            source_names=source_names,
            store=store,
            config=config,
            source=source,
            written_concept_slugs=written_concept_slugs,
            drafts_dir=drafts_dir,
            shared_parsed_citations=shared_parsed_citations,
        )
        if page is not None:
            pages.append(page)

    for entity in entities:
        if entity.label not in seen_labels:
            marker = (
                f"{_PENDING_PARSE_MARKER_PREFIX} for source {source}, "
                f"entity/concept {entity.label} - "
                "run wiki build again or manually accept via wiki drafts accept -->"
            )
            # Route through ``yaml.safe_dump`` so a label or source
            # containing a colon, quote, or newline does not produce a
            # frontmatter block that ``parse_frontmatter`` silently drops.
            frontmatter_body = yaml.safe_dump(
                {
                    "pending_source": source,
                    "pending_label": entity.label,
                    "pending_kind": PENDING_KIND_PARSE,
                },
                sort_keys=False,
            )
            frontmatter = f"---\n{frontmatter_body}---\n"
            path = _write_pending_marker(drafts_dir, entity.slug, marker, frontmatter)
            log.info("Wrote PENDING-PARSE marker for %s -> %s", entity.slug, path)

    return pages


def _finalize_section(
    *,
    header_label: str,
    kind: EntityKind,
    body: str,
    chunks: list[SearchChunk],
    citation_resolver: Callable[[list[ParsedCitation]], list[CitationRecord]],
    source_names: list[str],
    store: Store,
    config: Config,
    source: str,
    written_concept_slugs: dict[str, str],
    drafts_dir: Path,
    shared_parsed_citations: list[ParsedCitation],
) -> Path | None:
    """Citation-check, faithfulness-check, write one batched section.

    Shared by entity and concept sections from the per-source batched
    call. Returns the written page path, or ``None`` if the section
    failed any gate (no citations, empty body, slug collision marker
    handled via side channel). ``shared_parsed_citations`` is the
    definition list parsed once over the whole response — every
    section replays it so pages other than the last one still have
    their footnotes resolved.
    """
    slug = make_slug(header_label)
    if not slug:
        log.info("Empty slug for batched section %r; skipping", header_label)
        return None

    # Only replay citation keys that this section actually references
    # in the body; otherwise every section would claim every citation.
    section_keys = {ref.citation_key for ref in parse_wiki_citations(body)}
    # Fall back to in-body ``[^keyN]`` references when no definitions
    # live inside the section: count occurrences of the footnote
    # marker against the shared definition set.
    section_keys.update(_FOOTNOTE_MARKER_RE.findall(body))
    relevant = [c for c in shared_parsed_citations if c.citation_key in section_keys]
    verified = _verify_citations(citation_resolver(relevant), chunks, header_label, config)
    if not verified:
        log.info("No valid citations for batched section %s, skipping", header_label)
        return None

    score = _check_faithfulness(chunks, body, header_label, config)
    threshold = config.wiki_embedding_faithfulness_threshold
    page_type = CONCEPTS_SUBDIR if kind is EntityKind.CONCEPT else ENTITIES_SUBDIR
    subdir = page_type if score >= threshold else DRAFTS_SUBDIR
    if subdir == DRAFTS_SUBDIR:
        log.info(
            "Batched section %s scored %.2f (< %.2f), sending to drafts",
            header_label,
            score,
            threshold,
        )

    clean_body = strip_citation_block(body)
    frontmatter = _build_frontmatter(config, source_names, score, chunks=chunks)
    citation_block = render_citation_block(verified)
    full_content = _assemble_content(frontmatter, clean_body, citation_block)

    # Concept collision: the second source proposing a slug loses
    # and writes to a drafts collision marker; the winning source's
    # page stays untouched.
    if kind is EntityKind.CONCEPT and subdir == CONCEPTS_SUBDIR:
        first_source = written_concept_slugs.get(slug)
        if first_source is not None and first_source != source:
            return _divert_concept_collision(
                slug=slug,
                source=source,
                first_source=first_source,
                content=full_content,
                drafts_dir=drafts_dir,
            )
        written_concept_slugs.setdefault(slug, source)

    # Successful regen of a previously-PENDING slug: remove the old
    # marker so the drafts surface no longer lists it.
    _delete_pending_marker_if_present(drafts_dir, slug)

    wiki_root = config.data_root / config.wiki_dir
    target = PageTarget(
        wiki_root=wiki_root,
        subdir=subdir,
        slug=slug,
        wiki_source=f"{config.wiki_dir}/{subdir}/{slug}.md",
        page_type=page_type,
        label=header_label,
    )
    page_path = _persist_and_finalize(full_content, target, verified, source_names, store, config)
    log.info(
        "Generated batched page for %s -> %s (score=%.2f, citations=%d)",
        header_label,
        target.subdir,
        score,
        len(verified),
    )
    return page_path


def _divert_concept_collision(
    *,
    slug: str,
    source: str,
    first_source: str,
    content: str,
    drafts_dir: Path,
) -> Path:
    """Write the losing concept to ``drafts/<slug>-collision-<hash>.md``.

    The winning source's page is unchanged on disk. Hash is the
    first 8 hex of sha256(source_filename); stable per source so a
    retry on the same two sources lands at the same draft path,
    letting the user iterate without marker sprawl.
    """
    short = _short_source_hash(source)
    collision_slug = f"{slug}-collision-{short}"
    marker = (
        f"{_PENDING_COLLISION_MARKER_PREFIX} with source {first_source}, "
        f"content from {source} held for review -->\n\n"
    )
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = drafts_dir / f"{collision_slug}.md"
    path.write_text(marker + content, encoding="utf-8")
    log.warning(
        "Concept slug collision: %s already written by %s; diverted %s's version to %s",
        slug,
        first_source,
        source,
        path,
    )
    return path


def build_wiki(
    entities: list[ExtractedEntity],
    provider: LLMProvider,
    store: Store,
    config: Config | None = None,
    *,
    extract_concepts: bool = True,
) -> list[Path]:
    """Produce entity and LLM-curated concept pages per source.

    Phase D replaces the per-entity / per-concept fan-out with a
    per-source batched call: for each source in ``entities``' chunk
    refs, one LLM call identifies 3-5 concepts AND writes a wiki
    section for every pre-extracted entity belonging to that source.
    Output sections are split, citation-verified, embedding-scored,
    and landed under ``wiki/entities/`` or ``wiki/concepts/``
    depending on kind.

    ``extract_concepts=False`` (used by the incremental-ingest hook)
    drops the concept-curation paragraph from the prompt so a
    touched source does not churn concept slugs.

    A one-time archive migration runs first (idempotently, gated by
    ``{data_dir}/.phase-d-migrated``), moving pre-Phase-D concept
    pages under ``wiki/archive/concepts/`` and unwrapping stale
    ``[[archived-slug]]`` links across the remaining pages.
    """
    if config is None:
        config = cfg
    wiki_root = config.data_root / config.wiki_dir
    _maybe_run_phase_d_migration(wiki_root, config.data_dir)

    grouped = _group_entities_by_primary_source(entities)
    all_sources = _all_sources_in_scope(entities, grouped, store, config, extract_concepts)
    written_concept_slugs: dict[str, str] = {}
    pages: list[Path] = []

    for source in sorted(all_sources):
        source_entities = grouped.get(source, [])
        chunks = store.get_chunks_by_source(source)
        chunk_count = len(chunks)
        source_extract = extract_concepts and chunk_count >= config.wiki_batch_min_chunks
        if not source_entities and not source_extract:
            log.info(
                "Skipping source %s: %d entities, %d chunks, min=%d, extract=%s",
                source,
                len(source_entities),
                chunk_count,
                config.wiki_batch_min_chunks,
                source_extract,
            )
            continue
        source_pages = _generate_source_batch(
            source=source,
            entities=source_entities,
            chunks=chunks,
            provider=provider,
            store=store,
            config=config,
            extract_concepts=source_extract,
            written_concept_slugs=written_concept_slugs,
        )
        pages.extend(source_pages)

    _rewrite_links_across_wiki(entities, config)
    log.info("Generated %d batched wiki pages", len(pages))
    return pages


def _all_sources_in_scope(
    entities: list[ExtractedEntity],
    grouped: dict[str, list[ExtractedEntity]],
    store: Store,
    config: Config,
    extract_concepts: bool,
) -> set[str]:
    """Union of sources with entities and (when enabled) eligible for concept curation.

    Seed the union with every entity's primary source. When
    ``extract_concepts`` is True AND ``wiki_batch_min_chunks`` is
    satisfied, add any source in the store that passes the floor.
    This gives concept-only sources (no extracted entities) their
    chance at curation while keeping zero-entity short sources
    skipped entirely.
    """
    sources: set[str] = set(grouped)
    if not extract_concepts:
        return sources
    try:
        records = store.get_sources()
    except Exception as exc:
        log.warning("get_sources failed; sticking to entity-grouped sources: %s", exc)
        return sources
    for record in records:
        name = record.get("filename", "") if isinstance(record, dict) else ""
        if not name:
            continue
        if name in sources:
            continue
        chunk_count = record.get("chunk_count", 0) if isinstance(record, dict) else 0
        if chunk_count >= config.wiki_batch_min_chunks:
            sources.add(name)
    _ = entities  # silences linters on unused pass-through; kept for doc clarity
    return sources


def _entity_surface_map(entities: list[ExtractedEntity]) -> dict[str, str]:
    """Build the surface-form -> slug map for the ``[[link]]`` rewriter.

    Includes both the entity's human label (e.g. *"Henry Ford"*) and
    the slug-with-hyphens-as-spaces variant (*"henry ford"*) so the
    rewriter catches either form in body text.
    """
    mapping: dict[str, str] = {}
    for entity in entities:
        mapping[entity.label] = entity.slug
        spaced = entity.slug.replace("-", " ")
        if spaced and spaced != entity.label:
            mapping[spaced] = entity.slug
    return mapping


_ENTITY_LIKE_SUBDIRS: tuple[str, ...] = (CONCEPTS_SUBDIR, ENTITIES_SUBDIR)


def _augment_surface_map_with_existing_pages(
    surface_to_slug: dict[str, str], wiki_root: Path
) -> None:
    """Add slugs for pages already on disk so an incremental rebuild of
    one concept still links to its unchanged neighbors. **Mutates
    surface_to_slug in place.** Only enriches the map with the
    hyphen-to-space surface form because frontmatter labels aren't
    read here; body prose typically uses the spaced form so this
    covers the common case.
    """
    for subdir in _ENTITY_LIKE_SUBDIRS:
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        for md_path in subdir_path.rglob("*.md"):
            slug = md_path.stem
            spaced = slug.replace("-", " ")
            surface_to_slug.setdefault(spaced, slug)


def _rewrite_links_across_wiki(entities: list[ExtractedEntity], config: Config) -> None:
    """Rewrite ``[[slug]]`` links on every page under ``wiki/`` content subdirs.

    A page never receives a link to itself: the rewriter takes the
    owning slug and drops it inside its match callback, so the
    surface map is shared unmodified across every page in the walk
    (no O(M) dict rebuild per file). The map is augmented with
    slugs from the existing on-disk corpus so a touched page still
    links to untouched neighbors. The alternation regex + lookup are
    compiled once per build and reused across pages.
    """
    surface_to_slug = _entity_surface_map(entities)
    wiki_root = config.data_root / config.wiki_dir
    _augment_surface_map_with_existing_pages(surface_to_slug, wiki_root)
    rewriter = compile_rewriter(surface_to_slug)
    if rewriter is None:
        return

    for subdir in WIKI_CONTENT_SUBDIRS:
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        is_entity_subdir = subdir in _ENTITY_LIKE_SUBDIRS
        for md_path in subdir_path.rglob("*.md"):
            owning_slug = md_path.stem if is_entity_subdir else None
            original = md_path.read_text(encoding="utf-8")
            rewritten = apply_rewriter(original, rewriter, skip_slug=owning_slug)
            if rewritten != original:
                md_path.write_text(rewritten, encoding="utf-8")


class WikiBuildSummary(TypedDict):
    """Result of a full wiki build/update."""

    paths: list[str]
    entities: int
    count: int


def run_full_build(config: Config | None = None) -> WikiBuildSummary:
    """Extract entities and build wiki pages for every ingested source."""
    if config is None:
        config = cfg
    from lilbee.wiki.entity_extractor import get_entity_extractor
    from lilbee.wiki.shared import WIKI_LOG_ACTION_BUILD

    svc = get_services()
    chunks: list[SearchChunk] = []
    for record in svc.store.get_sources():
        chunks.extend(svc.store.get_chunks_by_source(record["filename"]))

    extractor = get_entity_extractor(config.wiki_entity_mode, svc.provider, config)
    entities = extractor.extract(chunks)
    pages = build_wiki(
        entities,
        svc.provider,
        svc.store,
        config,
        extract_concepts=config.wiki_extract_concepts,
    )
    update_wiki_index()
    append_wiki_log(WIKI_LOG_ACTION_BUILD, f"{len(pages)} pages from {len(entities)} records")
    return {
        "paths": [str(p) for p in pages],
        "entities": len(entities),
        "count": len(pages),
    }


class WikiSynthesizeSummary(TypedDict):
    """Result of running synthesis-page generation."""

    paths: list[str]
    count: int


def run_full_synthesize(config: Config | None = None) -> WikiSynthesizeSummary:
    """Generate synthesis pages for cross-source clusters of 3+ documents.

    Shared entry point for MCP ``wiki_synthesize`` and ``POST
    /api/wiki/synthesize``. Concurrency contract matches
    :func:`run_full_build`: not safe to run in parallel with itself or
    with other wiki write paths; callers serialize via an external lock
    on shared event loops.
    """
    if config is None:
        config = cfg
    svc = get_services()
    paths = generate_synthesis_pages(svc.provider, svc.store, svc.clusterer, config)
    return {
        "paths": [str(p) for p in paths],
        "count": len(paths),
    }
