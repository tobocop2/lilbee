"""Single-page generation pipeline for wiki summary and synthesis pages.

Given a label, prompt, and grounding chunks, drives the LLM call,
parses + verifies citations, scores faithfulness, builds the
frontmatter / body / citation block, and lands the page on disk via
:mod:`lilbee.wiki.persistence`. Also owns ``index_wiki_page``: the
post-write step that chunks, embeds, and stores the wiki body itself
so wiki content participates in retrieval.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from lilbee.core.config import CHUNKS_TABLE, DEFAULT_NUM_CTX, Config
from lilbee.core.services import get_services
from lilbee.data.chunk import chunk_text
from lilbee.data.store import (
    CHUNK_TYPE_WIKI,
    CitationRecord,
    SearchChunk,
    Store,
    escape_sql_string,
)
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.reasoning import strip_reasoning
from lilbee.wiki.citation import (
    ParsedCitation,
    extract_body,
    parse_wiki_citations,
    render_citation_block,
    strip_citation_block,
)
from lilbee.wiki.citations import (
    _render_provenance,
    _verify_citations,
)
from lilbee.wiki.persistence import _divert_to_drafts, _persist_and_finalize
from lilbee.wiki.quality import _check_faithfulness, _content_change_ratio, _diff_summary
from lilbee.wiki.shared import (
    DRAFTS_SUBDIR,
    WIKI_CONTENT_SUBDIRS,
    PageTarget,
)

log = logging.getLogger(__name__)

WikiProgressCallback = Callable[[str, dict[str, object]], None]
"""Callback for wiki generation progress: (stage, data) -> None."""

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
    # circular: persistence -> page via _subdir_from_wiki_source
    from lilbee.wiki.persistence import _subdir_from_wiki_source

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
