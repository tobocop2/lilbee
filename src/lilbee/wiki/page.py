"""Single-page generation pipeline for wiki summary and synthesis pages.

Given a label, prompt, and grounding chunks, drives the LLM call,
parses + verifies citations, scores faithfulness, builds the
frontmatter / body / citation block, and lands the page on disk via
:mod:`lilbee.wiki.persistence`. Also owns ``index_wiki_page``: the
post-write step that chunks, embeds, and stores the wiki body itself
so wiki content participates in retrieval.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lilbee.app.services import get_services
from lilbee.core.config import CHUNKS_TABLE, DEFAULT_NUM_CTX, Config
from lilbee.data.extract.chunk import chunk_text
from lilbee.data.store import (
    ChunkType,
    CitationRecord,
    SearchChunk,
    Store,
    escape_sql_string,
)
from lilbee.providers.base import LLMProvider
from lilbee.retrieval.reasoning import strip_reasoning
from lilbee.wiki.citations import (
    ParsedCitation,
    extract_body,
    parse_wiki_citations,
    render_citation_block,
    render_provenance,
    scrub_unverified_markers,
    strip_citation_block,
    verify_citations,
    wiki_sourced_count,
)
from lilbee.wiki.persistence import (
    delete_drift_draft_if_present,
    divert_concept_collision,
    divert_to_drafts,
    draft_source_names,
    origin_marker,
    persist_and_finalize,
    subdir_from_wiki_source,
)
from lilbee.wiki.quality import check_faithfulness, content_change_ratio, diff_summary
from lilbee.wiki.shared import (
    WIKI_CONTENT_SUBDIRS,
    PageTarget,
    WikiSubdir,
    atomic_write_text,
)
from lilbee.wiki.stats import BuildStats

log = logging.getLogger(__name__)

WikiProgressCallback = Callable[[str, dict[str, object]], None]
"""Callback for wiki generation progress: (stage, data) -> None."""

# Floor on the chunk budget as a fraction of the context window, applied when
# the output cap plus the template reserve already exceed ``num_ctx``.
_MIN_CONTEXT_BUDGET_FRACTION = 0.25

# Approximate characters per token for budget estimation. 4 chars/token
# is a widely used heuristic for English text.
_CHARS_PER_TOKEN = 4

# Separator ``chunks_to_text`` puts between formatted chunk blocks.
_CHUNK_SEPARATOR = "\n\n"

# Seed used for wiki generation when the user set none, so an unchanged
# corpus regenerates identically at the low wiki sampling temperature.
WIKI_DEFAULT_SEED = 1

# Directive recognized by chat templates that support a reasoning mode
# (Qwen3, DeepSeek-R1, etc.). Wiki generation is a summarization task
# where chain-of-thought adds wall-clock cost without improving output,
# so we suppress it whenever the provider reports the capability.
_NO_THINK_DIRECTIVE = "/no_think"

# Capability string a provider's get_capabilities reports for reasoning
# models (Qwen3, DeepSeek-R1).
_CAPABILITY_THINKING = "thinking"


def build_wiki_messages(prompt: str, provider: LLMProvider, config: Config) -> list[dict[str, str]]:
    """Build the chat messages list for a wiki-gen call.

    When the provider reports the ``thinking`` capability for the active
    chat model, prepends ``/no_think`` so the chat template disables the
    reasoning mode. Otherwise the prompt passes through unchanged.
    """
    capabilities = provider.get_capabilities(config.chat_model)
    if _CAPABILITY_THINKING in capabilities:
        prompt = f"{_NO_THINK_DIRECTIVE}\n\n{prompt}"
    return [{"role": "user", "content": prompt}]


def wiki_generation_options(config: Config) -> dict[str, Any]:
    """Sampling options for a wiki LLM call: wiki temperature, output cap, fixed seed."""
    return config.generation_options(
        temperature=config.wiki_temperature,
        max_tokens=config.wiki_summary_max_tokens,
        seed=WIKI_DEFAULT_SEED if config.seed is None else config.seed,
    )


def prompt_overhead_tokens(config: Config, prompt_chars: int | None = None) -> int:
    """Token cost of the non-chunk prompt text plus the ``/no_think`` prefix.

    ``prompt_chars`` is the rendered length of everything the prompt carries
    besides the chunks, so per-call substitutions (concept instruction, entity
    list, source list) are charged against the budget. Without it the longest
    raw template stands in, measured rather than assumed so overriding a prompt
    from settings moves the chunk budget with it.
    """
    longest = max(
        len(config.wiki_synthesis_prompt),
        len(config.wiki_entity_batch_prompt),
    )
    chars = longest if prompt_chars is None else prompt_chars
    return -(-(chars + len(_NO_THINK_DIRECTIVE)) // _CHARS_PER_TOKEN)


def truncate_chunks_to_budget(
    chunks: list[SearchChunk],
    config: Config,
    prompt_chars: int | None = None,
) -> list[SearchChunk]:
    """Drop trailing chunks so the prompt plus its generation fits the context window.

    The budget is ``num_ctx`` less the output cap (``wiki_summary_max_tokens``)
    less the prompt overhead. The quarter-window floor applies only when that
    leaves nothing: a positive budget is used as-is, so the floor can never
    raise the budget past what the window actually has room for. Chunks are
    measured as :func:`chunks_to_text` formats them, numbering and separators
    included. Uses a chars/4 heuristic for token estimation.
    """
    context_window = config.num_ctx or DEFAULT_NUM_CTX
    overhead = prompt_overhead_tokens(config, prompt_chars)
    available = context_window - config.wiki_summary_max_tokens - overhead
    budget_tokens = (
        available if available > 0 else int(context_window * _MIN_CONTEXT_BUDGET_FRACTION)
    )
    budget_chars = budget_tokens * _CHARS_PER_TOKEN

    total_chars = 0
    kept: list[SearchChunk] = []
    for index, chunk in enumerate(chunks):
        chunk_chars = len(_format_chunk(chunk, index)) + len(_CHUNK_SEPARATOR)
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


def _format_chunk(chunk: SearchChunk, index: int) -> str:
    """Render one chunk as the numbered block the prompt embeds."""
    location = ""
    if chunk.page_start:
        location = f" (page {chunk.page_start})"
    elif chunk.line_start:
        location = f" (lines {chunk.line_start}-{chunk.line_end})"
    return f"[Chunk {index + 1}]{location}:\n{chunk.chunk}"


def chunks_to_text(chunks: list[SearchChunk]) -> str:
    """Format chunks as numbered text blocks for the LLM prompt."""
    return _CHUNK_SEPARATOR.join(_format_chunk(chunk, i) for i, chunk in enumerate(chunks))


def build_frontmatter(
    config: Config,
    source_names: list[str],
    score: float,
    chunks: list[SearchChunk] | None = None,
) -> str:
    """Build YAML frontmatter for a wiki page.

    When ``chunks`` is provided the frontmatter carries a ``provenance``
    block naming the source/chunk-index pairs that fed the generator and
    the extraction method from config, so a bad page is auditable
    without re-running the pipeline.
    """
    # A JSON array is valid YAML flow syntax and escapes quotes/backslashes/
    # unicode, so a filename like ``a"b\c.txt`` cannot corrupt the frontmatter.
    sources_yaml = json.dumps(sorted(source_names))
    provenance_block = render_provenance(config, chunks) if chunks is not None else ""
    return (
        f"---\n"
        f"generated_by: {config.chat_model}\n"
        f"generated_at: {datetime.now(UTC).isoformat()}\n"
        f"sources: {sources_yaml}\n"
        f"faithfulness_score: {score:.2f}\n"
        f"{provenance_block}"
        f"---\n\n"
    )


def write_page(
    wiki_root: Path,
    subdir: str,
    slug: str,
    full_content: str,
    drift_threshold: float,
    source_names: list[str],
    page_type: str,
) -> Path:
    """Write page to disk with drift detection. Returns path written to.

    ``slug`` may contain forward slashes (e.g. ``cv-manual/page-0042``);
    any intermediate directories are created before writing. Publishing
    retires this page's own drift draft: that proposal predates this body
    and accepting it would undo the regen. Another source's draft under the
    same slug is kept. A ``drafts/`` target skips drift entirely and routes
    through :func:`_write_draft_page`; ``page_type`` is the subdir the page
    would have published to, which a diverted page carries in its marker.
    """
    page_path = wiki_root / subdir / f"{slug}.md"
    drafts_dir = wiki_root / WikiSubdir.DRAFTS
    if subdir == WikiSubdir.DRAFTS:
        return _write_draft_page(page_path, drafts_dir, slug, full_content, source_names, page_type)

    if page_path.exists():
        old_content = page_path.read_text(encoding="utf-8")
        # Frontmatter carries a fresh timestamp and score on every regen, so the
        # ratio is taken over body prose only.
        ratio = content_change_ratio(extract_body(old_content), extract_body(full_content))
        if ratio > drift_threshold:
            diff_text = diff_summary(old_content, full_content)
            return divert_to_drafts(
                full_content, drafts_dir, slug, ratio, diff_text, subdir, source_names
            )

    atomic_write_text(page_path, full_content)
    delete_drift_draft_if_present(drafts_dir, slug, source_names)
    return page_path


def _write_draft_page(
    draft_path: Path,
    drafts_dir: Path,
    slug: str,
    full_content: str,
    source_names: list[str],
    page_type: str,
) -> Path:
    """Land a below-threshold page in ``drafts/``, keeping another source's draft.

    A newer proposal from the same sources supersedes the draft in place; one
    from different sources lands at a collision draft rather than overwriting a
    page awaiting review. No drift ratio is computed: a drafts target is a
    proposal, not a published body it could be drifting from.
    """
    owner = draft_source_names(draft_path)
    if owner is not None and owner != sorted(source_names):
        return divert_concept_collision(
            slug=slug,
            source=", ".join(sorted(source_names)),
            first_source=f"{WikiSubdir.DRAFTS}/{slug}.md",
            content=full_content,
            drafts_dir=drafts_dir,
            origin_subdir=page_type,
        )
    atomic_write_text(draft_path, f"{origin_marker(page_type)}\n\n{full_content}")
    return draft_path


def assemble_content(
    frontmatter: str,
    wiki_text: str,
    citation_block: str,
) -> str:
    """Combine frontmatter, body, and citations into the full page content."""
    full = frontmatter + wiki_text
    if citation_block:
        full += "\n\n" + citation_block
    return full


def indexable_chunks(content: str) -> list[str]:
    """The chunks a page's body would index as; empty means nothing to index.

    One definition of "indexable" for the indexer and for the accept-time
    refusal, because a non-empty body can still chunk to nothing ("#", "---")
    and the two must not disagree about that.
    """
    body = extract_body(content).strip()
    return chunk_text(body, mime_type="text/markdown", use_semantic=True) if body else []


def index_wiki_page(
    content: str,
    wiki_source: str,
    store: Store,
    config: Config,
    chunks: list[str] | None = None,
) -> int:
    """Chunk a wiki page body, embed it, and write rows with ``chunk_type="wiki"``.

    ``wiki_source`` must follow the ``<config.wiki_dir>/<subdir>/<slug>.md``
    shape (see :attr:`PageTarget.wiki_source`). Three branches:

    - subdir in :data:`WIKI_CONTENT_SUBDIRS`: chunk, embed, then swap
      the page's rows in one locked replace, so an embedder failure
      leaves the previous rows searchable. Returns the row count.
    - subdir is ``drafts/`` or ``archive/``: skip without touching the
      store. Returns 0.
    - malformed ``wiki_source`` (no subdir component): log.warning and
      return 0. Does not raise because the caller set is narrow (only
      internal wiki paths reach here) and surfacing the bad input in
      the log is sufficient triage.

    Record shape matches the markdown-ingest convention in
    ``lilbee.data.ingest``: ``content_type="text"``, all four page/line
    positions ``0`` (wiki pages are not paginated).

    ``chunks`` lets a caller that already chunked this content pass the result
    in rather than paying for a second semantic pass under the build lock.
    """
    subdir = subdir_from_wiki_source(wiki_source, config.wiki_dir)
    if subdir is None:
        log.warning("index_wiki_page: malformed wiki_source %r (no subdir)", wiki_source)
        return 0
    if subdir not in WIKI_CONTENT_SUBDIRS:
        return 0

    predicate = f"source = '{escape_sql_string(wiki_source)}' AND chunk_type = '{ChunkType.WIKI}'"
    if chunks is None:
        chunks = indexable_chunks(content)
    if not chunks:
        store.clear_table(CHUNKS_TABLE, predicate)
        return 0

    vectors = get_services().embedder.embed_batch(chunks)
    records = [
        {
            "source": wiki_source,
            "content_type": "text",
            "chunk_type": ChunkType.WIKI,
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
    return store.replace_chunks(records, predicate)


def generate_page(
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
    stats: BuildStats | None = None,
) -> Path | None:
    """Core generation pipeline shared by summary and synthesis pages."""
    stats = BuildStats.ensure(stats)

    def _emit(stage: str, **data: object) -> None:
        if on_progress is not None:
            on_progress(stage, data)

    _emit("preparing", chunks=len(chunks), source=label)

    messages = build_wiki_messages(prompt, provider, config)
    _emit("generating", source=label)
    options = wiki_generation_options(config)
    try:
        response = provider.chat(messages, stream=False, options=options)
        wiki_text = strip_reasoning(response.text).strip()
    except Exception as exc:
        log.warning("LLM failed to generate wiki page for %s: %s", label, exc)
        _emit("failed", error=str(exc))
        return None

    if not wiki_text:
        log.warning("LLM returned empty response for wiki page %s", label)
        _emit("failed", error="Model returned empty response")
        return None

    parsed_citations = parse_wiki_citations(wiki_text)
    resolved = citation_resolver(parsed_citations)
    verified = verify_citations(resolved, chunks, label, config)
    stats.record_citations(
        len(verified),
        len(parsed_citations) - wiki_sourced_count(resolved, config) - len(verified),
    )
    if not verified:
        log.warning("No valid citations for %s, skipping", label)
        _emit("failed", error="No valid citations found")
        return None

    _emit("faithfulness_check")
    score = check_faithfulness(chunks, wiki_text, label, config)
    threshold = config.wiki_embedding_faithfulness_threshold
    subdir = page_type if score >= threshold else WikiSubdir.DRAFTS
    if subdir == WikiSubdir.DRAFTS:
        log.info("Wiki page %s scored %.2f (< %.2f), sending to drafts", label, score, threshold)

    wiki_text = scrub_unverified_markers(strip_citation_block(wiki_text), verified)
    frontmatter = build_frontmatter(config, source_names, score, chunks=chunks)
    citation_block = render_citation_block(verified)
    full_content = assemble_content(frontmatter, wiki_text, citation_block)

    wiki_root = config.data_root / config.wiki_dir
    target = PageTarget(
        wiki_root=wiki_root,
        subdir=subdir,
        slug=slug,
        wiki_source=f"{config.wiki_dir}/{subdir}/{slug}.md",
        page_type=page_type,
        label=label,
    )
    page_path = persist_and_finalize(
        full_content, target, verified, source_names, store, config, stats=stats
    )

    log.info(
        "Generated wiki page for %s -> %s (score=%.2f, citations=%d)",
        label,
        target.subdir,
        score,
        len(verified),
    )
    return page_path
