"""RAG query pipeline — embed question, search, generate answer with citations."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from lilbee.reasoning import StreamToken

from pydantic import BaseModel
from typing_extensions import TypedDict

from lilbee import embedder, store
from lilbee.config import cfg
from lilbee.providers import get_provider
from lilbee.store import SearchChunk


class ChatMessage(TypedDict):
    """A single chat message with role and content."""

    role: str
    content: str


CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


def format_source(result: SearchChunk) -> str:
    """Format a search result as a source citation line."""
    if result.content_type == "pdf":
        ps, pe = result.page_start, result.page_end
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"  → {result.source}, {pages}"

    if result.content_type == "code":
        ls, le = result.line_start, result.line_end
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"  → {result.source}, {lines}"

    return f"  → {result.source}"


def deduplicate_sources(results: list[SearchChunk], max_citations: int = 5) -> list[str]:
    """Merge results from same source into deduplicated citation lines."""
    seen: set[str] = set()
    citations: list[str] = []
    for r in results:
        line = format_source(r)
        if line not in seen:
            seen.add(line)
            citations.append(line)
            if len(citations) >= max_citations:
                break
    return citations


def _sort_key(r: SearchChunk) -> float:
    """Sort key: lower = more relevant.

    Hybrid results have relevance_score (higher = better) → negate.
    Vector results have distance (lower = better) → use directly.
    """
    if r.relevance_score is not None:
        return -r.relevance_score
    if r.distance is not None:
        return r.distance
    return float("inf")


def sort_by_relevance(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort search results by relevance (works for both hybrid and vector results)."""
    return sorted(results, key=_sort_key)


def diversify_sources(
    results: list[SearchChunk], max_per_source: int | None = None
) -> list[SearchChunk]:
    """Cap results per source document to ensure diversity.

    Standard IR diversity technique; see Zhai 2008,
    "Towards a Game-Theoretic Framework for Information Retrieval."
    """
    if max_per_source is None:
        max_per_source = cfg.diversity_max_per_source
    counts: dict[str, int] = {}
    diverse: list[SearchChunk] = []
    for r in results:
        count = counts.get(r.source, 0)
        if count < max_per_source:
            diverse.append(r)
            counts[r.source] = count + 1
    return diverse


def _apply_temporal_filter(results: list[SearchChunk], question: str) -> list[SearchChunk]:
    """Filter results by date range if temporal keywords detected in question."""
    if not cfg.temporal_filtering:
        return results
    from lilbee.temporal import detect_temporal, resolve_date_range

    keyword = detect_temporal(question)
    if keyword is None:
        return results
    date_range = resolve_date_range(keyword)
    # Filter by ingestion date (stored in sources table, not on chunks directly).
    # Since chunks don't carry dates, we filter by source ingestion time via store.
    source_dates = {s["filename"]: s.get("ingested_at", "") for s in store.get_sources()}
    filtered: list[SearchChunk] = []
    for r in results:
        ingested_at = source_dates.get(r.source, "")
        if not ingested_at:
            filtered.append(r)  # keep if no date info
            continue
        try:
            doc_date = datetime.fromisoformat(ingested_at)
            if date_range.start <= doc_date <= date_range.end:
                filtered.append(r)
        except (ValueError, TypeError):
            filtered.append(r)  # keep on parse failure
    return filtered if filtered else results  # fall back to unfiltered if nothing matches


def prepare_results(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort by relevance and apply source diversity cap."""
    return diversify_sources(sort_by_relevance(results))


def build_context(results: list[SearchChunk]) -> str:
    """Build context block from search results."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.chunk}")
    return "\n\n".join(parts)


# Multi-query expansion generates alternative search phrasings to
# improve recall. Based on standard multi-query retrieval techniques.
_EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

# Max tokens the LLM can generate for expansion variants.
# 200 tokens is enough for 3 one-line query variants (~50 tokens each
# plus formatting). Not configurable — this is an internal budget.
_EXPANSION_MAX_TOKENS = 200


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "after",
        "before",
        "above",
        "below",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "then",
        "than",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
    }
)

# Minimum token overlap ratio between variant and original query.
# Below 0.3, the variant has drifted too far from the original question.
_MIN_OVERLAP_RATIO = 0.3


def _tokenize_query(text: str) -> set[str]:
    """Tokenize into lowercase words, removing stop words."""
    return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 1}


def _apply_guardrails(variants: list[str], question: str) -> list[str]:
    """Validate expansion variants to prevent drift.

    Drops variants with less than 30% token overlap with the original query.
    Falls back to empty list if all variants are filtered.
    """
    if not cfg.expansion_guardrails:
        return variants

    original_tokens = _tokenize_query(question)
    if not original_tokens:
        return variants

    validated: list[str] = []
    for variant in variants:
        variant_tokens = _tokenize_query(variant)
        if not variant_tokens:
            continue
        overlap = len(original_tokens & variant_tokens) / len(original_tokens)
        if overlap >= _MIN_OVERLAP_RATIO:
            validated.append(variant)

    return validated


def _expand_query(question: str) -> list[str]:
    """Use the LLM to generate alternative query phrasings.

    Returns up to ``cfg.query_expansion_count`` variants.
    When ``cfg.expansion_guardrails`` is True, validates variants for drift.
    Set ``LILBEE_QUERY_EXPANSION_COUNT=0`` to disable expansion entirely.
    """
    count = cfg.query_expansion_count
    if count == 0:
        return []
    try:
        provider = get_provider()
        prompt = _EXPANSION_PROMPT.format(count=count, question=question)
        messages = [{"role": "user", "content": prompt}]
        response = provider.chat(
            messages, stream=False, options={"num_predict": _EXPANSION_MAX_TOKENS}
        )
        if not isinstance(response, str):
            return []
        variants = [line.strip() for line in response.strip().split("\n") if line.strip()]
        variants = variants[:count]
        variants = _apply_guardrails(variants, question)
        variants.extend(_concept_query_expansion(question))
        return variants
    except Exception:
        return []


def _should_skip_expansion(question: str) -> bool:
    """Check if BM25 results are confident enough to skip query expansion.

    Probes the FTS index with the original query. If the top result
    scores above ``cfg.expansion_skip_threshold`` and the gap to the
    second result exceeds ``cfg.expansion_skip_gap``, expansion is
    skipped to save an LLM call.

    Based on early termination patterns standard in search engines
    (Lucene/Elasticsearch). Thresholds derived from sigmoid-normalized
    BM25 score distribution (90th percentile).
    """
    if cfg.expansion_skip_threshold <= 0:
        return False
    results = store.bm25_probe(question, top_k=2)
    if not results:
        return False
    top_score = results[0].relevance_score or 0
    if top_score < cfg.expansion_skip_threshold:
        return False
    if len(results) < 2:
        return True  # only one result and it's confident
    second_score = results[1].relevance_score or 0
    return (top_score - second_score) >= cfg.expansion_skip_gap


def select_context(
    results: list[SearchChunk], question: str, max_sources: int | None = None
) -> list[SearchChunk]:
    """Select chunks that maximize query term coverage.

    Greedy set-cover: pick chunks adding the most uncovered query terms.
    Falls through unchanged if results ≤ max_sources.
    """
    if max_sources is None:
        max_sources = cfg.max_context_sources
    if len(results) <= max_sources:
        return results

    query_terms = _tokenize_query(question)
    if not query_terms:
        return results[:max_sources]

    selected: list[SearchChunk] = []
    covered: set[str] = set()
    remaining = list(results)

    for _ in range(max_sources):
        if not remaining or covered == query_terms:
            break
        best_idx = 0
        best_gain = -1
        for i, chunk in enumerate(remaining):
            chunk_terms = _tokenize_query(chunk.chunk)
            gain = len((chunk_terms & query_terms) - covered)
            if gain > best_gain or (gain == best_gain and i < best_idx):
                best_gain = gain
                best_idx = i
        if best_gain <= 0 and selected:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered |= _tokenize_query(chosen.chunk) & query_terms

    return selected


_HYDE_PROMPT = (
    "Write a 50-100 word passage that directly answers this question as if "
    "it were an excerpt from a real document. Do not include any preamble, "
    "just write the passage.\n\nQuestion: {question}"
)


def _hyde_search(question: str, top_k: int) -> list[SearchChunk]:
    """HyDE: generate a hypothetical document, embed it, search with it.

    Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without
    Relevance Labels" (https://arxiv.org/abs/2212.10496).

    Returns results from the hypothetical embedding search, or empty
    list on failure. Results should be weighted at ``cfg.hyde_weight``
    relative to original search results.
    """
    try:
        provider = get_provider()
        response = provider.chat(
            [{"role": "user", "content": _HYDE_PROMPT.format(question=question)}],
            stream=False,
            options={"num_predict": _EXPANSION_MAX_TOKENS},
        )
        if not isinstance(response, str) or not response.strip():
            return []
        hyde_vec = embedder.embed(response.strip())
        return store.search(hyde_vec, top_k=top_k, query_text=None)
    except Exception:
        return []


def _parse_structured_query(question: str) -> tuple[str | None, str]:
    """Parse structured query mode prefix.

    Returns (mode, query) where mode is "term", "vec", "hyde", or None.
    """
    for prefix in ("term:", "vec:", "hyde:"):
        if question.strip().lower().startswith(prefix):
            return prefix[:-1], question.strip()[len(prefix) :].strip()
    return None, question


def _search_structured(mode: str, query: str, top_k: int) -> list[SearchChunk]:
    """Execute a structured query mode search."""
    if mode == "term":
        return store.bm25_probe(query, top_k=top_k)
    if mode == "vec":
        query_vec = embedder.embed(query)
        return store.search(query_vec, top_k=top_k, query_text=None)
    if mode == "hyde":
        hyde_results = _hyde_search(query, top_k)
        return hyde_results
    return []


def _apply_concept_boost(results: list[SearchChunk], question: str) -> list[SearchChunk]:
    """Boost search results by concept overlap. No-op if disabled."""
    if not cfg.concept_graph:
        return results
    try:
        from lilbee.concepts import extract_concepts, get_graph

        graph = get_graph()
        if graph is None:
            return results
        query_concepts = extract_concepts(question)
        return graph.boost_results(results, query_concepts)
    except Exception:
        return results


def _concept_query_expansion(question: str) -> list[str]:
    """Get additional query terms from concept graph. Returns empty on failure."""
    if not cfg.concept_graph:
        return []
    try:
        from lilbee.concepts import get_graph

        graph = get_graph()
        if graph is None:
            return []
        return graph.expand_query(question)
    except Exception:
        return []


def search_context(question: str, top_k: int = 0) -> list[SearchChunk]:
    """Embed question and return top-K matching chunks.

    Supports structured query modes (``term:``, ``vec:``, ``hyde:`` prefixes)
    and query expansion with confidence-based skipping.
    """
    if top_k == 0:
        top_k = cfg.top_k

    # Check for structured query mode
    mode, clean_query = _parse_structured_query(question)
    if mode is not None:
        return _search_structured(mode, clean_query, top_k)

    query_vec = embedder.embed(question)
    results = store.search(query_vec, top_k=top_k, query_text=question)

    # Skip expansion if BM25 already found a confident match
    if _should_skip_expansion(question):
        return results[: top_k * 2]

    seen = {(r.source, r.chunk_index) for r in results}
    variants = _expand_query(question)
    if variants:
        for variant in variants:
            variant_vec = embedder.embed(variant)
            variant_results = store.search(variant_vec, top_k=top_k, query_text=variant)
            for r in variant_results:
                key = (r.source, r.chunk_index)
                if key not in seen:
                    results.append(r)
                    seen.add(key)

    # HyDE: search with hypothetical document embedding (if enabled).
    # Discount distances by hyde_weight since these are from a fabricated passage.
    if cfg.hyde:
        hyde_results = _hyde_search(question, top_k)
        for r in hyde_results:
            key = (r.source, r.chunk_index)
            if key not in seen:
                if r.distance is not None and cfg.hyde_weight > 0:
                    r.distance = r.distance / cfg.hyde_weight
                results.append(r)
                seen.add(key)

    results = _apply_concept_boost(results, question)

    # Cap total results to prevent context overflow from expansion
    return results[: top_k * 2]


class AskResult(BaseModel):
    """Structured result from ask_raw — answer text + raw search results."""

    answer: str
    sources: list[SearchChunk]


def _build_rag_context(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
) -> tuple[list[SearchChunk], list[ChatMessage]] | None:
    """Shared RAG pipeline: search → prepare → rerank → filter → select → build.

    Returns (results, messages) or None if no results found.
    """
    results = search_context(question, top_k=top_k)
    if not results:
        return None

    results = prepare_results(results)
    if cfg.reranker_model:
        from lilbee.reranker import rerank

        results = rerank(question, results)
    results = _apply_temporal_filter(results, question)
    results = select_context(results, question)
    context = build_context(results)
    prompt = CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[ChatMessage] = [{"role": "system", "content": cfg.system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return results, messages


def ask_raw(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> AskResult:
    """One-shot question returning structured answer + raw sources."""
    rag = _build_rag_context(question, top_k=top_k, history=history)
    if rag is None:
        return AskResult(
            answer="No relevant documents found. Try ingesting some documents first.",
            sources=[],
        )

    results, messages = rag
    opts = options if options is not None else cfg.generation_options()
    provider = get_provider()
    answer = provider.chat(cast(list[dict[str, Any]], messages), options=opts or None)
    return AskResult(answer=str(answer) or "", sources=results)


def ask(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """One-shot question: returns full answer with source citations."""
    result = ask_raw(question, top_k=top_k, history=history, options=options)
    if not result.sources:
        return result.answer
    citations = deduplicate_sources(result.sources)
    return f"{result.answer}\n\nSources:\n" + "\n".join(citations)


def ask_stream(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> Generator[StreamToken, None, None]:
    """Streaming question: yields classified tokens, then source citations.

    Each yielded ``StreamToken`` has ``.content`` (text) and ``.is_reasoning``
    (True for ``<think>...</think>`` blocks when ``cfg.show_reasoning`` is True).
    """
    from lilbee.reasoning import StreamToken, filter_reasoning

    rag = _build_rag_context(question, top_k=top_k, history=history)
    if rag is None:
        yield StreamToken(
            content="No relevant documents found. Try ingesting some documents first.",
            is_reasoning=False,
        )
        return

    results, messages = rag
    opts = options if options is not None else cfg.generation_options()
    provider = get_provider()
    raw_stream = provider.chat(
        cast(list[dict[str, Any]], messages), stream=True, options=opts or None
    )

    try:
        for st in filter_reasoning(cast(Iterator[str], raw_stream), show=cfg.show_reasoning):
            if st.content:
                yield st
    except (ConnectionError, OSError) as exc:
        yield StreamToken(content=f"\n\n[Connection lost: {exc}]", is_reasoning=False)

    citations = deduplicate_sources(results)
    yield StreamToken(content="\n\nSources:\n" + "\n".join(citations), is_reasoning=False)
