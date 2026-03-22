"""RAG query pipeline — embed question, search, generate answer with citations."""

from __future__ import annotations

from collections.abc import Generator, Iterator
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


_CONTEXT_TEMPLATE = """Context:
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

# Token budget for the expansion LLM call
_EXPANSION_MAX_TOKENS = 200


def _expand_query(question: str) -> list[str]:
    """Use the LLM to generate alternative query phrasings.

    Returns up to ``cfg.query_expansion_count`` variants.
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
        return variants[:count]
    except Exception:
        return []


def search_context(question: str, top_k: int = 0) -> list[SearchChunk]:
    """Embed question and return top-K matching chunks.

    Uses query expansion: generates alternative phrasings via LLM,
    searches with each, and merges results (deduped by source+index).
    """
    if top_k == 0:
        top_k = cfg.top_k
    query_vec = embedder.embed(question)
    results = store.search(query_vec, top_k=top_k, query_text=question)

    variants = _expand_query(question)
    if variants:
        seen = {(r.source, r.chunk_index) for r in results}
        for variant in variants:
            variant_vec = embedder.embed(variant)
            variant_results = store.search(variant_vec, top_k=top_k, query_text=variant)
            for r in variant_results:
                key = (r.source, r.chunk_index)
                if key not in seen:
                    results.append(r)
                    seen.add(key)

    # Cap total results to prevent context overflow from expansion
    return results[: top_k * 2]


class AskResult(BaseModel):
    """Structured result from ask_raw — answer text + raw search results."""

    answer: str
    sources: list[SearchChunk]


def ask_raw(
    question: str,
    top_k: int = 0,
    history: list[ChatMessage] | None = None,
    options: dict[str, Any] | None = None,
) -> AskResult:
    """One-shot question returning structured answer + raw sources."""
    results = search_context(question, top_k=top_k)
    if not results:
        return AskResult(
            answer="No relevant documents found. Try ingesting some documents first.",
            sources=[],
        )

    results = prepare_results(results)
    context = build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[ChatMessage] = [{"role": "system", "content": cfg.system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

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

    results = search_context(question, top_k=top_k)
    if not results:
        yield StreamToken(
            content="No relevant documents found. Try ingesting some documents first.",
            is_reasoning=False,
        )
        return

    results = prepare_results(results)
    context = build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[ChatMessage] = [{"role": "system", "content": cfg.system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

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
