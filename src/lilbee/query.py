"""RAG query pipeline — embed question, search, generate answer with citations."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import ollama
from pydantic import BaseModel
from typing_extensions import TypedDict

from lilbee import embedder, store
from lilbee.config import cfg
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
    source = result["source"]
    content_type = result["content_type"]

    if content_type == "pdf":
        ps, pe = result["page_start"], result["page_end"]
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"  → {source}, {pages}"

    if content_type == "code":
        ls, le = result["line_start"], result["line_end"]
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"  → {source}, {lines}"

    return f"  → {source}"


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

    Hybrid results have _relevance_score (higher = better) → negate.
    Vector results have _distance (lower = better) → use directly.
    """
    if "_relevance_score" in r:
        return -r["_relevance_score"]
    return r.get("_distance", float("inf"))


def sort_by_relevance(results: list[SearchChunk]) -> list[SearchChunk]:
    """Sort search results by relevance (works for both hybrid and vector results)."""
    return sorted(results, key=_sort_key)


def build_context(results: list[SearchChunk]) -> str:
    """Build context block from search results."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['chunk']}")
    return "\n\n".join(parts)


def search_context(question: str, top_k: int = 0) -> list[SearchChunk]:
    """Embed question and return top-K matching chunks."""
    if top_k == 0:
        top_k = cfg.top_k
    query_vec = embedder.embed(question)
    return store.search(query_vec, top_k=top_k, query_text=question)


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

    results = sort_by_relevance(results)
    context = build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[ChatMessage] = [{"role": "system", "content": cfg.system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    opts = options if options is not None else cfg.generation_options()
    try:
        response = ollama.chat(model=cfg.chat_model, messages=messages, options=opts or None)
    except ollama.ResponseError as exc:
        raise RuntimeError(
            f"Model '{cfg.chat_model}' not found in Ollama. Run: ollama pull {cfg.chat_model}"
        ) from exc
    return AskResult(answer=response.message.content or "", sources=results)


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
) -> Generator[str, None, None]:
    """Streaming question: yields answer tokens, then source citations."""
    results = search_context(question, top_k=top_k)
    if not results:
        yield "No relevant documents found. Try ingesting some documents first."
        return

    results = sort_by_relevance(results)
    context = build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[ChatMessage] = [{"role": "system", "content": cfg.system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    opts = options if options is not None else cfg.generation_options()
    try:
        stream = ollama.chat(
            model=cfg.chat_model,
            messages=messages,
            stream=True,
            options=opts or None,
        )
    except ollama.ResponseError as exc:
        raise RuntimeError(
            f"Model '{cfg.chat_model}' not found in Ollama. Run: ollama pull {cfg.chat_model}"
        ) from exc

    try:
        for chunk in stream:
            token = chunk.message.content
            if token:
                yield token
    except ollama.ResponseError as exc:
        raise RuntimeError(
            f"Model '{cfg.chat_model}' not found in Ollama. Run: ollama pull {cfg.chat_model}"
        ) from exc
    except (ConnectionError, OSError) as exc:
        yield f"\n\n[Connection lost: {exc}]"

    citations = deduplicate_sources(results)
    yield "\n\nSources:\n" + "\n".join(citations)
