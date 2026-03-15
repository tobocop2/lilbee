"""RAG query pipeline — embed question, search, generate answer with citations."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import ollama

from lilbee import embedder, store
from lilbee.config import cfg

_CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


def format_source(result: dict) -> str:
    """Format a search result as a source citation line."""
    source = result.get("source", "unknown")
    content_type = result.get("content_type", "")

    if content_type == "pdf":
        ps, pe = result.get("page_start", 0), result.get("page_end", 0)
        pages = f"page {ps}" if ps == pe else f"pages {ps}-{pe}"
        return f"  → {source}, {pages}"

    if content_type == "code":
        ls, le = result.get("line_start", 0), result.get("line_end", 0)
        lines = f"line {ls}" if ls == le else f"lines {ls}-{le}"
        return f"  → {source}, {lines}"

    return f"  → {source}"


def deduplicate_sources(results: list[dict], max_citations: int = 5) -> list[str]:
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


def sort_by_relevance(results: list[dict]) -> list[dict]:
    """Sort search results by distance (lower = more relevant)."""
    return sorted(results, key=lambda r: r.get("_distance", float("inf")))


def build_context(results: list[dict]) -> str:
    """Build context block from search results."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['chunk']}")
    return "\n\n".join(parts)


def search_context(question: str, top_k: int = 0) -> list[dict]:
    """Embed question and return top-K matching chunks."""
    if top_k == 0:
        top_k = cfg.top_k
    query_vec = embedder.embed(question)
    return store.search(query_vec, top_k=top_k)


@dataclass
class AskResult:
    """Structured result from ask_raw — answer text + raw search results."""

    answer: str
    sources: list[dict]


def ask_raw(
    question: str,
    top_k: int = 0,
    history: list[dict] | None = None,
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

    messages: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
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
    history: list[dict] | None = None,
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
    history: list[dict] | None = None,
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

    messages: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
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
