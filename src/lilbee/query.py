"""RAG query pipeline — embed question, search, generate answer with citations."""

from collections.abc import Generator

import ollama

from lilbee import embedder, store
from lilbee.config import CHAT_MODEL, TOP_K

_SYSTEM_PROMPT = (
    "You are a helpful technical assistant. Answer questions using "
    "the provided context. Be specific — prefer exact numbers, part numbers, "
    "and measurements over vague references. Cite facts directly from the context. "
    "Do not make up information."
)

_CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}"""


def _format_source(result: dict) -> str:
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


def _deduplicate_sources(results: list[dict], max_citations: int = 5) -> list[str]:
    """Merge results from same source into deduplicated citation lines."""
    seen: set[str] = set()
    citations: list[str] = []
    for r in results:
        line = _format_source(r)
        if line not in seen:
            seen.add(line)
            citations.append(line)
            if len(citations) >= max_citations:
                break
    return citations


def _sort_by_relevance(results: list[dict]) -> list[dict]:
    """Sort search results by distance (lower = more relevant)."""
    return sorted(results, key=lambda r: r.get("_distance", float("inf")))


def _build_context(results: list[dict]) -> str:
    """Build context block from search results."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['chunk']}")
    return "\n\n".join(parts)


def search_context(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed question and return top-K matching chunks."""
    query_vec = embedder.embed(question)
    return store.search(query_vec, top_k=top_k)


def ask(question: str, top_k: int = TOP_K, history: list[dict] | None = None) -> str:
    """One-shot question: returns full answer with source citations."""
    results = search_context(question, top_k=top_k)
    if not results:
        return "No relevant documents found. Try ingesting some documents first."

    results = _sort_by_relevance(results)
    context = _build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
    )

    answer = response["message"]["content"]
    citations = _deduplicate_sources(results)
    return f"{answer}\n\nSources:\n" + "\n".join(citations)


def ask_stream(
    question: str, top_k: int = TOP_K, history: list[dict] | None = None
) -> Generator[str, None, None]:
    """Streaming question: yields answer tokens, then source citations."""
    results = search_context(question, top_k=top_k)
    if not results:
        yield "No relevant documents found. Try ingesting some documents first."
        return

    results = _sort_by_relevance(results)
    context = _build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
    )

    try:
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except (ConnectionError, OSError) as exc:
        yield f"\n\n[Connection lost: {exc}]"

    citations = _deduplicate_sources(results)
    yield "\n\nSources:\n" + "\n".join(citations)
