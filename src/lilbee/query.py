"""RAG query pipeline — embed question, search, generate answer with citations."""

from collections.abc import Generator

import ollama

from lilbee import embedder, store
from lilbee.config import CHAT_MODEL, TOP_K

_SYSTEM_PROMPT = (
    "You are a helpful technical assistant. Answer questions based ONLY on "
    "the provided context.\n"
    "If the context doesn't contain enough information to answer, say so.\n"
    "Be specific and cite facts from the context. Do not make up information."
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


def _deduplicate_sources(results: list[dict]) -> list[str]:
    """Merge results from same source into deduplicated citation lines."""
    seen: set[str] = set()
    citations: list[str] = []
    for r in results:
        line = _format_source(r)
        if line not in seen:
            seen.add(line)
            citations.append(line)
    return citations


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


def ask(question: str, top_k: int = TOP_K) -> str:
    """One-shot question: returns full answer with source citations."""
    results = search_context(question, top_k=top_k)
    if not results:
        return "No relevant documents found. Try ingesting some documents first."

    context = _build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response["message"]["content"]
    citations = _deduplicate_sources(results)
    return f"{answer}\n\nSources:\n" + "\n".join(citations)


def ask_stream(question: str, top_k: int = TOP_K) -> Generator[str, None, None]:
    """Streaming question: yields answer tokens, then source citations."""
    results = search_context(question, top_k=top_k)
    if not results:
        yield "No relevant documents found. Try ingesting some documents first."
        return

    context = _build_context(results)
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token

    citations = _deduplicate_sources(results)
    yield "\n\nSources:\n" + "\n".join(citations)
