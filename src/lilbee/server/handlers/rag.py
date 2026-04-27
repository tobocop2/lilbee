"""Search, ask, and chat handlers (one-shot and streaming)."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING, Any, cast

from lilbee.cli.helpers import clean_result
from lilbee.core.config import cfg
from lilbee.core.results import DocumentResult, group
from lilbee.core.services import get_services
from lilbee.progress import SseEvent
from lilbee.server.handlers.sse import (
    SseStream,
    _resolve_generation_options,
    sse_done,
    sse_error,
    sse_event,
)
from lilbee.server.models import AskResponse, CleanedChunk

if TYPE_CHECKING:
    from lilbee.retrieval.query import ChatMessage

log = logging.getLogger(__name__)


async def search(q: str, top_k: int = 5, chunk_type: str | None = None) -> list[DocumentResult]:
    """Search and return grouped DocumentResults."""
    if not q or not q.strip():
        raise ValueError("query must not be empty")
    results = get_services().searcher.search(q, top_k=top_k, chunk_type=chunk_type)
    results = [r for r in results if r.distance is None or r.distance <= cfg.max_distance]
    return group(results)


async def ask(
    question: str,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AskResponse:
    """One-shot RAG answer. Returns answer and sources."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    opts = _resolve_generation_options(options)
    result = get_services().searcher.ask_raw(
        question, top_k=top_k, options=opts, chunk_type=chunk_type
    )
    return AskResponse(
        answer=result.answer,
        sources=[CleanedChunk(**clean_result(s)) for s in result.sources],
    )


def _run_llm_stream(
    messages: list[ChatMessage],
    opts: dict[str, Any] | None,
    queue: asyncio.Queue[str | None],
    cancel: threading.Event,
    error_holder: list[str],
) -> None:
    """Stream LLM tokens into a queue from a worker thread."""
    from lilbee.retrieval.reasoning import filter_reasoning

    try:
        provider = get_services().provider
        stream = provider.chat(
            cast("list[dict[str, Any]]", messages),
            stream=True,
            options=opts or None,
            model=cfg.chat_model,
        )
        for st in filter_reasoning(cast(Iterator[str], stream), show=cfg.show_reasoning):
            if cancel.is_set():
                break
            if st.content:
                event_type = SseEvent.REASONING if st.is_reasoning else SseEvent.TOKEN
                queue.put_nowait(sse_event(event_type, {"token": st.content}))
    except Exception as exc:
        error_holder.append(str(exc))
    finally:
        queue.put_nowait(None)


async def _stream_rag_response(
    question: str,
    history: list[ChatMessage] | None = None,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Shared SSE streaming for ask_stream and chat_stream."""
    yield ""  # force generator

    rag = get_services().searcher.build_rag_context(
        question, top_k=top_k, history=history, chunk_type=chunk_type
    )
    if rag is None:
        yield sse_error("No relevant documents found.")
        return

    results, messages = rag
    opts = _resolve_generation_options(options) or cfg.generation_options()

    sse = SseStream()
    error_holder: list[str] = []

    executor_fut = sse.loop.run_in_executor(
        None, _run_llm_stream, messages, opts, sse.queue, sse.cancel, error_holder
    )
    task = asyncio.ensure_future(executor_fut)
    async for event in sse.drain(task, "RAG stream"):
        yield event

    if error_holder:
        log.warning("Stream error: %s", error_holder[0])
        yield sse_error("Internal error")
        sse.cancel.set()
        return

    # Ensure executor thread has finished before yielding final events
    await executor_fut

    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in results])
    yield sse_done({})


def ask_stream(
    question: str,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events: token, sources, done."""
    return _stream_rag_response(question, top_k=top_k, options=options, chunk_type=chunk_type)


async def chat(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AskResponse:
    """Chat with history. Returns answer and sources."""
    opts = _resolve_generation_options(options)
    result = get_services().searcher.ask_raw(
        question, top_k=top_k, history=history, options=opts, chunk_type=chunk_type
    )
    return AskResponse(
        answer=result.answer,
        sources=[CleanedChunk(**clean_result(s)) for s in result.sources],
    )


def chat_stream(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events with chat history support."""
    return _stream_rag_response(
        question, history=history, top_k=top_k, options=options, chunk_type=chunk_type
    )
