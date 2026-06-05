"""Search, ask, and chat handlers (one-shot and streaming)."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from lilbee.app.memory import auto_extract, auto_extract_enabled
from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.results import DocumentResult, group
from lilbee.data.store import ChunkType, EmbeddingModelMismatchError
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.retrieval.reasoning import (
    CAP_NOTICE_TEMPLATE,
    CapNotice,
    effective_reasoning_cap,
    stream_chat_with_cap,
)
from lilbee.runtime.progress import SseErrorCode, SseEvent
from lilbee.server.handlers.sse import (
    SseStream,
    _resolve_generation_options,
    classify_load_error,
    sse_done,
    sse_error,
    sse_event,
)
from lilbee.server.models import (
    AskResponse,
    CleanedChunk,
    MemoryExtractedEvent,
    MemoryExtractedItem,
)

if TYPE_CHECKING:
    from lilbee.retrieval.query import ChatMessage

log = logging.getLogger(__name__)


async def search(
    q: str, top_k: int = 5, chunk_type: ChunkType | None = None
) -> list[DocumentResult]:
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
    chunk_type: ChunkType | None = None,
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
    error_holder: list[Exception],
    answer_parts: list[str],
) -> None:
    """Forward tokens from the cap-aware chat orchestrator into the SSE queue.

    Answer tokens (not reasoning) are also accumulated into *answer_parts* so the
    caller can feed the finished answer to auto-extraction.
    """
    try:
        events = stream_chat_with_cap(
            get_services().provider,
            cast("list[dict[str, Any]]", messages),
            options=opts,
            model=cfg.chat_model,
            show_reasoning=cfg.show_reasoning,
            cap_chars=effective_reasoning_cap(),
        )
        for event in events:
            if cancel.is_set():
                events.close()
                break
            if isinstance(event, CapNotice):
                queue.put_nowait(
                    sse_event(
                        SseEvent.REASONING,
                        {"token": CAP_NOTICE_TEMPLATE.format(chars=event.cap_chars)},
                    )
                )
            elif event.content:
                kind = SseEvent.REASONING if event.is_reasoning else SseEvent.TOKEN
                if kind is SseEvent.TOKEN:
                    answer_parts.append(event.content)
                queue.put_nowait(sse_event(kind, {"token": event.content}))
    except Exception as exc:
        error_holder.append(exc)
    finally:
        queue.put_nowait(None)


async def _emit_extracted_memories(question: str, answer: str) -> AsyncGenerator[str, None]:
    """Yield a ``memory_extracted`` SSE event if the turn auto-saved any memories.

    Runs the extraction LLM pass off the event loop. Silent (yields nothing)
    when the answer is empty, auto-extraction is off, or nothing was extracted,
    so existing consumers are unaffected.
    """
    if not answer or not auto_extract_enabled():
        return
    stored = await asyncio.to_thread(auto_extract, question, answer)
    if not stored:
        return
    event = MemoryExtractedEvent(
        count=len(stored),
        items=[MemoryExtractedItem(id=m.id, kind=m.kind, text=m.text) for m in stored],
    )
    yield sse_event(SseEvent.MEMORY_EXTRACTED, event.model_dump(mode="json"))


def _error_event(exc: Exception) -> str:
    """Build the SSE error event for a stream failure.

    Provider errors already carry a user-facing message (rate limits, auth,
    bad model), so surface it verbatim. Everything else goes through the
    llama.cpp OOM classifier and otherwise collapses to a generic message.
    """
    if isinstance(exc, ProviderError):
        log.warning("Provider error during stream: %s", exc)
        kind_code = exc.kind if exc.kind is not ProviderErrorKind.UNKNOWN else None
        return sse_error(str(exc), code=kind_code)
    raw = str(exc)
    code, user_message = classify_load_error(raw)
    log.warning("Stream error: %s", raw)
    return sse_error(user_message, code=code, detail=raw if code else None)


async def _stream_rag_response(
    question: str,
    history: list[ChatMessage] | None = None,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: ChunkType | None = None,
) -> AsyncGenerator[str, None]:
    """Shared SSE streaming for ask_stream and chat_stream."""
    yield ""  # force generator

    try:
        rag = get_services().searcher.build_rag_context(
            question, top_k=top_k, history=history, chunk_type=chunk_type
        )
    except EmbeddingModelMismatchError as exc:
        yield sse_error(str(exc), code=SseErrorCode.INDEX_EMBEDDER_MISMATCH)
        return
    if rag is None:
        yield sse_error("No relevant documents found.")
        return

    results, messages = rag
    opts = _resolve_generation_options(options) or cfg.generation_options()

    sse = SseStream()
    error_holder: list[Exception] = []
    answer_parts: list[str] = []

    executor_fut = sse.loop.run_in_executor(
        None, _run_llm_stream, messages, opts, sse.queue, sse.cancel, error_holder, answer_parts
    )
    task = asyncio.ensure_future(executor_fut)
    async for event in sse.drain(task, "RAG stream"):
        yield event

    if error_holder:
        yield _error_event(error_holder[0])
        sse.cancel.set()
        return

    # Ensure executor thread has finished before yielding final events
    await executor_fut

    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in results])
    yield sse_done({})

    # Auto-extraction (and its notification) trails ``done`` so clients that stop
    # at ``done`` are unaffected; the memories are stored regardless.
    async for event in _emit_extracted_memories(question, "".join(answer_parts)):
        yield event


def ask_stream(
    question: str,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: ChunkType | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events: token, sources, done."""
    return _stream_rag_response(question, top_k=top_k, options=options, chunk_type=chunk_type)


async def chat(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: ChunkType | None = None,
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
    chunk_type: ChunkType | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events with chat history support."""
    return _stream_rag_response(
        question, history=history, top_k=top_k, options=options, chunk_type=chunk_type
    )
