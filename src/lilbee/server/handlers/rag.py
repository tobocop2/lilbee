"""Search, ask, and chat handlers (one-shot and streaming)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.results import DocumentResult, group
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers.sse import (
    SseStream,
    _resolve_generation_options,
    classify_load_error,
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


_CAP_NOTICE_TEMPLATE = "\n[reasoning capped at {chars} chars, asking for a direct answer]\n"
_CAP_CONTINUATION_PROMPT = (
    "Stop thinking now. Give your final answer directly, without any further <think> blocks."
)


def _resolve_reasoning_cap() -> int:
    """Effective reasoning cap: per-model override beats the global setting."""
    defaults = cfg.model_defaults
    override = defaults.max_reasoning_chars if defaults is not None else None
    return override if isinstance(override, int) and override > 0 else cfg.max_reasoning_chars


def _stream_continuation(
    messages: list[ChatMessage],
    opts: dict[str, Any] | None,
    queue: asyncio.Queue[str | None],
    cancel: threading.Event,
    captured_reasoning: str,
) -> None:
    """Re-issue the chat with a 'stop thinking' nudge after the cap fires.

    Sends the full original turn back to the model with the partial reasoning
    surfaced as the previous assistant turn so the user can read what was
    captured, then a fresh user message asking for a direct answer. Tokens
    from the second pass stream as plain TOKEN events.
    """
    provider = get_services().provider
    nudged_messages: list[dict[str, Any]] = [
        *cast("list[dict[str, Any]]", messages),
        {"role": "assistant", "content": f"<think>{captured_reasoning}</think>"},
        {"role": "user", "content": _CAP_CONTINUATION_PROMPT},
    ]
    second_stream = provider.chat(
        nudged_messages,
        stream=True,
        options=opts or None,
        model=cfg.chat_model,
    )
    try:
        for chunk in second_stream:
            if cancel.is_set():
                break
            if chunk:
                queue.put_nowait(sse_event(SseEvent.TOKEN, {"token": chunk}))
    finally:
        with contextlib.suppress(Exception):
            second_stream.close()


def _run_llm_stream(
    messages: list[ChatMessage],
    opts: dict[str, Any] | None,
    queue: asyncio.Queue[str | None],
    cancel: threading.Event,
    error_holder: list[str],
) -> None:
    """Stream LLM tokens into a queue from a worker thread."""
    from lilbee.retrieval.reasoning import filter_reasoning

    cap_chars = _resolve_reasoning_cap()
    cap_holder: list[str] = []

    try:
        provider = get_services().provider
        stream = provider.chat(
            cast("list[dict[str, Any]]", messages),
            stream=True,
            options=opts or None,
            model=cfg.chat_model,
        )
        for st in filter_reasoning(
            stream,
            show=cfg.show_reasoning,
            cap_chars=cap_chars,
            on_cap=cap_holder.append,
        ):
            if cancel.is_set():
                break
            if st.content:
                event_type = SseEvent.REASONING if st.is_reasoning else SseEvent.TOKEN
                queue.put_nowait(sse_event(event_type, {"token": st.content}))

        if cap_holder and not cancel.is_set():
            queue.put_nowait(
                sse_event(
                    SseEvent.REASONING,
                    {"token": _CAP_NOTICE_TEMPLATE.format(chars=cap_chars)},
                )
            )
            _stream_continuation(messages, opts, queue, cancel, cap_holder[0])
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
        raw = error_holder[0]
        code, user_message = classify_load_error(raw)
        log.warning("Stream error: %s", raw)
        yield sse_error(user_message, code=code, detail=raw if code else None)
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
