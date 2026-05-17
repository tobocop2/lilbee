"""Search, ask, and chat handlers (one-shot and streaming)."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.core.results import DocumentResult, group
from lilbee.retrieval.reasoning import (
    CAP_NOTICE_TEMPLATE,
    CapNotice,
    effective_reasoning_cap,
    stream_chat_with_cap,
    strip_reasoning,
)
from lilbee.runtime.progress import SseEvent
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    ContentBlockDelta,
    TextBlock,
    TextDelta,
)
from lilbee.server.chat_dispatch.dispatch import (
    dispatch_chat,
    dispatch_chat_stream,
)
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
    from lilbee.core.results import SearchChunk
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
    """Forward tokens from the cap-aware chat orchestrator into the SSE queue."""
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
                queue.put_nowait(sse_event(kind, {"token": event.content}))
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
    """Chat with history. Returns answer and sources via canonical dispatch."""
    sources, messages = _build_chat_messages(question, history, top_k, chunk_type)
    req = _build_canonical_request(messages, options)
    response = await asyncio.to_thread(dispatch_chat, req)
    text = _join_text_blocks(response.content)
    answer = text if cfg.show_reasoning else strip_reasoning(text)
    return AskResponse(
        answer=answer,
        sources=[CleanedChunk(**clean_result(s)) for s in sources],
    )


def chat_stream(
    question: str,
    history: list[ChatMessage],
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream RAG chat tokens through canonical dispatch as token/sources/done events."""
    return _stream_chat_response(
        question, history=history, top_k=top_k, options=options, chunk_type=chunk_type
    )


async def _stream_chat_response(
    question: str,
    history: list[ChatMessage],
    top_k: int,
    options: dict[str, Any] | None,
    chunk_type: str | None,
) -> AsyncGenerator[str, None]:
    """Drive ``dispatch_chat_stream`` and emit token/sources/done SSE events."""
    rag = get_services().searcher.build_rag_context(
        question, top_k=top_k, history=history, chunk_type=chunk_type
    )
    if rag is None:
        yield sse_error("No relevant documents found.")
        return
    sources, messages = rag

    req = _build_canonical_request(messages, options)
    canonical_stream = dispatch_chat_stream(req)
    try:
        async for event in canonical_stream:
            text = _text_from_event(event)
            if text:
                yield sse_event(SseEvent.TOKEN, {"token": text})
    except Exception as exc:
        raw = str(exc)
        code, user_message = classify_load_error(raw)
        log.warning("Stream error: %s", raw)
        yield sse_error(user_message, code=code, detail=raw if code else None)
        return

    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in sources])
    yield sse_done({})


def _text_from_event(event: Any) -> str:
    """Return the text payload of a canonical event, or '' if not a text delta."""
    if isinstance(event, ContentBlockDelta) and isinstance(event.delta, TextDelta):
        return event.delta.text
    return ""


def _retrieval_skipped(question: str) -> bool:
    """Re-derive whether retrieval would have run; mirrors ``Searcher.ask_raw`` branches."""
    services = get_services()
    if cfg.chat_mode == ChatMode.CHAT.value:
        return True
    return not services.embedder.embedding_available()


def _build_chat_messages(
    question: str,
    history: list[ChatMessage],
    top_k: int,
    chunk_type: str | None,
) -> tuple[list[SearchChunk], list[ChatMessage]]:
    """Run retrieval and return (sources, message_list).

    Empty ``sources`` plus a direct-chat message list when retrieval is
    disabled or returns nothing; otherwise the augmented prompt from
    ``Searcher.build_rag_context``.
    """
    services = get_services()
    if _retrieval_skipped(question):
        return [], _direct_messages(question, history)
    rag = services.searcher.build_rag_context(
        question, top_k=top_k, history=history, chunk_type=chunk_type
    )
    if rag is None:
        return [], _direct_messages(question, history)
    return rag


def _direct_messages(question: str, history: list[ChatMessage]) -> list[ChatMessage]:
    """Direct-chat messages: general system prompt + history + user question."""
    msgs: list[ChatMessage] = [{"role": "system", "content": cfg.general_system_prompt}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": question})
    return msgs


def _build_canonical_request(
    messages: list[ChatMessage], options: dict[str, Any] | None
) -> CanonicalChatRequest:
    """Convert a wire-shaped message list to a no-tools ``CanonicalChatRequest``."""
    opts = _resolve_generation_options(options) or cfg.generation_options() or {}
    system, chat_msgs = _split_system(messages)
    return CanonicalChatRequest(
        model=cfg.chat_model,
        messages=[
            CanonicalMessage.from_string(role=cast("Any", m["role"]), text=m["content"])
            for m in chat_msgs
        ],
        system=system,
        temperature=opts.get("temperature"),
        top_p=opts.get("top_p"),
        top_k=opts.get("top_k"),
        max_tokens=opts.get("num_predict"),
        stop=opts.get("stop"),
    )


def _split_system(
    messages: list[ChatMessage],
) -> tuple[str | None, list[ChatMessage]]:
    """Pull the leading system message out, returning (system, rest)."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, list(messages)


def _join_text_blocks(content: list[Any]) -> str:
    """Concatenate the text from every ``TextBlock`` in a canonical content list."""
    return "".join(block.text for block in content if isinstance(block, TextBlock))
