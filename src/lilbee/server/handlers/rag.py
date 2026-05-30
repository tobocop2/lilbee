"""Search, ask, and chat handlers (one-shot and streaming)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast

from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.core.results import DocumentResult, group
from lilbee.data.store import ChunkType, EmbeddingModelMismatchError
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.roles import WorkerRole
from lilbee.retrieval.reasoning import (
    CAP_CONTINUATION_PROMPT,
    CAP_NOTICE_TEMPLATE,
    CapNotice,
    StreamToken,
    TagParser,
    effective_reasoning_cap,
    stream_chat_with_cap,
    strip_reasoning,
)
from lilbee.runtime.progress import SseErrorCode, SseEvent
from lilbee.server.chat_completions_api.errors import classify_provider_error
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
    SseErrorCodeValue,
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


def _classify_stream_error(exc: BaseException) -> tuple[SseErrorCodeValue | None, str]:
    """Return ``(code, user_message)`` for an SSE error event, typed-exception aware."""
    classified = classify_provider_error(exc)
    if classified is not None:
        return classified.code, classified.message
    if isinstance(exc, ProviderError):
        # An unmapped ProviderError already carries a user-facing message (rate
        # limit, auth, bad request). Surface it verbatim; the kind becomes a
        # machine-readable code unless the backend couldn't classify it.
        code = None if exc.kind is ProviderErrorKind.UNKNOWN else exc.kind
        return code, str(exc)
    return classify_load_error(str(exc))


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


def _chat_warming_events() -> list[str]:
    """One ``warming`` SSE event when the chat server is cold, else nothing.

    A cold chat server blocks the first token while it loads; the early event
    lets the client show a warming state instead of an apparently-dead stream.
    """
    if get_services().provider.role_ready(WorkerRole.CHAT):
        return []
    log.info("Chat engine cold; streaming a warming notice before the first token.")
    return [sse_event(SseEvent.WARMING, {"role": WorkerRole.CHAT.value})]


def _run_llm_stream(
    messages: list[ChatMessage],
    opts: dict[str, Any] | None,
    queue: asyncio.Queue[str | None],
    cancel: threading.Event,
    error_holder: list[BaseException],
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
        error_holder.append(exc)
    finally:
        queue.put_nowait(None)


async def _stream_rag_response(
    question: str,
    history: list[ChatMessage] | None = None,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: ChunkType | None = None,
) -> AsyncGenerator[str, None]:
    """Shared SSE streaming for ask_stream and chat_stream."""
    yield ""  # force generator

    for warming in _chat_warming_events():
        yield warming

    try:
        rag = get_services().searcher.build_rag_context(
            question, top_k=top_k, history=history, chunk_type=chunk_type
        )
    except EmbeddingModelMismatchError as mismatch:
        yield sse_error(str(mismatch), code=SseErrorCode.INDEX_EMBEDDER_MISMATCH)
        return
    if rag is None:
        yield sse_error("No relevant documents found.")
        return

    results, messages = rag
    opts = _resolve_generation_options(options) or cfg.generation_options()

    sse = SseStream()
    error_holder: list[BaseException] = []

    executor_fut = sse.loop.run_in_executor(
        None, _run_llm_stream, messages, opts, sse.queue, sse.cancel, error_holder
    )
    task = asyncio.ensure_future(executor_fut)
    async for event in sse.drain(task, "RAG stream"):
        yield event

    if error_holder:
        exc = error_holder[0]
        raw = str(exc)
        code, user_message = _classify_stream_error(exc)
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
    chunk_type: ChunkType | None = None,
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
    chunk_type: ChunkType | None,
) -> AsyncGenerator[str, None]:
    """Drive ``dispatch_chat_stream`` and emit reasoning/token/sources/done SSE events."""
    for warming in _chat_warming_events():
        yield warming

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
    sources, messages = rag

    req = _build_canonical_request(messages, options)
    try:
        async for event in _cap_aware_chat_events(req):
            yield _sse_for_chat_event(event)
    except Exception as exc:
        raw = str(exc)
        code, user_message = _classify_stream_error(exc)
        log.warning("Stream error: %s", raw)
        yield sse_error(user_message, code=code, detail=raw if code else None)
        return

    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in sources])
    yield sse_done({})


async def _cap_aware_chat_events(
    req: CanonicalChatRequest,
) -> AsyncIterator[StreamToken | CapNotice]:
    """Run ``dispatch_chat_stream``, split reasoning, and re-issue on cap-fire.

    Mirrors :func:`stream_chat_with_cap` but consumes the canonical async
    stream. ``CapNotice`` is yielded once between the truncated reasoning
    and the continuation answer; ``StreamToken`` carries the
    reasoning-vs-response split for downstream SSE shaping.
    """
    cap_chars = effective_reasoning_cap()
    show = cfg.show_reasoning

    first_parser = TagParser(show=show)
    async for tok in _drive_stream(dispatch_chat_stream(req), first_parser, cap_chars):
        yield tok
    if not (cap_chars > 0 and first_parser.reasoning_chars > cap_chars):
        return

    yield CapNotice(cap_chars=cap_chars)
    nudged = _nudged_request(req)
    cont_parser = TagParser(show=show)
    async for tok in _drive_stream(dispatch_chat_stream(nudged), cont_parser, cap_chars=0):
        # Continuation tokens are always treated as final-answer text.
        yield StreamToken(content=tok.content, is_reasoning=False)


async def _drive_stream(
    stream: AsyncIterator[Any],
    parser: TagParser,
    cap_chars: int,
) -> AsyncIterator[StreamToken]:
    """Feed *stream* through *parser*; yield ``StreamToken``s; stop on cap-fire."""
    cap_fired = False
    try:
        async for event in stream:
            text = _text_from_event(event)
            if not text:
                continue
            for tok in parser.feed(text):
                if tok.content:
                    yield tok
            if cap_chars > 0 and parser.reasoning_chars > cap_chars:
                cap_fired = True
                break
    finally:
        if cap_fired:
            await _aclose(stream)
    tail = parser.flush()
    if tail is not None and tail.content:
        yield tail


def _nudged_request(req: CanonicalChatRequest) -> CanonicalChatRequest:
    """Append the cap-continuation user prompt to *req*'s messages."""
    return dataclasses.replace(
        req,
        messages=[
            *req.messages,
            CanonicalMessage.from_string(role="user", text=CAP_CONTINUATION_PROMPT),
        ],
    )


async def _aclose(stream: AsyncIterator[Any]) -> None:
    """Best-effort close for async-generator-shaped streams."""
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    with contextlib.suppress(Exception):
        await closer()


def _sse_for_chat_event(event: StreamToken | CapNotice) -> str:
    """Render one orchestrator event as an SSE frame with the right channel."""
    if isinstance(event, CapNotice):
        return sse_event(
            SseEvent.REASONING,
            {"token": CAP_NOTICE_TEMPLATE.format(chars=event.cap_chars)},
        )
    kind = SseEvent.REASONING if event.is_reasoning else SseEvent.TOKEN
    return sse_event(kind, {"token": event.content})


def _text_from_event(event: Any) -> str:
    """Return the text payload of a canonical event, or '' if not a text delta."""
    if isinstance(event, ContentBlockDelta) and isinstance(event.delta, TextDelta):
        return event.delta.text
    return ""


def _retrieval_skipped() -> bool:
    """Re-derive whether retrieval would have run; mirrors ``Searcher.ask_raw`` branches."""
    services = get_services()
    if cfg.chat_mode == ChatMode.CHAT.value:
        return True
    return not services.embedder.embedding_available()


def _build_chat_messages(
    question: str,
    history: list[ChatMessage],
    top_k: int,
    chunk_type: ChunkType | None,
) -> tuple[list[SearchChunk], list[ChatMessage]]:
    """Run retrieval and return (sources, message_list).

    Empty ``sources`` plus a direct-chat message list when retrieval is
    disabled or returns nothing; otherwise the augmented prompt from
    ``Searcher.build_rag_context``.
    """
    services = get_services()
    if _retrieval_skipped():
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


_CANONICAL_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool"})


def _build_canonical_request(
    messages: list[ChatMessage], options: dict[str, Any] | None
) -> CanonicalChatRequest:
    """Convert a wire-shaped message list to a no-tools ``CanonicalChatRequest``."""
    opts = _resolve_generation_options(options) or cfg.generation_options() or {}
    system, chat_msgs = _split_system(messages)
    return CanonicalChatRequest(
        model=cfg.chat_model,
        messages=[
            CanonicalMessage.from_string(role=_canonical_role(m["role"]), text=m["content"])
            for m in chat_msgs
        ],
        system=system,
        temperature=opts.get("temperature"),
        top_p=opts.get("top_p"),
        top_k=opts.get("top_k"),
        max_tokens=opts.get("num_predict"),
        stop=opts.get("stop"),
    )


def _canonical_role(role: str) -> Literal["user", "assistant", "tool"]:
    """Narrow a raw wire role string to the canonical literal set or raise."""
    if role not in _CANONICAL_ROLES:
        raise ValueError(f"Unsupported message role {role!r}")
    return cast("Literal['user', 'assistant', 'tool']", role)


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
