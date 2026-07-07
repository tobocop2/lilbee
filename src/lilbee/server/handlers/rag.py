"""Search, ask, and chat handlers (one-shot and streaming)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from lilbee.app.memory import auto_extract, auto_extract_enabled
from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.results import DocumentResult, group
from lilbee.data.store import ChunkType, EmbeddingModelMismatchError
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.roles import WorkerRole
from lilbee.retrieval.query.formatting import cited_subset
from lilbee.retrieval.query.searcher import SEARCH_NEEDS_EMBEDDER
from lilbee.retrieval.reasoning import (
    CAP_CONTINUATION_PROMPT,
    CAP_NOTICE_TEMPLATE,
    REASONING_EXHAUSTED_NOTICE,
    CapNotice,
    StreamToken,
    TagParser,
    effective_reasoning_cap,
    stream_chat_with_cap,
    strip_reasoning,
)
from lilbee.runtime.progress import SseErrorCode, SseEvent
from lilbee.server.chat_completions_api.errors import CompletionsErrorCode
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    ContentBlockDelta,
    TextBlock,
    TextDelta,
)
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
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
from lilbee.server.models import (
    AskResponse,
    CleanedChunk,
    MemoryExtractedEvent,
    MemoryExtractedItem,
)

if TYPE_CHECKING:
    from lilbee.core.results import SearchChunk
    from lilbee.retrieval.query import ChatMessage
    from lilbee.retrieval.query.searcher import Searcher

log = logging.getLogger(__name__)


# Unmapped kinds surface as their ProviderErrorKind string; shipped clients branch on it.
_STREAM_KIND_CODES: dict[ProviderErrorKind, CompletionsErrorCode] = {
    ProviderErrorKind.CONTEXT_OVERFLOW: CompletionsErrorCode.CONTEXT_LENGTH_EXCEEDED,
    ProviderErrorKind.NOT_FOUND: CompletionsErrorCode.MODEL_NOT_FOUND,
}


def _classify_stream_error(exc: BaseException) -> tuple[SseErrorCodeValue | None, str]:
    """Return ``(code, user_message)`` for an SSE error event, typed-exception aware."""
    if isinstance(exc, ModelNotFoundError):
        return CompletionsErrorCode.MODEL_NOT_FOUND, str(exc)
    if isinstance(exc, ModelDoesNotSupportToolsError):
        return CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS, str(exc)
    if isinstance(exc, ProviderError):
        mapped = _STREAM_KIND_CODES.get(exc.kind)
        if mapped is not None:
            return mapped, str(exc)
        code = None if exc.kind is ProviderErrorKind.UNKNOWN else exc.kind
        return code, str(exc)
    return classify_load_error(str(exc))


async def search(
    q: str, top_k: int = 5, chunk_type: ChunkType | None = None
) -> list[DocumentResult]:
    """Search and return grouped DocumentResults."""
    if not q or not q.strip():
        raise ValueError("query must not be empty")
    # search() blocks on retrieval; run it off the event loop so other admitted
    # requests stay responsive, matching the sibling ask() handler.
    results = await asyncio.to_thread(
        get_services().searcher.search, q, top_k=top_k, chunk_type=chunk_type
    )
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
    searcher = get_services().searcher
    # ask_raw blocks for retrieval plus the whole generation; run it off the
    # event loop so other admitted requests stay responsive.
    result = await asyncio.to_thread(
        searcher.ask_raw,
        question,
        top_k=top_k,
        options=opts,
        chunk_type=chunk_type,
    )
    # Mirror the streaming ask path: auto-extract memories from a real answer,
    # but never from the search-needs-embedder refusal ask_raw returns.
    if not searcher.search_unavailable():
        await _store_extracted_memories(question, result.answer)
    return AskResponse(
        answer=result.answer,
        sources=[CleanedChunk(**clean_result(s)) for s in result.sources],
        cited_sources=[CleanedChunk(**clean_result(s)) for s in result.cited_sources],
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
    put: Callable[[str | None], None],
    cancel: threading.Event,
    error_holder: list[BaseException],
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
                put(
                    sse_event(
                        SseEvent.REASONING,
                        {"token": CAP_NOTICE_TEMPLATE.format(chars=event.cap_chars)},
                    )
                )
            elif event.content:
                kind = SseEvent.REASONING if event.is_reasoning else SseEvent.TOKEN
                if kind is SseEvent.TOKEN:
                    answer_parts.append(event.content)
                put(sse_event(kind, {"token": event.content}))
    except Exception as exc:
        error_holder.append(exc)
    finally:
        put(None)


async def _store_extracted_memories(question: str, answer: str) -> list[Any]:
    """Run the auto-extraction LLM pass off the event loop and return stored memories.

    Returns an empty list (no-op) when the answer is empty or auto-extraction is
    off, so one-shot and streaming callers share one extraction path.
    """
    if not answer or not auto_extract_enabled():
        return []
    return await asyncio.to_thread(auto_extract, question, answer)


async def _emit_extracted_memories(question: str, answer: str) -> AsyncGenerator[str, None]:
    """Yield a ``memory_extracted`` SSE event if the turn auto-saved any memories.

    Silent (yields nothing) when the answer is empty, auto-extraction is off, or
    nothing was extracted, so existing consumers are unaffected.
    """
    stored = await _store_extracted_memories(question, answer)
    if not stored:
        return
    event = MemoryExtractedEvent(
        count=len(stored),
        items=[MemoryExtractedItem(id=m.id, kind=m.kind, text=m.text) for m in stored],
    )
    yield sse_event(SseEvent.MEMORY_EXTRACTED, event.model_dump(mode="json"))


async def _stream_rag_response(
    question: str,
    history: list[ChatMessage] | None = None,
    top_k: int = 0,
    options: dict[str, Any] | None = None,
    chunk_type: ChunkType | None = None,
) -> AsyncGenerator[str, None]:
    """SSE streaming for the ask (search) endpoint.

    Mirrors ``Searcher.ask_stream`` so streaming, one-shot, and CLI ask agree:
    search mode with no embedder refuses cleanly, chat mode answers ungrounded,
    otherwise the answer is grounded in retrieved sources.
    """
    yield ""  # force generator

    for warming in _chat_warming_events():
        yield warming

    searcher = get_services().searcher
    if searcher.search_unavailable():
        # Search needs an embedder to ground. Mirror Searcher.ask_stream by
        # returning the refusal as a normal answer token (not an SSE error) so the
        # streaming, one-shot, and CLI ask paths all surface it the same way.
        yield sse_event(SseEvent.TOKEN, {"token": SEARCH_NEEDS_EMBEDDER})
        yield sse_event(SseEvent.SOURCES, [])
        yield sse_done({})
        return
    if searcher.skip_retrieval():
        # Chat mode with an embedder present: answer ungrounded, mirroring ask_raw.
        results: list[SearchChunk] = []
        messages = searcher.direct_messages(question, history)
    else:
        try:
            rag = searcher.build_rag_context(
                question, top_k=top_k, history=history, chunk_type=chunk_type
            )
        except EmbeddingModelMismatchError as mismatch:
            # detail carries the index's embedder so the client can offer to adopt it.
            detail = mismatch.persisted_model if mismatch.dims_match else None
            yield sse_error(str(mismatch), code=SseErrorCode.INDEX_EMBEDDER_MISMATCH, detail=detail)
            return
        if rag is None:
            yield sse_error("No relevant documents found.")
            return
        results, messages = rag

    opts = _resolve_generation_options(options) or cfg.generation_options()

    sse = SseStream()
    error_holder: list[BaseException] = []
    answer_parts: list[str] = []

    executor_fut = sse.loop.run_in_executor(
        None,
        _run_llm_stream,
        messages,
        opts,
        sse.put_threadsafe,
        sse.cancel,
        error_holder,
        answer_parts,
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

    # SOURCES carries the cited subset (what the answer referenced), falling back
    # to the full retrieved set when the answer carries no inline citations.
    # Mirrors Searcher.ask_stream's ``used if used else results``.
    cited = cited_subset("".join(answer_parts), results)
    source_list = cited if cited else results
    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in source_list])
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
    top_k: int | None = None,
    options: dict[str, Any] | None = None,
    chunk_type: ChunkType | None = None,
) -> AskResponse:
    """Chat with history. Returns answer and sources via canonical dispatch."""
    searcher = get_services().searcher
    if searcher.search_unavailable():
        # Search mode with no embedder can't ground; refuse cleanly with the same
        # message ask returns instead of silently answering off-corpus.
        return AskResponse(answer=SEARCH_NEEDS_EMBEDDER, sources=[], cited_sources=[])
    sources, messages = _build_chat_messages(question, history, top_k, chunk_type)
    req = _build_canonical_request(messages, options)
    response = await asyncio.to_thread(dispatch_chat, req)
    text = _join_text_blocks(response.content)
    answer = text if cfg.show_reasoning else strip_reasoning(text)
    if not answer.strip() and text.strip():
        # The model emitted only reasoning (stripped to nothing) and no final
        # answer. Surface that distinctly instead of a silent empty string the
        # caller can't tell apart from a legitimate empty response (bb-cpu). The
        # synthetic notice is not an answer, so -- like the search-needs-embedder
        # refusal -- it doesn't seed memory.
        answer = REASONING_EXHAUSTED_NOTICE
    else:
        await _store_extracted_memories(question, answer)
    return AskResponse(
        answer=answer,
        sources=[CleanedChunk(**clean_result(s)) for s in sources],
        cited_sources=[CleanedChunk(**clean_result(s)) for s in cited_subset(answer, sources)],
    )


def chat_stream(
    question: str,
    history: list[ChatMessage],
    top_k: int | None = None,
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
    top_k: int | None,
    options: dict[str, Any] | None,
    chunk_type: ChunkType | None,
) -> AsyncGenerator[str, None]:
    """Drive ``dispatch_chat_stream`` and emit reasoning/token/sources/done SSE events."""
    for warming in _chat_warming_events():
        yield warming

    searcher = get_services().searcher
    if searcher.search_unavailable():
        # Search mode with no embedder can't ground; refuse cleanly with the same
        # token the ask stream emits instead of silently answering off-corpus.
        yield sse_event(SseEvent.TOKEN, {"token": SEARCH_NEEDS_EMBEDDER})
        yield sse_event(SseEvent.SOURCES, [])
        yield sse_done({})
        return
    if _retrieval_off(searcher, top_k):
        # Chat-only mode, no embedder, or an explicit top_k:0 pure-LLM call:
        # answer directly with no RAG context.
        sources: list[SearchChunk] = []
        messages = searcher.direct_messages(question, history)
    else:
        try:
            rag = searcher.build_rag_context(
                question, top_k=top_k or 0, history=history, chunk_type=chunk_type
            )
        except EmbeddingModelMismatchError as exc:
            # detail carries the index's embedder so the client can offer to adopt it.
            detail = exc.persisted_model if exc.dims_match else None
            yield sse_error(str(exc), code=SseErrorCode.INDEX_EMBEDDER_MISMATCH, detail=detail)
            return
        if rag is None:
            yield sse_error("No relevant documents found.")
            return
        sources, messages = rag

    req = _build_canonical_request(messages, options)
    answer_parts: list[str] = []
    try:
        async for event in _cap_aware_chat_events(req):
            if (
                isinstance(event, StreamToken)
                and not event.is_reasoning
                # The reasoning-exhausted notice is streamed to the client but is
                # not a real answer: keep it out of answer_parts so it seeds no
                # memory and isn't treated as a citation source (bb-cpu).
                and event.content != REASONING_EXHAUSTED_NOTICE
            ):
                answer_parts.append(event.content)
            yield _sse_for_chat_event(event)
    except Exception as exc:
        raw = str(exc)
        code, user_message = _classify_stream_error(exc)
        log.warning("Stream error: %s", raw)
        yield sse_error(user_message, code=code, detail=raw if code else None)
        return

    # SOURCES carries the cited subset (what the answer referenced), falling back
    # to the full retrieved set when the answer carries no inline citations.
    # Mirrors Searcher.ask_stream's ``used if used else results``.
    cited = cited_subset("".join(answer_parts), sources)
    source_list = cited if cited else sources
    yield sse_event(SseEvent.SOURCES, [clean_result(s) for s in source_list])
    yield sse_done({})
    async for mem_event in _emit_extracted_memories(question, "".join(answer_parts)):
        yield mem_event


async def _cap_aware_chat_events(
    req: CanonicalChatRequest,
) -> AsyncIterator[StreamToken | CapNotice]:
    """Run ``dispatch_chat_stream``, split reasoning, and re-issue on cap-fire.

    Mirrors :func:`stream_chat_with_cap` but consumes the canonical async
    stream. ``CapNotice`` is yielded once between the truncated reasoning
    and the continuation answer; ``StreamToken`` carries the
    reasoning-vs-response split for downstream SSE shaping. When reasoning
    runs but no final answer follows, a closing ``StreamToken`` carrying
    ``REASONING_EXHAUSTED_NOTICE`` is yielded so the run isn't silent (bb-cpu).
    """
    cap_chars = effective_reasoning_cap()
    show = cfg.show_reasoning
    answered = False

    first_parser = TagParser(show=show)
    async for tok in _drive_stream(dispatch_chat_stream(req), first_parser, cap_chars):
        answered = answered or (not tok.is_reasoning and bool(tok.content))
        yield tok

    if cap_chars > 0 and first_parser.reasoning_chars > cap_chars:
        yield CapNotice(cap_chars=cap_chars)
        nudged = _nudged_request(req)
        cont_parser = TagParser(show=show)
        async for tok in _drive_stream(dispatch_chat_stream(nudged), cont_parser, cap_chars=0):
            answered = answered or bool(tok.content)
            # Continuation tokens are always treated as final-answer text.
            yield StreamToken(content=tok.content, is_reasoning=False)

    if first_parser.reasoning_chars > 0 and not answered:
        # The model spent its budget reasoning and produced no final answer;
        # a distinct notice tells a reasoning-only run apart from a completed one.
        yield StreamToken(content=REASONING_EXHAUSTED_NOTICE, is_reasoning=False)


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
    if not isinstance(stream, AsyncGenerator):
        return
    with contextlib.suppress(Exception):
        await stream.aclose()


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


def _retrieval_off(searcher: Searcher, top_k: int | None) -> bool:
    """Whether this /api/chat turn bypasses RAG.

    An explicit ``top_k == 0`` is a pure-LLM call: answer without retrieval. An
    unspecified ``top_k`` (``None``) uses the configured default and grounds
    normally. Chat-only mode or a missing embedder also bypass.
    """
    return top_k == 0 or searcher.skip_retrieval()


def _build_chat_messages(
    question: str,
    history: list[ChatMessage],
    top_k: int | None,
    chunk_type: ChunkType | None,
) -> tuple[list[SearchChunk], list[ChatMessage]]:
    """Run retrieval and return (sources, message_list).

    Empty ``sources`` plus a direct-chat message list when retrieval is
    disabled or returns nothing; otherwise the augmented prompt from
    ``Searcher.build_rag_context``.
    """
    searcher = get_services().searcher
    if _retrieval_off(searcher, top_k):
        return [], searcher.direct_messages(question, history)
    rag = searcher.build_rag_context(
        question, top_k=top_k or 0, history=history, chunk_type=chunk_type
    )
    if rag is None:
        return [], searcher.direct_messages(question, history)
    return rag


_CANONICAL_ROLE_BY_WIRE: dict[str, Literal["user", "assistant", "tool"]] = {
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


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


def _canonical_role(wire_role: str) -> Literal["user", "assistant", "tool"]:
    """Narrow a raw wire role string to the canonical literal set or raise."""
    try:
        return _CANONICAL_ROLE_BY_WIRE[wire_role]
    except KeyError:
        raise ValueError(f"Unsupported message role {wire_role!r}") from None


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
