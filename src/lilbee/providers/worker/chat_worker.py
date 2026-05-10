"""Long-lived chat worker subprocess body, with token streaming."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

from lilbee.providers.worker.transport import ChatRequest, RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import WireKind
from lilbee.providers.worker.worker_runtime import Reply, WorkerLoopState, run_worker

_ABORT_BRIDGE_POLL_S = 0.025
"""How often the abort bridge polls the parent's mp.Value flag.

25 ms is the budget between the user pressing Esc and ggml's next
abort_callback poll on a slow-token chat. Faster than this stops being
visible to a human; slower than this leaves the user staring at an
unresponsive UI for the duration of one stuck token.
"""

_STREAM_BATCH_MAX_CHUNKS = 16
"""Flush a streaming-chat batch once this many tokens have queued.

Bounded so a long answer at high tok/s doesn't grow the buffer without
limit. Sized so each flush write batches ~16 syscalls into 1.
"""

_STREAM_BATCH_MAX_INTERVAL_S = 0.05
"""Flush a streaming-chat batch at least this often (50 ms).

The check fires only when the *next* token arrives (we never wake on a
timer), so a generator that stalls after token N keeps token N+1's
buffer parked until the next token lands. The eager-first-flush at
the top of :func:`_handle_chat_streaming` guarantees the user sees
something within the very first token, so the parked-tail case only
delays subsequent batches, not initial output.
"""


class _ChatSession:
    """Lazy-loaded Llama chat handle, kept alive for the worker's lifetime.

    Reloads in place when the parent passes a per-call ``model`` override
    different from the currently loaded one.
    """

    def __init__(self, role_config: RoleConfig, abort_flag: Any) -> None:
        self._role_config = role_config
        self._abort_flag = abort_flag
        self._llm: Any = None
        self._model_path: str = ""

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        stream: bool,
        options: dict[str, Any] | None,
        model: str | None,
    ) -> Any:
        """Run one chat completion and return the llama-cpp response."""
        llm = self._ensure_loaded(model)
        kwargs: dict[str, Any] = dict(options) if options else {}
        return llm.create_chat_completion(messages=messages, stream=stream, **kwargs)

    def _ensure_loaded(self, model_override: str | None) -> Any:
        from lilbee.providers.llama_cpp.provider import load_llama, resolve_model_path
        from lilbee.providers.model_cache import LoaderMode

        target_path = (
            resolve_model_path(model_override) if model_override else self._role_config.model_path
        )
        target_str = str(target_path)
        if self._llm is None or target_str != self._model_path:
            self._close_model()
            # No abort_callback_override: routing the cancel signal through
            # ggml's mid-token abort path crashes the worker on macOS Metal.
            # Cancel is enforced one token boundary later by the Python-side
            # polling loop in _handle_chat_streaming.
            self._llm = load_llama(target_path, mode=LoaderMode.CHAT)
            self._model_path = target_str
        return self._llm

    def _close_model(self) -> None:
        if self._llm is not None:
            with contextlib.suppress(Exception):
                self._llm.close()
            self._llm = None

    def close(self) -> None:
        """Release the loaded model. Idempotent."""
        self._close_model()


def _extract_stream_content(chunk: Any) -> str | None:
    """Pull the text content out of one llama-cpp streaming chunk."""
    choices = chunk.get("choices") if isinstance(chunk, dict) else None
    if not choices:
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _handle_chat_streaming(reply: Reply, response_iter: Any, state: WorkerLoopState) -> None:
    """Drain *response_iter* and emit batched stream_chunk frames on the data pipe.

    Polls ``state.session._abort_flag`` between chunks so a cancel from the
    parent flushes a clean ``stream_end`` at the next token boundary.
    Tokens are accumulated and flushed every ``_STREAM_BATCH_MAX_CHUNKS``
    or ``_STREAM_BATCH_MAX_INTERVAL_S``, whichever comes first, so the
    pipe sees ~one syscall per batch instead of one per token.

    Cancel path: ``break`` exits the for loop normally, control falls
    through to ``completed_cleanly = True``, the finally clause flushes
    any buffered tail, and ``stream_end`` fires. The parent's
    ``stream()`` reader then returns cleanly without a hang.
    """
    abort_flag = state.session._abort_flag
    buffer: list[str] = []
    last_flush = time.monotonic()
    seen_first_token = False
    completed_cleanly = False
    try:
        for raw_chunk in response_iter:
            if abort_flag.value:
                with contextlib.suppress(Exception):
                    response_iter.close()
                break
            content = _extract_stream_content(raw_chunk)
            if content is None:
                continue
            buffer.append(content)
            now = time.monotonic()
            # Flush the very first token immediately so a generator that
            # stalls after one token still surfaces something to the user.
            should_flush = (
                not seen_first_token
                or len(buffer) >= _STREAM_BATCH_MAX_CHUNKS
                or (now - last_flush) >= _STREAM_BATCH_MAX_INTERVAL_S
            )
            if should_flush:
                reply.send(WireKind.STREAM_CHUNK, "".join(buffer))
                buffer.clear()
                last_flush = now
                seen_first_token = True
        completed_cleanly = True
    finally:
        # Flush any buffered tokens regardless of how the loop exited so
        # the user sees partial output before the error frame the outer
        # handler may emit.
        if buffer:
            reply.send(WireKind.STREAM_CHUNK, "".join(buffer))
    if completed_cleanly:
        reply.send(WireKind.STREAM_END, None)


def _extract_non_streaming_content(response: Any) -> str:
    """Pull the assistant text out of one llama-cpp non-streaming response."""
    if not isinstance(response, dict):
        raise TypeError(f"chat response must be dict, got {type(response).__name__}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("chat response missing 'choices' list")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError(f"chat choices[0] must be dict, got {type(first).__name__}")
    message = first.get("message")
    if not isinstance(message, dict):
        raise TypeError("chat choices[0].message missing or not dict")
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _handle_chat_non_streaming(reply: Reply, response: Any) -> None:
    """Emit one result frame with the full assistant message text."""
    text = _extract_non_streaming_content(response)
    reply.send(WireKind.RESULT, text)


class _AbortBridge:
    """Mirror the parent's mp.Value abort flag into ggml's threading.Event.

    Without this, cancel only takes effect at Python-loop boundaries
    between yielded tokens. A token that takes 30+ seconds inside the
    ggml decode (full context, slow GPU, big buffer) keeps generating
    because the Python loop never gets a chance to read the flag.
    Calling ``request_abort()`` flips the threading.Event that the
    loaded llama's ``abort_callback`` polls inside ggml, so cancel
    takes effect at the next ggml poll point (every few tokens) instead
    of waiting for the next Python yield.
    """

    def __init__(self, abort_flag: Any) -> None:
        self._abort_flag = abort_flag
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _AbortBridge:
        from lilbee.providers.llama_cpp.abort_signal import clear_abort

        # Reset both flags before the chat starts: a stale parent-side
        # cancel from a prior call must not abort the new request.
        clear_abort()
        self._abort_flag.value = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="chat-abort-bridge", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        from lilbee.providers.llama_cpp.abort_signal import clear_abort

        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        # Reset for the next request so a cancelled prior call doesn't
        # latch onto the next inference.
        clear_abort()
        self._abort_flag.value = 0

    def _poll(self) -> None:
        from lilbee.providers.llama_cpp.abort_signal import request_abort

        while not self._stop.wait(_ABORT_BRIDGE_POLL_S):
            if self._abort_flag.value:
                request_abort()
                return


def _handle_chat(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
    """Run one chat request and dispatch to the streaming/non-streaming handler."""
    if not isinstance(payload, ChatRequest):
        try:
            raise TypeError(f"chat payload must be ChatRequest, got {type(payload).__name__}")
        except TypeError as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))
        return
    session: _ChatSession = state.session
    with _AbortBridge(session._abort_flag):
        try:
            response = session.chat(
                messages=payload.messages,
                stream=payload.stream,
                options=payload.options,
                model=payload.model,
            )
        except Exception as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))
            return
        try:
            if payload.stream:
                _handle_chat_streaming(reply, response, state)
            else:
                _handle_chat_non_streaming(reply, response)
        except Exception as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))


def chat_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Chat worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    run_worker(
        data_conn,
        health_conn,
        abort_flag,
        role_config,
        session_factory=_ChatSession,
        kind_handlers={WireKind.CHAT: _handle_chat},
    )


__all__ = ["chat_worker_main"]
