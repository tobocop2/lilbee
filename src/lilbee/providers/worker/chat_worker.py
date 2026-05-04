"""Long-lived chat worker subprocess body, with token streaming."""

from __future__ import annotations

import contextlib
from typing import Any

from lilbee.providers.worker.transport import ChatRequest, RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import (
    CHAT_KIND,
    ERROR_KIND,
    RESULT_KIND,
    STREAM_CHUNK_KIND,
    STREAM_END_KIND,
)
from lilbee.providers.worker.worker_runtime import WorkerLoopState, run_worker, stream_window


def _make_abort_callback(abort_flag: Any) -> Any:
    """Return a llama-cpp abort_callback bound to the shared mp.Value flag."""

    def _callback(_user_data: Any = None) -> bool:
        return bool(abort_flag.value)

    return _callback


class _ChatSession:
    """Lazy-loaded Llama chat handle, kept alive for the worker's lifetime.

    Reloads in place when the parent passes a per-call ``model`` override
    different from the currently loaded one. The pool's standard model-
    swap path (``invalidate_load_cache`` + lazy respawn) still applies
    when ``cfg.chat_model`` itself changes; this is just the per-call
    override.
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
        """Run one chat completion and return the llama-cpp response.

        For ``stream=True`` returns the iterator the caller must drain.
        For ``stream=False`` returns the full result dict.
        """
        llm = self._ensure_loaded(model)
        kwargs: dict[str, Any] = dict(options) if options else {}
        return llm.create_chat_completion(messages=messages, stream=stream, **kwargs)

    def _ensure_loaded(self, model_override: str | None) -> Any:
        from lilbee.providers.llama_cpp.provider import load_llama, resolve_model_path
        from lilbee.providers.model_cache import MODE_CHAT

        target_path = (
            resolve_model_path(model_override) if model_override else self._role_config.model_path
        )
        target_str = str(target_path)
        if self._llm is None or target_str != self._model_path:
            self._close_model()
            # The abort flag lives in shared memory (mp.Value), so the
            # callback bound here lets the parent's pool.cancel() reach
            # llama-cpp's inference loop in this subprocess.
            self._llm = load_llama(
                target_path,
                mode=MODE_CHAT,
                abort_callback_override=_make_abort_callback(self._abort_flag),
            )
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
    """Pull the text content out of one llama-cpp streaming chunk.

    Mirrors the existing in-process ``_LockedStreamIterator`` shape so
    the parent's pool-backed iterator yields the same per-chunk strings
    consumers already expect.
    """
    choices = chunk.get("choices") if isinstance(chunk, dict) else None
    if not choices:
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _handle_chat_streaming(conn: Any, response_iter: Any, state: WorkerLoopState) -> None:
    """Drain *response_iter* and emit per-token stream_chunk frames.

    Marks *state* as actively streaming for the emission window. Any
    handler dispatch that happens to land while the flag is set drops
    health pings rather than emitting pong frames the parent's stream
    consumer would read out of band. The defense is paired with the
    parent-side ``stream`` reader's silent consumption of pong frames
    (which absorbs orphan pongs from pings buffered into the pipe before
    the worker entered this window).
    """
    with stream_window(state):
        for raw_chunk in response_iter:
            content = _extract_stream_content(raw_chunk)
            if content is not None:
                conn.send((STREAM_CHUNK_KIND, content))
        conn.send((STREAM_END_KIND, None))


def _extract_non_streaming_content(response: Any) -> str:
    """Pull the assistant text out of one llama-cpp non-streaming response.

    Mirrors :func:`_extract_stream_content`'s defensive walk so a
    malformed (or truncated) response surfaces as a typed
    :class:`TypeError` we can serialize back to the parent, instead of
    a raw :class:`KeyError` / :class:`IndexError` deep in the worker.
    """
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


def _handle_chat_non_streaming(conn: Any, response: Any) -> None:
    """Emit one result frame with the full assistant message text."""
    text = _extract_non_streaming_content(response)
    conn.send((RESULT_KIND, text))


def _handle_chat(conn: Any, payload: Any, state: WorkerLoopState) -> None:
    """Run one chat request and dispatch to the streaming/non-streaming handler."""
    if not isinstance(payload, ChatRequest):
        try:
            raise TypeError(f"chat payload must be ChatRequest, got {type(payload).__name__}")
        except TypeError as exc:
            conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    session: _ChatSession = state.session
    try:
        response = session.chat(
            messages=payload.messages,
            stream=payload.stream,
            options=payload.options,
            model=payload.model,
        )
    except Exception as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    try:
        if payload.stream:
            _handle_chat_streaming(conn, response, state)
        else:
            _handle_chat_non_streaming(conn, response)
    except Exception as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))


def chat_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    """Chat worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    run_worker(
        conn,
        abort_flag,
        role_config,
        session_factory=_ChatSession,
        kind_handlers={CHAT_KIND: _handle_chat},
    )


__all__ = ["chat_worker_main"]
