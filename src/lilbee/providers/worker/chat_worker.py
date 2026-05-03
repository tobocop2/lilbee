"""Persistent chat worker subprocess entrypoint with token streaming.

Runs in a child process spawned by :class:`PipeSpawner`. Loads the chat
GGUF on first request and serves chat completions, including streaming,
for the TUI's lifetime. The shared ``mp.Value`` abort flag from the spawn
is read by llama-cpp's ``abort_callback`` so the parent's pool-side
cancel takes effect mid-generation.

Wire protocol (each message is a tuple ``(kind, payload)``):

* ``("ping", None)`` -> ``("pong", None)``
* ``("shutdown", None)`` -> ``("ack", None)`` then exit
* ``("chat", payload_dict)`` -> ``("result", str)`` (non-streaming) or
  a sequence of ``("stream_chunk", str)`` followed by
  ``("stream_end", None)`` (streaming). Errors at any point yield
  ``("error", _SerializedException)``.

The chat ``payload_dict`` is::

    {
        "messages": list[dict[str, str]],
        "stream": bool,
        "options": dict[str, Any] | None,
        "model": str | None,
    }

``model`` is the optional override for the worker's role-config model.
A non-None value triggers a transparent reload inside the worker (one
extra cold-load) rather than respawning the whole worker process.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import (
    ACK_KIND,
    CHAT_KIND,
    ERROR_KIND,
    PING_KIND,
    PONG_KIND,
    RESULT_KIND,
    SHUTDOWN_KIND,
    STREAM_CHUNK_KIND,
    STREAM_END_KIND,
)
from lilbee.providers.worker.worker_runtime import (
    configure_worker_logging,
    redirect_stdio_to_devnull,
)

log = logging.getLogger(__name__)


_POLL_TIMEOUT_S = 0.5


def _make_abort_callback(abort_flag: Any) -> Any:
    """Return a llama-cpp abort_callback bound to the shared mp.Value flag.

    Discipline rule: the flag uses ``lock=True`` so reads are atomic on
    every supported architecture (x86 + ARM). The cost of one acquire
    per token-generation tick is negligible vs llama-cpp inference.
    """

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
        kwargs: dict[str, Any] = {"abort_callback": _make_abort_callback(self._abort_flag)}
        if options:
            kwargs.update(options)
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
            self._llm = load_llama(target_path, mode=MODE_CHAT)
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


def _handle_chat_streaming(conn: Any, response_iter: Any) -> None:
    """Drain *response_iter* and emit per-token stream_chunk frames."""
    for raw_chunk in response_iter:
        content = _extract_stream_content(raw_chunk)
        if content is not None:
            conn.send((STREAM_CHUNK_KIND, content))
    conn.send((STREAM_END_KIND, None))


def _handle_chat_non_streaming(conn: Any, response: Any) -> None:
    """Emit one result frame with the full assistant message text."""
    text = response["choices"][0]["message"]["content"] or ""
    conn.send((RESULT_KIND, text))


def _handle_chat(conn: Any, payload: Any, session: _ChatSession) -> None:
    """Run one chat request and dispatch to the streaming/non-streaming handler."""
    if not isinstance(payload, dict):
        try:
            raise TypeError(f"chat payload must be dict, got {type(payload).__name__}")
        except TypeError as exc:
            conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    try:
        response = session.chat(
            messages=payload.get("messages", []),
            stream=bool(payload.get("stream", False)),
            options=payload.get("options"),
            model=payload.get("model"),
        )
    except Exception as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    try:
        if payload.get("stream"):
            _handle_chat_streaming(conn, response)
        else:
            _handle_chat_non_streaming(conn, response)
    except Exception as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))


def _dispatch(conn: Any, kind: str, payload: Any, session: _ChatSession) -> bool:
    """Handle one request. Return False to stop the worker loop."""
    if kind == SHUTDOWN_KIND:
        conn.send((ACK_KIND, None))
        return False
    if kind == PING_KIND:
        conn.send((PONG_KIND, None))
        return True
    if kind == CHAT_KIND:
        _handle_chat(conn, payload, session)
        return True
    try:
        raise ValueError(f"Chat worker received unknown kind {kind!r}")
    except ValueError as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
    return True


def chat_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    """Chat worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    redirect_stdio_to_devnull()
    configure_worker_logging(role_config.role)
    log.info("chat worker online (pid=%s, model=%s)", os.getpid(), role_config.model_path)
    session = _ChatSession(role_config, abort_flag)
    try:
        while True:
            if not conn.poll(timeout=_POLL_TIMEOUT_S):
                continue
            try:
                kind, payload = conn.recv()
            except EOFError:
                return
            if not _dispatch(conn, kind, payload, session):
                return
    finally:
        session.close()
        with contextlib.suppress(Exception):
            conn.close()


__all__ = ["chat_worker_main"]
