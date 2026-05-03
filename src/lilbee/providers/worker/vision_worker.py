"""Persistent vision-OCR worker subprocess entrypoint.

Runs in a child process spawned by :class:`PipeSpawner`. Loads the
vision GGUF (chat handler with embedded mtmd template) on first
request and serves single-image OCR for the TUI's lifetime.

Wire protocol:

* ``("ping", None)`` -> ``("pong", None)``
* ``("shutdown", None)`` -> ``("ack", None)`` then exit
* ``("vision_ocr", payload_dict)`` -> ``("result", str)`` or
  ``("error", _SerializedException)``

The ``payload_dict`` is::

    {
        "png_bytes": bytes,
        "model": str | None,    # override; falls back to role-config path
        "prompt": str,          # OCR_PROMPT default supplied by caller
    }

Per-call ``model`` triggers a transparent reload inside the worker if it
differs from the currently loaded one. The pool's standard model-swap
path (``invalidate_load_cache`` + lazy respawn) still applies when
``cfg.vision_model`` itself changes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any

from lilbee.providers.worker.embed_worker import (
    _configure_worker_logging,
    _redirect_stdio_to_devnull,
)
from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception

log = logging.getLogger(__name__)


_POLL_TIMEOUT_S = 0.5
_VISION_KIND = "vision_ocr"
_PING_KIND = "ping"
_SHUTDOWN_KIND = "shutdown"
_RESULT_KIND = "result"
_ERROR_KIND = "error"
_PONG_KIND = "pong"
_ACK_KIND = "ack"


class _VisionSession:
    """Lazy-loaded vision Llama, kept alive for the worker's lifetime."""

    def __init__(self, role_config: RoleConfig) -> None:
        self._role_config = role_config
        self._llm: Any = None
        self._model_path: str = ""

    def ocr(self, *, png_bytes: bytes, prompt: str, model: str | None) -> str:
        """Run OCR on one image, loading the model on first use."""
        llm = self._ensure_loaded(model)
        from lilbee.vision import OCR_PROMPT, build_vision_messages

        messages = build_vision_messages(prompt or OCR_PROMPT, png_bytes)
        start = time.monotonic()
        response = llm.create_chat_completion(messages=messages, stream=False)
        text: str = response["choices"][0]["message"]["content"] or ""
        usage = response.get("usage", {}) or {}
        log.info(
            "vision_ocr wall=%.1fs prompt_tokens=%s completion_tokens=%s chars=%d",
            time.monotonic() - start,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            len(text),
        )
        return text

    def _ensure_loaded(self, model_override: str | None) -> Any:
        from lilbee.providers.llama_cpp.provider import resolve_model_path
        from lilbee.providers.mtmd_backend import load_vision_llama

        target_path = (
            resolve_model_path(model_override) if model_override else self._role_config.model_path
        )
        target_str = str(target_path)
        if self._llm is None or target_str != self._model_path:
            self._close_model()
            self._llm = load_vision_llama(target_path)
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


def _handle_vision(conn: Any, payload: Any, session: _VisionSession) -> None:
    """Run one vision OCR request and send the typed reply (or error)."""
    if not isinstance(payload, dict):
        try:
            raise TypeError(f"vision_ocr payload must be dict, got {type(payload).__name__}")
        except TypeError as exc:
            conn.send((_ERROR_KIND, _serialize_exception(exc)))
        return
    png_bytes = payload.get("png_bytes")
    if not isinstance(png_bytes, (bytes, bytearray)):
        try:
            raise TypeError("vision_ocr payload.png_bytes must be bytes")
        except TypeError as exc:
            conn.send((_ERROR_KIND, _serialize_exception(exc)))
        return
    try:
        text = session.ocr(
            png_bytes=bytes(png_bytes),
            prompt=str(payload.get("prompt", "")),
            model=payload.get("model"),
        )
    except Exception as exc:
        conn.send((_ERROR_KIND, _serialize_exception(exc)))
        return
    conn.send((_RESULT_KIND, text))


def _dispatch(conn: Any, kind: str, payload: Any, session: _VisionSession) -> bool:
    """Handle one request. Return False to stop the worker loop."""
    if kind == _SHUTDOWN_KIND:
        conn.send((_ACK_KIND, None))
        return False
    if kind == _PING_KIND:
        conn.send((_PONG_KIND, None))
        return True
    if kind == _VISION_KIND:
        _handle_vision(conn, payload, session)
        return True
    try:
        raise ValueError(f"Vision worker received unknown kind {kind!r}")
    except ValueError as exc:
        conn.send((_ERROR_KIND, _serialize_exception(exc)))
    return True


def vision_worker_main(conn: Any, _abort_flag: Any, role_config: RoleConfig) -> None:
    """Vision-OCR worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    _redirect_stdio_to_devnull()
    _configure_worker_logging(role_config.role)
    log.info("vision worker online (pid=%s, model=%s)", os.getpid(), role_config.model_path)
    session = _VisionSession(role_config)
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


__all__ = ["vision_worker_main"]
