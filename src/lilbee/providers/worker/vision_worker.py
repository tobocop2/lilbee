"""Long-lived vision-OCR worker subprocess body.

Hosts both single-image OCR (``VISION_KIND``, used by the live wiki /
catalog flows) and multi-page PDF vision OCR (``PDF_OCR_KIND``, used by
the ingest pipeline). The PDF path streams one chunk per page so
subscribers see incremental progress. Tesseract OCR has no shared model
state and runs inline via ``asyncio.to_thread`` in the ingest caller,
not through this worker.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from lilbee.providers.worker.transport import PdfOcrRequest, RoleConfig, VisionRequest
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import (
    ERROR_KIND,
    PDF_OCR_KIND,
    RESULT_KIND,
    STREAM_CHUNK_KIND,
    STREAM_END_KIND,
    VISION_KIND,
)
from lilbee.providers.worker.worker_runtime import Reply, WorkerLoopState, run_worker

log = logging.getLogger(__name__)


def _make_abort_callback(abort_flag: Any) -> Any:
    """Return a llama-cpp abort_callback bound to the shared mp.Value flag."""

    def _callback(_user_data: Any = None) -> bool:
        return bool(abort_flag.value)

    return _callback


class _VisionSession:
    """Lazy-loaded vision Llama, kept alive for the worker's lifetime."""

    def __init__(self, role_config: RoleConfig, abort_flag: Any) -> None:
        self._role_config = role_config
        self._abort_flag = abort_flag
        self._llm: Any = None
        self._model_path: str = ""

    def ocr(self, *, png_bytes: bytes, prompt: str, model: str | None) -> str:
        """Run OCR on one image, loading the model on first use."""
        llm = self._ensure_loaded(model)
        from lilbee.vision import OCR_PROMPT, build_vision_messages

        messages = build_vision_messages(prompt or OCR_PROMPT, png_bytes)
        start = time.monotonic()
        response = llm.create_chat_completion(messages=messages, stream=False)
        text = _extract_vision_content(response)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
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
            # The abort flag lives in shared memory (mp.Value), so the
            # callback bound here lets the parent's pool.cancel() reach
            # llama-cpp's vision inference loop in this subprocess.
            self._llm = load_vision_llama(
                target_path,
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


def _extract_vision_content(response: Any) -> str:
    """Pull the OCR text out of one llama-cpp vision response.

    Mirrors the chat path's defensive walk so a malformed response
    surfaces as a typed :class:`TypeError` we can serialize, instead of
    a raw :class:`KeyError` / :class:`IndexError` deep in the worker.
    """
    if not isinstance(response, dict):
        raise TypeError(f"vision response must be dict, got {type(response).__name__}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("vision response missing 'choices' list")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError(f"vision choices[0] must be dict, got {type(first).__name__}")
    message = first.get("message")
    if not isinstance(message, dict):
        raise TypeError("vision choices[0].message missing or not dict")
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _handle_vision(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
    """Run one vision OCR request and send the typed reply (or error)."""
    if not isinstance(payload, VisionRequest):
        try:
            raise TypeError(
                f"vision_ocr payload must be VisionRequest, got {type(payload).__name__}"
            )
        except TypeError as exc:
            reply.send(ERROR_KIND, _serialize_exception(exc))
        return
    if not isinstance(payload.png_bytes, (bytes, bytearray)):
        try:
            raise TypeError("vision_ocr payload.png_bytes must be bytes")
        except TypeError as exc:
            reply.send(ERROR_KIND, _serialize_exception(exc))
        return
    session: _VisionSession = state.session
    try:
        text = session.ocr(
            png_bytes=bytes(payload.png_bytes),
            prompt=payload.prompt,
            model=payload.model,
        )
    except Exception as exc:
        reply.send(ERROR_KIND, _serialize_exception(exc))
        return
    reply.send(RESULT_KIND, text)


def _handle_pdf_ocr(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
    """Stream multi-page vision PDF OCR results, one chunk per page, then stream_end.

    Iterates rasterised PDF pages and OCRs each via the loaded vision
    Llama. ``payload.backend`` must be ``"vision"``; tesseract is run
    inline by the ingest caller, not here, because tesseract has no
    shared model state and pool routing buys it nothing.
    """
    if not isinstance(payload, PdfOcrRequest):
        try:
            raise TypeError(f"pdf_ocr payload must be PdfOcrRequest, got {type(payload).__name__}")
        except TypeError as exc:
            reply.send(ERROR_KIND, _serialize_exception(exc))
        return
    if payload.backend != "vision":
        try:
            raise ValueError(
                f"vision worker only handles backend='vision', got {payload.backend!r}"
            )
        except ValueError as exc:
            reply.send(ERROR_KIND, _serialize_exception(exc))
        return
    session: _VisionSession = state.session
    try:
        from lilbee.vision import OCR_PROMPT, pdf_page_count, rasterize_pdf

        path = Path(payload.path)
        total = pdf_page_count(path)
        model_override = payload.model or None
        for idx, png_bytes in rasterize_pdf(path):
            text = session.ocr(
                png_bytes=bytes(png_bytes),
                prompt=OCR_PROMPT,
                model=model_override,
            )
            # 1-based page index matches how the rest of lilbee numbers
            # pages (PageText, ExtractEvent, etc.). Total ships in every
            # chunk so consumers don't need a separate header frame.
            reply.send(STREAM_CHUNK_KIND, (idx + 1, total, text))
    except Exception as exc:
        reply.send(ERROR_KIND, _serialize_exception(exc))
        return
    reply.send(STREAM_END_KIND, None)


def vision_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Vision-OCR worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    run_worker(
        data_conn,
        health_conn,
        abort_flag,
        role_config,
        session_factory=_VisionSession,
        kind_handlers={
            VISION_KIND: _handle_vision,
            PDF_OCR_KIND: _handle_pdf_ocr,
        },
    )


__all__ = ["vision_worker_main"]
