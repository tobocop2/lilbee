"""Subprocess worker entry point and request handlers.

Runs in a child process spawned by ``WorkerManager``. Loads llama-cpp
models lazily, processes requests off the parent's queue, and writes
responses back. The parent stays free of llama-cpp's GIL contention
and stdout corruption.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lilbee.core.config import cfg
from lilbee.providers.worker.wm_protocol import (
    ConfigSnapshot,
    EmbedRequest,
    EmbedResponse,
    LoadModelRequest,
    ShutdownRequest,
    VisionRequest,
    VisionResponse,
    WorkerRequest,
    WorkerResponse,
)

log = logging.getLogger(__name__)


def _redirect_stdio() -> None:  # pragma: no cover
    """Redirect stdout/stderr to /dev/null for the worker subprocess.

    Suppresses llama-cpp's C-level prints that would corrupt the parent TUI.
    Queues use pipes, not stdout.

    Covered by ``test_redirect_stdio_points_stdout_stderr_to_devnull`` in
    a subprocess (closing fds 1/2 in-process would deadlock pytest-xdist).
    Subprocess execution doesn't report back to coverage.py, so the body
    carries ``# pragma: no cover``.
    """
    import os
    import sys

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 1)  # stdout
    os.dup2(devnull_fd, 2)  # stderr
    os.close(devnull_fd)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def _configure_worker_logging() -> None:
    """Route the worker's Python logs to ``$LILBEE_DATA/worker.log``."""
    data_dir = os.environ.get("LILBEE_DATA") or ""
    if not data_dir:
        return
    log_path = os.path.join(data_dir, "worker.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


@dataclass
class _ModelSlot:
    """Lazy-loaded model handle plus the name it was loaded for."""

    llm: Any = None
    name: str = ""

    def reload_for(self, requested: str, loader: Callable[[str], Any]) -> Any:
        """Return the cached LLM for *requested*, swapping it in if stale."""
        if self.llm is None or requested != self.name:
            _close_model(self.llm)
            self.llm = loader(requested)
            self.name = requested
        return self.llm

    def reset(self) -> None:
        """Drop the cached LLM."""
        _close_model(self.llm)
        self.llm = None
        self.name = ""


def _handle_load_model_request(
    request: LoadModelRequest, embed_slot: _ModelSlot, vision_slot: _ModelSlot
) -> None:
    """Reset the appropriate slot when an explicit reload is requested."""
    if request.model_type == "embed":
        embed_slot.reset()
    else:
        vision_slot.reset()


def _handle_embed_request(
    request: EmbedRequest,
    config: ConfigSnapshot,
    embed_slot: _ModelSlot,
    resp_q: multiprocessing.Queue[WorkerResponse],
) -> None:
    """Resolve the embed model and dispatch."""
    model_name = request.model or config.embedding_model
    embed_llm = embed_slot.reload_for(model_name, _load_embed_model)
    resp_q.put(_handle_embed(embed_llm, request))


def _handle_vision_request(
    request: VisionRequest,
    config: ConfigSnapshot,
    vision_slot: _ModelSlot,
    resp_q: multiprocessing.Queue[WorkerResponse],
) -> None:
    """Resolve the vision model and dispatch (or report "no vision model")."""
    model_name = request.model or config.vision_model
    if not model_name:
        resp_q.put(
            VisionResponse(
                request_id=request.request_id,
                text="",
                error="No vision model configured; vision OCR is disabled.",
            )
        )
        return
    vision_llm = vision_slot.reload_for(model_name, _load_vision_model)
    resp_q.put(_handle_vision(vision_llm, request))


def _drain_request_loop(
    req_q: multiprocessing.Queue[WorkerRequest],
    resp_q: multiprocessing.Queue[WorkerResponse],
    config: ConfigSnapshot,
    embed_slot: _ModelSlot,
    vision_slot: _ModelSlot,
) -> None:
    """Pull from req_q until shutdown or queue failure."""
    while True:
        try:
            request = req_q.get()
        except (EOFError, OSError):
            return
        if isinstance(request, ShutdownRequest):
            return
        if isinstance(request, LoadModelRequest):
            _handle_load_model_request(request, embed_slot, vision_slot)
            continue
        if isinstance(request, EmbedRequest):
            _handle_embed_request(request, config, embed_slot, resp_q)
            continue
        if isinstance(request, VisionRequest):
            _handle_vision_request(request, config, vision_slot, resp_q)


def _close_queues(*queues: multiprocessing.Queue[Any]) -> None:
    """Close + join_thread each queue, swallowing any errors."""
    for q in queues:
        with contextlib.suppress(Exception):
            q.close()
        with contextlib.suppress(Exception):
            q.join_thread()


def _worker_main(
    req_q: multiprocessing.Queue[WorkerRequest],
    resp_q: multiprocessing.Queue[WorkerResponse],
    config: ConfigSnapshot,
) -> None:
    """Child process entry point. Loads models lazily, processes requests."""
    _redirect_stdio()
    _configure_worker_logging()
    _apply_config_snapshot(config)
    log.info("Worker subprocess online (pid=%s)", os.getpid())

    embed_slot = _ModelSlot()
    vision_slot = _ModelSlot()

    try:
        _drain_request_loop(req_q, resp_q, config, embed_slot, vision_slot)
    finally:
        embed_slot.reset()
        vision_slot.reset()
        _close_queues(req_q, resp_q)


def _close_model(model: Any) -> None:
    """Safely close a llama-cpp model instance."""
    if model is not None:
        with contextlib.suppress(Exception):
            model.close()


def _apply_config_snapshot(config: ConfigSnapshot) -> None:
    """Apply parent-process config to the child's cfg singleton.
    Called exactly once at child startup so later load paths can use
    ``cfg.models_dir`` etc. without per-request mutation. Per-request
    mutation would violate the "no mutable module-level globals in
    request paths" rule in CLAUDE.md.
    """
    from pathlib import Path

    cfg.models_dir = Path(config.models_dir)
    cfg.embedding_model = config.embedding_model
    cfg.num_ctx = config.num_ctx


def _load_embed_model(model_name: str) -> Any:
    """Load an embedding model in the child process."""
    from lilbee.providers.llama_cpp.provider import load_llama, resolve_model_path
    from lilbee.providers.model_cache import MODE_EMBED

    return load_llama(resolve_model_path(model_name), mode=MODE_EMBED)


def _load_vision_model(model_name: str) -> Any:
    """Load a vision model in the child process via the mtmd backend."""
    from lilbee.providers.llama_cpp.provider import resolve_model_path
    from lilbee.providers.mtmd_backend import load_vision_llama

    return load_vision_llama(resolve_model_path(model_name))


def _handle_embed(llm: Any, request: EmbedRequest) -> EmbedResponse:
    """Process a single embed request, returning response with vectors or error."""
    try:
        from lilbee.providers.llama_cpp.batching import embed_one

        vectors = [embed_one(llm, text) for text in request.texts]
        return EmbedResponse(vectors=vectors, request_id=request.request_id)
    except Exception as exc:
        return EmbedResponse(error=str(exc), request_id=request.request_id)


def _handle_vision(llm: Any, request: VisionRequest) -> VisionResponse:
    """Process a single vision OCR request.

    Generation is bounded by the model's ``n_ctx`` (llama.cpp stops when
    the remaining context is exhausted) and by EOT, which the GGUF's own
    chat template now fires correctly. No extra per-request cap.
    """
    try:
        from lilbee.vision import OCR_PROMPT, build_vision_messages

        prompt = request.prompt or OCR_PROMPT
        messages = build_vision_messages(prompt, request.png_bytes)
        start = time.monotonic()
        response = llm.create_chat_completion(messages=messages, stream=False)
        text: str = response["choices"][0]["message"]["content"] or ""
        usage = response.get("usage", {}) or {}
        log.info(
            "vision_ocr request_id=%s wall=%.1fs prompt_tokens=%s completion_tokens=%s chars=%d",
            request.request_id,
            time.monotonic() - start,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            len(text),
        )
        return VisionResponse(text=text, request_id=request.request_id)
    except Exception as exc:
        return VisionResponse(error=str(exc), request_id=request.request_id)
