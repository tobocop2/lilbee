"""Subprocess worker for llama-cpp embedding and vision operations.

Isolates llama-cpp native code (which holds the GIL) in a separate
process so the parent event loop stays responsive. Communication
uses multiprocessing queues with spawn start method (fork-unsafe
with threads).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import multiprocessing.queues
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

log = logging.getLogger(__name__)

_EMBED_TIMEOUT_S = 30.0
_VISION_TIMEOUT_S = 120.0
_JOIN_TIMEOUT_S = 5.0
_RESTART_DELAY_S = 0.1


@dataclass(frozen=True)
class EmbedRequest:
    """Request to embed a list of texts."""

    texts: list[str]
    model: str
    request_id: int = 0


@dataclass(frozen=True)
class EmbedResponse:
    """Response with embedding vectors or an error."""

    vectors: list[list[float]] = field(default_factory=list)
    error: str = ""
    request_id: int = 0


@dataclass(frozen=True)
class VisionRequest:
    """Request to run vision OCR on a PNG image."""

    png_bytes: bytes
    model: str
    prompt: str = ""
    request_id: int = 0


@dataclass(frozen=True)
class VisionResponse:
    """Response with extracted text or an error."""

    text: str = ""
    error: str = ""
    request_id: int = 0


@dataclass(frozen=True)
class LoadModelRequest:
    """Request to (re)load a specific model in the worker."""

    model: str
    model_type: str = "embed"  # "embed" or "vision"


@dataclass(frozen=True)
class ShutdownRequest:
    """Sentinel to tell the worker to exit."""


@dataclass(frozen=True)
class ConfigSnapshot:
    """Minimal config fields needed by the child process."""

    models_dir: str
    embedding_model: str
    embedding_dim: int
    num_ctx: int | None
    gpu_memory_fraction: float
    vision_model: str


_WorkerRequest = EmbedRequest | VisionRequest | LoadModelRequest | ShutdownRequest
_WorkerResponse = EmbedResponse | VisionResponse


def config_snapshot_from_cfg() -> ConfigSnapshot:
    """Build a ConfigSnapshot from the current cfg singleton."""
    from lilbee.config import cfg

    return ConfigSnapshot(
        models_dir=str(cfg.models_dir),
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        num_ctx=cfg.num_ctx,
        gpu_memory_fraction=cfg.gpu_memory_fraction,
        vision_model=cfg.vision_model,
    )


class WorkerProcess:
    """Manages a child process for embedding and vision inference.

    The child loads llama-cpp models independently, avoiding GIL
    contention and stdout corruption in the parent process.
    """

    def __init__(self, config: ConfigSnapshot | None = None) -> None:
        self._config = config
        self._process: Any = None
        self._ctx = get_context("spawn")
        self._request_queue: multiprocessing.Queue[_WorkerRequest] | None = None
        self._response_queue: multiprocessing.Queue[_WorkerResponse] | None = None
        self._next_id = 0
        self._started = False

    def _ensure_config(self) -> ConfigSnapshot:
        if self._config is None:
            self._config = config_snapshot_from_cfg()
        return self._config

    def start(self) -> None:
        """Launch the child process. No-op if already running."""
        if self._started and self.is_alive():
            return
        config = self._ensure_config()
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._request_queue, self._response_queue, config),
            daemon=True,
        )
        self._process.start()
        self._started = True
        log.info("Worker process started (pid=%s)", self._process.pid)

    def stop(self) -> None:
        """Send shutdown request, join, terminate if needed."""
        if self._process is None:
            self._started = False
            return
        with contextlib.suppress(OSError, ValueError, AttributeError):
            if self._request_queue is not None:
                self._request_queue.put(ShutdownRequest())
        self._process.join(timeout=_JOIN_TIMEOUT_S)
        if self._process.is_alive():
            log.warning("Worker did not exit gracefully, terminating")
            self._process.terminate()
            self._process.join(timeout=2)
        self._process = None
        self._started = False
        log.info("Worker process stopped")

    def restart(self) -> None:
        """Stop and restart the worker (e.g. after model change)."""
        self.stop()
        time.sleep(_RESTART_DELAY_S)
        self.start()

    def is_alive(self) -> bool:
        """Return True if the child process is running."""
        return self._process is not None and self._process.is_alive()

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _ensure_started(self) -> None:
        """Lazy start: launch worker on first request."""
        if not self._started or not self.is_alive():
            self.start()

    def embed(self, texts: list[str], model: str = "") -> list[list[float]]:
        """Send an embed request and wait for the response.

        Auto-starts the worker if not running. Retries once on crash.
        """
        self._ensure_started()
        rid = self._next_request_id()
        req = EmbedRequest(texts=texts, model=model, request_id=rid)
        return self._send_and_receive_embed(req)

    def _send_and_receive_embed(self, req: EmbedRequest) -> list[list[float]]:
        """Put request, get response, handle crash with one retry."""
        assert self._request_queue is not None
        self._request_queue.put(req)
        resp = self._get_response(timeout=_EMBED_TIMEOUT_S)
        if resp is None:
            return self._retry_embed(req)
        if not isinstance(resp, EmbedResponse):
            raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
        if resp.error:
            raise RuntimeError(resp.error)
        return resp.vectors

    def _retry_embed(self, req: EmbedRequest) -> list[list[float]]:
        """Restart worker and retry a failed embed request once."""
        log.warning("Worker crashed during embed, restarting and retrying")
        self.restart()
        assert self._request_queue is not None
        self._request_queue.put(req)
        resp = self._get_response(timeout=_EMBED_TIMEOUT_S)
        if resp is None:
            raise RuntimeError("Worker crashed again after restart")
        if not isinstance(resp, EmbedResponse):
            raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
        if resp.error:
            raise RuntimeError(resp.error)
        return resp.vectors

    def vision_ocr(self, png_bytes: bytes, model: str, prompt: str = "") -> str:
        """Send a vision OCR request and wait for the response.

        Auto-starts the worker if not running. Retries once on crash.
        """
        self._ensure_started()
        rid = self._next_request_id()
        req = VisionRequest(png_bytes=png_bytes, model=model, prompt=prompt, request_id=rid)
        return self._send_and_receive_vision(req)

    def _send_and_receive_vision(self, req: VisionRequest) -> str:
        """Put request, get response, handle crash with one retry."""
        assert self._request_queue is not None
        self._request_queue.put(req)
        resp = self._get_response(timeout=_VISION_TIMEOUT_S)
        if resp is None:
            return self._retry_vision(req)
        if not isinstance(resp, VisionResponse):
            raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
        if resp.error:
            raise RuntimeError(resp.error)
        return resp.text

    def _retry_vision(self, req: VisionRequest) -> str:
        """Restart worker and retry a failed vision request once."""
        log.warning("Worker crashed during vision OCR, restarting and retrying")
        self.restart()
        assert self._request_queue is not None
        self._request_queue.put(req)
        resp = self._get_response(timeout=_VISION_TIMEOUT_S)
        if resp is None:
            raise RuntimeError("Worker crashed again after restart")
        if not isinstance(resp, VisionResponse):
            raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
        if resp.error:
            raise RuntimeError(resp.error)
        return resp.text

    def _get_response(self, timeout: float) -> _WorkerResponse | None:
        """Read from response queue. Return None if worker died."""
        assert self._response_queue is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
                return None
            try:
                return self._response_queue.get(timeout=min(1.0, timeout))
            except Exception:
                continue
        raise TimeoutError(f"Worker did not respond within {timeout}s")

    def load_model(self, model: str, model_type: str = "embed") -> None:
        """Tell the worker to (re)load a model."""
        self._ensure_started()
        assert self._request_queue is not None
        self._request_queue.put(LoadModelRequest(model=model, model_type=model_type))


def _worker_main(
    req_q: multiprocessing.Queue[_WorkerRequest],
    resp_q: multiprocessing.Queue[_WorkerResponse],
    config: ConfigSnapshot,
) -> None:
    """Child process entry point. Loads models lazily, processes requests."""
    # Redirect stdout/stderr to devnull so llama-cpp's C-level prints
    # don't corrupt the parent TUI. Queues use pipes, not stdout.
    import os
    import sys

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 1)  # stdout
    os.dup2(devnull_fd, 2)  # stderr
    os.close(devnull_fd)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

    embed_llm: Any = None
    vision_llm: Any = None
    current_embed_model = ""
    current_vision_model = ""

    while True:
        try:
            request = req_q.get()
        except (EOFError, OSError):
            break

        if isinstance(request, ShutdownRequest):
            _close_model(embed_llm)
            _close_model(vision_llm)
            break

        if isinstance(request, LoadModelRequest):
            if request.model_type == "embed":
                _close_model(embed_llm)
                embed_llm = None
                current_embed_model = ""
            else:
                _close_model(vision_llm)
                vision_llm = None
                current_vision_model = ""
            continue

        if isinstance(request, EmbedRequest):
            model_name = request.model or config.embedding_model
            if embed_llm is None or model_name != current_embed_model:
                _close_model(embed_llm)
                embed_llm = _load_embed_model(config, model_name)
                current_embed_model = model_name
            embed_resp = _handle_embed(embed_llm, request)
            resp_q.put(embed_resp)
            continue

        if isinstance(request, VisionRequest):
            model_name = request.model or config.vision_model
            if vision_llm is None or model_name != current_vision_model:
                _close_model(vision_llm)
                vision_llm = _load_vision_model(config, model_name)
                current_vision_model = model_name
            vision_resp = _handle_vision(vision_llm, request)
            resp_q.put(vision_resp)
            continue


def _close_model(model: Any) -> None:
    """Safely close a llama-cpp model instance."""
    if model is not None and hasattr(model, "close"):
        with contextlib.suppress(Exception):
            model.close()


def _load_embed_model(config: ConfigSnapshot, model_name: str) -> Any:
    """Load an embedding model in the child process."""
    from pathlib import Path

    from lilbee.config import cfg
    from lilbee.providers.llama_cpp_provider import _load_llama, _resolve_model_path

    cfg.models_dir = Path(config.models_dir)
    cfg.embedding_model = config.embedding_model
    cfg.num_ctx = config.num_ctx

    model_path = _resolve_model_path(model_name)
    return _load_llama(model_path, embedding=True)


def _load_vision_model(config: ConfigSnapshot, model_name: str) -> Any:
    """Load a vision model in the child process."""
    from pathlib import Path

    from lilbee.config import cfg
    from lilbee.providers.llama_cpp_provider import _load_vision_llama, _resolve_model_path

    cfg.models_dir = Path(config.models_dir)
    cfg.vision_model = config.vision_model
    cfg.num_ctx = config.num_ctx

    model_path = _resolve_model_path(model_name)
    return _load_vision_llama(model_path)


def _handle_embed(llm: Any, request: EmbedRequest) -> EmbedResponse:
    """Process a single embed request, returning response with vectors or error."""
    try:
        from lilbee.providers.llama_cpp_provider import _embed_one

        vectors = [_embed_one(llm, text) for text in request.texts]
        return EmbedResponse(vectors=vectors, request_id=request.request_id)
    except Exception as exc:
        return EmbedResponse(error=str(exc), request_id=request.request_id)


def _handle_vision(llm: Any, request: VisionRequest) -> VisionResponse:
    """Process a single vision OCR request."""
    try:
        from lilbee.vision import _OCR_PROMPT, _build_vision_messages

        prompt = request.prompt or _OCR_PROMPT
        messages = _build_vision_messages(prompt, request.png_bytes)
        response = llm.create_chat_completion(messages=messages, stream=False)
        text: str = response["choices"][0]["message"]["content"] or ""
        return VisionResponse(text=text, request_id=request.request_id)
    except Exception as exc:
        return VisionResponse(error=str(exc), request_id=request.request_id)
