"""Parent-side manager for the llama-cpp worker subprocess.

Owns the child process, request/response queues, and the round-trip
protocol used by ``LlamaCppProvider`` to delegate embedding and vision
work off the main interpreter.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import multiprocessing.queues
import queue
import time
from multiprocessing import get_context
from typing import Any, TypeVar

from lilbee.core.config import cfg
from lilbee.providers.worker.protocol import (
    ConfigSnapshot,
    EmbedRequest,
    EmbedResponse,
    LoadModelRequest,
    ShutdownRequest,
    VisionRequest,
    VisionResponse,
    WorkerRequest,
    WorkerResponse,
    config_snapshot_from_cfg,
)
from lilbee.providers.worker.worker import _worker_main

log = logging.getLogger(__name__)

_ResponseT = TypeVar("_ResponseT", "EmbedResponse", "VisionResponse")

_EMBED_TIMEOUT_S = 30.0
_JOIN_TIMEOUT_S = 5.0
_RESTART_DELAY_S = 0.1
# Substituted when ``cfg.ocr_timeout == 0`` ("no limit"). The round-trip
# wait loop needs a finite deadline; one day is effectively unlimited.
_NO_CAP_TIMEOUT_S = 86_400.0


class WorkerManager:
    """Manages a child process for embedding and vision inference.
    The child loads llama-cpp models independently, avoiding GIL
    contention and stdout corruption in the parent process.
    """

    def __init__(self, config: ConfigSnapshot | None = None) -> None:
        self._config = config
        self._process: Any = None
        self._ctx = get_context("spawn")
        self._request_queue: multiprocessing.Queue[WorkerRequest] | None = None
        self._response_queue: multiprocessing.Queue[WorkerResponse] | None = None
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
        req = EmbedRequest(texts=texts, model=model, request_id=self._next_request_id())
        resp = self._round_trip(req, EmbedResponse, _EMBED_TIMEOUT_S, label="embed")
        return resp.vectors

    def vision_ocr(self, png_bytes: bytes, model: str, prompt: str = "") -> str:
        """Run vision OCR in the worker, honouring ``cfg.ocr_timeout``.

        Auto-starts the worker and retries once on crash. ``cfg.ocr_timeout
        == 0`` means no cap; substituted with ``_NO_CAP_TIMEOUT_S`` for
        the round-trip wait loop.
        """
        self._ensure_started()
        req = VisionRequest(
            png_bytes=png_bytes,
            model=model,
            prompt=prompt,
            request_id=self._next_request_id(),
        )
        timeout = cfg.ocr_timeout if cfg.ocr_timeout > 0 else _NO_CAP_TIMEOUT_S
        resp = self._round_trip(req, VisionResponse, timeout, label="vision OCR")
        return resp.text

    def _round_trip(
        self,
        req: WorkerRequest,
        response_type: type[_ResponseT],
        timeout: float,
        *,
        label: str,
    ) -> _ResponseT:
        """Send a request, wait for the typed response, restart once on crash."""
        resp = self._put_and_get(req, timeout)
        if resp is None:
            log.warning("Worker crashed during %s, restarting and retrying", label)
            self.restart()
            resp = self._put_and_get(req, timeout)
            if resp is None:
                raise RuntimeError("Worker crashed again after restart")
        if not isinstance(resp, response_type):
            raise RuntimeError(f"Unexpected response type: {type(resp).__name__}")
        if resp.error:
            raise RuntimeError(resp.error)
        return resp

    def _put_and_get(self, req: WorkerRequest, timeout: float) -> WorkerResponse | None:
        """Enqueue a request and block for the next response. None if worker died."""
        if self._request_queue is None:
            raise RuntimeError("Worker not started")
        self._request_queue.put(req)
        return self._get_response(timeout=timeout)

    def _get_response(self, timeout: float) -> WorkerResponse | None:
        """Read from response queue. Return None if worker died."""
        if self._response_queue is None:
            raise RuntimeError("Worker not started")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
                return None
            try:
                return self._response_queue.get(timeout=min(1.0, timeout))
            except queue.Empty:
                continue
        raise TimeoutError(f"Worker did not respond within {timeout}s")

    def load_model(self, model: str, model_type: str = "embed") -> None:
        """Tell the worker to (re)load a model."""
        self._ensure_started()
        if self._request_queue is None:
            raise RuntimeError("Worker not started")
        self._request_queue.put(LoadModelRequest(model=model, model_type=model_type))
