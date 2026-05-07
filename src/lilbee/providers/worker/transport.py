"""Bidirectional channel and spawner protocols for worker IPC.

Concrete impl lives in :mod:`lilbee.providers.worker.transport_pipe`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

WorkerEntrypoint = Callable[..., None]
"""Signature of a worker subprocess main function.

Concrete signature is ``(child_conn, abort_flag, role_config) -> None`` for
the pipe transport. Kept as ``Callable[..., None]`` here so other transports
(zmq, in-process for tests) can supply their own argument shape without the
Protocol leaking mp-specific types.
"""


@dataclass(frozen=True)
class RoleConfig:
    """Spawn-time configuration handed to a worker subprocess.

    Must be picklable: it crosses the process boundary at spawn. ``role`` is
    the short identifier (``embed``, ``chat``, ``rerank``, ``vision``).
    ``model_path`` is the absolute on-disk path to the GGUF file the worker
    should load. ``mode`` is the loader hint (``"embed"``, ``"chat"``,
    ``"vision"``) consumed by ``providers.model_cache``. ``extras`` carries
    any additional pickle-friendly payload a specific role needs (kept open
    so adding fields does not change the Protocol signature).
    """

    role: str
    model_path: Path
    mode: str
    extras: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatRequest:
    """Pickle-friendly chat request that crosses the parent->worker pipe.

    Replaces the inline ``{"messages": ..., "stream": ..., ...}`` dict
    so a typo on either side surfaces as a type error (or attribute
    miss) instead of a silent ``payload.get("missing", default)`` that
    masks the bug. ``messages`` is the standard llama-cpp message list;
    ``stream`` decides between single-result and chunked replies;
    ``options`` is the post-``filter_options`` kwarg dict the worker
    forwards to ``create_chat_completion``; ``model`` triggers a
    transparent reload inside the worker if it differs from the
    role-config model.
    """

    messages: list[dict[str, str]]
    stream: bool = False
    options: dict[str, Any] | None = None
    model: str | None = None


@dataclass(frozen=True)
class VisionRequest:
    """Pickle-friendly vision-OCR request that crosses the parent->worker pipe.

    Replaces the inline ``{"png_bytes": ..., "model": ..., "prompt": ...}``
    dict. ``model`` is optional: ``None`` means "use the role-config
    model unchanged".
    """

    png_bytes: bytes
    prompt: str = ""
    model: str | None = None


@dataclass(frozen=True)
class PdfOcrRequest:
    """Pickle-friendly multi-page PDF-OCR request.

    Replaces the per-call ``python -m lilbee.runtime.pdf_extract``
    subprocess. ``path`` is a string because ``pathlib.Path`` pickles
    via a chained reducer; the worker re-wraps it. ``backend`` is one
    of ``"vision"`` (uses the role's loaded vision Llama) or
    ``"tesseract"`` (no model load, runs kreuzberg). The worker
    validates the value and raises ``ValueError`` for anything else.
    ``model`` overrides ``cfg.vision_model`` for the call when set;
    ignored when ``backend == "tesseract"``.
    """

    path: str
    backend: str
    model: str = ""
    per_page_timeout_s: float | None = None
    quiet: bool = True


@dataclass(frozen=True)
class RerankPayload:
    """Pickle-friendly rerank request.

    Replaces the bare ``(query, candidates)`` tuple so the worker can
    type-check the shape via attribute access instead of length / index
    juggling.
    """

    query: str
    candidates: list[str]


@dataclass(frozen=True)
class WorkerHandle:
    """Opaque handle to a spawned worker, returned alongside the channel.

    Carries the bookkeeping the pool needs for restart-on-crash and idle
    reaping without exposing transport-specific types (``mp.Process``,
    ``threading.Thread``, etc.) to the pool. ``pid`` is informational and
    may be ``None`` for transports that do not have a single OS process
    (e.g. a hypothetical in-process test transport).
    """

    pid: int | None
    role: str


@runtime_checkable
class WorkerChannel(Protocol):
    """Bidirectional message channel to one running worker.

    Lifecycle: built by a :class:`WorkerSpawner`, kept alive for the
    worker's lifetime, torn down via :meth:`close`. Methods are ordered
    by the typical call sequence (call/stream during inference, ping for
    health, cancel/clear_abort to interrupt, close on shutdown).

    Call-ordering contract
    ----------------------

    1. The spawner returns a channel ready for ``call`` / ``stream`` /
       ``ping`` immediately. There is no ``initialize`` step.
    2. ``is_alive`` and ``in_flight`` are safe to read at any time,
       including concurrently with an in-flight ``call`` / ``stream``.
    3. ``call``, ``stream``, and ``ping`` may run concurrently with each
       other (each acquires its own send/recv lock internally), but the
       worker process serializes them on the wire.
    4. ``cancel`` / ``clear_abort`` are best-effort and never block on a
       lock; safe to call from any thread, including during a streaming
       ``__anext__`` so the consumer can interrupt itself.
    5. ``close`` is final. After it returns, ``call`` / ``stream`` /
       ``ping`` must raise :class:`WorkerError`. ``close`` is idempotent.
    """

    @property
    def is_alive(self) -> bool:
        """Return True iff the worker process is still running."""
        ...

    @property
    def pid(self) -> int | None:
        """OS process id of the worker, or None for transports without one."""
        ...

    @property
    def in_flight(self) -> int:
        """Number of requests sent but not yet fully replied to.

        The pool's idle reaper checks this is zero before timing out a
        worker. A pending ``stream()`` counts as in-flight until its
        terminator (``stream_end`` / ``error``) arrives.
        """
        ...

    def call(self, kind: str, payload: Any, *, timeout: float) -> Awaitable[Any]:
        """Send one request, await one reply. Raises on worker error or timeout."""
        ...

    def stream(self, kind: str, payload: Any) -> AsyncIterator[Any]:
        """Send one request, yield streamed chunks until the worker terminates the stream."""
        ...

    def ping(self, *, timeout: float) -> Awaitable[None]:
        """Send ping, await pong. Raises on timeout (worker considered hung)."""
        ...

    def cancel(self) -> None:
        """Flip the worker's abort flag.

        Best-effort: in-flight ``stream_chunk`` messages already in the
        pipe will still drain (typically a few extra tokens). The
        user-facing toast should say "Cancelling..." until the worker
        confirms with a terminator.
        """
        ...

    def clear_abort(self) -> None:
        """Reset the abort flag to 0 so the next request runs to completion."""
        ...

    def close(self, *, timeout: float) -> Awaitable[None]:
        """Send shutdown, await graceful exit, terminate stragglers past *timeout*."""
        ...


@runtime_checkable
class WorkerSpawner(Protocol):
    """Spawns worker subprocesses and returns their channels.

    One spawner instance per :class:`WorkerPool`; each call to
    :meth:`spawn` produces one new worker. The spawner owns transport-
    specific knowledge (which mp.Pipe end the child gets, which port a
    zmq worker should bind, etc.); the pool only sees Protocols.
    """

    def spawn(
        self,
        worker_main: WorkerEntrypoint,
        role_config: RoleConfig,
    ) -> tuple[WorkerChannel, WorkerHandle]:
        """Start a worker process and return its channel + handle."""
        ...


def default_spawner() -> WorkerSpawner:
    """Return a fresh :class:`PipeSpawner`. Lazy import to avoid a transport_pipe cycle."""
    from lilbee.providers.worker.transport_pipe import PipeSpawner

    return PipeSpawner()


__all__ = [
    "ChatRequest",
    "PdfOcrRequest",
    "RerankPayload",
    "RoleConfig",
    "VisionRequest",
    "WorkerChannel",
    "WorkerEntrypoint",
    "WorkerHandle",
    "WorkerSpawner",
    "default_spawner",
]
