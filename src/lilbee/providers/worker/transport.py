"""IPC abstraction for the persistent worker pool.

Defines the two thin Protocols that :class:`lilbee.providers.worker.pool.WorkerPool`,
the per-role worker entrypoints, and all consumer code (LlamaCppProvider,
mtmd_backend, ingest, services, app) talk to. The current concrete impl lives
in :mod:`lilbee.providers.worker.transport_pipe` and is backed by
:class:`multiprocessing.Pipe`. A future ``transport_zmq.py`` (pyzmq) can drop
in without touching consumer code: only the spawner selection in
``WorkerPool.__init__`` changes.

Nothing in this module imports ``multiprocessing`` directly. The Protocols
describe the contract a transport must satisfy; the concrete file owns the
mp.Pipe / mp.Process / mp.Value details.
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
    """

    @property
    def is_alive(self) -> bool:
        """Return True iff the worker process is still running."""
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


__all__ = [
    "RoleConfig",
    "WorkerChannel",
    "WorkerEntrypoint",
    "WorkerHandle",
    "WorkerSpawner",
]
