"""``multiprocessing.Pipe``-backed worker channel and spawner.

Concrete impl of the ``WorkerChannel`` / ``WorkerSpawner`` Protocols
from :mod:`lilbee.providers.worker.transport`. Pipe-specific discipline
rules are documented in ``docs/architecture.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import pickle
import threading
import traceback
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from lilbee.providers.worker.transport import (
    RoleConfig,
    WorkerChannel,
    WorkerEntrypoint,
    WorkerHandle,
)
from lilbee.providers.worker.wire_kinds import (
    ERROR_KIND,
    PING_KIND,
    PONG_KIND,
    RESULT_KIND,
    SHUTDOWN_KIND,
    STREAM_CHUNK_KIND,
    STREAM_END_KIND,
)

log = logging.getLogger(__name__)


_PICKLE_MAX_BYTES = 32 * 1024 * 1024
"""Conservative cap on a single :func:`pickle.dumps` payload before send.

``Connection.send`` raises ``ValueError`` past about 32 MiB on POSIX. Token
streams and embedding vectors fit comfortably; vision images and rerank
batches must stay under this limit (or move to ``send_bytes``).
"""


@dataclass(frozen=True)
class _SerializedException:
    """Pickle-friendly stand-in for an exception that crossed the wire.

    Worker-side exceptions go through :func:`_serialize_exception` so
    unpicklable types (``_thread.RLock`` in tracebacks, some ``OSError``
    subclasses, structlog wrappers) do not silently kill the worker.
    Reconstruction on the parent side raises a :class:`WorkerError` whose
    ``__cause__`` is the (best-effort) original.
    """

    type_name: str
    message: str
    traceback_str: str


class WorkerError(RuntimeError):
    """Raised on the parent side when a worker reports an exception.

    ``original_type`` is the best-effort name of the worker-side exception
    type (``"ValueError"``, ``"RuntimeError"``, etc.). ``traceback_str`` is
    the worker's formatted traceback for diagnostic logs.
    """

    def __init__(self, original_type: str, message: str, traceback_str: str) -> None:
        super().__init__(f"{original_type}: {message}")
        self.original_type = original_type
        self.traceback_str = traceback_str


class WorkerCrashError(WorkerError):
    """Raised when a worker process dies mid-request (EOF on the pipe)."""

    def __init__(self, role: str) -> None:
        super().__init__(
            "WorkerCrashError",
            f"Worker '{role}' subprocess exited unexpectedly.",
            "",
        )
        self.role = role


def _serialize_exception(exc: BaseException) -> _SerializedException:
    """Reduce an exception to a pickle-safe ``(type_name, message, traceback)`` triple.

    The live exception itself is not picklable for several common types
    (``_thread.RLock`` references in tracebacks, several ``OSError``
    subclasses, structlog wrappers, `cpython#101159`_), so the triple is
    what crosses the pipe.

    .. _cpython#101159: https://github.com/python/cpython/issues/101159
    """
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _SerializedException(
        type_name=type(exc).__name__,
        message=str(exc),
        traceback_str=tb_str,
    )


def _deserialize_exception(payload: _SerializedException) -> WorkerError:
    """Rebuild a parent-side exception from a serialized worker exception."""
    return WorkerError(payload.type_name, payload.message, payload.traceback_str)


def _check_pickle_size(payload: Any, kind: str) -> None:
    """Raise ``ValueError`` early if *payload* would exceed the pipe send cap."""
    try:
        size = len(pickle.dumps((kind, payload)))
    except Exception as exc:
        raise WorkerError("PickleError", f"Failed to pickle {kind!r} payload: {exc}", "") from exc
    if size > _PICKLE_MAX_BYTES:
        raise WorkerError(
            "PayloadTooLarge",
            f"{kind!r} payload is {size} bytes; pipe send cap is {_PICKLE_MAX_BYTES}.",
            "",
        )


class PipeChannel:
    """One worker process talked to via a duplex :class:`multiprocessing.Pipe`.

    Owns the parent end of the pipe, the abort flag (a
    ``multiprocessing.Value('b', 0, lock=True)`` shared with the worker),
    and a per-channel :class:`ThreadPoolExecutor` (``max_workers=2``: one
    read thread, one write thread). Per-channel executor is required so
    multiplexing four roles plus streaming chat plus UI calls does not
    starve the asyncio default thread pool, which is shared by everything
    in the process and capped at ``min(32, cpu_count + 4)``.

    Constructed only by :class:`PipeSpawner`; user code reaches the channel
    through :class:`lilbee.providers.worker.pool.WorkerPool` accessors.
    """

    def __init__(
        self,
        *,
        role: str,
        process: multiprocessing.process.BaseProcess,
        parent_conn: Any,
        abort_flag: Any,
    ) -> None:
        self._role = role
        self._process = process
        self._conn = parent_conn
        self._abort = abort_flag
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"pipechan-{role}",
        )
        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._closed = False
        self._closed_lock = threading.Lock()

    @property
    def role(self) -> str:
        """Short identifier for this channel (``embed``, ``chat``, ...)."""
        return self._role

    @property
    def is_alive(self) -> bool:
        """Return True iff the underlying process is still running."""
        return self._process.is_alive()

    @property
    def pid(self) -> int | None:
        """Worker process id (``None`` until ``start`` returns)."""
        return self._process.pid

    @property
    def in_flight(self) -> int:
        """Number of requests sent but not yet fully replied to."""
        with self._in_flight_lock:
            return self._in_flight

    def _bump_in_flight(self, delta: int) -> None:
        with self._in_flight_lock:
            self._in_flight += delta

    def _ensure_open(self) -> None:
        with self._closed_lock:
            if self._closed:
                raise WorkerError(
                    "PoolShutdownError",
                    f"Channel for worker '{self._role}' is closed.",
                    "",
                )

    async def _send(self, kind: str, payload: Any) -> None:
        """Pickle-pre-check + thread-bounded ``conn.send`` under the send lock."""
        _check_pickle_size(payload, kind)
        loop = asyncio.get_running_loop()
        async with self._send_lock:
            try:
                await loop.run_in_executor(self._executor, self._conn.send, (kind, payload))
            except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
                raise WorkerCrashError(self._role) from exc

    async def _recv(self) -> tuple[str, Any]:
        """Thread-bounded ``conn.recv`` under the recv lock; raises on EOF/crash."""
        loop = asyncio.get_running_loop()
        async with self._recv_lock:
            try:
                return await loop.run_in_executor(self._executor, self._conn.recv)
            except (EOFError, OSError, ConnectionResetError, BrokenPipeError) as exc:
                raise WorkerCrashError(self._role) from exc

    async def call(self, kind: str, payload: Any, *, timeout: float) -> Any:
        """Send one request, await one reply, return the unpacked result.

        Raises :class:`WorkerError` if the worker reported an exception,
        :class:`WorkerCrashError` if the worker died, or
        :class:`asyncio.TimeoutError` if the reply did not arrive in
        *timeout* seconds.
        """
        self._ensure_open()
        self._bump_in_flight(1)
        try:
            await self._send(kind, payload)
            msg_kind, value = await asyncio.wait_for(self._recv(), timeout=timeout)
            if msg_kind == ERROR_KIND:
                raise _deserialize_exception(value)
            if msg_kind != RESULT_KIND:
                raise WorkerError(
                    "ProtocolError",
                    f"Worker '{self._role}' replied with unexpected kind {msg_kind!r}.",
                    "",
                )
            return value
        finally:
            self._bump_in_flight(-1)

    async def stream(self, kind: str, payload: Any) -> AsyncIterator[Any]:
        """Send one request, yield streamed chunks until the worker terminates.

        The terminator is one of ``stream_end`` (clean), ``error`` (worker
        exception), or pipe EOF (worker crash). The in-flight counter
        stays positive for the entire streaming window so the idle reaper
        does not race with a long generation.
        """
        self._ensure_open()
        self._bump_in_flight(1)
        await self._send(kind, payload)
        try:
            while True:
                msg_kind, value = await self._recv()
                if msg_kind == STREAM_CHUNK_KIND:
                    yield value
                elif msg_kind == STREAM_END_KIND:
                    return
                elif msg_kind == ERROR_KIND:
                    raise _deserialize_exception(value)
                else:
                    raise WorkerError(
                        "ProtocolError",
                        f"Worker '{self._role}' streamed unexpected kind {msg_kind!r}.",
                        "",
                    )
        finally:
            self._bump_in_flight(-1)

    async def ping(self, *, timeout: float) -> None:
        """Round-trip a ping and verify the pong; raise on timeout."""
        self._ensure_open()
        self._bump_in_flight(1)
        try:
            await self._send(PING_KIND, None)
            msg_kind, _ = await asyncio.wait_for(self._recv(), timeout=timeout)
            if msg_kind != PONG_KIND:
                raise WorkerError(
                    "ProtocolError",
                    f"Worker '{self._role}' ping reply was {msg_kind!r}, want 'pong'.",
                    "",
                )
        finally:
            self._bump_in_flight(-1)

    def cancel(self) -> None:
        """Flip the abort flag to 1; in-flight tokens may still drain (rule 8)."""
        self._abort.value = 1

    def clear_abort(self) -> None:
        """Reset the abort flag to 0 before the next request."""
        self._abort.value = 0

    async def close(self, *, timeout: float) -> None:
        """Send shutdown, await ack, terminate the process if it overruns *timeout*.

        Idempotent: calling close on an already-closed channel is a no-op.
        Always closes the underlying pipe + executor at the end so callers
        can drop the channel even if the worker hung.
        """
        with self._closed_lock:
            if self._closed:
                return
            self._closed = True
        try:
            with contextlib.suppress(asyncio.TimeoutError, WorkerError):
                await self._send(SHUTDOWN_KIND, None)
                with contextlib.suppress(asyncio.TimeoutError, WorkerError):
                    await asyncio.wait_for(self._recv(), timeout=timeout)
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._join_process, timeout
            )
        finally:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _join_process(self, timeout: float) -> None:
        """Wait *timeout* seconds for the process; terminate if still alive."""
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            log.warning("Worker '%s' did not exit gracefully; terminating", self._role)
            self._process.terminate()
            self._process.join(timeout=2.0)


class PipeSpawner:
    """Spawns worker subprocesses connected to the parent via :class:`multiprocessing.Pipe`.

    Always uses :func:`multiprocessing.get_context` with ``"spawn"`` so:

    * Metal/CUDA contexts that the worker initializes are isolated. Fork
      inheritance crashes them (see vllm#8893 for the reference report).
    * Python 3.14 deprecates fork as the POSIX default; relying on the
      per-OS default is forward-incompatible.

    Cost: spawn re-imports Python in the child, adding ~1-3s cold start
    per worker. Mitigated by the pool's lazy spawn + idle reaping.
    """

    def __init__(self, *, daemon: bool = True) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._daemon = daemon

    def spawn(
        self,
        worker_main: WorkerEntrypoint,
        role_config: RoleConfig,
    ) -> tuple[WorkerChannel, WorkerHandle]:
        """Start a worker subprocess and return its channel + handle.

        Always builds the abort flag with the default ``lock=True``: the
        per-token-tick acquire cost is negligible vs llama-cpp inference,
        and lockless ``Value`` access is not documented atomic on ARM
        (M-series Macs, Snapdragon Windows). The cost of a missed abort
        on those platforms is too high.
        """
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        abort_flag = self._ctx.Value("b", 0, lock=True)
        process = self._ctx.Process(
            target=worker_main,
            args=(child_conn, abort_flag, role_config),
            daemon=self._daemon,
            name=f"lilbee-worker-{role_config.role}",
        )
        process.start()
        child_conn.close()
        channel = PipeChannel(
            role=role_config.role,
            process=process,
            parent_conn=parent_conn,
            abort_flag=abort_flag,
        )
        handle = WorkerHandle(pid=process.pid, role=role_config.role)
        log.info("Spawned worker role=%s pid=%s", role_config.role, process.pid)
        return channel, handle


__all__ = [
    "PipeChannel",
    "PipeSpawner",
    "WorkerCrashError",
    "WorkerError",
]
