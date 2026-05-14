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
    WorkerRole,
)
from lilbee.providers.worker.wire_kinds import WireKind

log = logging.getLogger(__name__)


_PICKLE_MAX_BYTES = 32 * 1024 * 1024
"""``Connection.send`` raises past about 32 MiB on POSIX."""


@dataclass(frozen=True)
class _SerializedException:
    """Pickle-friendly stand-in for an exception that crossed the wire."""

    type_name: str
    message: str
    traceback_str: str


class WorkerError(RuntimeError):
    """Raised on the parent side when a worker reports an exception."""

    def __init__(self, original_type: str, message: str, traceback_str: str) -> None:
        super().__init__(f"{original_type}: {message}")
        self.original_type = original_type
        self.traceback_str = traceback_str


class WorkerCrashError(WorkerError):
    """Raised when a worker process dies mid-request (EOF on the pipe).

    Carries an optional ``log_path`` so the surfaced message can point the
    user at the worker log file that contains the underlying traceback or
    signal info.
    """

    def __init__(self, role: WorkerRole, *, log_path: str | None = None) -> None:
        suffix = f" See {log_path} for details." if log_path else ""
        super().__init__(
            "WorkerCrashError",
            f"Worker '{role}' subprocess exited unexpectedly.{suffix}",
            "",
        )
        self.role = role
        self.log_path = log_path


def _serialize_exception(exc: BaseException) -> _SerializedException:
    """Reduce an exception to a pickle-safe ``(type_name, message, traceback)`` triple."""
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _SerializedException(
        type_name=type(exc).__name__,
        message=str(exc),
        traceback_str=tb_str,
    )


def _deserialize_exception(payload: _SerializedException) -> WorkerError:
    """Rebuild a parent-side exception from a serialized worker exception."""
    return WorkerError(payload.type_name, payload.message, payload.traceback_str)


def _check_pickle_size(payload: Any, kind: WireKind) -> None:
    """Raise ``WorkerError`` early if *payload* would exceed the pipe send cap."""
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


def _worker_log_path(role: WorkerRole) -> str | None:
    """Return the worker's log file path if ``LILBEE_DATA`` is set."""
    import os

    data_dir = os.environ.get("LILBEE_DATA")
    if not data_dir:
        return None
    return os.path.join(data_dir, "logs", f"worker-{role}.log")


class PipeChannel:
    """One worker process talked to via a duplex :class:`multiprocessing.Pipe`.

    Owns the parent end of two pipes (data and health), the abort flag,
    and a per-channel :class:`ThreadPoolExecutor`. The data pipe carries
    one call at a time: ``call`` and ``stream`` acquire ``_call_lock`` for
    their full request/reply or request/stream lifetime, so a reply (or
    stream chunk) can only ever belong to the call currently holding the
    lock. The health pipe carries ping/pong and shutdown/ack and is
    served by a dedicated daemon thread on the worker side, so a long
    inference never starves liveness or shutdown.
    """

    def __init__(
        self,
        *,
        role: WorkerRole,
        process: multiprocessing.process.BaseProcess,
        parent_conn: Any,
        health_conn: Any,
        abort_flag: Any,
    ) -> None:
        self._role = role
        self._process = process
        self._conn = parent_conn
        self._health_conn = health_conn
        self._abort = abort_flag
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"pipechan-{role}",
        )
        self._call_lock = asyncio.Lock()
        self._health_lock = asyncio.Lock()
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._closed = False
        self._closed_lock = threading.Lock()

    @property
    def role(self) -> WorkerRole:
        """Worker role this channel addresses."""
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

    def _crash(self) -> WorkerCrashError:
        return WorkerCrashError(self._role, log_path=_worker_log_path(self._role))

    async def _send_data(self, kind: WireKind, payload: Any) -> None:
        """Pickle-pre-check + thread-bounded ``conn.send`` on the data pipe."""
        _check_pickle_size(payload, kind)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._conn.send, (kind, payload))
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
            raise self._crash() from exc

    async def _recv_data(self) -> tuple[WireKind, Any]:
        """Block in a worker thread until the next ``(kind, payload)`` frame arrives."""
        loop = asyncio.get_running_loop()
        try:
            frame = await loop.run_in_executor(self._executor, self._conn.recv)
        except (EOFError, OSError, ConnectionResetError, BrokenPipeError) as exc:
            raise self._crash() from exc
        return frame  # type: ignore[no-any-return]

    async def call(self, kind: WireKind, payload: Any, *, timeout: float) -> Any:
        """Send one request, await one reply on the data pipe.

        The ``_call_lock`` is held for the full request/reply window so a
        reply that lands on the pipe can only belong to the call holding
        the lock. New callers queue on the lock behind the in-flight one.
        """
        self._ensure_open()
        async with self._call_lock:
            self._bump_in_flight(1)
            try:
                await self._send_data(kind, payload)
                msg_kind, value = await asyncio.wait_for(self._recv_data(), timeout=timeout)
                if msg_kind == WireKind.ERROR:
                    raise _deserialize_exception(value)
                if msg_kind != WireKind.RESULT:
                    raise WorkerError(
                        "ProtocolError",
                        f"Worker '{self._role}' replied with unexpected kind {msg_kind!r}.",
                        "",
                    )
                return value
            finally:
                self._bump_in_flight(-1)

    async def stream(self, kind: WireKind, payload: Any) -> AsyncIterator[Any]:
        """Send one request, yield streamed chunks until the terminator arrives.

        The ``_call_lock`` is held for the full stream lifetime, so frames
        recv'd by this coroutine belong to this stream by construction.
        New callers queue behind the active stream.
        """
        self._ensure_open()
        async with self._call_lock:
            self._bump_in_flight(1)
            try:
                await self._send_data(kind, payload)
                while True:
                    msg_kind, value = await self._recv_data()
                    if msg_kind == WireKind.STREAM_CHUNK:
                        yield value
                    elif msg_kind == WireKind.STREAM_END:
                        return
                    elif msg_kind == WireKind.ERROR:
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
        """Round-trip a ping over the health pipe; raise on timeout / crash.

        The worker dedicates a daemon thread to the health pipe so pings
        and shutdown can be served while the data loop is busy in
        ``session.embed`` / ``create_chat_completion``.
        """
        self._ensure_open()
        kind = await self._health_round_trip(WireKind.PING, None, timeout=timeout)
        if kind != WireKind.PONG:
            raise WorkerError(
                "ProtocolError",
                f"Worker '{self._role}' ping reply was {kind!r}, want 'pong'.",
                "",
            )

    async def _health_round_trip(
        self, send_kind: WireKind, send_payload: Any, *, timeout: float
    ) -> WireKind:
        """Send one frame on the health pipe and await its reply within *timeout*."""
        loop = asyncio.get_running_loop()
        async with self._health_lock:
            try:
                await loop.run_in_executor(
                    self._executor, self._health_conn.send, (send_kind, send_payload)
                )
            except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
                raise self._crash() from exc
            try:
                frame = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._health_conn.recv),
                    timeout=timeout,
                )
            except (EOFError, OSError, ConnectionResetError, BrokenPipeError) as exc:
                raise self._crash() from exc
        reply_kind, _ = frame
        return reply_kind  # type: ignore[no-any-return]

    def cancel(self) -> None:
        """Flip the abort flag to 1; in-flight tokens may still drain."""
        self._abort.value = 1

    def clear_abort(self) -> None:
        """Reset the abort flag to 0 before the next request."""
        self._abort.value = 0

    async def close(self, *, timeout: float) -> None:
        """Send shutdown on the health pipe, await ack, then join the process.

        The data pipe is never used for shutdown: a long in-flight call
        on the data pipe would otherwise serialize behind a shutdown
        request. Health-pipe shutdown is served by the worker's
        dedicated heartbeat thread, so this returns within the timeout
        regardless of what the data loop is doing. Any in-flight data
        call sees the process exit and surfaces as :class:`WorkerCrashError`.
        """
        with self._closed_lock:
            if self._closed:
                return
            self._closed = True
        try:
            with contextlib.suppress(TimeoutError, WorkerError):
                await self._health_round_trip(WireKind.SHUTDOWN, None, timeout=timeout)
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._join_process, timeout
            )
        finally:
            with contextlib.suppress(Exception):
                self._conn.close()
            with contextlib.suppress(Exception):
                self._health_conn.close()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _join_process(self, timeout: float) -> None:
        """Wait *timeout* seconds for the process; terminate if still alive.

        On non-clean exit (signal, non-zero code) record the exit reason in
        the worker log so the user has something to attach to a bug report.
        """
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            log.warning("Worker '%s' did not exit gracefully; terminating", self._role)
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._record_exit_reason()

    def _record_exit_reason(self) -> None:
        """Append worker exit reason (signal or non-zero code) to the worker log."""
        code = self._process.exitcode
        if code is None or code == 0:
            return
        log_path = _worker_log_path(self._role)
        message = self._format_exit_reason(code)
        log.warning("Worker '%s' %s", self._role, message)
        if log_path is None:
            return
        with contextlib.suppress(OSError), open(log_path, "a") as handle:
            handle.write(f"\n[supervisor] {message}\n")

    @staticmethod
    def _format_exit_reason(exit_code: int) -> str:
        if exit_code >= 0:
            return f"exited with code {exit_code}"
        import signal

        signum = -exit_code
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"SIG{signum}"
        return f"killed by signal {name} ({signum})"


class PipeSpawner:
    """Spawns worker subprocesses connected to the parent via :class:`multiprocessing.Pipe`."""

    def __init__(self, *, daemon: bool = True) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._daemon = daemon

    def spawn(
        self,
        worker_main: WorkerEntrypoint,
        role_config: RoleConfig,
    ) -> tuple[WorkerChannel, WorkerHandle]:
        """Start a worker subprocess and return its channel + handle.

        Two pipes per worker: ``data_pipe`` carries call/stream traffic,
        ``health_pipe`` carries ping/pong and shutdown/ack. The worker
        dedicates a daemon thread to the health pipe so heartbeats and
        shutdown stay live during long inference.
        """
        parent_data, child_data = self._ctx.Pipe(duplex=True)
        parent_health, child_health = self._ctx.Pipe(duplex=True)
        abort_flag = self._ctx.Value("b", 0, lock=True)
        process = self._ctx.Process(
            target=worker_main,
            args=(child_data, child_health, abort_flag, role_config),
            daemon=self._daemon,
            name=f"lilbee-worker-{role_config.role}",
        )
        process.start()
        child_data.close()
        child_health.close()
        channel = PipeChannel(
            role=role_config.role,
            process=process,
            parent_conn=parent_data,
            health_conn=parent_health,
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
