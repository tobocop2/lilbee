"""``multiprocessing.Pipe``-backed worker channel and spawner.

Concrete impl of the ``WorkerChannel`` / ``WorkerSpawner`` Protocols
from :mod:`lilbee.providers.worker.transport`. Pipe-specific discipline
rules are documented in ``docs/architecture.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
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
"""``Connection.send`` raises past about 32 MiB on POSIX."""

_CONTROL_CALL_ID = 0
"""Sentinel call-id for shutdown/ack frames not associated with a user call."""


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

    def __init__(self, role: str, *, log_path: str | None = None) -> None:
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


def _check_pickle_size(payload: Any, kind: str, call_id: int) -> None:
    """Raise ``ValueError`` early if *payload* would exceed the pipe send cap."""
    try:
        size = len(pickle.dumps((call_id, kind, payload)))
    except Exception as exc:
        raise WorkerError("PickleError", f"Failed to pickle {kind!r} payload: {exc}", "") from exc
    if size > _PICKLE_MAX_BYTES:
        raise WorkerError(
            "PayloadTooLarge",
            f"{kind!r} payload is {size} bytes; pipe send cap is {_PICKLE_MAX_BYTES}.",
            "",
        )


def _worker_log_path(role: str) -> str | None:
    """Return the worker's log file path if ``LILBEE_DATA`` is set."""
    import os

    data_dir = os.environ.get("LILBEE_DATA")
    if not data_dir:
        return None
    return os.path.join(data_dir, "logs", f"worker-{role}.log")


class PipeChannel:
    """One worker process talked to via a duplex :class:`multiprocessing.Pipe`.

    Owns the parent end of two pipes (data and health), the abort flag,
    and a per-channel :class:`ThreadPoolExecutor`. Frames carry a monotonic
    ``call_id`` so leftover frames from a cancelled prior call can be
    discarded by the next call's reader instead of poisoning the protocol.
    """

    def __init__(
        self,
        *,
        role: str,
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
            max_workers=4,
            thread_name_prefix=f"pipechan-{role}",
        )
        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()
        self._recv_thread_lock = threading.Lock()
        self._health_send_lock = asyncio.Lock()
        self._health_recv_lock = asyncio.Lock()
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._closed = False
        self._closed_lock = threading.Lock()
        self._call_ids = itertools.count(start=1)
        self._call_id_lock = threading.Lock()

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

    def _next_call_id(self) -> int:
        with self._call_id_lock:
            return next(self._call_ids)

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

    async def _send(self, call_id: int, kind: str, payload: Any) -> None:
        """Pickle-pre-check + thread-bounded ``conn.send`` under the send lock."""
        _check_pickle_size(payload, kind, call_id)
        loop = asyncio.get_running_loop()
        async with self._send_lock:
            try:
                await loop.run_in_executor(
                    self._executor, self._conn.send, (call_id, kind, payload)
                )
            except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
                raise self._crash() from exc

    def _conn_recv_serialized(self) -> Any:
        """Serialize ``conn.recv`` across executor threads with a threading lock."""
        with self._recv_thread_lock:
            return self._conn.recv()

    async def _recv(self) -> tuple[int, str, Any]:
        """Thread-bounded ``conn.recv`` under the recv lock; raises on EOF/crash."""
        loop = asyncio.get_running_loop()
        async with self._recv_lock:
            try:
                return await loop.run_in_executor(self._executor, self._conn_recv_serialized)
            except (EOFError, OSError, ConnectionResetError, BrokenPipeError) as exc:
                raise self._crash() from exc

    async def _recv_for(self, expected_call_id: int) -> tuple[str, Any]:
        """Read frames until one matches *expected_call_id*; drain stale ones."""
        while True:
            call_id, kind, value = await self._recv()
            if call_id == expected_call_id:
                return kind, value
            log.debug(
                "Worker %s: discarding stale frame call_id=%s kind=%s (expected %s)",
                self._role,
                call_id,
                kind,
                expected_call_id,
            )

    async def call(self, kind: str, payload: Any, *, timeout: float) -> Any:
        """Send one request, await one reply on the data pipe.

        Frames carry a per-call id; leftover frames from a cancelled prior
        call are silently drained instead of raising :class:`ProtocolError`.
        """
        self._ensure_open()
        call_id = self._next_call_id()
        self._bump_in_flight(1)
        try:
            await self._send(call_id, kind, payload)
            msg_kind, value = await asyncio.wait_for(self._recv_for(call_id), timeout=timeout)
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
        """Send one request, yield streamed chunks on the data pipe."""
        self._ensure_open()
        call_id = self._next_call_id()
        self._bump_in_flight(1)
        await self._send(call_id, kind, payload)
        try:
            while True:
                msg_kind, value = await self._recv_for(call_id)
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
        """Round-trip a ping over the dedicated health pipe; raise on timeout.

        The health pipe is owned by a worker-side daemon thread that
        answers ping → pong without depending on the data-frame handler,
        so a long inference cannot starve the heartbeat.
        """
        self._ensure_open()
        loop = asyncio.get_running_loop()
        try:
            async with self._health_send_lock:
                await loop.run_in_executor(
                    self._executor,
                    self._health_conn.send,
                    (_CONTROL_CALL_ID, PING_KIND, None),
                )
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
            raise self._crash() from exc
        async with self._health_recv_lock:
            try:
                _call_id, msg_kind, _ = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._health_conn.recv),
                    timeout=timeout,
                )
            except (EOFError, OSError, ConnectionResetError, BrokenPipeError) as exc:
                raise self._crash() from exc
        if msg_kind != PONG_KIND:
            raise WorkerError(
                "ProtocolError",
                f"Worker '{self._role}' ping reply was {msg_kind!r}, want 'pong'.",
                "",
            )

    def cancel(self) -> None:
        """Flip the abort flag to 1; in-flight tokens may still drain."""
        self._abort.value = 1

    def clear_abort(self) -> None:
        """Reset the abort flag to 0 before the next request."""
        self._abort.value = 0

    async def close(self, *, timeout: float) -> None:
        """Send shutdown, await ack, terminate if it overruns *timeout*."""
        with self._closed_lock:
            if self._closed:
                return
            self._closed = True
        try:
            with contextlib.suppress(asyncio.TimeoutError, WorkerError):
                await self._send(_CONTROL_CALL_ID, SHUTDOWN_KIND, None)
                with contextlib.suppress(asyncio.TimeoutError, WorkerError):
                    await asyncio.wait_for(self._recv(), timeout=timeout)
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

        Two pipes per worker: ``data_pipe`` carries call/stream/shutdown
        traffic, ``health_pipe`` carries ping/pong. The worker dedicates a
        daemon thread to the health pipe so heartbeats stay live during
        long inference.
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
