"""Persistent worker-pool manager.

Owns the lifecycle of all per-role workers (embed, chat, rerank, vision)
in the TUI process. Talks to workers exclusively through the
:class:`lilbee.providers.worker.transport.WorkerSpawner` /
:class:`lilbee.providers.worker.transport.WorkerChannel` Protocols, so
swapping the IPC primitive (mp.Pipe today, pyzmq tomorrow) is a one-file
change in the transport layer.

Lifecycle contract
==================

1. ``WorkerPool(spawner=...)``: builds the pool object. **No subprocesses
   spawned yet.** Roles are registered with their entrypoint + role config
   factory but not started.
2. ``await pool.start_eager()``: spawns one process per registered role
   concurrently. Returns when all are up. Optional, gated on the caller's
   own config (``cfg.worker_pool_eager_start``).
3. ``await pool.<role>.call(...)``: lazy-spawn the role's worker on first
   call. Subsequent calls reuse the live channel.
4. ``await pool.shutdown(timeout=5.0)``: send shutdown to every live
   worker, await graceful exit, terminate stragglers. Idempotent.
5. Per-role accessors raise :class:`PoolShutdownError` after ``shutdown``.

The pool itself is async-safe: per-role accessor lookups and lazy spawn
serialize on a per-role asyncio.Lock so two concurrent first-callers do
not race to spawn two workers.

Restart-on-crash, idle reaping, and health pings ride on top of the same
accessor: if a channel reports ``is_alive == False`` (or raises
:class:`WorkerCrashError`), the accessor drops it and the next call
re-spawns. Concrete restart bookkeeping (attempts within a window, mark-
as-degraded) lands with the per-role workers in subsequent commits.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lilbee.providers.worker.transport import (
    RoleConfig,
    WorkerChannel,
    WorkerEntrypoint,
    WorkerSpawner,
)
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerCrashError,
    WorkerError,
)

if TYPE_CHECKING:
    from lilbee.providers.worker.transport import WorkerHandle

log = logging.getLogger(__name__)


_DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0
_DEFAULT_CALL_TIMEOUT_S = 300.0


class PoolShutdownError(WorkerError):
    """Raised when a caller tries to use a pool that has been shut down."""

    def __init__(self) -> None:
        super().__init__(
            "PoolShutdownError",
            "Inference pool is shutting down. Please wait for current tasks to finish.",
            "",
        )


@dataclass
class _Role:
    """Per-role registration: how to spawn it plus its live channel (if any)."""

    name: str
    worker_main: WorkerEntrypoint
    config_factory: Callable[[], RoleConfig]
    channel: WorkerChannel | None = None
    handle: WorkerHandle | None = None
    spawn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RoleAccessor:
    """Per-role facade returned by ``pool.<role>``.

    Wraps the lazy-spawn dance so callers write ``await pool.embed.call(...)``
    instead of ``await pool._spawn_then_call("embed", ...)``. Per-role
    workers in subsequent commits add typed convenience methods (e.g.
    ``await pool.embed.batch(texts)``) on top of the generic ``call`` /
    ``stream`` here.
    """

    def __init__(self, pool: WorkerPool, role: str) -> None:
        self._pool = pool
        self._role = role

    async def call(
        self,
        kind: str,
        payload: object,
        *,
        timeout: float = _DEFAULT_CALL_TIMEOUT_S,
    ) -> object:
        """Lazy-spawn the worker on first call, then dispatch one request."""
        channel = await self._pool._ensure_channel(self._role)
        try:
            return await channel.call(kind, payload, timeout=timeout)
        except WorkerCrashError:
            await self._pool._on_crash(self._role)
            raise

    def stream(self, kind: str, payload: object) -> object:
        """Lazy-spawn (synchronously async) and return the channel's async iterator.

        The returned object is the ``stream`` async iterator from the
        underlying :class:`WorkerChannel`; the spawn step is folded into
        the iterator's first ``__anext__`` via :func:`_spawn_and_stream`.
        """
        return _spawn_and_stream(self._pool, self._role, kind, payload)

    async def ping(self, *, timeout: float) -> None:
        """Health check: lazy-spawn if needed, then ping the worker."""
        channel = await self._pool._ensure_channel(self._role)
        try:
            await channel.ping(timeout=timeout)
        except WorkerCrashError:
            await self._pool._on_crash(self._role)
            raise

    def cancel(self) -> None:
        """Flip the worker's abort flag if it is alive; no-op otherwise."""
        channel = self._pool._channel_if_alive(self._role)
        if channel is not None:
            channel.cancel()

    def clear_abort(self) -> None:
        """Reset the worker's abort flag if it is alive; no-op otherwise."""
        channel = self._pool._channel_if_alive(self._role)
        if channel is not None:
            channel.clear_abort()

    @property
    def is_alive(self) -> bool:
        """True iff the worker has been spawned and its process is alive."""
        return self._pool._channel_if_alive(self._role) is not None


async def _spawn_and_stream(
    pool: WorkerPool,
    role: str,
    kind: str,
    payload: object,
) -> AsyncIterator[object]:
    """Async generator that spawns the worker on first iteration, then streams."""
    channel = await pool._ensure_channel(role)
    try:
        async for chunk in channel.stream(kind, payload):
            yield chunk
    except WorkerCrashError:
        await pool._on_crash(role)
        raise


class WorkerPool:
    """Owns every long-lived worker process.

    Constructor takes an explicit *spawner* so tests can plug in an
    in-process fake (see ``tests/test_worker_pool.py``). Production callers
    pass nothing and get the default :class:`PipeSpawner`.

    Roles are registered with :meth:`register` before the first call; the
    pool itself does not import the per-role worker entrypoint modules so
    we keep the worker-side and parent-side code paths clearly separated.
    Registration is intentionally explicit: callers (typically ``Services``)
    decide which roles exist in this process.
    """

    def __init__(self, *, spawner: WorkerSpawner | None = None) -> None:
        self._spawner: WorkerSpawner = spawner if spawner is not None else PipeSpawner()
        self._roles: dict[str, _Role] = {}
        self._shutdown = False
        self._shutdown_lock = asyncio.Lock()

    def register(
        self,
        role: str,
        worker_main: WorkerEntrypoint,
        config_factory: Callable[[], RoleConfig],
    ) -> RoleAccessor:
        """Register a role's worker entrypoint and return its accessor.

        ``config_factory`` is called every time the role spawns (lazy or
        on restart) so model swaps in cfg propagate without an explicit
        invalidation: the next spawn picks up whatever
        ``config_factory()`` returns.
        """
        if role in self._roles:
            raise ValueError(f"Role {role!r} is already registered on this pool.")
        self._roles[role] = _Role(
            name=role,
            worker_main=worker_main,
            config_factory=config_factory,
        )
        return RoleAccessor(self, role)

    def accessor(self, role: str) -> RoleAccessor:
        """Return the :class:`RoleAccessor` for *role*; must already be registered."""
        if role not in self._roles:
            raise KeyError(f"Role {role!r} is not registered on this pool.")
        return RoleAccessor(self, role)

    @property
    def registered_roles(self) -> tuple[str, ...]:
        """Names of every role registered on this pool, in registration order."""
        return tuple(self._roles)

    async def start_eager(self) -> None:
        """Spawn every registered role concurrently; raise on first spawn failure.

        Optional. Most callers rely on lazy spawn via the accessors. Use
        this when you want to absorb the per-worker cold-start cost up
        front (e.g. just after the TUI mounts so the first user action
        does not also pay for spawn).
        """
        self._raise_if_shutdown()
        await asyncio.gather(*(self._ensure_channel(role) for role in self._roles))

    async def shutdown(self, *, timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT_S) -> None:
        """Send shutdown to every live worker, terminate stragglers past *timeout*.

        Idempotent. Safe to register on ``atexit``.
        """
        async with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
        live: list[WorkerChannel] = [
            role.channel for role in self._roles.values() if role.channel is not None
        ]
        for role in self._roles.values():
            role.channel = None
            role.handle = None
        await asyncio.gather(
            *(channel.close(timeout=timeout) for channel in live),
            return_exceptions=True,
        )

    async def _ensure_channel(self, role: str) -> WorkerChannel:
        """Return the role's live channel, spawning it on first use or after crash."""
        self._raise_if_shutdown()
        registration = self._roles.get(role)
        if registration is None:
            raise KeyError(f"Role {role!r} is not registered on this pool.")
        if registration.channel is not None and registration.channel.is_alive:
            return registration.channel
        async with registration.spawn_lock:
            if registration.channel is not None and registration.channel.is_alive:
                return registration.channel
            self._raise_if_shutdown()
            channel, handle = await asyncio.get_running_loop().run_in_executor(
                None,
                self._spawner.spawn,
                registration.worker_main,
                registration.config_factory(),
            )
            registration.channel = channel
            registration.handle = handle
            log.info("Worker pool spawned role=%s pid=%s", role, handle.pid)
            return channel

    def _channel_if_alive(self, role: str) -> WorkerChannel | None:
        """Return the role's live channel without spawning; None if absent or dead."""
        registration = self._roles.get(role)
        if registration is None or registration.channel is None:
            return None
        if not registration.channel.is_alive:
            return None
        return registration.channel

    async def _on_crash(self, role: str) -> None:
        """Drop a crashed channel so the next call respawns; safe to call repeatedly."""
        registration = self._roles.get(role)
        if registration is None:
            return
        async with registration.spawn_lock:
            channel = registration.channel
            registration.channel = None
            registration.handle = None
        if channel is not None:
            with contextlib.suppress(WorkerError):
                await channel.close(timeout=_DEFAULT_SHUTDOWN_TIMEOUT_S)

    def _raise_if_shutdown(self) -> None:
        if self._shutdown:
            raise PoolShutdownError()


__all__ = [
    "PoolShutdownError",
    "RoleAccessor",
    "WorkerPool",
]
