"""Lifecycle for per-role inference subprocess workers.

Owns the embed, chat, rerank, and vision worker processes. Lifecycle
contract, restart-budget policy, idle reaping, and health pings are
documented in ``docs/architecture.md`` under "Inference worker pool".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

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
    # circular: health_ticker -> pool via WorkerPool/PoolRuntime
    from lilbee.providers.worker.health_ticker import HealthTickerHandle

log = logging.getLogger(__name__)

_T = TypeVar("_T")


_DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0
_DEFAULT_CALL_TIMEOUT_S = 300.0
_DEFAULT_MAX_IDLE_S = 0.0  # 0 = no idle reaping by default
_HEALTH_TIMEOUT_S = 5.0
_RESTART_BUDGET = 3
_RESTART_WINDOW_S = 60.0
_RUNTIME_THREAD_NAME = "lilbee-worker-pool-loop"


class PoolShutdownError(WorkerError):
    """Raised when a caller tries to use a pool that has been shut down."""

    def __init__(self) -> None:
        super().__init__(
            "PoolShutdownError",
            "Inference pool is shutting down. Please wait for current tasks to finish.",
            "",
        )


class RoleDegradedError(WorkerError):
    """Raised when a role has burned through its restart budget."""

    def __init__(self, role: str, attempts: int, window_s: float) -> None:
        super().__init__(
            "RoleDegradedError",
            (
                f"The {role} worker crashed {attempts} times in the last "
                f"{window_s:.0f} seconds and is now disabled. Restart "
                "lilbee to recover."
            ),
            "",
        )
        self.role = role


@dataclass
class _Role:
    """Per-role registration: how to spawn it plus its live channel (if any)."""

    name: str
    worker_main: WorkerEntrypoint
    config_factory: Callable[[], RoleConfig]
    channel: WorkerChannel | None = None
    spawn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = 0.0
    crash_history: deque[float] = field(default_factory=deque)
    degraded: bool = False


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
            result = await channel.call(kind, payload, timeout=timeout)
        except WorkerCrashError:
            await self._pool._on_crash(self._role)
            raise
        self._pool._stamp_used(self._role)
        return result

    def stream(self, kind: str, payload: object) -> object:
        """Lazy-spawn (synchronously async) and return the channel's async iterator.

        The returned object is the ``stream`` async iterator from the
        underlying :class:`WorkerChannel`; the spawn step is folded into
        the iterator's first ``__anext__`` via :func:`_spawn_and_stream`.
        """
        return _spawn_and_stream(self._pool, self._role, kind, payload)

    async def ping(self, *, timeout: float) -> None:
        """Health check: lazy-spawn if needed, then ping the worker.

        Skips when the channel already has in-flight work: that work is
        itself liveness evidence, and an out-of-band ping would race the
        in-flight frames on the pipe (the parent's ping recv could read
        a stream chunk, the worker's pong reply could land in front of a
        subsequent stream's frames).
        """
        channel = await self._pool._ensure_channel(self._role)
        if channel.in_flight > 0:
            self._pool._stamp_used(self._role)
            return
        try:
            await channel.ping(timeout=timeout)
        except WorkerCrashError:
            await self._pool._on_crash(self._role)
            raise
        self._pool._stamp_used(self._role)

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
    pool._stamp_used(role)


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

    def __init__(
        self,
        *,
        spawner: WorkerSpawner | None = None,
        max_idle_s: float = _DEFAULT_MAX_IDLE_S,
    ) -> None:
        self._spawner: WorkerSpawner = spawner if spawner is not None else PipeSpawner()
        self._roles: dict[str, _Role] = {}
        self._shutdown = False
        self._shutdown_lock = asyncio.Lock()
        self._max_idle_s = max_idle_s

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
        if registration.degraded:
            raise RoleDegradedError(role, _RESTART_BUDGET, _RESTART_WINDOW_S)
        if registration.channel is not None and registration.channel.is_alive:
            return registration.channel
        async with registration.spawn_lock:
            if registration.channel is not None and registration.channel.is_alive:
                return registration.channel
            if registration.degraded:
                raise RoleDegradedError(role, _RESTART_BUDGET, _RESTART_WINDOW_S)
            self._raise_if_shutdown()
            channel, _handle = await asyncio.get_running_loop().run_in_executor(
                None,
                self._spawner.spawn,
                registration.worker_main,
                registration.config_factory(),
            )
            registration.channel = channel
            registration.last_used = time.monotonic()
            log.info("Worker pool spawned role=%s pid=%s", role, channel.pid)
            return channel

    def _stamp_used(self, role: str) -> None:
        """Update *role*'s ``last_used`` timestamp; called by the accessor."""
        registration = self._roles.get(role)
        if registration is not None:
            registration.last_used = time.monotonic()

    def _channel_if_alive(self, role: str) -> WorkerChannel | None:
        """Return the role's live channel without spawning; None if absent or dead."""
        registration = self._roles.get(role)
        if registration is None or registration.channel is None:
            return None
        if not registration.channel.is_alive:
            return None
        return registration.channel

    async def _on_crash(self, role: str) -> None:
        """Drop a crashed channel; mark degraded if the restart budget is exhausted."""
        registration = self._roles.get(role)
        if registration is None:
            return
        async with registration.spawn_lock:
            channel = registration.channel
            registration.channel = None
            now = time.monotonic()
            cutoff = now - _RESTART_WINDOW_S
            while registration.crash_history and registration.crash_history[0] < cutoff:
                registration.crash_history.popleft()
            registration.crash_history.append(now)
            if len(registration.crash_history) > _RESTART_BUDGET:
                registration.degraded = True
                log.error(
                    "Worker pool marking role=%s degraded after %d crashes in %.0fs",
                    role,
                    len(registration.crash_history),
                    _RESTART_WINDOW_S,
                )
        if channel is not None:
            with contextlib.suppress(WorkerError):
                await channel.close(timeout=_DEFAULT_SHUTDOWN_TIMEOUT_S)

    def reset_role_failures(self, role: str) -> None:
        """Clear *role*'s degraded mark and crash history.

        Used by an explicit "retry" UI affordance so the user can recover
        without restarting lilbee. Returns silently if the role is
        unregistered.
        """
        registration = self._roles.get(role)
        if registration is None:
            return
        registration.crash_history.clear()
        registration.degraded = False

    def is_degraded(self, role: str) -> bool:
        """Return True iff *role* is currently disabled by the restart-budget rule."""
        registration = self._roles.get(role)
        return registration is not None and registration.degraded

    async def reap_idle(self) -> tuple[str, ...]:
        """Close any role idle longer than ``max_idle_s`` with zero in-flight.

        Caller (typically a background async task) decides cadence; the
        pool does not own a recurring timer because the host TUI's loop
        is the authoritative scheduler.

        Returns the role names that were reaped (informational; useful
        for tests). No-op when ``max_idle_s == 0``.
        """
        if self._max_idle_s <= 0.0:
            return ()
        now = time.monotonic()
        reaped: list[str] = []
        for role_name, registration in list(self._roles.items()):
            channel = registration.channel
            if channel is None or not channel.is_alive:
                continue
            if channel.in_flight > 0:
                continue
            if registration.last_used <= 0.0:
                continue
            if now - registration.last_used < self._max_idle_s:
                continue
            async with registration.spawn_lock:
                # Re-check inside the lock; another coroutine may have just used it.
                if channel.in_flight > 0:
                    continue
                if now - registration.last_used < self._max_idle_s:
                    continue
                registration.channel = None
            with contextlib.suppress(WorkerError):
                await channel.close(timeout=_DEFAULT_SHUTDOWN_TIMEOUT_S)
            log.info(
                "Worker pool reaped idle role=%s after %.0fs",
                role_name,
                now - registration.last_used,
            )
            reaped.append(role_name)
        return tuple(reaped)

    async def ping_role(
        self,
        role: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """Round-trip a ping/pong against *role*; raise on timeout / crash.

        ``timeout`` defaults to the module-level ``_HEALTH_TIMEOUT_S``.
        Spawns the worker on first use, same as a real call. Caller
        (typically a background health monitor) decides cadence and
        whether to respond by reaping/restarting; this method only
        propagates the round-trip outcome.
        """
        accessor = self.accessor(role)
        budget = timeout if timeout is not None else _HEALTH_TIMEOUT_S
        await accessor.ping(timeout=budget)

    async def release(self, role: str) -> None:
        """Close *role*'s live worker and forget the registration entirely.

        Used by callers (notably ``LlamaCppProvider.invalidate_load_cache``)
        that want the next request to respawn with a fresh model picked
        from the current cfg. The role can be re-registered immediately
        after with :meth:`register`. No-op if the role is unregistered.
        """
        registration = self._roles.pop(role, None)
        if registration is None:
            return
        channel = registration.channel
        registration.channel = None
        if channel is not None:
            with contextlib.suppress(WorkerError):
                await channel.close(timeout=_DEFAULT_SHUTDOWN_TIMEOUT_S)

    def _raise_if_shutdown(self) -> None:
        if self._shutdown:
            raise PoolShutdownError()


class PoolRuntime:
    """Background asyncio loop dedicated to a single :class:`WorkerPool`.

    Sync callers (``LlamaCppProvider.embed`` and friends) invoke pool
    coroutines via :meth:`run_sync`, which submits them onto this loop
    and blocks the caller's thread for the result. Because every pool
    operation runs on the same loop, the per-role asyncio.Lock instances
    inside :class:`_Role` retain their semantics across concurrent
    sync callers.

    Constructed once per pool. :meth:`shutdown` stops the loop and
    joins the thread; subsequent :meth:`run_sync` calls raise
    :class:`PoolShutdownError`.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Spin up the background thread + loop. Idempotent."""
        with self._lock:
            if self._thread is not None:
                return
            if self._stopped:
                raise PoolShutdownError()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=_RUNTIME_THREAD_NAME,
                daemon=True,
            )
            self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def run_sync(self, coro: Coroutine[Any, Any, _T], *, timeout: float | None = None) -> _T:
        """Submit *coro* to the background loop and block for the result.

        On timeout, cancels the underlying asyncio task before raising so
        the loop does not log "Task was destroyed but it is pending".
        """
        if self._stopped:
            coro.close()
            raise PoolShutdownError()
        if self._thread is None:
            self.start()
        loop = self._loop
        assert loop is not None  # _ready signaled, loop is set  # noqa: S101
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise

    def submit(self, coro: Coroutine[Any, Any, _T]) -> Future[_T]:
        """Schedule *coro* on the background loop without blocking the caller.

        Returns the :class:`concurrent.futures.Future` for the call so the
        caller can await it (via :func:`asyncio.wrap_future` from another
        loop) or cancel it. Used by the Services-owned health ticker so a
        long pool ping does not stall the bg-loop.
        """
        if self._stopped:
            coro.close()
            raise PoolShutdownError()
        if self._thread is None:
            self.start()
        loop = self._loop
        assert loop is not None  # _ready signaled, loop is set  # noqa: S101
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def shutdown(self, *, timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT_S) -> None:
        """Stop the loop and join the thread. Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            loop = self._loop
            thread = self._thread
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)


def shutdown_pool_runtime(
    pool: WorkerPool,
    runtime: PoolRuntime,
    ticker: HealthTickerHandle,
    *,
    drain_timeout_s: float = 10.0,
    runtime_timeout_s: float = 5.0,
) -> None:
    """Stop the health ticker, drain the pool through the runtime, then stop the runtime.

    Order matters: cancel the ticker first so it cannot schedule a fresh
    pool op against a draining runtime; then drain the pool via the
    runtime; then stop the runtime thread. Idempotent.
    """
    from lilbee.providers.worker.health_ticker import stop_health_ticker

    stop_health_ticker(ticker)
    try:
        runtime.run_sync(pool.shutdown(), timeout=drain_timeout_s)
    except (TimeoutError, RuntimeError, OSError) as exc:
        log.warning("Pool shutdown raised %s; forcing runtime stop", exc)
    runtime.shutdown(timeout=runtime_timeout_s)


__all__ = [
    "PoolRuntime",
    "PoolShutdownError",
    "RoleAccessor",
    "WorkerPool",
    "shutdown_pool_runtime",
]
