"""Tests for the WorkerPool lifecycle.

Use an in-process fake :class:`WorkerSpawner` so pool lifecycle (lazy
spawn, eager start, shutdown, crash detection, idle re-spawn) can be
exercised at unit-test speed without spawning real subprocesses. The
pipe transport itself is covered by ``test_worker_transport_pipe.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from lilbee.providers.worker.pool import (
    PoolRuntime,
    PoolShutdownError,
    RoleAccessor,
    WorkerPool,
)
from lilbee.providers.worker.transport import (
    RoleConfig,
    WorkerChannel,
    WorkerEntrypoint,
    WorkerHandle,
)
from lilbee.providers.worker.transport_pipe import WorkerCrashError, WorkerError

# =====================================================================
# In-process fake transport. Honors the WorkerChannel Protocol; the
# WorkerSpawner Protocol returns one such channel per spawn() call.
# =====================================================================


@dataclass
class FakeChannel:
    """Records every interaction the pool drives through it."""

    role_config: RoleConfig
    pid: int = 0
    alive: bool = True
    in_flight_count: int = 0
    closed: bool = False
    close_calls: int = 0
    cancel_calls: int = 0
    clear_calls: int = 0
    canned_call: dict[str, Any] = field(default_factory=dict)
    canned_stream: dict[str, list[Any]] = field(default_factory=dict)
    raise_on_call: dict[str, BaseException] = field(default_factory=dict)
    raise_on_stream: dict[str, BaseException] = field(default_factory=dict)
    call_log: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.alive and not self.closed

    @property
    def in_flight(self) -> int:
        return self.in_flight_count

    async def call(self, kind: str, payload: Any, *, timeout: float) -> Any:
        self.call_log.append((kind, payload))
        if kind in self.raise_on_call:
            raise self.raise_on_call[kind]
        return self.canned_call.get(kind, payload)

    async def stream(self, kind: str, payload: Any) -> AsyncIterator[Any]:
        self.call_log.append((f"stream:{kind}", payload))
        if kind in self.raise_on_stream:
            raise self.raise_on_stream[kind]
        for chunk in self.canned_stream.get(kind, []):
            yield chunk

    async def ping(self, *, timeout: float) -> None:
        self.call_log.append(("ping", None))
        if "ping" in self.raise_on_call:
            raise self.raise_on_call["ping"]

    def cancel(self) -> None:
        self.cancel_calls += 1

    def clear_abort(self) -> None:
        self.clear_calls += 1

    async def close(self, *, timeout: float) -> None:
        self.close_calls += 1
        self.closed = True


@dataclass
class FakeSpawner:
    """Spawner that constructs :class:`FakeChannel` instances on demand."""

    spawned: list[FakeChannel] = field(default_factory=list)
    next_pid: int = 1000
    fail_with: BaseException | None = None
    spawn_delay_s: float = 0.0

    def spawn(
        self,
        worker_main: WorkerEntrypoint,
        role_config: RoleConfig,
    ) -> tuple[WorkerChannel, WorkerHandle]:
        if self.fail_with is not None:
            raise self.fail_with
        if self.spawn_delay_s > 0.0:
            import time as _time

            _time.sleep(self.spawn_delay_s)
        channel = FakeChannel(role_config=role_config, pid=self.next_pid)
        self.next_pid += 1
        self.spawned.append(channel)
        return channel, WorkerHandle(pid=channel.pid, role=role_config.role)


def _entrypoint(_conn: Any, _abort: Any, _config: RoleConfig) -> None:  # pragma: no cover
    """Placeholder; FakeSpawner ignores the entrypoint."""


def _config_factory(role: str, tmp_path) -> Any:
    def _make() -> RoleConfig:
        return RoleConfig(role=role, model_path=tmp_path / f"{role}.gguf", mode="embed")

    return _make


# =====================================================================
# Lifecycle: lazy spawn, eager start, shutdown.
# =====================================================================


@pytest.mark.asyncio
async def test_register_returns_accessor_and_does_not_spawn(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    accessor = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        assert isinstance(accessor, RoleAccessor)
        assert spawner.spawned == []
        assert pool.registered_roles == ("embed",)
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_register_same_role_twice_raises(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        with pytest.raises(ValueError, match="already registered"):
            pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_accessor_unknown_role_raises(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    try:
        with pytest.raises(KeyError, match="not registered"):
            pool.accessor("missing")
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_accessor_returns_usable_role_handle_after_register(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        accessor = pool.accessor("embed")
        assert isinstance(accessor, RoleAccessor)
        result = await accessor.call("echo", "via-accessor-method")
        assert result == "via-accessor-method"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_ensure_channel_unknown_role_raises_keyerror(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    try:
        with pytest.raises(KeyError, match="not registered"):
            await pool._ensure_channel("missing")
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_on_crash_unknown_role_is_noop(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    try:
        # Must not raise; defensive no-op when the role was never registered.
        await pool._on_crash("missing")
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_call_lazy_spawns_on_first_use(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        assert spawner.spawned == []
        result = await embed.call("echo", {"value": 42})
        assert result == {"value": 42}
        assert len(spawner.spawned) == 1
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_repeated_calls_reuse_one_channel(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.call("echo", "a")
        await embed.call("echo", "b")
        await embed.call("echo", "c")
        assert len(spawner.spawned) == 1
        assert spawner.spawned[0].call_log == [("echo", "a"), ("echo", "b"), ("echo", "c")]
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_concurrent_first_callers_share_one_spawn(tmp_path) -> None:
    """Spawn is slow on purpose so the second caller exercises the inner-lock alive check."""
    spawner = FakeSpawner(spawn_delay_s=0.05)
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        results = await asyncio.gather(*(embed.call("echo", i) for i in range(5)))
        assert sorted(results) == [0, 1, 2, 3, 4]
        assert len(spawner.spawned) == 1
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_start_eager_spawns_every_registered_role(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    pool.register("rerank", _entrypoint, _config_factory("rerank", tmp_path))
    pool.register("chat", _entrypoint, _config_factory("chat", tmp_path))
    try:
        await pool.start_eager()
        assert {ch.role_config.role for ch in spawner.spawned} == {"embed", "rerank", "chat"}
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_every_spawned_channel(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    rerank = pool.register("rerank", _entrypoint, _config_factory("rerank", tmp_path))
    await embed.call("warm", None)
    await rerank.call("warm", None)
    assert len(spawner.spawned) == 2

    await pool.shutdown()
    assert all(ch.closed for ch in spawner.spawned)
    assert all(ch.close_calls == 1 for ch in spawner.spawned)


@pytest.mark.asyncio
async def test_shutdown_is_idempotent(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    await embed.call("warm", None)

    await pool.shutdown()
    await pool.shutdown()
    # Second shutdown must NOT close the channel again.
    assert spawner.spawned[0].close_calls == 1


@pytest.mark.asyncio
async def test_call_after_shutdown_raises_pool_shutdown_error(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    await pool.shutdown()
    with pytest.raises(PoolShutdownError):
        await embed.call("echo", "x")


@pytest.mark.asyncio
async def test_start_eager_after_shutdown_raises_pool_shutdown_error(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    await pool.shutdown()
    with pytest.raises(PoolShutdownError):
        await pool.start_eager()


# =====================================================================
# Crash recovery: a dead channel is dropped, next call respawns.
# =====================================================================


@pytest.mark.asyncio
async def test_crashed_channel_is_dropped_and_next_call_respawns(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.call("warm", None)
        first = spawner.spawned[0]
        first.raise_on_call["doomed"] = WorkerCrashError("embed")

        with pytest.raises(WorkerCrashError):
            await embed.call("doomed", None)

        assert pool._channel_if_alive("embed") is None
        # Next call must spawn a fresh channel.
        await embed.call("warm-again", None)
        assert len(spawner.spawned) == 2
        assert spawner.spawned[1].call_log[-1] == ("warm-again", None)
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_dead_channel_is_replaced_on_next_use(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.call("warm", None)
        first = spawner.spawned[0]
        # Simulate the OS process dying without us seeing a crash exception
        # (e.g. SIGKILL between requests).
        first.alive = False
        await embed.call("post-death", None)
        assert len(spawner.spawned) == 2
        assert spawner.spawned[1].call_log[-1] == ("post-death", None)
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_stream_propagates_crash_and_drops_channel(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    chat = pool.register("chat", _entrypoint, _config_factory("chat", tmp_path))
    try:
        await chat.call("warm", None)
        first = spawner.spawned[0]
        first.raise_on_stream["tokens"] = WorkerCrashError("chat")
        with pytest.raises(WorkerCrashError):
            async for _ in chat.stream("tokens", None):
                pass
        assert pool._channel_if_alive("chat") is None
    finally:
        await pool.shutdown()


# =====================================================================
# Streaming behavior through the accessor.
# =====================================================================


@pytest.mark.asyncio
async def test_stream_yields_canned_chunks(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    chat = pool.register("chat", _entrypoint, _config_factory("chat", tmp_path))
    try:
        await chat.call("warm", None)
        spawner.spawned[0].canned_stream["gen"] = ["a", "b", "c"]
        chunks = [chunk async for chunk in chat.stream("gen", {"prompt": "hi"})]
        assert chunks == ["a", "b", "c"]
    finally:
        await pool.shutdown()


# =====================================================================
# Cancel / clear_abort.
# =====================================================================


@pytest.mark.asyncio
async def test_cancel_flips_flag_when_channel_alive(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    chat = pool.register("chat", _entrypoint, _config_factory("chat", tmp_path))
    try:
        await chat.call("warm", None)
        chat.cancel()
        chat.clear_abort()
        assert spawner.spawned[0].cancel_calls == 1
        assert spawner.spawned[0].clear_calls == 1
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_cancel_is_noop_when_channel_not_spawned(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    chat = pool.register("chat", _entrypoint, _config_factory("chat", tmp_path))
    try:
        # No call() yet, so no spawn.
        chat.cancel()
        chat.clear_abort()
        assert spawner.spawned == []
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_is_alive_reflects_channel_state(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    chat = pool.register("chat", _entrypoint, _config_factory("chat", tmp_path))
    try:
        assert chat.is_alive is False
        await chat.call("warm", None)
        assert chat.is_alive is True
        spawner.spawned[0].alive = False
        assert chat.is_alive is False
    finally:
        await pool.shutdown()


# =====================================================================
# Health check pass-through.
# =====================================================================


@pytest.mark.asyncio
async def test_ping_lazy_spawns_and_records_call(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.ping(timeout=1.0)
        assert spawner.spawned[0].call_log == [("ping", None)]
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_ping_propagates_crash_and_drops_channel(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.call("warm", None)
        spawner.spawned[0].raise_on_call["ping"] = WorkerCrashError("embed")
        with pytest.raises(WorkerCrashError):
            await embed.ping(timeout=1.0)
        assert pool._channel_if_alive("embed") is None
    finally:
        await pool.shutdown()


# =====================================================================
# Worker errors that aren't crashes don't drop the channel.
# =====================================================================


@pytest.mark.asyncio
async def test_worker_error_does_not_drop_channel(tmp_path) -> None:
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.call("warm", None)
        spawner.spawned[0].raise_on_call["bad"] = WorkerError("ValueError", "nope", "")
        with pytest.raises(WorkerError):
            await embed.call("bad", None)
        # Channel stays up; only crashes drop it.
        assert pool._channel_if_alive("embed") is spawner.spawned[0]
    finally:
        await pool.shutdown()


# =====================================================================
# PoolRuntime: sync bridge that hosts the background asyncio loop.
# =====================================================================


def test_pool_runtime_runs_coroutine_to_completion() -> None:
    runtime = PoolRuntime()
    try:

        async def _double(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        assert runtime.run_sync(_double(21)) == 42
    finally:
        runtime.shutdown()


def test_pool_runtime_starts_lazily_on_first_call() -> None:
    runtime = PoolRuntime()
    try:
        assert runtime._thread is None

        async def _identity() -> str:
            return "ready"

        assert runtime.run_sync(_identity()) == "ready"
        assert runtime._thread is not None
        assert runtime._thread.is_alive()
    finally:
        runtime.shutdown()


def test_pool_runtime_propagates_exceptions() -> None:
    runtime = PoolRuntime()
    try:

        async def _boom() -> None:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            runtime.run_sync(_boom())
    finally:
        runtime.shutdown()


def test_pool_runtime_run_sync_after_shutdown_raises() -> None:
    runtime = PoolRuntime()
    runtime.shutdown()

    async def _noop() -> None:
        pass

    coro = _noop()
    try:
        with pytest.raises(PoolShutdownError):
            runtime.run_sync(coro)
    finally:
        coro.close()


def test_pool_runtime_start_after_shutdown_raises() -> None:
    runtime = PoolRuntime()
    runtime.shutdown()
    with pytest.raises(PoolShutdownError):
        runtime.start()


def test_pool_runtime_shutdown_is_idempotent() -> None:
    runtime = PoolRuntime()
    runtime.shutdown()
    runtime.shutdown()


def test_pool_runtime_start_is_idempotent() -> None:
    runtime = PoolRuntime()
    try:
        runtime.start()
        first = runtime._thread
        runtime.start()
        assert runtime._thread is first
    finally:
        runtime.shutdown()


def test_pool_runtime_run_sync_respects_timeout() -> None:
    runtime = PoolRuntime()
    try:

        async def _hang() -> None:
            await asyncio.sleep(60)

        from concurrent.futures import TimeoutError as FuturesTimeoutError

        with pytest.raises(FuturesTimeoutError):
            runtime.run_sync(_hang(), timeout=0.05)
    finally:
        runtime.shutdown()
