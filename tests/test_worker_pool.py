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


@pytest.mark.asyncio
async def test_ping_runs_even_when_data_pipe_has_in_flight_work(tmp_path) -> None:
    """Pings travel on the dedicated health pipe so they fire regardless of
    data-pipe activity. With separate pipes, no orphan-pong defense is
    needed in the accessor."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    embed = pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await embed.call("warm", None)
        channel = spawner.spawned[0]
        channel.in_flight_count = 1
        await embed.ping(timeout=1.0)
        # The channel observed the ping despite in_flight > 0 on the data pipe.
        assert ("ping", None) in channel.call_log
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


def test_pool_runtime_submit_lazy_starts_thread() -> None:
    """submit() must spin up the runtime thread on first call, like run_sync."""
    runtime = PoolRuntime()
    try:

        async def _noop() -> str:
            return "ok"

        future = runtime.submit(_noop())
        assert future.result(timeout=2.0) == "ok"
        assert runtime._thread is not None
    finally:
        runtime.shutdown()


def test_pool_runtime_submit_after_shutdown_raises() -> None:
    """submit() on a shut-down runtime must surface PoolShutdownError."""
    runtime = PoolRuntime()
    runtime.start()
    runtime.shutdown()

    async def _noop() -> None:
        return None

    with pytest.raises(PoolShutdownError):
        runtime.submit(_noop())


def test_pool_runtime_drain_failure_still_closes_loop(monkeypatch, caplog) -> None:
    """If task drain raises during shutdown the loop still closes and
    the exception is logged, not propagated. Otherwise a crashed drain
    would leak the asyncio loop on Ctrl-C teardown."""
    import logging

    from lilbee.providers.worker import pool as pool_module

    original_gather = asyncio.gather

    def _raising_gather(*args, **kwargs):
        raise RuntimeError("simulated drain failure")

    monkeypatch.setattr(pool_module.asyncio, "gather", _raising_gather)
    runtime = PoolRuntime()
    runtime.start()

    async def _spawn_pending() -> None:
        await asyncio.sleep(60)

    runtime.submit(_spawn_pending())
    with caplog.at_level(logging.ERROR, logger=pool_module.__name__):
        runtime.shutdown()
    assert any("Pool runtime loop drain failed" in r.message for r in caplog.records)
    monkeypatch.setattr(pool_module.asyncio, "gather", original_gather)


# =====================================================================
# Lifecycle features: idle reap, restart bookkeeping, health pings.
# =====================================================================


@pytest.mark.asyncio
async def test_reap_idle_no_op_when_max_idle_zero(tmp_path) -> None:
    """``max_idle_s == 0`` disables idle reaping entirely."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.0)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        # Even with last_used in the past, reap is a no-op.
        reaped = await pool.reap_idle()
        assert reaped == ()
        assert spawner.spawned[0].closed is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_closes_idle_role_with_zero_in_flight(tmp_path) -> None:
    """Roles past ``max_idle_s`` with no in-flight work get closed."""
    import time as _time

    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.01)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        channel = spawner.spawned[0]
        # Backdate last_used so the reaper considers it stale.
        pool._roles["embed"].last_used = _time.monotonic() - 10.0
        reaped = await pool.reap_idle()
        assert reaped == ("embed",)
        assert channel.closed is True
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_skips_in_flight_role(tmp_path) -> None:
    """Reap leaves a role alone while it still has in-flight work."""
    import time as _time

    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.01)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        channel = spawner.spawned[0]
        channel.in_flight_count = 1
        pool._roles["embed"].last_used = _time.monotonic() - 10.0
        reaped = await pool.reap_idle()
        assert reaped == ()
        assert channel.closed is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_role_marks_degraded_after_restart_budget(tmp_path, monkeypatch) -> None:
    """A role exceeding _RESTART_BUDGET within the window is marked degraded."""
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_BUDGET", 2)
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_WINDOW_S", 10.0)
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        # Three crashes in the window: degraded after the third.
        await pool._on_crash("embed")
        await pool._on_crash("embed")
        assert pool.is_degraded("embed") is False
        await pool._on_crash("embed")
        assert pool.is_degraded("embed") is True

        from lilbee.providers.worker.pool import RoleDegradedError

        with pytest.raises(RoleDegradedError, match="embed"):
            await pool.accessor("embed").call("echo", "x")
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reset_role_failures_clears_degraded_state(tmp_path, monkeypatch) -> None:
    """Manual reset re-enables a degraded role."""
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_BUDGET", 1)
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_WINDOW_S", 10.0)
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool._on_crash("embed")
        await pool._on_crash("embed")
        assert pool.is_degraded("embed") is True
        pool.reset_role_failures("embed")
        assert pool.is_degraded("embed") is False
        # Next call now spawns a fresh worker.
        result = await pool.accessor("embed").call("echo", "post-reset")
        assert result == "post-reset"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reset_role_failures_silent_for_unregistered_role(tmp_path) -> None:
    pool = WorkerPool(spawner=FakeSpawner())
    try:
        # Must not raise.
        pool.reset_role_failures("missing")
        assert pool.is_degraded("missing") is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_crash_history_evicts_stale_entries(tmp_path, monkeypatch) -> None:
    """Crashes outside the window do not count toward the budget."""
    import time as _time

    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_BUDGET", 2)
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_WINDOW_S", 0.05)
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool._on_crash("embed")
        await pool._on_crash("embed")
        # Wait past the window so the next crash is the only one in the budget.
        _time.sleep(0.1)
        await pool._on_crash("embed")
        assert pool.is_degraded("embed") is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_ping_role_round_trips(tmp_path) -> None:
    """``ping_role`` lazy-spawns and round-trips one ping/pong."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.ping_role("embed", timeout=5.0)
        channel = spawner.spawned[0]
        assert ("ping", None) in channel.call_log
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_call_stamps_last_used(tmp_path) -> None:
    """A successful call updates the role's last_used timestamp."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=10.0)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        before = pool._roles["embed"].last_used
        await pool.accessor("embed").call("echo", "x")
        after = pool._roles["embed"].last_used
        assert after > before
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_skips_dead_channel(tmp_path) -> None:
    """Reap walks past a registered role whose channel is None or not alive."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.01)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        # Channel never spawned: reap must walk past without crashing.
        reaped = await pool.reap_idle()
        assert reaped == ()

        await pool.accessor("embed").call("echo", "x")
        channel = spawner.spawned[0]
        # Mark the channel dead; reap must skip it.
        channel.alive = False
        reaped = await pool.reap_idle()
        assert reaped == ()
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_skips_role_with_zero_last_used(tmp_path) -> None:
    """A registered role that never serviced a call has last_used=0.0; reap leaves it."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.01)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        # Spawn the channel directly without going through accessor.call (which stamps).
        await pool._ensure_channel("embed")
        pool._roles["embed"].last_used = 0.0
        reaped = await pool.reap_idle()
        assert reaped == ()
        assert spawner.spawned[0].closed is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_skips_recently_used_role(tmp_path) -> None:
    """A role used inside the max_idle window is left alone."""
    import time as _time

    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=60.0)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        # last_used is now-ish; max_idle=60s; reap must skip.
        pool._roles["embed"].last_used = _time.monotonic()
        reaped = await pool.reap_idle()
        assert reaped == ()
        assert spawner.spawned[0].closed is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_skips_when_in_flight_appears_inside_lock(tmp_path) -> None:
    """A racy in_flight bump after the outer check still aborts the reap.

    The reap re-checks ``in_flight`` inside the spawn_lock so a coroutine
    that started a call between the outer pass and the lock acquisition
    does not get its channel yanked.
    """
    import time as _time

    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.01)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        channel = spawner.spawned[0]
        registration = pool._roles["embed"]
        registration.last_used = _time.monotonic() - 10.0

        # Pre-acquire the spawn_lock from this coroutine. Start reap as a task;
        # it will pass the outer check (in_flight=0, last_used old) and queue
        # on the lock. While it waits, bump in_flight, then release. The reap
        # acquires the lock, inner check sees in_flight > 0, takes line 440.
        await registration.spawn_lock.acquire()
        try:
            reap_task = asyncio.create_task(pool.reap_idle())
            await asyncio.sleep(0)  # let reap reach the lock acquisition
            channel.in_flight_count = 1
        finally:
            registration.spawn_lock.release()
        reaped = await reap_task
        assert reaped == ()
        assert channel.closed is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_reap_idle_skips_when_last_used_refreshes_inside_lock(tmp_path) -> None:
    """A racy last_used refresh after the outer check still aborts the reap.

    The reap re-checks ``last_used`` inside the spawn_lock so a coroutine
    that just stamped the role does not get its channel yanked.
    """
    import time as _time

    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner, max_idle_s=0.01)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        channel = spawner.spawned[0]
        registration = pool._roles["embed"]
        registration.last_used = _time.monotonic() - 10.0

        # Same pattern: pre-acquire the lock, queue the reap, refresh last_used
        # mid-wait, then release. Inner check sees the fresh stamp and continues.
        await registration.spawn_lock.acquire()
        try:
            reap_task = asyncio.create_task(pool.reap_idle())
            await asyncio.sleep(0)
            registration.last_used = _time.monotonic()
        finally:
            registration.spawn_lock.release()
        reaped = await reap_task
        assert reaped == ()
        assert channel.closed is False
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_release_unregistered_role_is_noop(tmp_path) -> None:
    """``release(role)`` on an unregistered role returns silently."""
    pool = WorkerPool(spawner=FakeSpawner())
    try:
        # Must not raise; release for a never-registered role is a no-op.
        await pool.release("nope")
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_release_drops_registration_and_closes_live_channel(tmp_path) -> None:
    """``release`` closes the live channel and forgets the role."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool.accessor("embed").call("echo", "x")
        channel = spawner.spawned[0]
        await pool.release("embed")
        assert channel.closed is True
        # Re-register works after release.
        pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_ensure_channel_raises_when_role_degraded_after_outer_check(
    tmp_path, monkeypatch
) -> None:
    """The inner cooldown check inside spawn_lock catches the race where the role
    was tripped into cooldown between the pre-lock check and lock acquisition."""
    import time as _time

    from lilbee.providers.worker.pool import RoleDegradedError

    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_BUDGET", 1)
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_WINDOW_S", 10.0)
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        registration = pool._roles["embed"]

        # Acquire the spawn_lock so _ensure_channel must wait. Inside our hold,
        # arm a cooldown; release the lock; the queued _ensure_channel
        # re-checks the cooldown inside the lock.
        await registration.spawn_lock.acquire()
        try:
            ensure_task = asyncio.create_task(pool._ensure_channel("embed"))
            # Yield so the task gets to the spawn_lock.acquire() await point.
            await asyncio.sleep(0)
            registration.degraded_until = _time.monotonic() + 60.0
        finally:
            registration.spawn_lock.release()
        with pytest.raises(RoleDegradedError, match="embed"):
            await ensure_task
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_cooldown_expiry_allows_one_retry(tmp_path, monkeypatch) -> None:
    """After the cooldown deadline elapses the next call gets a fresh spawn attempt."""
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_BUDGET", 1)
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_WINDOW_S", 10.0)
    monkeypatch.setattr("lilbee.providers.worker.pool._DEGRADED_COOLDOWN_S", 0.05)
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        # Trip the breaker.
        await pool._on_crash("embed")
        await pool._on_crash("embed")
        assert pool.is_degraded("embed") is True
        # Wait past the cooldown; the next call should succeed (half-open
        # attempt clears the breaker).
        import time as _time

        _time.sleep(0.1)
        assert pool.is_degraded("embed") is False
        result = await pool.accessor("embed").call("echo", "after-cooldown")
        assert result == "after-cooldown"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_role_degraded_error_advertises_retry_window(tmp_path, monkeypatch) -> None:
    """The error message surfaces how long until auto-retry, not 'restart lilbee'."""
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_BUDGET", 1)
    monkeypatch.setattr("lilbee.providers.worker.pool._RESTART_WINDOW_S", 10.0)
    monkeypatch.setattr("lilbee.providers.worker.pool._DEGRADED_COOLDOWN_S", 30.0)
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))
    try:
        await pool._on_crash("embed")
        await pool._on_crash("embed")
        from lilbee.providers.worker.pool import RoleDegradedError

        with pytest.raises(RoleDegradedError) as excinfo:
            await pool.accessor("embed").call("echo", "x")
        assert "Retry in" in str(excinfo.value)
        # Should NOT instruct the user to restart; the breaker auto-recovers.
        assert "Restart lilbee" not in str(excinfo.value)
    finally:
        await pool.shutdown()


# Lifecycle: shutdown_pool_runtime helper.


def test_shutdown_pool_runtime_warns_and_forces_stop_when_drain_raises(caplog) -> None:
    """``shutdown_pool_runtime`` warns and force-stops when the pool drain raises."""
    from lilbee.providers.worker.health_ticker import HealthTickerHandle
    from lilbee.providers.worker.pool import shutdown_pool_runtime

    runtime_calls: list[str] = []

    class _FailingRuntime:
        def run_sync(self, coro, *, timeout):
            runtime_calls.append("run_sync")
            # Close the coro so pytest does not warn about "never awaited".
            coro.close()
            raise RuntimeError("simulated drain failure")

        def shutdown(self, *, timeout=5.0):
            runtime_calls.append("shutdown")

    class _FakePool:
        async def shutdown(self):
            return None

    with caplog.at_level("WARNING", logger="lilbee.providers.worker.pool"):
        shutdown_pool_runtime(_FakePool(), _FailingRuntime(), HealthTickerHandle())
    assert runtime_calls == ["run_sync", "shutdown"]
    assert any("forcing runtime stop" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Spawn lifecycle listeners (cold-start UX hook).


@pytest.mark.asyncio
async def test_spawn_listeners_fire_around_first_call(tmp_path) -> None:
    """``add_listener`` callbacks fire spawning-then-spawned around the spawn."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))

    events: list[tuple[str, str]] = []
    pool.add_listener(
        on_spawning=lambda role: events.append(("spawning", role)),
        on_spawned=lambda role: events.append(("spawned", role)),
    )
    try:
        accessor = pool.accessor("embed")
        await accessor.call("embed", ["hi"], timeout=5.0)
        assert events == [("spawning", "embed"), ("spawned", "embed")]
    finally:
        await pool.shutdown(timeout=2.0)


@pytest.mark.asyncio
async def test_spawn_listeners_fire_only_on_actual_spawn(tmp_path) -> None:
    """The second call to a live worker must not re-fire either listener."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))

    events: list[tuple[str, str]] = []
    pool.add_listener(
        on_spawning=lambda role: events.append(("spawning", role)),
        on_spawned=lambda role: events.append(("spawned", role)),
    )
    try:
        accessor = pool.accessor("embed")
        await accessor.call("embed", ["hi"], timeout=5.0)
        await accessor.call("embed", ["bye"], timeout=5.0)
        # One spawn, one pair of events.
        assert events == [("spawning", "embed"), ("spawned", "embed")]
    finally:
        await pool.shutdown(timeout=2.0)


@pytest.mark.asyncio
async def test_spawn_listener_exception_does_not_break_pool(tmp_path, caplog) -> None:
    """A misbehaving listener logs and the spawn still completes."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.register("embed", _entrypoint, _config_factory("embed", tmp_path))

    def _boom(role: str) -> None:
        raise RuntimeError(f"listener exploded for {role}")

    pool.add_listener(on_spawning=_boom, on_spawned=_boom)
    try:
        with caplog.at_level("ERROR", logger="lilbee.providers.worker.pool"):
            accessor = pool.accessor("embed")
            await accessor.call("embed", ["hi"], timeout=5.0)
        # Pool kept the channel alive even though both listeners raised.
        assert spawner.spawned, "spawn must have completed"
        assert any("listener for role=embed raised" in rec.message for rec in caplog.records)
    finally:
        await pool.shutdown(timeout=2.0)


def test_add_listener_accepts_either_kwarg_independently() -> None:
    """``add_listener`` allows one or both callbacks; defaults are empty lists."""
    spawner = FakeSpawner()
    pool = WorkerPool(spawner=spawner)
    pool.add_listener(on_spawning=lambda r: None)
    pool.add_listener(on_spawned=lambda r: None)
    assert len(pool._on_role_spawning) == 1
    assert len(pool._on_role_spawned) == 1
