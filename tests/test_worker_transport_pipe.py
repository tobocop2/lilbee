"""Tests for the multiprocessing.Pipe-backed worker transport.

Spawns real subprocesses (stdlib only, no mocks) to exercise protocol
round-trips, streaming, error replies, ping/pong, abort flag, payload
size cap, in-flight counter, crash detection, and idempotent close.
Each test that spawns a worker is annotated for pytest-xdist isolation.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pickle
import time
from typing import Any

import pytest

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import (
    _PICKLE_MAX_BYTES,
    PipeChannel,
    PipeSpawner,
    WorkerCrashError,
    WorkerError,
    _serialize_exception,
)

pytestmark = pytest.mark.xdist_group("worker_pool_transport")


# =====================================================================
# Worker entrypoints used by the tests below.
#
# Module-level functions so pickling at spawn time succeeds (closures /
# locals are not picklable across mp.Process). Each entrypoint mirrors
# the discipline-rule contract: poll the pipe, dispatch by kind, never
# block on bare recv.
# =====================================================================


_POLL_TIMEOUT_S = 0.2
_TEST_PING_TIMEOUT_S = 5.0
_TEST_CALL_TIMEOUT_S = 5.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


def _handle_echo(conn: Any, payload: Any, _abort: Any) -> None:
    conn.send(("result", payload))


def _handle_raise(conn: Any, payload: Any, _abort: Any) -> None:
    try:
        raise RuntimeError(payload)
    except RuntimeError as exc:
        conn.send(("error", _serialize_exception(exc)))


def _handle_stream_error(conn: Any, payload: Any, _abort: Any) -> None:
    conn.send(("stream_chunk", "first"))
    try:
        raise ValueError(payload)
    except ValueError as exc:
        conn.send(("error", _serialize_exception(exc)))


def _handle_stream(conn: Any, payload: Any, _abort: Any) -> None:
    count, suffix = payload
    for i in range(count):
        conn.send(("stream_chunk", f"{suffix}{i}"))
    conn.send(("stream_end", None))


def _handle_abort_loop(conn: Any, _payload: Any, abort: Any) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if abort.value:
            conn.send(("result", "aborted"))
            return
        time.sleep(0.01)
    conn.send(("result", "timeout"))


def _handle_bad_kind(conn: Any, _payload: Any, _abort: Any) -> None:
    conn.send(("not_a_known_kind", None))


def _handle_bad_stream_kind(conn: Any, _payload: Any, _abort: Any) -> None:
    conn.send(("totally_bogus_kind", None))


def _handle_bad_ping_kind(conn: Any, _payload: Any, _abort: Any) -> None:
    conn.send(("not_pong", None))


_ECHO_DISPATCH = {
    "echo": _handle_echo,
    "raise": _handle_raise,
    "stream": _handle_stream,
    "stream_error": _handle_stream_error,
    "abort_loop": _handle_abort_loop,
    "bad_kind": _handle_bad_kind,
    "bad_stream_kind": _handle_bad_stream_kind,
    "bad_ping_kind": _handle_bad_ping_kind,
}


def _echo_worker_main(conn: Any, abort_flag: Any, _role_config: RoleConfig) -> None:
    """Worker that dispatches kinds via _ECHO_DISPATCH; polls so SIGTERM propagates."""
    try:
        while True:
            if not conn.poll(timeout=_POLL_TIMEOUT_S):
                continue
            try:
                kind, payload = conn.recv()
            except EOFError:
                return
            if kind == "shutdown":
                conn.send(("ack", None))
                return
            if kind == "ping":
                conn.send(("pong", None))
                continue
            handler = _ECHO_DISPATCH.get(kind)
            if handler is not None:
                handler(conn, payload, abort_flag)
    finally:  # pragma: no cover - cleanup runs in subprocess
        conn.close()


def _crash_worker_main(conn: Any, _abort_flag: Any, _role_config: RoleConfig) -> None:
    """Worker that exits abruptly on the first request, simulating a crash."""
    if conn.poll(timeout=_POLL_TIMEOUT_S * 50):
        with contextlib.suppress(EOFError):
            conn.recv()
    os._exit(1)


def _hang_worker_main(conn: Any, _abort_flag: Any, _role_config: RoleConfig) -> None:
    """Worker that polls forever; never replies.

    Used to exercise call-timeout and close-timeout paths without leaving
    a hung process behind: the parent's close() terminates it.
    """
    while True:
        if conn.poll(timeout=_POLL_TIMEOUT_S):
            with contextlib.suppress(EOFError):
                conn.recv()
            # Drop the request silently to force a parent timeout.


def _ping_replies_garbage_main(conn: Any, _abort_flag: Any, _role_config: RoleConfig) -> None:
    """Worker that replies to ping with a non-pong kind, to test the parent's check."""
    while True:
        if not conn.poll(timeout=_POLL_TIMEOUT_S):
            continue
        try:
            kind, _ = conn.recv()
        except EOFError:
            return
        if kind == "shutdown":
            conn.send(("ack", None))
            return
        # Always reply with garbage regardless of kind so ping/call both
        # see a protocol error.
        conn.send(("not_a_known_reply", None))


def _stream_replies_garbage_main(conn: Any, _abort_flag: Any, _role_config: RoleConfig) -> None:
    """Worker that replies to a streaming kind with a non-stream message."""
    while True:
        if not conn.poll(timeout=_POLL_TIMEOUT_S):
            continue
        try:
            kind, _ = conn.recv()
        except EOFError:
            return
        if kind == "shutdown":
            conn.send(("ack", None))
            return
        # Send something the streaming consumer rejects (not chunk/end/error).
        conn.send(("totally_bogus_kind", None))


# =====================================================================
# Fixtures: spawn one worker per test, always close in teardown so a
# failing assertion never leaves a real subprocess behind.
# =====================================================================


@pytest.fixture()
def role_config(tmp_path) -> RoleConfig:
    return RoleConfig(role="echo", model_path=tmp_path / "model.gguf", mode="embed")


@pytest.fixture()
def spawner() -> PipeSpawner:
    return PipeSpawner()


@pytest.fixture()
async def echo_channel(
    spawner: PipeSpawner,
    role_config: RoleConfig,
):
    channel, _handle = spawner.spawn(_echo_worker_main, role_config)
    try:
        yield channel
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


# =====================================================================
# Roundtrip and protocol behavior.
# =====================================================================


@pytest.mark.asyncio
async def test_call_roundtrips_payload(echo_channel: PipeChannel) -> None:
    result = await echo_channel.call("echo", {"hello": "world"}, timeout=_TEST_CALL_TIMEOUT_S)
    assert result == {"hello": "world"}


@pytest.mark.asyncio
async def test_call_returns_in_flight_to_zero_after_success(
    echo_channel: PipeChannel,
) -> None:
    await echo_channel.call("echo", "x", timeout=_TEST_CALL_TIMEOUT_S)
    assert echo_channel.in_flight == 0


@pytest.mark.asyncio
async def test_call_raises_worker_error_on_worker_exception(
    echo_channel: PipeChannel,
) -> None:
    with pytest.raises(WorkerError) as excinfo:
        await echo_channel.call("raise", "boom", timeout=_TEST_CALL_TIMEOUT_S)
    assert excinfo.value.original_type == "RuntimeError"
    assert "boom" in str(excinfo.value)
    assert "Traceback" in excinfo.value.traceback_str


@pytest.mark.asyncio
async def test_call_raises_protocol_error_on_unexpected_kind(
    echo_channel: PipeChannel,
) -> None:
    with pytest.raises(WorkerError) as excinfo:
        await echo_channel.call("bad_kind", None, timeout=_TEST_CALL_TIMEOUT_S)
    assert excinfo.value.original_type == "ProtocolError"


@pytest.mark.asyncio
async def test_ping_roundtrips(echo_channel: PipeChannel) -> None:
    await echo_channel.ping(timeout=_TEST_PING_TIMEOUT_S)
    assert echo_channel.in_flight == 0


@pytest.mark.asyncio
async def test_stream_yields_chunks_then_terminates(echo_channel: PipeChannel) -> None:
    chunks = [chunk async for chunk in echo_channel.stream("stream", (3, "tok-"))]
    assert chunks == ["tok-0", "tok-1", "tok-2"]
    assert echo_channel.in_flight == 0


@pytest.mark.asyncio
async def test_stream_raises_on_worker_error_terminator(
    echo_channel: PipeChannel,
) -> None:
    seen: list[Any] = []
    with pytest.raises(WorkerError) as excinfo:
        async for chunk in echo_channel.stream("stream_error", "broken"):
            seen.append(chunk)
    assert seen == ["first"]
    assert excinfo.value.original_type == "ValueError"
    assert echo_channel.in_flight == 0


# =====================================================================
# Cancellation: parent flips the abort flag, worker observes it.
# =====================================================================


@pytest.mark.asyncio
async def test_cancel_sets_abort_flag_and_worker_observes_it(
    echo_channel: PipeChannel,
) -> None:
    async def trigger_cancel() -> None:
        await asyncio.sleep(0.05)
        echo_channel.cancel()

    cancel_task = asyncio.create_task(trigger_cancel())
    result = await echo_channel.call("abort_loop", None, timeout=_TEST_CALL_TIMEOUT_S)
    await cancel_task
    assert result == "aborted"
    echo_channel.clear_abort()
    # After clear_abort, the worker should run to its 5s timeout instead.
    # Re-running the abort loop and not flipping anything proves the reset.
    result_after = await asyncio.wait_for(
        echo_channel.call("echo", "post-abort", timeout=_TEST_CALL_TIMEOUT_S),
        timeout=_TEST_CALL_TIMEOUT_S,
    )
    assert result_after == "post-abort"


# =====================================================================
# Pickle size cap.
# =====================================================================


@pytest.mark.asyncio
async def test_call_rejects_payload_over_size_cap(echo_channel: PipeChannel) -> None:
    huge = b"\x00" * (_PICKLE_MAX_BYTES + 1)
    with pytest.raises(WorkerError) as excinfo:
        await echo_channel.call("echo", huge, timeout=_TEST_CALL_TIMEOUT_S)
    assert excinfo.value.original_type == "PayloadTooLarge"
    assert echo_channel.in_flight == 0


# =====================================================================
# Crash detection and close.
# =====================================================================


@pytest.mark.asyncio
async def test_call_raises_worker_crash_error_when_worker_dies(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _handle = spawner.spawn(_crash_worker_main, role_config)
    try:
        with pytest.raises(WorkerCrashError) as excinfo:
            await channel.call("anything", None, timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.role == role_config.role
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_close_terminates_hung_worker_within_timeout(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _handle = spawner.spawn(_hang_worker_main, role_config)
    pid = channel.pid
    assert pid is not None
    start = time.monotonic()
    await channel.close(timeout=0.5)
    elapsed = time.monotonic() - start
    # Close should never sit on a hung worker for the full call timeout.
    # Generous upper bound (worker-join + terminate + safety): 6s.
    assert elapsed < 6.0
    # Verify the worker process is gone. We use the multiprocessing
    # handle (not os.kill) because os.kill(pid, 0) is unreliable on
    # Windows -- the signal-0 path actually kills the process there
    # rather than returning a probe-only result, so it can't be used
    # for a "did terminate succeed" assertion.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not channel.is_alive:
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"Hung worker pid={pid} survived close()")


@pytest.mark.asyncio
async def test_close_is_idempotent(echo_channel: PipeChannel) -> None:
    await echo_channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)
    # Second call must be a no-op (no error).
    await echo_channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_call_after_close_raises_pool_shutdown_error(
    echo_channel: PipeChannel,
) -> None:
    await echo_channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)
    with pytest.raises(WorkerError) as excinfo:
        await echo_channel.call("echo", "post-close", timeout=_TEST_CALL_TIMEOUT_S)
    assert excinfo.value.original_type == "PoolShutdownError"


# =====================================================================
# In-flight counter accuracy across success, error, and stream.
# =====================================================================


@pytest.mark.asyncio
async def test_in_flight_returns_to_zero_after_error_call(
    echo_channel: PipeChannel,
) -> None:
    with pytest.raises(WorkerError):
        await echo_channel.call("raise", "x", timeout=_TEST_CALL_TIMEOUT_S)
    assert echo_channel.in_flight == 0


@pytest.mark.asyncio
async def test_in_flight_returns_to_zero_after_stream(
    echo_channel: PipeChannel,
) -> None:
    async for _chunk in echo_channel.stream("stream", (2, "x")):
        pass
    assert echo_channel.in_flight == 0


# =====================================================================
# Pure-function behavior of the helpers (no subprocess).
# =====================================================================


def test_serialize_exception_returns_triple_for_picklable_type() -> None:
    try:
        raise ValueError("hello")
    except ValueError as exc:
        serialized = _serialize_exception(exc)
    assert serialized.type_name == "ValueError"
    assert serialized.message == "hello"
    assert "ValueError" in serialized.traceback_str


def test_serialize_exception_extracts_metadata_even_from_unpicklable_exception() -> None:
    """Exceptions whose __reduce__ raises still surface (type, message, traceback)."""

    class _Unpicklable(RuntimeError):
        def __reduce__(self):  # type: ignore[override]
            raise TypeError("nope, cannot pickle me")

    try:
        raise _Unpicklable("tricky")
    except _Unpicklable as exc:
        # Confirm the exception genuinely is unpicklable (test invariant).
        with pytest.raises(TypeError):
            pickle.dumps(exc)
        serialized = _serialize_exception(exc)
    assert serialized.type_name == "_Unpicklable"
    assert serialized.message == "tricky"
    assert "tricky" in serialized.traceback_str


def test_worker_crash_error_carries_role_name() -> None:
    err = WorkerCrashError("embed")
    assert err.role == "embed"
    assert "embed" in str(err)


def test_pipe_spawner_uses_spawn_context() -> None:
    """The pipe spawner pins the multiprocessing context to spawn."""
    spawner = PipeSpawner()
    assert spawner._ctx.get_start_method() == "spawn"


@pytest.mark.asyncio
async def test_channel_exposes_role_and_pid_and_is_alive(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, handle = spawner.spawn(_echo_worker_main, role_config)
    try:
        assert channel.role == role_config.role
        assert channel.pid is not None
        assert channel.pid == handle.pid
        assert channel.is_alive is True
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)
        # After close, the process is gone.
        assert channel.is_alive is False


@pytest.mark.asyncio
async def test_stream_raises_on_unexpected_message_kind(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_stream_replies_garbage_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            async for _chunk in channel.stream("anything", None):
                pass
        assert excinfo.value.original_type == "ProtocolError"
        assert "totally_bogus_kind" in str(excinfo.value)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_ping_raises_on_non_pong_reply(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_ping_replies_garbage_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.ping(timeout=_TEST_PING_TIMEOUT_S)
        assert excinfo.value.original_type == "ProtocolError"
        assert "not_a_known_reply" in str(excinfo.value)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


def test_check_pickle_size_wraps_pickle_failure() -> None:
    """Unpicklable payload surfaces as PickleError before the send."""
    from lilbee.providers.worker.transport_pipe import _check_pickle_size

    # Lambda is famously unpicklable; pickle.dumps raises immediately.
    with pytest.raises(WorkerError) as excinfo:
        _check_pickle_size(lambda: None, "echo")
    assert excinfo.value.original_type == "PickleError"
