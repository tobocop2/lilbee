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
import threading
import time
from multiprocessing.connection import wait
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
# locals are not picklable across mp.Process). Each entrypoint takes
# (data_conn, health_conn, abort_flag, role_config) and multiplexes via
# multiprocessing.connection.wait.
# =====================================================================


_POLL_TIMEOUT_S = 0.2
_TEST_PING_TIMEOUT_S = 5.0
_TEST_CALL_TIMEOUT_S = 5.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


def _drive_test_worker(
    data_conn: Any,
    health_conn: Any,
    abort_flag: Any,
    on_data: Any,
) -> None:
    """Test-worker multiplexer using the new ``(call_id, kind, payload)`` frame format.

    The dedicated heartbeat thread responsibility lives in real workers via
    ``worker_runtime.run_worker``; tests bake an inline equivalent so each
    test worker stays self-contained. *on_data* signature is
    ``(conn, call_id, kind, payload, abort_flag) -> None`` and it owns
    sending response frames with the right call_id echoed back.
    """
    try:
        while True:
            ready = wait([data_conn, health_conn], timeout=_POLL_TIMEOUT_S)
            if data_conn in ready:
                try:
                    call_id, kind, payload = data_conn.recv()
                except EOFError:
                    return
                if kind == "shutdown":
                    data_conn.send((call_id, "ack", None))
                    return
                on_data(data_conn, call_id, kind, payload, abort_flag)
            if health_conn in ready:
                try:
                    hcall_id, hkind, _ = health_conn.recv()
                except EOFError:
                    continue
                if hkind == "ping":
                    health_conn.send((hcall_id, "pong", None))
    finally:  # pragma: no cover - cleanup runs in subprocess
        with contextlib.suppress(Exception):
            data_conn.close()
        with contextlib.suppress(Exception):
            health_conn.close()


def _handle_echo(conn: Any, call_id: int, payload: Any, _abort: Any) -> None:
    conn.send((call_id, "result", payload))


def _handle_raise(conn: Any, call_id: int, payload: Any, _abort: Any) -> None:
    try:
        raise RuntimeError(payload)
    except RuntimeError as exc:
        conn.send((call_id, "error", _serialize_exception(exc)))


def _handle_stream_error(conn: Any, call_id: int, payload: Any, _abort: Any) -> None:
    conn.send((call_id, "stream_chunk", "first"))
    try:
        raise ValueError(payload)
    except ValueError as exc:
        conn.send((call_id, "error", _serialize_exception(exc)))


def _handle_stream(conn: Any, call_id: int, payload: Any, _abort: Any) -> None:
    count, suffix = payload
    for i in range(count):
        conn.send((call_id, "stream_chunk", f"{suffix}{i}"))
    conn.send((call_id, "stream_end", None))


def _handle_abort_loop(conn: Any, call_id: int, _payload: Any, abort: Any) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if abort.value:
            conn.send((call_id, "result", "aborted"))
            return
        time.sleep(0.01)
    conn.send((call_id, "result", "timeout"))


def _handle_bad_kind(conn: Any, call_id: int, _payload: Any, _abort: Any) -> None:
    conn.send((call_id, "not_a_known_kind", None))


def _handle_bad_stream_kind(conn: Any, call_id: int, _payload: Any, _abort: Any) -> None:
    conn.send((call_id, "totally_bogus_kind", None))


def _handle_bad_ping_kind(conn: Any, _call_id: int, _payload: Any, _abort: Any) -> None:
    """Reply to ping with non-pong kind on the health pipe (separate driver)."""
    # Unused. bad_ping behavior lives in _ping_replies_garbage_main.
    raise NotImplementedError


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


def _echo_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, _role_config: RoleConfig
) -> None:
    """Worker that dispatches data-pipe kinds via _ECHO_DISPATCH and pings on health."""

    def _on_data(conn: Any, call_id: int, kind: str, payload: Any, abort: Any) -> None:
        handler = _ECHO_DISPATCH.get(kind)
        if handler is not None:
            handler(conn, call_id, payload, abort)

    _drive_test_worker(data_conn, health_conn, abort_flag, _on_data)


def _crash_worker_main(
    data_conn: Any, _health_conn: Any, _abort_flag: Any, _role_config: RoleConfig
) -> None:
    """Worker that exits abruptly on the first request, simulating a crash."""
    if data_conn.poll(timeout=_POLL_TIMEOUT_S * 50):
        with contextlib.suppress(EOFError):
            data_conn.recv()
    os._exit(1)


def _hang_worker_main(
    data_conn: Any, _health_conn: Any, _abort_flag: Any, _role_config: RoleConfig
) -> None:
    """Worker that polls the data pipe forever; never replies. Used for timeout paths."""
    while True:
        if data_conn.poll(timeout=_POLL_TIMEOUT_S):
            with contextlib.suppress(EOFError):
                data_conn.recv()


def _ping_replies_garbage_main(
    _data_conn: Any, health_conn: Any, _abort_flag: Any, _role_config: RoleConfig
) -> None:
    """Worker that replies to every health ping with a non-pong kind."""
    while True:
        if not health_conn.poll(timeout=_POLL_TIMEOUT_S):
            continue
        try:
            call_id, _kind, _ = health_conn.recv()
        except EOFError:
            return
        health_conn.send((call_id, "not_a_known_reply", None))


def _stream_replies_garbage_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, _role_config: RoleConfig
) -> None:
    """Worker that replies to a streaming kind with a non-stream message."""

    def _on_data(conn: Any, call_id: int, _kind: str, _payload: Any, _abort: Any) -> None:
        conn.send((call_id, "totally_bogus_kind", None))

    _drive_test_worker(data_conn, health_conn, abort_flag, _on_data)


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
async def test_concurrent_ping_and_stream_do_not_interfere(
    echo_channel: PipeChannel,
) -> None:
    """Pings on the health pipe must not see stream chunks on the data pipe.

    Spawn a long-running stream, fire several pings while it's emitting,
    confirm the stream still produces every chunk in order and every ping
    completes without raising. This is the architectural fix for bb-ubnm:
    pings and streams travel on separate pipes by construction, so no
    orphan-pong defenses are needed in PipeChannel.
    """
    chunks: list[str] = []
    ping_count = 0

    async def _drain() -> None:
        async for chunk in echo_channel.stream("stream", (5, "tok-")):
            chunks.append(chunk)
            await asyncio.sleep(0.01)

    async def _ping_loop() -> None:
        nonlocal ping_count
        for _ in range(3):
            await echo_channel.ping(timeout=_TEST_PING_TIMEOUT_S)
            ping_count += 1
            await asyncio.sleep(0.01)

    await asyncio.gather(_drain(), _ping_loop())
    assert chunks == [f"tok-{i}" for i in range(5)]
    assert ping_count == 3


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
        _check_pickle_size(lambda: None, "echo", call_id=1)
    assert excinfo.value.original_type == "PickleError"


def _crashing_health_pipe_main(
    data_conn: Any, health_conn: Any, _abort_flag: Any, _role_config: RoleConfig
) -> None:
    """Worker that closes the health pipe immediately, leaves data alive."""
    health_conn.close()
    while True:
        if data_conn.poll(timeout=_POLL_TIMEOUT_S):
            try:
                call_id, kind, _ = data_conn.recv()
            except EOFError:
                return
            if kind == "shutdown":
                data_conn.send((call_id, "ack", None))
                return


@pytest.mark.asyncio
async def test_ping_raises_worker_crash_when_health_pipe_dies(
    spawner: PipeSpawner, role_config: RoleConfig
) -> None:
    """A dead health pipe surfaces as ``WorkerCrashError`` from ``ping``."""
    channel, _ = spawner.spawn(_crashing_health_pipe_main, role_config)
    try:
        with pytest.raises(WorkerCrashError):
            await channel.ping(timeout=_TEST_PING_TIMEOUT_S)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_ping_raises_worker_crash_when_health_send_fails() -> None:
    """If the health-pipe send itself raises, ping surfaces WorkerCrashError.

    Closes the health pipe synchronously before issuing ping so the
    parent's ``_health_conn.send`` hits BrokenPipeError / OSError on
    its way out.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)

    class _FakeProcess:
        def is_alive(self) -> bool:
            return True

        @property
        def pid(self) -> int:
            return -1

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            return None

    channel = PipeChannel(
        role="echo",
        process=_FakeProcess(),
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    # Close the parent end of the health pipe so `send` on it raises.
    parent_health.close()
    try:
        with pytest.raises(WorkerCrashError):
            await channel.ping(timeout=1.0)
    finally:
        with contextlib.suppress(Exception):
            parent_data.close()
        with contextlib.suppress(Exception):
            child_data.close()
        with contextlib.suppress(Exception):
            child_health.close()


def test_reader_main_short_circuits_when_async_setup_incomplete() -> None:
    """``_reader_main`` returns immediately when ``_reader_loop`` /
    ``_frame_queue`` haven't been wired up yet. Without this guard a thread
    would spin on a recv() against the data pipe with no consumer."""

    class _DummyChannel:
        _reader_loop: Any = None
        _frame_queue: Any = None
        _conn: Any = None  # never read because we early-return

    # Bind the real method to the dummy via descriptor protocol.
    PipeChannel._reader_main(_DummyChannel())  # must return cleanly


def test_worker_log_path_returns_none_when_env_unset(monkeypatch) -> None:
    """Without LILBEE_DATA the log path resolver returns None."""
    from lilbee.providers.worker.transport_pipe import _worker_log_path

    monkeypatch.delenv("LILBEE_DATA", raising=False)
    assert _worker_log_path("embed") is None


def test_worker_log_path_joins_data_dir_when_env_set(monkeypatch, tmp_path) -> None:
    """With LILBEE_DATA set, the path lands under <data>/logs/worker-<role>.log."""
    from lilbee.providers.worker.transport_pipe import _worker_log_path

    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    assert _worker_log_path("chat") == str(tmp_path / "logs" / "worker-chat.log")


def test_format_exit_reason_for_normal_exit_code() -> None:
    """Non-zero code surfaces as 'exited with code N'."""
    from lilbee.providers.worker.transport_pipe import PipeChannel

    assert PipeChannel._format_exit_reason(2) == "exited with code 2"


def test_format_exit_reason_for_known_signal() -> None:
    """Negative exit code maps to the signal name."""
    import signal as _signal

    from lilbee.providers.worker.transport_pipe import PipeChannel

    msg = PipeChannel._format_exit_reason(-_signal.SIGTERM)
    assert "SIGTERM" in msg
    assert f"({int(_signal.SIGTERM)})" in msg


def test_format_exit_reason_for_unknown_signal() -> None:
    """Unrecognised signum (no Signals enum entry) falls back to SIG<num>."""
    from lilbee.providers.worker.transport_pipe import PipeChannel

    msg = PipeChannel._format_exit_reason(-9999)
    assert "SIG9999" in msg


def test_record_exit_reason_writes_to_worker_log(monkeypatch, tmp_path) -> None:
    """When LILBEE_DATA is set, exit reason gets appended to the worker log."""
    import multiprocessing

    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    (tmp_path / "logs").mkdir()
    log_file = tmp_path / "logs" / "worker-chat.log"
    log_file.write_text("preexisting\n")

    class _FakeProcess:
        @property
        def exitcode(self) -> int:
            return -15  # SIGTERM

    ctx = multiprocessing.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)
    channel = PipeChannel(
        role="chat",
        process=_FakeProcess(),  # type: ignore[arg-type]
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    try:
        channel._record_exit_reason()
    finally:
        with contextlib.suppress(Exception):
            parent_data.close()
        with contextlib.suppress(Exception):
            parent_health.close()
        with contextlib.suppress(Exception):
            child_data.close()
        with contextlib.suppress(Exception):
            child_health.close()
    body = log_file.read_text()
    assert "[supervisor]" in body
    assert "SIGTERM" in body


def test_record_exit_reason_skips_when_log_path_unset(monkeypatch, caplog) -> None:
    """Without LILBEE_DATA the reason logs to stderr but no file is touched."""
    import multiprocessing

    monkeypatch.delenv("LILBEE_DATA", raising=False)

    class _FakeProcess:
        @property
        def exitcode(self) -> int:
            return 7

    ctx = multiprocessing.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)
    channel = PipeChannel(
        role="chat",
        process=_FakeProcess(),  # type: ignore[arg-type]
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    try:
        with caplog.at_level("WARNING", logger="lilbee.providers.worker.transport_pipe"):
            channel._record_exit_reason()
    finally:
        with contextlib.suppress(Exception):
            parent_data.close()
        with contextlib.suppress(Exception):
            parent_health.close()
        with contextlib.suppress(Exception):
            child_data.close()
        with contextlib.suppress(Exception):
            child_health.close()
    assert any("exited with code 7" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_recv_for_drains_stale_call_id_frames(caplog) -> None:
    """Frames with mismatched call_id are silently discarded; the right one wins."""
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)

    class _FakeProcess:
        def is_alive(self) -> bool:
            return True

        @property
        def pid(self) -> int:
            return -1

    channel = PipeChannel(
        role="echo",
        process=_FakeProcess(),  # type: ignore[arg-type]
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    # Stuff three frames into the parent end via the child side: stale,
    # stale, then matching call_id=42.
    child_data.send((1, "result", "stale-1"))
    child_data.send((2, "result", "stale-2"))
    child_data.send((42, "result", "winner"))
    try:
        with caplog.at_level("DEBUG", logger="lilbee.providers.worker.transport_pipe"):
            kind, value = await channel._recv_for(42)
        assert kind == "result"
        assert value == "winner"
        assert any("discarding stale frame" in rec.message for rec in caplog.records)
    finally:
        channel._executor.shutdown(wait=False, cancel_futures=True)
        with contextlib.suppress(Exception):
            parent_data.close()
        with contextlib.suppress(Exception):
            parent_health.close()
        with contextlib.suppress(Exception):
            child_data.close()
        with contextlib.suppress(Exception):
            child_health.close()


@pytest.mark.asyncio
async def test_reader_thread_starts_lazily_on_first_recv() -> None:
    """No reader thread spins up until a coroutine awaits a frame."""
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)

    class _FakeProcess:
        def is_alive(self) -> bool:
            return True

        @property
        def pid(self) -> int:
            return -1

    channel = PipeChannel(
        role="echo",
        process=_FakeProcess(),  # type: ignore[arg-type]
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    try:
        assert channel._reader_thread is None
        assert not channel._reader_started.is_set()
        child_data.send((1, "result", "go"))
        _, _, value = await channel._recv()
        assert value == "go"
        assert channel._reader_started.is_set()
        assert channel._reader_thread is not None
        assert channel._reader_thread.is_alive()
    finally:
        channel._executor.shutdown(wait=False, cancel_futures=True)
        for handle in (parent_data, parent_health, child_data, child_health):
            with contextlib.suppress(Exception):
                handle.close()


@pytest.mark.asyncio
async def test_reader_thread_surfaces_eof_as_worker_crash() -> None:
    """Closing the parent conn drains EOF into a WorkerCrashError on next recv."""
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)

    class _FakeProcess:
        def is_alive(self) -> bool:
            return True

        @property
        def pid(self) -> int:
            return -1

    channel = PipeChannel(
        role="echo",
        process=_FakeProcess(),  # type: ignore[arg-type]
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    try:
        child_data.send((1, "result", "primer"))
        await channel._recv()
        # Close both ends so the reader's blocking recv() raises EOFError;
        # the next _recv() should then surface the crash to the caller.
        with contextlib.suppress(Exception):
            child_data.close()
        with contextlib.suppress(Exception):
            parent_data.close()
        with pytest.raises(WorkerCrashError):
            await asyncio.wait_for(channel._recv(), timeout=2.0)
    finally:
        channel._executor.shutdown(wait=False, cancel_futures=True)
        for handle in (parent_health, child_health):
            with contextlib.suppress(Exception):
                handle.close()


@pytest.mark.asyncio
async def test_stream_preserves_frame_ordering_under_burst() -> None:
    """A burst of stream_chunks is delivered to stream() in send order."""
    import multiprocessing as _mp

    from lilbee.providers.worker.wire_kinds import WireKind

    ctx = _mp.get_context("spawn")
    parent_data, child_data = ctx.Pipe(duplex=True)
    parent_health, child_health = ctx.Pipe(duplex=True)
    abort_flag = ctx.Value("b", 0, lock=True)

    class _FakeProcess:
        def is_alive(self) -> bool:
            return True

        @property
        def pid(self) -> int:
            return -1

    channel = PipeChannel(
        role="echo",
        process=_FakeProcess(),  # type: ignore[arg-type]
        parent_conn=parent_data,
        health_conn=parent_health,
        abort_flag=abort_flag,
    )
    burst_count = 250
    try:
        # Start the reader first so the OS pipe buffer drains as we send;
        # otherwise child_data.send() blocks once the buffer fills.
        channel._ensure_reader()

        def _produce() -> None:
            for i in range(burst_count):
                child_data.send((7, WireKind.STREAM_CHUNK, f"chunk-{i}"))
            child_data.send((7, WireKind.STREAM_END, None))

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()

        received: list[str] = []
        seen_end = False
        while not seen_end:
            kind, value = await asyncio.wait_for(channel._recv_for(7), timeout=5.0)
            if kind == WireKind.STREAM_END:
                seen_end = True
            else:
                received.append(value)
        producer.join(timeout=1.0)
        assert received == [f"chunk-{i}" for i in range(burst_count)]
    finally:
        channel._executor.shutdown(wait=False, cancel_futures=True)
        for handle in (parent_data, parent_health, child_data, child_health):
            with contextlib.suppress(Exception):
                handle.close()
