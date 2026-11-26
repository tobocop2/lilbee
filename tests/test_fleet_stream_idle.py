"""The silence of a chat stream's body ends at the idle bound.

The client timeout is the cold-load budget, and it holds until the headers arrive: llama-swap
sends no byte while it loads a model. After the headers, the silence between two reads of the
body is bounded by the idle bound. Each test runs the real client against a loopback server
that sends headers and frames on a script.

The tests that assert a time also guard the mechanism. The client lowers the read timeout in
the request's own ``timeout`` extension after the headers, and httpcore reads it at the first
body read. If httpcore took the read timeout earlier, the body would wait the client timeout,
which is far over ``_QUICK_S``.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, replace
from typing import BinaryIO

import httpx
import pytest

from lilbee.providers.fleet import client as fleet_client
from lilbee.providers.fleet.client import LlamaServerClient, StreamInterruptedError
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole

# The client timeout of a test: the cold-load budget in the real client. At the base commit a
# silent stream waits this long, which is well above `_QUICK_S` and below the test timeout.
_COLD_LOAD_S = 20.0
# The idle bound of a test.
_IDLE_S = 0.5
# A stream bounded by the idle bound must end within this time of the last frame.
_QUICK_S = 5.0
_MESSAGES = [{"role": "user", "content": "hi"}]
_HEADERS = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
    b"Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
)
_ANSWER = json.dumps(
    {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ]
    }
).encode()


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


_ROLE_FRAME = _chunk(
    b'data: {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n'
)


def _frame(text: str, *, finish: bool = False) -> bytes:
    choice = {"index": 0, "delta": {"content": text}, "finish_reason": "stop" if finish else None}
    return _chunk(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())


@dataclass(frozen=True)
class _Script:
    """What the server does with each request; every wait is in seconds."""

    head_wait_s: float = 0.0  # before a stream's headers, as llama-swap holds a cold load
    wait_s: float = 0.0  # after the headers, before the first frame
    heartbeat_s: float = 0.0  # a ":" comment this often during wait_s; 0 sends none
    frames: int = 1
    gap_s: float = 0.0
    end: bool = False  # end the stream; otherwise go silent
    close: bool = False  # close the connection after the frames, with the body not ended
    answer_wait_s: float = 0.0  # before the answer to a request that does not stream
    role_first: bool = False  # a first data line with no content, as OpenAI-style servers send


class _ScriptedServer:
    """A keep-alive loopback server that answers every request on its script."""

    def __init__(self, script: _Script) -> None:
        self._script = script
        self._release = threading.Event()
        # The client port of each request, in order: one port means one connection.
        self.peers: list[int] = []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, peer = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn, peer[1]), daemon=True).start()

    def _serve(self, conn: socket.socket, peer: int) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn, contextlib.suppress(OSError), conn.makefile("rb") as reader:
            while (body := _read_request(reader)) is not None:
                self.peers.append(peer)
                if not self._answer(conn, body):
                    return

    def _answer(self, conn: socket.socket, body: dict[str, object]) -> bool:
        """Send one answer; whether the connection stays open for the next request."""
        script = self._script
        if not body.get("stream"):
            self._release.wait(script.answer_wait_s)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(_ANSWER)}\r\n\r\n".encode()
                + _ANSWER
            )
            return True
        self._release.wait(script.head_wait_s)
        out = bytearray(_HEADERS + (_ROLE_FRAME if script.role_first else b""))

        def pause(seconds: float) -> None:
            if seconds > 0:
                conn.sendall(out)
                out.clear()
                self._release.wait(seconds)

        if script.heartbeat_s:
            deadline = time.monotonic() + script.wait_s
            while time.monotonic() < deadline:
                pause(script.heartbeat_s)
                out.extend(_chunk(b": keep-alive\n\n"))
        else:
            pause(script.wait_s)
        for number in range(script.frames):
            if number:
                pause(script.gap_s)
            out.extend(_frame(f"w{number}"))
        if script.end:
            out.extend(_frame("", finish=True) + b"0\r\n\r\n")
            conn.sendall(out)
            return True
        conn.sendall(out)
        if not script.close:
            self._release.wait(60)
        return False

    def close(self) -> None:
        self._release.set()
        self._sock.close()


def _read_request(reader: BinaryIO) -> dict[str, object] | None:
    """The JSON body of the next request on *reader*, or None once the client closes."""
    length = 0
    line = reader.readline()
    if not line:
        return None
    while line not in (b"\r\n", b""):
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
        line = reader.readline()
    return json.loads(reader.read(length) or b"{}")


@pytest.fixture(autouse=True)
def _short_idle_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # raising=False: the same tests run at the base commit, which has no idle bound.
    monkeypatch.setattr(fleet_client, "_STREAM_IDLE_TIMEOUT_S", _IDLE_S, raising=False)


@contextlib.contextmanager
def _serve(
    script: _Script, timeout: float = _COLD_LOAD_S, *, silent_prefill: bool = False
) -> Iterator[tuple[_ScriptedServer, LlamaServerClient]]:
    server = _ScriptedServer(script)
    profile = replace(LLAMA.request_profile(WorkerRole.CHAT), silent_prefill=silent_prefill)
    client = LlamaServerClient(server.url, "m", timeout=timeout, profile=profile)
    client._needs_alternation = False
    try:
        yield server, client
    finally:
        server.close()
        client.close()


# The three streaming chat sites, each read to its end.
_SITES: dict[str, Callable[[LlamaServerClient], Collection[object]]] = {
    "items": lambda client: list(client.chat_stream_items(_MESSAGES)),
    "text": lambda client: list(client.chat(_MESSAGES, stream=True)),
    "abortable": lambda client: client.chat_abortable(_MESSAGES),
}


def test_a_stream_quiet_after_its_first_frame_ends_with_the_cut_answer_error() -> None:
    with _serve(_Script()) as (_server, client):
        stream = client.chat_stream_items(_MESSAGES)
        assert next(stream) == "w0"
        start = time.monotonic()
        with pytest.raises(StreamInterruptedError) as caught:
            list(stream)
        waited = time.monotonic() - start
    assert "stopped answering" in str(caught.value)
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    assert waited < _QUICK_S


def test_a_quiet_wait_after_the_headers_ends_at_the_idle_bound() -> None:
    """A prefill that sends nothing after the headers is cut, though frames would come."""
    with _serve(_Script(wait_s=_IDLE_S * 3, end=True)) as (_server, client):
        start = time.monotonic()
        with pytest.raises(httpx.ReadTimeout) as caught:
            list(client.chat_stream_items(_MESSAGES))
        waited = time.monotonic() - start
    assert str(caught.value) == "The model server sent no answer for 0.5 s, so the request stopped."
    assert caught.value.request.url.path == "/v1/chat/completions"
    assert waited < _QUICK_S


@pytest.mark.parametrize("site", sorted(_SITES))
def test_headers_then_silence_ends_at_the_idle_bound(site: str) -> None:
    with _serve(_Script(wait_s=60.0)) as (_server, client):
        start = time.monotonic()
        with pytest.raises(httpx.ReadTimeout, match=r"model server sent no answer for 0\.5 s"):
            _SITES[site](client)
        waited = time.monotonic() - start
    assert waited < _QUICK_S


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_cold_load_past_the_client_timeout_ends_with_the_plain_read_timeout(site: str) -> None:
    """No headers came, so the server's silence after the headers is not the cause."""
    with (
        _serve(_Script(head_wait_s=_IDLE_S * 3, end=True), timeout=_IDLE_S) as (_server, client),
        pytest.raises(httpx.ReadTimeout) as caught,
    ):
        _SITES[site](client)
    assert not isinstance(caught.value, fleet_client.StreamSilentError)


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_prefill_that_sends_comments_is_not_cut(site: str) -> None:
    """llama-server sends a ":" comment during a long prefill; each one resets the bound."""
    script = _Script(wait_s=_IDLE_S * 3, heartbeat_s=_IDLE_S / 4, end=True)
    with _serve(script) as (_server, client):
        assert "w0" in _SITES[site](client)


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_cold_load_before_the_headers_keeps_the_cold_load_budget(site: str) -> None:
    with _serve(_Script(head_wait_s=_IDLE_S * 3, end=True)) as (_server, client):
        assert "w0" in _SITES[site](client)


def test_a_slow_steady_stream_is_not_cut() -> None:
    """Each gap is under the idle bound; the whole stream is far over it."""
    with _serve(_Script(frames=6, gap_s=_IDLE_S * 0.6, end=True)) as (_server, client):
        start = time.monotonic()
        items = [item for item in client.chat_stream_items(_MESSAGES) if isinstance(item, str)]
        total = time.monotonic() - start
    assert items == [f"w{number}" for number in range(6)]
    assert total > _IDLE_S * 2


def test_a_shorter_client_timeout_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle bound only lowers a longer read timeout.

    The kept value is read from the request's timeout extension. A lower bound on the measured
    wait would depend on the clock: on Windows the monotonic clock ticks every 15.6 ms, so a wait
    of 0.5 s can read as less.
    """
    monkeypatch.setattr(fleet_client, "_STREAM_IDLE_TIMEOUT_S", 4.0, raising=False)
    kept: list[float | None] = []
    bound_body_reads = fleet_client._bound_body_reads

    def record(resp: httpx.Response) -> httpx.Request:
        request = bound_body_reads(resp)
        kept.append(request.extensions["timeout"]["read"])
        return request

    monkeypatch.setattr(fleet_client, "_bound_body_reads", record)
    with _serve(_Script(), timeout=0.5) as (_server, client):
        stream = client.chat_stream_items(_MESSAGES)
        assert next(stream) == "w0"
        start = time.monotonic()
        with pytest.raises(StreamInterruptedError):
            list(stream)
        waited = time.monotonic() - start
    assert kept == [0.5]
    assert waited < 2.5


def test_a_read_timeout_of_none_is_bounded_too() -> None:
    """A client built without a read timeout still gets the idle bound after the headers."""
    with _serve(_Script()) as (server, client):
        client._http.close()
        client._http = httpx.Client(base_url=server.url, timeout=httpx.Timeout(None))
        stream = client.chat_stream_items(_MESSAGES)
        assert next(stream) == "w0"
        start = time.monotonic()
        with pytest.raises(StreamInterruptedError):
            list(stream)
        waited = time.monotonic() - start
    assert waited < _QUICK_S


def test_the_text_stream_is_bounded_after_its_first_frame() -> None:
    with _serve(_Script()) as (_server, client):
        stream = client.chat(_MESSAGES, stream=True)
        assert next(stream) == "w0"
        start = time.monotonic()
        with pytest.raises(StreamInterruptedError) as caught:
            list(stream)
        waited = time.monotonic() - start
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    assert waited < _QUICK_S


def test_the_abortable_chat_is_bounded_after_its_first_frame() -> None:
    with _serve(_Script()) as (_server, client):
        start = time.monotonic()
        with pytest.raises(StreamInterruptedError) as caught:
            client.chat_abortable(_MESSAGES)
        waited = time.monotonic() - start
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    assert waited < _QUICK_S


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_stream_closed_after_its_first_frame_ends_with_the_cut_answer_error(site: str) -> None:
    """Any transport error after a frame cuts the answer, at all three sites alike."""
    with (
        _serve(_Script(close=True)) as (_server, client),
        pytest.raises(StreamInterruptedError) as caught,
    ):
        _SITES[site](client)
    assert isinstance(caught.value.__cause__, httpx.RemoteProtocolError)


def test_a_later_slow_request_on_the_same_connection_is_not_cut() -> None:
    """The bound ends with its stream: the pooled connection keeps the client timeout."""
    with _serve(_Script(end=True, answer_wait_s=_IDLE_S * 3)) as (server, client):
        assert "w0" in list(client.chat_stream_items(_MESSAGES))
        assert client.chat(_MESSAGES) == "ok"
    assert len(server.peers) == 2
    assert len(set(server.peers)) == 1


def test_many_streams_on_one_connection_end_without_an_error() -> None:
    with _serve(_Script(end=True)) as (server, client):
        for _ in range(1500):
            assert "w0" in list(client.chat_stream_items(_MESSAGES))
    assert len(server.peers) == 1500
    assert len(set(server.peers)) == 1


# An engine with a silent prefill (Lemonade on the NPU) sends its headers, then no byte until
# its first frame. The fake server holds the stream silent for three idle bounds.


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_silent_prefill_past_the_idle_bound_is_not_cut(site: str) -> None:
    with _serve(_Script(wait_s=_IDLE_S * 3, end=True), silent_prefill=True) as (_server, client):
        assert "w0" in _SITES[site](client)


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_silent_prefill_still_waits_only_the_client_timeout(site: str) -> None:
    with (
        _serve(_Script(wait_s=60.0), timeout=_IDLE_S * 2, silent_prefill=True) as (_s, client),
        pytest.raises(httpx.ReadTimeout),
    ):
        _SITES[site](client)


@pytest.mark.parametrize("site", sorted(_SITES))
def test_silence_after_the_first_frame_ends_at_the_idle_bound_for_a_silent_prefill(
    site: str,
) -> None:
    with _serve(_Script(wait_s=_IDLE_S), silent_prefill=True) as (_server, client):
        start = time.monotonic()
        with pytest.raises(StreamInterruptedError):
            _SITES[site](client)
        waited = time.monotonic() - start
    assert _IDLE_S <= waited < _QUICK_S


@pytest.mark.parametrize("site", sorted(_SITES))
def test_a_comment_is_not_the_first_frame_of_a_silent_prefill(site: str) -> None:
    """The bound starts at a ``data:`` line, so a ping before it does not start it."""
    script = _Script(wait_s=_IDLE_S * 3, heartbeat_s=_IDLE_S * 2, end=True)
    with _serve(script, silent_prefill=True) as (_server, client):
        assert "w0" in _SITES[site](client)


@pytest.mark.parametrize("site", sorted(_SITES))
def test_silence_after_a_first_frame_without_content_is_the_silence_error(site: str) -> None:
    """A role-only first frame starts the bound; its cut is a read timeout, not a broken body."""
    with (
        _serve(_Script(role_first=True, wait_s=60.0), silent_prefill=True) as (_s, client),
        pytest.raises(fleet_client.StreamSilentError),
    ):
        _SITES[site](client)


def test_a_slow_reader_of_a_silent_prefill_stream_is_not_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound runs only while the client waits for a line, not while the caller holds one.

    The caller holds the second frame past the bound. The fake server then still has frames to
    send, so a bound that ran during the hold has shut the socket.
    """
    shutdowns: list[httpx.Response] = []
    real = fleet_client._shut_down_socket

    def record(resp: httpx.Response) -> bool:
        shutdowns.append(resp)
        return real(resp)

    monkeypatch.setattr(fleet_client, "_shut_down_socket", record)
    script = _Script(frames=8, gap_s=_IDLE_S * 0.3, end=True)
    with _serve(script, silent_prefill=True) as (_server, client):
        stream = client.chat_stream_items(_MESSAGES)
        assert [next(stream), next(stream)] == ["w0", "w1"]
        time.sleep(_IDLE_S * 1.5)
        assert "w7" in list(stream)
    assert shutdowns == []


def test_a_bound_never_fires_after_its_stream_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """The next request on the pooled connection is not cut by the bound of the one before."""
    shutdowns: list[httpx.Response] = []
    real = fleet_client._shut_down_socket

    def record(resp: httpx.Response) -> bool:
        shutdowns.append(resp)
        return real(resp)

    monkeypatch.setattr(fleet_client, "_shut_down_socket", record)
    with _serve(_Script(end=True), silent_prefill=True) as (server, client):
        assert "w0" in list(client.chat_stream_items(_MESSAGES))
        time.sleep(_IDLE_S * 3)
        assert "w0" in list(client.chat_stream_items(_MESSAGES))
        time.sleep(_IDLE_S * 3)
    assert shutdowns == []
    assert len(server.peers) == 2
    assert len(set(server.peers)) == 1


def test_a_silent_prefill_stream_starts_one_thread_for_all_its_lines() -> None:
    started: list[int] = []
    real = threading.Thread.start

    def count(thread: threading.Thread) -> None:
        started.append(1)
        real(thread)

    with _serve(_Script(frames=50, end=True), silent_prefill=True) as (_server, client):
        list(client.chat_stream_items(_MESSAGES))  # opens the connection and its server thread
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(threading.Thread, "start", count)
            assert "w49" in list(client.chat_stream_items(_MESSAGES))
    assert len(started) == 1


def test_a_connection_lost_after_the_first_frame_of_a_silent_prefill_is_a_cut_answer() -> None:
    with _serve(_Script(close=True), silent_prefill=True) as (_server, client):
        stream = client.chat_stream_items(_MESSAGES)
        assert next(stream) == "w0"
        with pytest.raises(StreamInterruptedError) as caught:
            list(stream)
    assert not isinstance(caught.value.__cause__, fleet_client.StreamSilentError)


@pytest.fixture
def watchdog_threads(monkeypatch: pytest.MonkeyPatch) -> list[threading.Thread]:
    """Every watchdog thread the test starts, kept by identity when the watchdog is armed.

    The thread body then sleeps 50 ms after its loop, so a stop that does not join leaves
    the thread alive when the stream returns, in every ending.
    """
    seen: list[threading.Thread] = []
    watchdog = fleet_client._SilenceWatchdog
    real_arm, real_run = watchdog.arm, watchdog._run

    def arm(self: fleet_client._SilenceWatchdog) -> None:
        real_arm(self)
        if self._thread is not None and self._thread not in seen:
            seen.append(self._thread)

    def run(self: fleet_client._SilenceWatchdog) -> None:
        real_run(self)
        time.sleep(0.05)

    monkeypatch.setattr(watchdog, "arm", arm)
    monkeypatch.setattr(watchdog, "_run", run)
    return seen


def _end_normally(client: LlamaServerClient, threads: list[threading.Thread]) -> None:
    assert "w2" in list(client.chat_stream_items(_MESSAGES))


def _end_cut(client: LlamaServerClient, threads: list[threading.Thread]) -> None:
    with pytest.raises(StreamInterruptedError):
        list(client.chat_stream_items(_MESSAGES))


def _end_by_caller_error(client: LlamaServerClient, threads: list[threading.Thread]) -> None:
    stream = client.chat_stream_items(_MESSAGES)
    assert [next(stream), next(stream)] == ["w0", "w1"]
    assert threads[-1].is_alive()
    with pytest.raises(RuntimeError):
        stream.throw(RuntimeError("the caller failed"))


def _end_by_close(client: LlamaServerClient, threads: list[threading.Thread]) -> None:
    stream = client.chat_stream_items(_MESSAGES)
    assert [next(stream), next(stream)] == ["w0", "w1"]
    assert threads[-1].is_alive()
    stream.close()


@pytest.mark.parametrize(
    ("script", "end"),
    [
        pytest.param(_Script(frames=3, end=True), _end_normally, id="normal-end"),
        pytest.param(_Script(frames=3), _end_cut, id="cut-by-silence"),
        pytest.param(_Script(frames=3), _end_by_caller_error, id="caller-error"),
        pytest.param(_Script(frames=3), _end_by_close, id="close"),
    ],
)
def test_no_watchdog_thread_outlives_its_stream(
    script: _Script,
    end: Callable[[LlamaServerClient, list[threading.Thread]], None],
    watchdog_threads: list[threading.Thread],
) -> None:
    """The stream joins its watchdog before it returns: the thread it started is not alive.

    Each pass starts one watchdog and the test holds that thread object, so a renamed target
    cannot hide it. The thread ends 50 ms late, so a missing join shows in every ending.
    """
    with _serve(script, silent_prefill=True) as (_server, client):
        for number in range(1, 6):
            end(client, watchdog_threads)
            assert len(watchdog_threads) == number
            assert not watchdog_threads[-1].is_alive()
