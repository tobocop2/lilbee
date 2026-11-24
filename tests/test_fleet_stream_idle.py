"""A chat stream that goes quiet after its first frame ends at the idle bound.

The client timeout is the cold-load budget, and it holds until the first frame: the engine
sends its headers before it ends prompt evaluation. After the first frame, the silence
between two reads is bounded by the idle bound. Each test runs the real client against a
loopback server that sends headers and frames on a script.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time

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


def _frame(text: str, *, finish: bool = False) -> bytes:
    choice = {"index": 0, "delta": {"content": text}, "finish_reason": "stop" if finish else None}
    body = f"data: {json.dumps({'choices': [choice]})}\n\n".encode()
    return f"{len(body):x}\r\n".encode() + body + b"\r\n"


class _ScriptedServer:
    """A one-connection loopback server: headers, a wait, frames at a gap, then silence or end."""

    def __init__(
        self, *, wait_s: float = 0.0, frames: int = 1, gap_s: float = 0.0, end: bool = False
    ) -> None:
        self._wait_s = wait_s
        self._frames = frames
        self._gap_s = gap_s
        self._end = end
        self._release = threading.Event()
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        conn, _ = self._sock.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            time.sleep(self._wait_s)
            for number in range(self._frames):
                if number:
                    time.sleep(self._gap_s)
                conn.sendall(_frame(f"w{number}"))
            if self._end:
                conn.sendall(_frame("", finish=True) + b"0\r\n\r\n")
                # Read to the client's close: a close with unread request bytes resets the
                # connection, and the reset would discard the frames the client has not read.
                conn.shutdown(socket.SHUT_WR)
                conn.settimeout(5)
                with contextlib.suppress(OSError):
                    while conn.recv(65536):
                        pass
                return
            self._release.wait(timeout=60)

    def close(self) -> None:
        self._release.set()
        self._thread.join(timeout=5)
        self._sock.close()


@pytest.fixture(autouse=True)
def _short_idle_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # raising=False: the same tests run at the base commit, which has no idle bound.
    monkeypatch.setattr(fleet_client, "_STREAM_IDLE_TIMEOUT_S", _IDLE_S, raising=False)


def _client(server: _ScriptedServer, timeout: float = _COLD_LOAD_S) -> LlamaServerClient:
    client = LlamaServerClient(
        server.url, "m", timeout=timeout, profile=LLAMA.request_profile(WorkerRole.CHAT)
    )
    client._needs_alternation = False
    return client


def test_a_stream_quiet_after_its_first_frame_ends_with_the_cut_answer_error() -> None:
    server = _ScriptedServer(frames=1)
    client = _client(server)
    stream = client.chat_stream_items(_MESSAGES)
    assert next(stream) == "w0"
    start = time.monotonic()
    with pytest.raises(StreamInterruptedError) as caught:
        list(stream)
    waited = time.monotonic() - start
    server.close()
    client.close()
    assert "stopped answering" in str(caught.value)
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    assert waited < _QUICK_S


def test_a_quiet_wait_before_the_first_frame_keeps_the_cold_load_budget() -> None:
    """Prompt evaluation sends headers early and no frame; the idle bound must not cut it."""
    server = _ScriptedServer(wait_s=_IDLE_S * 3, frames=1, end=True)
    client = _client(server)
    items = list(client.chat_stream_items(_MESSAGES))
    server.close()
    client.close()
    assert "w0" in items


def test_headers_then_silence_ends_at_the_cold_load_budget() -> None:
    server = _ScriptedServer(wait_s=60.0, frames=1)
    client = _client(server, timeout=1.0)
    start = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        list(client.chat_stream_items(_MESSAGES))
    waited = time.monotonic() - start
    server.close()
    client.close()
    assert 1.0 <= waited < _QUICK_S


def test_a_slow_steady_stream_is_not_cut() -> None:
    """Each gap is under the idle bound; the whole stream is far over it."""
    server = _ScriptedServer(frames=6, gap_s=_IDLE_S * 0.6, end=True)
    client = _client(server)
    start = time.monotonic()
    items = [item for item in client.chat_stream_items(_MESSAGES) if isinstance(item, str)]
    total = time.monotonic() - start
    server.close()
    client.close()
    assert items == [f"w{number}" for number in range(6)]
    assert total > _IDLE_S * 2


def test_a_shorter_client_timeout_is_kept() -> None:
    """The idle bound only lowers a longer read timeout."""
    server = _ScriptedServer(frames=1)
    client = _client(server, timeout=_IDLE_S / 5)
    stream = client.chat_stream_items(_MESSAGES)
    assert next(stream) == "w0"
    with pytest.raises(StreamInterruptedError):
        list(stream)
    server.close()
    client.close()


def test_a_read_timeout_of_none_is_bounded_too() -> None:
    """A client built without a read timeout still gets the idle bound after a frame."""
    server = _ScriptedServer(frames=1)
    client = _client(server)
    client._http.close()
    client._http = httpx.Client(base_url=server.url, timeout=httpx.Timeout(None))
    stream = client.chat_stream_items(_MESSAGES)
    assert next(stream) == "w0"
    start = time.monotonic()
    with pytest.raises(StreamInterruptedError):
        list(stream)
    waited = time.monotonic() - start
    server.close()
    client.close()
    assert waited < _QUICK_S


def test_the_text_stream_is_bounded_after_its_first_frame() -> None:
    server = _ScriptedServer(frames=1)
    client = _client(server)
    stream = client.chat(_MESSAGES, stream=True)
    assert next(stream) == "w0"
    start = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        list(stream)
    waited = time.monotonic() - start
    server.close()
    client.close()
    assert waited < _QUICK_S


def test_the_abortable_chat_is_bounded_after_its_first_frame() -> None:
    server = _ScriptedServer(frames=1)
    client = _client(server)
    start = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        client.chat_abortable(_MESSAGES)
    waited = time.monotonic() - start
    server.close()
    client.close()
    assert waited < _QUICK_S
