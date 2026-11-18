"""A chat stream cut after it sent text, over a real local socket and the real fleet client."""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider

_HEADERS = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
)
_CUT_MESSAGE = "answer is cut"


def _chunk(text: str) -> bytes:
    body = f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n'.encode()
    return b"%x\r\n" % len(body) + body + b"\r\n"


class _CuttingServer:
    """Sends *frames* SSE text frames, then ends the connection the way *how* says."""

    def __init__(self, frames: int, how: str) -> None:
        self._frames = frames
        self._how = how
        self.streams = 0
        self.cut = threading.Event()
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._listener.close()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        request = b""
        while b"\r\n\r\n" not in request:
            request += conn.recv(65536)
        head, _, body = request.partition(b"\r\n\r\n")
        length = next(
            int(line.split(b":")[1])
            for line in head.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
        while len(body) < length:
            body += conn.recv(65536)
        streaming = b'"stream":true' in body.replace(b" ", b"")
        if streaming:
            self.streams += 1
        conn.sendall(_HEADERS)
        for index in range(self._frames):
            conn.sendall(_chunk(f"word{index} "))
        if streaming:
            # The cut waits for the test to read every frame: a reset can discard unread data.
            assert self.cut.wait(10)
        if self._how == "reset":
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        elif self._how == "ended":
            conn.sendall(b"0\r\n\r\n")
        conn.close()


def _routing_over(monkeypatch: pytest.MonkeyPatch, server: _CuttingServer) -> RoutingProvider:
    monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
    routing = RoutingProvider()
    fleet = routing._get_local()
    assert isinstance(fleet, FleetProvider)
    client = LlamaServerClient(server.url, "chat-0", profile=LLAMA.request_profile(WorkerRole.CHAT))
    # The fleet start-up layer is out; the client and its socket are real.
    monkeypatch.setattr(fleet, "_require_clients", lambda role: [client])
    monkeypatch.setattr(fleet, "_clients", {WorkerRole.CHAT: [client]})
    monkeypatch.setattr(fleet, "_drop_swap_refs", lambda **_k: None)
    monkeypatch.setattr(routing, "_announce", lambda _role, _ref: None)
    return routing


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("how", ["reset", "closed", "ended"])
def test_a_stream_cut_after_text_keeps_the_text_and_ends_with_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, frames: int, how: str
) -> None:
    server = _CuttingServer(frames, how)
    try:
        routing = _routing_over(monkeypatch, server)
        received: list[str] = []
        stream = routing.chat([{"role": "user", "content": "hi"}], stream=True)
        with pytest.raises(ProviderError, match=_CUT_MESSAGE):
            for item in stream:
                assert isinstance(item, str)
                received.append(item)
                if len(received) == frames:
                    server.cut.set()
        assert received == [f"word{index} " for index in range(frames)]
        # A stream that already sent text is never retried: a retry would repeat it.
        assert server.streams == 1
    finally:
        server.close()
