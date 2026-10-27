"""The URL path an Ollama request reaches, read at a stub server with the real client."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.factory import sdk_library
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.sdk_backend import CompletionRequest, EmbeddingRequest

pytest.importorskip(sdk_library().module)

_CHAT_BODY = {
    "model": "m",
    "created_at": "2024-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
    "done_reason": "stop",
}
_EMBED_BODY = {"model": "m", "embeddings": [[0.5, 0.5]]}


class _Stub(BaseHTTPRequestHandler):
    paths: ClassVar[list[str]] = []

    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).paths.append(self.path)
        body = json.dumps(_EMBED_BODY if self.path.endswith("/embed") else _CHAT_BODY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub() -> Iterator[tuple[str, list[str]]]:
    handler = type("Handler", (_Stub,), {"paths": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler.paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/", "/proxy", "/proxy/v1"])
def test_ollama_requests_reach_the_native_routes(stub: tuple[str, list[str]], suffix: str) -> None:
    root, paths = stub
    base = root + suffix
    prefix = suffix.removesuffix("/").removesuffix("/v1")
    backend = LiterLlmBackend()
    fields: dict[str, Any] = {"ref": parse_model_ref("ollama/m"), "api_base": base}
    backend.complete(CompletionRequest(messages=[{"role": "user", "content": "hi"}], **fields))
    backend.embed(EmbeddingRequest(inputs=["x"], **fields))
    assert paths == [f"{prefix}/api/chat", f"{prefix}/api/embed"]
