"""The Authorization header a model listing request carries, read at a stub server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from lilbee.core.config import cfg
from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.sdk_llm_provider import SdkLLMProvider

_UNREACHABLE = "http://127.0.0.1:9"


class _Stub(BaseHTTPRequestHandler):
    seen: ClassVar[list[tuple[str, str | None]]] = []

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        if self.path.endswith("/api/tags"):
            body = json.dumps({"models": [{"name": "tagged:latest"}]}).encode()
        else:
            body = json.dumps({"data": [{"id": "listed"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub() -> Iterator[tuple[str, list[tuple[str, str | None]]]]:
    handler = type("Handler", (_Stub,), {"seen": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Neither "localhost:11434" nor "ollama" appears in this address, like a LAN host.
        yield f"http://127.0.0.1:{server.server_port}", handler.seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ollama_listing_carries_no_key_at_any_address(
    stub: tuple[str, list[tuple[str, str | None]]],
) -> None:
    url, seen = stub
    cfg.llm_api_key = "sk-secret"
    cfg.ollama_base_url = url
    cfg.lm_studio_base_url = _UNREACHABLE
    names = SdkLLMProvider(LiterLlmBackend()).list_models()
    assert names == ["tagged:latest"]
    assert seen == [("/api/tags", None)]


def test_lm_studio_listing_still_carries_the_key(
    stub: tuple[str, list[tuple[str, str | None]]],
) -> None:
    url, seen = stub
    cfg.llm_api_key = "sk-secret"
    cfg.ollama_base_url = _UNREACHABLE
    cfg.lm_studio_base_url = url
    names = SdkLLMProvider(LiterLlmBackend()).list_models()
    assert names == ["listed"]
    assert seen == [("/v1/models", "Bearer sk-secret")]
