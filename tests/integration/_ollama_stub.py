#!/usr/bin/env python3
"""Minimal stand-in for an Ollama daemon, used by the SDK integration tests.

Parses ``--port`` from argv and serves the subset of the Ollama API the
provider calls: ``GET /api/tags``, ``POST /api/show``, and the OpenAI-compatible
``POST /v1/chat/completions`` (plain and streaming SSE) and ``POST /v1/embeddings``.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

_STUB_CHAT_TEXT = "stub-chat"
_STUB_EMBEDDING = [0.5, 0.5]
_STUB_MODELS = ["qwen3:0.6b", "nomic-embed-text"]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send_json({"models": [{"name": name} for name in _STUB_MODELS]})
        elif self.path == "/api/version":
            self._send_json({"version": "stub"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions":
            model = body.get("model", "qwen3:0.6b")
            if body.get("stream"):
                self._send_sse(
                    [
                        self._chunk(model, {"role": "assistant", "content": "stub-"}, None),
                        self._chunk(model, {"content": "chat"}, None),
                        self._chunk(model, {}, "stop"),
                    ]
                )
            else:
                self._send_json(
                    {
                        "id": "stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": _STUB_CHAT_TEXT},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                )
        elif self.path == "/v1/embeddings":
            inputs = body.get("input", [])
            count = len(inputs) if isinstance(inputs, list) else 1
            self._send_json(
                {
                    "object": "list",
                    "model": body.get("model", "nomic-embed-text"),
                    "data": [
                        {"object": "embedding", "index": i, "embedding": _STUB_EMBEDDING}
                        for i in range(count)
                    ],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            )
        elif self.path == "/api/show":
            self._send_json({"parameters": "stub-params", "capabilities": ["completion"]})
        else:
            self.send_error(404)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _chunk(model: str, delta: dict[str, object], finish: str | None) -> dict[str, object]:
        return {
            "id": "stub",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def _send_sse(self, chunks: list[dict[str, object]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _port_from_argv(argv: list[str]) -> int:
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            return int(argv[index + 1])
    raise SystemExit("--port is required")


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", _port_from_argv(sys.argv)), _Handler).serve_forever()
