#!/usr/bin/env python3
"""Minimal stand-in for an Ollama daemon, used by the litellm integration tests.

Parses ``--port`` from argv and serves the subset of the Ollama API the
litellm SDK calls: ``GET /api/tags``, ``POST /api/generate`` (plain and
streaming NDJSON), ``POST /api/embed``, and ``POST /api/show``. Chat goes
through ``/api/generate`` because litellm routes the ``ollama/`` prefix to
its completion handler, not its chat handler.
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
        if self.path == "/api/generate":
            model = body.get("model", "qwen3:0.6b")
            if body.get("stream"):
                self._send_ndjson(
                    [
                        {"model": model, "response": "stub-", "done": False},
                        {"model": model, "response": "chat", "done": False},
                        {"model": model, "response": "", "done": True},
                    ]
                )
            else:
                self._send_json(
                    {
                        "model": model,
                        "response": _STUB_CHAT_TEXT,
                        "done": True,
                        "done_reason": "stop",
                    }
                )
        elif self.path == "/api/embed":
            inputs = body.get("input", [])
            count = len(inputs) if isinstance(inputs, list) else 1
            # prompt_eval_count is required: without it litellm calls a
            # logging method its own Logging object does not have.
            self._send_json(
                {"embeddings": [_STUB_EMBEDDING for _ in range(count)], "prompt_eval_count": 8}
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

    def _send_ndjson(self, chunks: list[dict[str, object]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write((json.dumps(chunk) + "\n").encode())


def _port_from_argv(argv: list[str]) -> int:
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            return int(argv[index + 1])
    raise SystemExit("--port is required")


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", _port_from_argv(sys.argv)), _Handler).serve_forever()
