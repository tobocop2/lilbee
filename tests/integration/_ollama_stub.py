#!/usr/bin/env python3
"""Minimal stand-in for an Ollama daemon, used by the SDK integration tests.

Parses ``--port`` from argv and serves the subset of the Ollama API the
provider calls: ``GET /api/tags``, ``POST /api/show``, ``POST /api/chat`` (plain
and streaming NDJSON) and ``POST /api/embed``.
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
        if self.path == "/api/chat":
            model = body.get("model", "qwen3:0.6b")
            if body.get("stream", True):
                self._send_ndjson(
                    [
                        self._chat_line(model, "stub-", done=False),
                        self._chat_line(model, "chat", done=False),
                        self._chat_line(model, "", done=True),
                    ]
                )
            else:
                self._send_json(self._chat_line(model, _STUB_CHAT_TEXT, done=True))
        elif self.path == "/api/embed":
            inputs = body.get("input", [])
            count = len(inputs) if isinstance(inputs, list) else 1
            self._send_json(
                {
                    "model": body.get("model", "nomic-embed-text"),
                    "embeddings": [_STUB_EMBEDDING for _ in range(count)],
                    "prompt_eval_count": 8,
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
    def _chat_line(model: str, content: str, *, done: bool) -> dict[str, object]:
        line: dict[str, object] = {
            "model": model,
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": done,
        }
        if done:
            line.update({"done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1})
        return line

    def _send_ndjson(self, lines: list[dict[str, object]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for line in lines:
            self.wfile.write(f"{json.dumps(line)}\n".encode())


def _port_from_argv(argv: list[str]) -> int:
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            return int(argv[index + 1])
    raise SystemExit("--port is required")


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", _port_from_argv(sys.argv)), _Handler).serve_forever()
