#!/usr/bin/env python3
"""Minimal stand-in for llama-server, used by the multi-gpu integration tests.

Parses ``--port`` from argv, ignores every other llama-server flag, and serves
the subset of the OpenAI surface the fleet calls: ``/health``,
``/v1/chat/completions`` (streaming, plain, and tool-calling), and
``/v1/embeddings`` (which also backs rank-pooling rerank). Vision OCR reuses the
chat endpoint, so a multipart image request gets the same stub answer.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions":
            if body.get("stream"):
                self._send_sse(["stub", "-chat"])
            elif body.get("tools"):
                # --jinja-style structured tool call: echo the first tool's name.
                tool_name = body["tools"][0]["function"]["name"]
                self._send_json(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "stub_call",
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": "{}",
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
            else:
                self._send_json({"choices": [{"message": {"content": "stub-chat"}}]})
        elif self.path == "/v1/embeddings":
            count = len(body.get("input", []))
            self._send_json({"data": [{"embedding": [0.5, 0.5]} for _ in range(count)]})
        else:
            self.send_error(404)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, tokens: list[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for token in tokens:
            chunk = json.dumps({"choices": [{"delta": {"content": token}}]})
            self.wfile.write(f"data: {chunk}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _port_from_argv(argv: list[str]) -> int:
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            return int(argv[index + 1])
    raise SystemExit("--port is required")


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", _port_from_argv(sys.argv)), _Handler).serve_forever()
