"""The template step of the preset check reads only the answer, through the real provider."""

from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.factory import sdk_library
from tests.conformance import preset_check
from tests.conformance.preset_check import Setup

# The step drives the real provider; a missing library fails the import.
importlib.import_module(sdk_library().module)

_MODELS = {role: "m" for role in ("chat", "embed", "rerank", "vision")}


def _template_outcome(
    tmp_path: Path, message: dict[str, Any], seen: list[dict[str, Any]] | None = None
) -> preset_check.Outcome:
    """Run the template step through the real provider against a server that sends *message*.

    Each request body the server receives is appended to *seen*.
    """
    setup = Setup(("chat",), dict(_MODELS))
    choice = {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}
    reply = json.dumps(
        {"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if seen is not None:
                seen.append(json.loads(body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with preset_check.configured(
            f"http://127.0.0.1:{server.server_address[1]}", setup, tmp_path
        ):
            return preset_check._STEP_FUNCTIONS["template"](setup)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("message", "ok"),
    [
        ({"content": "Rome."}, True),
        ({"content": "<think>No idea.</think>Rome."}, True),
        ({"content": "Rome.", "reasoning_content": "No idea."}, True),
        ({"content": "<think>Rome, surely.</think>Florence."}, False),
        ({"content": "<think>Rome</think>"}, False),
        ({"content": "<think>Rome, so"}, False),
        ({"content": "Florence.", "reasoning_content": "Rome, surely."}, False),
    ],
    ids=[
        "answer",
        "inline-think-right",
        "reasoning-field-right",
        "inline-think-wrong",
        "no-answer",
        "unterminated-think",
        "reasoning-field-wrong",
    ],
)
def test_preset_template_step_reads_only_the_answer(
    tmp_path: Path, message: dict[str, Any], ok: bool
) -> None:
    assert _template_outcome(tmp_path, message).ok is ok


def test_preset_template_step_request_is_deterministic_and_capped(tmp_path: Path) -> None:
    """A sampling engine must not decide the step: temperature 0, and a cap the client sends."""
    seen: list[dict[str, Any]] = []
    _template_outcome(tmp_path, {"content": "Rome."}, seen)
    assert len(seen) == 1
    assert seen[0]["temperature"] == 0
    assert seen[0]["max_tokens"] == 1024
