"""Start a mistral.rs binary on the CPU and ask one chat question.

Usage: check_mistralrs_chat.py <mistralrs binary> <model id>

The binary starts `serve` with one text model and answers the OpenAI chat route under the one
model name it serves. The check passes when the answer holds a non-empty message.
"""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import time

HOST = "127.0.0.1"
SERVED_NAME = "default"
START_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 300
_POLL_S = 3
_OK = 200


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def argv(binary: str, model: str, port: int) -> list[str]:
    return [binary, "serve", "--host", HOST, "--port", str(port), "text", "--cpu", "-m", model]


def request(
    port: int, method: str, path: str, body: dict | None, timeout: float
) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the server stopped with code {proc.returncode} before it was ready")
        try:
            if request(port, "GET", "/v1/models", None, 10)[0] == _OK:
                return
        except OSError:
            pass
        time.sleep(_POLL_S)
    raise SystemExit(f"the server was not ready after {timeout:.0f} s")


def chat(port: int) -> str:
    body = {
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 16,
    }
    status, raw = request(port, "POST", "/v1/chat/completions", body, REQUEST_TIMEOUT_S)
    if status != _OK:
        raise SystemExit(f"the chat route answered {status}: {raw[:300]!r}")
    message = json.loads(raw)["choices"][0]["message"]["content"]
    if not isinstance(message, str) or not message.strip():
        raise SystemExit(f"the chat answer holds no text: {raw[:300]!r}")
    return message


def main(args: list[str]) -> int:
    if len(args) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    port = free_port()
    proc = subprocess.Popen(argv(args[1], args[2], port))
    try:
        wait_ready(port, proc, START_TIMEOUT_S)
        print("chat answer:", chat(port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
