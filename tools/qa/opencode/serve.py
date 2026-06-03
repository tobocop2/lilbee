"""lilbee serve lifecycle and the launcher-serve.log scrapers (errors, chat-200s)."""

from __future__ import annotations

import contextlib
import subprocess
import time
from pathlib import Path

import httpx
from harness_config import _SERVE_BOOT_TIMEOUT_S, _SERVE_TERMINATE_TIMEOUT_S


def boot_serve(workspace: Path, port: int, log_path: Path) -> subprocess.Popen[bytes]:
    """Spawn lilbee serve on *port* in its own process group and wait for /api/health.

    The process group lets :func:`stop_serve` reap the whole tree (``uv`` plus
    the lilbee child it forks) rather than orphaning the child python.
    """
    log_file = log_path.open("ab")
    proc = subprocess.Popen(
        ["uv", "run", "lilbee", "serve", "--port", str(port)],
        cwd=workspace,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + _SERVE_BOOT_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"lilbee serve exited before becoming ready (rc={proc.returncode})")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1.0)
            if resp.status_code == 200:
                return proc
        except (httpx.HTTPError, httpx.RequestError):
            pass
        time.sleep(0.5)
    stop_serve(proc)
    raise TimeoutError("lilbee serve did not become ready in time")


def stop_serve(proc: subprocess.Popen[bytes]) -> None:
    """Reap the full lilbee-serve process group (uv parent + lilbee python child)."""
    import os
    import signal

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=_SERVE_TERMINATE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def _scrape_serve_errors(workspace: Path) -> str:
    """Pull worker/dispatch exceptions out of the cell's launcher-serve.log.

    A cell whose scenarios pass the pane substring check but whose lilbee serve
    raised a chat-worker exception (e.g., tool-call shape mismatch, context
    overflow, ProviderError) cannot be marked "supported" -- the model
    + parser combination is broken, even if opencode happened to render
    enough text to satisfy the smoke substrings.
    """
    log_file = workspace / ".lilbee" / "data" / "logs" / "launcher-serve.log"
    if not log_file.exists():
        return ""
    text = log_file.read_text(encoding="utf-8", errors="replace")
    needles = ("Traceback (most recent call last)", "ProviderError:", "WorkerError:", "TypeError:")
    hits = [line for line in text.splitlines() if any(n in line for n in needles)]
    return "\n".join(hits[-8:]) if hits else ""


def _count_ok_chat_completions(workspace: Path) -> int:
    """Count successful ``POST /v1/chat/completions`` responses in launcher-serve.log.

    uvicorn logs each request as ``... "POST /v1/chat/completions HTTP/1.1" 200 OK``.
    A cell where opencode never received a 200 chat back (model never loaded,
    every turn 500'd) has a count of zero and cannot be a real PASS. Paired
    with :func:`_scrape_serve_errors`, this catches the SSE-200-then-crash case
    too: the header logs 200 but the mid-stream exception lands in serve_errors.
    """
    log_file = workspace / ".lilbee" / "data" / "logs" / "launcher-serve.log"
    if not log_file.exists():
        return 0
    text = log_file.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if 'POST /v1/chat/completions HTTP/1.1" 200' in line)
