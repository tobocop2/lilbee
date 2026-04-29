"""MCP stdio JSON-RPC client.

Minimal sync client for `lilbee mcp` over stdio. One subprocess per session,
shared across all MCP scenarios in a single test file (the conftest hook
groups MCP tests by file under xdist's loadgroup).

Protocol: JSON-RPC 2.0 line-delimited, with the standard MCP initialization
handshake (initialize, response, notifications/initialized).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping
from queue import Empty, Queue
from types import TracebackType
from typing import Any, Self

_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_NAME = "lilbee-qa-matrix"
_CLIENT_VERSION = "0.0.1"
_DEFAULT_TIMEOUT = 30.0
_RECV_QUEUE_TIMEOUT = 1.0


class MCPError(RuntimeError):
    """JSON-RPC error returned by the MCP server."""


class MCPStdioClient:
    """Synchronous JSON-RPC client over `lilbee mcp` stdio."""

    def __init__(
        self,
        cmd: list[str],
        *,
        env: Mapping[str, str] | None = None,
        startup_timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=dict(env) if env is not None else None,
        )
        self._next_id = 0
        self._inbox: Queue[dict[str, Any]] = Queue()
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        self._initialize(timeout=startup_timeout)

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                # The MCP server occasionally emits non-JSON debug lines on
                # stdout. Drop them rather than failing; stderr surfaces them.
                self._stderr_lines.append(f"[non-json stdout] {line[:200]}")

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_lines.append(line.rstrip())

    def _send_message(self, message: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send_message(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params or {})}
        )
        return self._await_response(request_id, timeout=timeout)

    def _notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self._send_message({"jsonrpc": "2.0", "method": method, "params": dict(params or {})})

    def _await_response(self, request_id: int, *, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        unmatched: list[dict[str, Any]] = []
        try:
            while time.monotonic() < deadline:
                try:
                    msg = self._inbox.get(timeout=_RECV_QUEUE_TIMEOUT)
                except Empty:
                    if self._proc.poll() is not None:
                        raise MCPError(self._format_diagnostic("MCP process exited")) from None
                    continue
                if msg.get("id") == request_id:
                    if "error" in msg:
                        raise MCPError(self._format_diagnostic(f"MCP error: {msg['error']}"))
                    return msg
                # Server-initiated notifications or responses to other ids.
                unmatched.append(msg)
            raise MCPError(self._format_diagnostic(f"timeout waiting for response id={request_id}"))
        finally:
            for msg in unmatched:
                self._inbox.put(msg)

    def _initialize(self, *, timeout: float) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": _CLIENT_NAME, "version": _CLIENT_VERSION},
            },
            timeout=timeout,
        )
        self._notify("notifications/initialized")

    def list_tools(self, *, timeout: float = _DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
        response = self._request("tools/list", {}, timeout=timeout)
        result = response.get("result") or {}
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPError(f"tools/list returned no tools array: {result!r}")
        return tools

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        response = self._request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
            timeout=timeout,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPError(f"tools/call {name!r} returned no result: {response!r}")
        return result

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines)

    def _format_diagnostic(self, message: str) -> str:
        tail = self.stderr_tail()
        if tail:
            return f"{message}\nstderr tail:\n{tail}"
        return message

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        with contextlib.suppress(Exception):
            self._proc.terminate()
            self._proc.wait(timeout=5.0)
        with contextlib.suppress(Exception):
            self._proc.kill()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
