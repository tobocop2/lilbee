"""HTTP + SSE driver helpers for the QA matrix.

Wraps `httpx` for sync HTTP and `httpx-sse` for parsing event streams emitted
by `/api/ask/stream`, `/api/chat/stream`, `/api/sync`, `/api/wiki/build`, etc.
Streaming-output assertions live here, not in the TUI driver — see plan.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass

import httpx
from httpx_sse import EventSource

_DEFAULT_TIMEOUT = 60.0
_HEALTH_POLL_INTERVAL = 0.2


@dataclass
class SSEEvent:
    """One server-sent event captured from an SSE stream."""

    event: str
    data: str
    id: str | None
    retry: int | None


def find_free_port() -> int:
    """Bind to ephemeral port 0 and return what the OS gave us. Race-free."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_health(url: str, *, timeout: float = 30.0) -> None:
    """Poll a health endpoint until it returns 200 or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            last_err = exc
        else:
            if response.status_code == httpx.codes.OK:
                return
            last_err = httpx.HTTPStatusError(
                f"unexpected status {response.status_code}",
                request=response.request,
                response=response,
            )
        time.sleep(_HEALTH_POLL_INTERVAL)
    raise TimeoutError(
        f"health endpoint {url!r} not ready within {timeout}s; last error: {last_err}"
    )


def stream_sse(
    method: str,
    url: str,
    *,
    json: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Iterator[SSEEvent]:
    """Yield SSEEvent objects for the duration of one request.

    Caller is responsible for collecting / asserting; events stop when the
    server closes the stream or `timeout` elapses.
    """
    with (
        httpx.Client(timeout=timeout) as client,
        client.stream(method, url, json=json, headers=headers) as response,
    ):
        response.raise_for_status()
        source = EventSource(response)
        for event in source.iter_sse():
            yield SSEEvent(
                event=event.event,
                data=event.data,
                id=event.id,
                retry=event.retry,
            )


def collect_event_types(events: Iterator[SSEEvent]) -> list[str]:
    """Drain a stream and return only the event-type names, in order.

    Use for assertions like:
        assert "PROGRESS" in collect_event_types(stream)
        assert collect_event_types(stream)[-1] == "DONE"
    """
    return [event.event for event in events]
