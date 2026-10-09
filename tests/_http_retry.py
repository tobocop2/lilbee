"""A GET that retries the transport, never the answer, for tests that call a live host."""

from __future__ import annotations

import time

import httpx

ATTEMPTS = 4
PAUSE_SECONDS = 2.0


def get_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Response:
    """GET *url*, retrying a connection failure or a 5xx up to ``ATTEMPTS`` times.

    A 4xx answer returns at once, so the caller still sees a repo that is gone.
    The last failure is raised or returned when every attempt fails.
    """
    with httpx.Client(transport=transport, timeout=timeout, headers=headers) as client:
        for attempt in range(1, ATTEMPTS + 1):
            last = attempt == ATTEMPTS
            try:
                resp = client.get(url)
            except httpx.TransportError:
                if last:
                    raise
            else:
                if resp.status_code < httpx.codes.INTERNAL_SERVER_ERROR or last:
                    return resp
            time.sleep(PAUSE_SECONDS * attempt)
    raise AssertionError("unreachable: the last attempt returns or raises")
