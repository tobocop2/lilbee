"""Shared test helpers for server tests."""

from __future__ import annotations

import json


def parse_sse_events(body: bytes) -> list[tuple[str, dict]]:
    """Parse raw SSE bytes into a list of (event_type, data_dict) tuples."""
    events = []
    current_event = ""
    current_data = ""
    for line in body.decode().split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "" and current_event:
            events.append((current_event, json.loads(current_data)))
            current_event = ""
            current_data = ""
    return events
