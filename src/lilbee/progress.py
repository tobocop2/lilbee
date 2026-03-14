"""Granular progress callback protocol for streaming pipeline events."""

from collections.abc import Callable
from typing import Any

# Callback signature: (event_type: str, data: dict[str, Any]) -> None
DetailedProgressCallback = Callable[[str, dict[str, Any]], None]


def noop_callback(event_type: str, data: dict[str, Any]) -> None:
    """Default no-op callback — discards all events."""
