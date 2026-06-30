"""Shared types for per-vendor GPU utilization backends."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class UtilSample:
    """Live utilization + temperature reading for one GPU index."""

    index: int
    utilization_pct: int | None
    temperature_c: int | None
    # VRAM: optional; populated when the backend's tool also reports memory.
    # 0 is used as sentinel (no data), not "0 bytes used".
    free_bytes: int
    total_bytes: int


class UtilBackend(Protocol):
    """One vendor's live-util probe: given indices, returns per-index samples."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]: ...


def extract_int(obj: object, keys: tuple[str, ...]) -> int | None:
    """Return the first key's value as int, or None."""
    if not isinstance(obj, dict):
        return None
    for key in keys:
        val = obj.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def parse_device_index(key: str) -> int | None:
    """Extract a GPU index from keys like 'card0', 'GPU[0]', '0'."""
    m = re.search(r"\d+", key)
    return int(m.group()) if m else None
