"""Shared types and helpers for per-vendor GPU utilization backends."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_SMI_TIMEOUT_S = 5.0


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


def run_smi(tool: str, args: Sequence[str], timeout: float = _SMI_TIMEOUT_S) -> str:
    """Resolve tool via shutil.which, run it, return stdout on rc==0 else "".

    Returns "" when the tool is not found, exits non-zero, or raises.
    """
    binary = shutil.which(tool) or tool
    try:
        proc = subprocess.run(  # noqa: S603 - caller supplies fixed args from named constants
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def extract_int(obj: object, keys: tuple[str, ...]) -> int | None:
    """Return the first key's value coerced to int, or None.

    Accepts integer, float, and decimal-string values (e.g. "35.0" from rocm-smi).
    """
    if not isinstance(obj, dict):
        return None
    for key in keys:
        val = obj.get(key)
        if val is not None:
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
    return None


def parse_device_index(key: str) -> int | None:
    """Extract a GPU index from keys like 'card0', 'GPU[0]', '0'."""
    m = re.search(r"\d+", key)
    return int(m.group()) if m else None


def _coerce_metric(val: object) -> int | None:
    """Coerce a metric value to int: a bare number, a decimal string, or {"value": N}."""
    if isinstance(val, dict):
        val = val.get("value")
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            return None
    return None


def find_metric(obj: object, keys: tuple[str, ...]) -> int | None:
    """Find the first of *keys* anywhere in a nested dict and coerce it to int.

    SMI tools nest the same reading differently across versions -- flat
    (``{"gfx_activity": 45}``), value-wrapped (``{"gfx_activity": {"value": 45}}``),
    or under a block (``{"usage": {"gfx_activity": {"value": 45}}}``). Searching by
    key at any depth reads all three without hard-coding one layout.
    """
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj:
            found = _coerce_metric(obj[key])
            if found is not None:
                return found
    for val in obj.values():
        if isinstance(val, dict):
            found = find_metric(val, keys)
            if found is not None:
                return found
    return None
