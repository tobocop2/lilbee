"""Generic DRM fdinfo GPU-utilization reader (no root, no vendor tool).

Modern kernels publish per-client GPU engine-busy counters under
``/proc/<pid>/fdinfo/<fd>`` for any DRM driver (i915, xe, amdgpu, ...). Each open
DRM file reports monotonic nanosecond counters per engine::

    drm-driver:	i915
    drm-pdev:	0000:00:02.0
    drm-engine-render:	9288864723 ns
    drm-engine-video:	0 ns

Utilization is the busiest engine's share of wall-clock time over a short window:
we snapshot the counters, wait, snapshot again, and divide the delta by the
elapsed time. Counters are summed across every DRM client of the matching driver,
so a workload split across processes still reads correctly.

This needs no elevated privileges (a process can read its own and same-user
clients' fdinfo) and no extra tool, but the engine counters only exist on kernels
new enough to publish them (i915 landed them in ~6.2); older kernels omit the
``drm-engine-*`` lines and this reader reports nothing, letting a caller fall back.
"""

from __future__ import annotations

import time
from pathlib import Path

_PROC = Path("/proc")
_ENGINE_PREFIX = "drm-engine-"
_DRIVER_KEY = "drm-driver"
_DEFAULT_INTERVAL_S = 0.2


def read_drm_util(
    driver: str,
    interval_s: float = _DEFAULT_INTERVAL_S,
    proc: Path = _PROC,
) -> int | None:
    """Busiest-engine utilization percent for *driver*'s GPU, or None.

    Returns None when the kernel publishes no engine counters for this driver
    (too old, or no active DRM clients), so the caller can fall back to a tool.
    """
    first = _snapshot(driver, proc)
    if first is None:
        return None
    time.sleep(interval_s)
    second = _snapshot(driver, proc)
    if second is None:
        return None
    busy1, wall1 = first
    busy2, wall2 = second
    elapsed = wall2 - wall1
    if elapsed <= 0:
        return None
    peak = 0.0
    for engine, ns2 in busy2.items():
        delta = ns2 - busy1.get(engine, 0)
        if delta > 0:
            peak = max(peak, delta / elapsed)
    return round(min(peak, 1.0) * 100)


def _snapshot(driver: str, proc: Path) -> tuple[dict[str, int], int] | None:
    """Sum engine-busy ns per engine across all DRM clients of *driver*.

    Returns (totals_by_engine, monotonic_ns), or None when no client publishes
    engine counters for this driver.
    """
    totals: dict[str, int] = {}
    found = False
    try:
        pids = [entry for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:
        return None
    for pid in pids:
        for engine, ns in _client_engine_ns(pid, driver):
            totals[engine] = totals.get(engine, 0) + ns
            found = True
    if not found:
        return None
    return totals, time.monotonic_ns()


def _client_engine_ns(pid: Path, driver: str) -> list[tuple[str, int]]:
    """(engine, ns) pairs from every *driver* DRM client fd under this pid."""
    pairs: list[tuple[str, int]] = []
    try:
        entries = list((pid / "fdinfo").iterdir())
    except OSError:
        return pairs
    for entry in entries:
        try:
            text = entry.read_text()
        except OSError:
            continue
        if not _driver_matches(text, driver):
            continue
        for line in text.splitlines():
            if line.startswith(_ENGINE_PREFIX):
                engine, ns = _parse_engine_line(line)
                if ns is not None:
                    pairs.append((engine, ns))
    return pairs


def _driver_matches(text: str, driver: str) -> bool:
    """True when the fdinfo names *driver* on its drm-driver line."""
    for line in text.splitlines():
        if line.startswith(_DRIVER_KEY):
            _, _, val = line.partition(":")
            return val.strip() == driver
    return False


def _parse_engine_line(line: str) -> tuple[str, int | None]:
    """Parse 'drm-engine-render:\\t123 ns' into ('render', 123)."""
    key, _, val = line.partition(":")
    engine = key[len(_ENGINE_PREFIX) :]
    fields = val.split()
    if not fields:
        return engine, None
    try:
        return engine, int(fields[0])
    except ValueError:
        return engine, None
