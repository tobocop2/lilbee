"""Generic DRM fdinfo GPU-utilization reader (no root, no vendor tool).

Modern kernels publish per-client GPU engine-busy counters under
``/proc/<pid>/fdinfo/<fd>`` for any DRM driver (i915, xe, amdgpu, ...). Each open
DRM file reports monotonic nanosecond counters per engine::

    drm-driver:	i915
    drm-client-id:	42
    drm-pdev:	0000:00:02.0
    drm-engine-render:	9288864723 ns
    drm-engine-video:	0 ns

Utilization is the busiest engine's share of wall-clock time over a short window:
we snapshot the counters, wait, snapshot again, and divide the delta by the
elapsed time. Counters are summed across every DRM client of the matching driver
on one device, so a workload split across processes still reads correctly. A
client is keyed by its ``drm-pdev`` and ``drm-client-id`` lines and counts once,
however many descriptors (duplicated or inherited by a child) name it.

This needs no elevated privileges (a process can read its own and same-user
clients' fdinfo) and no extra tool, but the engine counters only exist on kernels
new enough to publish them (i915 landed them in 5.19 for execlists submission,
6.5 where GuC submission is active); older kernels omit the
``drm-engine-*`` lines and this reader reports nothing, letting a caller fall back.
A kernel that omits ``drm-client-id`` gets one client for each descriptor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

_PROC = Path("/proc")
_ENGINE_PREFIX = "drm-engine-"
_DRIVER_KEY = "drm-driver"
_CLIENT_ID_KEY = "drm-client-id"
_PDEV_KEY = "drm-pdev"
_DEFAULT_INTERVAL_S = 0.2

# Engine-busy ns per engine, for each device (keyed by drm-pdev; None when unnamed).
_Devices = dict[str | None, dict[str, int]]


class _Client(NamedTuple):
    """One DRM fdinfo entry: its device, client identity and engine counters."""

    pdev: str | None
    ident: str
    engines: dict[str, int]


def read_drm_util(
    driver: str,
    interval_s: float = _DEFAULT_INTERVAL_S,
    proc: Path = _PROC,
    pdev: str | None = None,
) -> int | None:
    """Busiest-engine utilization percent for *driver*'s GPU, or None.

    With *pdev* the number is for that device only. Without it the number is for
    the driver's one device, and None when the driver shows two or more devices.
    Returns None when the kernel publishes no engine counters for this device
    (too old, or no active DRM clients), so the caller can fall back to a tool.
    """
    first = _snapshot(driver, proc)
    if first is None:
        return None
    busy1 = _pick_device(first[0], pdev)
    if busy1 is None:
        return None
    time.sleep(interval_s)
    second = _snapshot(driver, proc)
    if second is None:
        return None
    busy2 = _pick_device(second[0], pdev)
    if busy2 is None:
        return None
    elapsed = second[1] - first[1]
    if elapsed <= 0:
        return None
    peak = 0.0
    for engine, ns2 in busy2.items():
        delta = ns2 - busy1.get(engine, 0)
        if delta > 0:
            peak = max(peak, delta / elapsed)
    return round(min(peak, 1.0) * 100)


def _pick_device(devices: _Devices, pdev: str | None) -> dict[str, int] | None:
    """Counters of the named device, or of the only device when none is named."""
    if pdev is not None:
        return devices.get(pdev)
    if len(devices) > 1:
        return None
    return next(iter(devices.values()))


def _snapshot(driver: str, proc: Path) -> tuple[_Devices, int] | None:
    """Sum engine-busy ns per engine and device across the DRM clients of *driver*.

    Each client counts once. Returns (totals_by_device, monotonic_ns), or None
    when no client publishes engine counters for this driver.
    """
    seen: set[tuple[str | None, str]] = set()
    devices: _Devices = {}
    try:
        pids = [entry for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:
        return None
    for pid in pids:
        for client in _pid_clients(pid, driver):
            key = (client.pdev, client.ident)
            if key in seen:
                continue
            seen.add(key)
            totals = devices.setdefault(client.pdev, {})
            for engine, ns in client.engines.items():
                totals[engine] = totals.get(engine, 0) + ns
    if not devices:
        return None
    return devices, time.monotonic_ns()


def _pid_clients(pid: Path, driver: str) -> list[_Client]:
    """The client of every *driver* DRM fd under this pid that has engine counters."""
    clients: list[_Client] = []
    try:
        entries = list((pid / "fdinfo").iterdir())
    except OSError:
        return clients
    for entry in entries:
        try:
            text = entry.read_text(encoding="utf-8")
        except OSError:
            continue
        if not _driver_matches(text, driver):
            continue
        engines = _engine_ns(text)
        if engines:
            client_id = _value(text, _CLIENT_ID_KEY)
            ident = f"fd:{pid.name}/{entry.name}" if client_id is None else f"id:{client_id}"
            clients.append(_Client(_value(text, _PDEV_KEY), ident, engines))
    return clients


def _engine_ns(text: str) -> dict[str, int]:
    """Engine-busy ns by engine name from one fdinfo text."""
    engines: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith(_ENGINE_PREFIX):
            engine, ns = _parse_engine_line(line)
            if ns is not None:
                engines[engine] = ns
    return engines


def _value(text: str, key: str) -> str | None:
    """Value of the fdinfo line named *key*, or None when the line is absent."""
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            return line.partition(":")[2].strip()
    return None


def _driver_matches(text: str, driver: str) -> bool:
    """True when the fdinfo names *driver* on its drm-driver line."""
    return _value(text, _DRIVER_KEY) == driver


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
