"""CPU concurrency policy for compute-bound parallelism.

Use ``cpu_quota()`` to bound thread pools and asyncio semaphores that
schedule CPU-heavy work (chat inference, PDF rasterization, embedding,
tokenization). Capping at half of cpu_count keeps the asyncio main
thread scheduler share so the TUI stays responsive while a worker
storm is in flight.

Do NOT use this for HTTP request concurrency: that lives under
``cfg.crawl_max_concurrent`` and is governed by remote-side rate
limits, not local CPU.

``engine_thread_count()`` is the out-of-process counterpart, sizing a
spawned llama-server's thread count.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from pathlib import Path

log = logging.getLogger(__name__)

_ENV_VAR = "LILBEE_CPU_QUOTA"

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_cpu_quota(root: Path = _CGROUP_ROOT) -> int | None:
    """CPUs the CFS quota allows, or None when unlimited or unreadable.

    cgroup v2 keeps ``<quota> <period>`` (or ``max`` for unlimited) in
    ``cpu.max``; v1 splits it across ``cpu.cfs_quota_us`` (-1 = unlimited) and
    ``cpu.cfs_period_us``. The effective core count is ``ceil(quota / period)``.
    """
    try:
        v2 = (root / "cpu.max").read_text().split()
    except OSError:
        v2 = []
    if v2:
        if v2[0] == "max":
            return None
        try:
            quota, period = int(v2[0]), int(v2[1])
        except (ValueError, IndexError):
            return None
        return max(1, math.ceil(quota / period)) if period > 0 else None
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def available_cpu_count() -> int:
    """Usable CPUs for this process, honoring cgroup quota and CPU affinity.

    ``os.cpu_count()`` reports the host's cores, which over-reports inside a
    CPU-limited container (a rented multi-vCPU box hands a pod a fraction of the
    machine). Fold in the process's scheduling affinity and the cgroup CFS quota
    so a container-bound run sizes to its real budget rather than the host's.
    Always at least 1.

    ``os.process_cpu_count()`` folds these in for us, but it landed in 3.13 and
    the project floor is 3.11, so the cgroup read is done by hand here.
    """
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            limits.append(len(os.sched_getaffinity(0)))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        limits.append(quota)
    return max(1, min(limits))


def cpu_quota() -> int:
    """Return the CPU concurrency cap; honors ``LILBEE_CPU_QUOTA`` override.

    Default is ``max(1, available_cpu_count() // 2)``. The override accepts a
    positive integer; non-positive or unparseable values fall back to
    the default and a warning is logged once per call.
    """
    override = os.environ.get(_ENV_VAR)
    if override is not None:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass  # bad override falls through to the warning + default below
        log.warning(
            "Ignoring %s=%r: must be a positive integer; using default.",
            _ENV_VAR,
            override,
        )
    return max(1, available_cpu_count() // 2)


def engine_thread_count() -> int | None:
    """Threads for a spawned engine, or None to keep the engine's own default.

    llama.cpp sizes its default from host topology (physical cores, minus
    efficiency cores) and cannot see a CFS quota or an affinity mask, so inside a
    CPU-limited container it starts several times more threads than the quota
    admits and generation slows down: 33.75 tok/s at 8 threads against 13.45 at
    32, on a pod with 96 visible cores and a 7.65-core quota. Where nothing caps
    this process, upstream's count is the better informed one and stays.

    Not ``cpu_quota()``: that halves the budget to leave the asyncio scheduler a
    share, which an out-of-process engine does not compete with.
    """
    quota = available_cpu_count()
    return quota if quota < (os.cpu_count() or 1) else None
