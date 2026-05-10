"""CPU concurrency policy for compute-bound parallelism.

Use ``cpu_quota()`` to bound thread pools and asyncio semaphores that
schedule CPU-heavy work (chat inference, PDF rasterization, embedding,
tokenization). Capping at half of cpu_count keeps the asyncio main
thread scheduler share so the TUI stays responsive while a worker
storm is in flight.

Do NOT use this for HTTP request concurrency: that lives under
``cfg.crawl_max_concurrent`` and is governed by remote-side rate
limits, not local CPU.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_ENV_VAR = "LILBEE_CPU_QUOTA"


def cpu_quota() -> int:
    """Return the CPU concurrency cap; honors ``LILBEE_CPU_QUOTA`` override.

    Default is ``max(1, os.cpu_count() // 2)``. The override accepts a
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
    return max(1, (os.cpu_count() or 2) // 2)
