"""Apple Metal GPU utilization via ioreg PerformanceStatistics."""

from __future__ import annotations

import re

from lilbee.providers.fleet.gpu_backends.base import UtilSample, run_smi

# llama-server --list-devices emits "MTL0: Apple M1 Pro (21845 MiB, ...)" -> prefix
# "MTL". The build-dependent "Metal" variant is also registered in __init__.py.
BACKEND_KEY = "MTL"

_TOOL = "ioreg"
# Root at the GPU accelerator (AGXAccelerator conforms to IOAccelerator on every
# Apple Silicon generation) and read its properties. A plain "-d 1 -k
# PerformanceStatistics" never reaches the GPU node -- it caps traversal at depth
# 1 and matches a shallow always-zero entry, so the bar read 0% even under load.
_ARGS = ("-r", "-c", "IOAccelerator", "-d", "1")
_TIMEOUT_S = 5.0

# Apple Silicon exposes one integrated GPU; its load is "Device Utilization %".
_UTIL_RE = re.compile(r'"Device Utilization %"=(\d+)')


class AppleBackend:
    """Apple Metal util via ioreg; VRAM stays structural via the orchestrator."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        return _ioreg_samples(indices)


def _ioreg_output() -> str:
    """ioreg PerformanceStatistics stdout, or "" when it can't run."""
    return run_smi(_TOOL, list(_ARGS), _TIMEOUT_S)


def _parse_ioreg(raw: str, indices: frozenset[int]) -> dict[int, UtilSample]:
    """Apply the single Apple GPU's utilization to every requested index."""
    match = _UTIL_RE.search(raw)
    if match is None:
        return {}
    util = int(match.group(1))
    # Apple has no per-card temperature without sudo, and unified-memory totals
    # come from the structural probe, so leave temp/VRAM as the 0 sentinel.
    return {
        index: UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=None,
            free_bytes=0,
            total_bytes=0,
        )
        for index in indices
    }


def _ioreg_samples(indices: frozenset[int]) -> dict[int, UtilSample]:
    return _parse_ioreg(_ioreg_output(), indices)
