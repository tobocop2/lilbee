"""CUDA GPU utilization via nvidia-smi."""

from __future__ import annotations

import threading
import time

from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends.base import UtilSample, run_smi

_TOOL = "nvidia-smi"
_QUERY = "index,utilization.gpu,memory.used,memory.total"
_FIELDS = 4
_TIMEOUT_S = 5.0
# Coalesce concurrent streams: cache one probe just under the SSE tick interval.
_CACHE_TTL_S = 0.9


class SmiCache:
    """Shares one nvidia-smi probe across concurrent callers within the TTL."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at = 0.0
        self._value: dict[int, UtilSample] = {}
        self._primed = False

    def stats(self) -> dict[int, UtilSample]:
        with self._lock:
            now = time.monotonic()
            if not self._primed or now - self._at >= _CACHE_TTL_S:
                self._value = _parse_smi_output(_smi_output())
                self._at = now
                self._primed = True
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._primed = False


_cache = SmiCache()


class NvidiaBackend:
    """CUDA util backend: nvidia-smi with TTL caching."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        return {i: s for i, s in _cache.stats().items() if i in indices}


def _smi_output() -> str:
    """nvidia-smi CSV stdout, or "" when it can't run."""
    return run_smi(_TOOL, [f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"], _TIMEOUT_S)


def _parse_smi_output(out: str) -> dict[int, UtilSample]:
    """Parse four-column nvidia-smi CSV into UtilSample per index."""
    samples: dict[int, UtilSample] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != _FIELDS:
            continue
        try:
            index, util, used_mib, total_mib = (int(p) for p in parts)
        except ValueError:
            continue
        total = total_mib * MIB
        samples[index] = UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=None,  # four-column query omits temperature
            free_bytes=max(total - used_mib * MIB, 0),
            total_bytes=total,
        )
    return samples
