"""CUDA GPU utilization via nvidia-smi."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends.base import UtilSample, run_smi
from lilbee.providers.fleet.gpu_select import PCIVendorID

_TOOL = "nvidia-smi"
_QUERY = "index,utilization.gpu,temperature.gpu,memory.used,memory.total"
_FIELDS = 5
_TIMEOUT_S = 5.0
# Coalesce concurrent streams: cache one probe just under the SSE tick interval.
_CACHE_TTL_S = 0.9


class SmiCache:
    """Shares one nvidia-smi probe across concurrent callers within the TTL.

    Not a ``cachetools.TTLCache``: this holds the lock *across* the probe so
    concurrent callers share one result. ``@cached(cache, lock=...)`` releases
    the lock around the call, so three concurrent misses spawn three
    subprocesses (measured). ``test_concurrent_threads_probe_once`` pins it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at = 0.0
        self._value: dict[int, UtilSample] = {}
        self._cards = 0
        self._primed = False

    def stats(self) -> dict[int, UtilSample]:
        with self._lock:
            self._refresh()
            return self._value

    def cards(self) -> int:
        """Rows with an index in the last probe, whether or not their memory read."""
        with self._lock:
            self._refresh()
            return self._cards

    def _refresh(self) -> None:
        if self._primed and time.monotonic() - self._at < _CACHE_TTL_S:
            return
        out = _smi_output()
        self._value = _parse_smi_output(out)
        self._cards = sum(1 for _ in _rows(out))
        # Stamp after the probe: a probe that takes seconds must not age its own entry.
        self._at = time.monotonic()
        self._primed = True

    def reset(self) -> None:
        with self._lock:
            self._primed = False


_cache = SmiCache()


class NvidiaBackend:
    """CUDA util backend: nvidia-smi with TTL caching."""

    pci_vendor = PCIVendorID.NVIDIA

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        return {i: s for i, s in _cache.stats().items() if i in indices}


def card_count() -> int:
    """Cards the driver shows, from the cached nvidia-smi probe; 0 when it cannot run."""
    return _cache.cards()


def _smi_output() -> str:
    """nvidia-smi CSV stdout, or "" when it can't run."""
    return run_smi(_TOOL, [f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"], _TIMEOUT_S)


def _as_int(field: str) -> int | None:
    """A CSV field as an int, or ``None`` for nvidia-smi's [N/A]."""
    try:
        return int(field)
    except ValueError:
        return None


def _rows(out: str) -> Iterator[tuple[int, list[str]]]:
    """(index, fields) for each full CSV row; a row with no index names no card."""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != _FIELDS:
            continue
        index = _as_int(parts[0])
        if index is not None:
            yield index, parts


def _parse_smi_output(out: str) -> dict[int, UtilSample]:
    """Parse five-column nvidia-smi CSV into UtilSample per index."""
    samples: dict[int, UtilSample] = {}
    for index, parts in _rows(out):
        # Each column degrades on its own. nvidia-smi prints [N/A] for
        # utilization on some cards and inside some VMs, and dropping the row for
        # it threw away that card's memory reading too, so a GPU reporting its
        # VRAM perfectly well disappeared from the panel entirely.
        util = _as_int(parts[1])
        temperature = _as_int(parts[2])
        used_mib = _as_int(parts[3])
        total_mib = _as_int(parts[4])
        if used_mib is None or total_mib is None:
            continue
        total = total_mib * MIB
        samples[index] = UtilSample(
            index=index,
            utilization_pct=util,
            temperature_c=temperature,
            free_bytes=max(total - used_mib * MIB, 0),
            total_bytes=total,
        )
    return samples
