"""NVIDIA card facts and utilization via bounded nvidia-smi probes.

Every probe runs the tool in a child that ``run_bounded`` kills when it hangs, so a
wedged driver costs one bounded wait and never blocks a caller on a library call.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends.base import UtilSample, run_smi
from lilbee.providers.fleet.gpu_select import PCIVendorID

_TOOL = "nvidia-smi"
_QUERY = "index,utilization.gpu,temperature.gpu,memory.used,memory.total"
_FIELDS = 5
# The query and the parser both read this tuple. The name is the last column: it is the only
# field that can hold a comma, and the row splits at the commas before it only.
_CARD_COLUMNS = ("index", "uuid", "memory.total", "memory.free", "name")
_CARD_QUERY = ",".join(_CARD_COLUMNS)
_COMPUTE_QUERY = "index,compute_cap"
_COMPUTE_FIELDS = 2
_TIMEOUT_S = 5.0
# Coalesce concurrent streams: cache one probe just under the SSE tick interval.
_CACHE_TTL_S = 0.9
# A probe that timed out means the driver is wedged: callers must not each pay the wait again.
_WEDGED_TTL_S = 30.0
_CSV = "--format=csv,noheader,nounits"
_STATS_ARGS = (f"--query-gpu={_QUERY}", _CSV)
_CARD_ARGS = (f"--query-gpu={_CARD_QUERY}", _CSV)
_COMPUTE_ARGS = (f"--query-gpu={_COMPUTE_QUERY}", _CSV)
_CUDA_ARGS = ("-q", "-x")
_CUDA_VERSION = re.compile(r"<cuda_version>\s*(\d+)\.(\d+)\s*</cuda_version>")
# CUDA 12.4 is 12040: the encoding the host answer reads its driver CUDA version in.
CUDA_MAJOR_UNIT = 1000
CUDA_MINOR_UNIT = 10


class Reading(NamedTuple):
    """What one probe printed, and whether it ended by hitting the time limit."""

    out: str
    timed_out: bool


class _Probe:
    """One nvidia-smi query whose answer is shared by every caller within its TTL.

    Not a ``cachetools.TTLCache``: this holds the lock *across* the probe so
    concurrent callers share one result. ``@cached(cache, lock=...)`` releases
    the lock around the call, so three concurrent misses spawn three
    subprocesses (measured). ``test_concurrent_threads_probe_once`` pins it.
    """

    def __init__(self, args: Sequence[str], ttl_s: float) -> None:
        self._args = list(args)
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._at = 0.0
        self._reading = Reading("", False)
        self._primed = False

    def read(self) -> Reading:
        with self._lock:
            if self._primed and time.monotonic() - self._at < self._ttl():
                return self._reading
            started = time.monotonic()
            out = run_smi(_TOOL, self._args, _TIMEOUT_S)
            # run_bounded returns "" at the limit; a tool that fails at once returns "" quickly.
            self._reading = Reading(out, not out and time.monotonic() - started >= _TIMEOUT_S)
            # Stamp after the probe: a probe that takes seconds must not age its own entry.
            self._at = time.monotonic()
            self._primed = True
            return self._reading

    def _ttl(self) -> float:
        """How long the last reading stays valid; only an answer lives for the full TTL."""
        if self._reading.timed_out:
            return _WEDGED_TTL_S
        return self._ttl_s if self._reading.out else min(self._ttl_s, _CACHE_TTL_S)

    def reset(self) -> None:
        with self._lock:
            self._primed = False
            self._reading = Reading("", False)


class SmiCache:
    """Utilization and card count from one shared nvidia-smi probe, parsed once per probe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._probe = _Probe(_STATS_ARGS, _CACHE_TTL_S)
        self._parsed_from: Reading | None = None
        self._value: dict[int, UtilSample] = {}
        self._cards = 0

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
        reading = self._probe.read()
        if reading is self._parsed_from:
            return
        self._value = _parse_smi_output(reading.out)
        self._cards = sum(1 for _ in _rows(reading.out, _FIELDS))
        self._parsed_from = reading

    def reset(self) -> None:
        with self._lock:
            self._probe.reset()
            self._parsed_from = None


_cache = SmiCache()
_cards_probe = _Probe(_CARD_ARGS, _CACHE_TTL_S)
# The compute capability and the driver CUDA version do not change while the process runs;
# an answer stays for the process and a failure is read again.
_compute_probe = _Probe(_COMPUTE_ARGS, math.inf)
_cuda_probe = _Probe(_CUDA_ARGS, math.inf)


@dataclass(frozen=True)
class NvidiaCard:
    """One NVIDIA card as nvidia-smi reports its memory now."""

    index: int
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class _Listed:
    """One row of the card query; a memory figure is ``None`` when nvidia-smi prints [N/A]."""

    index: int
    uuid: str
    name: str
    total_mib: int | None
    free_mib: int | None

    def card(self) -> NvidiaCard | None:
        if self.total_mib is None or self.free_mib is None:
            return None
        return NvidiaCard(
            self.index, self.uuid, self.name, self.total_mib * MIB, self.free_mib * MIB
        )


class NvidiaBackend:
    """CUDA util backend: nvidia-smi with TTL caching."""

    pci_vendor = PCIVendorID.NVIDIA

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        return {i: s for i, s in _cache.stats().items() if i in indices}


def card_count() -> int:
    """Cards the driver shows, from the cached nvidia-smi probe; 0 when it cannot run."""
    return _cache.cards()


def cards() -> list[NvidiaCard]:
    """Each card in driver order; empty when nvidia-smi cannot run or any card's memory is unread.

    A card whose memory did not read is not dropped: dropping it would renumber the
    cards after it, so the whole answer is unknown instead.
    """
    listed = _listed(_cards_probe.read().out)
    found = [card for card in map(_Listed.card, listed) if card is not None]
    return found if len(found) == len(listed) else []


def card_totals() -> list[tuple[str, int]]:
    """``(uuid, total bytes)`` of each card whose total memory reads, in driver order."""
    listed = _listed(_cards_probe.read().out)
    return [(c.uuid, c.total_mib * MIB) for c in listed if c.total_mib is not None]


def host_facts() -> dict[str, Any] | None:
    """The driver CUDA version and each card's compute capability, or ``None`` with no driver.

    ``None`` means nvidia-smi is absent or fails at once. A probe that hit its time limit
    gives every field ``None``: the driver is there and its state is unknown. The
    ``cuda_driver`` field is the CUDA version as ``major * 1000 + minor * 10``.
    """
    listed = _cards_probe.read()
    if listed.timed_out:
        return {"cuda_driver": None, "gpus": None}
    found = _listed(listed.out)
    if not found:
        return None
    caps = _compute_caps(_compute_probe.read().out)
    gpus = [{"name": c.name, "cc": caps[c.index]} for c in found if c.index in caps]
    return {
        "cuda_driver": _cuda_code(_cuda_probe.read().out),
        "gpus": gpus if len(gpus) == len(found) else None,
    }


def _cuda_code(xml: str) -> int | None:
    """The driver CUDA version in ``nvidia-smi -q -x`` output, or ``None``."""
    match = _CUDA_VERSION.search(xml)
    if match is None:
        return None
    return int(match[1]) * CUDA_MAJOR_UNIT + int(match[2]) * CUDA_MINOR_UNIT


def _compute_caps(out: str) -> dict[int, list[int]]:
    """Compute capability ``[major, minor]`` per card index; a card that prints none is absent."""
    caps: dict[int, list[int]] = {}
    for index, parts in _rows(out, _COMPUTE_FIELDS):
        major, _, minor = parts[1].partition(".")
        if major.isdigit() and minor.isdigit():
            caps[index] = [int(major), int(minor)]
    return caps


def _listed(out: str) -> list[_Listed]:
    """Every card row of the card query, whether or not its memory read."""
    return [_listed_row(index, parts) for index, parts in _rows(out, len(_CARD_COLUMNS))]


def _listed_row(index: int, parts: list[str]) -> _Listed:
    """One card row, each field read by its column name in the card query."""
    row = dict(zip(_CARD_COLUMNS, parts, strict=True))
    return _Listed(
        index, row["uuid"], row["name"], _as_int(row["memory.total"]), _as_int(row["memory.free"])
    )


def _as_int(field: str) -> int | None:
    """A CSV field as an int, or ``None`` for nvidia-smi's [N/A]."""
    try:
        return int(field)
    except ValueError:
        return None


def _rows(out: str, fields: int) -> Iterator[tuple[int, list[str]]]:
    """(index, fields) for each CSV row of *fields* columns; a row with no index names no card."""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",", fields - 1)]
        if len(parts) != fields:
            continue
        index = _as_int(parts[0])
        if index is not None:
            yield index, parts


def _parse_smi_output(out: str) -> dict[int, UtilSample]:
    """Parse five-column nvidia-smi CSV into UtilSample per index."""
    samples: dict[int, UtilSample] = {}
    for index, parts in _rows(out, _FIELDS):
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
