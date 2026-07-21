"""What actually ran: machines, versions, timings, throughput, cost.

A benchmark that publishes numbers without publishing the machine that produced
them is asking to be taken on trust. This captures the run's physical facts on
the pod itself, while it runs, so the report states them rather than someone
reconstructing them afterwards from memory and shell history.

Everything is read from the live system rather than declared in advance. A
config file states an intention; ``nvidia-smi`` states what the job actually had.
The two diverge exactly when it matters: a pod that provisioned with fewer GPUs
than requested, a driver that differs from the image's, a stage that took three
times its estimate.

Timing is a context manager rather than a pair of timestamps, because a
hand-written ``t1 - t0`` drifts from the thing it claims to measure the moment
anybody edits the block between them. While a stage is open a sampler records
GPU utilisation and memory in the background, so the record can say whether four
hours were spent computing or waiting on I/O -- which is the difference between
a slow model and a badly-shaped pipeline, and is invisible in a duration alone.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# RunPod hourly rates are not discoverable from inside the container, so the
# launcher that knows the price passes it in. Recorded rather than estimated: a
# benchmark's cost claim should be what was billed.
HOURLY_RATE_ENV = "LILBEE_POD_HOURLY_USD"
POD_ID_ENV = "RUNPOD_POD_ID"

# How often the background sampler reads the GPUs. Fast enough to catch a stall
# in a multi-minute stage, slow enough that the sampling itself is free.
SAMPLE_INTERVAL_SECONDS = 5.0

# getrusage reports ru_maxrss in KiB on Linux and in bytes on macOS. The pods are
# Linux, but a developer checking the harness on a laptop would otherwise read a
# number a thousand times too large and have no way to tell, so the unit is
# resolved rather than assumed.
_RSS_PER_GIB = 1024**2 if platform.system() == "Linux" else 1024**3


@dataclass(frozen=True)
class GPU:
    """One accelerator as the driver reports it."""

    index: int
    name: str
    memory_mib: int
    driver: str


@dataclass(frozen=True)
class Machine:
    """The box the work ran on."""

    host: str
    platform: str
    python: str
    cpu_count: int
    memory_gib: float
    gpus: list[GPU] = field(default_factory=list)
    pod_id: str = ""

    @property
    def gpu_summary(self) -> str:
        """One line, as a reader wants it in a table."""
        if not self.gpus:
            return "none (CPU-only)"
        return f"{len(self.gpus)}x {self.gpus[0].name} ({self.gpus[0].memory_mib // 1024} GiB each)"


def _query_gpus() -> list[GPU]:
    """Read the GPUs from the driver, or none if there is no driver.

    Absence is an empty list, not an error: hydration runs on a CPU box on
    purpose, and a record that refused to exist without a GPU would be unusable
    for the stage where cheapness is the whole point.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus: list[GPU] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            GPU(index=int(parts[0]), name=parts[1], memory_mib=int(parts[2]), driver=parts[3])
        )
    return gpus


def _sample_gpus() -> tuple[list[float], list[float]]:
    """One instantaneous (utilisation %, memory-used MiB) reading per GPU."""
    if shutil.which("nvidia-smi") is None:
        return [], []
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return [], []
    utilisation: list[float] = []
    memory: list[float] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        utilisation.append(float(parts[0]))
        memory.append(float(parts[1]))
    return utilisation, memory


def describe_machine() -> Machine:
    """The current box, read live."""
    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return Machine(
        host=platform.node(),
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        memory_gib=round(memory_bytes / (1024**3), 1),
        gpus=_query_gpus(),
        pod_id=os.environ.get(POD_ID_ENV, ""),
    )


class _GpuSampler:
    """Polls GPU utilisation on a background thread while a stage is open."""

    def __init__(self, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.utilisation: list[float] = []
        self.memory_mib: list[float] = []

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            utilisation, memory = _sample_gpus()
            if utilisation:
                # Mean across GPUs per sample: a data-parallel embed should keep
                # every card busy, and an average that sits well below the peak
                # is the signature of one card doing the work.
                self.utilisation.append(sum(utilisation) / len(utilisation))
                self.memory_mib.append(max(memory))

    def __enter__(self) -> _GpuSampler:
        if shutil.which("nvidia-smi") is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)


@dataclass
class Stage:
    """One timed step, with enough context to make the duration mean something.

    ``documents`` and ``bytes_out`` are what turn a number into a rate: four
    hours says nothing until it is four hours for nine million passages.
    """

    name: str
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    documents: int = 0
    bytes_out: int = 0
    peak_rss_gib: float = 0.0
    gpu_utilisation_mean: float = 0.0
    gpu_utilisation_peak: float = 0.0
    gpu_memory_peak_mib: float = 0.0
    command: str = ""
    notes: str = ""

    @property
    def documents_per_second(self) -> float:
        return self.documents / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def mib_per_second(self) -> float:
        return (self.bytes_out / 1024**2) / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def cpu_utilisation(self) -> float:
        """CPU seconds over wall seconds; below 1.0 on one core means waiting."""
        return self.cpu_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def cost_usd(self, hourly_rate: float) -> float:
        return hourly_rate * self.wall_seconds / 3600.0

    def to_dict(self, hourly_rate: float) -> dict[str, Any]:
        return {
            **asdict(self),
            "wall_seconds": round(self.wall_seconds, 3),
            "cpu_seconds": round(self.cpu_seconds, 3),
            "documents_per_second": round(self.documents_per_second, 1),
            "mib_per_second": round(self.mib_per_second, 1),
            "cpu_utilisation": round(self.cpu_utilisation, 2),
            "cost_usd": round(self.cost_usd(hourly_rate), 2),
        }


@dataclass
class RunProvenance:
    """The physical record of one hydration, ingest, or benchmark run."""

    stage_group: str
    machine: Machine
    hourly_rate_usd: float = 0.0
    stages: list[Stage] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, *, documents: int = 0, command: str = "", notes: str = ""):
        """Time a block, sampling the GPUs throughout, and record the result.

        Yields the ``Stage`` so the body can fill in what it could not know up
        front, most often ``bytes_out`` and a corrected ``documents`` once the
        work has actually counted them.
        """
        entry = Stage(name=name, documents=documents, command=command, notes=notes)
        before_cpu = resource.getrusage(resource.RUSAGE_SELF)
        started = time.perf_counter()
        with _GpuSampler() as sampler:
            try:
                yield entry
            finally:
                # Full precision here, rounded only on the way out. Rounding at
                # capture turns a sub-millisecond stage into exactly 0.0, which
                # then reads as "took no time and cost nothing" rather than
                # "was fast" -- and a stage that died immediately would record
                # itself as free.
                entry.wall_seconds = time.perf_counter() - started
                after_cpu = resource.getrusage(resource.RUSAGE_SELF)
                entry.cpu_seconds = (after_cpu.ru_utime - before_cpu.ru_utime) + (
                    after_cpu.ru_stime - before_cpu.ru_stime
                )
                entry.peak_rss_gib = round(after_cpu.ru_maxrss / _RSS_PER_GIB, 2)
                if sampler.utilisation:
                    entry.gpu_utilisation_mean = round(
                        sum(sampler.utilisation) / len(sampler.utilisation), 1
                    )
                    entry.gpu_utilisation_peak = round(max(sampler.utilisation), 1)
                    entry.gpu_memory_peak_mib = round(max(sampler.memory_mib), 1)
                self.stages.append(entry)

    @property
    def total_seconds(self) -> float:
        return sum(stage.wall_seconds for stage in self.stages)

    @property
    def total_cost_usd(self) -> float:
        return self.hourly_rate_usd * self.total_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_type": "provenance",
            "stage_group": self.stage_group,
            "machine": asdict(self.machine),
            "gpu_summary": self.machine.gpu_summary,
            "hourly_rate_usd": self.hourly_rate_usd,
            "total_seconds": round(self.total_seconds, 1),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "stages": [stage.to_dict(self.hourly_rate_usd) for stage in self.stages],
        }

    def write(self, path: Path) -> None:
        """Append this record to the JSONL the report reads.

        Appended, not overwritten: hydration and ingest run on different
        machines and the report wants both, so each adds a line rather than the
        second machine erasing the first.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(self.to_dict()) + "\n")


def start(stage_group: str) -> RunProvenance:
    """Begin a provenance record on the current machine."""
    rate = os.environ.get(HOURLY_RATE_ENV, "").strip()
    return RunProvenance(
        stage_group=stage_group,
        machine=describe_machine(),
        hourly_rate_usd=float(rate) if rate else 0.0,
    )
