"""What a run started and did not clean up: processes, temp entries and listening sockets."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

POLL_SECONDS = 0.05
GRACE_SECONDS = 2.0
STOP_WAIT_SECONDS = 3.0
LISTEN = psutil.CONN_LISTEN
UNKNOWN_PORTS = (-1,)
# What a read of another process can raise. On macOS psutil lets a refused sysctl out as a
# SystemError or an OSError instead of its own AccessDenied.
UNREADABLE = (psutil.Error, OSError, SystemError)
TEMP_VARIABLES = ("TMPDIR", "TEMP", "TMP")


@dataclass(frozen=True)
class ProcessLeft:
    """One process of the run that still lives after the run ended."""

    pid: int
    name: str
    listening_ports: tuple[int, ...]


@dataclass(frozen=True)
class LeftBehind:
    """Everything one run left."""

    processes: tuple[ProcessLeft, ...] = ()
    temp_entries: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        """True when the run left nothing."""
        return not self.processes and not self.temp_entries


@dataclass(frozen=True)
class _Identity:
    pid: int
    created: float


def _identity(process: psutil.Process) -> _Identity | None:
    try:
        return _Identity(process.pid, process.create_time())
    except UNREADABLE:
        return None


def _all_processes() -> set[_Identity]:
    return {found for process in psutil.process_iter() if (found := _identity(process)) is not None}


def _listening_ports(process: psutil.Process) -> tuple[int, ...]:
    """The ports *process* listens on; ``UNKNOWN_PORTS`` when the system refuses the read."""
    try:
        connections = process.net_connections(kind="inet")
    except UNREADABLE:
        return UNKNOWN_PORTS
    return tuple(sorted({conn.laddr.port for conn in connections if conn.status == LISTEN}))


def _mentions(process: psutil.Process, temp_root: Path) -> bool:
    """Whether the process inherited the run's environment or names its temp root in its command."""
    try:
        if process.environ().get(TEMP_VARIABLES[0]) == str(temp_root):
            return True
    except UNREADABLE:
        pass  # the command line below is the second way to know
    try:
        return str(temp_root) in " ".join(process.cmdline())
    except UNREADABLE:
        return False


@dataclass
class Tracker:
    """Follows one run: its own temp root and the descendants of its process.

    A process counts as the run's own only when it did not exist at the baseline and one of
    three things holds: it was seen as a descendant of the run, it inherited the run's temp
    variable, or its command line names the run's temp root.
    """

    temp_root: Path
    _baseline: set[_Identity] = field(default_factory=_all_processes)
    _descendants: set[_Identity] = field(default_factory=set)
    _stop: threading.Event = field(default_factory=threading.Event)
    _poller: threading.Thread | None = None

    def environment(self) -> dict[str, str]:
        """The variables that point a child's temp files at this run's temp root."""
        self.temp_root.mkdir(parents=True, exist_ok=True)
        return dict.fromkeys(TEMP_VARIABLES, str(self.temp_root))

    def watch(self, pid: int) -> None:
        """Record every descendant of *pid* until :meth:`collect` is called."""
        self._poller = threading.Thread(target=self._poll, args=(pid,), daemon=True)
        self._poller.start()

    def _poll(self, pid: int) -> None:
        while not self._stop.is_set():
            try:
                children = psutil.Process(pid).children(recursive=True)
            except UNREADABLE:
                children = []
            self._descendants.update(
                found for child in children if (found := _identity(child)) is not None
            )
            time.sleep(POLL_SECONDS)

    def _own_survivors(self) -> list[psutil.Process]:
        survivors: list[psutil.Process] = []
        for process in psutil.process_iter():
            identity = _identity(process)
            if identity is None or identity in self._baseline:
                continue
            if identity in self._descendants or _mentions(process, self.temp_root):
                survivors.append(process)
        return survivors

    def collect(self, grace_seconds: float) -> LeftBehind:
        """What is left after the run's process exited; the run's own survivors are then stopped."""
        self._stop.set()
        if self._poller is not None:
            self._poller.join()
        time.sleep(grace_seconds)
        survivors = self._own_survivors()
        left = tuple(
            ProcessLeft(process.pid, _name(process), _listening_ports(process))
            for process in survivors
        )
        temp_entries = tuple(sorted(entry.name for entry in self.temp_root.iterdir()))
        _stop_processes(survivors)
        return LeftBehind(left, temp_entries)


def _name(process: psutil.Process) -> str:
    try:
        return str(process.name())
    except UNREADABLE:
        return "unknown"


def _stop_processes(processes: list[psutil.Process]) -> None:
    """Terminate, then kill, exactly the processes given."""
    for process in processes:
        try:
            process.terminate()
        except UNREADABLE:
            continue
    _, alive = psutil.wait_procs(processes, timeout=STOP_WAIT_SECONDS)
    for process in alive:
        try:
            process.kill()
        except UNREADABLE:
            continue
