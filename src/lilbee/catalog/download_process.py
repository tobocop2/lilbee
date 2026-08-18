"""One model download per child process, so cancelling terminates the child.

hf_xet cancels only at session granularity within a process (one session per
PID), so a terminatable child is what makes per-download cancel real.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from lilbee.catalog.models import CatalogModel
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from lilbee.catalog.download_progress import ProgressCallback

_POLL_INTERVAL_S = 0.2

_EXIT_GRACE_S = 10.0

_PROGRESS_MIN_INTERVAL_S = 0.1

# The child's translated errors, rebuilt in the parent by type name.
_ERRORS_BY_NAME: dict[str, type[Exception]] = {PermissionError.__name__: PermissionError}


@dataclass(frozen=True)
class _Progress:
    """A child's byte counters as it transfers."""

    kind: Literal["progress"]
    downloaded: int
    total: int


@dataclass(frozen=True)
class _Done:
    """A child's success verdict carrying the downloaded model's path."""

    kind: Literal["done"]
    path: str


@dataclass(frozen=True)
class _Failed:
    """A child's failure, serialized because exception objects may not unpickle."""

    kind: Literal["failed"]
    error_type: str
    message: str


_ChildMessage = _Progress | _Done | _Failed


class _Worker(Protocol):
    """The slice of ``multiprocessing.Process`` the parent relay drives."""

    @property
    def exitcode(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


class _PipeProgress:
    """Byte-progress callback that relays over the pipe at ~10 Hz plus the final event."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._last_sent: float | None = None

    def __call__(self, downloaded: int, total: int) -> None:
        now = time.monotonic()
        final = total > 0 and downloaded >= total
        throttled = self._last_sent is not None and now - self._last_sent < _PROGRESS_MIN_INTERVAL_S
        if not final and throttled:
            return
        self._last_sent = now
        self._conn.send(_Progress(kind="progress", downloaded=downloaded, total=total))


def download_in_subprocess(
    entry: CatalogModel,
    models_dir: Path,
    token: str | None,
    *,
    on_progress: ProgressCallback | None,
    cancel: CancelSignal,
) -> Path:
    """Run one download in its own process, relaying progress until it finishes.

    A set *cancel* signal terminates the child, which is the only way to free
    the bandwidth of a running hf_xet transfer mid-flight.
    """
    if cancel.is_set():
        raise TaskCancelledError
    worker, receiver = _start_worker(entry, models_dir, token)
    try:
        return _relay_until_done(entry, worker, receiver, on_progress, cancel)
    finally:
        _stop_worker(worker)
        receiver.close()


def _start_worker(
    entry: CatalogModel, models_dir: Path, token: str | None
) -> tuple[_Worker, Connection]:
    """Spawn the download child; fork is unsafe under the parent's threads."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_run_download_child,
        args=(sender, entry, str(models_dir), token),
        name=f"lilbee-download-{entry.hf_repo}",
    )
    worker.start()
    sender.close()
    return worker, receiver


def _relay_until_done(
    entry: CatalogModel,
    worker: _Worker,
    receiver: Connection,
    on_progress: ProgressCallback | None,
    cancel: CancelSignal,
) -> Path:
    """Forward child messages until its verdict, polling *cancel* between them."""
    while True:
        if cancel.is_set():
            raise TaskCancelledError
        if receiver.poll(_POLL_INTERVAL_S):
            verdict = _apply(receiver.recv(), on_progress)
            if verdict is not None:
                return verdict
        elif not worker.is_alive() and not receiver.poll():
            raise RuntimeError(
                f"Download of {entry.hf_repo} stopped: "
                f"its process exited with code {worker.exitcode}."
            )


def _apply(message: _ChildMessage, on_progress: ProgressCallback | None) -> Path | None:
    """Act on one child message, returning the path once the child reports done."""
    if message.kind == "progress":
        if on_progress is not None:
            on_progress(message.downloaded, message.total)
        return None
    if message.kind == "done":
        return Path(message.path)
    raise _ERRORS_BY_NAME.get(message.error_type, RuntimeError)(message.message)


def _stop_worker(worker: _Worker) -> None:
    """Terminate a live child and reap it, escalating to kill if TERM is ignored."""
    if worker.is_alive():
        worker.terminate()
    worker.join(_EXIT_GRACE_S)
    if worker.is_alive():
        worker.kill()
        worker.join(_EXIT_GRACE_S)


def _run_download_child(
    conn: Connection, entry: CatalogModel, models_dir: str, token: str | None
) -> None:
    """Child-process entry: fetch the files and report the verdict over *conn*."""
    _silence_output()
    # heavy: lilbee.catalog.download (>50ms; huggingface_hub fanout)
    from lilbee.catalog.download import fetch_model_files

    try:
        path = fetch_model_files(entry, Path(models_dir), token, on_progress=_PipeProgress(conn))
        conn.send(_Done(kind="done", path=str(path)))
    except Exception as exc:
        conn.send(_Failed(kind="failed", error_type=type(exc).__name__, message=str(exc)))


def _silence_output() -> None:
    """Point stdout/stderr at devnull; the parent may own a Textual screen."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)
