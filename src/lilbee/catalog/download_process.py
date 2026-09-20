"""One model download per child process, so cancelling terminates the child.

hf_xet cancels only at session granularity within a process (one session per
PID), so a terminatable child is what makes per-download cancel real, and it
is also the only stop a wedged transfer cannot refuse.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from lilbee.catalog.download_progress import ProgressCallback
from lilbee.catalog.models import CatalogModel
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

log = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.2

_EXIT_GRACE_S = 10.0

_PROGRESS_MIN_INTERVAL_S = 0.1

STALL_FLOOR_BYTES = 256 * 1024
"""Minimum new bytes a child must report for its transfer to count as alive.

A wedged connection can trickle a few bytes a minute, which an any-activity
check reads as progress; ~2 KB/s is far below any usable model download."""

STALL_ATTEMPTS = 3
"""Children the parent spawns for one download before it reports a stall."""

_STALL_DEADLINE_S = 90.0
"""Seconds of no reported progress after which the child is terminated.

Well past the hub's own 10 second read timeout and the retries it makes under
it, so a child only expires on a transfer those mechanisms cannot wake."""

_STARTUP_DEADLINE_S = 120.0
"""Seconds a child may stay silent before its first bytes.

It spawns, imports huggingface_hub, lists the repo, and resolves the
filename, the dominant term, bounded by lilbee's own 30 second listing
timeout and 10 second header-probe timeout, not the hub's. Every file the
child then resolves restarts this clock, so the budget does not grow with
the shard count."""

# The child's translated errors, rebuilt in the parent by type name.
_ERRORS_BY_NAME: dict[str, type[Exception]] = {PermissionError.__name__: PermissionError}


@dataclass(frozen=True)
class _Progress:
    """A child's byte counters as it transfers."""

    kind: Literal["progress"]
    downloaded: int
    total: int


@dataclass(frozen=True)
class _Probed:
    """A child's note that it resolved one file with the Hub before transferring.

    *blob* names the cache file that transfer writes, so the parent knows which
    temporary files the child owns and can clear them after terminating it.
    """

    kind: Literal["probed"]
    blob: str | None


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


_ChildMessage = _Progress | _Probed | _Done | _Failed


class _ChildStalledError(Exception):
    """A child stopped reporting bytes, so its transfer is retried in a new child."""


class _StallDeadline:
    """Times a child against the bytes it reports, so a wedged transfer expires."""

    def __init__(
        self,
        deadline_s: float = _STALL_DEADLINE_S,
        floor_bytes: int = STALL_FLOOR_BYTES,
        startup_s: float = _STARTUP_DEADLINE_S,
    ) -> None:
        self._deadline_s = deadline_s
        self._floor_bytes = floor_bytes
        self._startup_s = startup_s
        self._since = time.monotonic()
        self._base = 0
        self._moving = False

    def saw(self, downloaded: int) -> None:
        """Restart the clock once the byte floor of new data has arrived."""
        if downloaded - self._base < self._floor_bytes:
            return
        self._base = downloaded
        self._since = time.monotonic()
        self._moving = True

    def probed(self) -> None:
        """Restart the clock for a child that resolved a file before its first bytes."""
        self._since = time.monotonic()

    def expired(self) -> bool:
        """Whether the child has reported nothing for the budget of its phase."""
        budget = self._deadline_s if self._moving else self._startup_s
        return time.monotonic() - self._since >= budget


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


class _PipeProbe:
    """Callback that tells the parent one file was resolved with the Hub."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def __call__(self, blob: str | None) -> None:
        self._conn.send(_Probed(kind="probed", blob=blob))


def download_in_subprocess(
    entry: CatalogModel,
    models_dir: Path,
    token: str | None,
    *,
    on_progress: ProgressCallback | None,
    cancel: CancelSignal,
) -> Path:
    """Run one download in its own process, retrying in a fresh child after a stall.

    A set *cancel* signal terminates the child, which is the only way to free
    the bandwidth of a running hf_xet transfer mid-flight. The same terminate
    ends a child that stops reporting bytes, so stopping a wedged transfer does
    not depend on it answering an abort.
    """
    if cancel.is_set():
        raise TaskCancelledError
    for attempt in range(STALL_ATTEMPTS):
        try:
            return _run_one_attempt(entry, models_dir, token, on_progress, cancel)
        except _ChildStalledError:
            log.warning(
                "Transfer of %s stalled (attempt %d/%d); starting again in a new process.",
                entry.hf_repo,
                attempt + 1,
                STALL_ATTEMPTS,
            )
    raise RuntimeError(_stalled_download_message(entry.hf_repo))


def _stalled_download_message(hf_repo: str) -> str:
    """The user-facing error for a transfer that never started moving again."""
    return (
        f"Download of {hf_repo} kept stalling with almost no data arriving. "
        "Check the network connection and retry. Files that finished are kept; "
        "the file in flight starts again."
    )


def _run_one_attempt(
    entry: CatalogModel,
    models_dir: Path,
    token: str | None,
    on_progress: ProgressCallback | None,
    cancel: CancelSignal,
) -> Path:
    """Spawn one child, relay it, and clear the partials it owned on any failure.

    ``KeyboardInterrupt`` is caught because Ctrl-C during a download leaves the
    same partial file any other interrupted attempt leaves. ``SystemExit`` is
    not, so interpreter shutdown skips the filesystem work.
    """
    worker, receiver = _start_worker(entry, models_dir, token)
    owned: list[str] = []
    try:
        try:
            return _relay_until_done(entry, worker, receiver, on_progress, cancel, owned)
        finally:
            _stop_worker(worker)
            receiver.close()
    # The inner finally must terminate the child before this delete runs, or the
    # delete races the child's own writer.
    except (Exception, KeyboardInterrupt):
        _discard_partial_blobs(models_dir, entry.hf_repo, owned)
        raise


def _discard_partial_blobs(models_dir: Path, hf_repo: str, owned: list[str]) -> None:
    """Delete the partial blobs the child reported owning, which nothing reads."""
    # heavy: lilbee.catalog.download (>50ms; huggingface_hub fanout)
    from lilbee.catalog.download import discard_partial_blobs

    discard_partial_blobs(models_dir, hf_repo, owned)


def _start_worker(
    entry: CatalogModel, models_dir: Path, token: str | None
) -> tuple[_Worker, Connection]:
    """Spawn the download child; fork is unsafe under the parent's threads.

    Daemonic on purpose: at interpreter exit multiprocessing terminates daemon
    children but joins live non-daemon ones, and quitting the app mid-download
    must not wait for a multi-GB transfer.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_run_download_child,
        args=(sender, entry, str(models_dir), token),
        name=f"lilbee-download-{entry.hf_repo}",
        daemon=True,
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
    owned: list[str],
) -> Path:
    """Forward child messages until its verdict, polling *cancel* between them.

    The budget is tested only when the pipe holds nothing, so a child whose
    verdict is already queued is never terminated with its answer unread.
    """
    deadline = _StallDeadline()
    while True:
        if cancel.is_set():
            raise TaskCancelledError
        if receiver.poll(_POLL_INTERVAL_S):
            try:
                message = receiver.recv()
            except (EOFError, OSError):
                # No verdict is reachable once the read fails, however it
                # fails: POSIX raises EOFError at the closed pipe, Windows a
                # BrokenPipeError.
                raise _died_silently(entry, worker) from None
            verdict = _apply(message, on_progress, deadline, owned)
            if verdict is not None:
                return verdict
        elif receiver.poll():
            continue  # a message landed while the poll was expiring; read it next pass
        elif not worker.is_alive():
            raise _died_silently(entry, worker)
        elif deadline.expired():
            raise _ChildStalledError


def _died_silently(entry: CatalogModel, worker: _Worker) -> RuntimeError:
    """The error for a child that exited without reporting a verdict."""
    return RuntimeError(
        f"Download of {entry.hf_repo} stopped: its process exited with code {worker.exitcode}."
    )


def _apply(
    message: _ChildMessage,
    on_progress: ProgressCallback | None,
    deadline: _StallDeadline,
    owned: list[str],
) -> Path | None:
    """Act on one child message, returning the path once the child reports done."""
    if message.kind == "progress":
        deadline.saw(message.downloaded)
        if on_progress is not None:
            on_progress(message.downloaded, message.total)
        return None
    if message.kind == "probed":
        deadline.probed()
        if message.blob is not None:
            owned.append(message.blob)
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
        path = fetch_model_files(
            entry,
            Path(models_dir),
            token,
            on_progress=_PipeProgress(conn),
            on_probe=_PipeProbe(conn),
        )
        conn.send(_Done(kind="done", path=str(path)))
    except Exception as exc:
        conn.send(_Failed(kind="failed", error_type=type(exc).__name__, message=str(exc)))


def _silence_output() -> None:
    """Point stdout/stderr at devnull; the parent may own a Textual screen."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)
