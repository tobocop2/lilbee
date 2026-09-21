"""Cancelling a download has to stop its bytes, so each download owns a child process.

hf_xet cancels only at session granularity within a process, so the parent
relay terminates the child instead: the transfer dies with the process, and
concurrent downloads in sibling processes keep running.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import download_process as dp
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.runtime.cancellation import TaskCancelledError

pytestmark = pytest.mark.download_child


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/x-GGUF",
        gguf_filename="x.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )


class _Flag:
    """CancelSignal test double whose state the test flips directly."""

    def __init__(self, value: bool = False) -> None:
        self.value = value

    def is_set(self) -> bool:
        return self.value


class _FakeWorker:
    """Worker double backed by a plain flag instead of a real process."""

    def __init__(self, alive: bool = True, exitcode: int | None = None) -> None:
        self.alive = alive
        self.terminated = False
        self.killed = False
        self.ignores_term = False
        self._exitcode = exitcode

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_term:
            self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        return None


def _pipe() -> tuple[Any, Any]:
    receiver, sender = multiprocessing.get_context("spawn").Pipe(duplex=False)
    return receiver, sender


def _wire(monkeypatch: pytest.MonkeyPatch, worker: _FakeWorker, receiver: Any) -> None:
    monkeypatch.setattr(dp, "_start_worker", lambda entry, models_dir, token: (worker, receiver))


def test_progress_is_forwarded_and_done_returns_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = _pipe()
    worker = _FakeWorker()
    _wire(monkeypatch, worker, receiver)
    sender.send(dp._Progress(kind="progress", downloaded=10, total=100))
    sender.send(dp._Done(kind="done", path=str(tmp_path / "x.gguf")))
    seen: list[tuple[int, int]] = []

    path = dp.download_in_subprocess(
        _entry(), tmp_path, None, on_progress=lambda d, t: seen.append((d, t)), cancel=_Flag()
    )

    assert path == tmp_path / "x.gguf"
    assert seen == [(10, 100)]


def test_progress_without_a_callback_is_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = _pipe()
    _wire(monkeypatch, _FakeWorker(), receiver)
    sender.send(dp._Progress(kind="progress", downloaded=10, total=100))
    sender.send(dp._Done(kind="done", path=str(tmp_path / "x.gguf")))

    path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert path == tmp_path / "x.gguf"


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [("PermissionError", PermissionError), ("SomethingElse", RuntimeError)],
)
def test_a_child_failure_is_rebuilt_as_its_translated_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: str,
    expected: type[Exception],
) -> None:
    receiver, sender = _pipe()
    worker = _FakeWorker()
    _wire(monkeypatch, worker, receiver)
    sender.send(dp._Failed(kind="failed", error_type=error_type, message="boom"))

    with pytest.raises(expected, match="boom"):
        dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())
    assert not worker.alive


def test_a_cancel_that_lands_before_the_spawn_starts_no_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[str] = []

    def _spawn(entry: CatalogModel, models_dir: Path, token: str | None) -> Any:
        spawned.append(entry.hf_repo)
        raise AssertionError("must not spawn")

    monkeypatch.setattr(dp, "_start_worker", _spawn)

    with pytest.raises(TaskCancelledError):
        dp.download_in_subprocess(
            _entry(), tmp_path, None, on_progress=None, cancel=_Flag(value=True)
        )

    assert spawned == []


def test_a_cancel_mid_transfer_terminates_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = _pipe()
    worker = _FakeWorker()
    _wire(monkeypatch, worker, receiver)
    flag = _Flag()
    sender.send(dp._Progress(kind="progress", downloaded=10, total=100))

    def _cancel_on_progress(downloaded: int, total: int) -> None:
        flag.value = True

    with pytest.raises(TaskCancelledError):
        dp.download_in_subprocess(
            _entry(), tmp_path, None, on_progress=_cancel_on_progress, cancel=flag
        )

    assert worker.terminated


def test_a_progress_callback_that_raises_cancel_terminates_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = _pipe()
    worker = _FakeWorker()
    _wire(monkeypatch, worker, receiver)
    sender.send(dp._Progress(kind="progress", downloaded=10, total=100))

    def _raise(downloaded: int, total: int) -> None:
        raise TaskCancelledError

    with pytest.raises(TaskCancelledError):
        dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=_raise, cancel=_Flag())

    assert worker.terminated


def test_a_child_that_dies_silently_reads_as_a_failure_naming_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, _sender = _pipe()  # held open so the relay sees silence, not EOF
    worker = _FakeWorker(alive=False, exitcode=-9)
    _wire(monkeypatch, worker, receiver)

    with pytest.raises(RuntimeError, match="exited with code -9"):
        dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())


def test_a_message_racing_the_child_death_is_still_delivered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receiver, sender = _pipe()
    worker = _FakeWorker(alive=False, exitcode=0)
    _wire(monkeypatch, worker, receiver)
    sender.send(dp._Done(kind="done", path=str(tmp_path / "x.gguf")))

    path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert path == tmp_path / "x.gguf"


def test_stop_worker_escalates_to_kill_when_term_is_ignored() -> None:
    worker = _FakeWorker()
    worker.ignores_term = True

    dp._stop_worker(worker)

    assert worker.terminated
    assert worker.killed
    assert not worker.alive


def test_pipe_progress_throttles_but_always_sends_first_and_final() -> None:
    receiver, sender = _pipe()
    progress = dp._PipeProgress(sender)

    progress(1, 100)
    progress(2, 100)  # inside the min interval: dropped
    progress(100, 100)  # final: always sent

    first = receiver.recv()
    final = receiver.recv()
    assert (first.downloaded, final.downloaded) == (1, 100)
    assert not receiver.poll()


def test_pipe_progress_sends_again_after_the_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver, sender = _pipe()
    progress = dp._PipeProgress(sender)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(dp.time, "monotonic", lambda: next(clock))

    progress(1, 100)
    progress(2, 100)

    assert receiver.recv().downloaded == 1
    assert receiver.recv().downloaded == 2


def test_the_child_body_reports_the_fetched_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lilbee.catalog.download as dl

    receiver, sender = _pipe()
    monkeypatch.setattr(dp, "_silence_output", lambda: None)
    monkeypatch.setattr(
        dl,
        "fetch_model_files",
        lambda entry, models_dir, token, on_progress, on_probe: tmp_path / "x.gguf",
    )

    dp._run_download_child(sender, _entry(), str(tmp_path), None)

    message = receiver.recv()
    assert message.kind == "done"
    assert message.path == str(tmp_path / "x.gguf")


def test_the_child_body_serializes_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lilbee.catalog.download as dl

    receiver, sender = _pipe()
    monkeypatch.setattr(dp, "_silence_output", lambda: None)

    def _boom(entry: Any, models_dir: Any, token: Any, on_progress: Any, on_probe: Any) -> Path:
        raise PermissionError("gated")

    monkeypatch.setattr(dl, "fetch_model_files", _boom)

    dp._run_download_child(sender, _entry(), str(tmp_path), None)

    message = receiver.recv()
    assert message.kind == "failed"
    assert (message.error_type, message.message) == ("PermissionError", "gated")


@pytest.mark.timeout(120)
def test_a_real_spawned_child_downloads_a_cached_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real spawn end to end: offline mode makes the pre-seeded file a cache hit."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "x.gguf").write_bytes(b"gguf")
    seen: list[tuple[int, int]] = []

    path = dp.download_in_subprocess(
        _entry(), tmp_path, None, on_progress=lambda d, t: seen.append((d, t)), cancel=_Flag()
    )

    assert path == tmp_path / "x.gguf"
    assert seen[-1] == (4, 4)


@pytest.mark.timeout(120)
def test_a_real_spawned_child_dies_on_terminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling mid-spawn really kills the child rather than orphaning it."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    flag = _Flag()
    worker_seen: list[dp._Worker] = []
    real_start = dp._start_worker

    def _spy(entry: CatalogModel, models_dir: Path, token: str | None) -> Any:
        started = real_start(entry, models_dir, token)
        worker_seen.append(started[0])
        return started

    monkeypatch.setattr(dp, "_start_worker", _spy)

    def _cancel_soon() -> None:
        time.sleep(0.5)
        flag.value = True

    canceller = threading.Thread(target=_cancel_soon)
    canceller.start()
    try:
        with pytest.raises((TaskCancelledError, RuntimeError)):
            dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=flag)
    finally:
        canceller.join()

    assert worker_seen and not worker_seen[0].is_alive()


def test_download_model_routes_to_a_child_when_cancellable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cancel signal selects the child-process path and still runs on_complete."""
    import lilbee.catalog.download as dl

    monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
    monkeypatch.setattr(dl, "hf_token", lambda: None)
    calls: list[Path] = []
    completed: list[Path] = []

    def _fake_run(
        entry: CatalogModel, models_dir: Path, token: str | None, *, on_progress: Any, cancel: Any
    ) -> Path:
        calls.append(models_dir)
        return tmp_path / "x.gguf"

    monkeypatch.setattr(dp, "download_in_subprocess", _fake_run)

    result = dl.download_model(
        _entry(), on_complete=lambda entry, path: completed.append(path), cancel=_Flag()
    )

    assert result == tmp_path / "x.gguf"
    assert calls == [tmp_path]
    assert completed == [tmp_path / "x.gguf"]


def test_silence_output_swallows_writes(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """The child's console output must never reach the parent's terminal."""
    import os

    with capfd.disabled():
        out_fd = sys.stdout.fileno()
        err_fd = sys.stderr.fileno()
        capture = os.open(str(tmp_path / "out"), os.O_CREAT | os.O_WRONLY)
        saved_out = os.dup(out_fd)
        saved_err = os.dup(err_fd)
        try:
            os.dup2(capture, out_fd)
            os.dup2(capture, err_fd)
            dp._silence_output()
            os.write(out_fd, b"leak")
            os.write(err_fd, b"leak")
        finally:
            os.dup2(saved_out, out_fd)
            os.dup2(saved_err, err_fd)
            os.close(saved_out)
            os.close(saved_err)
            os.close(capture)

    assert (tmp_path / "out").read_bytes() == b""


def test_the_spawned_child_is_daemonic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Interpreter exit terminates daemon children but joins live non-daemon
    ones; quitting the app mid-download must not wait for the transfer."""

    class _Recorder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs
            recorded.append(self)

        def start(self) -> None:
            return None

    real_context = multiprocessing.get_context("spawn")

    class _Context:
        Process = _Recorder

        @staticmethod
        def Pipe(duplex: bool) -> Any:
            return real_context.Pipe(duplex=duplex)

    recorded: list[_Recorder] = []
    monkeypatch.setattr(dp.multiprocessing, "get_context", lambda method: _Context)

    dp._start_worker(_entry(), tmp_path, None)

    assert recorded[0].kwargs["daemon"] is True


@pytest.mark.parametrize("error", [EOFError(), BrokenPipeError(109, "The pipe has been ended")])
def test_a_child_that_dies_mid_message_reads_as_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    """A pipe that closes mid-message must fail with the exit code, not a raw pipe error.

    POSIX raises EOFError there and Windows BrokenPipeError, so both are driven
    here rather than only whichever the host produces.
    """

    class _DeadPipe:
        def poll(self, timeout: float | None = None) -> bool:
            return True

        def recv(self) -> object:
            raise error

        def close(self) -> None:
            return None

    worker = _FakeWorker(alive=False, exitcode=-9)
    monkeypatch.setattr(dp, "_start_worker", lambda entry, models_dir, token: (worker, _DeadPipe()))

    with pytest.raises(RuntimeError, match="exited with code -9"):
        dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())


def _wire_attempts(
    monkeypatch: pytest.MonkeyPatch,
    scripts: list[list[Any]],
    workers: list[_FakeWorker],
) -> tuple[list[_FakeWorker], list[Any]]:
    """Give each attempt its own child and pipe, pre-loaded with that attempt's messages.

    The senders are handed back so the test keeps them open: a closed write end
    reads as a dead child rather than a quiet one.
    """
    started: list[_FakeWorker] = []
    senders: list[Any] = []

    def _start(entry: CatalogModel, models_dir: Path, token: str | None) -> tuple[Any, Any]:
        receiver, sender = _pipe()
        senders.append(sender)
        for message in scripts[len(started)]:
            sender.send(message)
        started.append(workers[len(started)])
        return started[-1], receiver

    monkeypatch.setattr(dp, "_start_worker", _start)
    return started, senders


def _shorten_deadline(
    monkeypatch: pytest.MonkeyPatch, deadline_s: float, floor: int, startup_s: float | None = None
) -> None:
    """Run the relay against a real deadline on a test-sized clock."""
    real = dp._StallDeadline
    monkeypatch.setattr(
        dp,
        "_StallDeadline",
        lambda: real(
            deadline_s=deadline_s,
            floor_bytes=floor,
            startup_s=deadline_s if startup_s is None else startup_s,
        ),
    )


def test_a_child_that_stops_reporting_bytes_is_terminated_and_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The quiet child never answers anything, so only killing it can end the transfer."""
    _shorten_deadline(monkeypatch, 0.05, 1024)
    workers = [_FakeWorker(), _FakeWorker()]
    started, _senders = _wire_attempts(
        monkeypatch, [[], [dp._Done(kind="done", path=str(tmp_path / "x.gguf"))]], workers
    )

    path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert path == tmp_path / "x.gguf"
    assert len(started) == 2
    assert workers[0].terminated and not workers[0].alive


def test_a_child_that_ignores_terminate_is_killed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wedged transfer cannot refuse the stop: TERM escalates to KILL."""
    _shorten_deadline(monkeypatch, 0.05, 1024)
    stubborn = _FakeWorker()
    stubborn.ignores_term = True
    started, _senders = _wire_attempts(
        monkeypatch,
        [[], [dp._Done(kind="done", path=str(tmp_path / "x.gguf"))]],
        [stubborn, _FakeWorker()],
    )

    dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert len(started) == 2
    assert stubborn.killed and not stubborn.alive


def test_a_persistently_quiet_download_fails_with_the_stalled_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _shorten_deadline(monkeypatch, 0.05, 1024)
    workers = [_FakeWorker() for _ in range(dp.STALL_ATTEMPTS)]
    started, _senders = _wire_attempts(monkeypatch, [[] for _ in workers], workers)

    with pytest.raises(RuntimeError, match=r"kept stalling with almost no data arriving"):
        dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert len(started) == dp.STALL_ATTEMPTS
    assert all(worker.terminated for worker in workers)


def test_a_slow_but_moving_transfer_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control: bytes above the floor restart the clock, so a slow pull still finishes."""
    _shorten_deadline(monkeypatch, 0.5, 1024)
    script = [
        dp._Progress(kind="progress", downloaded=2048, total=8192),
        dp._Progress(kind="progress", downloaded=4096, total=8192),
        dp._Progress(kind="progress", downloaded=8192, total=8192),
        dp._Done(kind="done", path=str(tmp_path / "x.gguf")),
    ]
    started, _senders = _wire_attempts(monkeypatch, [script], [_FakeWorker()])
    seen: list[tuple[int, int]] = []

    path = dp.download_in_subprocess(
        _entry(), tmp_path, None, on_progress=lambda d, t: seen.append((d, t)), cancel=_Flag()
    )

    assert path == tmp_path / "x.gguf"
    assert len(started) == 1
    assert seen[-1] == (8192, 8192)


def test_the_deadline_expires_only_after_its_whole_span() -> None:
    fresh = dp._StallDeadline(deadline_s=60.0, floor_bytes=1024)
    fresh.saw(4096)
    aged = dp._StallDeadline(deadline_s=1.0, floor_bytes=1024)
    aged.saw(4096)
    aged._since -= 2.0

    assert not fresh.expired()
    assert aged.expired()


def test_a_trickle_under_the_floor_does_not_restart_the_deadline() -> None:
    deadline = dp._StallDeadline(deadline_s=1.0, floor_bytes=1024)
    deadline.saw(4096)
    deadline._since -= 2.0
    deadline.saw(4096 + 1023)

    assert deadline.expired()


def test_the_floor_of_new_bytes_restarts_the_deadline() -> None:
    deadline = dp._StallDeadline(deadline_s=1.0, floor_bytes=1024)
    deadline._since -= 2.0
    deadline.saw(1024)

    assert not deadline.expired()


def test_the_floor_counts_bytes_since_the_last_restart() -> None:
    """Progress is cumulative, so each restart measures from the count that caused it."""
    deadline = dp._StallDeadline(deadline_s=1.0, floor_bytes=1024)
    deadline.saw(1024)
    deadline._since -= 2.0
    deadline.saw(1500)

    assert deadline.expired()


def test_a_transfer_that_goes_quiet_after_moving_is_terminated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tighter budget applies once bytes have arrived, not the startup one."""
    _shorten_deadline(monkeypatch, 0.05, 1024, startup_s=600.0)
    workers = [_FakeWorker(), _FakeWorker()]
    started, _senders = _wire_attempts(
        monkeypatch,
        [
            [dp._Progress(kind="progress", downloaded=4096, total=99999)],
            [dp._Done(kind="done", path=str(tmp_path / "x.gguf"))],
        ],
        workers,
    )

    path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert path == tmp_path / "x.gguf"
    assert len(started) == 2
    assert workers[0].terminated


def test_a_child_still_resolving_file_sizes_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control: a child that has not reached its first bytes gets the startup budget."""
    _shorten_deadline(monkeypatch, 0.01, 1024, startup_s=600.0)
    worker = _FakeWorker()
    started, senders = _wire_attempts(monkeypatch, [[]], [worker])

    def _finish_late() -> None:
        time.sleep(0.5)
        senders[0].send(dp._Done(kind="done", path=str(tmp_path / "x.gguf")))

    late = threading.Thread(target=_finish_late)
    late.start()
    try:
        path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())
    finally:
        late.join()

    assert path == tmp_path / "x.gguf"
    assert len(started) == 1


def test_the_startup_budget_applies_before_the_first_bytes() -> None:
    deadline = dp._StallDeadline(deadline_s=1.0, floor_bytes=1024, startup_s=60.0)
    deadline._since -= 2.0

    assert not deadline.expired()


def test_a_child_resolving_file_after_file_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolved files are the only sign of life a many-shard model gives before bytes."""
    _shorten_deadline(monkeypatch, 0.05, 1024, startup_s=0.4)
    started, senders = _wire_attempts(
        monkeypatch,
        [[], [dp._Done(kind="done", path=str(tmp_path / "wrong.gguf"))]],
        [_FakeWorker(), _FakeWorker()],
    )

    def _probe_then_finish() -> None:
        try:
            for _ in range(4):
                time.sleep(0.15)
                senders[0].send(dp._Probed(kind="probed", blob=None))
            senders[0].send(dp._Done(kind="done", path=str(tmp_path / "x.gguf")))
        except OSError:
            return  # the relay gave up on this child and closed its pipe

    probing = threading.Thread(target=_probe_then_finish)
    probing.start()
    try:
        path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())
    finally:
        probing.join()

    assert path == tmp_path / "x.gguf"
    assert len(started) == 1


def test_the_child_reports_every_file_it_resolves_and_the_blob_it_writes() -> None:
    receiver, sender = _pipe()

    dp._PipeProbe(sender)("abc123")

    message = receiver.recv()
    assert (message.kind, message.blob) == ("probed", "abc123")


def test_a_terminated_child_leaves_no_partial_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A killed child unwinds nothing, so the parent that killed it clears the bytes."""
    blobs = tmp_path / "models--acme--x-GGUF" / "blobs"
    blobs.mkdir(parents=True)
    partial = blobs / "abc123.4f2a91cc.incomplete"
    partial.write_bytes(b"x" * 16)
    finished = blobs / "abc123"
    finished.write_bytes(b"y" * 16)
    _shorten_deadline(monkeypatch, 0.05, 1024)
    started, _senders = _wire_attempts(
        monkeypatch,
        [
            [dp._Probed(kind="probed", blob="abc123")],
            [dp._Done(kind="done", path=str(tmp_path / "x.gguf"))],
        ],
        [_FakeWorker(), _FakeWorker()],
    )

    dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert len(started) == 2
    assert not partial.exists()
    assert finished.exists()


def test_a_sibling_download_of_the_same_repo_keeps_its_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One repo ships several quants, so a second child is writing in the same folder.

    Sweeping the whole folder unlinks that child's temporary file under it, and
    on POSIX it keeps writing to an unlinked inode and fails at the final move
    after transferring the entire file.
    """
    blobs = tmp_path / "models--acme--x-GGUF" / "blobs"
    blobs.mkdir(parents=True)
    mine = blobs / "abc123.4f2a91cc.incomplete"
    mine.write_bytes(b"x" * 16)
    sibling = blobs / "def456.90bb17de.incomplete"
    sibling.write_bytes(b"y" * 16)
    _shorten_deadline(monkeypatch, 0.05, 1024)
    _wire_attempts(
        monkeypatch,
        [
            [dp._Probed(kind="probed", blob="abc123")],
            [dp._Done(kind="done", path=str(tmp_path / "x.gguf"))],
        ],
        [_FakeWorker(), _FakeWorker()],
    )

    dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert not mine.exists()
    assert sibling.exists()


def test_a_child_that_never_named_a_blob_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A child killed before it resolved a file owns no temporary file to clear."""
    blobs = tmp_path / "models--acme--x-GGUF" / "blobs"
    blobs.mkdir(parents=True)
    stranger = blobs / "def456.90bb17de.incomplete"
    stranger.write_bytes(b"y" * 16)
    _shorten_deadline(monkeypatch, 0.05, 1024)
    _wire_attempts(
        monkeypatch,
        [[], [dp._Done(kind="done", path=str(tmp_path / "x.gguf"))]],
        [_FakeWorker(), _FakeWorker()],
    )

    dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert stranger.exists()


def test_a_done_already_in_the_pipe_beats_the_expired_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The watchdog must not fire between the last byte and the clean exit.

    A child that moved the blob and sent its verdict one millisecond before the
    budget elapsed would otherwise be killed and its answer thrown away.
    """
    _shorten_deadline(monkeypatch, 0.0, 1024)  # already expired on the first pass
    started, _senders = _wire_attempts(
        monkeypatch, [[dp._Done(kind="done", path=str(tmp_path / "x.gguf"))]], [_FakeWorker()]
    )

    path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert path == tmp_path / "x.gguf"
    assert len(started) == 1


def test_a_failure_already_in_the_pipe_beats_the_expired_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gated repo must report itself, not be replaced by the stall message."""
    _shorten_deadline(monkeypatch, 0.0, 1024)
    started, _senders = _wire_attempts(
        monkeypatch,
        [[dp._Failed(kind="failed", error_type="PermissionError", message="gated")]],
        [_FakeWorker()],
    )

    with pytest.raises(PermissionError, match="gated"):
        dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert len(started) == 1


def test_a_message_that_lands_while_the_poll_expires_is_read_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The timed poll can miss by a microsecond what the next untimed poll sees."""
    done = dp._Done(kind="done", path=str(tmp_path / "x.gguf"))

    class _LatePipe:
        """A pipe whose first timed poll misses the message its untimed poll sees."""

        def __init__(self) -> None:
            self.timed_polls = 0

        def poll(self, timeout: float | None = None) -> bool:
            if timeout is None:
                return True
            self.timed_polls += 1
            return self.timed_polls > 1

        def recv(self) -> object:
            return done

        def close(self) -> None:
            return None

    pipe = _LatePipe()
    worker = _FakeWorker(alive=False, exitcode=0)
    monkeypatch.setattr(dp, "_start_worker", lambda entry, models_dir, token: (worker, pipe))

    path = dp.download_in_subprocess(_entry(), tmp_path, None, on_progress=None, cancel=_Flag())

    assert path == tmp_path / "x.gguf"


def test_download_model_routes_every_pull_through_a_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller with no cancel signal still gets the watchdog that ends a quiet child."""
    import lilbee.catalog.download as dl

    monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
    monkeypatch.setattr(dl, "hf_token", lambda: None)
    signals: list[bool] = []

    def _fake_run(
        entry: CatalogModel, models_dir: Path, token: str | None, *, on_progress: Any, cancel: Any
    ) -> Path:
        signals.append(cancel.is_set())
        return tmp_path / "x.gguf"

    monkeypatch.setattr(dp, "download_in_subprocess", _fake_run)

    assert dl.download_model(_entry()) == tmp_path / "x.gguf"
    assert signals == [False]
