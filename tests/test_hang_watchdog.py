"""A hung test prints every thread's stack before its timeout ends it."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent
_TEST_TIMEOUT_S = 4
_BACKSTOP_S = 90
_RUN_LIMIT_S = 60
# The parent run's pytest and coverage state must not leak into the child run.
_PARENT_ENV_PREFIXES = ("PYTEST_", "COV_CORE_")
_HUNG_TEST = """
import threading
import time

def test_stuck_on_a_lock():
    held = threading.Lock()
    held.acquire()
    def parked_waiter():
        held.acquire()
    threading.Thread(target=parked_waiter, name="parked-waiter", daemon=True).start()
    time.sleep(60)
"""
# The worker stalls after its last test's watchdog is cancelled: pytest-timeout is
# off, so only the controller's stall watchdog can end the run. The teardown blocks
# on a FIFO that nothing writes until the controller has exited, so a controller
# that exits did not wait for the worker.
_STALLED_TEARDOWN = """
import os

import pytest


@pytest.fixture(scope="session")
def slow_to_close():
    yield
    with open(os.environ["ENTERED"], "w", encoding="utf-8") as marker:
        marker.write(str(os.getpid()))
    with open(os.environ["RELEASE"], "rb") as fifo:
        fifo.read()


def test_last(slow_to_close):
    pass
"""
# Skip the stall timer armed at session start, so worker start-up is not in the race.
_ARM_AFTER_A_REPORT = """
import _hang_watchdog

_arm = _hang_watchdog._arm_stall


def _arm_after_a_report(last_report):
    if last_report != "session start":
        _arm(last_report)


_hang_watchdog._arm_stall = _arm_after_a_report
"""
_STALL_BACKSTOP_S = 1
# The test fails, which stops pytest-timeout, then its teardown never returns.
_FAILED_THEN_WEDGED = """
import threading

import pytest


@pytest.fixture
def wedged_teardown():
    yield
    threading.Event().wait()


def test_fails(wedged_teardown):
    assert False
"""
# The controller stall limit is four times this, so the worker backstop fires first.
_TEARDOWN_BACKSTOP_S = 2
_FAILED_FAST = "def test_fails():\n    assert False\n"
# The session ends after the failed test's teardown, slower than the backstop.
_SLOW_SESSION_FINISH = f"""
import time


def pytest_sessionfinish(session):
    time.sleep({_TEARDOWN_BACKSTOP_S * 2})
"""


def _child_run(
    tmp_path: Path, source: str, *extra: str, backstop_s: int
) -> tuple[list[str], dict[str, str], Path]:
    """The command, environment and dump directory of a child pytest run."""
    test_file = tmp_path / "test_case.py"
    test_file.write_text(source, encoding="utf-8")
    dump_dir = tmp_path / "dumps"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(_PARENT_ENV_PREFIXES)},
        "LILBEE_TEST_HANG_DUMP_S": str(backstop_s),
        "LILBEE_TEST_HANG_DUMP_DIR": str(dump_dir),
        "PYTHONPATH": str(_TESTS_DIR),
        "ENTERED": str(tmp_path / "entered"),
        "RELEASE": str(tmp_path / "release"),
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "_hang_watchdog",
        "-p",
        "no:cacheprovider",
        f"--timeout={_TEST_TIMEOUT_S}",
        "--rootdir",
        str(tmp_path),
        "-c",
        os.devnull,
        str(test_file),
        *extra,
    ]
    return command, env, dump_dir


def _run_pytest(
    tmp_path: Path, source: str, *extra: str, backstop_s: int = _BACKSTOP_S
) -> tuple[subprocess.CompletedProcess[str], Path]:
    command, env, dump_dir = _child_run(tmp_path, source, *extra, backstop_s=backstop_s)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=_RUN_LIMIT_S,
    )
    return result, dump_dir


@pytest.mark.parametrize("extra", [(), ("-n", "1")], ids=["single-process", "xdist-worker"])
def test_hung_test_dumps_stacks_to_the_job_log_before_its_timeout(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    result, dump_dir = _run_pytest(tmp_path, _HUNG_TEST, *extra)
    log = result.stdout + result.stderr

    assert result.returncode != 0
    if sys.platform != "win32":
        # pytest-timeout defaults to its thread method on Windows, which ends the
        # process after the dump, so no summary line is printed.
        assert "1 failed" in result.stdout
    assert f"still running after {_TEST_TIMEOUT_S * 0.75:.0f}s" in log
    assert "test_stuck_on_a_lock" in log
    assert "parked_waiter" in log
    assert "open file descriptors:" in log
    assert "locks held by this process:" in log
    dump_files = list(dump_dir.glob("hang-*.txt"))
    assert any("parked_waiter" in f.read_text(encoding="utf-8") for f in dump_files), (
        "the dump file is lost when the test ends"
    )


@pytest.mark.parametrize(
    ("extra", "dump_name"),
    [((), "hang-master.txt"), (("-n", "1"), "hang-gw0.txt")],
    ids=["single-process", "xdist-worker"],
)
def test_a_failed_test_whose_teardown_blocks_ends_at_the_backstop(
    tmp_path: Path, extra: tuple[str, ...], dump_name: str
) -> None:
    result, dump_dir = _run_pytest(
        tmp_path, _FAILED_THEN_WEDGED, *extra, backstop_s=_TEARDOWN_BACKSTOP_S
    )

    assert result.returncode != 0
    dump = (dump_dir / dump_name).read_text(encoding="utf-8")
    assert "hang watchdog armed for the teardown of test_case.py::test_fails" in dump
    assert "wedged_teardown" in dump
    controller = (dump_dir / "hang-master.txt").read_text(encoding="utf-8")
    assert "no test report for" not in controller


@pytest.mark.parametrize("extra", [(), ("-n", "1")], ids=["single-process", "xdist-worker"])
def test_a_failed_test_with_a_fast_teardown_leaves_no_dump(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    result, dump_dir = _run_pytest(tmp_path, _FAILED_FAST, *extra)

    assert "1 failed" in result.stdout
    dumps = {f.name: f.read_text(encoding="utf-8") for f in dump_dir.glob("hang-*.txt")}
    assert dumps
    assert all(text == "" for text in dumps.values()), dumps


def test_a_failed_test_does_not_arm_the_backstop_past_its_teardown(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(_SLOW_SESSION_FINISH, encoding="utf-8")
    result, dump_dir = _run_pytest(tmp_path, _FAILED_FAST, backstop_s=_TEARDOWN_BACKSTOP_S)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout
    assert (dump_dir / "hang-master.txt").read_text(encoding="utf-8") == ""


def test_quiet_test_leaves_no_dump(tmp_path: Path) -> None:
    result, dump_dir = _run_pytest(tmp_path, "def test_ok():\n    assert True\n")

    assert "1 passed" in result.stdout
    assert all(f.stat().st_size == 0 for f in dump_dir.glob("hang-*.txt"))


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no signal to make a worker dump")
def test_a_stall_outside_any_test_ends_the_run_with_every_stack(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "release")
    (tmp_path / "conftest.py").write_text(_ARM_AFTER_A_REPORT, encoding="utf-8")
    command, env, dump_dir = _child_run(
        tmp_path, _STALLED_TEARDOWN, "-n", "1", "--timeout=0", backstop_s=_STALL_BACKSTOP_S
    )
    with (tmp_path / "run.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=tmp_path
        )
        try:
            # The worker is still blocked, so the run ends only if the controller exits itself.
            # pytest-timeout ends a controller that never exits.
            returncode = child.wait()
            entered = (tmp_path / "entered").exists()
        finally:
            _release_stalled_worker(tmp_path / "release")
            child.wait()

    assert returncode != 0
    assert entered
    controller = (dump_dir / "hang-master.txt").read_text(encoding="utf-8")
    assert "no test report for 4s after test_case.py::test_last (call)" in controller
    assert "Thread" in controller
    worker = (dump_dir / "hang-gw0.txt").read_text(encoding="utf-8")
    assert "slow_to_close" in worker


def _release_stalled_worker(fifo: Path) -> None:
    """Unblock the worker's teardown so no process outlives the test."""
    # The open fails when the worker already exited, so nothing reads the FIFO.
    with contextlib.suppress(OSError):
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))


def test_a_single_process_run_arms_no_stall_watchdog(tmp_path: Path) -> None:
    result, dump_dir = _run_pytest(
        tmp_path, "def test_ok():\n    assert True\n", backstop_s=_STALL_BACKSTOP_S
    )

    assert "1 passed" in result.stdout
    assert (dump_dir / "hang-master.txt").read_text(encoding="utf-8") == ""


class _RecordedTimer:
    """Stands in for ``threading.Timer``: records its arguments and never fires."""

    def __init__(self, delay: float, fn: object, args: tuple[object, ...]) -> None:
        self.delay = delay
        self.args = args
        self.cancelled = False
        self.daemon = False

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True


class _Recorders:
    def __init__(self) -> None:
        self.timers: list[_RecordedTimer] = []
        self.exit_dumps: list[tuple[float, bool]] = []
        self.cancelled_dumps = 0


@pytest.fixture
def watchdog() -> types.ModuleType:
    """A private copy of the watchdog, so patching it cannot reach the live plugin."""
    spec = importlib.util.spec_from_file_location(
        "_hang_watchdog_copy", _TESTS_DIR / "_hang_watchdog.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stall_recorders(watchdog: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Recorders:
    """Run the controller's stall arming with recorder timers instead of real ones."""
    recorders = _Recorders()

    def make_timer(delay: float, fn: object, args: tuple[object, ...]) -> _RecordedTimer:
        timer = _RecordedTimer(delay, fn, args)
        recorders.timers.append(timer)
        return timer

    def dump_later(delay: float, *, file: object, exit: bool) -> None:
        recorders.exit_dumps.append((delay, exit))

    def cancel_dump() -> None:
        recorders.cancelled_dumps += 1

    monkeypatch.setattr(watchdog, "threading", types.SimpleNamespace(Timer=make_timer))
    monkeypatch.setattr(
        watchdog,
        "faulthandler",
        types.SimpleNamespace(
            dump_traceback_later=dump_later, cancel_dump_traceback_later=cancel_dump
        ),
    )
    monkeypatch.setattr(watchdog, "_dump_handle", object())
    monkeypatch.setattr(watchdog, "_timeout_s", float(_STALL_BACKSTOP_S))
    monkeypatch.setattr(watchdog, "_stall_s", 0.0)
    monkeypatch.setattr(watchdog, "_stall_timer", None)
    return recorders


def _xdist_controller_session() -> pytest.Session:
    manager = types.SimpleNamespace(has_plugin=lambda name: name == "dsession")
    return types.SimpleNamespace(config=types.SimpleNamespace(pluginmanager=manager))  # type: ignore[return-value]


def test_session_start_arms_the_stall_timer_and_the_exit(
    watchdog: types.ModuleType, stall_recorders: _Recorders
) -> None:
    watchdog.pytest_sessionstart(_xdist_controller_session())

    assert [(t.delay, t.args) for t in stall_recorders.timers] == [(4.0, ("session start",))]
    assert stall_recorders.exit_dumps == [(4.0 + watchdog._STALL_EXIT_GRACE_S, True)]


def test_a_report_re_arms_the_stall_timer_under_its_own_name(
    watchdog: types.ModuleType, stall_recorders: _Recorders
) -> None:
    watchdog.pytest_sessionstart(_xdist_controller_session())
    report = types.SimpleNamespace(nodeid="test_case.py::test_a", when="call")
    watchdog.pytest_runtest_logreport(report)  # type: ignore[arg-type]

    first, second = stall_recorders.timers
    assert first.cancelled
    assert not second.cancelled
    assert (second.delay, second.args) == (4.0, ("test_case.py::test_a (call)",))
    assert len(stall_recorders.exit_dumps) == 2


def test_session_finish_cancels_the_stall_timer_and_the_exit(
    watchdog: types.ModuleType, stall_recorders: _Recorders
) -> None:
    session = _xdist_controller_session()
    watchdog.pytest_sessionstart(session)
    watchdog.pytest_sessionfinish(session)

    assert stall_recorders.timers[-1].cancelled
    assert stall_recorders.cancelled_dumps == 1
    assert watchdog._stall_s == 0.0
