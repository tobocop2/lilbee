"""A hung test prints every thread's stack before its timeout ends it."""

from __future__ import annotations

import os
import subprocess
import sys
import time
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
# off, so only the controller's stall watchdog can end the run.
_STALLED_TEARDOWN = """
import time

import pytest


@pytest.fixture(scope="session")
def slow_to_close():
    yield
    time.sleep(_STALL_SLEEP_S)


def test_last(slow_to_close):
    pass
"""
_STALL_BACKSTOP_S = 1
_STALL_SLEEP_S = 30
_STALL_ENDS_WITHIN_S = 20
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


def _run_pytest(
    tmp_path: Path, source: str, *extra: str, backstop_s: int = _BACKSTOP_S
) -> tuple[subprocess.CompletedProcess[str], Path]:
    test_file = tmp_path / "test_case.py"
    test_file.write_text(source, encoding="utf-8")
    dump_dir = tmp_path / "dumps"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(_PARENT_ENV_PREFIXES)},
        "LILBEE_TEST_HANG_DUMP_S": str(backstop_s),
        "LILBEE_TEST_HANG_DUMP_DIR": str(dump_dir),
        "PYTHONPATH": str(_TESTS_DIR),
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


def test_quiet_test_leaves_no_dump(tmp_path: Path) -> None:
    result, dump_dir = _run_pytest(tmp_path, "def test_ok():\n    assert True\n")

    assert "1 passed" in result.stdout
    assert all(f.stat().st_size == 0 for f in dump_dir.glob("hang-*.txt"))


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no signal to make a worker dump")
def test_a_stall_outside_any_test_ends_the_run_with_every_stack(tmp_path: Path) -> None:
    source = _STALLED_TEARDOWN.replace("_STALL_SLEEP_S", str(_STALL_SLEEP_S))
    started = time.monotonic()
    result, dump_dir = _run_pytest(
        tmp_path, source, "-n", "1", "--timeout=0", backstop_s=_STALL_BACKSTOP_S
    )
    elapsed = time.monotonic() - started

    assert result.returncode != 0
    assert elapsed < _STALL_ENDS_WITHIN_S, result.stdout + result.stderr
    controller = (dump_dir / "hang-master.txt").read_text(encoding="utf-8")
    assert "no test report for 4s after test_case.py::test_last (call)" in controller
    assert "Thread" in controller
    worker = (dump_dir / "hang-gw0.txt").read_text(encoding="utf-8")
    assert "slow_to_close" in worker


def test_a_single_process_run_arms_no_stall_watchdog(tmp_path: Path) -> None:
    result, dump_dir = _run_pytest(
        tmp_path, "def test_ok():\n    assert True\n", backstop_s=_STALL_BACKSTOP_S
    )

    assert "1 passed" in result.stdout
    assert (dump_dir / "hang-master.txt").read_text(encoding="utf-8") == ""
