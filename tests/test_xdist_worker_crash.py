"""The suite's xdist flags end the run when a worker dies mid-run."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from lilbee.providers.fleet.proc import run_bounded
from tests._stall_diagnostics import StallWatch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_BUDGET_S = 60.0
_REAP_WAIT_S = 5.0
_GRANDCHILD_EXIT_WAIT_S = 5.0
_GRANDCHILD_POLL_S = 0.05
_GRANDCHILD_SLEEP_S = 600
_CHILD_START_LIMIT_S = 5.0
# Coverage flags and the suite's own test paths are irrelevant to the toy suite. --forked moves
# the test into a child process, so the child dies and the xdist worker does not; the worker is
# what is under test.
_DROPPED_WORD_PREFIXES = ("--cov", "--forked", "tests/")
_RULE_LINE = re.compile(r"^([A-Za-z][\w.-]*):(?!=)")
_CHILD_ENV_PREFIXES = ("PYTEST_", "COV_CORE_")
_PYTEST_WORD = "pytest"
_PYTEST_LAUNCHERS = ("run", "-m")
_AUTO_WORKERS = ("auto", "logical")
_NO_WORKER_RESTARTS = 0
_XDIST_PLUGIN = "xdist"
_EXIT_OK = 0
_FIRST_SESSION = 1
_MAKE_RECURSION_ENV_VARS = ("MAKEFLAGS", "MAKELEVEL")
_MAKE_STDOUT_TAIL_CHARS = 500
_CRASH_NODEID = "test_toy.py::test_worker_dies"
_CRASH_REPORT = "crashed while running"
_EXIT_TESTS_FAILED = 1
_MARKER_ENV = "TOY_DIED_ONCE"
# The toy run needs one worker to die and one to survive. `-n logical` would start one worker per
# CPU, and on a loaded host each worker's start competes with the others for the same CPUs.
_TOY_WORKERS = 2
_START_COST_ENV = "TOY_WORKER_START_COST_S"
_START_LOCK_ENV = "TOY_WORKER_START_LOCK"
_START_COST_S = 4.0
_TOY_CONFTEST = f"""
import os
import time

import filelock


def pytest_configure(config):
    # A loaded host starts the workers one after another on shared CPUs. One lock models that:
    # each worker holds it for the cost of its start.
    cost = float(os.environ.get("{_START_COST_ENV}", "0"))
    if cost and os.environ.get("PYTEST_XDIST_WORKER"):
        with filelock.FileLock(os.environ["{_START_LOCK_ENV}"]):
            time.sleep(cost)
"""
_TOY_SUITE = """
import os
import pathlib

import pytest


@pytest.mark.parametrize("index", range(200))
def test_before(index):
    pass


def test_worker_dies():
    # Dies once, like a timeout, so a rerun of this test on another worker passes.
    marker = pathlib.Path(os.environ["TOY_DIED_ONCE"])
    if not marker.exists():
        marker.touch()
        os._exit(1)


@pytest.mark.parametrize("index", range(2))
def test_after(index):
    pass
"""


def _make(*args: str, ok: tuple[int, ...] = (_EXIT_OK,)) -> str:
    """Run make in the repo root, outside any parent make, and return stdout."""
    make_env = {k: v for k, v in os.environ.items() if k not in _MAKE_RECURSION_ENV_VARS}
    command = ["make", "--no-print-directory", *args]
    result = subprocess.run(
        command, cwd=_REPO_ROOT, capture_output=True, encoding="utf-8", env=make_env
    )
    if result.returncode not in ok:
        stdout_tail = result.stdout[-_MAKE_STDOUT_TAIL_CHARS:]
        raise RuntimeError(
            f"{shlex.join(command)} exited {result.returncode}:\n"
            f"{result.stderr}\n"
            f"stdout (last {_MAKE_STDOUT_TAIL_CHARS} characters of {len(result.stdout)}):\n"
            f"{stdout_tail}"
        )
    return result.stdout


def _rule_targets() -> list[str]:
    """Every target make's rule database names (`-q` exits 1 for stale targets)."""
    targets: dict[str, None] = {}
    for line in _make("-pRrq", ok=(_EXIT_OK, 1)).splitlines():
        rule = _RULE_LINE.match(line)
        if rule:
            targets[rule.group(1)] = None
    return list(targets)


def _line_pytest_args(line: str) -> list[str] | None:
    """The pytest arguments of one recipe line, or None if the line runs no pytest."""
    words = shlex.split(line)
    for index, word in enumerate(words):
        if word == _PYTEST_WORD and (index == 0 or words[index - 1] in _PYTEST_LAUNCHERS):
            rest = words[index + 1 :]
            return [w for w in rest if not w.startswith(_DROPPED_WORD_PREFIXES)]
    return None


def _pytest_runs(target: str, all_targets: list[str]) -> list[list[str]]:
    """The pytest arguments of each pytest line in the target's own expanded recipe.

    `make -n` expands variables, so `$(PYTEST)` and `$(XDIST)` are read as the shell sees them.
    `-o` marks every other target as up to date, so a prerequisite's recipe is not attributed here.
    """
    others = [flag for other in all_targets if other != target for flag in ("-o", other)]
    lines = _make("-n", *others, target).splitlines()
    parsed = [_line_pytest_args(line) for line in lines if _PYTEST_WORD in line]
    return [args for args in parsed if args is not None]


def _run_name(target: str, session: int) -> str:
    """The target's name for its first pytest run, and `target#n` for a later run."""
    return target if session == _FIRST_SESSION else f"{target}#{session}"


class _OptionProbe:
    """A pytest plugin that records what pytest and xdist parsed, then stops before any run."""

    def __init__(self) -> None:
        self.workers: int | str | None = None
        self.max_worker_restart: int | str | None = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_cmdline_main(self, config: pytest.Config) -> int:
        if config.pluginmanager.hasplugin(_XDIST_PLUGIN):
            self.workers = config.option.numprocesses
            self.max_worker_restart = config.option.maxworkerrestart
        return _EXIT_OK


def _parsed_options(args: list[str]) -> _OptionProbe:
    """Let pytest and xdist parse the argument list, so no spelling of a flag is missed."""
    probe = _OptionProbe()
    code = pytest.main(["-p", "no:cacheprovider", "-o", "addopts=", *args], plugins=[probe])
    if code != _EXIT_OK:
        raise RuntimeError(f"pytest rejected the arguments {shlex.join(args)} (exit {code})")
    return probe


def _is_distributed(probe: _OptionProbe) -> bool:
    """True when xdist starts more than one worker."""
    if probe.workers in _AUTO_WORKERS:
        return True
    return isinstance(probe.workers, int) and probe.workers > 1


def _xdist_targets() -> dict[str, list[str]]:
    """Every pytest run of a target that uses more than one xdist worker, with its arguments."""
    all_targets = _rule_targets()
    args_by_run = {
        _run_name(target, session): args
        for target in all_targets
        for session, args in enumerate(_pytest_runs(target, all_targets), start=_FIRST_SESSION)
    }
    return {
        run: args for run, args in args_by_run.items() if _is_distributed(_parsed_options(args))
    }


_XDIST_TARGETS = _xdist_targets()


def test_the_makefile_has_xdist_targets_to_check() -> None:
    assert {"test", "test-ci", "test-ci-forked", "test-ci-split", "test-ci-split#2"} <= set(
        _XDIST_TARGETS
    )
    assert "test-ci-serial" not in _XDIST_TARGETS


@pytest.mark.parametrize("target", sorted(_XDIST_TARGETS))
def test_a_distributed_target_never_restarts_a_dead_worker(target: str) -> None:
    restarts = _parsed_options(_XDIST_TARGETS[target]).max_worker_restart
    assert restarts is not None
    assert int(restarts) == _NO_WORKER_RESTARTS


def test_a_failed_make_reports_the_command_and_its_output() -> None:
    message = r"-n no-such-target exited 2:\n.*No rule to make target"
    with pytest.raises(RuntimeError, match=message):
        _make("-n", "no-such-target")


def test_a_failed_make_puts_stderr_first_and_bounds_stdout() -> None:
    # `-p` makes stdout the whole rule database (tens of kilobytes); the error is on stderr.
    with pytest.raises(RuntimeError) as raised:
        _make("-pRrq", "no-such-target")
    message = str(raised.value)
    assert message.startswith("make --no-print-directory -pRrq no-such-target exited 2:\n")
    assert len(message) < 2 * _MAKE_STDOUT_TAIL_CHARS + 500
    assert message.index("No rule to make target") < message.index("stdout (last")


def _run_bounded_child(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout_s: float = _RUN_BUDGET_S
) -> tuple[str, int]:
    """Run a child run to its end or its limit; at the limit, kill its whole process group.

    The child starts its own xdist workers. `subprocess.run` kills only the child at the limit,
    so the workers it started stay alive and the reap can wait on them.

    At the limit the raised error carries the tail of the child's output, the process tree and the
    stacks of each descendant, taken before the kill. The limit still fails the test.
    """
    with StallWatch(timeout_s) as watch:
        try:
            return run_bounded(
                argv,
                timeout_s=timeout_s,
                kill_wait_s=_REAP_WAIT_S,
                env=env,
                merge_stderr=True,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as expired:
            raise watch.stalled(expired) from expired


def _is_running(pid: int) -> bool:
    """True while the process exists and is not a zombie waiting for a parent to reap it."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(os.name != "posix", reason="only POSIX has a process group to kill")
def test_a_child_run_that_hits_its_limit_leaves_no_worker_alive(tmp_path: Path) -> None:
    pid_file = tmp_path / "worker.pid"
    # One interpreter start: the worker is a fork of the child, not a second interpreter.
    # The pid file is renamed into place, so a reader never sees a half-written pid.
    spawn_worker_then_wait = (
        "import os, pathlib, time\n"
        "worker = os.fork()\n"
        f"if worker == 0:\n    time.sleep({_GRANDCHILD_SLEEP_S})\n    os._exit(0)\n"
        f"partial = pathlib.Path({str(pid_file) + '.partial'!r})\n"
        "partial.write_text(str(worker))\n"
        f"partial.rename({str(pid_file)!r})\n"
        f"time.sleep({_GRANDCHILD_SLEEP_S})\n"
    )
    worker_pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_bounded_child(
                [sys.executable, "-c", spawn_worker_then_wait],
                cwd=tmp_path,
                env=dict(os.environ),
                timeout_s=_CHILD_START_LIMIT_S,
            )
        assert pid_file.exists(), "the child did not start its worker before the limit"
        worker_pid = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + _GRANDCHILD_EXIT_WAIT_S
        while _is_running(worker_pid) and time.monotonic() < deadline:
            time.sleep(_GRANDCHILD_POLL_S)
        assert not _is_running(worker_pid)
    finally:
        if worker_pid is not None and _is_running(worker_pid):
            psutil.Process(worker_pid).kill()


@pytest.mark.skipif(os.name != "posix", reason="only POSIX has a process group to kill")
def test_a_child_run_that_hits_its_limit_reports_its_output_and_process_tree(
    tmp_path: Path,
) -> None:
    stalls = "import sys, time\nprint('last words of the child', flush=True)\ntime.sleep(600)\n"
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _run_bounded_child(
            [sys.executable, "-c", stalls], cwd=tmp_path, env=dict(os.environ), timeout_s=3.0
        )
    message = str(raised.value)
    assert "last words of the child" in message
    assert "process tree of pid" in message
    assert "status=sleeping" in message
    assert "py-spy" in message
    if sys.platform.startswith("linux"):
        assert "wchan=" in message
        assert "SigCgt" in message


def test_a_child_run_that_ends_in_time_returns_its_output_and_exit_code(tmp_path: Path) -> None:
    output, code = _run_bounded_child(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout_s=_CHILD_START_LIMIT_S,
    )
    assert (output.strip(), code) == ("done", _EXIT_OK)


def test_the_presets_target_reaches_the_toy_run_without_its_test_file() -> None:
    assert "tests/integration/test_preset_fixtures.py" in _make("-n", "test-presets")
    args = _XDIST_TARGETS["test-presets"]
    assert not any(a.endswith(".py") or a.startswith("tests/") for a in args)


def test_a_target_naming_no_test_file_keeps_all_its_arguments() -> None:
    line = "uv run pytest -v -n logical --dist loadgroup --max-worker-restart=0"
    assert _line_pytest_args(line) == [
        "-v",
        "-n",
        "logical",
        "--dist",
        "loadgroup",
        "--max-worker-restart=0",
    ]
    assert {"-v", "-n", "--dist", "--max-worker-restart=0"} <= set(_XDIST_TARGETS["test"])


def _toy_run(target: str, tmp_path: Path, *, start_cost_s: float = 0.0) -> tuple[str, int]:
    """Run the target's pytest arguments against the toy suite, whose one test kills its worker.

    The worker count is replaced: the last `-n` on the command line wins.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_toy.py").write_text(_TOY_SUITE, encoding="utf-8")
    (tmp_path / "conftest.py").write_text(_TOY_CONFTEST, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CHILD_ENV_PREFIXES)}
    env[_MARKER_ENV] = str(tmp_path / "died-once")
    env[_START_COST_ENV] = str(start_cost_s)
    env[_START_LOCK_ENV] = str(tmp_path / "start.lock")
    args = [*_XDIST_TARGETS[target], "-n", str(_TOY_WORKERS)]
    return _run_bounded_child(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=tmp_path, env=env
    )


def _assert_the_dead_worker_failed_the_run(output: str, returncode: int) -> None:
    assert returncode == _EXIT_TESTS_FAILED, output
    assert _CRASH_REPORT in output
    assert _CRASH_NODEID in output


@pytest.mark.timeout(_RUN_BUDGET_S * 2)
@pytest.mark.parametrize("target", sorted(_XDIST_TARGETS))
def test_a_dead_worker_fails_the_run_instead_of_hanging_it(target: str, tmp_path: Path) -> None:
    _assert_the_dead_worker_failed_the_run(*_toy_run(target, tmp_path))


@pytest.mark.timeout(_RUN_BUDGET_S * 2)
def test_a_slow_worker_start_does_not_fail_the_dead_worker_run(tmp_path: Path) -> None:
    # Every worker start costs _START_COST_S of a shared CPU, as on a host at load 40. With one
    # worker per CPU the starts alone pass the run budget on any host of 16 CPUs or more.
    _assert_the_dead_worker_failed_the_run(*_toy_run("test", tmp_path, start_cost_s=_START_COST_S))
