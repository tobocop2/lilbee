"""The suite's xdist flags end the run when a worker dies mid-run."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_BUDGET_S = 60.0
# Coverage flags are irrelevant to the toy suite. --forked moves the test into a child
# process, so the child dies and the xdist worker does not; the worker is what is under test.
_DROPPED_FLAG_PREFIXES = ("--cov", "--forked")
_RULE_LINE = re.compile(r"^([A-Za-z][\w.-]*):(?!=)")
_CHILD_ENV_PREFIXES = ("PYTEST_", "COV_CORE_")
_PYTEST_WORD = "pytest"
_PYTEST_LAUNCHERS = ("run", "-m")
_AUTO_WORKERS = ("auto", "logical")
_NO_WORKER_RESTARTS = 0
_XDIST_PLUGIN = "xdist"
_EXIT_OK = 0
_MAKE_RECURSION_ENV_VARS = ("MAKEFLAGS", "MAKELEVEL")
_CRASH_NODEID = "test_toy.py::test_worker_dies"
_CRASH_REPORT = "crashed while running"
_EXIT_TESTS_FAILED = 1
_MARKER_ENV = "TOY_DIED_ONCE"
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
        raise RuntimeError(
            f"{shlex.join(command)} exited {result.returncode}:\n{result.stdout}{result.stderr}"
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


def _pytest_args(target: str, all_targets: list[str]) -> list[str] | None:
    """The pytest arguments of the target's own expanded recipe, or None if it has no pytest.

    `make -n` expands variables, so `$(PYTEST)` and `$(XDIST)` are read as the shell sees them.
    `-o` marks every other target as up to date, so a prerequisite's recipe is not attributed here.
    """
    others = [flag for other in all_targets if other != target for flag in ("-o", other)]
    for line in _make("-n", *others, target).splitlines():
        if _PYTEST_WORD not in line:
            continue
        words = shlex.split(line)
        for index, word in enumerate(words):
            if word == _PYTEST_WORD and (index == 0 or words[index - 1] in _PYTEST_LAUNCHERS):
                rest = words[index + 1 :]
                return [w for w in rest if not w.startswith(_DROPPED_FLAG_PREFIXES)]
    return None


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
    """Every target whose pytest run uses more than one xdist worker, with its arguments."""
    all_targets = _rule_targets()
    args_by_target = {target: _pytest_args(target, all_targets) for target in all_targets}
    return {
        target: args
        for target, args in args_by_target.items()
        if args is not None and _is_distributed(_parsed_options(args))
    }


_XDIST_TARGETS = _xdist_targets()


def test_the_makefile_has_xdist_targets_to_check() -> None:
    assert {"test", "test-ci", "test-ci-forked"} <= set(_XDIST_TARGETS)
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


@pytest.mark.timeout(_RUN_BUDGET_S * 2)
@pytest.mark.parametrize("target", sorted(_XDIST_TARGETS))
def test_a_dead_worker_fails_the_run_instead_of_hanging_it(target: str, tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_toy.py").write_text(_TOY_SUITE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CHILD_ENV_PREFIXES)}
    env[_MARKER_ENV] = str(tmp_path / "died-once")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *_XDIST_TARGETS[target]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=_RUN_BUDGET_S,
    )
    assert result.returncode == _EXIT_TESTS_FAILED, result.stdout + result.stderr
    assert _CRASH_REPORT in result.stdout
    assert _CRASH_NODEID in result.stdout
