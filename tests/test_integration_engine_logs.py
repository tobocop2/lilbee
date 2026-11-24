"""The integration suite prints the engine logs of a test that fails or times out."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.integration import _engine_logs as engine_logs

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_log(directory: Path, name: str, lines: list[str]) -> Path:
    log_dir = directory / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_timed_out_test_report_carries_the_engine_log_tails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real pytest-timeout failure shows both engine logs in its report."""
    if sys.platform == "win32":
        pytest.skip("pytest-timeout kills the process on Windows, so no report is written")
    data = tmp_path / "data"
    _write_log(data, "llama-swap-chat.log", [f"swap line {n}" for n in range(300)])
    _write_log(data, "engine-qwen.log", ["engine says: model stalled"])
    monkeypatch.setenv("LILBEE_DATA", str(data))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(_REPO_ROOT), *sys.path]))
    pytester.makeconftest(
        "from tests.integration.conftest import _remember_basetemp, pytest_runtest_makereport\n"
    )
    pytester.makepyfile(
        "import time\n"
        "import pytest\n"
        "@pytest.mark.timeout(1)\n"
        "def test_stalls():\n"
        "    time.sleep(30)\n"
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Engine logs*", "*Timeout*", "*engine says: model stalled*"])
    result.stdout.fnmatch_lines(["*swap line 299*"])
    assert "swap line 99\n" not in result.stdout.str()


def test_find_logs_returns_nothing_without_logs(tmp_path: Path) -> None:
    assert engine_logs.find_logs([tmp_path, tmp_path / "missing"]) == []
    assert engine_logs.failure_section([tmp_path]) is None


def test_find_logs_ignores_other_log_files(tmp_path: Path) -> None:
    _write_log(tmp_path, "cli.log", ["not an engine"])
    _write_log(tmp_path, "llama-swap-chat.log", ["swap"])
    found = engine_logs.find_logs([tmp_path])
    assert [path.name for path in found] == ["llama-swap-chat.log"]


def test_find_logs_lists_a_log_once_and_caps_the_count(tmp_path: Path) -> None:
    for number in range(engine_logs.MAX_FILES + 3):
        _write_log(tmp_path, f"engine-m{number}.log", ["x"])
    found = engine_logs.find_logs([tmp_path, tmp_path])
    assert len(found) == engine_logs.MAX_FILES
    assert len(set(found)) == len(found)


def test_find_logs_puts_the_newest_log_first(tmp_path: Path) -> None:
    old = _write_log(tmp_path, "engine-old.log", ["old"])
    new = _write_log(tmp_path, "engine-new.log", ["new"])
    os.utime(old, (1_000_000, 1_000_000))
    assert [path.name for path in engine_logs.find_logs([tmp_path])] == [new.name, old.name]


def test_tail_keeps_the_last_lines(tmp_path: Path) -> None:
    path = _write_log(tmp_path, "engine-a.log", [f"line {n}" for n in range(500)])
    text = engine_logs.tail(path)
    assert text.splitlines()[0] == "line 300"
    assert text.splitlines()[-1] == "line 499"
    assert len(text.splitlines()) == engine_logs.TAIL_LINES


def test_tail_replaces_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "engine-b.log"
    path.write_bytes(b"ok \xff\xfe end\n")
    assert engine_logs.tail(path) == "ok �� end\n"


def test_failure_section_names_each_log(tmp_path: Path) -> None:
    swap = _write_log(tmp_path, "llama-swap-chat.log", ["swap up"])
    section = engine_logs.failure_section([tmp_path])
    assert section is not None
    assert str(swap.resolve()) in section
    assert "swap up" in section
