"""The integration suite prints the engine logs of a test that fails or times out."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conftest import child_pythonpath
from tests.integration import _engine_logs as engine_logs

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_log(directory: Path, name: str, lines: list[str]) -> Path:
    log_dir = directory / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_inner(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
) -> pytest.RunResult:
    """Run the tests in *source* under the integration hooks, logs where the fleet writes them.

    The fleet writes to the machine engine slot, which LILBEE_ENGINE_DIR moves; the
    logs go there and nowhere else. The inner conftest touches the file named by
    EMIT_MARKER each time the integration conftest prints, so a test can wait for
    the print instead of racing it with a sleep.
    """
    if sys.platform == "win32":
        pytest.skip("pytest-timeout kills the process on Windows, so no report is written")
    slot = tmp_path / "engine-slot"
    _write_log(slot, "llama-swap-chat.log", [f"swap line {n}" for n in range(300)])
    _write_log(slot, "engine-qwen.log", ["engine says: model stalled"])
    _write_log(slot, "engine-reaper.log", ["reaper noise"])
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(slot))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PYTHONPATH", child_pythonpath(_REPO_ROOT, *sys.path))
    monkeypatch.setenv("EMIT_MARKER", str(tmp_path / "emitted"))
    pytester.makeconftest(
        "import os\n"
        "from pathlib import Path\n"
        "import tests.integration.conftest as integration\n"
        "from tests.integration.conftest import (\n"
        "    _print_engine_logs_when_stalled, _remember_basetemp, pytest_runtest_makereport,\n"
        ")\n"
        "real_emit = integration.emit\n"
        "def emit(text):\n"
        "    real_emit(text)\n"
        "    if text:\n"
        "        Path(os.environ['EMIT_MARKER']).touch()\n"
        "integration.emit = emit\n"
    )
    pytester.makepyfile(
        "import os, time\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "def wait_for_emit(seconds):\n"
        "    end = time.monotonic() + seconds\n"
        "    while time.monotonic() < end and not Path(os.environ['EMIT_MARKER']).exists():\n"
        "        time.sleep(0.05)\n" + source
    )
    return pytester.runpytest_subprocess("-p", "no:cacheprovider")


def test_timed_out_test_prints_the_engine_log_tails_at_once(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real pytest-timeout failure writes both engine logs to stderr when it fails."""
    source = "@pytest.mark.timeout(5)\ndef test_stalls():\n    time.sleep(60)\n"
    result = _run_inner(pytester, monkeypatch, tmp_path, source)

    result.assert_outcomes(failed=1)
    result.stderr.fnmatch_lines(["*Engine logs*", "*engine says: model stalled*"])
    result.stderr.fnmatch_lines(["*swap line 299*"])
    assert "swap line 99\n" not in result.stderr.str()
    assert "reaper noise" not in result.stderr.str()


def test_a_failing_test_prints_the_tails_from_the_failure_hook_alone(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stall timer cannot fire in a test with a 60 s limit, so the hook prints these."""
    source = "@pytest.mark.timeout(60)\ndef test_fails():\n    assert False\n"
    result = _run_inner(pytester, monkeypatch, tmp_path, source)

    result.assert_outcomes(failed=1)
    result.stderr.fnmatch_lines(["*Engine logs*", "*engine says: model stalled*"])


def test_a_stalled_test_prints_the_engine_log_tails_before_it_ends(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A test still running 75% into its timeout has printed the tails, and it passes."""
    source = (
        "@pytest.mark.timeout(8)\n"
        "def test_waits_for_the_print():\n"
        "    wait_for_emit(7)\n"
        "    assert Path(os.environ['EMIT_MARKER']).exists()\n"
    )
    result = _run_inner(pytester, monkeypatch, tmp_path, source)

    result.assert_outcomes(passed=1)
    result.stderr.fnmatch_lines(["*Engine logs*", "*engine says: model stalled*"])


def test_a_test_that_ends_early_prints_nothing_after_its_stall_window(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The first test ends at once; a timer it leaves behind would print inside the second."""
    source = (
        "@pytest.mark.timeout(2)\n"
        "def test_ends_at_once():\n"
        "    pass\n"
        "@pytest.mark.timeout(30)\n"
        "def test_outlives_the_first_window():\n"
        "    time.sleep(2.5)\n"
    )
    result = _run_inner(pytester, monkeypatch, tmp_path, source)

    result.assert_outcomes(passed=2)
    assert "Engine logs" not in result.stderr.str()


def test_log_roots_name_the_machine_engine_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    roots = engine_logs.log_roots(None)
    assert roots[0] == tmp_path / "slot"
    assert engine_logs.log_roots(tmp_path / "base")[-1] == tmp_path / "base"
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    assert tmp_path / "data" in engine_logs.log_roots(None)


def test_emit_writes_text_to_stderr_and_ignores_none(capfd: pytest.CaptureFixture[str]) -> None:
    engine_logs.emit(None)
    engine_logs.emit("tail text")
    assert "tail text" in capfd.readouterr().err


def test_find_logs_skips_the_reaper_log(tmp_path: Path) -> None:
    _write_log(tmp_path, "engine-reaper.log", ["reaper"])
    _write_log(tmp_path, "engine-chat-0.log", ["engine"])
    assert [path.name for path in engine_logs.find_logs([tmp_path])] == ["engine-chat-0.log"]


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
