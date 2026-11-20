"""The engine-log check reads the process table: it asks real processes what they hold open."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.conformance import record_presets

# Prints the port once the socket listens and the file, when one is named, is open.
_LISTEN = (
    "import socket, sys, time\n"
    "s = socket.socket()\n"
    "s.bind(('127.0.0.1', 0))\n"
    "s.listen(1)\n"
    "held = open(sys.argv[1], 'a') if sys.argv[1] else None\n"
    "print(s.getsockname()[1], flush=True)\n"
    "time.sleep(300)\n"
)
# Prints once the file is open; listens on nothing.
_HOLD = (
    "import sys, time\nheld = open(sys.argv[1], 'a')\nprint('held', flush=True)\ntime.sleep(300)\n"
)
_NOT_HELD = "does not hold this file open"


@contextmanager
def _process(code: str, held: Path | None) -> Iterator[subprocess.Popen[str]]:
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(held or "")],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        yield child
    finally:
        child.kill()
        child.wait()
        assert child.stdout is not None
        child.stdout.close()


def _ready(child: subprocess.Popen[str]) -> str:
    assert child.stdout is not None
    return child.stdout.readline().strip()


def _url(listener: subprocess.Popen[str]) -> str:
    return f"http://127.0.0.1:{_ready(listener)}"


def _log(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "engine.log"
    path.write_text("startup output", encoding="utf-8")
    return path


def test_the_listener_that_holds_the_log_is_accepted(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with _process(_LISTEN, log) as listener:
        record_presets.require_engine_output(_url(listener), log)


def test_a_hard_link_to_the_held_file_is_the_same_file(tmp_path: Path) -> None:
    held = _log(tmp_path)
    link = tmp_path / "link.log"
    link.hardlink_to(held)
    with _process(_LISTEN, held) as listener:
        record_presets.require_engine_output(_url(listener), link)


def test_a_listener_that_does_not_hold_the_log_is_refused(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with _process(_LISTEN, None) as listener, pytest.raises(SystemExit, match=_NOT_HELD):
        record_presets.require_engine_output(_url(listener), log)


def test_a_log_held_by_a_process_that_does_not_listen_is_refused(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with _process(_HOLD, log) as holder, _process(_LISTEN, None) as listener:
        assert _ready(holder) == "held"
        with pytest.raises(SystemExit, match=_NOT_HELD):
            record_presets.require_engine_output(_url(listener), log)


def test_a_log_held_by_a_listener_on_another_port_is_refused(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with _process(_LISTEN, log) as other, _process(_LISTEN, None) as engine:
        assert _ready(other)
        with pytest.raises(SystemExit, match=_NOT_HELD):
            record_presets.require_engine_output(_url(engine), log)


def test_a_file_of_the_same_name_in_another_folder_is_refused(tmp_path: Path) -> None:
    held = _log(tmp_path / "held")
    other = _log(tmp_path / "other")
    with _process(_LISTEN, held) as listener, pytest.raises(SystemExit, match=_NOT_HELD):
        record_presets.require_engine_output(_url(listener), other)


def test_a_log_that_does_not_exist_is_refused_not_raised(tmp_path: Path) -> None:
    """A file that cannot be read is a refusal, never an OSError out of the check."""
    held = _log(tmp_path)
    with _process(_LISTEN, held) as listener, pytest.raises(SystemExit, match=_NOT_HELD):
        record_presets.require_engine_output(_url(listener), tmp_path / "missing.log")
