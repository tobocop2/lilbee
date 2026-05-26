"""Tests for ``lilbee.cli.launchers.server.spawn_server`` stdout/stderr wiring.

``subprocess.Popen`` is patched so no real ``lilbee serve`` process is spawned;
the tests assert how the child's stdout/stderr are routed (DEVNULL in quiet
mode, a size-capped log file otherwise).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from lilbee.cli.launchers import server as server_mod
from lilbee.core.config import cfg


@pytest.fixture()
def _capture_popen():
    """Patch Popen to record the stdout/stderr kwargs without spawning."""
    captured: dict = {}

    def fake_popen(cmd, *, stdout, stderr):
        captured["cmd"] = cmd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return mock.MagicMock()

    with mock.patch.object(server_mod.subprocess, "Popen", side_effect=fake_popen):
        yield captured


def test_quiet_mode_routes_output_to_devnull(monkeypatch, _capture_popen) -> None:
    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
    server_mod.spawn_server(8080)
    assert _capture_popen["stdout"] == subprocess.DEVNULL
    assert _capture_popen["stderr"] == subprocess.DEVNULL


def test_non_quiet_writes_to_log_file(monkeypatch, tmp_path: Path, _capture_popen) -> None:
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    server_mod.spawn_server(8080)
    log_path = tmp_path / "logs" / "launcher-serve.log"
    assert log_path.exists()  # the log dir + file were created
    # stdout is an open binary file handle, stderr is folded into it.
    assert _capture_popen["stdout"].name == str(log_path)
    assert _capture_popen["stderr"] == subprocess.STDOUT
    _capture_popen["stdout"].close()


def test_oversized_log_is_truncated_before_reopen(
    monkeypatch, tmp_path: Path, _capture_popen
) -> None:
    """A pre-existing log past the 5 MB cap is unlinked, so the session starts fresh."""
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "launcher-serve.log"
    log_path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))  # one byte over the cap

    server_mod.spawn_server(8080)

    # The oversized file was unlinked then reopened empty (append mode, nothing
    # written by the patched Popen), so the stale firehose did not survive.
    assert log_path.stat().st_size == 0
    _capture_popen["stdout"].close()
