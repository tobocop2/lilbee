"""Tests for ensuring hermes has HTTP MCP support before the launcher wires lilbee in."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from lilbee.cli.launchers import hermes_mcp


def _script(tmp_path, shebang: str):
    p = tmp_path / "hermes"
    p.write_text(f"{shebang}\nprint('hermes')\n")
    return str(p)


def _ensure(binary, msgs, *, allow_lazy_installs=True):
    return hermes_mcp.ensure_hermes_http_mcp(
        binary, allow_lazy_installs=allow_lazy_installs, echo=msgs.append
    )


def test_interpreter_read_from_shebang(tmp_path):
    binary = _script(tmp_path, "#!/opt/venv/bin/python3")
    assert hermes_mcp.hermes_interpreter(binary) == "/opt/venv/bin/python3"


def test_interpreter_none_for_non_python_shebang(tmp_path):
    assert hermes_mcp.hermes_interpreter(_script(tmp_path, "#!/bin/sh")) is None


def test_interpreter_none_for_missing_binary():
    assert hermes_mcp.hermes_interpreter("/no/such/hermes") is None


def test_has_http_mcp_true_on_success():
    with patch.object(hermes_mcp.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
        assert hermes_mcp.has_http_mcp("/py") is True


def test_has_http_mcp_false_on_import_error():
    with patch.object(hermes_mcp.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
        assert hermes_mcp.has_http_mcp("/py") is False


def test_extra_requirements_reads_pinned_extra():
    out = '["mcp==1.26.0", "starlette==1.0.1"]'
    with patch.object(hermes_mcp.subprocess, "run", return_value=SimpleNamespace(stdout=out)):
        assert hermes_mcp._mcp_extra_requirements("/py") == ["mcp==1.26.0", "starlette==1.0.1"]


def test_extra_requirements_falls_back_to_mcp_when_unreadable():
    with patch.object(hermes_mcp.subprocess, "run", return_value=SimpleNamespace(stdout="")):
        assert hermes_mcp._mcp_extra_requirements("/py") == ["mcp"]


def test_ensure_noops_when_already_supported(tmp_path):
    binary = _script(tmp_path, "#!/opt/venv/bin/python")
    msgs: list[str] = []
    with (
        patch.object(hermes_mcp, "has_http_mcp", return_value=True),
        patch.object(hermes_mcp.subprocess, "run") as run,
    ):
        assert _ensure(binary, msgs) is True
    run.assert_not_called()  # already supported -> no install attempt
    assert msgs == []


def test_ensure_installs_pinned_extra_when_missing_and_allowed(tmp_path):
    binary = _script(tmp_path, "#!/opt/venv/bin/python")
    msgs: list[str] = []
    with (
        patch.object(hermes_mcp, "has_http_mcp", side_effect=[False, True]),
        patch.object(hermes_mcp, "_mcp_extra_requirements", return_value=["mcp==1.26.0"]),
        patch.object(hermes_mcp.subprocess, "run") as run,
    ):
        assert _ensure(binary, msgs) is True
    # installs hermes's pinned extra, not an unpinned `mcp`
    assert run.call_args[0][0][1:] == ["-m", "pip", "install", "mcp==1.26.0"]
    assert any("Setting up hermes MCP support" in m for m in msgs)
    assert any("ready" in m for m in msgs)


def test_ensure_respects_security_gate_and_guides(tmp_path):
    binary = _script(tmp_path, "#!/opt/venv/bin/python")
    msgs: list[str] = []
    with (
        patch.object(hermes_mcp, "has_http_mcp", return_value=False),
        patch.object(hermes_mcp.subprocess, "run") as run,
    ):
        assert _ensure(binary, msgs, allow_lazy_installs=False) is False
    run.assert_not_called()  # gate off -> never pip-installs behind hermes's setting
    assert any("hermes-agent[mcp]" in m for m in msgs)


def test_ensure_guides_when_install_did_not_take(tmp_path):
    binary = _script(tmp_path, "#!/opt/venv/bin/python")
    msgs: list[str] = []
    with (
        patch.object(hermes_mcp, "has_http_mcp", side_effect=[False, False]),
        patch.object(hermes_mcp, "_mcp_extra_requirements", return_value=["mcp"]),
        patch.object(hermes_mcp.subprocess, "run"),
    ):
        assert _ensure(binary, msgs) is False
    assert any("hermes-agent[mcp]" in m for m in msgs)


def test_ensure_guides_when_interpreter_unknown(tmp_path):
    binary = _script(tmp_path, "#!/bin/sh")  # not a python shebang
    msgs: list[str] = []
    assert _ensure(binary, msgs) is False
    assert any("hermes-agent[mcp]" in m for m in msgs)


def test_has_http_mcp_false_on_oserror():
    with patch.object(hermes_mcp.subprocess, "run", side_effect=OSError("no such file")):
        assert hermes_mcp.has_http_mcp("/py") is False


def test_extra_requirements_falls_back_on_oserror():
    with patch.object(hermes_mcp.subprocess, "run", side_effect=OSError("no such file")):
        assert hermes_mcp._mcp_extra_requirements("/py") == ["mcp"]


def test_ensure_guides_on_install_oserror(tmp_path):
    binary = _script(tmp_path, "#!/opt/venv/bin/python")
    msgs: list[str] = []
    with (
        patch.object(hermes_mcp, "has_http_mcp", return_value=False),
        patch.object(hermes_mcp, "_mcp_extra_requirements", return_value=["mcp"]),
        patch.object(hermes_mcp.subprocess, "run", side_effect=OSError("no such file")),
    ):
        assert _ensure(binary, msgs) is False
    assert any("hermes-agent[mcp]" in m for m in msgs)


def test_interpreter_none_for_no_shebang_non_windows(tmp_path):
    """A binary with no python shebang returns None on non-Windows platforms."""
    binary = _script(tmp_path, "#!/bin/sh")
    # Explicitly confirm non-Windows behavior: no shebang -> None
    with patch.object(hermes_mcp.sys, "platform", "linux"):
        assert hermes_mcp.hermes_interpreter(str(binary)) is None


def test_interpreter_resolved_for_windows_cmd(tmp_path, monkeypatch):
    """On Windows, a .cmd wrapper with an embedded python path resolves the interpreter."""
    cmd = tmp_path / "hermes.cmd"
    cmd.write_text('@"C:\\venv\\Scripts\\python.exe" "%~dp0hermes-script.py" %*\n')
    monkeypatch.setattr(hermes_mcp.sys, "platform", "win32")
    result = hermes_mcp.hermes_interpreter(str(cmd))
    assert result == "C:\\venv\\Scripts\\python.exe"


def test_interpreter_falls_back_to_sys_executable_for_windows_exe(tmp_path, monkeypatch):
    """On Windows, a .exe wrapper with no parseable python path falls back to sys.executable."""
    exe = tmp_path / "hermes.exe"
    exe.write_bytes(b"\x4d\x5a")  # MZ header; no shebang, no @"python" line
    monkeypatch.setattr(hermes_mcp.sys, "platform", "win32")
    result = hermes_mcp.hermes_interpreter(str(exe))
    assert result == hermes_mcp.sys.executable
