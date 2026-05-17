"""End-to-end matrix: a real ``opencode`` binary against ``lilbee serve``.

The suite asserts protocol correctness, not coding quality. A tiny
tool-capable model (Qwen3 0.6B) is used because it is the smallest
model the catalog ships that completes the tool roundtrip.

All tests skip when ``LILBEE_QA_OPENCODE`` is unset or ``opencode`` is
not on ``PATH``; the gating lives in the session-scoped fixtures.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx
import pytest

from tests.qa.conftest import TOOL_CAPABLE_MODEL

_OPENCODE_TIMEOUT = 300.0


def _opencode_run(
    binary: str,
    config: Path,
    *,
    prompt: str,
    cwd: Path | None = None,
) -> str:
    """Invoke ``opencode run`` with the lilbee provider and return stdout."""
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(config)
    result = subprocess.run(
        [binary, "run", "--model", f"lilbee/{TOOL_CAPABLE_MODEL}", prompt],
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=_OPENCODE_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "opencode run exited with "
            f"{result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


def _tool_capable_model_installed(base_url: str, headers: dict[str, str]) -> bool:
    """True iff the tool-capable model is listed in ``/v1/models``."""
    response = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=10.0)
    if response.status_code != httpx.codes.OK:
        return False
    return any(entry.get("id") == TOOL_CAPABLE_MODEL for entry in response.json()["data"])


def test_opencode_lists_models_via_v1_models(
    lilbee_serve: str,
    opencode_config: Path,
    auth_headers: dict[str, str],
) -> None:
    """``/v1/models`` returns the same model ids opencode sees in its config."""
    response = httpx.get(f"{lilbee_serve}/v1/models", headers=auth_headers, timeout=10.0)
    assert response.status_code == httpx.codes.OK
    body = response.json()
    assert body["object"] == "list"
    server_refs = {entry["id"] for entry in body["data"]}

    import json as _json

    opencode_refs = set(
        _json.loads(opencode_config.read_text())["provider"]["lilbee"]["models"].keys()
    )
    assert server_refs == opencode_refs, (
        f"opencode config and /v1/models disagree: "
        f"server={sorted(server_refs)} opencode={sorted(opencode_refs)}"
    )


def test_opencode_completes_a_simple_prompt(
    opencode_binary: str,
    opencode_config: Path,
    lilbee_serve: str,
    auth_headers: dict[str, str],
) -> None:
    """opencode completes a no-tool prompt end-to-end."""
    if not _tool_capable_model_installed(lilbee_serve, auth_headers):
        pytest.skip(f"{TOOL_CAPABLE_MODEL} not installed; pull it to run this test")
    output = _opencode_run(opencode_binary, opencode_config, prompt="say hi in one word")
    assert any(line.strip() for line in output.splitlines()), (
        f"opencode produced no non-empty output: {output!r}"
    )


def test_opencode_lists_files_using_tools(
    opencode_binary: str,
    opencode_config: Path,
    lilbee_serve: str,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    """opencode invokes a tool call and the model names the directory's files."""
    if not _tool_capable_model_installed(lilbee_serve, auth_headers):
        pytest.skip(f"{TOOL_CAPABLE_MODEL} not installed; pull it to run this test")
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.txt").write_text("b")
    output = _opencode_run(
        opencode_binary,
        opencode_config,
        prompt="list files in this directory",
        cwd=tmp_path,
    )
    # Tiny models call tools but sometimes drop one filename in the
    # final summary. Pass if at least one of the two seeded names
    # shows up; this still proves the tool roundtrip executed.
    found = [name for name in ("alpha.txt", "beta.txt") if name in output]
    assert found, f"opencode output mentions neither alpha.txt nor beta.txt:\n{output}"


def test_opencode_reads_a_file_using_tools(
    opencode_binary: str,
    opencode_config: Path,
    lilbee_serve: str,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    """opencode invokes a read tool and surfaces the file's contents."""
    if not _tool_capable_model_installed(lilbee_serve, auth_headers):
        pytest.skip(f"{TOOL_CAPABLE_MODEL} not installed; pull it to run this test")
    secret = "magic-token-87234"
    (tmp_path / "secret.txt").write_text(secret)
    output = _opencode_run(
        opencode_binary,
        opencode_config,
        prompt="what's in secret.txt?",
        cwd=tmp_path,
    )
    assert secret in output, f"opencode output does not contain the secret value:\n{output}"
