"""``lilbee launch <client>``: start a server if needed, wire a config, exec the client."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

import httpx
import typer

from lilbee.cli.agent_configs.opencode import opencode_config
from lilbee.cli.app import console
from lilbee.cli.commands.agent_config import installed_chat_model_refs, running_server_session

launch_app = typer.Typer(help="Launch a third-party AI client wired to lilbee.")
log = logging.getLogger(__name__)

_LOCAL_HOST = "127.0.0.1"
_SERVER_BOOT_TIMEOUT_S = 60.0
_SERVER_POLL_INTERVAL_S = 0.5
_HTTP_OK = 200
_OPENCODE_INSTALL_HINT = (
    "opencode binary not found. Install it with: npm i -g opencode-ai "
    "(or: brew install sst/tap/opencode)."
)
_SKILL_PACKAGE = "lilbee.skills.lilbee_mcp"
_OPENCODE_STATE_RECENT_CAP = 10
_OPENCODE_PROVIDER_ID = "lilbee"


def _opencode_skill_dest() -> Path:
    return Path.home() / ".config" / "opencode" / "skills" / "lilbee-mcp"


def _opencode_state_file() -> Path:
    return Path.home() / ".local" / "state" / "opencode" / "model.json"


class OpencodeNotInstalledError(Exception):
    """opencode binary is not on PATH."""


def _opencode_binary() -> str:
    path = shutil.which("opencode")
    if path is None:
        raise OpencodeNotInstalledError
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_LOCAL_HOST, 0))
        return int(s.getsockname()[1])


def _wait_for_health(port: int, timeout_s: float = _SERVER_BOOT_TIMEOUT_S) -> bool:
    url = f"http://{_LOCAL_HOST}:{port}/api/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == _HTTP_OK:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(_SERVER_POLL_INTERVAL_S)
    return False


def _spawn_server(port: int) -> subprocess.Popen[bytes]:
    # Command line built from sys.executable plus literal flags; only caller-
    # controlled value is the integer port. No untrusted input.
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "lilbee", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_spawned_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _install_lilbee_skill() -> Path | None:
    """Copy the bundled lilbee MCP skill into opencode's global skills dir.

    Skip when the destination already exists so user customizations are
    preserved. Returns the destination path on a fresh copy, else ``None``.
    """
    dest = _opencode_skill_dest()
    if dest.exists():
        return None
    dest.mkdir(parents=True)
    source = resources.files(_SKILL_PACKAGE)
    for entry in source.iterdir():
        if entry.is_file() and not entry.name.startswith("__"):
            (dest / entry.name).write_bytes(entry.read_bytes())
    return dest


def _update_opencode_picker_state(model_refs: list[str]) -> Path | None:
    """Make lilbee models appear in opencode's model picker on first run.

    Reads opencode's ``model.json`` state file (best-effort parse), prepends
    each lilbee model under the ``recent`` list, and writes the result back
    atomically. Skipped on Windows where opencode stores state elsewhere.
    Returns the state path on success, ``None`` on skip or failure.
    """
    if sys.platform.startswith("win") or not model_refs:
        return None
    path = _opencode_state_file()
    state = _read_opencode_state(path)
    state["recent"] = _merge_recent(state.get("recent"), model_refs)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, state)
    return path


def _read_opencode_state(path: Path) -> dict:
    fallback: dict = {"recent": [], "favorite": [], "variant": {}}
    if not path.exists():
        return fallback
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback
    return loaded if isinstance(loaded, dict) else fallback


def _merge_recent(existing: object, model_refs: list[str]) -> list[dict]:
    """Prepend lilbee entries, drop stale lilbee entries, cap the list length."""
    prior: list = existing if isinstance(existing, list) else []
    new_set = set(model_refs)
    kept = [
        entry
        for entry in prior
        if not (
            isinstance(entry, dict)
            and entry.get("providerID") == _OPENCODE_PROVIDER_ID
            and entry.get("modelID") in new_set
        )
    ]
    fresh = [{"providerID": _OPENCODE_PROVIDER_ID, "modelID": ref} for ref in model_refs]
    return (fresh + kept)[:_OPENCODE_STATE_RECENT_CAP]


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _ensure_server_running(port: int) -> tuple[tuple[str, int], subprocess.Popen[bytes] | None]:
    """Return ``(session, spawned_proc)``. Spawns lilbee serve if not already running."""
    existing = running_server_session()
    if existing is not None:
        return existing, None
    chosen_port = port if port > 0 else _free_port()
    console.print(f"Starting lilbee server on port {chosen_port}...")
    spawned = _spawn_server(chosen_port)
    if not _wait_for_health(chosen_port):
        _stop_spawned_server(spawned)
        typer.secho(
            f"lilbee server failed to start on port {chosen_port}; check the logs.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    session = running_server_session()
    if session is None:
        _stop_spawned_server(spawned)
        typer.secho(
            "lilbee server started but did not write a session file; cannot continue.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    return session, spawned


def _resolve_opencode_or_exit() -> str:
    try:
        return _opencode_binary()
    except OpencodeNotInstalledError:
        typer.secho(_OPENCODE_INSTALL_HINT, err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from None


def _opencode_env(token: str, server_port: int, model_refs: list[str]) -> dict[str, str]:
    block = opencode_config(
        base_url=f"http://{_LOCAL_HOST}:{server_port}",
        api_key=token,
        model_refs=model_refs,
    )
    return {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(block)}


def _run_opencode(
    opencode_bin: str,
    env: dict[str, str],
    spawned: subprocess.Popen[bytes] | None,
    *,
    keep_serving: bool,
) -> int:
    try:
        # opencode_bin resolved via shutil.which on PATH; no shell interpolation.
        result = subprocess.run([opencode_bin], env=env, check=False)  # noqa: S603
    finally:
        if spawned is not None and not keep_serving:
            _stop_spawned_server(spawned)
    return result.returncode


@launch_app.command("opencode")
def opencode_cmd(
    keep_serving: bool = typer.Option(
        False,
        "--keep-serving",
        help="Do not stop the spawned lilbee server when opencode exits.",
    ),
    port: int = typer.Option(
        0,
        "--port",
        "-p",
        help=(
            "Port to bind lilbee serve on. 0 picks a free port. "
            "Ignored when a server is already running."
        ),
    ),
) -> None:
    """Launch opencode with lilbee as its model provider."""
    opencode_bin = _resolve_opencode_or_exit()
    (token, server_port), spawned = _ensure_server_running(port)
    model_refs = installed_chat_model_refs()
    _install_lilbee_skill()
    _update_opencode_picker_state(model_refs)
    env = _opencode_env(token, server_port, model_refs)
    exit_code = _run_opencode(opencode_bin, env, spawned, keep_serving=keep_serving)
    raise typer.Exit(exit_code)
