"""`lilbee launch <client>`: start a server if needed, wire a config, exec the client."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import typer

from lilbee.cli.agent_configs.opencode import LILBEE_PRIMING, opencode_config
from lilbee.cli.app import console
from lilbee.cli.commands.agent_config import _chat_model_refs, _server_session

launch_app = typer.Typer(help="Launch a third-party AI client wired to lilbee.")

_LOCAL_HOST = "127.0.0.1"
_MCP_COMMAND = ["lilbee", "mcp"]
_SERVER_BOOT_TIMEOUT_S = 60.0
_SERVER_POLL_INTERVAL_S = 0.5
_HTTP_OK = 200


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
    # Fixed command line built from sys.executable plus literal flags; the only
    # caller-controlled value is the validated integer port. No untrusted input.
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "lilbee", "serve", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_opencode_config(config_dir: Path, *, base_url: str, api_key: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    priming_path = config_dir / "AGENTS.md"
    priming_path.write_text(LILBEE_PRIMING)
    block = opencode_config(
        base_url=base_url,
        api_key=api_key,
        model_refs=_chat_model_refs(),
        mcp_command=_MCP_COMMAND,
        instructions_paths=[str(priming_path)],
    )
    target = config_dir / "opencode.json"
    target.write_text(json.dumps(block, indent=2))
    return target


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
    """Launch opencode with lilbee as its model provider.

    Reuses an already-running ``lilbee serve`` when one exists. Otherwise spawns
    a server on a free port for the duration of the opencode session and stops
    it on exit. Writes a temporary ``opencode.json`` that points opencode at
    the running server and at the lilbee MCP tools.
    """
    try:
        opencode_bin = _opencode_binary()
    except OpencodeNotInstalledError:
        typer.secho(
            "opencode binary not found. Install it with: npm i -g opencode-ai "
            "(or: brew install sst/tap/opencode).",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from None

    existing = _server_session()
    spawned: subprocess.Popen[bytes] | None = None
    if existing is not None:
        token, server_port = existing
    else:
        chosen_port = port if port > 0 else _free_port()
        console.print(f"Starting lilbee server on port {chosen_port}...")
        spawned = _spawn_server(chosen_port)
        if not _wait_for_health(chosen_port):
            if spawned.poll() is None:
                spawned.terminate()
                spawned.wait(timeout=10)
            typer.secho(
                f"lilbee server failed to start on port {chosen_port}; check the logs.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        session = _server_session()
        if session is None:
            if spawned.poll() is None:
                spawned.terminate()
                spawned.wait(timeout=10)
            typer.secho(
                "lilbee server started but did not write a session file; cannot continue.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        token, server_port = session

    base_url = f"http://{_LOCAL_HOST}:{server_port}"
    config_dir = Path.home() / ".cache" / "lilbee" / "opencode"
    config_path = _write_opencode_config(config_dir, base_url=base_url, api_key=token)

    env = {**os.environ, "OPENCODE_CONFIG": str(config_path)}
    try:
        # opencode_bin resolved via shutil.which on PATH; no shell interpolation.
        result = subprocess.run([opencode_bin], env=env, check=False)  # noqa: S603
    finally:
        if spawned is not None and not keep_serving and spawned.poll() is None:
            spawned.terminate()
            try:
                spawned.wait(timeout=10)
            except subprocess.TimeoutExpired:
                spawned.kill()
                spawned.wait(timeout=5)
    raise typer.Exit(result.returncode)
