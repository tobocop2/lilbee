"""Driving opencode: tool scoping, model pinning, tmux control, session reset."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from harness_config import (
    _OPENCODE_BOOT_SETTLE_S,
    _OPENCODE_PICKER_STATE,
    _OPENCODE_SHARE_DIR,
    _OPENCODE_UI_TIMEOUT_S,
    _PANE_EXCERPT_TAIL,
    _POLL_INTERVAL_S,
    _POST_SEND_SLEEP_S,
    _TMUX_COMMAND_TIMEOUT_S,
    _TMUX_HISTORY_LINES,
    _TMUX_WINDOW_COLS,
    _TMUX_WINDOW_ROWS,
    _TOOLS_OFF,
    _UI_WAIT_HEARTBEAT_S,
)


def scope_opencode_tools(workspace: Path) -> None:
    """Disable opencode's built-in tools for the cell so the model uses lilbee_search.

    Models drift to opencode's built-in webfetch/read/grep over the lilbee MCP
    search unless those are turned off (search mode). Written as the cell
    workspace's project-level ``opencode.json`` (opencode merges it below the
    launcher's injected env config), never the user's global config: a global
    write outlives the QA run and disables the developer's own opencode tools.
    ``autoupdate`` is pinned off so the binary cannot change mid-matrix.
    """
    config = {
        "$schema": "https://opencode.ai/config.json",
        "tools": {tool: False for tool in _TOOLS_OFF},
        "autoupdate": False,
    }
    (workspace / "opencode.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def tmux_kill(name: str) -> None:
    if tmux_session_exists(name):
        subprocess.run(["tmux", "kill-session", "-t", name], check=False)


def tmux_capture(name: str) -> str:
    # A wedged tmux server blocks capture-pane forever, which freezes the whole
    # matrix without a single log line; bound it and treat a hang as "no pane".
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", name, "-p", "-S", f"-{_TMUX_HISTORY_LINES}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_TMUX_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"tmux capture-pane timed out after {_TMUX_COMMAND_TIMEOUT_S:.0f}s for {name}")
        return ""
    return result.stdout if result.returncode == 0 else ""


def tmux_send(name: str, keys: str) -> None:
    subprocess.run(["tmux", "send-keys", "-t", name, keys], check=False)
    time.sleep(_POST_SEND_SLEEP_S)
    subprocess.run(["tmux", "send-keys", "-t", name, "Enter"], check=False)


def launch_opencode_in_tmux(workspace: Path, session: str) -> None:
    """Boot opencode in a tmux session pinned to the per-cell lilbee data dir.

    The ``LILBEE_DATA=workspace/.lilbee`` env override is critical: matrix.py
    imports ``lilbee.core.config``, whose module-import side effect sets
    ``LILBEE_DATA`` in matrix.py's own env to the GLOBAL data root. Every
    tmux session and subprocess matrix.py spawns inherits that polluted env,
    so the launched lilbee serve would read the global config.toml (default
    chat_model=Qwen3-0.6B) instead of the workspace's. Setting the env
    explicitly per cell breaks the inheritance: the launched serve resolves
    workspace/.lilbee/config.toml, picks up chat_model=<cell.ref>, and the
    worker pool spawns with the right model. See bb-hef0.
    """
    import os

    tmux_kill(session)
    workspace_data = workspace / ".lilbee"
    env_flags = ["-e", f"LILBEE_DATA={workspace_data}"]
    # The session runs `bash -lc`, a login shell that does not inherit the
    # matrix's environment, so anything the cell's `uv run` needs must be
    # forwarded explicitly: the models dir (or the launcher serves no models)
    # and the uv project env (or `uv run` syncs a FRESH venv whose lilbee-engine
    # wheel is the empty placeholder, leaving the serve with no llama-server).
    for env_var in (
        "LILBEE_MODELS_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "UV_LINK_MODE",
        "UV_NO_SYNC",
    ):
        value = os.environ.get(env_var)
        if value:
            env_flags += ["-e", f"{env_var}={value}"]
    # Forward QA diagnostic flags into the launched serve + its worker
    # subprocesses (multiprocessing-spawn inherits the tmux session env), so
    # LILBEE_QA_LOG_RAW reaches the chat worker where the raw-output tap lives.
    if os.environ.get("LILBEE_QA_LOG_RAW"):
        env_flags += ["-e", "LILBEE_QA_LOG_RAW=1"]
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            str(_TMUX_WINDOW_COLS),
            "-y",
            str(_TMUX_WINDOW_ROWS),
            *env_flags,
            "bash",
            "-lc",
            f"cd {workspace} && exec uv run lilbee launch opencode --no-prompt",
        ],
        check=True,
    )
    _wait_for_opencode_ui(session, workspace)


def _pane_in_alternate_screen(session: str) -> bool:
    """True once the pane's terminal is in the alternate screen.

    A full-screen TUI flips the terminal into the alternate screen when it
    takes over; the launcher's inline warm spinner never does. tmux exposes
    the flag directly, so this is a content- and version-independent "the TUI
    has painted" signal (footer text shifts between opencode releases).
    """
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "-t", session, "#{alternate_on}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_TMUX_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and result.stdout.strip() == "1"


def _wait_for_opencode_ui(session: str, workspace: Path) -> None:
    """Block until opencode is up (first tap record or alternate-screen flip), then settle.

    The wait must outlast the launcher's warm spinner (a giant's cold load runs
    minutes); the heartbeat keeps the wait visible in the matrix log.
    """
    from events import plugin_active

    deadline = time.monotonic() + _OPENCODE_UI_TIMEOUT_S
    started = time.monotonic()
    next_heartbeat = started + _UI_WAIT_HEARTBEAT_S
    while time.monotonic() < deadline:
        if plugin_active(workspace) or _pane_in_alternate_screen(session):
            time.sleep(_OPENCODE_BOOT_SETTLE_S)
            return
        if time.monotonic() >= next_heartbeat:
            elapsed = time.monotonic() - started
            pane = tmux_capture(session)
            tail = " | ".join(pane.strip().splitlines()[-2:]) if pane.strip() else "(empty pane)"
            print(f"waiting for opencode UI ({elapsed:.0f}s): {tail}")
            next_heartbeat = time.monotonic() + _UI_WAIT_HEARTBEAT_S
        time.sleep(_POLL_INTERVAL_S)
    raise RuntimeError(
        f"opencode TUI did not appear within {_OPENCODE_UI_TIMEOUT_S:.0f}s; "
        f"pane tail: {tmux_capture(session)[-_PANE_EXCERPT_TAIL:]}"
    )


def reset_opencode_session_state() -> None:
    """Wipe opencode's per-user state so the prior cell can't bleed into the new pane.

    Two persistence locations need scrubbing:

    1. ``~/.local/share/opencode/`` -- session DB + storage. Holds the prior
       cell's conversation transcripts. Without the wipe opencode's recent-
       sessions panel surfaces tokens from the previous PASS (e.g.
       ``KnownModelCache``) and the next cell's smoke matches them without
       its own model ever loading.

    2. ``~/.local/state/opencode/model.json`` -- the model-selection state.
       The launcher pins the boot model via the injected config, but a stale
       recent/variant selection from the prior cell is one more input opencode
       may consult, so each cell starts from none.
    """
    if _OPENCODE_SHARE_DIR.exists():
        shutil.rmtree(_OPENCODE_SHARE_DIR)
    if _OPENCODE_PICKER_STATE.exists():
        _OPENCODE_PICKER_STATE.unlink()
