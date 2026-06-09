"""Opencode launcher: wires the inline config, installs the skill, runs opencode."""

from __future__ import annotations

import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Any, TypedDict

import typer

from lilbee.cli.agent_configs.opencode import opencode_config
from lilbee.cli.launchers.launcher import run_launcher
from lilbee.cli.launchers.server import LOOPBACK, served_chat_ctx
from lilbee.core.config import cfg

_OPENCODE_INSTALL_HINT = "opencode binary not found on PATH. Install it from https://opencode.ai/."
_SKILL_PACKAGE = "lilbee.skills.lilbee_mcp"
_OPENCODE_PROVIDER_ID = "lilbee"
_OPENCODE_CONFIG_ENV_VAR = "OPENCODE_CONFIG_CONTENT"
_PICKER_STATE_RECENT_CAP = 10
_SETUP_MARKER_NAME = "opencode-setup.json"


def _opencode_config_path() -> Path:
    """Path to opencode's persistent user config."""
    return Path.home() / ".config" / "opencode" / "opencode.json"


class PickerEntry(TypedDict):
    providerID: str
    modelID: str


class PickerState(TypedDict):
    recent: list[PickerEntry]
    favorite: list[PickerEntry]
    variant: dict[str, Any]


def _opencode_skill_dest() -> Path:
    return Path.home() / ".config" / "opencode" / "skills" / "lilbee-mcp"


def _opencode_state_file() -> Path:
    return Path.home() / ".local" / "state" / "opencode" / "model.json"


def _setup_marker_path() -> Path:
    """lilbee's record that opencode setup already ran (so launch doesn't re-prompt)."""
    return cfg.data_dir / "launchers" / _SETUP_MARKER_NAME


def _setup_recorded() -> bool:
    return _setup_marker_path().exists()


def _record_setup() -> None:
    """Persist that the user accepted opencode setup; idempotent."""
    path = _setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"accepted": True}), encoding="utf-8")


def _print_setup_plan() -> None:
    """Tell the user exactly which files the first-run setup writes."""
    typer.secho("First-time opencode setup will write:", fg=typer.colors.CYAN)
    typer.echo(f"  - lilbee MCP skill -> {_opencode_skill_dest()}")
    typer.echo(f"  - provider + MCP config -> {_opencode_config_path()}")
    typer.echo(f"  - model picker state -> {_opencode_state_file()}")
    typer.echo(
        "Each write is skipped if already present. To undo, delete the skill dir "
        "and the `lilbee` keys in opencode.json."
    )


def _is_interactive() -> bool:
    """True when stdin is a TTY, so a confirmation prompt can be answered."""
    return sys.stdin.isatty()


def _confirm_setup(assume_yes: bool) -> bool:
    """Prompt before the first opencode setup; True means proceed.

    Skipped when already recorded, when *assume_yes* is set, or when stdin is
    not a TTY (scripts/CI: invoking ``launch opencode`` is the consent there).
    The choice is remembered so later launches don't re-prompt.
    """
    if _setup_recorded():
        return True
    _print_setup_plan()
    if assume_yes or not _is_interactive():
        _record_setup()
        return True
    if not typer.confirm("Proceed with opencode setup?", default=True):
        typer.secho("Skipped opencode setup.", fg=typer.colors.YELLOW)
        return False
    _record_setup()
    return True


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


def _update_opencode_picker_state(model_refs: list[str], default_ref: str) -> Path | None:
    """Put lilbee models in opencode's picker, the configured chat model first.

    opencode opens on ``recent[0]``; leading with *default_ref* makes it select the
    chat model lilbee actually serves instead of the alphabetically-first installed
    one (which otherwise leaves opencode on its own fallback provider). No-op when no
    models are installed; same XDG-style state path on every platform.
    """
    if not model_refs:
        return None
    path = _opencode_state_file()
    state = _read_opencode_state(path)
    state["recent"] = _merge_recent(state.get("recent"), _default_first(model_refs, default_ref))
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, state)
    return path


def _default_first(model_refs: list[str], default_ref: str) -> list[str]:
    """Order so *default_ref* leads, leaving the rest in their existing order."""
    if default_ref not in model_refs:
        return model_refs
    return [default_ref, *(ref for ref in model_refs if ref != default_ref)]


def _read_opencode_state(path: Path) -> PickerState:
    fallback: PickerState = {"recent": [], "favorite": [], "variant": {}}
    if not path.exists():
        return fallback
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    if not isinstance(loaded, dict):
        return fallback
    return PickerState(
        recent=loaded.get("recent") or [],
        favorite=loaded.get("favorite") or [],
        variant=loaded.get("variant") or {},
    )


def _merge_recent(existing: object, model_refs: list[str]) -> list[PickerEntry]:
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
    fresh: list[PickerEntry] = [
        {"providerID": _OPENCODE_PROVIDER_ID, "modelID": ref} for ref in model_refs
    ]
    return (fresh + kept)[:_PICKER_STATE_RECENT_CAP]


def _atomic_write_json(path: Path, payload: PickerState) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _merge_lilbee_provider_into_config(
    *,
    config_path: Path,
    provider_block: dict[str, Any],
) -> None:
    """Write the lilbee provider into opencode's persistent config file.

    Picker rendering is driven by the on-disk ``opencode.json``; the
    env-var injection alone is not enough for opencode to render a
    section header for our provider. Existing providers (ollama, etc.)
    and any other top-level config keys (``plugin``, ``$schema``) are
    preserved; only ``provider.lilbee`` is replaced. The file is created
    if absent.
    """
    existing: dict[str, Any] = {"$schema": "https://opencode.ai/config.json"}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded
    providers = existing.get("provider")
    if not isinstance(providers, dict):
        # ``provider`` is missing or the wrong shape; reset so the merge writes
        # a valid provider section rather than silently dropping the lilbee entry.
        providers = {}
        existing["provider"] = providers
    providers[_OPENCODE_PROVIDER_ID] = provider_block
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    os.replace(tmp, config_path)


class OpencodeLauncher:
    """``Launcher`` implementation for opencode (https://opencode.ai/)."""

    name = "opencode"
    install_hint = _OPENCODE_INSTALL_HINT

    def __init__(self, *, assume_yes: bool = False) -> None:
        self._assume_yes = assume_yes

    def find_binary(self) -> str | None:
        return shutil.which("opencode")

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        if not _confirm_setup(self._assume_yes):
            raise typer.Exit(0)
        _install_lilbee_skill()
        _update_opencode_picker_state(model_refs, str(cfg.chat_model))
        block = opencode_config(
            base_url=f"http://{LOOPBACK}:{port}",
            api_key=token,
            model_refs=model_refs,
            chat_ctx=served_chat_ctx(port),
        )
        provider_block = block["provider"][_OPENCODE_PROVIDER_ID]
        _merge_lilbee_provider_into_config(
            config_path=_opencode_config_path(),
            provider_block=provider_block,
        )
        env = {**os.environ, _OPENCODE_CONFIG_ENV_VAR: json.dumps(block)}
        return ([], env)


def opencode_cmd(
    yes: bool = typer.Option(
        False,
        "--no-prompt",
        "--yes",
        "-y",
        help="Proceed with first-run setup without the interactive prompt (for scripts/CI).",
    ),
) -> None:
    """Launch opencode with lilbee as its model provider."""
    run_launcher(OpencodeLauncher(assume_yes=yes))
