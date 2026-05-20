"""Opencode launcher: wires the inline config, installs the skill, runs opencode."""

from __future__ import annotations

import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Any, TypedDict

from lilbee.cli.agent_configs.opencode import opencode_config
from lilbee.cli.launchers.launcher import run_launcher
from lilbee.cli.launchers.server import LOOPBACK

_OPENCODE_INSTALL_HINT = "opencode binary not found on PATH. Install it from https://opencode.ai/."
_SKILL_PACKAGE = "lilbee.skills.lilbee_mcp"
_OPENCODE_PROVIDER_ID = "lilbee"
_OPENCODE_CONFIG_ENV_VAR = "OPENCODE_CONFIG_CONTENT"
_PICKER_STATE_RECENT_CAP = 10


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

    Skipped on Windows where opencode stores state at a different path.
    """
    if sys.platform.startswith("win") or not model_refs:
        return None
    path = _opencode_state_file()
    state = _read_opencode_state(path)
    state["recent"] = _merge_recent(state.get("recent"), model_refs)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, state)
    return path


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

    def find_binary(self) -> str | None:
        return shutil.which("opencode")

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        _install_lilbee_skill()
        _update_opencode_picker_state(model_refs)
        block = opencode_config(
            base_url=f"http://{LOOPBACK}:{port}",
            api_key=token,
            model_refs=model_refs,
        )
        provider_block = block["provider"][_OPENCODE_PROVIDER_ID]
        _merge_lilbee_provider_into_config(
            config_path=_opencode_config_path(),
            provider_block=provider_block,
        )
        env = {**os.environ, _OPENCODE_CONFIG_ENV_VAR: json.dumps(block)}
        return ([], env)


def opencode_cmd() -> None:
    """Launch opencode with lilbee as its model provider."""
    run_launcher(OpencodeLauncher())
