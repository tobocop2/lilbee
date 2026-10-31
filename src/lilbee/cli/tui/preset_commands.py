"""The /preset slash command: text in, text out, over the preset service."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from lilbee.app.presets import (
    ServerPreset,
    add_preset,
    edit_preset,
    failure_text,
    list_presets,
    remove_preset,
    split_roles,
)
from lilbee.cli.tui import messages as msg
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import RerankWire

_COMMAND = "/preset"
_KEY_FLAG = "--key"
_VALUE_FLAGS = ("--url", "--roles", "--wire")
_CLEAR_FLAG = "--clear-key"
_VERB_AND_NAME = 2
_NAME_AND_URL = 2


@dataclass(frozen=True)
class PresetCommandResult:
    """What a /preset command did: whether it worked, and the line to show."""

    ok: bool
    text: str


@dataclass(frozen=True)
class DialogRequest:
    """A /preset form that opens a dialog; *name* is None for a new preset."""

    name: str | None


def key_state_text(preset: ServerPreset) -> str:
    """Whether a key is set; the key itself never reaches a screen."""
    return msg.PRESET_KEY_SET if preset.has_key else msg.PRESET_KEY_NONE


def wire_text(preset: ServerPreset) -> str:
    """The rerank wire, or a dash when the preset has no rerank role."""
    if preset.rerank_wire is None or WorkerRole.RERANK not in preset.roles:
        return msg.PRESET_NO_VALUE
    return preset.rerank_wire.value


def roles_text(preset: ServerPreset) -> str:
    """The roles of a preset as one comma-separated line."""
    return ", ".join(role.value for role in preset.roles) or msg.PRESET_NO_VALUE


def preset_line(preset: ServerPreset) -> str:
    """One preset as a line of text."""
    return msg.SETTINGS_PRESETS_ROW.format(
        name=preset.name,
        url=preset.base_url or msg.PRESET_INVALID_URL,
        roles=roles_text(preset),
        wire=wire_text(preset),
        key=key_state_text(preset),
    )


def _tokens(args: str) -> list[str] | None:
    """The words of *args* with quotes honoured, or None when a quote is left open."""
    try:
        return shlex.split(args)
    except ValueError:
        return None


def key_typed(args: str) -> bool:
    """Whether *args* holds a `--key` flag, with the value joined by a space or an equals sign."""
    words = _tokens(args)
    if words is None:
        words = args.split()
    return any(word == _KEY_FLAG or word.startswith(f"{_KEY_FLAG}=") for word in words)


def is_preset_line(text: str) -> bool:
    """Whether the chat line *text* is a /preset command, with the dispatcher's word split."""
    words = text.split()
    return bool(words) and words[0].lower() == _COMMAND


def dialog_request(args: str) -> DialogRequest | None:
    """The dialog that *args* asks for: `add` alone, or `edit NAME` with no flag."""
    words = _tokens(args)
    if words == ["add"]:
        return DialogRequest(None)
    if words is not None and len(words) == _VERB_AND_NAME and words[0] == "edit":
        return DialogRequest(words[1])
    return None


def remove_target(args: str) -> str | None:
    """The preset name in `remove NAME`, else None."""
    words = _tokens(args)
    if words is not None and len(words) == _VERB_AND_NAME and words[0] == "remove":
        return words[1]
    return None


def _flags(words: list[str]) -> tuple[dict[str, str], bool] | None:
    """The `--flag value` pairs and the clear flag in *words*; None for any other shape."""
    values: dict[str, str] = {}
    clear = False
    rest = iter(words)
    for word in rest:
        if word == _CLEAR_FLAG:
            clear = True
        elif word in _VALUE_FLAGS and (value := next(rest, None)) is not None:
            values[word] = value
        else:
            return None
    return values, clear


def _list() -> PresetCommandResult:
    presets = list_presets()
    if not presets:
        return PresetCommandResult(True, msg.PRESET_LIST_EMPTY)
    return PresetCommandResult(True, "\n".join(preset_line(p) for p in presets))


def _add(words: list[str]) -> PresetCommandResult | None:
    """Add from `NAME URL --roles ROLES`; words before the URL join into the name."""
    first_flag = next((i for i, word in enumerate(words) if word.startswith("--")), len(words))
    if first_flag < _NAME_AND_URL:
        return None
    parsed = _flags(words[first_flag:])
    if parsed is None:
        return None
    flags, clear = parsed
    if clear or "--url" in flags:
        return None
    preset = add_preset(
        " ".join(words[: first_flag - 1]),
        words[first_flag - 1],
        roles=split_roles(flags.get("--roles", "")),
        rerank_wire=flags.get("--wire", RerankWire.V1),
    )
    return PresetCommandResult(True, msg.PRESET_ADDED.format(name=preset.name))


def _edit(words: list[str]) -> PresetCommandResult | None:
    if not words:
        return None
    parsed = _flags(words[1:])
    if parsed is None:
        return None
    flags, clear = parsed
    roles = flags.get("--roles")
    preset = edit_preset(
        words[0],
        url=flags.get("--url"),
        roles=None if roles is None else split_roles(roles),
        rerank_wire=flags.get("--wire"),
        clear_key=clear,
    )
    return PresetCommandResult(True, msg.PRESET_UPDATED.format(name=preset.name))


def _remove(words: list[str]) -> PresetCommandResult | None:
    if len(words) != 1:
        return None
    remove_preset(words[0])
    return PresetCommandResult(True, msg.PRESET_REMOVED.format(name=words[0]))


def run_preset_command(args: str) -> PresetCommandResult:
    """Run `/preset ARGS` against the service. A refusal returns the service's own text."""
    usage = PresetCommandResult(False, msg.PRESET_USAGE)
    words = _tokens(args)
    if not words:
        return usage
    verb, rest = words[0], words[1:]
    if key_typed(args):
        return PresetCommandResult(False, msg.PRESET_KEY_IN_DIALOG)
    try:
        if verb == "list" and not rest:
            return _list()
        result = {"add": _add, "edit": _edit, "remove": _remove}.get(verb, lambda _w: None)(rest)
    except (ValueError, OSError) as exc:
        return PresetCommandResult(False, failure_text(exc))
    return usage if result is None else result
