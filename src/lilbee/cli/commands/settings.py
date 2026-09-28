"""Settings commands: list, get, set, and unset through the shared write boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

import typer
from rich.table import Table
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.commands._shared import REBUILD_HINT, emit, shown_value
from lilbee.cli.helpers import json_output, print_prefixed
from lilbee.core.config import cfg

if TYPE_CHECKING:
    from lilbee.app.settings import SettingInfo, SettingsUpdateResult
    from lilbee.server.models import ConfigUpdateResponse

T = TypeVar("T")

settings_app = typer.Typer(help="Show and change settings, with each value's source.")

_MASKED_VALUE = "************"
_key_argument = typer.Argument(help="A setting's name, as shown by 'lilbee settings list'.")
_keys_argument = typer.Argument(
    help="One or more settings to remove your value of; env, profile or default applies."
)


def _setup(data_dir: Path | None, use_global: bool) -> None:
    apply_overrides(data_dir=data_dir, use_global=use_global)


def _fail(message: str) -> NoReturn:
    if cfg.json_mode:
        json_output({"error": message})
    else:
        print_prefixed(console, "Error: ", message, style=theme.ERROR)
    raise typer.Exit(1)


def _run(operation: Callable[[], T]) -> T:
    """Run a settings operation; a refusal prints its reason, as JSON or text, and exits 1."""
    try:
        return operation()
    except (ValueError, KeyError) as exc:
        _fail(str(exc))
    except OSError as exc:
        from lilbee.app.settings import config_write_failure_message

        _fail(config_write_failure_message(exc))


def _line(text: str, style: str | None = None) -> None:
    console.print(Text(text, style=style or ""), soft_wrap=True)


def _is_secret(key: str) -> bool:
    from lilbee.app.settings_map import SETTINGS_MAP

    definition = SETTINGS_MAP.get(key)
    return definition is not None and definition.secret


def _render_list(infos: list[SettingInfo]) -> None:
    table = Table("Key", "Group", "Value", "Source")
    for info in infos:
        source = info.source.value.replace("_", " ")
        table.add_row(info.key, info.group.value, Text(shown_value(info.value)), source)
    console.print(table)


def _render_get(info: SettingInfo) -> None:
    source = info.source.value.replace("_", " ")
    _line(f"{info.key} = {shown_value(info.value)}", theme.ACCENT)
    _line(f"source: {source}")
    _line(info.help_text, theme.MUTED)


def _render_update(result: SettingsUpdateResult) -> None:
    if result.reindex_required:
        _line(REBUILD_HINT, theme.WARNING)
    for warning in result.warnings:
        _line(warning, theme.WARNING)


def _update_response(result: SettingsUpdateResult) -> ConfigUpdateResponse:
    from lilbee.server.models import ConfigUpdateResponse

    return ConfigUpdateResponse(
        updated=result.updated,
        reindex_required=result.reindex_required,
        warnings=list(result.warnings),
    )


@settings_app.command(name="list")
def settings_list(
    group: str | None = typer.Option(
        None, "--group", help="Filter to one group (case-insensitive); omit for every group."
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List every writable setting with its current value and source."""
    from lilbee.app.settings import list_settings
    from lilbee.server.models import SettingsListResponse

    _setup(data_dir, use_global)
    infos = _run(lambda: list_settings(group))
    emit(SettingsListResponse.from_infos(infos), lambda: _render_list(infos))


@settings_app.command(name="get")
def settings_get(
    key: str = _key_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show one setting's current value and source."""
    from lilbee.app.settings import get_setting
    from lilbee.server.models import SettingValueResponse

    _setup(data_dir, use_global)
    info = _run(lambda: get_setting(key))
    emit(SettingValueResponse.from_info(info), lambda: _render_get(info))


def _parse_cli_value(key: str, raw: str) -> Any:
    """A CLI ``set`` argument as the config write boundary expects it.

    List-typed fields are newline-separated, matching how config.toml persists
    them; every other type is left to pydantic's own coercion on assignment.
    """
    from lilbee.core.config.schema import field_type_name

    try:
        is_list = field_type_name(key) == "list"
    except KeyError:
        is_list = False
    return raw.split("\n") if is_list else raw


@settings_app.command(name="set")
def settings_set(
    key: str = _key_argument,
    value: str = typer.Argument(help="The new value."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Set a writable setting; model roles are refused (use 'lilbee model' instead)."""
    from lilbee.app.settings import apply_settings_update

    _setup(data_dir, use_global)
    parsed = _parse_cli_value(key, value)
    result = _run(lambda: apply_settings_update({key: parsed}, allow_model_roles=False))
    shown = _MASKED_VALUE if _is_secret(key) and value else value

    def _render() -> None:
        _line(f"Set {key} to {shown}.", theme.ACCENT)
        _render_update(result)

    emit(_update_response(result), _render)


@settings_app.command(name="unset")
def settings_unset(
    keys: list[str] = _keys_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remove your value of one or more settings; the next source applies."""
    from lilbee.app.settings import get_setting, reset_settings

    _setup(data_dir, use_global)
    result = _run(lambda: reset_settings(keys, allow_model_roles=False))

    def _render() -> None:
        for key in result.updated:
            try:
                info = get_setting(key)
            except KeyError:
                _line(f"{key}: removed your value (write-only; new value not shown)")
                continue
            source = info.source.value.replace("_", " ")
            _line(f"{key} = {shown_value(info.value)} ({source})")
        _render_update(result)

    emit(_update_response(result), _render)
