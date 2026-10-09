"""Settings commands: list, get, set, and unset through the shared write boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.table import Table
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import console, data_dir_option, global_option
from lilbee.cli.commands._shared import REBUILD_HINT, emit, line, run_or_fail, setup, shown_value
from lilbee.cli.tui import messages as msg

if TYPE_CHECKING:
    from lilbee.app.settings import SettingInfo, SettingsUpdateResult
    from lilbee.server.models import ConfigUpdateResponse

settings_app = typer.Typer(help="Show and change settings, with each value's source.")

_key_argument = typer.Argument(help="A setting's name, as shown by 'lilbee settings list'.")
_keys_argument = typer.Argument(
    help="One or more settings to remove your value of; env, profile or default applies."
)


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
    line(f"{info.key} = {shown_value(info.value)}", theme.ACCENT)
    line(f"source: {source}")
    line(info.help_text, theme.MUTED)


def _render_update(result: SettingsUpdateResult) -> None:
    if result.reindex_required:
        line(REBUILD_HINT, theme.WARNING)
    for warning in result.warnings:
        line(warning, theme.WARNING)


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
    from lilbee.app.settings import config_write_failure_message, list_settings
    from lilbee.server.models import SettingsListResponse

    setup(data_dir, use_global)
    infos = run_or_fail(lambda: list_settings(group), config_write_failure_message)
    emit(SettingsListResponse.from_infos(infos), lambda: _render_list(infos))


@settings_app.command(name="get")
def settings_get(
    key: str = _key_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show one setting's current value and source."""
    from lilbee.app.settings import config_write_failure_message, get_setting
    from lilbee.server.models import SettingValueResponse

    setup(data_dir, use_global)
    info = run_or_fail(lambda: get_setting(key), config_write_failure_message)
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
    from lilbee.app.settings import apply_settings_update, config_write_failure_message

    setup(data_dir, use_global)
    parsed = _parse_cli_value(key, value)
    result = run_or_fail(
        lambda: apply_settings_update({key: parsed}, allow_model_roles=False),
        config_write_failure_message,
    )
    shown = msg.MASKED_VALUE if _is_secret(key) and value else value

    def _render() -> None:
        line(f"Set {key} to {shown}.", theme.ACCENT)
        _render_update(result)

    emit(_update_response(result), _render)


@settings_app.command(name="unset")
def settings_unset(
    keys: list[str] = _keys_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remove your value of one or more settings; the next source applies."""
    from lilbee.app.settings import config_write_failure_message, get_setting, reset_settings

    setup(data_dir, use_global)
    result = run_or_fail(
        lambda: reset_settings(keys, allow_model_roles=False), config_write_failure_message
    )

    def _render() -> None:
        for key in result.updated:
            try:
                info = get_setting(key)
            except KeyError:
                line(f"{key}: removed your value (write-only; new value not shown)")
                continue
            source = info.source.value.replace("_", " ")
            line(f"{key} = {shown_value(info.value)} ({source})")
        _render_update(result)

    emit(_update_response(result), _render)
