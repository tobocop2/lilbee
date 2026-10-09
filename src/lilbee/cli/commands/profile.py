"""Profile commands: show, list, diff, apply, and the profile file operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.table import Table
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import console, data_dir_option, global_option
from lilbee.cli.commands._shared import REBUILD_HINT, emit, line, run_or_fail, setup, shown_value
from lilbee.cli.helpers import json_output, print_prefixed
from lilbee.core.config import cfg
from lilbee.core.profile_files import ProfileFolder, ProfileStore, profile_key

if TYPE_CHECKING:
    from lilbee.server.models import (
        ActiveProfileResponse,
        ProfileApplyResponse,
        ProfileChangeRowResponse,
        ProfileDiffResponse,
        ProfileDiffRowResponse,
        ProfileDiscardResponse,
        ProfileEntryResponse,
        ProfileLocationResponse,
        ProfileSaveResponse,
        ProfileValidationResponse,
    )

_CURRENT_DIR = Path()

profile_app = typer.Typer(
    help="Manage profiles: named sets of ingest, OCR, chunking and retrieval settings."
)

_target_option = typer.Option(
    ProfileFolder.GLOBAL,
    "--target",
    help="Where to save it: global (all projects, the default) or project (this project).",
)
_folder_option = typer.Option(
    ProfileFolder.GLOBAL,
    "--folder",
    help="Check it as a profile in this folder; builtin applies the evidence rule.",
)
_export_path_argument = typer.Argument(_CURRENT_DIR, help="A file, or a folder for <name>.toml.")


def _store() -> ProfileStore:
    """The profile store; it scans the folders on each call and needs no running services."""
    return ProfileStore()


def _about(entry: ProfileEntryResponse) -> list[str]:
    """The description, credit, tested-on and problem lines shown for a profile."""
    from lilbee.app import profiles

    tested_on = profiles.tested_on_line(entry.tested_on, "Tested on {text}")
    lines = [entry.description, entry.credit, tested_on]
    if entry.error:
        lines.append(f"Broken: {entry.error}")
    if entry.shadowed_by is not None:
        lines.append(f"Hidden by the {entry.shadowed_by.value} profile with this name")
    return [text for text in lines if text]


def _render_entry(entry: ProfileEntryResponse) -> None:
    line(entry.name, theme.ACCENT)
    for about_line in _about(entry):
        line(about_line)
    line(f"{entry.folder.value} profile: {entry.path}", theme.MUTED)
    for key, value in entry.values.items():
        line(f"  {key} = {shown_value(value)}")


def _render_list(entries: list[ProfileEntryResponse], active_name: str) -> None:
    active_key = profile_key(active_name)
    table = Table("", "Name", "Folder", "About")
    for entry in entries:
        picked = entry.shadowed_by is None and profile_key(entry.name) == active_key
        marker = "*" if picked else ""
        table.add_row(marker, Text(entry.name), entry.folder.value, Text("\n".join(_about(entry))))
    console.print(table)


def _render_active(current: ActiveProfileResponse) -> None:
    from lilbee.app.profiles import status_note

    line(f"Profile: {current.name}", theme.ACCENT)
    note = status_note(current.status)
    if note:
        line(note, theme.WARNING)
    if current.error:
        line(current.error, theme.WARNING)
    if current.profile is not None:
        for about_line in _about(current.profile):
            line(about_line)
    if current.changes:
        _render_your_changes(current.changes)


def _render_your_changes(rows: list[ProfileChangeRowResponse]) -> None:
    line("Your changes:", theme.ACCENT)
    table = Table("Setting", "Yours", "Without yours", "Takes effect")
    for row in rows:
        source = row.profile_source.value.replace("_", " ")
        fallback = f"{shown_value(row.profile_value)} ({source})"
        effect = row.effect.value.replace("_", " ")
        table.add_row(row.key, Text(shown_value(row.yours)), Text(fallback), effect)
    console.print(table)


def render_changes(rows: list[ProfileDiffRowResponse]) -> None:
    """Print the settings a profile apply changes, as a table."""
    if not rows:
        line("No settings change.")
        return
    table = Table("Setting", "Now", "After", "Takes effect")
    for row in rows:
        source = row.current_source.value.replace("_", " ")
        now = f"{shown_value(row.current)} ({source})"
        effect = row.effect.value.replace("_", " ")
        table.add_row(row.key, Text(now), Text(shown_value(row.new)), effect)
    console.print(table)


def _render_diff(diff: ProfileDiffResponse) -> None:
    line(f"Applying {diff.name}:", theme.ACCENT)
    render_changes(diff.changes)
    if diff.kept:
        line(f"Keeps your values of: {', '.join(diff.kept)}")
    line(f"{diff.untouched_count} settings are never touched by profiles.", theme.MUTED)


def _render_location(verb: str, location: ProfileLocationResponse) -> None:
    line(f"{verb} {location.name}: {location.path}")


def _render_save(verb: str, result: ProfileSaveResponse) -> None:
    _render_location(verb, result)
    if result.absorbed:
        line(f"It now holds your settings of: {', '.join(result.absorbed)}")
    for warning in result.warnings:
        line(warning, theme.WARNING)


def _render_discard(result: ProfileDiscardResponse) -> None:
    if not result.dropped:
        line("You have no values of profile settings to remove.")
        return
    line(f"Removed your values of: {', '.join(result.dropped)}")
    if result.reindex_required:
        line(REBUILD_HINT, theme.WARNING)
    for warning in result.warnings:
        line(warning, theme.WARNING)


@profile_app.command(name="show")
def profile_show(
    name: str | None = typer.Argument(None, help="A profile name; omit it for this project's."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show this project's profile, or the profile a name picks."""
    from lilbee.app import profiles
    from lilbee.server.models import ActiveProfileResponse, ProfileEntryResponse

    setup(data_dir, use_global)
    if name is None:
        current = ActiveProfileResponse.from_active(
            run_or_fail(lambda: profiles.active(_store()), profiles.file_failure_message)
        )
        emit(current, lambda: _render_active(current))
        return
    entry = ProfileEntryResponse.from_entry(
        run_or_fail(lambda: profiles.show(_store(), name), profiles.file_failure_message)
    )
    emit(entry, lambda: _render_entry(entry))


@profile_app.command(name="list")
def profile_list(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List every profile, highest precedence first; * marks this project's."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileEntryResponse, ProfileListResponse

    setup(data_dir, use_global)
    catalog = run_or_fail(lambda: profiles.list_profiles(_store()), profiles.file_failure_message)
    active_name = run_or_fail(lambda: profiles.active(_store()), profiles.file_failure_message).name
    listing = ProfileListResponse(
        profiles=[ProfileEntryResponse.from_entry(e) for e in catalog.entries]
    )
    emit(listing, lambda: _render_list(listing.profiles, active_name))


@profile_app.command(name="diff")
def profile_diff(
    name: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show what applying a profile changes and which of your values it keeps."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileDiffResponse

    setup(data_dir, use_global)
    diff = ProfileDiffResponse.from_diff(
        run_or_fail(lambda: profiles.diff(_store(), name), profiles.file_failure_message)
    )
    emit(diff, lambda: _render_diff(diff))


def _apply_json(
    result: ProfileApplyResponse, reindexed: int | None, reindex_error: str | None
) -> dict[str, Any]:
    payload = {**result.model_dump(mode="json"), "reindexed": reindexed}
    if reindex_error is not None:
        payload["reindex_error"] = reindex_error
    return payload


@profile_app.command(name="apply")
def profile_apply(
    name: str,
    reindex: bool = typer.Option(
        False, "--reindex", help="Rebuild the index afterwards when a change needs it."
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Make a profile this project's; your own values and LILBEE_* variables stay."""
    from lilbee.app import profiles
    from lilbee.cli.commands.ingest_sync import rebuild_or_raise
    from lilbee.server.models import ProfileApplyResponse

    setup(data_dir, use_global)
    result = ProfileApplyResponse.from_result(
        run_or_fail(lambda: profiles.apply(_store(), name), profiles.file_failure_message)
    )
    if not cfg.json_mode:
        line(f"Applied {result.name}.", theme.ACCENT)
        render_changes(result.changes)
    reindexed: int | None = None
    reindex_error: str | None = None
    if reindex and result.reindex_required:
        try:
            reindexed = len(rebuild_or_raise().added)
        except RuntimeError as exc:
            reindex_error = str(exc)
    if cfg.json_mode:
        json_output(_apply_json(result, reindexed, reindex_error))
    else:
        if reindex_error is not None:
            print_prefixed(console, "Error: ", reindex_error, style=theme.ERROR)
        elif reindexed is not None:
            line(f"Rebuilt: {reindexed} documents ingested")
        elif result.reindex_required:
            line(REBUILD_HINT, theme.WARNING)
        for warning in result.warnings:
            line(warning, theme.WARNING)
    if reindex_error is not None:
        raise typer.Exit(1)


@profile_app.command(name="new")
def profile_new(
    name: str,
    from_profile: str | None = typer.Option(
        None, "--from", help="Start from this profile's values."
    ),
    target: ProfileFolder = _target_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Write a profile file listing every profile setting, ready to edit."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        run_or_fail(
            lambda: profiles.new(_store(), name, target, from_name=from_profile),
            profiles.file_failure_message,
        )
    )
    emit(location, lambda: _render_location("Wrote", location))


@profile_app.command(name="save")
def profile_save(
    name: str,
    target: ProfileFolder = _target_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Save this project's profile values plus yours as a new profile and switch to it."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileSaveResponse

    setup(data_dir, use_global)
    result = ProfileSaveResponse.from_save(
        run_or_fail(lambda: profiles.save_as(name, target), profiles.file_failure_message)
    )
    emit(result, lambda: _render_save("Saved", result))


@profile_app.command(name="update")
def profile_update(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Write this project's profile values plus yours into the profile's own file."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileSaveResponse

    setup(data_dir, use_global)
    result = ProfileSaveResponse.from_save(
        run_or_fail(lambda: profiles.update(_store()), profiles.file_failure_message)
    )
    emit(result, lambda: _render_save("Updated", result))


@profile_app.command(name="discard")
def profile_discard(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remove your values of profile settings so the profile's values show through."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileDiscardResponse

    setup(data_dir, use_global)
    result = ProfileDiscardResponse.from_result(
        run_or_fail(profiles.discard, profiles.file_failure_message)
    )
    emit(result, lambda: _render_discard(result))


@profile_app.command(name="duplicate")
def profile_duplicate(
    name: str,
    new_name: str,
    target: ProfileFolder = _target_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Copy a profile under a new name."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        run_or_fail(
            lambda: profiles.duplicate(_store(), name, new_name, target),
            profiles.file_failure_message,
        )
    )
    emit(location, lambda: _render_location("Wrote", location))


@profile_app.command(name="rename")
def profile_rename(
    name: str,
    new_name: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Rename a project or global profile."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        run_or_fail(
            lambda: profiles.rename(_store(), name, new_name), profiles.file_failure_message
        )
    )
    emit(location, lambda: _render_location("Renamed to", location))


@profile_app.command(name="delete")
def profile_delete(
    name: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete a project or global profile file; projects keep their recorded copy."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        run_or_fail(lambda: profiles.delete(_store(), name), profiles.file_failure_message)
    )
    emit(location, lambda: _render_location("Deleted", location))


@profile_app.command(name="export")
def profile_export(
    name: str,
    path: Path = _export_path_argument,
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing file."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Write a profile to a file anyone can import."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        run_or_fail(
            lambda: profiles.export(_store(), name, path, overwrite=overwrite),
            profiles.file_failure_message,
        )
    )
    emit(location, lambda: _render_location("Exported", location))


@profile_app.command(name="import")
def profile_import(
    file: Path,
    target: ProfileFolder = _target_option,
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace a profile with the same name in the target folder."
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Validate a profile file and copy it into a profile folder."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        run_or_fail(
            lambda: profiles.import_profile(_store(), file, target, overwrite=overwrite),
            profiles.file_failure_message,
        )
    )
    emit(location, lambda: _render_location("Imported", location))


def _render_validation(result: ProfileValidationResponse) -> None:
    if result.valid:
        line(f"{result.name} is a valid profile.")
        return
    line(f"{result.name} is not a valid profile:", theme.ERROR)
    for problem in result.problems:
        line(f"  {problem}")


@profile_app.command(name="validate")
def profile_validate(
    file: Path,
    folder: ProfileFolder = _folder_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Check a profile file and list every problem; exits 1 when there is one."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileValidationResponse

    setup(data_dir, use_global)
    result = ProfileValidationResponse.from_validation(profiles.validate(file, folder))
    emit(result, lambda: _render_validation(result))
    if not result.valid:
        raise typer.Exit(1)
