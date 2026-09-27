"""Profile commands: show, list, diff, apply, and the profile file operations."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

import typer
from pydantic_core import to_jsonable_python
from rich.table import Table
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output, print_prefixed
from lilbee.core.config import cfg
from lilbee.core.profile_files import ProfileFolder, ProfileStore, profile_key

if TYPE_CHECKING:
    from pydantic import BaseModel

    from lilbee.server.models import (
        ActiveProfileResponse,
        ProfileApplyResponse,
        ProfileDiffResponse,
        ProfileDiffRowResponse,
        ProfileEntryResponse,
        ProfileLocationResponse,
        ProfileSaveResponse,
        ProfileValidationResponse,
    )

T = TypeVar("T")

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
    help="Check it as a profile in this folder; community applies the evidence rule.",
)
_export_path_argument = typer.Argument(_CURRENT_DIR, help="A file, or a folder for <name>.toml.")


def _store() -> ProfileStore:
    """The profile store; it scans the folders on each call and needs no running services."""
    return ProfileStore()


def _setup(data_dir: Path | None, use_global: bool) -> None:
    apply_overrides(data_dir=data_dir, use_global=use_global)


def _run(operation: Callable[[], T]) -> T:
    """Run a profile operation; a refusal prints its reason, as JSON or text, and exits 1."""
    from lilbee.app.profiles import file_failure_message

    try:
        return operation()
    except ValueError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(file_failure_message(exc))


def _fail(message: str) -> NoReturn:
    if cfg.json_mode:
        json_output({"error": message})
    else:
        print_prefixed(console, "Error: ", message, style=theme.ERROR)
    raise typer.Exit(1)


def _emit(model: BaseModel, render: Callable[[], None]) -> None:
    """Print *model* as JSON in --json mode, else run *render*."""
    if cfg.json_mode:
        json_output(model.model_dump(mode="json"))
    else:
        render()


def _shown(value: Any) -> str:
    """A setting value as a profile file writes it: "auto", true, 512."""
    return json.dumps(to_jsonable_python(value))


def _line(text: str, style: str | None = None) -> None:
    console.print(Text(text, style=style or ""), soft_wrap=True)


def _about(entry: ProfileEntryResponse) -> list[str]:
    """The description, credit, tested-on and problem lines shown for a profile."""
    lines = [entry.description, entry.credit]
    if entry.tested_on:
        lines.append(f"Tested on {entry.tested_on}")
    if entry.error:
        lines.append(f"Broken: {entry.error}")
    if entry.shadowed_by is not None:
        lines.append(f"Hidden by the {entry.shadowed_by.value} profile with this name")
    return [line for line in lines if line]


def _render_entry(entry: ProfileEntryResponse) -> None:
    _line(entry.name, theme.ACCENT)
    for line in _about(entry):
        _line(line)
    _line(f"{entry.folder.value} profile: {entry.path}", theme.MUTED)
    for key, value in entry.values.items():
        _line(f"  {key} = {_shown(value)}")


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

    _line(f"Profile: {current.name}", theme.ACCENT)
    note = status_note(current.status)
    if note:
        _line(note, theme.WARNING)
    if current.error:
        _line(current.error, theme.WARNING)
    if current.profile is not None:
        for line in _about(current.profile):
            _line(line)


def _render_changes(rows: list[ProfileDiffRowResponse]) -> None:
    if not rows:
        _line("No settings change.")
        return
    table = Table("Setting", "Now", "After", "Takes effect")
    for row in rows:
        source = row.current_source.value.replace("_", " ")
        now = f"{_shown(row.current)} ({source})"
        table.add_row(row.key, Text(now), Text(_shown(row.new)), row.effect.value.replace("_", " "))
    console.print(table)


def _render_diff(diff: ProfileDiffResponse) -> None:
    _line(f"Applying {diff.name}:", theme.ACCENT)
    _render_changes(diff.changes)
    if diff.kept:
        _line(f"Keeps your values of: {', '.join(diff.kept)}")
    _line(f"{diff.untouched_count} settings are never touched by profiles.", theme.MUTED)


def _render_location(verb: str, location: ProfileLocationResponse) -> None:
    _line(f"{verb} {location.name}: {location.path}")


def _render_save(verb: str, result: ProfileSaveResponse) -> None:
    _render_location(verb, result)
    if result.absorbed:
        _line(f"It now holds your settings of: {', '.join(result.absorbed)}")


@profile_app.command(name="show")
def profile_show(
    name: str | None = typer.Argument(None, help="A profile name; omit it for this project's."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show this project's profile, or the profile a name picks."""
    from lilbee.app import profiles
    from lilbee.server.models import ActiveProfileResponse, ProfileEntryResponse

    _setup(data_dir, use_global)
    if name is None:
        current = ActiveProfileResponse.from_active(_run(lambda: profiles.active(_store())))
        _emit(current, lambda: _render_active(current))
        return
    entry = ProfileEntryResponse.from_entry(_run(lambda: profiles.show(_store(), name)))
    _emit(entry, lambda: _render_entry(entry))


@profile_app.command(name="list")
def profile_list(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List every profile, highest precedence first; * marks this project's."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileEntryResponse, ProfileListResponse

    _setup(data_dir, use_global)
    catalog = _run(lambda: profiles.list_profiles(_store()))
    active_name = _run(lambda: profiles.active(_store())).name
    listing = ProfileListResponse(
        profiles=[ProfileEntryResponse.from_entry(e) for e in catalog.entries]
    )
    _emit(listing, lambda: _render_list(listing.profiles, active_name))


@profile_app.command(name="diff")
def profile_diff(
    name: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show what applying a profile changes and which of your values it keeps."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileDiffResponse

    _setup(data_dir, use_global)
    diff = ProfileDiffResponse.from_diff(_run(lambda: profiles.diff(_store(), name)))
    _emit(diff, lambda: _render_diff(diff))


def _apply_json(result: ProfileApplyResponse, reindexed: int | None) -> dict[str, Any]:
    return {**result.model_dump(mode="json"), "reindexed": reindexed}


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
    from lilbee.cli.commands.ingest_sync import run_rebuild
    from lilbee.server.models import ProfileApplyResponse

    _setup(data_dir, use_global)
    result = ProfileApplyResponse.from_result(_run(lambda: profiles.apply(_store(), name)))
    if not cfg.json_mode:
        _line(f"Applied {result.name}.", theme.ACCENT)
        _render_changes(result.changes)
    rebuild = reindex and result.reindex_required
    reindexed = len(run_rebuild().added) if rebuild else None
    if cfg.json_mode:
        json_output(_apply_json(result, reindexed))
    elif reindexed is not None:
        _line(f"Rebuilt: {reindexed} documents ingested")
    elif result.reindex_required:
        _line("Run lilbee rebuild so the index uses the new values.", theme.WARNING)


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

    _setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        _run(lambda: profiles.new(_store(), name, target, from_name=from_profile))
    )
    _emit(location, lambda: _render_location("Wrote", location))


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

    _setup(data_dir, use_global)
    result = ProfileSaveResponse.from_save(_run(lambda: profiles.save_as(name, target)))
    _emit(result, lambda: _render_save("Saved", result))


@profile_app.command(name="update")
def profile_update(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Write this project's profile values plus yours into the profile's own file."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileSaveResponse

    _setup(data_dir, use_global)
    result = ProfileSaveResponse.from_save(_run(lambda: profiles.update(_store())))
    _emit(result, lambda: _render_save("Updated", result))


@profile_app.command(name="discard")
def profile_discard(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remove your values of profile settings so the profile's values show through."""
    from lilbee.app import profiles

    _setup(data_dir, use_global)
    dropped = list(_run(profiles.discard).dropped)
    if cfg.json_mode:
        json_output({"dropped": dropped})
    elif dropped:
        _line(f"Removed your values of: {', '.join(dropped)}")
    else:
        _line("You have no values of profile settings to remove.")


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

    _setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        _run(lambda: profiles.duplicate(_store(), name, new_name, target))
    )
    _emit(location, lambda: _render_location("Wrote", location))


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

    _setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        _run(lambda: profiles.rename(_store(), name, new_name))
    )
    _emit(location, lambda: _render_location("Renamed to", location))


@profile_app.command(name="delete")
def profile_delete(
    name: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete a project or global profile file; projects keep their recorded copy."""
    from lilbee.app import profiles
    from lilbee.server.models import ProfileLocationResponse

    _setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(_run(lambda: profiles.delete(_store(), name)))
    _emit(location, lambda: _render_location("Deleted", location))


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

    _setup(data_dir, use_global)
    written = _run(lambda: profiles.export(_store(), name, path, overwrite=overwrite))
    if cfg.json_mode:
        json_output({"name": name, "path": written.as_posix()})
    else:
        _line(f"Exported {name}: {written}")


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

    _setup(data_dir, use_global)
    location = ProfileLocationResponse.from_location(
        _run(lambda: profiles.import_profile(_store(), file, target, overwrite=overwrite))
    )
    _emit(location, lambda: _render_location("Imported", location))


def _render_validation(result: ProfileValidationResponse) -> None:
    if result.valid:
        _line(f"{result.name} is a valid profile.")
        return
    _line(f"{result.name} is not a valid profile:", theme.ERROR)
    for problem in result.problems:
        _line(f"  {problem}")


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

    _setup(data_dir, use_global)
    result = ProfileValidationResponse.from_validation(profiles.validate(file, folder))
    _emit(result, lambda: _render_validation(result))
    if not result.valid:
        raise typer.Exit(1)
