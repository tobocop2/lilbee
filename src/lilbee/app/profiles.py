"""Profile use cases: list, show, the active profile, diff, apply, and the file operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w

from lilbee.app.settings import apply_profile_layer, list_settings, reset_settings
from lilbee.config_meta import PUBLIC_CONFIG_FIELDS, REINDEX_FIELDS
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import ProfileScope, SettingSource
from lilbee.core.config.resolve import (
    PROFILE_FIELDS,
    SettingLayers,
    builtin_value,
    read_layers,
    read_profile_table,
    resolve,
)
from lilbee.core.profile_files import (
    DEFAULT_PROFILE_NAME,
    META_TABLE,
    PACKAGE_FOLDERS,
    PROFILE_FORMAT,
    PROFILE_SUFFIX,
    VALUES_TABLE,
    PlannedWrite,
    ProfileCatalog,
    ProfileEntry,
    ProfileFile,
    ProfileFolder,
    ProfileStore,
    ProfileValidation,
    normalized_values,
    parse_text,
    plan_write,
    profile_folders,
    profile_key,
    profile_text,
    read_profile_text,
    validate_file,
)

_OVERRIDE_SOURCES = frozenset({SettingSource.ENV, SettingSource.USER})
_SAVE_FOLDERS = frozenset({ProfileFolder.PROJECT, ProfileFolder.GLOBAL})


class ProfileEffect(StrEnum):
    """When a changed profile value takes effect."""

    REINDEX = "reindex"
    NEW_FILES_ONLY = "new_files_only"
    NOW = "now"


class ProfileStatus(StrEnum):
    """How the applied profile's file compares with the copy recorded on apply."""

    CURRENT = "current"
    CHANGED = "changed"
    MISSING = "missing"
    BROKEN = "broken"


@dataclass(frozen=True)
class DiffRow:
    """One setting a profile apply changes."""

    key: str
    current: Any
    current_source: SettingSource
    new: Any
    effect: ProfileEffect


@dataclass(frozen=True)
class ProfileDiff:
    """What applying a profile changes, which set values it keeps, and how many it never touches."""

    name: str
    changes: tuple[DiffRow, ...]
    kept: tuple[str, ...]
    untouched_count: int


@dataclass(frozen=True)
class ActiveProfile:
    """The project's applied profile, its recorded values, and the state of its file."""

    name: str
    values: Mapping[str, Any]
    status: ProfileStatus
    error: str | None = None


@dataclass(frozen=True)
class ProfileLocation:
    """A profile file an operation wrote or removed."""

    name: str
    folder: ProfileFolder
    path: Path


@dataclass(frozen=True)
class SaveResult:
    """A saved profile file and the settings of yours it took over from config.toml."""

    location: ProfileLocation
    absorbed: tuple[str, ...]


@dataclass(frozen=True)
class DiscardResult:
    """The settings of yours removed so the profile's values show through."""

    dropped: tuple[str, ...]


@dataclass(frozen=True)
class ApplyResult:
    """The outcome of a profile apply."""

    name: str
    changes: tuple[DiffRow, ...]
    reindex_required: bool
    new_files_only: tuple[str, ...]


def _effect(key: str) -> ProfileEffect:
    if key in REINDEX_FIELDS:
        return ProfileEffect.REINDEX
    if PROFILE_FIELDS[key] is ProfileScope.INGEST:
        return ProfileEffect.NEW_FILES_ONLY
    return ProfileEffect.NOW


def list_profiles(store: ProfileStore) -> ProfileCatalog:
    """Every profile file in the profile folders, in precedence order."""
    return store.scan()


def show(store: ProfileStore, name: str) -> ProfileEntry:
    """The entry *name* picks; raises ``ValueError`` when no profile has that name."""
    entry = store.scan().find(name)
    if entry is None:
        raise ValueError(f"No profile named {name!r}")
    return entry


def _valid_file(entry: ProfileEntry) -> ProfileFile:
    """The profile file of *entry*; raises ``ValueError`` when it is broken."""
    if entry.file is None:
        raise ValueError(f"Profile {entry.name!r} cannot be used: {entry.error}")
    return entry.file


def _usable(store: ProfileStore, name: str) -> ProfileFile:
    """The valid profile file *name* picks; raises ``ValueError`` when missing or broken."""
    return _valid_file(show(store, name))


def _file_status(entry: ProfileEntry | None, recorded: Mapping[str, Any]) -> ProfileStatus:
    if entry is None:
        return ProfileStatus.MISSING
    if entry.file is None:
        return ProfileStatus.BROKEN
    if dict(entry.file.values) != dict(recorded):
        return ProfileStatus.CHANGED
    return ProfileStatus.CURRENT


def active(store: ProfileStore) -> ActiveProfile:
    """The applied profile and whether its file is current, changed, gone or broken."""
    table = read_profile_table(cfg.data_root)
    name = table.name or DEFAULT_PROFILE_NAME
    if profile_key(name) == profile_key(DEFAULT_PROFILE_NAME):
        return ActiveProfile(name, table.values, ProfileStatus.CURRENT)
    entry = store.scan().find(name)
    error = entry.error if entry is not None else None
    return ActiveProfile(name, table.values, _file_status(entry, table.values), error)


def _diff(profile: ProfileFile) -> ProfileDiff:
    layers = read_layers(cfg.data_root)
    current = {key: resolve(key, layers) for key in PROFILE_FIELDS}
    kept = tuple(key for key in profile.values if current[key].source in _OVERRIDE_SOURCES)
    after = replace(layers, profile=dict(profile.values))
    open_keys = [key for key in PROFILE_FIELDS if current[key].source not in _OVERRIDE_SOURCES]
    new_values = normalized_values({key: resolve(key, after).value for key in open_keys})
    changes = tuple(
        DiffRow(key, getattr(cfg, key), current[key].source, new_values[key], _effect(key))
        for key in open_keys
        if new_values[key] != getattr(cfg, key)
    )
    return ProfileDiff(
        name=profile.name,
        changes=changes,
        kept=kept,
        untouched_count=len(PUBLIC_CONFIG_FIELDS - set(PROFILE_FIELDS)),
    )


def diff(store: ProfileStore, name: str) -> ProfileDiff:
    """What applying *name* changes; raises ``ValueError`` when it is missing or broken."""
    return _diff(_usable(store, name))


def apply(store: ProfileStore, name: str) -> ApplyResult:
    """Record *name* as the project's profile and set cfg from it; user and env values stay."""
    profile = _usable(store, name)
    planned = _diff(profile)
    apply_profile_layer(profile.name, profile.values)
    effects = {row.key: row.effect for row in planned.changes}
    return ApplyResult(
        name=profile.name,
        changes=planned.changes,
        reindex_required=ProfileEffect.REINDEX in effects.values(),
        new_files_only=tuple(k for k, e in effects.items() if e is ProfileEffect.NEW_FILES_ONLY),
    )


def _target_dir(target: ProfileFolder) -> Path:
    """The folder a new profile file goes to; only this project's or the global folder."""
    if target not in _SAVE_FOLDERS:
        raise ValueError("Profiles are saved to this project or to all projects")
    folders = dict(profile_folders(cfg.data_root))
    if target not in folders:
        raise ValueError(
            "lilbee is using the global data folder, so there is no project folder; "
            "save the profile for all projects instead"
        )
    return folders[target]


def _location(planned: PlannedWrite) -> ProfileLocation:
    return ProfileLocation(planned.profile.name, planned.folder, planned.path)


def _written(planned: PlannedWrite) -> ProfileLocation:
    planned.write()
    return _location(planned)


def _owned(store: ProfileStore, name: str, instead: str = "duplicate it") -> ProfileEntry:
    """The project or global profile *name* picks; raises for one that ships with lilbee."""
    entry = show(store, name)
    if entry.folder in PACKAGE_FOLDERS:
        raise ValueError(f"{entry.name} ships with lilbee and cannot be changed; {instead} instead")
    return entry


def _owned_file(
    store: ProfileStore, name: str, instead: str = "duplicate it"
) -> tuple[ProfileEntry, ProfileFile]:
    entry = _owned(store, name, instead)
    return entry, _valid_file(entry)


def _commented(text: str) -> list[str]:
    return [f"# {line}" for line in text.strip().splitlines()]


def _template_lines(key: str, value: Any, help_text: str) -> list[str]:
    """A template entry: the help text, then *value* set, or the default commented out."""
    if value is not None:
        return [*_commented(help_text), tomli_w.dumps({key: value}).strip()]
    default = builtin_value(key)
    if default is None:
        return [
            *_commented(help_text),
            f"# {key} has no default: leaving it out lets lilbee decide",
        ]
    return [*_commented(help_text), *_commented(tomli_w.dumps({key: default}))]


def _template_text(name: str, values: Mapping[str, Any]) -> str:
    """A profile file that lists every profile setting, set from *values* or commented out."""
    help_texts = {info.key: info.help_text for info in list_settings()}
    blocks = [
        "\n".join(_template_lines(key, values.get(key), help_texts[key])) for key in PROFILE_FIELDS
    ]
    head = tomli_w.dumps({META_TABLE: {"name": name, "format": PROFILE_FORMAT}})
    return "\n\n".join([head.strip(), f"[{VALUES_TABLE}]", *blocks]) + "\n"


def new(
    store: ProfileStore,
    name: str,
    target: ProfileFolder = ProfileFolder.GLOBAL,
    from_name: str | None = None,
) -> ProfileLocation:
    """Write a template for *name* listing every profile setting; *from_name* sets its values."""
    values = _usable(store, from_name).values if from_name is not None else {}
    text = _template_text(name, values)
    return _written(plan_write(_target_dir(target), target, text, stem=name))


def _your_profile_keys(layers: SettingLayers) -> tuple[str, ...]:
    """The profile settings config.toml sets, in profile-field order."""
    return tuple(key for key in PROFILE_FIELDS if key in layers.user)


def _project_values() -> tuple[dict[str, Any], tuple[str, ...]]:
    """The project's profile values with yours laid over them, and which keys are yours."""
    layers = read_layers(cfg.data_root)
    yours = _your_profile_keys(layers)
    return {**layers.profile, **{key: layers.user[key] for key in yours}}, yours


def _switch_to(planned: PlannedWrite, yours: tuple[str, ...]) -> SaveResult:
    """Write *planned*, then make it the project's profile, taking *yours* out of config.toml.

    A failure after the file is written keeps the file and says so.
    """
    written: list[Path] = []
    try:
        apply_profile_layer(
            planned.profile.name,
            planned.profile.values,
            absorb=yours,
            write_first=lambda: written.append(planned.write()),
        )
    except OSError as exc:
        if not written:
            raise
        raise ValueError(
            f"Saved the profile to {planned.path}, but switching this project to it failed: {exc}"
        ) from exc
    return SaveResult(_location(planned), yours)


def _bare(name: str, values: Mapping[str, Any]) -> ProfileFile:
    return ProfileFile(
        name=name,
        description=None,
        authors=(),
        tested_on=None,
        format=PROFILE_FORMAT,
        min_lilbee=None,
        evidence=None,
        values=values,
    )


def save_as(name: str, target: ProfileFolder = ProfileFolder.GLOBAL) -> SaveResult:
    """Save the project's profile values plus yours as *name* and switch the project to it.

    ``absorbed`` names your settings the new profile now holds; they leave config.toml.
    Environment variables are not saved.
    """
    values, yours = _project_values()
    text = profile_text(_bare(name, values))
    return _switch_to(plan_write(_target_dir(target), target, text, stem=name), yours)


def update(store: ProfileStore) -> SaveResult:
    """Write the project's profile values plus yours into the active profile's own file."""
    table = read_profile_table(cfg.data_root)
    name = table.name or DEFAULT_PROFILE_NAME
    entry, profile = _owned_file(store, name, "save your settings as a new profile")
    if dict(profile.values) != dict(table.values):
        raise ValueError(
            f"{entry.name} changed on disk since it was applied; apply it again to use "
            "the file, or save your settings as a new profile"
        )
    values, yours = _project_values()
    text = profile_text(replace(profile, values=values))
    directory = entry.path.parent
    planned = plan_write(directory, entry.folder, text, stem=entry.path.stem, replacing=entry.path)
    return _switch_to(planned, yours)


def discard() -> DiscardResult:
    """Remove your settings of profile keys from config.toml; environment variables stay."""
    yours = _your_profile_keys(read_layers(cfg.data_root))
    if yours:
        reset_settings(list(yours))
    return DiscardResult(yours)


def duplicate(
    store: ProfileStore, name: str, new_name: str, target: ProfileFolder = ProfileFolder.GLOBAL
) -> ProfileLocation:
    """Copy the profile *name* picks, with its metadata, as *new_name*."""
    text = profile_text(replace(_usable(store, name), name=new_name))
    return _written(plan_write(_target_dir(target), target, text, stem=new_name))


def rename(store: ProfileStore, name: str, new_name: str) -> ProfileLocation:
    """Rename a project or global profile in its folder; the project follows when it is active."""
    entry, profile = _owned_file(store, name)
    text = profile_text(replace(profile, name=new_name))
    planned = plan_write(entry.path.parent, entry.folder, text, stem=new_name, replacing=entry.path)
    location = _written(planned)
    table = read_profile_table(cfg.data_root)
    if table.name is not None and profile_key(table.name) == profile_key(entry.name):
        persistent_settings.write_profile_table(cfg.data_root, location.name, table.values)
    return location


def delete(store: ProfileStore, name: str) -> ProfileLocation:
    """Remove a project or global profile file; projects keep their recorded copy of it."""
    entry = _owned(store, name)
    entry.path.unlink()
    return ProfileLocation(entry.name, entry.folder, entry.path)


def export(store: ProfileStore, name: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Write the profile *name* picks as a clean file at *dest*, or in it when it is a folder."""
    entry = show(store, name)
    profile = _valid_file(entry)
    path = dest / f"{profile_key(entry.name)}{PROFILE_SUFFIX}" if dest.is_dir() else dest
    if path.exists() and not overwrite:
        raise ValueError(f"{path} already exists")
    text = profile_text(profile)
    return PlannedWrite(
        path, entry.folder, parse_text(text, path.stem, entry.folder), text, None
    ).write()


def _same_name_in(store: ProfileStore, folder: ProfileFolder, name: str) -> Path | None:
    key = profile_key(name)
    return next(
        (e.path for e in store.scan().entries if e.folder is folder and profile_key(e.name) == key),
        None,
    )


def import_profile(
    store: ProfileStore,
    source: Path,
    target: ProfileFolder = ProfileFolder.GLOBAL,
    *,
    overwrite: bool = False,
) -> ProfileLocation:
    """Validate the file at *source* and copy it into *target*; a taken name needs *overwrite*.

    A file that names no profile takes the slug of its file name as its name.
    """
    text = read_profile_text(source)
    directory = _target_dir(target)
    stem = profile_key(source.stem)
    profile = parse_text(text, stem, target)
    replacing = _same_name_in(store, target, profile.name) if overwrite else None
    return _written(plan_write(directory, target, text, stem=stem, replacing=replacing))


def validate(path: Path, folder: ProfileFolder = ProfileFolder.GLOBAL) -> ProfileValidation:
    """Every problem with the file at *path* as a profile in *folder*."""
    return validate_file(path, folder)
