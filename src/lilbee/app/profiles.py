"""Profile use cases: list, show, the active profile, diff and apply."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from lilbee.app.settings import apply_profile_layer
from lilbee.config_meta import PUBLIC_CONFIG_FIELDS, REINDEX_FIELDS
from lilbee.core.config import cfg
from lilbee.core.config.enums import ProfileScope, SettingSource
from lilbee.core.config.resolve import (
    PROFILE_FIELDS,
    read_layers,
    read_profile_table,
    resolve,
)
from lilbee.core.profile_files import (
    DEFAULT_PROFILE_NAME,
    ProfileCatalog,
    ProfileEntry,
    ProfileFile,
    ProfileStore,
    normalized_values,
    profile_key,
)

_OVERRIDE_SOURCES = frozenset({SettingSource.ENV, SettingSource.USER})


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


def _usable(store: ProfileStore, name: str) -> ProfileFile:
    """The valid profile file *name* picks; raises ``ValueError`` when missing or broken."""
    entry = show(store, name)
    if entry.file is None:
        raise ValueError(f"Profile {entry.name!r} cannot be used: {entry.error}")
    return entry.file


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
