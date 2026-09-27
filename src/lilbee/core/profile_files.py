"""Profile files: the file format, its validation, and discovery across the profile folders."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
from pydantic import ValidationError

from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import ProfileScope
from lilbee.core.config.resolve import PROFILE_FIELDS, builtin_value
from lilbee.core.system import canonical_data_root, default_data_dir

PROFILES_DIRNAME = "profiles"
BUILTIN_DIRNAME = "builtin"
COMMUNITY_DIRNAME = "community"
PROFILE_SUFFIX = ".toml"
PACKAGE_PROFILES_DIR = Path(__file__).resolve().parent.parent / PROFILES_DIRNAME

DEFAULT_PROFILE_NAME = "Default"
PROFILE_FORMAT = 1

META_TABLE = "profile"
VALUES_TABLE = "values"
_TOP_LEVEL_KEYS = frozenset({META_TABLE, VALUES_TABLE})
_STRING_META_KEYS = ("name", "description", "tested_on", "min_lilbee", "evidence")
_META_KEYS = frozenset({*_STRING_META_KEYS, "authors", "format"})
_AUTHOR_KEYS = frozenset({"name", "github"})

_NAME_PATTERN = re.compile(r"[A-Za-z0-9 _()-]{1,40}")
_KEY_DROPPED = re.compile(r"[()]")
_KEY_SEPARATORS = re.compile(r"[\s_-]+")


class ProfileFolder(StrEnum):
    """A folder lilbee reads profiles from, highest precedence first."""

    PROJECT = "project"
    GLOBAL = "global"
    COMMUNITY = "community"
    BUILTIN = "builtin"


PACKAGE_FOLDERS = frozenset({ProfileFolder.COMMUNITY, ProfileFolder.BUILTIN})


class ProfileFileError(ValueError):
    """A profile file lilbee cannot use, with the reason as its message."""


@dataclass(frozen=True)
class ProfileAuthor:
    """One credited author of a profile."""

    name: str
    github: str | None


@dataclass(frozen=True)
class ProfileFile:
    """A valid profile file: its metadata and the settings it holds."""

    name: str
    description: str | None
    authors: tuple[ProfileAuthor, ...]
    tested_on: str | None
    format: int
    min_lilbee: str | None
    evidence: str | None
    values: Mapping[str, Any]


@dataclass(frozen=True)
class ProfileEntry:
    """One profile file found in a folder; broken files carry the reason in ``error``."""

    name: str
    folder: ProfileFolder
    path: Path
    file: ProfileFile | None
    error: str | None
    shadowed_by: ProfileFolder | None = None


@dataclass(frozen=True)
class ProfileCatalog:
    """Every profile file found, in precedence order."""

    entries: tuple[ProfileEntry, ...]

    def find(self, name: str) -> ProfileEntry | None:
        """The entry a name picks: the highest-precedence one with that lookup key."""
        key = profile_key(name)
        return next(
            (e for e in self.entries if e.shadowed_by is None and profile_key(e.name) == key),
            None,
        )


def profile_key(name: str) -> str:
    """The lookup key of a name and the stem of a file lilbee writes for it: a lowercase slug."""
    words = _KEY_DROPPED.sub("", name.casefold())
    return _KEY_SEPARATORS.sub("-", words).strip("-")


def profile_folders(data_root: Path) -> list[tuple[ProfileFolder, Path]]:
    """The profile folders for *data_root*, highest precedence first."""
    global_root = default_data_dir()
    folders = [
        (ProfileFolder.GLOBAL, global_root / PROFILES_DIRNAME),
        (ProfileFolder.COMMUNITY, PACKAGE_PROFILES_DIR / COMMUNITY_DIRNAME),
        (ProfileFolder.BUILTIN, PACKAGE_PROFILES_DIR / BUILTIN_DIRNAME),
    ]
    if canonical_data_root(data_root) != canonical_data_root(global_root):
        folders.insert(0, (ProfileFolder.PROJECT, data_root / PROFILES_DIRNAME))
    return folders


def _needs_newer(min_lilbee: str | None) -> str | None:
    """The "needs a newer lilbee" message when *min_lilbee* is above the running version."""
    if min_lilbee is None or Version(min_lilbee) <= Version(installed_version("lilbee")):
        return None
    return f"Needs lilbee {min_lilbee} or newer"


def _optional_str(meta: Mapping[str, Any], key: str) -> str | None:
    value = meta.get(key)
    # untyped TOML: a metadata field can hold any TOML type
    if value is not None and not isinstance(value, str):
        raise ProfileFileError(f"{key} must be text")
    return value


def _parse_author(raw: Any) -> ProfileAuthor:
    # untyped TOML: each author must be a table with a text name
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        raise ProfileFileError("Each author needs a name")
    unknown = sorted(set(raw) - _AUTHOR_KEYS)
    if unknown:
        raise ProfileFileError(f"Unknown author field: {unknown[0]}")
    return ProfileAuthor(name=raw["name"], github=_optional_str(raw, "github"))


def _parse_authors(meta: Mapping[str, Any]) -> tuple[ProfileAuthor, ...]:
    raw = meta.get("authors", [])
    # untyped TOML: authors must be an array of tables
    if not isinstance(raw, list):
        raise ProfileFileError("authors must be a list of { name, github } tables")
    return tuple(_parse_author(item) for item in raw)


def _parse_format(meta: Mapping[str, Any]) -> int:
    value = meta.get("format", PROFILE_FORMAT)
    if value != PROFILE_FORMAT or isinstance(value, bool):
        raise ProfileFileError(
            f"Unknown profile format {value!r}; this lilbee reads format {PROFILE_FORMAT}"
        )
    return PROFILE_FORMAT


def _parse_min_lilbee(meta: Mapping[str, Any]) -> str | None:
    value = _optional_str(meta, "min_lilbee")
    if value is None:
        return None
    try:
        Version(value)
    except InvalidVersion:
        raise ProfileFileError(f"min_lilbee is not a version: {value!r}") from None
    return value


def _check_name(name: str) -> str:
    if _NAME_PATTERN.fullmatch(name) is None or not profile_key(name):
        raise ProfileFileError(
            f"Bad name {name!r}: use 1 to 40 letters, digits, spaces, hyphens, "
            "underscores or parentheses"
        )
    return name


def _check_key(key: str, min_lilbee: str | None) -> None:
    if key not in Config.model_fields:
        raise ProfileFileError(_needs_newer(min_lilbee) or f"Unknown setting: {key}")
    if key not in PROFILE_FIELDS:
        raise ProfileFileError(f"Profiles cannot set {key}")


def normalized_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Each value as Config holds it once its field validator runs; raises on a bad value."""
    trial = cfg.model_copy()
    for key, value in values.items():
        try:
            setattr(trial, key, value)
        except ValidationError as exc:
            raise ProfileFileError(f"Bad value for {key}: {exc.errors()[0]['msg']}") from None
        # A validator's own TypeError reaches here unwrapped; a bad value costs only its file
        except TypeError as exc:
            raise ProfileFileError(f"Bad value for {key}: {exc}") from None
    return {key: getattr(trial, key) for key in values}


def _check_overlap(values: Mapping[str, Any]) -> None:
    size, overlap = values.get("chunk_size"), values.get("chunk_overlap")
    if size is not None and overlap is not None and overlap >= size:
        raise ProfileFileError(f"chunk_overlap ({overlap}) must be < chunk_size ({size})")


def _check_evidence(values: Mapping[str, Any], evidence: str | None) -> None:
    """A package profile sets a retrieval value unequal to the built-in only with evidence."""
    if evidence is not None:
        return
    for key, value in values.items():
        if PROFILE_FIELDS[key] is ProfileScope.RETRIEVAL and value != builtin_value(key):
            raise ProfileFileError(f"Sets retrieval setting {key} without evidence")


def _parse_values(
    data: Mapping[str, Any], min_lilbee: str | None, folder: ProfileFolder, evidence: str | None
) -> dict[str, Any]:
    raw = data.get(VALUES_TABLE)
    # untyped TOML: [values] must be a table
    if not isinstance(raw, dict):
        raise ProfileFileError("Missing [values] table")
    for key in raw:
        _check_key(key, min_lilbee)
    try:
        normalized = normalized_values(raw)
    except ProfileFileError as exc:
        raise ProfileFileError(_needs_newer(min_lilbee) or str(exc)) from None
    _check_overlap(normalized)
    if folder in PACKAGE_FOLDERS:
        _check_evidence(normalized, evidence)
    return dict(raw)


def parse_profile(data: Mapping[str, Any], stem: str, folder: ProfileFolder) -> ProfileFile:
    """Validate parsed TOML as a profile file; raises ``ProfileFileError`` with the reason."""
    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ProfileFileError(f"Unknown table: {unknown[0]}")
    meta = data.get(META_TABLE, {})
    # untyped TOML: [profile] must be a table
    if not isinstance(meta, dict):
        raise ProfileFileError("[profile] must be a table")
    unknown = sorted(set(meta) - _META_KEYS)
    if unknown:
        raise ProfileFileError(f"Unknown profile field: {unknown[0]}")
    strings = {key: _optional_str(meta, key) for key in _STRING_META_KEYS}
    min_lilbee = _parse_min_lilbee(meta)
    return ProfileFile(
        name=_check_name(stem if strings["name"] is None else strings["name"]),
        description=strings["description"],
        authors=_parse_authors(meta),
        tested_on=strings["tested_on"],
        format=_parse_format(meta),
        min_lilbee=min_lilbee,
        evidence=strings["evidence"],
        values=_parse_values(data, min_lilbee, folder, strings["evidence"]),
    )


def read_entry(path: Path, folder: ProfileFolder) -> ProfileEntry:
    """Read and validate one profile file; a broken file is returned with its reason."""
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
        profile = parse_profile(data, path.stem, folder)
    # tomllib raises RecursionError on arrays nested a few hundred deep
    except (tomllib.TOMLDecodeError, RecursionError) as exc:
        return ProfileEntry(path.stem, folder, path, None, f"Not valid TOML: {exc}")
    except UnicodeDecodeError:
        return ProfileEntry(path.stem, folder, path, None, "Not UTF-8 text")
    except OSError as exc:
        return ProfileEntry(path.stem, folder, path, None, f"Cannot read the file: {exc}")
    except ProfileFileError as exc:
        return ProfileEntry(_entry_name(data, path.stem), folder, path, None, str(exc))
    return ProfileEntry(profile.name, folder, path, profile, None)


def _entry_name(data: Mapping[str, Any], stem: str) -> str:
    """The name a broken file shows under: its own name when readable, else the file stem."""
    meta = data.get(META_TABLE)
    # untyped TOML: a broken file's [profile] table may hold anything
    name = meta.get("name") if isinstance(meta, dict) else None
    return name if isinstance(name, str) else stem


def _read_folder(folder: ProfileFolder, path: Path) -> list[ProfileEntry]:
    """Every profile file in *path*, sorted by file name; a missing folder is empty."""
    if not path.is_dir():
        return []
    files = sorted(p for p in path.glob(f"*{PROFILE_SUFFIX}") if p.is_file())
    return _mark_duplicates([read_entry(p, folder) for p in files])


def _broken(entry: ProfileEntry, reason: str) -> ProfileEntry:
    return replace(entry, file=None, error=reason)


def _mark_duplicates(entries: list[ProfileEntry]) -> list[ProfileEntry]:
    """Mark every entry whose lookup key another entry in the same folder shares."""
    keys = [profile_key(e.name) for e in entries]
    return [
        _broken(e, f"Duplicate name in the {e.folder.value} folder") if keys.count(key) > 1 else e
        for e, key in zip(entries, keys, strict=True)
    ]


def _settle_names(entries: list[ProfileEntry]) -> list[ProfileEntry]:
    """Break non-builtin entries that take a built-in name; shadow each lower entry by name."""
    reserved = {profile_key(e.name) for e in entries if e.folder is ProfileFolder.BUILTIN}
    owner: dict[str, ProfileFolder] = {}
    settled: list[ProfileEntry] = []
    for entry in entries:
        key = profile_key(entry.name)
        if entry.folder is not ProfileFolder.BUILTIN and key in reserved:
            entry = _broken(entry, f"Reserved name: {entry.name} is a built-in profile")
            settled.append(replace(entry, shadowed_by=ProfileFolder.BUILTIN))
            continue
        winner = owner.setdefault(key, entry.folder)
        settled.append(entry if winner is entry.folder else replace(entry, shadowed_by=winner))
    return settled


def scan(folders: Sequence[tuple[ProfileFolder, Path]]) -> ProfileCatalog:
    """Read every profile file in *folders*, given highest precedence first."""
    entries = [entry for folder, path in folders for entry in _read_folder(folder, path)]
    return ProfileCatalog(entries=tuple(_settle_names(entries)))


@dataclass(frozen=True)
class ProfileStore:
    """Scans the profile folders of the current data root on each call."""

    def scan(self) -> ProfileCatalog:
        """Every profile visible from ``cfg.data_root`` now."""
        return scan(profile_folders(cfg.data_root))
