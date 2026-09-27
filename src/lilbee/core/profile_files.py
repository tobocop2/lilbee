"""Profile files: the file format, its validation, discovery across the folders, and writes."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any, TypeVar

import tomli_w
from packaging.version import InvalidVersion, Version
from pydantic import ValidationError

from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import ProfileScope
from lilbee.core.config.resolve import PROFILE_FIELDS, builtin_value
from lilbee.core.security import write_text_atomically
from lilbee.core.system import canonical_data_root, default_data_dir

T = TypeVar("T")

PROFILES_DIRNAME = "profiles"
BUILTIN_DIRNAME = "builtin"
COMMUNITY_DIRNAME = "community"
PROFILE_SUFFIX = ".toml"
PACKAGE_PROFILES_DIR = Path(__file__).resolve().parent.parent / PROFILES_DIRNAME
MAX_PROFILE_BYTES = 256 * 1024

DEFAULT_PROFILE_NAME = "Default"
PROFILE_FORMAT = 1

META_TABLE = "profile"
VALUES_TABLE = "values"
_TOP_LEVEL_KEYS = frozenset({META_TABLE, VALUES_TABLE})
_STRING_META_KEYS = ("name", "description", "tested_on", "min_lilbee", "evidence")
_META_KEYS = frozenset({*_STRING_META_KEYS, "authors", "format"})
_AUTHOR_KEYS = frozenset({"name", "github"})

_NAME_PATTERN = re.compile(r"[A-Za-z0-9 _()-]{1,40}")
# the fixed segments under /api/profiles/, and the file names Windows keeps for devices
RESERVED_NAME_KEYS = frozenset(
    {
        "active",
        "new",
        "discard",
        "import",
        "validate",
        "con",
        "prn",
        "aux",
        "nul",
        *(f"{port}{digit}" for port in ("com", "lpt") for digit in range(10)),
    }
)
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


def _check_name(meta: Mapping[str, Any], stem: str) -> str:
    raw = _optional_str(meta, "name")
    name = stem if raw is None else raw
    if _NAME_PATTERN.fullmatch(name) is None or not profile_key(name):
        raise ProfileFileError(
            f"Bad name {name!r}: use 1 to 40 letters, digits, spaces, hyphens, "
            "underscores or parentheses"
        )
    if profile_key(name) in RESERVED_NAME_KEYS:
        raise ProfileFileError(f"Reserved name: {name} cannot name a profile")
    return name


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


def _normalized(key: str, value: Any, min_lilbee: str | None) -> Any:
    """*value* as Config holds it; raises when *key* is not a profile setting or *value* is bad."""
    if key not in Config.model_fields:
        raise ProfileFileError(_needs_newer(min_lilbee) or f"Unknown setting: {key}")
    if key not in PROFILE_FIELDS:
        raise ProfileFileError(f"Profiles cannot set {key}")
    try:
        return normalized_values({key: value})[key]
    except ProfileFileError as exc:
        raise ProfileFileError(_needs_newer(min_lilbee) or str(exc)) from None


def _overlap_problems(values: Mapping[str, Any]) -> list[str]:
    size, overlap = values.get("chunk_size"), values.get("chunk_overlap")
    if size is not None and overlap is not None and overlap >= size:
        return [f"chunk_overlap ({overlap}) must be < chunk_size ({size})"]
    return []


def _evidence_problems(values: Mapping[str, Any], evidence: str | None) -> list[str]:
    """A package profile sets a retrieval value unequal to the built-in only with evidence."""
    if evidence is not None:
        return []
    return [
        f"Sets retrieval setting {key} without evidence"
        for key, value in values.items()
        if PROFILE_FIELDS[key] is ProfileScope.RETRIEVAL and value != builtin_value(key)
    ]


def _value_problems(
    raw: Mapping[str, Any], min_lilbee: str | None, folder: ProfileFolder, evidence: str | None
) -> list[str]:
    problems: list[str] = []
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            normalized[key] = _normalized(key, value, min_lilbee)
        except ProfileFileError as exc:
            problems.append(str(exc))
    problems += _overlap_problems(normalized)
    if folder in PACKAGE_FOLDERS:
        problems += _evidence_problems(normalized, evidence)
    return problems


def _checked(problems: list[str], check: Callable[[], T]) -> T | None:
    """The result of *check*, or None with its reason added to *problems*."""
    try:
        return check()
    except ProfileFileError as exc:
        problems.append(str(exc))
        return None


def _meta_problems(meta: Mapping[str, Any], stem: str) -> tuple[list[str], str | None]:
    """Every problem in the ``[profile]`` table, and its ``min_lilbee`` when that is valid."""
    problems = [f"Unknown profile field: {key}" for key in sorted(set(meta) - _META_KEYS)]
    _checked(problems, partial(_check_name, meta, stem))
    for key in ("description", "tested_on", "evidence"):
        _checked(problems, partial(_optional_str, meta, key))
    min_lilbee = _checked(problems, partial(_parse_min_lilbee, meta))
    _checked(problems, partial(_parse_authors, meta))
    _checked(problems, partial(_parse_format, meta))
    return problems, min_lilbee


def profile_problems(data: Mapping[str, Any], stem: str, folder: ProfileFolder) -> list[str]:
    """Every reason parsed TOML is not a valid profile file; empty when it is one."""
    problems = [f"Unknown table: {key}" for key in sorted(set(data) - _TOP_LEVEL_KEYS)]
    meta = data.get(META_TABLE, {})
    # untyped TOML: [profile] must be a table
    if not isinstance(meta, dict):
        problems.append("[profile] must be a table")
        meta = {}
    meta_problems, min_lilbee = _meta_problems(meta, stem)
    problems += meta_problems
    raw = data.get(VALUES_TABLE)
    # untyped TOML: [values] must be a table
    if not isinstance(raw, dict):
        problems.append("Missing [values] table")
    else:
        evidence = meta.get("evidence")
        # untyped TOML: a non-text evidence is already a problem and counts as none
        evidence = evidence if isinstance(evidence, str) else None
        problems += _value_problems(raw, min_lilbee, folder, evidence)
    return list(dict.fromkeys(problems))


def parse_profile(data: Mapping[str, Any], stem: str, folder: ProfileFolder) -> ProfileFile:
    """Validate parsed TOML as a profile file; raises ``ProfileFileError`` with the first reason."""
    problems = profile_problems(data, stem, folder)
    if problems:
        raise ProfileFileError(problems[0])
    meta = data.get(META_TABLE, {})
    return ProfileFile(
        name=_check_name(meta, stem),
        description=meta.get("description"),
        authors=_parse_authors(meta),
        tested_on=meta.get("tested_on"),
        format=PROFILE_FORMAT,
        min_lilbee=meta.get("min_lilbee"),
        evidence=meta.get("evidence"),
        values=dict(data[VALUES_TABLE]),
    )


def _check_size(size: int) -> None:
    """Raise ``ProfileFileError`` when *size* bytes is over the profile file cap."""
    if size > MAX_PROFILE_BYTES:
        raise ProfileFileError(
            f"The file is over {MAX_PROFILE_BYTES // 1024} KB, too large for a profile file"
        )


def _parse_toml(text: str) -> dict[str, Any]:
    """The parsed TOML in *text*; raises ``ProfileFileError`` when it is too large or not TOML."""
    _check_size(len(text.encode("utf-8")))
    try:
        return tomllib.loads(text)
    # tomllib raises RecursionError on arrays nested a few hundred deep
    except (tomllib.TOMLDecodeError, RecursionError) as exc:
        raise ProfileFileError(f"Not valid TOML: {exc}") from None


def read_profile_text(path: Path) -> str:
    """The text of the file at *path*; raises ``ProfileFileError`` when it cannot be read."""
    try:
        with path.open("rb") as f:
            data = f.read(MAX_PROFILE_BYTES + 1)
    except OSError as exc:
        raise ProfileFileError(f"Cannot read the file: {exc}") from None
    _check_size(len(data))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ProfileFileError("Not UTF-8 text") from None


def _load_toml(path: Path) -> dict[str, Any]:
    """The parsed TOML at *path*; raises ``ProfileFileError`` when it cannot be read or parsed."""
    return _parse_toml(read_profile_text(path))


def read_entry(path: Path, folder: ProfileFolder) -> ProfileEntry:
    """Read and validate one profile file; a broken file is returned with its reason."""
    try:
        data = _load_toml(path)
    except ProfileFileError as exc:
        return ProfileEntry(path.stem, folder, path, None, str(exc))
    try:
        profile = parse_profile(data, path.stem, folder)
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


def _reserved_reason(name: str) -> str:
    return f"Reserved name: {name} is a built-in profile"


def _settle_names(entries: list[ProfileEntry]) -> list[ProfileEntry]:
    """Break non-builtin entries that take a built-in name; shadow each lower entry by name."""
    reserved = {profile_key(e.name) for e in entries if e.folder is ProfileFolder.BUILTIN}
    owner: dict[str, ProfileFolder] = {}
    settled: list[ProfileEntry] = []
    for entry in entries:
        key = profile_key(entry.name)
        if entry.folder is not ProfileFolder.BUILTIN and key in reserved:
            entry = _broken(entry, _reserved_reason(entry.name))
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


@dataclass(frozen=True)
class ProfileValidation:
    """Every problem found in one profile file; empty ``problems`` means it is valid."""

    path: Path
    name: str
    problems: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """True when the file has no problems."""
        return not self.problems


@dataclass(frozen=True)
class PlannedWrite:
    """A validated profile file and the free path it goes to; nothing is written until ``write``."""

    path: Path
    folder: ProfileFolder
    profile: ProfileFile
    text: str
    replacing: Path | None

    def write(self) -> Path:
        """Write the file atomically, then remove the file it replaces when that is another file."""
        stale = (
            None
            if self.replacing is None or _same_file(self.replacing, self.path)
            else self.replacing
        )
        write_text_atomically(self.path, self.text)
        if stale is not None:
            stale.unlink(missing_ok=True)
        return self.path


def _same_file(first: Path, second: Path) -> bool:
    """True when both paths exist and name one file, whatever their spelling or case."""
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def builtin_keys() -> frozenset[str]:
    """The lookup keys of the built-in profiles, which no other profile may take."""
    folder = PACKAGE_PROFILES_DIR / BUILTIN_DIRNAME
    return frozenset(profile_key(e.name) for e in _read_folder(ProfileFolder.BUILTIN, folder))


def validate_file(path: Path, folder: ProfileFolder) -> ProfileValidation:
    """Every problem with *path* as a profile file in *folder*."""
    try:
        text = read_profile_text(path)
    except ProfileFileError as exc:
        return ProfileValidation(path, path.stem, (str(exc),))
    return validate_text(text, path, folder)


def validate_text(text: str, path: Path, folder: ProfileFolder) -> ProfileValidation:
    """Every problem with *text*, the content of the file at *path*, as a profile in *folder*."""
    try:
        data = _parse_toml(text)
    except ProfileFileError as exc:
        return ProfileValidation(path, path.stem, (str(exc),))
    name = _entry_name(data, path.stem)
    problems = profile_problems(data, path.stem, folder)
    if folder is not ProfileFolder.BUILTIN and profile_key(name) in builtin_keys():
        problems.append(_reserved_reason(name))
    return ProfileValidation(path, name, tuple(problems))


def profile_text(profile: ProfileFile) -> str:
    """*profile* as the text of a clean profile file."""
    authors = [
        {k: v for k, v in (("name", a.name), ("github", a.github)) if v is not None}
        for a in profile.authors
    ]
    meta = {
        "name": profile.name,
        "description": profile.description,
        "authors": authors or None,
        "tested_on": profile.tested_on,
        "format": profile.format,
        "min_lilbee": profile.min_lilbee,
        "evidence": profile.evidence,
    }
    table = {key: value for key, value in meta.items() if value is not None}
    return tomli_w.dumps({META_TABLE: table, VALUES_TABLE: dict(profile.values)})


def parse_text(text: str, stem: str, folder: ProfileFolder) -> ProfileFile:
    """Validate *text* as a profile file named *stem* when it names none; raises on a problem."""
    return parse_profile(_parse_toml(text), stem, folder)


def _clash(
    directory: Path, folder: ProfileFolder, planned: Path, replacing: Path | None
) -> Path | None:
    """A file in *directory*, other than *replacing*, at *planned* or holding the same name."""
    if not directory.is_dir():
        return None
    key = planned.stem
    for path in sorted(directory.glob(f"*{PROFILE_SUFFIX}")):
        # a case-insensitive file system treats Mine.toml and mine.toml as one file
        same_file = path.name.casefold() == planned.name.casefold()
        if path != replacing and (same_file or profile_key(read_entry(path, folder).name) == key):
            return path
    return None


def plan_write(
    directory: Path, folder: ProfileFolder, text: str, *, stem: str, replacing: Path | None = None
) -> PlannedWrite:
    """Validate *text* and pick its file in *directory*, named by the profile's slug.

    Raises ``ProfileFileError`` when the text is invalid or another file there has the name;
    *replacing* is the file this write supersedes.
    """
    profile = parse_text(text, stem, folder)
    key = profile_key(profile.name)
    if key in builtin_keys():
        raise ProfileFileError(_reserved_reason(profile.name))
    path = directory / f"{key}{PROFILE_SUFFIX}"
    clash = _clash(directory, folder, path, replacing)
    if clash is not None:
        raise ProfileFileError(f"A profile named {profile.name} already exists: {clash}")
    return PlannedWrite(path, folder, profile, text, replacing)
