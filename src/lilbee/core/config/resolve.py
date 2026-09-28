"""Each setting's effective value and the layer that supplies it: env > user > profile > built-in.

Per-run CLI flags and ``apply_ephemeral_model_swap`` set cfg for one process and are not sources.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .defaults import CONFIG_FILE_NAME, SKIP_TOML_ENV, env_var_name
from .enums import ProfileScope, SettingSource
from .model import Config, value_is_set

log = logging.getLogger(__name__)

PROFILE_TABLE = "profile"
PROFILE_NAME_KEY = "name"
PROFILE_VALUES_KEY = "values"

# Built from data_root at construction; their pydantic default is an unresolved sentinel.
ROOT_DERIVED_FIELDS: frozenset[str] = frozenset(
    {"data_root", "documents_dir", "data_dir", "lancedb_dir", "models_dir"}
)


def _profile_scope(info: FieldInfo) -> ProfileScope | None:
    extra = info.json_schema_extra
    # pydantic types json_schema_extra as a dict or a callable; ConfigField always sets a dict
    scope = extra.get("profile") if isinstance(extra, dict) else None
    return None if scope is None else ProfileScope(str(scope))


# The settings a profile may hold, each with the part of lilbee it tunes.
PROFILE_FIELDS: Mapping[str, ProfileScope] = MappingProxyType(
    {
        name: scope
        for name, info in Config.model_fields.items()
        if (scope := _profile_scope(info)) is not None
    }
)


@dataclass(frozen=True)
class ProfileTable:
    """The ``[profile]`` table of config.toml: the applied profile's name and recorded values."""

    name: str | None
    values: Mapping[str, Any]


@dataclass(frozen=True)
class SettingLayers:
    """The explicit layers read once: env vars, config.toml keys, and the ``[profile]`` values."""

    env: Mapping[str, str]
    user: Mapping[str, Any]
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class Resolved:
    """A setting's effective value and the layer it came from."""

    value: Any
    source: SettingSource


def _env_layer() -> dict[str, str]:
    """The set ``LILBEE_<FIELD>`` values, keyed by field name; a blank value counts as empty."""
    layer: dict[str, str] = {}
    for name in Config.model_fields:
        raw = os.environ.get(env_var_name(name))
        if raw is not None and value_is_set(name, raw.strip()):
            layer[name] = raw
    return layer


def _present(table: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unset values: an empty string, except on a model role that can be off."""
    return {key: value for key, value in table.items() if value_is_set(key, value)}


def _read_toml(path: Path) -> dict[str, Any]:
    """The parsed config.toml; empty when skipped, missing or unreadable."""
    if os.environ.get(SKIP_TOML_ENV) == "1" or not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (ValueError, OSError):
        log.warning("Failed to read %s, ignoring", path)
        return {}


def _profile_table(data: Mapping[str, Any], path: Path) -> ProfileTable:
    """The ``[profile]`` table in *data*, keeping only the values a profile may hold."""
    table = data.get(PROFILE_TABLE)
    # untyped TOML: a hand edit can put any type under [profile], its name or its values
    if not isinstance(table, dict):
        return ProfileTable(name=None, values={})
    name = table.get(PROFILE_NAME_KEY)
    values = table.get(PROFILE_VALUES_KEY)
    kept: dict[str, Any] = {}
    for key, value in (values if isinstance(values, dict) else {}).items():
        if key in PROFILE_FIELDS:
            kept[key] = value
        else:
            log.warning(
                "Ignoring %s in the [profile] values of %s: profiles cannot set it", key, path
            )
    return ProfileTable(name=name if isinstance(name, str) else None, values=kept)


def read_profile_table(root: Path) -> ProfileTable:
    """The ``[profile]`` table of ``root/config.toml``."""
    path = root / CONFIG_FILE_NAME
    return _profile_table(_read_toml(path), path)


def read_layers(root: Path) -> SettingLayers:
    """Read the env vars and ``root/config.toml`` once."""
    path = root / CONFIG_FILE_NAME
    data = _read_toml(path)
    user = {key: value for key, value in data.items() if key != PROFILE_TABLE}
    return SettingLayers(
        env=_env_layer(),
        user=_present(user),
        profile=_present(_profile_table(data, path).values),
    )


def builtin_value(key: str) -> Any:
    """The pydantic default for ``key``, or None when the field has none."""
    info = Config.model_fields[key]
    if info.default_factory is not None:
        return info.default_factory()  # type: ignore[call-arg]
    if info.default is PydanticUndefined:
        return None
    return info.default


def _is_derived(info: FieldInfo) -> bool:
    extra = info.json_schema_extra
    return isinstance(extra, dict) and bool(extra.get("derived", False))


def resolve(key: str, layers: SettingLayers) -> Resolved:
    """The effective value of ``key`` and its source."""
    for source, layer in (
        (SettingSource.ENV, layers.env),
        (SettingSource.USER, layers.user),
        (SettingSource.PROFILE, layers.profile),
    ):
        if key in layer:
            return Resolved(layer[key], source)
    fallback = (
        SettingSource.AUTO if _is_derived(Config.model_fields[key]) else SettingSource.BUILT_IN
    )
    return Resolved(builtin_value(key), fallback)


def resolve_all(layers: SettingLayers) -> dict[str, Resolved]:
    """Resolve every Config field."""
    return {key: resolve(key, layers) for key in Config.model_fields}
