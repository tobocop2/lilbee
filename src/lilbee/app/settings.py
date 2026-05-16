"""Canonical write boundary for lilbee configuration.

Every entry point that writes settings (HTTP ``PATCH /api/config``, the
TUI's ``set_setting`` action, the MCP ``settings_set`` tool) routes
through ``apply_settings_update`` so the validation, persistence, and
cache-invalidation policy lives in one place. Read-side surfaces use
``list_settings`` / ``get_setting`` to introspect the writable schema
without re-deriving metadata from pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_core import PydanticUndefined

from lilbee.cli.settings_map import SETTINGS_MAP, SettingDef
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS,
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core import settings as persistent_settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.keys import PROVIDER_API_KEYS

_MIN_CHUNK_SIZE = 64


@dataclass(frozen=True)
class SettingInfo:
    """Externally-facing description of a single writable setting."""

    key: str
    value: Any
    default: Any
    type: str
    nullable: bool
    group: str
    help_text: str
    choices: tuple[str, ...] | None
    reindex_required: bool


@dataclass(frozen=True)
class SettingsUpdateResult:
    """Outcome of an ``apply_settings_update`` call."""

    updated: list[str]
    reindex_required: bool


def _annotation_name(annotation: Any) -> str:
    """Render a pydantic field annotation as a short MCP-friendly type string."""
    import types
    from pathlib import Path
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        parts = [_annotation_name(a) for a in get_args(annotation)]
        return "|".join(parts)
    if annotation is type(None):
        return "null"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    if annotation is Path:
        return "str"
    if origin in (list, frozenset, set, tuple):
        return "list"
    name = getattr(annotation, "__name__", None)
    return name or str(annotation)


def _setting_default(key: str) -> Any:
    """Return the pydantic default for ``key``, or ``None`` if unset."""
    info = Config.model_fields[key]
    if info.default_factory is not None:
        return info.default_factory()  # type: ignore[call-arg]
    if info.default is PydanticUndefined:
        return None
    return info.default


def _writable_keys() -> list[str]:
    """Names of every writable config field, including model role slots."""
    return sorted(set(WRITABLE_CONFIG_FIELDS) | set(MODEL_ROLE_FIELDS))


def _setting_info(key: str, definition: SettingDef | None) -> SettingInfo:
    field_info = Config.model_fields[key]
    nullable = WRITABLE_CONFIG_FIELDS.get(key, False) or key in MODEL_ROLE_FIELDS
    group = definition.group if definition else "Models"
    help_text = definition.help_text if definition else ""
    choices = definition.choices if definition else None
    return SettingInfo(
        key=key,
        value=getattr(cfg, key),
        default=_setting_default(key),
        type=_annotation_name(field_info.annotation),
        nullable=nullable,
        group=group,
        help_text=help_text,
        choices=choices,
        reindex_required=key in REINDEX_FIELDS,
    )


def list_settings(group: str | None = None) -> list[SettingInfo]:
    """List every writable setting, optionally filtered by group."""
    infos = [_setting_info(key, SETTINGS_MAP.get(key)) for key in _writable_keys()]
    if group is not None:
        infos = [info for info in infos if info.group.lower() == group.lower()]
    return sorted(infos, key=lambda info: (info.group, info.key))


def get_setting(key: str) -> SettingInfo:
    """Return the ``SettingInfo`` for one writable key. Raises ``KeyError`` if unknown."""
    if key not in WRITABLE_CONFIG_FIELDS and key not in MODEL_ROLE_FIELDS:
        raise KeyError(f"Unknown or read-only setting: {key}")
    return _setting_info(key, SETTINGS_MAP.get(key))


def _validate(updates: dict[str, Any]) -> None:
    """Reject unknown keys, null on non-nullable, and out-of-range chunk sizes."""
    for key, value in updates.items():
        if key not in WRITABLE_CONFIG_FIELDS:
            raise ValueError(f"Unknown or read-only setting: {key}")
        if value is None and not WRITABLE_CONFIG_FIELDS[key]:
            raise ValueError(f"Setting '{key}' does not accept null")
    chunk_val = updates.get("chunk_size")
    if isinstance(chunk_val, int) and chunk_val < _MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {_MIN_CHUNK_SIZE}")


def _apply_with_rollback(
    updates: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Set each key on cfg with snapshot/rollback. Returns (persist, delete)."""
    snapshot = {k: getattr(cfg, k) for k in updates}
    to_persist: dict[str, str] = {}
    to_delete: list[str] = []
    try:
        for key, value in updates.items():
            if value is None:
                setattr(cfg, key, None)
                to_delete.append(key)
            else:
                setattr(cfg, key, value)
                normalized = getattr(cfg, key)
                if isinstance(normalized, list):
                    to_persist[key] = "\n".join(str(x) for x in normalized)
                else:
                    to_persist[key] = str(normalized)
    except Exception:
        for k, v in snapshot.items():
            setattr(cfg, k, v)
        raise
    return to_persist, to_delete


def _invalidate_caches(changed_keys: set[str]) -> None:
    """Drop every read-side cache that depends on a changed setting.

    Mirrors the historical TUI ``settings_changed_signal`` subscribers but
    runs unconditionally so non-TUI surfaces (HTTP, MCP, CLI) get the same
    invalidation. Importing the heavy provider/modelhub modules lazily so
    ``lilbee mcp`` boot stays fast.
    """
    if not changed_keys:
        return
    if changed_keys & MODEL_ROLE_FIELDS:
        from lilbee.modelhub.model_info import invalidate_cache as invalidate_arch_cache

        invalidate_arch_cache()
    from lilbee.providers.llama_cpp.provider import (
        LOAD_AFFECTING_KEYS,
        PER_CALL_RELOADABLE_KEYS,
    )

    load_affecting = (changed_keys & LOAD_AFFECTING_KEYS) - PER_CALL_RELOADABLE_KEYS
    if load_affecting:
        from lilbee.app.services import peek_services

        services = peek_services()
        if services is not None:
            services.provider.invalidate_load_cache()
    if changed_keys & PROVIDER_API_KEYS:
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        inject_provider_keys()


def apply_settings_update(updates: dict[str, Any]) -> SettingsUpdateResult:
    """Validate, apply, persist, and invalidate caches for a batch of updates.

    Atomicity contract: if any value fails pydantic validation, every
    other field in this batch is rolled back to its prior value and
    nothing is persisted to disk. Callers see either a ``ValueError``
    (no state changed) or a successful result (every key applied and
    flushed to ``config.toml``).
    """
    _validate(updates)
    to_persist, to_delete = _apply_with_rollback(updates)
    if to_persist:
        persistent_settings.update_values(cfg.data_root, to_persist)
    if to_delete:
        persistent_settings.delete_values(cfg.data_root, to_delete)
    _invalidate_caches(set(updates))
    reindex_required = bool(REINDEX_FIELDS & set(updates))
    return SettingsUpdateResult(
        updated=sorted(updates),
        reindex_required=reindex_required,
    )


def reset_settings(keys: list[str]) -> SettingsUpdateResult:
    """Reset each key to its pydantic default and apply through the write boundary.

    Nullable fields collapse to ``None`` so the next process load picks
    up the field default; non-nullable fields are written to disk with
    their canonical default value.
    """
    for key in keys:
        if key not in WRITABLE_CONFIG_FIELDS:
            raise ValueError(f"Unknown or read-only setting: {key}")
    updates: dict[str, Any] = {}
    for key in keys:
        if WRITABLE_CONFIG_FIELDS[key]:
            updates[key] = None
        else:
            updates[key] = _setting_default(key)
    return apply_settings_update(updates)
