"""Canonical write boundary for lilbee configuration.

API-key fields declared with ``ConfigField(..., write_only=True)`` in
``core.config.model`` (every ``*_api_key`` and ``hf_token``) are
filtered out of ``list_settings`` and refused by ``get_setting``.
``apply_settings_update`` still accepts writes so an agent can
configure a key on the user's behalf; nothing reads it back.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from pydantic_core import PydanticUndefined

from lilbee.app.settings_map import SETTINGS_MAP, SettingDef, SettingGroup
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS,
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core import settings as persistent_settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.keys import PROVIDER_API_KEYS

_MIN_CHUNK_SIZE = 64

# Path-typed writable fields whose pydantic "default" is the unresolved
# sentinel ``Path()`` (a literal "."). The actual default is computed by
# the model_validator at process start (data_root/documents, vault_base
# stays as None). Resetting these via the boundary would corrupt the
# install, so they are refused at the reset gate.
_NO_RESET_FIELDS: frozenset[str] = frozenset({"documents_dir"})


@dataclass(frozen=True)
class SettingInfo:
    """Externally-facing description of a single writable setting."""

    key: str
    value: Any
    default: Any
    type: str
    nullable: bool
    group: SettingGroup
    help_text: str
    choices: tuple[str, ...] | None
    reindex_required: bool


@dataclass(frozen=True)
class SettingsUpdateResult:
    """Outcome of an ``apply_settings_update`` call."""

    updated: list[str]
    reindex_required: bool


_SCALAR_TYPE_NAMES: dict[type, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    Path: "str",
    type(None): "null",
}
_COLLECTION_ORIGINS = (list, frozenset, set, tuple)
_UNION_ORIGINS = (Union, types.UnionType)


def _annotation_name(annotation: Any) -> str:
    """Render a pydantic field annotation as a short MCP-friendly type string."""
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        return "|".join(_annotation_name(a) for a in get_args(annotation))
    scalar = _SCALAR_TYPE_NAMES.get(annotation)
    if scalar is not None:
        return scalar
    if origin in _COLLECTION_ORIGINS:
        return "list"
    return getattr(annotation, "__name__", None) or str(annotation)


def _setting_default(key: str) -> Any:
    """Return the pydantic default for ``key``, or ``None`` if unset."""
    info = Config.model_fields[key]
    if info.default_factory is not None:
        return info.default_factory()  # type: ignore[call-arg]
    if info.default is PydanticUndefined:
        return None
    return info.default


def _is_write_only(key: str) -> bool:
    """Return True for fields persisted but never read back (API keys, hf_token)."""
    extra = Config.model_fields[key].json_schema_extra
    if isinstance(extra, dict):
        return bool(extra.get("write_only", False))
    return False


def _public_writable_keys() -> list[str]:
    """Names of every writable config field minus write-only secrets."""
    keys = set(WRITABLE_CONFIG_FIELDS) | set(MODEL_ROLE_FIELDS)
    return sorted(k for k in keys if not _is_write_only(k))


def _setting_info(key: str, definition: SettingDef | None) -> SettingInfo:
    field_info = Config.model_fields[key]
    nullable = WRITABLE_CONFIG_FIELDS.get(key, False) or key in MODEL_ROLE_FIELDS
    group = definition.group if definition else SettingGroup.MODELS
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


def _parse_group(group: SettingGroup | str) -> SettingGroup:
    """Resolve a group value or label to a ``SettingGroup``. Case-insensitive on the value."""
    if isinstance(group, SettingGroup):
        return group
    normalized = group.strip().lower()
    for candidate in SettingGroup:
        if candidate.value.lower() == normalized:
            return candidate
    raise ValueError(
        f"Unknown setting group: {group!r}. Valid groups: "
        f"{', '.join(g.value for g in SettingGroup)}"
    )


def list_settings(group: SettingGroup | str | None = None) -> list[SettingInfo]:
    """List every writable non-secret setting, optionally filtered by group (case-insensitive)."""
    infos = [_setting_info(key, SETTINGS_MAP.get(key)) for key in _public_writable_keys()]
    if group is not None:
        wanted = _parse_group(group)
        infos = [info for info in infos if info.group == wanted]
    return sorted(infos, key=lambda info: (info.group.value, info.key))


def get_setting(key: str) -> SettingInfo:
    """Return the ``SettingInfo`` for one writable non-secret key."""
    if not _is_settable(key):
        raise KeyError(f"Unknown or read-only setting: {key}")
    if _is_write_only(key):
        raise KeyError(f"Setting '{key}' is write-only and cannot be read back")
    return _setting_info(key, SETTINGS_MAP.get(key))


def _is_settable(key: str) -> bool:
    return key in WRITABLE_CONFIG_FIELDS or key in MODEL_ROLE_FIELDS


def _is_nullable(key: str) -> bool:
    """Return True if ``key`` accepts ``None`` to clear the persisted entry."""
    if key in WRITABLE_CONFIG_FIELDS:
        return WRITABLE_CONFIG_FIELDS[key]
    return False


def _validate(updates: dict[str, Any]) -> None:
    """Reject unknown keys, null on non-nullable, and out-of-range chunk sizes."""
    for key, value in updates.items():
        if not _is_settable(key):
            raise ValueError(f"Unknown or read-only setting: {key}")
        if value is None and not _is_nullable(key):
            raise ValueError(f"Setting '{key}' does not accept null")
    new_chunk_size = updates.get("chunk_size")
    if isinstance(new_chunk_size, int) and new_chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {_MIN_CHUNK_SIZE}")
    effective_chunk_size = new_chunk_size if isinstance(new_chunk_size, int) else cfg.chunk_size
    new_overlap = updates.get("chunk_overlap")
    if isinstance(new_overlap, int) and new_overlap >= effective_chunk_size:
        raise ValueError(
            f"chunk_overlap ({new_overlap}) must be < chunk_size ({effective_chunk_size})"
        )


def _coerce_value(key: str, value: Any) -> Any:
    """Canonicalize value before cfg assignment; model-role slots run task validation."""
    if key in MODEL_ROLE_FIELDS and isinstance(value, str):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        return validate_model_task_assignment(key, value)
    return value


def _apply_with_rollback(
    updates: dict[str, Any],
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    """Set each key on cfg with snapshot/rollback. Returns (persist, delete, snapshot)."""
    snapshot = {k: getattr(cfg, k) for k in updates}
    to_persist: dict[str, str] = {}
    to_delete: list[str] = []
    try:
        for key, raw in updates.items():
            if raw is None:
                setattr(cfg, key, None)
                to_delete.append(key)
                continue
            setattr(cfg, key, _coerce_value(key, raw))
            normalized = getattr(cfg, key)
            if isinstance(normalized, list):
                to_persist[key] = "\n".join(str(x) for x in normalized)
            else:
                to_persist[key] = str(normalized)
    except Exception:
        _restore_snapshot(snapshot)
        raise
    return to_persist, to_delete, snapshot


def _restore_snapshot(snapshot: dict[str, Any]) -> None:
    for key, value in snapshot.items():
        setattr(cfg, key, value)


# Mirrors ``lilbee.providers.llama_cpp.provider.LOAD_AFFECTING_KEYS`` so the
# write boundary doesn't pay the llama-cpp import cost on every settings_set.
_LOAD_AFFECTING_KEYS = frozenset(
    {"num_ctx", "chat_model", "embedding_model", "vision_model", "reranker_model"}
)
_PER_CALL_RELOADABLE_KEYS = frozenset({"chat_model", "vision_model"})


def _invalidate_caches(changed_keys: set[str]) -> None:
    """Drop every read-side cache whose freshness depends on a changed setting."""
    if not changed_keys:
        return
    if changed_keys & MODEL_ROLE_FIELDS:
        from lilbee.modelhub.model_info import invalidate_cache as invalidate_arch_cache

        invalidate_arch_cache()
    load_affecting = (changed_keys & _LOAD_AFFECTING_KEYS) - _PER_CALL_RELOADABLE_KEYS
    if load_affecting:
        from lilbee.app.services import peek_services

        services = peek_services()
        if services is not None:
            services.provider.invalidate_load_cache()
    if changed_keys & PROVIDER_API_KEYS:
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        inject_provider_keys()


def apply_settings_update(
    updates: dict[str, Any],
    *,
    allow_model_roles: bool = True,
) -> SettingsUpdateResult:
    """Validate, apply, persist, and invalidate caches for a batch of updates.

    Atomic on validation: a rejection rolls every field back and writes
    nothing. Atomic on disk failure: an ``OSError`` from the TOML write
    also restores the in-memory snapshot before re-raising. Cache
    invalidation runs only after a successful persist.

    Pass ``allow_model_roles=False`` to reject ``chat_model`` /
    ``embedding_model`` / ``vision_model`` / ``reranker_model`` at the
    boundary; the HTTP PATCH /api/config surface uses this to route role
    writes through PUT /api/models/<role>.
    """
    if not allow_model_roles:
        rejected = MODEL_ROLE_FIELDS & set(updates)
        if rejected:
            offender = sorted(rejected)[0]
            raise ValueError(
                f"'{offender}' must be set through the dedicated model route, "
                "not the general settings update."
            )
    _validate(updates)
    to_persist, to_delete, snapshot = _apply_with_rollback(updates)
    try:
        if to_persist:
            persistent_settings.update_values(cfg.data_root, to_persist)
        if to_delete:
            persistent_settings.delete_values(cfg.data_root, to_delete)
    except OSError:
        _restore_snapshot(snapshot)
        raise
    _invalidate_caches(set(updates))
    reindex_required = bool(REINDEX_FIELDS & set(updates))
    return SettingsUpdateResult(
        updated=sorted(updates),
        reindex_required=reindex_required,
    )


def reset_settings(keys: list[str]) -> SettingsUpdateResult:
    """Reset each key to its pydantic default and apply through the write boundary.

    Fields whose default is a known sentinel (currently ``documents_dir``,
    which resolves to ``data_root/documents`` at process start) are
    refused so a reset doesn't write the literal sentinel back. The
    caller should ``settings_set`` an explicit value instead.
    """
    for key in keys:
        if not _is_settable(key):
            raise ValueError(f"Unknown or read-only setting: {key}")
        if key in _NO_RESET_FIELDS:
            raise ValueError(
                f"'{key}' has no resettable default; pass an explicit value via settings_set."
            )
    updates: dict[str, Any] = {}
    for key in keys:
        default = _setting_default(key)
        if default is None and _is_nullable(key):
            updates[key] = None
        else:
            updates[key] = default
    return apply_settings_update(updates)
