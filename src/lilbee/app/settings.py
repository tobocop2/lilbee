"""Canonical write boundary for lilbee configuration."""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from pydantic_core import PydanticUndefined

from lilbee.app.settings_map import SETTINGS_MAP, SettingDef, SettingGroup
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS,
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core import settings as persistent_settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.keys import (
    LOAD_AFFECTING_KEYS,
    PROVIDER_API_KEYS,
    PROVIDER_SWITCHING_KEYS,
)

if TYPE_CHECKING:
    from lilbee.modelhub.registry import ModelRegistry

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
    nullable = _is_nullable(key)
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


def _as_int_setting(value: Any) -> int | None:
    """Coerce a settings value to int the way pydantic will, or None if not numeric.

    MCP settings_set forwards raw JSON, so a numeric setting can arrive as a
    string (``{"chunk_overlap": "1000"}``). The cross-field guards must compare
    the coerced int, not skip on the string and let pydantic accept an
    unvalidated value downstream. ``bool`` is excluded (it is not a meaningful
    chunk size) and non-numeric strings fall through to pydantic's type error.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _validate(updates: dict[str, Any]) -> None:
    """Reject unknown keys, null on non-nullable, and out-of-range chunk sizes."""
    for key, value in updates.items():
        if not _is_settable(key):
            raise ValueError(f"Unknown or read-only setting: {key}")
        if value is None and not _is_nullable(key):
            raise ValueError(f"Setting '{key}' does not accept null")
    new_chunk_size = _as_int_setting(updates.get("chunk_size"))
    if new_chunk_size is not None and new_chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {_MIN_CHUNK_SIZE}")
    effective_chunk_size = new_chunk_size if new_chunk_size is not None else cfg.chunk_size
    new_overlap = _as_int_setting(updates.get("chunk_overlap"))
    if new_overlap is not None and new_overlap >= effective_chunk_size:
        raise ValueError(
            f"chunk_overlap ({new_overlap}) must be < chunk_size ({effective_chunk_size})"
        )


def _coerce_value(key: str, value: Any) -> Any:
    """Canonicalize value before cfg assignment; model-role slots run task validation."""
    if key in MODEL_ROLE_FIELDS and isinstance(value, str):
        # heavy: role_validator pulls catalog + modelhub transitively (~300 ms)
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


def _reload_changed_roles(changed_keys: set[str]) -> None:
    """Off-thread reload for each changed model-role server; full off-thread drop otherwise.

    A model-role change (chat_model/embedding_model/reranker_model/vision_model)
    respawns only that role's server via the per-role reload, so unrelated roles
    keep serving uninterrupted. A genuinely role-agnostic load key (num_ctx,
    kv_cache_type) has no single owning role, so it falls back to dropping the
    whole fleet. Both paths run off the caller's thread, so the settings write
    never blocks on a slow stop-and-respawn.
    """
    from lilbee.app.services import peek_services
    from lilbee.providers.roles import MODEL_FIELD_TO_ROLE

    services = peek_services()
    if services is None:
        return
    changed_role_fields = changed_keys & MODEL_ROLE_FIELDS
    for field in changed_role_fields:
        services.reload_role(MODEL_FIELD_TO_ROLE[field])
    role_agnostic = (changed_keys & LOAD_AFFECTING_KEYS) - MODEL_ROLE_FIELDS
    if role_agnostic:
        services.provider.drop_loaded_models_async()


def requires_services_reset(updates: dict[str, Any]) -> bool:
    """True if applying *updates* would tear down and rebuild the Services singleton.

    A provider switch reconstructs the provider via ``create_provider``, which
    only runs at services init, so it forces a full ``reset_services()``. Callers
    on the shared HTTP daemon use this to refuse the swap rather than tear the
    singleton down under concurrent in-flight handlers.
    """
    return bool(set(updates) & PROVIDER_SWITCHING_KEYS)


def provider_reset_refused_message(action: str) -> str:
    """Shared user-facing refusal for a provider *action* on the HTTP server.

    *action* is the verb shown to the user, e.g. ``"Switching"`` or
    ``"Resetting"``. Kept in one place so the daemon entry points (MCP
    settings_set / settings_reset, REST config) cannot drift apart.
    """
    return (
        f"{action} the model provider is unavailable on the HTTP server: it rebuilds "
        "the shared engine for every connected client. Change it from the CLI."
    )


def _invalidate_caches(changed_keys: set[str]) -> None:
    """Drop every read-side cache whose freshness depends on a changed setting."""
    if not changed_keys:
        return
    if changed_keys & MODEL_ROLE_FIELDS:
        # heavy: model_info reads GGUF headers with the gguf parser (~130 ms)
        from lilbee.modelhub.model_info import invalidate_cache as invalidate_arch_cache

        invalidate_arch_cache()
    if changed_keys & LOAD_AFFECTING_KEYS:
        # heavy: app.services pulls the provider stack + lancedb (~70 ms)
        _reload_changed_roles(changed_keys)
    if changed_keys & PROVIDER_API_KEYS:
        # heavy: sdk_llm_provider pulls litellm fanout (~145 ms)
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        inject_provider_keys()
    if changed_keys & PROVIDER_SWITCHING_KEYS:
        # Swap requires reconstructing the provider singleton via
        # providers.factory.create_provider, only called at services init.
        from lilbee.app.services import reset_services

        reset_services()


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
    embed_in_batch = "embedding_model" in updates
    # Derived (not user-writable) fields applied alongside the validated batch.
    effective_updates = dict(updates)
    if embed_in_batch:
        # Pin the OLD ref into store meta before mutation, otherwise the
        # next read lazy-initializes meta from the NEW cfg and silently
        # hides the dimension drift. Runs even when the value is unchanged
        # so a legacy meta row is always canonicalized on the first swap
        # attempt.
        _pin_legacy_store_meta()
        # Track the new embedder's output width so a fresh index is built at
        # the right dimension (embedding_dim is derived, not in SETTINGS_MAP).
        dim = _embedder_dim_from_gguf(updates["embedding_model"])
        if dim is not None:
            effective_updates["embedding_dim"] = dim
    to_persist, to_delete, snapshot = _apply_with_rollback(effective_updates)
    try:
        if to_persist:
            persistent_settings.update_values(cfg.data_root, to_persist)
        if to_delete:
            persistent_settings.delete_values(cfg.data_root, to_delete)
    except OSError:
        _restore_snapshot(snapshot)
        raise
    _invalidate_caches(set(effective_updates))
    reindex_required = bool(REINDEX_FIELDS & set(updates))
    if embed_in_batch:
        reindex_required = reindex_required or _embed_reindex_required(updates["embedding_model"])
    return SettingsUpdateResult(
        updated=sorted(updates),
        reindex_required=reindex_required,
    )


def _pin_legacy_store_meta() -> None:
    """Pin the current embedding ref into store meta before swapping it."""
    # heavy: ~100ms (lance + store init); only paid when embedding_model is in the batch.
    from lilbee.app.services import get_services

    get_services().store.initialize_meta_if_legacy()


def _embedder_dim_from_gguf(ref: str, registry: ModelRegistry | None = None) -> int | None:
    """The embedder's output width from its GGUF header (``<arch>.embedding_length``).

    None when the model can't be resolved or the header lacks the field. Cheap: a
    cached header read, no load. *registry* is forwarded to resolve the GGUF without
    ``get_services()`` (callers running inside its construction).
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    try:
        # resolve_model_path raises ProviderError for a non-native (ollama/SDK) ref,
        # which has no local GGUF -- those embedders carry no width to derive here.
        meta = read_gguf_metadata(resolve_model_path(ref, registry))
    except (ProviderError, ValueError, OSError, RuntimeError, TypeError):
        return None
    raw = meta.get("embedding_length") if meta else None
    if not raw:
        return None
    try:
        dim = int(raw)
    except (TypeError, ValueError):
        return None
    return dim if dim > 0 else None


def reconcile_embedding_dim(registry: ModelRegistry | None = None) -> None:
    """Pin ``cfg.embedding_dim`` to the native embedder's GGUF width before the store
    is built; no-op for non-native embedders or an already-matching dim."""
    dim = _embedder_dim_from_gguf(cfg.embedding_model, registry)
    if dim is not None and dim != cfg.embedding_dim:
        cfg.embedding_dim = dim


def _embed_reindex_required(new_ref: str) -> bool:
    """Compare *new_ref* to the persisted store meta; True if rebuild needed."""
    from lilbee.app.services import get_services
    from lilbee.data.store.lance_helpers import refs_compatible

    store = get_services().store
    store.canonicalize_meta_if_legacy()
    meta = store.get_meta()
    if meta is None:
        return False
    return not refs_compatible(
        meta["embedding_model"], new_ref, meta["embedding_dim"], meta["embedding_dim"]
    )


def reset_settings(keys: list[str], *, skip_unresettable: bool = False) -> SettingsUpdateResult:
    """Reset each key to its pydantic default and apply through the write boundary.

    Fields whose default is a known sentinel (currently ``documents_dir``,
    which resolves to ``data_root/documents`` at process start) are
    refused so a reset doesn't write the literal sentinel back. Pass
    ``skip_unresettable=True`` for bulk-reset gestures that should drop
    those fields rather than failing the whole batch.
    """
    for key in keys:
        if not _is_settable(key):
            raise ValueError(f"Unknown or read-only setting: {key}")
        if key in _NO_RESET_FIELDS and not skip_unresettable:
            raise ValueError(
                f"'{key}' has no resettable default; pass an explicit value via settings_set."
            )
    updates: dict[str, Any] = {}
    for key in keys:
        if key in _NO_RESET_FIELDS:
            continue
        default = _setting_default(key)
        if default is None and _is_nullable(key):
            updates[key] = None
        else:
            updates[key] = default
    return apply_settings_update(updates)
