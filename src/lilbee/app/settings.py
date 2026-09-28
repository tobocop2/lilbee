"""Canonical write boundary for lilbee configuration."""

from __future__ import annotations

import errno
import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from lilbee.app.settings_map import SETTINGS_MAP, SettingDef, SettingGroup
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS,
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, Config, cfg
from lilbee.core.config.enums import SettingSource
from lilbee.core.config.keys import (
    LOAD_AFFECTING_KEYS,
    PROVIDER_SWITCHING_KEYS,
)
from lilbee.core.config.resolve import (
    PROFILE_FIELDS,
    Resolved,
    SettingLayers,
    builtin_value,
    read_layers,
    resolve,
    resolve_all,
)
from lilbee.core.config.schema import field_type_name
from lilbee.core.project_state import dismiss_tip
from lilbee.providers.roles import MODEL_FIELD_TO_ROLE, ROLE_GATE_FIELD_TO_ROLE
from lilbee.runtime.progress import OcrBackendUsed

if TYPE_CHECKING:
    from lilbee.modelhub.registry import ModelRegistry

log = logging.getLogger(__name__)

_MIN_CHUNK_SIZE = 64

# Keys that decide which OCR engine runs, and whether a set vision model goes unused.
OCR_SETTING_KEYS = frozenset({"enable_ocr", "vision_model"})
OCR_OFF_WARNING = (
    "OCR is off (enable_ocr = false), so the vision model {model} is not used. "
    "Scanned PDFs without a text layer are skipped. Set enable_ocr to true to OCR them."
)
_OCR_ENGINE_NOTES = {
    OcrBackendUsed.VISION: (
        "A vision model is set ({model}), so it is used instead of Tesseract. "
        "Clear vision_model to use Tesseract."
    ),
    OcrBackendUsed.TESSERACT: (
        "No vision model is set, so Tesseract runs OCR. "
        "Set vision_model to use a vision model instead."
    ),
}

# Path-typed writable fields whose pydantic "default" is the unresolved
# sentinel ``Path()`` (a literal "."). The actual default is computed by
# the model_validator at process start (data_root/documents, vault_base
# stays as None). Resetting these via the boundary would corrupt the
# install, so they are refused at the reset gate.
_NO_RESET_FIELDS: frozenset[str] = frozenset({"documents_dir"})


class _LayerChange(StrEnum):
    """A change to the setting layers whose resolved values are checked before it is written."""

    RESET = "reset"
    APPLY = "apply"


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
    source: SettingSource


@dataclass(frozen=True)
class SettingsUpdateResult:
    """Outcome of an ``apply_settings_update`` call."""

    updated: list[str]
    reindex_required: bool
    warnings: tuple[str, ...] = ()


def ocr_off_warning() -> str | None:
    """The warning for a set vision model that OCR being off keeps unused, else None."""
    if cfg.vision_model and cfg.enable_ocr is False:
        return OCR_OFF_WARNING.format(model=cfg.vision_model)
    return None


def ocr_engine_note() -> str | None:
    """Which OCR engine runs for scanned pages, or None when OCR is off."""
    backend = OcrBackendUsed.chosen(cfg.enable_ocr, cfg.vision_model)
    note = _OCR_ENGINE_NOTES.get(backend)
    return note.format(model=cfg.vision_model) if note is not None else None


def _update_warnings(changed_keys: set[str]) -> tuple[str, ...]:
    """Warnings about the configuration an update leaves behind."""
    warning = ocr_off_warning() if changed_keys & OCR_SETTING_KEYS else None
    return (warning,) if warning is not None else ()


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


def _setting_help(key: str, definition: SettingDef | None) -> str:
    """The one documented description for *key*.

    ``SettingDef.help_text`` wins because it is what the TUI already shows;
    a field with no settings-map entry falls back to its own description.
    """
    if definition is not None and definition.help_text:
        return definition.help_text
    return Config.model_fields[key].description or ""


def setting_sources() -> dict[str, SettingSource]:
    """The source of every Config field's effective value, from one read of config.toml."""
    return _sources(read_layers(cfg.data_root))


def _sources(layers: SettingLayers) -> dict[str, SettingSource]:
    """The source of every Config field's effective value under *layers*."""
    return {key: entry.source for key, entry in resolve_all(layers).items()}


def _reset_target(key: str, layers: SettingLayers) -> Any:
    """The value a reset falls back to without an env var: the profile's, else the built-in."""
    return layers.profile[key] if key in layers.profile else builtin_value(key)


def _setting_info(key: str, layers: SettingLayers, source: SettingSource) -> SettingInfo:
    definition = SETTINGS_MAP.get(key)
    nullable = _is_nullable(key)
    group = definition.group if definition else SettingGroup.MODELS
    help_text = _setting_help(key, definition)
    choices = definition.choices if definition else None
    return SettingInfo(
        key=key,
        value=getattr(cfg, key),
        default=_reset_target(key, layers),
        type=field_type_name(key),
        nullable=nullable,
        group=group,
        help_text=help_text,
        choices=choices,
        reindex_required=key in REINDEX_FIELDS,
        source=source,
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
    layers = read_layers(cfg.data_root)
    sources = _sources(layers)
    infos = [_setting_info(key, layers, sources[key]) for key in _public_writable_keys()]
    if group is not None:
        wanted = _parse_group(group)
        infos = [info for info in infos if info.group == wanted]
    return sorted(infos, key=lambda info: (info.group.value, info.key))


def get_setting(key: str) -> SettingInfo:
    """Return the ``SettingInfo`` for one writable non-secret key."""
    if not _is_settable(key):
        raise ValueError(f"Unknown or read-only setting: {key}")
    if _is_write_only(key):
        raise KeyError(f"Setting '{key}' is write-only and cannot be read back")
    layers = read_layers(cfg.data_root)
    return _setting_info(key, layers, _sources(layers)[key])


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
    new_ttl = _as_int_setting(updates.get("engine_idle_ttl_minutes"))
    if new_ttl is not None and new_ttl < 0:
        raise ValueError("engine_idle_ttl_minutes must be >= 0 (0 keeps weights loaded)")
    new_chunk_size = _as_int_setting(updates.get("chunk_size"))
    if new_chunk_size is not None and new_chunk_size < _MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {_MIN_CHUNK_SIZE}")
    effective_chunk_size = new_chunk_size if new_chunk_size is not None else cfg.chunk_size
    new_overlap = _as_int_setting(updates.get("chunk_overlap"))
    # Compare the effective overlap against the effective chunk_size so that
    # lowering chunk_size alone (below the already-persisted overlap) is caught,
    # not just an explicit new overlap.
    effective_overlap = new_overlap if new_overlap is not None else cfg.chunk_overlap
    if effective_overlap >= effective_chunk_size:
        raise ValueError(
            f"chunk_overlap ({effective_overlap}) must be < chunk_size ({effective_chunk_size})"
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
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Set each key on cfg with snapshot/rollback. Returns (persist, delete, snapshot)."""
    snapshot = {k: getattr(cfg, k) for k in updates}
    to_persist: dict[str, Any] = {}
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
                # Hand the scalar over with its type intact so config.toml holds
                # `true` and `2560`, not `"True"` and `"2560"`.
                to_persist[key] = normalized
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
    keep serving uninterrupted; so does a setting that gates a role (enable_ocr).
    A genuinely role-agnostic load key (num_ctx, kv_cache_type) has no single
    owning role, so it falls back to dropping the whole fleet. Both paths run off
    the caller's thread, so the settings write never blocks on a slow
    stop-and-respawn.
    """
    from lilbee.app.services import peek_services

    services = peek_services()
    if services is None:
        return
    changed_role_fields = changed_keys & MODEL_ROLE_FIELDS
    field_to_role = MODEL_FIELD_TO_ROLE | ROLE_GATE_FIELD_TO_ROLE
    reloaded = {field_to_role[field] for field in changed_keys & field_to_role.keys()}
    for role in sorted(reloaded):
        services.reload_role(role)
    if "vision_model" in changed_role_fields:
        # Register/unregister lilbee's xberg OCR backend on any vision-model
        # change (REST/MCP/TUI/CLI all funnel here), not just the REST route.
        from lilbee.data.extract.backends import BackendKind, sync_xberg_backend

        sync_xberg_backend(BackendKind.OCR, services.provider)
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


def config_write_failure_message(exc: OSError) -> str:
    """User-facing text for a failed config write; names the fix when the file is locked."""
    detail = f"Could not write {CONFIG_FILE_NAME}: {exc}."
    if exc.errno in (errno.EACCES, errno.EPERM):
        detail += f" Close the program that holds {CONFIG_FILE_NAME} open and try again."
    return detail


def _invalidate_caches(changed_keys: set[str]) -> None:
    """Drop every read-side cache whose freshness depends on a changed setting."""
    if not changed_keys:
        return
    if changed_keys & MODEL_ROLE_FIELDS:
        # heavy: model_info reads GGUF headers with the gguf parser (~130 ms)
        from lilbee.modelhub.model_info import invalidate_cache as invalidate_arch_cache

        invalidate_arch_cache()
    if changed_keys & (LOAD_AFFECTING_KEYS | ROLE_GATE_FIELD_TO_ROLE.keys()):
        # heavy: app.services pulls the provider stack + lancedb (~70 ms)
        _reload_changed_roles(changed_keys)
    if "token_sizing" in changed_keys:
        # Unregister lilbee's xberg tokenizer backend when token_sizing is turned
        # off (via any settings path); the chunker binds it on demand when on.
        from lilbee.app.services import peek_services
        from lilbee.data.extract.backends import BackendKind, sync_xberg_backend

        services = peek_services()
        if services is not None:
            sync_xberg_backend(BackendKind.TOKENIZER, services.provider)
    if changed_keys & PROVIDER_SWITCHING_KEYS:
        # Swap requires reconstructing the provider singleton via
        # providers.factory.create_provider, only called at services init.
        from lilbee.app.services import reset_services

        reset_services()
    if "mcp_tool_threads" in changed_keys:
        # Resize the running server's thread pool now instead of only at startup.
        from lilbee.server.app import reapply_thread_pool_ceiling

        reapply_thread_pool_ceiling()
    if "include_uncensored" in changed_keys:
        # The picks memoize per process; drop them so the toggle takes
        # effect on the next read instead of the next restart.
        from lilbee.catalog.picks import reset_picks

        reset_picks()


def apply_settings_update(
    updates: dict[str, Any],
    *,
    allow_model_roles: bool = True,
) -> SettingsUpdateResult:
    """Validate, apply, persist, and invalidate caches for a batch of updates.

    Atomic on validation: a rejection rolls every field back and writes
    nothing. Atomic on disk failure: an ``OSError`` from the TOML write, or a
    parse error reloading a corrupt config.toml, restores the in-memory
    snapshot before re-raising. Cache invalidation runs only after a
    successful persist.

    Pass ``allow_model_roles=False`` to reject ``chat_model`` /
    ``embedding_model`` / ``vision_model`` / ``reranker_model`` at the
    boundary; the HTTP PATCH /api/config surface uses this to route role
    writes through PUT /api/models/<role>.
    """
    if not allow_model_roles:
        _refuse_model_roles(updates)
    _validate(updates)
    embed_in_batch = "embedding_model" in updates
    if embed_in_batch:
        # Pin the OLD ref into store meta before mutation, otherwise the
        # next read lazy-initializes meta from the NEW cfg and silently
        # hides the dimension drift. Runs even when the value is unchanged
        # so a legacy meta row is always canonicalized on the first swap
        # attempt.
        _pin_legacy_store_meta()
    to_persist, to_delete, snapshot = _apply_with_rollback(updates)
    try:
        if to_persist:
            persistent_settings.update_values(cfg.data_root, to_persist)
        if to_delete:
            persistent_settings.delete_values(cfg.data_root, to_delete)
    except (OSError, ValueError):
        # OSError from the write, or a TOMLDecodeError (ValueError) when
        # update/delete reloads a corrupt on-disk config.toml: either way the
        # in-memory snapshot must be restored so cfg matches what was persisted.
        _restore_snapshot(snapshot)
        raise
    return _settle(set(updates), embed_in_batch=embed_in_batch)


def _refuse_model_roles(keys: Iterable[str]) -> None:
    """Refuse model-role keys, which the dedicated model route owns."""
    rejected = MODEL_ROLE_FIELDS & set(keys)
    if rejected:
        offender = sorted(rejected)[0]
        raise ValueError(
            f"'{offender}' must be set through the dedicated model route, "
            "not the general settings update."
        )


def _settle(keys: set[str], *, embed_in_batch: bool) -> SettingsUpdateResult:
    """Set *keys* on cfg from the resolver, then rederive, invalidate and report."""
    persistent_settings.sync_from_resolver(keys)
    _rederive_from_resolved(keys)
    _invalidate_caches(keys)
    reindex_required = bool((REINDEX_FIELDS - _inert_reindex_keys()) & keys)
    if embed_in_batch:
        reindex_required = reindex_required or _embed_reindex_required()
    return SettingsUpdateResult(
        updated=sorted(keys),
        reindex_required=reindex_required,
        warnings=_update_warnings(keys),
    )


def apply_ephemeral_model_swap(field: str, ref: str) -> None:
    """Apply a chat/embedding model swap to cfg for this process only.

    Performs the same embedding side effects as the persisted path (legacy
    store meta pinned under the OLD ref first, then embedding_dim re-derived
    for the new one) so the mismatch gate and table width stay correct, but
    never writes config.toml.
    """
    if field == "embedding_model":
        _pin_legacy_store_meta()
        dim = _embedder_dim_from_gguf(ref)
        setattr(cfg, field, ref)
        if dim is not None:
            cfg.embedding_dim = dim
        return
    setattr(cfg, field, ref)


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


def _rederive_from_resolved(keys: set[str]) -> None:
    """Recompute each setting derived from one of *keys*, from that key's resolved value."""
    if "embedding_model" in keys:
        reconcile_embedding_dim()


def _inert_reindex_keys() -> set[str]:
    """Reindex keys that change no extraction output under the effective config.

    xberg reads ``table_model`` only inside layout detection, so a change to it
    while ``layout_detection`` is off is not worth a rebuild.
    """
    return set() if cfg.layout_detection else {"table_model"}


def _embed_reindex_required() -> bool:
    """True when the persisted index was built with another embedder than cfg now names.

    Runs after the swap is applied, so the store compares its meta row against
    the new ref and the new model's width: the same verdict search refuses on.
    """
    from lilbee.app.services import get_services

    store = get_services().store
    store.canonicalize_meta_if_legacy()
    return store.index_mismatch() is not None


def reset_settings(
    keys: list[str], *, skip_unresettable: bool = False, allow_model_roles: bool = True
) -> SettingsUpdateResult:
    """Remove each key from config.toml and set cfg to the value the resolver then gives.

    ``documents_dir`` has no default to fall back to, so it is refused; pass
    ``skip_unresettable=True`` for bulk gestures that skip it instead.
    """
    if not allow_model_roles:
        _refuse_model_roles(keys)
    for key in keys:
        if not _is_settable(key):
            raise ValueError(f"Unknown or read-only setting: {key}")
        if key in _NO_RESET_FIELDS and not skip_unresettable:
            raise ValueError(f"'{key}' has no default to reset to; set a folder path instead.")
    targets = [key for key in keys if key not in _NO_RESET_FIELDS]
    fallbacks = _values_after_reset(targets)
    _refuse_invalid_fallbacks(fallbacks, _LayerChange.RESET)
    _validate({key: entry.value for key, entry in fallbacks.items()})
    embed_in_batch = "embedding_model" in targets
    if embed_in_batch:
        _pin_legacy_store_meta()
    persistent_settings.delete_values(cfg.data_root, targets)
    return _settle(set(targets), embed_in_batch=embed_in_batch)


def _values_after_reset(keys: list[str]) -> dict[str, Resolved]:
    """The value and source each of *keys* resolves to once its user entry is gone."""
    layers = read_layers(cfg.data_root)
    remaining = replace(
        layers, user={key: value for key, value in layers.user.items() if key not in keys}
    )
    return {key: resolve(key, remaining) for key in keys}


def _refuse_invalid_fallbacks(fallbacks: dict[str, Resolved], action: _LayerChange) -> None:
    """Refuse an *action* when a key would resolve to a value its field rejects."""
    trial = cfg.model_copy()
    for key, entry in fallbacks.items():
        try:
            setattr(trial, key, entry.value)
        except ValueError as exc:
            raise ValueError(
                f"Cannot {action.value} '{key}': its {entry.source.value} value {entry.value!r} is "
                "invalid. Fix or remove that value first."
            ) from exc


def apply_profile_layer(
    name: str,
    values: Mapping[str, Any],
    *,
    absorb: Collection[str] = (),
    write_first: Callable[[], object] | None = None,
) -> SettingsUpdateResult:
    """Record *name* and *values* as the project's profile and set cfg from the new layers.

    Every key the old or new profile holds is validated at the value it resolves to
    under the new profile before anything is written; user and env values keep winning.
    Each *absorb* key, which *values* must hold, leaves config.toml in the same write.
    Applying a profile hides the analyze tip.
    *write_first* runs once validation passes, before config.toml changes.
    """
    refused = sorted(set(values) - set(PROFILE_FIELDS))
    if refused:
        raise ValueError(f"Profiles cannot set {refused[0]}")
    stray = sorted(set(absorb) - set(values))
    if stray:
        raise ValueError(f"The profile does not hold {stray[0]}, so it cannot take it over")
    layers = read_layers(cfg.data_root)
    user = {key: value for key, value in layers.user.items() if key not in absorb}
    after = replace(layers, user=user, profile=dict(values))
    keys = set(layers.profile) | set(values)
    resolved = {key: resolve(key, after) for key in sorted(keys)}
    _refuse_invalid_fallbacks(resolved, _LayerChange.APPLY)
    _validate({key: entry.value for key, entry in resolved.items()})
    if write_first is not None:
        write_first()
    persistent_settings.write_profile_table(cfg.data_root, name, values, drop=absorb)
    _hide_analyze_tip()
    return _settle(keys, embed_in_batch=False)


def _hide_analyze_tip() -> None:
    """Hide the analyze tip once a profile is applied; a failed write must not fail the apply."""
    try:
        dismiss_tip(cfg.data_root)
    except OSError as exc:
        log.warning("Could not record that the analyze tip is hidden: %s", exc)
