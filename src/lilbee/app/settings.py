"""Canonical write boundary for lilbee configuration."""

from __future__ import annotations

import errno
import logging
import os
import threading
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from lilbee.app.settings_map import SETTINGS_MAP, SettingDef, SettingGroup
from lilbee.config_meta import (
    MODEL_ROLE_FIELDS,
    PUBLIC_CONFIG_FIELDS,
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, Config, cfg
from lilbee.core.config.defaults import SKIP_TOML_ENV
from lilbee.core.config.enums import OcrMode, SettingSource
from lilbee.core.config.keys import (
    LOAD_AFFECTING_KEYS,
    PROVIDER_SWITCHING_KEYS,
)
from lilbee.core.config.login_host import refuse_unless_host
from lilbee.core.config.model import (
    canonical_model_ref,
    checked_write,
    default_refusal,
    refusal_sentence,
    root_folders,
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
from lilbee.core.crawl_patterns import exclude_patterns_refusal
from lilbee.core.project_state import dismiss_tip
from lilbee.providers.roles import MODEL_FIELD_TO_ROLE, ROLE_GATE_FIELD_TO_ROLE
from lilbee.runtime.progress import OcrBackendUsed

if TYPE_CHECKING:
    from lilbee.modelhub.registry import ModelRegistry

log = logging.getLogger(__name__)

_MIN_CHUNK_SIZE = 64

# Keys that decide whether scanned pages are read, and by which engine.
OCR_SETTING_KEYS = frozenset({"ocr", "vision_model"})
SCANNED_PAGES_LABEL = "Scanned pages"
_SCANNED_PAGES_OFF = "skipped (ocr = off)"
_SCANNED_PAGES_READ = "read by {engine}"
_SCANNED_PAGES_ALL = "every page read by {engine} (ocr = all)"
_TESSERACT_ENGINE = "Tesseract ({languages})"
OCR_OFF_VISION_NOTICE = (
    "Scanned pages stay skipped because ocr is off. Set ocr to auto to read them with {model}."
)

# Path-typed writable fields whose pydantic "default" is the unresolved
# sentinel ``Path()`` (a literal "."). The actual default is computed by
# the model_validator at process start (data_root/documents, vault_base
# stays as None). Resetting these via the boundary would corrupt the
# install, so they are refused at the reset gate.
_NO_RESET_FIELDS: frozenset[str] = frozenset({"documents_dir"})

# Serializes a whole settings update (apply, save, rollback, settle) across
# threads. Held only inside _settings_transaction, so a plain lock (not
# reentrant) is enough.
_settings_write_lock = threading.Lock()


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
    advanced: bool


@dataclass(frozen=True)
class SettingsUpdateResult:
    """Outcome of an ``apply_settings_update`` call."""

    updated: list[str]
    reindex_required: bool
    warnings: tuple[str, ...] = ()


def scanned_pages_state() -> str:
    """What happens to scanned pages under the current ocr and vision_model settings."""
    backend = OcrBackendUsed.chosen(cfg.ocr, cfg.vision_model)
    if backend is OcrBackendUsed.NONE:
        return _SCANNED_PAGES_OFF
    engine = (
        cfg.vision_model
        if backend is OcrBackendUsed.VISION
        else _TESSERACT_ENGINE.format(languages="+".join(cfg.ocr_language))
    )
    template = _SCANNED_PAGES_ALL if cfg.ocr is OcrMode.ALL else _SCANNED_PAGES_READ
    return template.format(engine=engine)


def scanned_pages_line() -> str:
    """The one status line for scanned pages, shown on every surface."""
    return f"{SCANNED_PAGES_LABEL}: {scanned_pages_state()}"


def ocr_off_vision_notice() -> tuple[str, ...]:
    """The notice for a vision model that ocr = off keeps unused; empty otherwise."""
    if cfg.vision_model and cfg.ocr is OcrMode.OFF:
        return (OCR_OFF_VISION_NOTICE.format(model=cfg.vision_model),)
    return ()


def list_setting_value(raw: str) -> list[str]:
    """The items of a list setting typed as text: one per line, trimmed, blank lines dropped.

    The CLI and both Settings views use this parse. A comma stays in the item; a
    field validator that accepts commas splits it on write.
    """
    return [line.strip() for line in raw.split("\n") if line.strip()]


def _update_warnings(changed_keys: set[str]) -> tuple[str, ...]:
    """The notice for a vision model picked while ocr = off keeps it unused."""
    return ocr_off_vision_notice() if "vision_model" in changed_keys else ()


def _is_write_only(key: str) -> bool:
    """Return True for fields persisted but never read back (API keys, hf_token)."""
    extra = Config.model_fields[key].json_schema_extra
    if isinstance(extra, dict):
        return bool(extra.get("write_only", False))
    return False


def _secret_refusal(key: str, error: ValidationError) -> str:
    """What the field's validation says about a secret value, without the value itself."""
    reasons = "; ".join(detail["msg"] for detail in error.errors(include_input=False))
    return f"Setting '{key}' is refused: {reasons}"


def _public_writable_keys() -> list[str]:
    """Names of every writable config field minus write-only secrets and non-public fields."""
    keys = set(WRITABLE_CONFIG_FIELDS) | set(MODEL_ROLE_FIELDS)
    return sorted(k for k in keys if k in PUBLIC_CONFIG_FIELDS and not _is_write_only(k))


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
    return {key: _true_source(key, entry) for key, entry in resolve_all(layers).items()}


def _true_source(key: str, entry: Resolved) -> SettingSource:
    """The source of *entry*; a live value that differs from the built-in is not a built-in."""
    if entry.source is SettingSource.BUILT_IN and getattr(cfg, key) != entry.value:
        return SettingSource.AUTO
    return entry.source


def _reset_target(key: str, layers: SettingLayers) -> Any:
    """The value a reset falls back to without an env var: the profile's, else the built-in."""
    return layers.profile[key] if key in layers.profile else builtin_value(key)


def profile_defaults() -> dict[str, Any]:
    """The value the active profile sets for each setting it names.

    A reset falls back to it instead of the built-in, so it is the default every surface shows.
    """
    return dict(read_layers(cfg.data_root).profile)


def _setting_info(key: str, layers: SettingLayers, source: SettingSource) -> SettingInfo:
    definition = SETTINGS_MAP.get(key)
    nullable = _is_nullable(key)
    group = definition.group if definition else SettingGroup.MODELS
    help_text = _setting_help(key, definition)
    choices = definition.choices if definition else None
    advanced = definition.advanced if definition else False
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
        advanced=advanced,
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
        raise ValueError(f"Setting '{key}' is write-only and cannot be read back")
    layers = read_layers(cfg.data_root)
    return _setting_info(key, layers, _sources(layers)[key])


def _is_settable(key: str) -> bool:
    return key in WRITABLE_CONFIG_FIELDS or key in MODEL_ROLE_FIELDS


def _refuse_unsettable(key: str) -> None:
    """Raise ValueError for a key no writer accepts."""
    if not _is_settable(key):
        raise ValueError(f"Unknown or read-only setting: {key}")


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


_CRAWL_LOGIN_KEYS = {
    "crawl_authorization": "authorization",
    "crawl_username": "username",
    "crawl_password": "password",
    "crawl_cookie": "cookie",
}


def _validate_crawl_login(updates: dict[str, Any]) -> None:
    """Reject a stored crawl login that CrawlLogin refuses once *updates* apply."""
    from lilbee.crawler.models import CrawlLogin

    host = updates.get("crawl_login_host")
    if isinstance(host, str):
        refuse_unless_host(host)
    # A value that is not text is left to the field's own type check, as for any string setting.
    if not _CRAWL_LOGIN_KEYS.keys() & updates.keys() or not all(
        isinstance(updates[key], str) for key in _CRAWL_LOGIN_KEYS if key in updates
    ):
        return
    CrawlLogin(
        **{
            field: updates[key] if key in updates else getattr(cfg, key)
            for key, field in _CRAWL_LOGIN_KEYS.items()
        }
    )


def _validate(updates: dict[str, Any]) -> None:
    """Reject unknown keys, null on non-nullable, and out-of-range chunk sizes."""
    for key, value in updates.items():
        _refuse_unsettable(key)
        if value is None and not _is_nullable(key):
            raise ValueError(f"Setting '{key}' does not accept null")
    refusal = exclude_patterns_refusal(updates.get("crawl_exclude_patterns"))
    if refusal is not None:
        raise ValueError(refusal)
    _validate_crawl_login(updates)
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
    return checked_write(key, value)


def _apply_with_rollback(
    updates: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Set each key on cfg, or none of them on a refusal. Returns (persist, delete)."""
    snapshot = {k: getattr(cfg, k) for k in updates}
    to_persist: dict[str, Any] = {}
    to_delete: list[str] = []
    try:
        for key, raw in updates.items():
            if raw is None:
                setattr(cfg, key, None)
                to_delete.append(key)
                continue
            try:
                setattr(cfg, key, _coerce_value(key, raw))
            except ValidationError as exc:
                if _is_write_only(key):
                    raise ValueError(_secret_refusal(key, exc)) from None
                raise ValueError(refusal_sentence(key, raw)) from None
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
    return to_persist, to_delete


def _restore_snapshot(snapshot: dict[str, Any]) -> None:
    for key, value in snapshot.items():
        setattr(cfg, key, value)


def _reload_changed_roles(changed_keys: set[str]) -> None:
    """Re-plan the fleet off-thread, so only the servers whose launch changed restart.

    A model-role change (chat_model/embedding_model/reranker_model/vision_model) or
    a setting that gates a role (ocr) names its role. A role-agnostic load key
    (num_ctx, kv_cache_type, chunk_size) names none, and the same re-plan finds the
    roles it moves. The reload applies current cfg, so one write needs one reload.
    It runs off the caller's thread: the write never blocks on a slow respawn.
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
    if role_agnostic and not reloaded:
        services.provider.replan_fleet()


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
    if isinstance(exc, persistent_settings.ReadOnlyConfigError):
        # The error text already names the file by its full path.
        return f"{exc}."
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
    if "mcp_tool_threads" in changed_keys:
        # Resize the running server's thread pool now instead of only at startup.
        from lilbee.server.app import reapply_thread_pool_ceiling

        reapply_thread_pool_ceiling()
    if "include_uncensored" in changed_keys:
        # The picks memoize per process; drop them so the toggle takes
        # effect on the next read instead of the next restart.
        from lilbee.catalog.picks import reset_picks

        reset_picks()


def _run_deferred_provider_reset(needed: bool, *, after_failure: bool = False) -> None:
    """Tear down the Services singleton for a provider switch, once the lock has released.

    Blocks on the whole fleet stopping (each engine's stop-then-reap can
    take a few seconds), which is why ``_settings_transaction`` defers it
    until after ``_settings_write_lock`` releases.
    """
    if not needed:
        return
    from lilbee.app.services import reset_services

    if not after_failure:
        reset_services()
        return
    try:
        reset_services()
    except Exception:
        log.exception("Provider reset failed while handling another exception")


@dataclass
class _SettingsTransaction:
    """State one locked settings update carries to its exit-time provider reset."""

    needs_provider_reset: bool = False


@contextmanager
def _settings_transaction() -> Iterator[_SettingsTransaction]:
    """Hold ``_settings_write_lock`` for one settings update.

    The body sets ``needs_provider_reset`` on the yielded transaction before
    returning. The reset then runs after the lock releases: if the body
    raised, a reset failure is logged and swallowed so the body's exception
    reaches the caller unchanged; otherwise a reset failure propagates
    normally.
    """
    txn = _SettingsTransaction()
    try:
        with _settings_write_lock:
            yield txn
    except Exception:
        _run_deferred_provider_reset(txn.needs_provider_reset, after_failure=True)
        raise
    else:
        _run_deferred_provider_reset(txn.needs_provider_reset)


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

    Runs inside ``_settings_transaction``, which holds the lock and owns
    the provider-reset exception contract.
    """
    with _settings_transaction() as txn:
        if not allow_model_roles:
            _refuse_model_roles(updates)
        _validate(updates)
        null_keys = [key for key, value in updates.items() if value is None]
        if null_keys:
            _refuse_invalid_fallbacks(_values_after_reset(null_keys), _LayerChange.APPLY)
        embed_in_batch = "embedding_model" in updates
        if embed_in_batch:
            # Pin the OLD ref into store meta before mutation, otherwise the
            # next read lazy-initializes meta from the NEW cfg and silently
            # hides the dimension drift. Runs even when the value is unchanged
            # so a legacy meta row is always canonicalized on the first swap
            # attempt.
            _pin_legacy_store_meta()
        before = snapshot_settings(updates)
        to_persist, to_delete = _apply_with_rollback(updates)
        try:
            # A write can fail part way: the update lands and the delete raises.
            if to_persist:
                persistent_settings.update_values(cfg.data_root, to_persist)
            if to_delete:
                persistent_settings.delete_values(cfg.data_root, to_delete)
        except Exception:
            # No cache saw the new values yet, so none is dropped.
            _undo_update(before, invalidate_caches=False)
            raise
        # Decide the reset now, before settle's other steps can raise.
        txn.needs_provider_reset = bool(set(updates) & PROVIDER_SWITCHING_KEYS)
        try:
            file_unread_values = _values_when_file_unread(updates)
            return _settle(set(updates), embed_in_batch=embed_in_batch, values=file_unread_values)
        except Exception:
            # A store error at the verdict comes after the write, so config.toml must go back too.
            _undo_update(before, invalidate_caches=True)
            raise


def _values_when_file_unread(updates: dict[str, Any]) -> dict[str, Any] | None:
    """The values to settle on under ``LILBEE_SKIP_TOML_CONFIG=1``; None when the file is read.

    The resolver does not read config.toml under the switch, so it cannot see the value just
    written. cfg holds the update, except where a variable names the setting (it still wins)
    or the update clears it (the resolver gives the default).
    """
    if os.environ.get(SKIP_TOML_ENV) != "1":
        return None
    resolved = resolve_all(read_layers(cfg.data_root))
    return {
        key: resolved[key].value
        if raw is None or resolved[key].source is SettingSource.ENV
        else getattr(cfg, key)
        for key, raw in updates.items()
    }


def _undo_update(before: SettingsSnapshot, *, invalidate_caches: bool) -> None:
    """Put a failed update back; cfg goes back even when config.toml refuses the undo."""
    try:
        _put_back(before, invalidate_caches=invalidate_caches)
    except Exception:
        # The caller sees the update fail, so no field may keep its new value.
        log.exception("Could not put the settings back after a failed update")
        _restore_snapshot(before.values)
        cfg.embedding_dim = before.embedding_dim


@dataclass(frozen=True)
class SettingsSnapshot:
    """Settings as they stood: the values in cfg and the entries in config.toml."""

    values: dict[str, Any]
    saved: dict[str, Any]
    # Derived from the embedder, and not always from a header a restore can read again.
    embedding_dim: int


def snapshot_settings(keys: Collection[str]) -> SettingsSnapshot:
    """Record *keys* for ``restore_settings``. A key with no config.toml entry stays without one."""
    user = read_layers(cfg.data_root).user
    return SettingsSnapshot(
        values={key: getattr(cfg, key) for key in keys},
        saved={key: user[key] for key in keys if key in user},
        embedding_dim=cfg.embedding_dim,
    )


def restore_settings(snapshot: SettingsSnapshot) -> None:
    """Put cfg and config.toml back as *snapshot* holds them, so each source is as it was.

    Raises ``OSError`` or ``ValueError`` when config.toml refuses the write, with cfg
    left alone so that it still agrees with the file.
    """
    with _settings_transaction() as txn:
        txn.needs_provider_reset = bool(snapshot.values.keys() & PROVIDER_SWITCHING_KEYS)
        _put_back(snapshot)


def _put_back(snapshot: SettingsSnapshot, *, invalidate_caches: bool = True) -> None:
    """The body of ``restore_settings``. Caller holds ``_settings_write_lock``."""
    if snapshot.saved:
        persistent_settings.update_values(cfg.data_root, snapshot.saved)
    persistent_settings.delete_values(
        cfg.data_root, [key for key in snapshot.values if key not in snapshot.saved]
    )
    _restore_snapshot(snapshot.values)
    cfg.embedding_dim = snapshot.embedding_dim
    if invalidate_caches:
        _invalidate_caches(set(snapshot.values))


def _refuse_model_roles(keys: Iterable[str]) -> None:
    """Refuse model-role keys, which the dedicated model route owns."""
    rejected = MODEL_ROLE_FIELDS & set(keys)
    if rejected:
        offender = sorted(rejected)[0]
        raise ValueError(
            f"'{offender}' must be set through the dedicated model route "
            "(on the command line, lilbee model), not the general settings update."
        )


def _settle(
    keys: set[str],
    *,
    embed_in_batch: bool,
    changed: set[str] | None = None,
    reindex_changed: set[str] | None = None,
    values: Mapping[str, Any] | None = None,
) -> SettingsUpdateResult:
    """Set *keys* on cfg; rederive, invalidate and report.

    The values come from the resolver, or from *values* when the caller has read them.
    *changed* (default *keys*) is the keys that differ in cfg; *reindex_changed*
    (default *changed*) is the keys that differ in config.toml.
    """
    effective = keys if changed is None else changed
    reindex_effective = effective if reindex_changed is None else reindex_changed
    if values is None:
        persistent_settings.sync_from_resolver(cfg, keys)
    else:
        for key in keys:
            setattr(cfg, key, values[key])
    _rederive_from_resolved(effective)
    _invalidate_caches(effective)
    reindex_required = bool((REINDEX_FIELDS - _inert_reindex_keys()) & reindex_effective)
    if embed_in_batch:
        reindex_required = reindex_required or _embed_reindex_required()
    return SettingsUpdateResult(
        updated=sorted(keys),
        reindex_required=reindex_required,
        warnings=_update_warnings(effective),
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

    ``table_model`` is read only inside layout detection, and ``topic_threshold``
    only inside semantic chunking, so a change to either while its gating flag is
    off is not worth a rebuild.
    """
    inert: set[str] = set()
    if not cfg.layout_detection:
        inert.add("table_model")
    if not cfg.semantic_chunking:
        inert.add("topic_threshold")
    return inert


def _index_has_chunks() -> bool:
    """False only when a live store holds no chunk; with no services built, assume it does.

    A settings write must not build the services (and their engines) to answer this.
    """
    from lilbee.app.services import peek_services

    services = peek_services()
    return services is None or services.store.has_chunks()


def _embed_reindex_required() -> bool:
    """True when the persisted index was built with another embedder than cfg now names.

    Runs after the swap is applied, so the store compares its meta row against
    the new ref and the new model's width: the same verdict search refuses on.
    """
    from lilbee.app.services import get_services

    store = get_services().store
    store.canonicalize_meta_if_legacy()
    return store.index_mismatch() is not None


def embed_pick_needs_rebuild(ref: str) -> bool:
    """True when picking *ref* as the embedder leaves the index unsearchable.

    The verdict ``_embed_reindex_required`` reads after a swap, asked before it:
    the store's meta row against *ref* and its width, with cfg left alone.
    """
    from lilbee.app.services import get_services

    store = get_services().store
    store.initialize_meta_if_legacy()
    try:
        canonical = canonical_model_ref(_coerce_value("embedding_model", ref))
    except ValueError:  # the write refuses this ref on its own
        canonical = ref
    dim = _embedder_dim_from_gguf(canonical) or cfg.embedding_dim
    return store.index_mismatch(model=canonical, dim=dim) is not None


def reset_settings(
    keys: list[str], *, skip_unresettable: bool = False, allow_model_roles: bool = True
) -> SettingsUpdateResult:
    """Remove each key from config.toml and set cfg to the value the resolver then gives.

    ``documents_dir`` has no default to fall back to, so it is refused; pass
    ``skip_unresettable=True`` for bulk gestures that skip it instead.

    Runs inside ``_settings_transaction``, which holds the lock and owns
    the provider-reset exception contract.
    """
    with _settings_transaction() as txn:
        if not allow_model_roles:
            _refuse_model_roles(keys)
        for key in keys:
            _refuse_unsettable(key)
            if key in _NO_RESET_FIELDS and not skip_unresettable:
                raise ValueError(f"'{key}' has no default to reset to; set a folder path instead.")
        targets = [key for key in keys if key not in _NO_RESET_FIELDS]
        before = _values_before_reset(targets)
        fallbacks = _values_after_reset(targets)
        coerced = _refuse_invalid_fallbacks(fallbacks, _LayerChange.RESET)
        _validate({key: entry.value for key, entry in fallbacks.items()})
        # compared in cfg's own terms
        changed = {key for key in targets if getattr(cfg, key) != coerced[key]}
        # compared on the file's resolved value
        reindex_changed = {key for key in targets if before[key].value != fallbacks[key].value}
        if "embedding_model" in changed:
            _pin_legacy_store_meta()
        persistent_settings.delete_values(cfg.data_root, targets)
        # Decide the reset now, before settle's other steps can raise.
        txn.needs_provider_reset = bool(changed & PROVIDER_SWITCHING_KEYS)
        return _settle(
            set(targets),
            embed_in_batch="embedding_model" in reindex_changed,
            changed=changed,
            reindex_changed=reindex_changed,
        )


def _values_before_reset(keys: list[str]) -> dict[str, Resolved]:
    """The value and source each of *keys* currently resolves to, saved value included."""
    layers = read_layers(cfg.data_root)
    return {key: resolve(key, layers) for key in keys}


def _values_after_reset(keys: list[str]) -> dict[str, Resolved]:
    """The value and source each of *keys* resolves to once its user entry is gone."""
    layers = read_layers(cfg.data_root)
    remaining = replace(
        layers, user={key: value for key, value in layers.user.items() if key not in keys}
    )
    return {key: resolve(key, remaining) for key in keys}


def _refuse_invalid_fallbacks(
    fallbacks: dict[str, Resolved], action: _LayerChange
) -> dict[str, Any]:
    """Refuse an *action* when a key would resolve to a value its field rejects.

    Returns each key's value after the same coercion a cfg assignment applies
    (e.g. an env var's raw string cast to int), so a caller can compare it
    against cfg's current value in the same terms.
    """
    trial = cfg.model_copy()
    for key, entry in fallbacks.items():
        value = builtin_value(key) if default_refusal(key, entry.value) else entry.value
        try:
            setattr(trial, key, value)
        except ValueError as exc:
            raise ValueError(
                f"Cannot {action.value} '{key}': its {entry.source.value} value {entry.value!r} is "
                "invalid. Fix or remove that value first."
            ) from exc
    return {key: getattr(trial, key) for key in fallbacks}


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

    Runs inside ``_settings_transaction``, which holds the lock and owns
    the provider-reset exception contract.
    """
    with _settings_transaction() as txn:
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
        checked = _refuse_invalid_fallbacks(resolved, _LayerChange.APPLY)
        _validate({key: entry.value for key, entry in resolved.items()})
        if write_first is not None:
            write_first()
        persistent_settings.write_profile_table(cfg.data_root, name, values, drop=absorb)
        _hide_analyze_tip()
        # Decide the reset now, before settle's other steps can raise.
        txn.needs_provider_reset = bool(keys & PROVIDER_SWITCHING_KEYS)
        # Under LILBEE_SKIP_TOML_CONFIG=1 the resolver cannot read the profile just written,
        # so settle on the checked values resolved from the layers built above.
        file_unread = os.environ.get(SKIP_TOML_ENV) == "1"
        settled = checked if file_unread else None
        return replace(
            _settle(keys, embed_in_batch=False, values=settled),
            warnings=ocr_off_vision_notice(),
        )


@dataclass(frozen=True)
class ConfigReading:
    """What config.toml set when this process last read it.

    ``values`` is what the startup overlay sets from the file, by key. ``held`` is the
    value a held key takes when the file stops setting it, as at startup: a folder
    derived from the data root, or the model cfg held when the TUI started.
    """

    stamp: tuple[int, int, int] | None
    values: Mapping[str, Any]
    held: Mapping[str, Any]


# The source registry has its own reader, which a running sync's hold on it stops.
_RELOADED_KEYS = (set(WRITABLE_CONFIG_FIELDS) | MODEL_ROLE_FIELDS) - {"linked_roots"}
_ABSENT = object()


def read_config_state() -> ConfigReading:
    """Read config.toml under cfg's data root the way startup does, in one locked read."""
    stamp, values = persistent_settings.read_persisted(cfg.data_root)
    folders = root_folders(cfg.data_root)
    held = {
        key: folders[key] if key in folders else getattr(cfg, key)
        for key in persistent_settings.HELD_AT_BUILT_IN & _RELOADED_KEYS
    }
    return ConfigReading(stamp, values or {}, held)


def _registry_refreshed() -> bool:
    """Read the source registry again unless a running sync holds it; whether it changed."""
    before = cfg.linked_roots
    persistent_settings.refresh_registry(cfg)
    return cfg.linked_roots != before


def adopt_external_changes(
    last: ConfigReading,
) -> tuple[ConfigReading, SettingsUpdateResult | None]:
    """Set on cfg each key another process changed in config.toml since *last*.

    Costs two stat() calls when the file is unchanged. The new values come from the
    same function startup uses, from one read under the lock a writer takes. A key is
    taken only when its value moved since *last* and cfg does not hold it yet, so a
    value this process set for itself alone (a one-off command-line flag) stays unless
    the file changed that key. A held key the file stops setting goes back to its
    startup value. The source registry is read by ``refresh_registry``: while a sync
    holds it, it keeps its folders, and a later check takes the change. The keys go
    through the same settle step as an update made here. A file that does not parse
    changes nothing and keeps *last*, as at startup.
    Returns the new reading and the update result, or None when no key changed.
    """
    roots_changed = _registry_refreshed()
    if persistent_settings.overlay_stamp(cfg.data_root) == last.stamp and not roots_changed:
        return last, None
    with _settings_transaction() as txn:
        stamp, values = persistent_settings.read_persisted(cfg.data_root)
        now = last if values is None else replace(last, stamp=stamp, values=values)
        targets: dict[str, Any] = {}
        for key in _RELOADED_KEYS:
            value = now.values.get(key, _ABSENT)
            if value == last.values.get(key, _ABSENT):
                continue
            if value is _ABSENT:
                value = last.held.get(key, _ABSENT)
            if value is not _ABSENT and value != getattr(cfg, key):
                targets[key] = value
        if not targets and not roots_changed:
            return now, None
        txn.needs_provider_reset = bool(targets.keys() & PROVIDER_SWITCHING_KEYS)
        result = _settle(set(targets), embed_in_batch=False, values=targets)
        if roots_changed:
            result = replace(result, updated=sorted({*result.updated, "linked_roots"}))
        return now, result


def _hide_analyze_tip() -> None:
    """Hide the analyze tip once a profile is applied; a failed write must not fail the apply."""
    try:
        dismiss_tip(cfg.data_root)
    except OSError as exc:
        log.warning("Could not record that the analyze tip is hidden: %s", exc)
