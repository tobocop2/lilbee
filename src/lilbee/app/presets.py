"""Surface-agnostic use-cases for user-defined OpenAI-compatible server presets."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from lilbee.app.settings import config_write_failure_message
from lilbee.core import settings
from lilbee.core.blocked_networks import check_url_form, url_origin, validate_preset_url
from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import (
    PRESETS_KEY,
    RESERVED_NAMES,
    Entry,
    RerankWire,
    StoredPreset,
    UnreadableEntry,
    checked_key,
    decode_entries,
    stored_entries,
)

log = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_UNREADABLE_ENTRY = "The stored entry is not a valid preset. Fix or remove it in config.toml."
_KEY_STAYS_WITH_ORIGIN = (
    "The new URL is a different server, and the stored key does not follow it. "
    "Give the key again, or clear the key first."
)


_NOTHING_TO_CHANGE = "Nothing to change. Give a URL, roles, a key, a rerank wire or clear the key."
_CLEAR_KEY_ALONE = "Clear the key in its own edit. Do not give other changes with it."

_ListChange = Callable[[list[Entry]], list[Entry]]


class PresetNotFoundError(ValueError):
    """No stored preset has the requested name."""


class PresetExistsError(ValueError):
    """A stored preset already has the requested name."""


@dataclass(frozen=True)
class ServerPreset:
    """One preset as a surface shows it: it says whether a key is set and never holds one.

    An entry no preset may hold has valid False, an empty base_url and no rerank_wire.
    """

    name: str
    base_url: str
    roles: tuple[WorkerRole, ...]
    has_key: bool
    valid: bool
    rerank_wire: RerankWire | None


def _check_name(name: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "A preset name uses lower-case letters, digits, - and _, "
            "and starts with a letter or a digit."
        )
    if name in RESERVED_NAMES:
        raise ValueError(f"The preset name {name!r} is a provider prefix. Choose another name.")


def _checked_roles(roles: Collection[str]) -> tuple[WorkerRole, ...]:
    """*roles* as role values, in their order; ValueError for none, an unknown or a repeat."""
    known = {role.value: role for role in WorkerRole}
    if not roles:
        raise ValueError(f"A preset needs at least one role: {', '.join(known)}.")
    unknown = [role for role in roles if role not in known]
    if unknown:
        raise ValueError(f"Unknown role {unknown[0]!r}. Valid roles: {', '.join(known)}.")
    if len(set(roles)) != len(roles):
        raise ValueError("A preset names each role once.")
    return tuple(known[role] for role in roles)


def _checked_wire(wire: str) -> RerankWire:
    """*wire* as a rerank wire; ValueError that names the valid wires."""
    try:
        return RerankWire(wire)
    except ValueError:
        valid = ", ".join(RerankWire)
        raise ValueError(f"Unknown rerank wire {wire!r}. Valid wires: {valid}.") from None


def _encoded(entries: list[Entry]) -> str | None:
    """The stored value for *entries*; None when there are none, so the key leaves the file."""
    if not entries:
        return None
    rows = []
    for entry in entries:
        if isinstance(entry, UnreadableEntry):
            rows.append(entry.raw)
            continue
        row: dict[str, Any] = {
            "name": entry.name,
            "url": entry.url,
            "roles": [role.value for role in entry.roles],
        }
        if entry.api_key is not None:
            row["api_key"] = entry.api_key.get_secret_value()
        if entry.rerank_wire is not RerankWire.V1:
            row["rerank_wire"] = entry.rerank_wire.value
        rows.append(row)
    return json.dumps(rows)


def _checked_url(entry: Entry) -> str:
    """The URL of *entry*; ValueError when no preset may hold it. No stored name is looked up."""
    if isinstance(entry, UnreadableEntry):
        raise ValueError(_UNREADABLE_ENTRY)
    _check_name(entry.name)
    check_url_form(entry.url)
    return entry.url


def _check_list(entries: list[Entry]) -> None:
    """Refuse a list to write that holds a repeated name or an entry no preset may hold.

    A write holds the config lock, and the request check refuses a name whose DNS answer
    turned bad.
    """
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("Two stored server presets share a name. Remove one of them.")
    for entry in entries:
        try:
            _checked_url(entry)
        except ValueError as exc:
            raise ValueError(f"The stored preset {entry.name!r} is refused: {exc}") from None


def _view(entry: Entry) -> ServerPreset:
    """What a surface shows of *entry*; the URL of a refused entry can hold a credential."""
    try:
        url = _checked_url(entry)
    except ValueError:
        log.warning("The stored server preset %r is not valid and cannot be used.", entry.name)
        url = ""
    wire = entry.rerank_wire if isinstance(entry, StoredPreset) and url else None
    return ServerPreset(entry.name or "", url, entry.roles, entry.has_key, bool(url), wire)


def _store(change: _ListChange) -> list[Entry]:
    """Apply *change* to the stored list under the config lock; return the list it made."""

    def apply(raw: Any) -> tuple[str | None, list[Entry]]:
        changed = change(decode_entries(raw))
        encoded = _encoded(changed)
        cfg.server_presets = encoded
        return encoded, changed

    return settings.mutate_value(cfg.data_root, PRESETS_KEY, apply)


def _named(entries: list[Entry], name: str) -> Entry:
    for entry in entries:
        if entry.name == name:
            return entry
    raise PresetNotFoundError(f"No server preset is named {name!r}.")


def _without_key(entry: Entry) -> Entry:
    if isinstance(entry, UnreadableEntry):
        return UnreadableEntry({k: v for k, v in entry.raw.items() if k != "api_key"})
    return entry.model_copy(update={"api_key": None})


def list_presets() -> list[ServerPreset]:
    """Every stored entry; a stored value that is no list of entries is refused."""
    return [_view(entry) for entry in stored_entries()]


def add_preset(
    name: str,
    url: str,
    *,
    roles: Collection[str],
    api_key: str | None = None,
    rerank_wire: str = RerankWire.V1,
) -> ServerPreset:
    """Store a new preset. An empty key is no key."""
    _check_name(name)
    checked_roles = _checked_roles(roles)
    validate_preset_url(url)
    entry = StoredPreset(
        name=name,
        url=url,
        roles=checked_roles,
        api_key=checked_key(api_key) if api_key else None,
        rerank_wire=_checked_wire(rerank_wire),
    )

    def change(entries: list[Entry]) -> list[Entry]:
        if any(existing.name == name for existing in entries):
            raise PresetExistsError(f"A server preset named {name!r} already exists.")
        changed = [*entries, entry]
        _check_list(changed)
        return changed

    _store(change)
    return _view(entry)


def _check_edit_shape(given: int, clear_key: bool) -> None:
    """Refuse an edit with no change, or a key clear that comes with *given* other changes."""
    if clear_key and given:
        raise ValueError(_CLEAR_KEY_ALONE)
    if not (clear_key or given):
        raise ValueError(_NOTHING_TO_CHANGE)


def edit_preset(
    name: str,
    *,
    url: str | None = None,
    roles: Collection[str] | None = None,
    api_key: str | None = None,
    rerank_wire: str | None = None,
    clear_key: bool = False,
) -> ServerPreset:
    """Change what is passed. An empty or omitted key keeps the stored key.

    With *clear_key* the key is removed and nothing else may change in the same call. A call
    that names no change is refused.

    A URL with a new scheme, host or port and no new key is refused: the stored key never
    follows the URL to a new origin. A key change alone checks only that preset's entry.
    """
    given = sum(value is not None for value in (url, roles, rerank_wire)) + bool(api_key)
    _check_edit_shape(given, clear_key)
    if clear_key:
        return _clear_key(name)
    changes: dict[str, Any] = {}
    if url is not None:
        validate_preset_url(url)
        changes["url"] = url
    if roles is not None:
        changes["roles"] = _checked_roles(roles)
    if api_key:
        changes["api_key"] = checked_key(api_key)
    if rerank_wire is not None:
        changes["rerank_wire"] = _checked_wire(rerank_wire)

    def change(entries: list[Entry]) -> list[Entry]:
        target = _named(entries, name)
        if isinstance(target, UnreadableEntry):
            raise ValueError(f"The stored preset {name!r} is refused: {_UNREADABLE_ENTRY}")
        moves_key = target.has_key and not api_key
        if url is not None and moves_key and url_origin(target.url) != url_origin(url):
            raise ValueError(_KEY_STAYS_WITH_ORIGIN)
        updated = target.model_copy(update=changes)
        changed = [updated if entry is target else entry for entry in entries]
        key_change_only = list(changes) == ["api_key"]
        _check_list([e for e in changed if e.name == name] if key_change_only else changed)
        return changed

    return _view(_named(_store(change), name))


def _clear_key(name: str) -> ServerPreset:
    """Remove the stored key of the preset *name*. A clear adds nothing, so no entry can stop it."""

    def change(entries: list[Entry]) -> list[Entry]:
        _named(entries, name)
        return [_without_key(entry) if entry.name == name else entry for entry in entries]

    return _view(_named(_store(change), name))


def remove_preset(name: str) -> None:
    """Remove the preset *name*. A removal adds nothing, so no other stored entry can stop it."""

    def change(entries: list[Entry]) -> list[Entry]:
        _named(entries, name)
        return [entry for entry in entries if entry.name != name]

    _store(change)


def split_roles(text: str) -> list[str]:
    """The role names in a comma-separated *text*; an empty segment stays for the role check."""
    if not text.strip():
        return []
    return [part.strip() for part in text.split(",")]


def engines_refused_message() -> str:
    """The refusal text for a preset change on the HTTP server while allow_http_engines is off."""
    return (
        "Changing presets on the HTTP server is unavailable: a preset sets where requests and "
        "API keys go. Enable allow_http_engines (LILBEE_ALLOW_HTTP_ENGINES) on a deployment you "
        "alone use, or change presets from the CLI or TUI."
    )


def failure_text(exc: ValueError | OSError) -> str:
    """The text a surface shows for a failed preset call; no surface writes its own."""
    if isinstance(exc, OSError):
        return config_write_failure_message(exc)
    return str(exc)
