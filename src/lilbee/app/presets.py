"""Surface-agnostic use-cases for user-defined OpenAI-compatible server presets."""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from lilbee.core import settings
from lilbee.core.blocked_networks import ALWAYS_BLOCKED_NETWORKS, IpAddress, blocked_form
from lilbee.core.config import cfg
from lilbee.providers.model_ref import LOCAL_PROVIDER, PROVIDER_PREFIXES
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

_PRESETS_KEY = "server_presets"
_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_RESERVED_NAMES = PROVIDER_PREFIXES | {LOCAL_PROVIDER}
_ALLOWED_SCHEMES = ("http", "https")
# What a base URL may hold: printable ASCII, less a space and a backslash. A URL standard
# parser reads these differently from urlsplit, or folds them into an address.
_URL_PATTERN = re.compile(r"[!-\[\]-~]+")
_IPV4_NUMBER = re.compile(r"0[xX][0-9a-fA-F]*|[0-9]+")
_IPV4_MAX_PARTS = 4
_IPV4_PART_BASE = 256
_NOT_AN_IPV4 = "The URL host is not a valid IPv4 address."
_NOT_AN_IPV6 = "The URL host is not a valid IPv6 address."
_UNPARSED_STORE = (
    "The stored server presets do not parse. Fix or remove server_presets in config.toml."
)


class _StoredPreset(BaseModel):
    """One preset as config.toml holds it; the key is a secret so no repr shows it."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    name: str
    url: str
    roles: Annotated[tuple[WorkerRole, ...], Field(min_length=1)]
    api_key: SecretStr | None = None


_STORED_LIST = TypeAdapter(list[_StoredPreset])
_ListChange = Callable[[list[_StoredPreset]], list[_StoredPreset]]


@dataclass(frozen=True)
class ServerPreset:
    """One preset as a surface shows it: it says whether a key is set and never holds one."""

    name: str
    base_url: str
    roles: tuple[WorkerRole, ...]
    has_key: bool


def _view(entry: _StoredPreset) -> ServerPreset:
    return ServerPreset(entry.name, entry.url, entry.roles, bool(entry.api_key))


def _ipv4_part(part: str) -> int:
    """The value of one part of a numeric host by the URL standard."""
    if part[:2].lower() == "0x":
        return int(part[2:] or "0", 16)
    return int(part, 8 if len(part) > 1 and part.startswith("0") else 10)


def _numeric_host(host: str) -> ipaddress.IPv4Address | None:
    """The IPv4 a host that ends in a number names by the URL standard; None for a host name.

    A connection library reads ``2852039166``, ``0xa9.254.0xa9fe`` and ``0251.0376.0251.0376``
    as addresses without asking DNS. ValueError when the host ends in a number and names none.
    """
    parts = host.removesuffix(".").split(".")
    *_, final_part = parts
    if not _IPV4_NUMBER.fullmatch(final_part):
        return None
    if len(parts) > _IPV4_MAX_PARTS or not all(_IPV4_NUMBER.fullmatch(part) for part in parts):
        raise ValueError(_NOT_AN_IPV4)
    try:
        *leading, last = [_ipv4_part(part) for part in parts]
    except ValueError:  # an octal part with an 8 or a 9
        raise ValueError(_NOT_AN_IPV4) from None
    # The last part fills every byte the parts before it leave.
    room = _IPV4_PART_BASE ** (_IPV4_MAX_PARTS - len(leading))
    if last >= room or any(part >= _IPV4_PART_BASE for part in leading):
        raise ValueError(_NOT_AN_IPV4)
    value = 0
    for part in leading:
        value = value * _IPV4_PART_BASE + part
    return ipaddress.IPv4Address(value * room + last)


def _url_host(url: str) -> str:
    """The host of a well-formed base URL; ValueError that repeats no part of *url*."""
    if not _URL_PATTERN.fullmatch(url):
        raise ValueError(
            "A server URL needs printable ASCII with no space or backslash. "
            "Write an international host name in its xn-- form."
        )
    if "?" in url or "#" in url:
        raise ValueError("A server URL cannot hold a query or a fragment.")
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018  reading it is the check: it refuses a bad port
    except ValueError:
        raise ValueError("The server URL does not parse, or its port is not valid.") from None
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError("A server URL must start with http:// or https://.")
    if "@" in parts.netloc:
        raise ValueError("A server URL cannot hold a user name or password. Use the API key.")
    if "%" in parts.netloc:
        raise ValueError("A server URL cannot hold a percent sign or a zone id in its host.")
    if not parts.hostname:
        raise ValueError("A server URL needs a host.")
    if parts.netloc.startswith("[") and ":" not in parts.hostname:
        raise ValueError(_NOT_AN_IPV6)  # urlsplit admits a bracketed [v1.abc]; a connection cannot
    return parts.hostname


def _literal_address(host: str) -> IpAddress | None:
    """The address *host* names without DNS, or None for a host name."""
    if ":" in host:
        try:
            return ipaddress.IPv6Address(host)
        except ValueError:
            raise ValueError(_NOT_AN_IPV6) from None
    return _numeric_host(host)


def _refuse_blocked(address: IpAddress) -> None:
    """Refuse *address* when it, or the IPv4 it embeds, lies in an always-blocked network."""
    blocked = blocked_form(address, ALWAYS_BLOCKED_NETWORKS)
    if blocked is not None:
        raise ValueError(f"A server preset cannot use the blocked address {blocked}.")


def _resolved_addresses(host: str) -> list[IpAddress]:
    """Every address DNS gives for *host*; empty when it gives none."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    except UnicodeError:
        raise ValueError("The URL host is not a valid host name.") from None
    return [ipaddress.ip_address(sockaddr[0]) for *_, sockaddr in infos]


def _check_form(url: str) -> str | None:
    """Check all of *url* that needs no DNS; return its host name, or None for an address."""
    host = _url_host(url)
    literal = _literal_address(host)
    if literal is None:
        return host
    _refuse_blocked(literal)
    return None


def validate_preset_url(url: str) -> None:
    """Refuse a URL no preset may save; a host with no DNS answer is accepted."""
    host = _check_form(url)
    if host is not None:
        for address in _resolved_addresses(host):
            _refuse_blocked(address)


def check_request_address(url: str) -> None:
    """Refuse a request unless every address of the URL's host is allowed at this moment."""
    host = _check_form(url)
    if host is None:
        return
    addresses = _resolved_addresses(host)
    if not addresses:
        raise ValueError("The server host has no DNS answer.")
    for address in addresses:
        _refuse_blocked(address)


def _check_name(name: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "A preset name uses lower-case letters, digits, - and _, "
            "and starts with a letter or a digit."
        )
    if name in _RESERVED_NAMES:
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


def _decoded(raw: Any) -> list[_StoredPreset]:
    """The presets of a stored value; a refusal and a log line that repeat none of it."""
    if raw is None:
        return []
    entries: list[_StoredPreset] | None = None
    if isinstance(raw, str):  # a hand-edited config.toml can hold any TOML type here
        try:
            entries = _STORED_LIST.validate_json(raw)
        except ValidationError:
            entries = None
    if entries is None:
        log.warning(_UNPARSED_STORE)
        raise ValueError(_UNPARSED_STORE)
    return entries


def _encoded(entries: list[_StoredPreset]) -> str | None:
    """The stored value for *entries*; None when there are none, so the key leaves the file."""
    if not entries:
        return None
    rows = []
    for entry in entries:
        row: dict[str, Any] = {
            "name": entry.name,
            "url": entry.url,
            "roles": [role.value for role in entry.roles],
        }
        if entry.api_key is not None:
            row["api_key"] = entry.api_key.get_secret_value()
        rows.append(row)
    return json.dumps(rows)


def _check_list(entries: list[_StoredPreset]) -> None:
    """Refuse a list to write that holds a repeated name or an entry no preset may hold.

    A stored host name is not looked up: a write holds the config lock, and the request
    check refuses a name whose DNS answer turned bad.
    """
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("Two stored server presets share a name. Remove one of them.")
    for entry in entries:
        try:
            _check_name(entry.name)
            _check_form(entry.url)
        except ValueError as exc:
            raise ValueError(f"The stored preset {entry.name!r} is refused: {exc}") from None


def _store(change: _ListChange) -> list[_StoredPreset]:
    """Apply *change* to the stored list under the config lock; return the list it made."""

    def apply(raw: Any) -> tuple[str | None, list[_StoredPreset]]:
        changed = change(_decoded(raw))
        encoded = _encoded(changed)
        cfg.server_presets = encoded
        return encoded, changed

    return settings.mutate_value(cfg.data_root, _PRESETS_KEY, apply)


def _write(change: _ListChange) -> list[_StoredPreset]:
    """Store what *change* makes when all of it passes the check for a list to write."""

    def checked(entries: list[_StoredPreset]) -> list[_StoredPreset]:
        changed = change(entries)
        _check_list(changed)
        return changed

    return _store(checked)


def _named(entries: list[_StoredPreset], name: str) -> _StoredPreset:
    for entry in entries:
        if entry.name == name:
            return entry
    raise ValueError(f"No server preset is named {name!r}.")


def _replace(name: str, **changes: Any) -> ServerPreset:
    """Change the fields *changes* names on the stored preset *name*."""

    def change(entries: list[_StoredPreset]) -> list[_StoredPreset]:
        target = _named(entries, name)
        updated = target.model_copy(update=changes)
        return [updated if entry is target else entry for entry in entries]

    return _view(_named(_write(change), name))


def list_presets() -> list[ServerPreset]:
    """Every stored preset; a stored value that does not parse is refused."""
    return [_view(entry) for entry in _decoded(settings.load(cfg.data_root).get(_PRESETS_KEY))]


def add_preset(
    name: str, url: str, *, roles: Collection[str], api_key: str | None = None
) -> ServerPreset:
    """Store a new preset. An empty key is no key."""
    _check_name(name)
    checked_roles = _checked_roles(roles)
    validate_preset_url(url)
    entry = _StoredPreset(
        name=name,
        url=url,
        roles=checked_roles,
        api_key=SecretStr(api_key) if api_key else None,
    )

    def change(entries: list[_StoredPreset]) -> list[_StoredPreset]:
        if any(existing.name == name for existing in entries):
            raise ValueError(f"A server preset named {name!r} already exists.")
        return [*entries, entry]

    _write(change)
    return _view(entry)


def edit_preset(
    name: str,
    *,
    url: str | None = None,
    roles: Collection[str] | None = None,
    api_key: str | None = None,
) -> ServerPreset:
    """Change what is passed. An empty or omitted key keeps the stored key."""
    changes: dict[str, Any] = {}
    if url is not None:
        validate_preset_url(url)
        changes["url"] = url
    if roles is not None:
        changes["roles"] = _checked_roles(roles)
    if api_key:
        changes["api_key"] = SecretStr(api_key)
    return _replace(name, **changes)


def clear_key(name: str) -> ServerPreset:
    """Remove the stored key of the preset *name*."""
    return _replace(name, api_key=None)


def remove_preset(name: str) -> None:
    """Remove the preset *name*. A removal adds nothing, so no other stored entry can stop it."""

    def change(entries: list[_StoredPreset]) -> list[_StoredPreset]:
        target = _named(entries, name)
        return [entry for entry in entries if entry.name != target.name]

    _store(change)
