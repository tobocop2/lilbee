"""Surface-agnostic use-cases for user-defined OpenAI-compatible server presets."""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from lilbee.core import settings
from lilbee.core.blocked_networks import ALWAYS_BLOCKED_NETWORKS, IpAddress, blocked_form
from lilbee.core.config import cfg
from lilbee.providers.model_ref import LOCAL_PROVIDER, PROVIDER_PREFIXES
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

_PRESETS_KEY = "server_presets"
_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_RESERVED_NAMES = PROVIDER_PREFIXES | {LOCAL_PROVIDER}
_DEFAULT_PORTS = {"http": 80, "https": 443}
# What a base URL may hold: printable ASCII, less a space and a backslash. A URL standard
# parser reads these differently from urlsplit, or folds them into an address.
_URL_PATTERN = re.compile(r"[!-\[\]-~]+")
_IPV4_NUMBER = re.compile(r"0[xX][0-9a-fA-F]*|[0-9]+")
_IPV4_MAX_PARTS = 4
_IPV4_PART_BASE = 256
_LOCAL_NAT64_PREFIX = ipaddress.ip_network("64:ff9b:1::/48")  # RFC 8215
_ISATAP_INTERFACE_PREFIXES = (0x00005EFE, 0x02005EFE)  # RFC 5214: bits 32 to 63 of the address
_NOT_AN_IPV4 = "The URL host is not a valid IPv4 address."
_NOT_AN_IPV6 = "The URL host is not a valid IPv6 address."
_UNPARSED_STORE = (
    "The stored server presets do not parse. Fix or remove server_presets in config.toml."
)
_UNREADABLE_ENTRY = "The stored entry is not a valid preset. Fix or remove it in config.toml."
_KEY_STAYS_WITH_ORIGIN = (
    "The new URL is a different server, and the stored key does not follow it. "
    "Give the key again, or clear the key first."
)


def _checked_key(api_key: str) -> SecretStr:
    """*api_key* as a secret; ValueError that does not repeat it when it cannot be stored."""
    # A compare and no caught UnicodeEncodeError: its object would keep the key on the chain.
    if api_key.encode("utf-8", "replace").decode("utf-8") != api_key:
        raise ValueError("The API key holds a character that cannot be stored.")
    return SecretStr(api_key)


class _StoredPreset(BaseModel):
    """One preset as config.toml holds it; the key and the URL credentials stay out of its repr."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    name: str
    url: str = Field(repr=False)
    roles: Annotated[tuple[WorkerRole, ...], Field(min_length=1)]
    api_key: SecretStr | None = None

    @field_validator("api_key")
    @classmethod
    def _key_can_be_stored(cls, value: SecretStr | None) -> SecretStr | None:
        return value and _checked_key(value.get_secret_value())

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class _UnreadableEntry:
    """A stored row that is no valid preset; a write puts it back as it came."""

    raw: Any = field(repr=False)
    roles: tuple[WorkerRole, ...] = ()

    @property
    def name(self) -> str | None:
        name = self.raw.get("name") if isinstance(self.raw, dict) else None
        return name if isinstance(name, str) else None

    @property
    def has_key(self) -> bool:
        return isinstance(self.raw, dict) and bool(self.raw.get("api_key"))


_Entry = _StoredPreset | _UnreadableEntry
_ListChange = Callable[[list[_Entry]], list[_Entry]]


@dataclass(frozen=True)
class ServerPreset:
    """One preset as a surface shows it: it says whether a key is set and never holds one.

    An entry no preset may hold has valid False and an empty base_url.
    """

    name: str
    base_url: str
    roles: tuple[WorkerRole, ...]
    has_key: bool
    valid: bool


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
        if parts.port == 0:
            raise ValueError
    except ValueError:
        raise ValueError("The server URL does not parse, or its port is not valid.") from None
    if parts.scheme.lower() not in _DEFAULT_PORTS:
        raise ValueError("A server URL must use the http or https scheme.")
    if "@" in parts.netloc:
        raise ValueError("A server URL cannot hold a user name or password. Use the API key.")
    if "%" in parts.netloc:
        raise ValueError("A server URL cannot hold a percent sign or a zone id in its host.")
    if not parts.hostname:
        raise ValueError("A server URL needs a host.")
    if parts.netloc.startswith("[") and ":" not in parts.hostname:
        raise ValueError(_NOT_AN_IPV6)  # urlsplit admits a bracketed [v1.abc]; a connection cannot
    return parts.hostname


def _origin(url: str) -> tuple[str, str, int] | None:
    """The scheme, host and port that *url* connects to; None when *url* is not well formed."""
    try:
        host = _url_host(url)
    except ValueError:
        return None
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, host, _DEFAULT_PORTS[scheme] if parts.port is None else parts.port


def _literal_address(host: str) -> IpAddress | None:
    """The address *host* names without DNS, or None for a host name."""
    if ":" in host:
        try:
            return ipaddress.IPv6Address(host)
        except ValueError:
            raise ValueError(_NOT_AN_IPV6) from None
    return _numeric_host(host)


def _tunnels_to_ipv4(address: IpAddress) -> bool:
    """True for a local-use NAT64, Teredo or ISATAP address: each can carry a blocked IPv4."""
    if not isinstance(address, ipaddress.IPv6Address):
        return False
    return (
        address in _LOCAL_NAT64_PREFIX
        or address.teredo is not None
        or (int(address) >> 32) & 0xFFFFFFFF in _ISATAP_INTERFACE_PREFIXES
    )


def _refuse_blocked(address: IpAddress) -> None:
    """Refuse *address* when it, or the IPv4 it embeds, lies in an always-blocked network."""
    blocked = blocked_form(address, ALWAYS_BLOCKED_NETWORKS)
    if blocked is None and _tunnels_to_ipv4(address):
        blocked = address
    if blocked is not None:
        raise ValueError(f"A server preset cannot use the blocked address {blocked}.")


def _resolved_addresses(host: str) -> list[IpAddress]:
    """Every address DNS gives for *host*; empty when it gives none."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
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


def _decoded_row(row: Any) -> _Entry:
    try:
        return _StoredPreset.model_validate(row)
    except ValidationError:
        return _UnreadableEntry(row)


def _decoded(raw: Any) -> list[_Entry]:
    """The entries of a stored value; a refusal and a log line that repeat none of it."""
    if raw is None:
        return []
    rows: Any = None
    if isinstance(raw, str):  # a hand-edited config.toml can hold any TOML type here
        try:
            rows = json.loads(raw)
        except (ValueError, RecursionError):
            rows = None
    if not isinstance(rows, list):
        log.warning(_UNPARSED_STORE)
        raise ValueError(_UNPARSED_STORE)
    return [_decoded_row(row) for row in rows]


def _encoded(entries: list[_Entry]) -> str | None:
    """The stored value for *entries*; None when there are none, so the key leaves the file."""
    if not entries:
        return None
    rows = []
    for entry in entries:
        if isinstance(entry, _UnreadableEntry):
            rows.append(entry.raw)
            continue
        row: dict[str, Any] = {
            "name": entry.name,
            "url": entry.url,
            "roles": [role.value for role in entry.roles],
        }
        if entry.api_key is not None:
            row["api_key"] = entry.api_key.get_secret_value()
        rows.append(row)
    return json.dumps(rows)


def _checked_url(entry: _Entry) -> str:
    """The URL of *entry*; ValueError when no preset may hold it. No stored name is looked up."""
    if isinstance(entry, _UnreadableEntry):
        raise ValueError(_UNREADABLE_ENTRY)
    _check_name(entry.name)
    _check_form(entry.url)
    return entry.url


def _check_list(entries: list[_Entry]) -> None:
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


def _view(entry: _Entry) -> ServerPreset:
    """What a surface shows of *entry*; the URL of a refused entry can hold a credential."""
    try:
        url = _checked_url(entry)
    except ValueError:
        log.warning("The stored server preset %r is not valid and cannot be used.", entry.name)
        url = ""
    return ServerPreset(entry.name or "", url, entry.roles, entry.has_key, bool(url))


def _store(change: _ListChange) -> list[_Entry]:
    """Apply *change* to the stored list under the config lock; return the list it made."""

    def apply(raw: Any) -> tuple[str | None, list[_Entry]]:
        changed = change(_decoded(raw))
        encoded = _encoded(changed)
        cfg.server_presets = encoded
        return encoded, changed

    return settings.mutate_value(cfg.data_root, _PRESETS_KEY, apply)


def _named(entries: list[_Entry], name: str) -> _Entry:
    for entry in entries:
        if entry.name == name:
            return entry
    raise ValueError(f"No server preset is named {name!r}.")


def _without_key(entry: _Entry) -> _Entry:
    if isinstance(entry, _UnreadableEntry):
        return _UnreadableEntry({k: v for k, v in entry.raw.items() if k != "api_key"})
    return entry.model_copy(update={"api_key": None})


def list_presets() -> list[ServerPreset]:
    """Every stored entry; a stored value that is no list of entries is refused."""
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
        api_key=_checked_key(api_key) if api_key else None,
    )

    def change(entries: list[_Entry]) -> list[_Entry]:
        if any(existing.name == name for existing in entries):
            raise ValueError(f"A server preset named {name!r} already exists.")
        changed = [*entries, entry]
        _check_list(changed)
        return changed

    _store(change)
    return _view(entry)


def edit_preset(
    name: str,
    *,
    url: str | None = None,
    roles: Collection[str] | None = None,
    api_key: str | None = None,
) -> ServerPreset:
    """Change what is passed. An empty or omitted key keeps the stored key.

    A URL with a new scheme, host or port and no new key is refused: the stored key never
    follows the URL to a new origin. A key change alone checks only that preset's entry.
    """
    changes: dict[str, Any] = {}
    if url is not None:
        validate_preset_url(url)
        changes["url"] = url
    if roles is not None:
        changes["roles"] = _checked_roles(roles)
    if api_key:
        changes["api_key"] = _checked_key(api_key)

    def change(entries: list[_Entry]) -> list[_Entry]:
        target = _named(entries, name)
        if isinstance(target, _UnreadableEntry):
            raise ValueError(f"The stored preset {name!r} is refused: {_UNREADABLE_ENTRY}")
        moves_key = target.has_key and not api_key
        if url is not None and moves_key and _origin(target.url) != _origin(url):
            raise ValueError(_KEY_STAYS_WITH_ORIGIN)
        updated = target.model_copy(update=changes)
        changed = [updated if entry is target else entry for entry in entries]
        key_change_only = list(changes) == ["api_key"]
        _check_list([e for e in changed if e.name == name] if key_change_only else changed)
        return changed

    return _view(_named(_store(change), name))


def clear_key(name: str) -> ServerPreset:
    """Remove the stored key of the preset *name*. A clear adds nothing, so no entry can stop it."""

    def change(entries: list[_Entry]) -> list[_Entry]:
        _named(entries, name)
        return [_without_key(entry) if entry.name == name else entry for entry in entries]

    return _view(_named(_store(change), name))


def remove_preset(name: str) -> None:
    """Remove the preset *name*. A removal adds nothing, so no other stored entry can stop it."""

    def change(entries: list[_Entry]) -> list[_Entry]:
        _named(entries, name)
        return [entry for entry in entries if entry.name != name]

    _store(change)
