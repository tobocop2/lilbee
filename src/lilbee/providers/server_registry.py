"""The stored server presets and the prefixes that route a model ref to a server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from lilbee.catalog.types import ModelSource
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.local_servers import LOCAL_SERVER_KEYS
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from lilbee.providers.model_ref import ProviderModelRef

log = logging.getLogger(__name__)

PRESETS_KEY = "server_presets"
API_PROVIDERS = frozenset({"openrouter", "gemini", "anthropic", "openai", "mistral", "deepseek"})
# Prefixes that route a ref away from the local registry; a constant at run time.
BUILTIN_PREFIXES: frozenset[str] = frozenset(API_PROVIDERS | LOCAL_SERVER_KEYS)

_UNPARSED_STORE = (
    "The stored server presets do not parse. Fix or remove server_presets in config.toml."
)


class RerankWire(StrEnum):
    """The request shape a preset's rerank endpoint reads."""

    V1 = "v1"  # POST /v1/rerank with documents: Jina, Cohere, llama-server, Lemonade
    TEI = "tei"  # POST /rerank with texts: Hugging Face text-embeddings-inference


def checked_key(api_key: str) -> SecretStr:
    """*api_key* as a secret; ValueError that does not repeat it when it cannot be stored."""
    # A compare and no caught UnicodeEncodeError: its object would keep the key on the chain.
    if api_key.encode("utf-8", "replace").decode("utf-8") != api_key:
        raise ValueError("The API key holds a character that cannot be stored.")
    return SecretStr(api_key)


class StoredPreset(BaseModel):
    """One preset as config.toml holds it; the key and the URL credentials stay out of its repr."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    name: str
    url: str = Field(repr=False)
    roles: Annotated[tuple[WorkerRole, ...], Field(min_length=1)]
    api_key: SecretStr | None = None
    rerank_wire: RerankWire = RerankWire.V1

    @field_validator("api_key")
    @classmethod
    def _key_can_be_stored(cls, value: SecretStr | None) -> SecretStr | None:
        return value and checked_key(value.get_secret_value())

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class UnreadableEntry:
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


Entry = StoredPreset | UnreadableEntry


def _decoded_row(row: Any) -> Entry:
    try:
        return StoredPreset.model_validate(row)
    except ValidationError:
        return UnreadableEntry(row)


def decode_entries(raw: Any) -> list[Entry]:
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


def stored_entries() -> list[Entry]:
    """Every stored entry, read from config.toml now; a value that is no list is refused."""
    # Deferred: the config builds its fields through parse_model_ref, which imports this module.
    from lilbee.core import settings
    from lilbee.core.config import cfg

    return decode_entries(settings.load(cfg.data_root).get(PRESETS_KEY))


@dataclass(frozen=True)
class ServerRoute:
    """Where a preset ref goes: the server, the key it sends, the roles it serves."""

    name: str
    base_url: str = field(repr=False)
    api_key: str | None = field(repr=False)
    roles: tuple[WorkerRole, ...]
    rerank_wire: RerankWire


def route_for(prefix: str) -> ServerRoute | None:
    """The route of the stored preset named *prefix*; None for a built-in prefix or no preset.

    The store is read on each call, so a preset another process changes is seen at once.
    A store that does not parse routes nothing; the read has logged the fault.
    """
    if prefix in BUILTIN_PREFIXES:
        return None
    try:
        entries = stored_entries()
    except ValueError:
        return None
    for entry in entries:
        if isinstance(entry, StoredPreset) and entry.name == prefix:
            key = entry.api_key.get_secret_value() if entry.api_key else None
            return ServerRoute(entry.name, entry.url, key, entry.roles, entry.rerank_wire)
    return None


def check_role(ref: ProviderModelRef, role: WorkerRole) -> ServerRoute:
    """The route of the preset *ref* names; ProviderError when it serves no *role*."""
    route = route_for(ref.provider)
    if route is None:
        raise ProviderError(
            f"No server preset is named {ref.provider!r}. Add the preset or change the model.",
            provider=ModelSource.REMOTE,
            kind=ProviderErrorKind.NOT_FOUND,
        )
    if role not in route.roles:
        raise ProviderError(
            f"The server preset {route.name!r} does not serve the {role.value} role. "
            "Add the role to the preset or choose another model.",
            provider=ModelSource.REMOTE,
            kind=ProviderErrorKind.BAD_REQUEST,
        )
    return route
