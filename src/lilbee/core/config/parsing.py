"""Parsing helpers shared by the Config field validators and the resolver's soft checks."""

from collections.abc import Mapping
from types import MappingProxyType

# Matches what pydantic itself accepts for a bool field. Every other bool on
# Config is coerced by pydantic, so a narrower vocabulary here would make the
# same env spelling mean different things on different fields of one object.
_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "on", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "off", "0"})
_AUTO_ALIASES = frozenset({"", "auto", "none"})


def parse_bool(raw: str) -> bool:
    """Parse a boolean env string; raises ValueError on anything else."""
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean: {raw!r}")


def parse_tristate_bool(raw: str) -> bool | None:
    """Parse an auto/on/off string: blank, ``auto`` or ``none`` means auto-detect."""
    if raw.strip().lower() in _AUTO_ALIASES:
        return None
    return parse_bool(raw)


def parse_optional_int(
    raw: str, *, aliases: Mapping[str, int] = MappingProxyType({})
) -> int | None:
    """Parse an optional integer: blank/``auto``/``none`` is None; *aliases* map other labels."""
    label = raw.strip().lower()
    if label in _AUTO_ALIASES:
        return None
    if label in aliases:
        return aliases[label]
    return int(label)


def parse_gpu_device_list(raw: str) -> str | None:
    """Parse a comma-separated device index list; blank/auto/all/none means every device."""
    label = raw.strip().lower()
    if label in _AUTO_ALIASES or label == "all":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    for part in parts:
        if not part.lstrip("-").isdigit():
            raise ValueError(f"Invalid device index: {part!r}")
    return ",".join(parts)
