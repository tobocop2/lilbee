"""Boolean parsing helpers used by :mod:`lilbee.config` validators."""

# Matches what pydantic itself accepts for a bool field. Every other bool on
# Config is coerced by pydantic, so a narrower vocabulary here would make the
# same env spelling mean different things on different fields of one object.
_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "on", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "off", "0"})


def parse_bool(raw: str) -> bool:
    """Parse a boolean env string; raises ValueError on anything else."""
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean: {raw!r}")
