"""Boolean parsing helpers used by :mod:`lilbee.config` validators."""

_BOOL_TRUE = frozenset({"true", "1", "yes"})
_BOOL_FALSE = frozenset({"false", "0", "no"})


def _parse_bool(raw: str) -> bool:
    """Parse true/1/yes or false/0/no; raises ValueError on anything else."""
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean: {raw!r}")
