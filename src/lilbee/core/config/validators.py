"""Pydantic ``Field`` wrapper with lilbee-specific schema metadata."""

from typing import Any

from pydantic import Field


def ConfigField(  # noqa: N802  pydantic Field wrapper; matches Field's PascalCase
    *args: Any,
    writable: bool = False,
    reindex: bool = False,
    write_only: bool = False,
    public: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap pydantic ``Field`` and attach metadata via ``json_schema_extra``."""
    extra: dict[str, bool] = {}
    if writable:
        extra["writable"] = True
    if reindex:
        extra["reindex"] = True
    if write_only:
        extra["write_only"] = True
    if not public:
        extra["public"] = False
    if extra:
        # Merge rather than assign: a caller passing its own json_schema_extra
        # had it silently dropped, with lilbee's flags winning.
        supplied = kwargs.get("json_schema_extra")
        kwargs["json_schema_extra"] = {**supplied, **extra} if isinstance(supplied, dict) else extra
    return Field(*args, **kwargs)
