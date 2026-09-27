"""Pydantic ``Field`` wrapper with lilbee-specific schema metadata."""

from typing import Any

from pydantic import Field

from .enums import ProfileScope


def ConfigField(  # noqa: N802  pydantic Field wrapper; matches Field's PascalCase
    *args: Any,
    writable: bool = False,
    reindex: bool = False,
    write_only: bool = False,
    public: bool = True,
    derived: bool = False,
    profile: ProfileScope | None = None,
    **kwargs: Any,
) -> Any:
    """Wrap pydantic ``Field`` and attach metadata via ``json_schema_extra``.

    ``derived`` marks a field whose default means lilbee computes the value at runtime.
    ``profile`` marks a field a profile may hold, and the part of lilbee it tunes.
    """
    extra: dict[str, bool | str] = {}
    if writable:
        extra["writable"] = True
    if derived:
        extra["derived"] = True
    if reindex:
        extra["reindex"] = True
    if write_only:
        extra["write_only"] = True
    if not public:
        extra["public"] = False
    if profile is not None:
        extra["profile"] = profile.value
    if extra:
        # Merge rather than assign: a caller passing its own json_schema_extra
        # had it silently dropped, with lilbee's flags winning.
        supplied = kwargs.get("json_schema_extra")
        kwargs["json_schema_extra"] = {**supplied, **extra} if isinstance(supplied, dict) else extra
    return Field(*args, **kwargs)
