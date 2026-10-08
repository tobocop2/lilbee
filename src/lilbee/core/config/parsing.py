"""Parsing helpers used by :mod:`lilbee.config` validators."""

from collections.abc import Iterable, Mapping
from typing import Any

from .enums import OcrMode
from .load_warnings import RefusedVariableError, warn_on_load

# Matches what pydantic itself accepts for a bool field. Every other bool on
# Config is coerced by pydantic, so a narrower vocabulary here would make the
# same env spelling mean different things on different fields of one object.
_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "on", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "off", "0"})

# The retired config.toml key that ``ocr`` replaces, and the retired env vars the CLI refuses.
_RETIRED_OCR_KEY = "enable_ocr"
_RETIRED_OCR_ENV_VARS = ("LILBEE_ENABLE_OCR", "LILBEE_OCR_FORCE")
_OCR_ENV_VAR = "LILBEE_OCR"


def parse_bool(raw: str) -> bool:
    """Parse a boolean env string; raises ValueError on anything else."""
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean: {raw!r}")


def _stored_bool(value: Any) -> bool | None:
    """A stored TOML bool or bool string; None for anything else (auto, empty, unparseable)."""
    if isinstance(value, bool):
        return value
    try:
        return parse_bool(str(value))
    except ValueError:
        return None


def _ocr_from_enable_ocr(stored: Any, vision_model: str) -> OcrMode:
    """The ocr mode a stored enable_ocr value stands for."""
    if _stored_bool(stored) is False and not vision_model:
        return OcrMode.OFF
    return OcrMode.AUTO


def migrate_ocr_keys(data: dict[str, Any], vision_model: str) -> dict[str, Any]:
    """*data* with a stored ``enable_ocr`` replaced by ``ocr``; an explicit ``ocr`` wins."""
    if _RETIRED_OCR_KEY not in data:
        return data
    migrated = {key: value for key, value in data.items() if key != _RETIRED_OCR_KEY}
    if "ocr" not in migrated:
        migrated["ocr"] = _ocr_from_enable_ocr(data[_RETIRED_OCR_KEY], vision_model).value
    return migrated


def without_refused_ocr(data: dict[str, Any]) -> dict[str, Any]:
    """*data* less an ``ocr`` value OcrMode refuses, when ``enable_ocr`` can stand in for it."""
    if "ocr" not in data or _RETIRED_OCR_KEY not in data:
        return data
    try:
        OcrMode(data["ocr"])
    except ValueError:
        return {key: value for key, value in data.items() if key != "ocr"}
    return data


def refused_value_fallback(key: str, values: dict[str, Any]) -> str:
    """What stands in for a refused value: a stored ``enable_ocr``, else the default."""
    if key == "ocr" and _RETIRED_OCR_KEY in values:
        return f"ocr comes from {_RETIRED_OCR_KEY}"
    return f"{key} uses its default"


def _replaced_by(retired: list[str], replacement: str = "ocr") -> str:
    """'<names> is/are replaced by <replacement>' for the retired names given."""
    verb = "is" if len(retired) == 1 else "are"
    return f"{' and '.join(retired)} {verb} replaced by {replacement}"


def _ocr_modes() -> str:
    """The ocr modes as a list for a message."""
    return ", ".join(mode.value for mode in OcrMode)


def warn_retired_ocr_keys(data: dict[str, Any]) -> None:
    """Warn that config.toml still carries ``enable_ocr``."""
    if _RETIRED_OCR_KEY in data:
        warn_on_load(
            f"config.toml: {_replaced_by([_RETIRED_OCR_KEY])}; the next settings write saves it"
        )


def refuse_retired_ocr_keys(keys: Iterable[str]) -> None:
    """Raise ValueError when a request or setting names ``enable_ocr`` instead of ``ocr``."""
    if _RETIRED_OCR_KEY in keys:
        raise ValueError(f"{_replaced_by([_RETIRED_OCR_KEY])}; set ocr to one of {_ocr_modes()}")


def refuse_retired_ocr_env(environ: Mapping[str, str]) -> None:
    """Raise RefusedVariableError when *environ* sets a retired OCR variable."""
    retired = [name for name in _RETIRED_OCR_ENV_VARS if environ.get(name, "").strip()]
    if retired:
        raise RefusedVariableError(
            f"{_replaced_by(retired, _OCR_ENV_VAR)}; set {_OCR_ENV_VAR} to one of {_ocr_modes()}"
        )
