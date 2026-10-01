"""Parsing helpers used by :mod:`lilbee.config` validators."""

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .enums import OcrMode

log = logging.getLogger(__name__)

# Matches what pydantic itself accepts for a bool field. Every other bool on
# Config is coerced by pydantic, so a narrower vocabulary here would make the
# same env spelling mean different things on different fields of one object.
_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "on", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "off", "0"})

# Retired config.toml keys that the ``ocr`` setting replaces, each with its retired env var.
RETIRED_OCR_KEYS = {"enable_ocr": "LILBEE_ENABLE_OCR", "force_ocr": "LILBEE_OCR_FORCE"}
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


def _ocr_from_retired(retired: dict[str, Any], vision_model: str) -> OcrMode:
    """The ocr mode that stored enable_ocr and force_ocr values stand for."""
    if _stored_bool(retired.get("enable_ocr")) is False and not vision_model:
        return OcrMode.OFF
    if _stored_bool(retired.get("force_ocr")):
        return OcrMode.ALL
    return OcrMode.AUTO


def migrate_ocr_keys(data: dict[str, Any], vision_model: str) -> dict[str, Any]:
    """*data* with the retired OCR keys replaced by ``ocr``; an explicit ``ocr`` wins."""
    retired = {key: data[key] for key in RETIRED_OCR_KEYS if key in data}
    if not retired:
        return data
    migrated = {key: value for key, value in data.items() if key not in retired}
    if "ocr" not in migrated:
        migrated["ocr"] = _ocr_from_retired(retired, vision_model).value
    return migrated


def _replaced_by(retired: list[str], replacement: str = "ocr") -> str:
    """'<names> is/are replaced by <replacement>' for the retired names given."""
    verb = "is" if len(retired) == 1 else "are"
    return f"{' and '.join(retired)} {verb} replaced by {replacement}"


def _ocr_modes() -> str:
    """The ocr modes as a list for a message."""
    return ", ".join(mode.value for mode in OcrMode)


def warn_retired_ocr_keys(data: dict[str, Any]) -> None:
    """Warn that config.toml still carries a retired OCR key."""
    retired = [key for key in RETIRED_OCR_KEYS if key in data]
    if retired:
        log.warning("config.toml: %s; the next settings write saves it", _replaced_by(retired))


def refuse_retired_ocr_keys(keys: Iterable[str]) -> None:
    """Raise ValueError when a request or setting names a retired OCR key instead of ``ocr``."""
    retired = sorted(set(keys) & set(RETIRED_OCR_KEYS))
    if retired:
        raise ValueError(f"{_replaced_by(retired)}; set ocr to one of {_ocr_modes()}")


def refuse_retired_ocr_env(environ: Mapping[str, str]) -> None:
    """Raise ValueError when *environ* sets a retired OCR variable instead of LILBEE_OCR."""
    retired = [name for name in RETIRED_OCR_KEYS.values() if environ.get(name, "").strip()]
    if retired:
        raise ValueError(
            f"{_replaced_by(retired, _OCR_ENV_VAR)}; set {_OCR_ENV_VAR} to one of {_ocr_modes()}"
        )
