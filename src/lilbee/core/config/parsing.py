"""Parsing helpers used by :mod:`lilbee.config` validators."""

import logging
from collections.abc import Iterable
from typing import Any

from .enums import OcrMode

log = logging.getLogger(__name__)

# Matches what pydantic itself accepts for a bool field. Every other bool on
# Config is coerced by pydantic, so a narrower vocabulary here would make the
# same env spelling mean different things on different fields of one object.
_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "on", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "off", "0"})

# Retired config.toml keys that the ``ocr`` setting replaces.
RETIRED_OCR_KEYS = ("enable_ocr", "force_ocr")


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


def warn_retired_ocr_keys(data: dict[str, Any]) -> None:
    """Warn that config.toml still carries a retired OCR key."""
    retired = [key for key in RETIRED_OCR_KEYS if key in data]
    if retired:
        log.warning(
            "config.toml key %s is replaced by ocr; the next settings write saves it",
            " and ".join(retired),
        )


def refuse_retired_ocr_keys(keys: Iterable[str]) -> None:
    """Raise ValueError when a request names a retired OCR key instead of ``ocr``."""
    retired = sorted(set(keys) & set(RETIRED_OCR_KEYS))
    if retired:
        raise ValueError(
            f"{' and '.join(retired)} is no longer accepted; "
            f"use ocr with one of {', '.join(mode.value for mode in OcrMode)}"
        )
