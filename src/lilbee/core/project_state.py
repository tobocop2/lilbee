"""The project's analyze state in state.toml: when it was analyzed and whether the tip is hidden."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli_w

from lilbee.core.security import file_lock_or_warn, write_private_text

log = logging.getLogger(__name__)

STATE_FILE_NAME = "state.toml"
_ANALYZED_AT_KEY = "analyzed_at"
_TIP_DISMISSED_KEY = "tip_dismissed"
_STATE_LOCK_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ProjectState:
    """When analyze last completed (ISO 8601 UTC) and whether the user hid the tip."""

    analyzed_at: str | None = None
    tip_dismissed: bool = False


def _state_path(root: Path) -> Path:
    return root / STATE_FILE_NAME


def _read_table(path: Path) -> dict[str, Any]:
    """The TOML table at *path*; a missing, unreadable or broken file reads as empty."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log.warning("Ignoring unreadable %s: %s", path, exc)
        return {}


def read_state(root: Path) -> ProjectState:
    """The analyze state of the project at *root*; values of the wrong type read as unset."""
    table = _read_table(_state_path(root))
    analyzed_at = table.get(_ANALYZED_AT_KEY)
    dismissed = table.get(_TIP_DISMISSED_KEY)
    # untyped TOML: a hand-edited file can hold any type under either key
    return ProjectState(
        analyzed_at=analyzed_at if isinstance(analyzed_at, str) else None,
        tip_dismissed=dismissed is True,
    )


def _update(root: Path, key: str, value: Any) -> None:
    """Set *key* in state.toml under its lock, keeping the other keys."""
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock_or_warn(path, _STATE_LOCK_TIMEOUT_S):
        table = _read_table(path)
        table[key] = value
        write_private_text(path, tomli_w.dumps(table))


def mark_analyzed(root: Path) -> None:
    """Record that analyze completed now."""
    _update(root, _ANALYZED_AT_KEY, datetime.now(UTC).isoformat(timespec="seconds"))


def dismiss_tip(root: Path) -> None:
    """Hide the analyze tip for the project at *root*."""
    _update(root, _TIP_DISMISSED_KEY, True)
