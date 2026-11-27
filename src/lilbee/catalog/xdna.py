"""The XDNA model format: an NPU model named by its tag, downloaded by the engine."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lilbee.catalog.types import ModelTask

XDNA_TAG_RE = re.compile(r"[a-z0-9][a-z0-9._-]*:[a-z0-9][a-z0-9._-]*")
"""The ``<name>:<tag>`` form of an NPU model, such as ``qwen3:0.6b``."""

_TICKET_DIR = "xdna"
_TICKET_SUFFIX = ".json"
_TICKET_KEY = "checkpoint"
_TICKET_TASK_KEY = "task"
_TICKET_FOOTPRINT_KEY = "footprint_bytes"
_TICKET_CONTEXT_KEY = "context_length"
_TICKET_TAG_SEPARATOR = "@"
_BYTES_PER_GIB = 1024**3


def ticket_path(models_dir: Path, tag: str) -> Path:
    """The file that records the installed NPU model *tag*."""
    name = tag.replace(":", _TICKET_TAG_SEPARATOR)
    return models_dir / _TICKET_DIR / f"{name}{_TICKET_SUFFIX}"


def ticket_paths(models_dir: Path) -> list[Path]:
    """Every ticket file under *models_dir*, sorted; empty when none was written."""
    return sorted((models_dir / _TICKET_DIR).glob(f"*{_TICKET_SUFFIX}"))


def write_ticket(
    models_dir: Path,
    tag: str,
    task: ModelTask,
    footprint_gb: float = 0.0,
    context_length: int = 0,
) -> Path:
    """Record *tag* as installed for *task*. The ticket holds no weights.

    *footprint_gb* is the memory the publisher states for the model. The ticket records it
    in bytes, and records none when the publisher states none. *context_length* is the
    window the publisher states, in tokens; the ticket records none when it is 0.
    """
    path = ticket_path(models_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, object] = {_TICKET_KEY: tag, _TICKET_TASK_KEY: str(task)}
    if footprint_gb > 0:
        fields[_TICKET_FOOTPRINT_KEY] = round(footprint_gb * _BYTES_PER_GIB)
    if context_length > 0:
        fields[_TICKET_CONTEXT_KEY] = context_length
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _positive_int(value: object) -> int | None:
    """*value* when it is a positive int; ``None`` for anything else, a bool included."""
    # isinstance: the value is untyped JSON from a file
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def read_footprint(ticket: Path) -> int | None:
    """The memory in bytes that a ticket records for its model; ``None`` when it records none.

    A ticket from a release that recorded no footprint, and an unreadable ticket, give ``None``.
    """
    return _positive_int(_read_field(ticket, _TICKET_FOOTPRINT_KEY))


def read_context_length(ticket: Path) -> int | None:
    """The window in tokens that a ticket records for its model; ``None`` when it records none.

    A ticket from a release that recorded no window, and an unreadable ticket, give ``None``.
    """
    return _positive_int(_read_field(ticket, _TICKET_CONTEXT_KEY))


def _read_field(ticket: Path, key: str) -> object:
    """The value under *key* in *ticket*; ``None`` when the file or the key is unreadable."""
    try:
        return json.loads(ticket.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError, AttributeError):
        return None


def read_checkpoint(ticket: Path) -> str:
    """The model tag a ticket records; ``ValueError`` when the file names none."""
    tag = _read_field(ticket, _TICKET_KEY)
    if not isinstance(tag, str) or not XDNA_TAG_RE.fullmatch(tag):
        raise ValueError(f"{ticket} names no NPU model tag.")
    return tag


def read_task(ticket: Path) -> ModelTask:
    """The task a ticket records; ``ValueError`` when the file names none."""
    task = _read_field(ticket, _TICKET_TASK_KEY)
    if isinstance(task, str) and task in set(ModelTask):
        return ModelTask(task)
    raise ValueError(f"{ticket} names no model task.")
