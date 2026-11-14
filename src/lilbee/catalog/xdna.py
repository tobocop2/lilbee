"""The XDNA model format: an NPU model named by its tag, downloaded by the engine."""

from __future__ import annotations

import json
import re
from pathlib import Path

XDNA_TAG_RE = re.compile(r"[a-z0-9][a-z0-9._-]*:[a-z0-9][a-z0-9._-]*")
"""The ``<name>:<tag>`` form of an NPU model, such as ``qwen3:0.6b``."""

_TICKET_DIR = "xdna"
_TICKET_SUFFIX = ".json"
_TICKET_KEY = "checkpoint"
_TICKET_TAG_SEPARATOR = "@"


def ticket_path(models_dir: Path, tag: str) -> Path:
    """The file that records the installed NPU model *tag*."""
    name = tag.replace(":", _TICKET_TAG_SEPARATOR)
    return models_dir / _TICKET_DIR / f"{name}{_TICKET_SUFFIX}"


def write_ticket(models_dir: Path, tag: str) -> Path:
    """Record *tag* as installed. The ticket holds no weights: the engine downloads them."""
    path = ticket_path(models_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({_TICKET_KEY: tag}), encoding="utf-8")
    return path


def read_checkpoint(ticket: Path) -> str:
    """The model tag a ticket records; ``ValueError`` when the file names none."""
    try:
        tag = json.loads(ticket.read_text(encoding="utf-8")).get(_TICKET_KEY)
    except (OSError, ValueError, AttributeError):
        tag = None
    if not isinstance(tag, str) or not XDNA_TAG_RE.fullmatch(tag):
        raise ValueError(f"{ticket} names no NPU model tag.")
    return tag
