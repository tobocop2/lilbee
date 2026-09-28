"""Constants and small rendering helpers shared across the CLI command submodules."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg

CHUNK_PREVIEW_LEN = 80  # characters shown in human-readable search output

REBUILD_HINT = "Run lilbee rebuild so the index uses the new values."


def shown_value(value: Any) -> str:
    """A setting value the way a profile or config file writes it: "auto", true, 512."""
    return json.dumps(to_jsonable_python(value))


def emit(model: BaseModel, render: Callable[[], None]) -> None:
    """Print *model* as JSON in --json mode, else run *render*."""
    if cfg.json_mode:
        json_output(model.model_dump(mode="json"))
    else:
        render()
