"""Read the shell of a workflow step the way the shell does, so a comment is not a command."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def steps(workflow: str, job: str) -> list[dict[str, Any]]:
    """Return the parsed steps of *job* in the workflow file *workflow*."""
    parsed = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    return list(parsed["jobs"][job]["steps"])


def commands(script: str) -> str:
    """Return *script* as one line of its words, with every comment dropped and quotes removed."""
    return " ".join(shlex.split(script, comments=True))
