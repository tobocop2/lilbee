"""Install the bundled lilbee-mcp guidance skill into an agent's skills directory."""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path

_SKILL_PACKAGE = "lilbee.skills.lilbee_mcp"


def install_bundled_skill(dest: Path) -> Path | None:
    """Copy the bundled lilbee-mcp skill into *dest*; skip (return None) if it exists."""
    if dest.exists():
        return None
    source = resources.files(_SKILL_PACKAGE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stage in a sibling temp dir and atomically rename so a partial copy never
    # leaves a half-written skill dir that exists() would then skip forever.
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".lilbee-mcp-"))
    try:
        for entry in source.iterdir():
            if entry.is_file() and not entry.name.startswith("__"):
                (staging / entry.name).write_bytes(entry.read_bytes())
        try:
            os.replace(staging, dest)
        except OSError:
            # On Windows, os.replace into an existing dest can race with another
            # installer. If dest now exists, the skill is already installed.
            shutil.rmtree(staging, ignore_errors=True)
            if dest.exists():
                return None
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest
