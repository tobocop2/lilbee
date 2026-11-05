"""Install the bundled lilbee-mcp skill, and locate the bundled skill and pi extension."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED_SKILL = _PACKAGE_ROOT / "skills" / "lilbee_mcp"
_CHECKOUT_SKILL = _PACKAGE_ROOT.parents[1] / "plugins" / "lilbee" / "skills" / "lilbee-mcp"
_PACKAGED_AGENT_PLUGIN = _PACKAGE_ROOT / "agent_plugin"
_CHECKOUT_AGENT_PLUGIN = _PACKAGE_ROOT.parents[1] / "packages" / "lilbee" / "plugin"
_PI_EXTENSION = Path("extensions") / "lilbee.js"


def _installed_or_checkout(packaged: Path, checkout: Path, marker: Path) -> Path:
    """Return whichever of *packaged* and *checkout* holds *marker*, the installed one first."""
    for candidate in (packaged, checkout):
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(f"{marker.as_posix()} not found under {packaged} or {checkout}")


def bundled_skill_dir() -> Path:
    """Return the installed skill directory, or the checkout one in a source tree."""
    return _installed_or_checkout(_PACKAGED_SKILL, _CHECKOUT_SKILL, Path("SKILL.md"))


def bundled_pi_extension() -> Path:
    """Return the lilbee pi extension the npm package ships, as the wheel or a checkout holds it."""
    plugin = _installed_or_checkout(_PACKAGED_AGENT_PLUGIN, _CHECKOUT_AGENT_PLUGIN, _PI_EXTENSION)
    return plugin / _PI_EXTENSION


def install_bundled_skill(dest: Path) -> Path | None:
    """Copy the bundled lilbee-mcp skill into *dest*; skip (return None) if it exists."""
    if dest.exists():
        return None
    source = bundled_skill_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stage in a sibling temp dir and atomically rename so a partial copy never
    # leaves a half-written skill dir that exists() would then skip forever.
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".lilbee-mcp-"))
    try:
        for entry in source.iterdir():
            if entry.is_file():
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
