"""Install the bundled lilbee-mcp skill, and locate the bundled skill and pi extension."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_MARKER = ".lilbee-skill-sha256"

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


def _digest(directory: Path) -> str:
    """Return a sha256 over the names and bytes of the files directly in *directory*."""
    total = hashlib.sha256()
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.name != _MARKER:
            total.update(
                f"{entry.name}\0{hashlib.sha256(entry.read_bytes()).hexdigest()}\0".encode()
            )
    return total.hexdigest()


def _recorded_digest(dest: Path) -> str | None:
    """Return the digest recorded when lilbee installed *dest*, or None if there is none."""
    marker = dest / _MARKER
    return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None


def _needs_refresh(dest: Path, bundled: str) -> bool:
    """Return True when *dest* is an unedited copy of an older bundled skill."""
    installed = _digest(dest)
    if installed == bundled:
        return False
    if _recorded_digest(dest) != installed:
        log.warning(
            "The lilbee skill at %s has local changes, so it is not updated. "
            "Delete the folder to get the bundled version.",
            dest,
        )
        return False
    return True


def _swap_in(staging: Path, dest: Path) -> None:
    """Replace the existing *dest* with *staging*, restoring the old copy on failure."""
    previous = dest.with_name(f".lilbee-mcp-old-{os.getpid()}")
    os.replace(dest, previous)
    try:
        os.replace(staging, dest)
    except BaseException:
        os.replace(previous, dest)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def install_bundled_skill(dest: Path) -> Path | None:
    """Install the bundled lilbee-mcp skill at *dest*, or refresh it if unedited and older."""
    source = bundled_skill_dir()
    bundled = _digest(source)
    replacing = dest.exists()
    if replacing and not _needs_refresh(dest, bundled):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stage in a sibling temp dir and rename it into place so a partial copy never
    # leaves a half-written skill dir.
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".lilbee-mcp-"))
    try:
        for entry in source.iterdir():
            if entry.is_file():
                (staging / entry.name).write_bytes(entry.read_bytes())
        (staging / _MARKER).write_text(bundled, encoding="utf-8")
        if replacing:
            _swap_in(staging, dest)
            return dest
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
