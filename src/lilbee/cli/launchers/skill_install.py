"""Install the bundled lilbee-mcp skill, and locate the bundled skill and pi extension."""

from __future__ import annotations

import errno
import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from lilbee.core.security import write_text_atomically

log = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED_SKILL = _PACKAGE_ROOT / "skills" / "lilbee_mcp"
_CHECKOUT_SKILL = _PACKAGE_ROOT.parents[1] / "plugins" / "lilbee" / "skills" / "lilbee-mcp"
_PACKAGED_AGENT_PLUGIN = _PACKAGE_ROOT / "agent_plugin"
_CHECKOUT_AGENT_PLUGIN = _PACKAGE_ROOT.parents[1] / "packages" / "lilbee" / "plugin"
_PI_EXTENSION = Path("extensions") / "lilbee.js"

# sha256 of every released or bundled version of each file lilbee owns in the skill folder.
_KNOWN_SKILL_FILES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "SKILL.md": frozenset(
            {
                "92bddb8e9d3b92c18c71747ba323d15d35a072ea3f04ef557b2b9a24cbe646a7",  # v0.6.90b692
                "eceed74a30d320e55506464fc8d800a1189647d336b4de0e5f99aa289646c5eb",
            }
        ),
        "reference-settings.md": frozenset(
            {"9aa05001c478c43da0189b89da456784b3fb1eebb3b6896a771835896281305c"}
        ),
        "reference-tuning.md": frozenset(
            {"105f78bf4cc87d90a0056208fe7f8642c57666d95ae312656b734d1dd6e298a5"}
        ),
    }
)
_PERMISSION_DENIED = "is not writable (permission denied)"
_SKIP_REASONS: Mapping[int | None, str] = MappingProxyType(
    {
        errno.EEXIST: "is not a folder",
        errno.ENOTDIR: "is not a folder",
        errno.EACCES: _PERMISSION_DENIED,
        errno.EPERM: _PERMISSION_DENIED,
        errno.EROFS: _PERMISSION_DENIED,
        errno.EBUSY: "is in use",
    }
)
_DEFAULT_SKIP_REASON = "could not be written"


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


def _bundled_skill_files() -> dict[str, bytes]:
    """Return the name and bytes of each file in the bundled skill."""
    return {
        entry.name: entry.read_bytes() for entry in bundled_skill_dir().iterdir() if entry.is_file()
    }


def _is_edited(path: Path) -> bool:
    """Return True when *path* holds bytes that no released or bundled skill shipped."""
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest not in _KNOWN_SKILL_FILES.get(path.name, frozenset())


def _refresh(bundled: dict[str, bytes], dest: Path) -> Path | None:
    """Write each bundled file that *dest* lacks or holds in an older version."""
    dest.mkdir(parents=True, exist_ok=True)
    if any(_is_edited(dest / name) for name in bundled):
        log.warning(
            "The lilbee skill at %s has local changes, so it is not updated. "
            "Delete the folder to get the bundled version.",
            dest,
        )
        return None
    stale = [
        name
        for name, data in bundled.items()
        if not (dest / name).is_file() or (dest / name).read_bytes() != data
    ]
    for name in stale:
        write_text_atomically(dest / name, bundled[name].decode("utf-8"))
    return dest if stale else None


def install_bundled_skill(dest: Path) -> Path | None:
    """Install the bundled skill at *dest*, or refresh its files if lilbee's copies are unedited.

    Files the user added to *dest* are never touched. A folder that cannot be
    written is skipped with a warning, so the launch goes on.
    """
    bundled = _bundled_skill_files()
    try:
        return _refresh(bundled, dest)
    except OSError as exc:
        reason = _SKIP_REASONS.get(exc.errno, _DEFAULT_SKIP_REASON)
        log.warning("The lilbee skill at %s %s, so it is not updated: %s", dest, reason, exc)
        return None
