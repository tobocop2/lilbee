"""Install the bundled lilbee-mcp skill, and locate the bundled skill and pi extension."""

from __future__ import annotations

import errno
import hashlib
import logging
import sys
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

# sha256 of each version of each file lilbee owns in the skill folder: every version a
# release installed (the PyPI wheels of v0.6.90b420.dev710 to v0.6.90b693) and the bundled one.
_KNOWN_SKILL_FILES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "SKILL.md": frozenset(
            {
                # v0.6.90b420.dev710 to v0.6.90b420.dev721
                "d00d77d707c1d304f0875aec70d23da8185bb10e3a2892590b3118d128fe78b1",
                # v0.6.90b420.dev722 to v0.6.90b420.dev725
                "eb5054bacfd91b3974b16a8e883ce6be4ae8f483ab6b6f68a17d1f39e4a0e894",
                # v0.6.90b420.dev726 to v0.6.90b420.dev728
                "dcbe37b289d61526630e4131cf41be40fae37f9707805e1e5cb66461832db120",
                # v0.6.90b420.dev729
                "9e3266549a9ca2063ed648f84d02a58baccdb4865fe68cdeb95c1aff17d74cce",
                # v0.6.90b420.dev730 to v0.6.90b423
                "aa00332947c4caed23f750a3b72779c072278c1c5c4fb23dc1eddee619371adb",
                # v0.6.90b424 to v0.6.90b425
                "49b82ef46ee302b20969431ef0f288be03bcd325778b41db05a049ec000e3e66",
                # v0.6.90b426 to v0.6.90b432
                "5241a8368c653904fbea8fd44b589da54902a8971070c1cd6a94efdf266ad12b",
                # v0.6.90b433 to v0.6.90b435
                "594ac5714cc5add0ae4ea9503c325b9c6b18f2a4f86640a958ffbbfe4455ae1c",
                # v0.6.90b436 to v0.6.90b445
                "21fd3f7c8c99a7562dfa6f1bef2a28816bf12e1ec34435459169e4539aa542bb",
                # v0.6.90b446 to v0.6.90b447
                "99411a0d234f453429f7c9fde42368ff32ed907108ce8556b072c5c6bac7a588",
                "dcf2b2213a73201385a248d018052d170aef87c1258c131854d04769e83a9fec",  # v0.6.90b448
                # v0.6.90b449 to v0.6.90b690
                "0c77b8b84e8317bdede1b77393e450088dc1fb0b0027c8d630205e1fe3aeeb9e",
                "a58cd4cab79874535b5983b1d5bedc05f7be66d632260d84e767dd4266bd6d04",  # v0.6.90b691
                "92bddb8e9d3b92c18c71747ba323d15d35a072ea3f04ef557b2b9a24cbe646a7",  # v0.6.90b692
                "eceed74a30d320e55506464fc8d800a1189647d336b4de0e5f99aa289646c5eb",  # v0.6.90b693
                "726a7f64240e0c72965a6892fd4bec76a59f7a43b39f837e7df5133c1ad8f950",  # bundled
            }
        ),
        "reference-settings.md": frozenset(
            {
                "9aa05001c478c43da0189b89da456784b3fb1eebb3b6896a771835896281305c",  # v0.6.90b693
                "cb953c3bc1f59dac4ffba8de04fc9d0d6ac0b3fbcc202bcf9b64baee97b6e1e2",  # bundled
            }
        ),
        "reference-tuning.md": frozenset(
            {"105f78bf4cc87d90a0056208fe7f8642c57666d95ae312656b734d1dd6e298a5"}
        ),
    }
)
_PERMISSION_DENIED = "is not writable (permission denied)"
_IN_USE = "is in use"
_SKIP_REASONS: Mapping[int | None, str] = MappingProxyType(
    {
        errno.EEXIST: "is not a folder",
        errno.ENOTDIR: "is not a folder",
        errno.EACCES: _PERMISSION_DENIED,
        errno.EPERM: _PERMISSION_DENIED,
        errno.EROFS: _PERMISSION_DENIED,
        errno.EBUSY: _IN_USE,
    }
)
_DEFAULT_SKIP_REASON = "could not be written"
# Windows ERROR_SHARING_VIOLATION and ERROR_LOCK_VIOLATION: another process holds the file.
_WINDOWS_IN_USE_ERRORS = frozenset({32, 33})


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


def _skip_reason(exc: OSError) -> str:
    """Return the words the skip warning uses for *exc*."""
    # A failed atomic write raises from the OS error, which holds the Windows code.
    origin = exc.__cause__ if isinstance(exc.__cause__, OSError) else exc
    if sys.platform == "win32" and origin.winerror in _WINDOWS_IN_USE_ERRORS:
        return _IN_USE
    return _SKIP_REASONS.get(exc.errno, _DEFAULT_SKIP_REASON)


def install_bundled_skill(dest: Path) -> Path | None:
    """Install the bundled skill at *dest*, or refresh its files if lilbee's copies are unedited.

    Files the user added to *dest* are never touched. A folder that cannot be
    written is skipped with a warning, so the launch goes on.
    """
    bundled = _bundled_skill_files()
    try:
        return _refresh(bundled, dest)
    except OSError as exc:
        log.warning(
            "The lilbee skill at %s %s, so it is not updated: %s", dest, _skip_reason(exc), exc
        )
        return None
