"""Startup sweep of the onefile extraction directories left by older releases."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import lilbee
from lilbee._frozen import is_frozen

logger = logging.getLogger(__name__)

# Written by tools/wheel-build/onefile-bootstrap-lilbee.patch into every payload directory.
BOOTSTRAP_MANIFEST_NAME = ".lilbee-bootstrap-manifest"
# Payload directories are named ``{VERSION}-<build key>`` by the build script.
_BUILD_KEY_SEPARATOR = "-"


def _extraction_dir() -> Path:
    """The payload directory of the running binary; the compiled package sits directly inside it."""
    return Path(lilbee.__file__).resolve().parent.parent


def _release_version(directory: Path) -> str:
    return directory.name.partition(_BUILD_KEY_SEPARATOR)[0]


def _stale_siblings(running: Path) -> list[Path]:
    """Payload directories beside *running* that another release wrote."""
    running_version = _release_version(running)
    return [
        path
        for path in running.parent.iterdir()
        if path != running
        and _release_version(path) != running_version
        and (path / BOOTSTRAP_MANIFEST_NAME).is_file()
    ]


def _remove(path: Path) -> bool:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.debug("Left the onefile cache %s in place: %s", path, exc)
        return False
    return True


def remove_stale_extractions(running: Path) -> list[Path]:
    """Delete the payload directories of other releases beside *running*; never raises."""
    try:
        stale = _stale_siblings(running)
    except OSError as exc:
        logger.debug("Skipped the onefile cache sweep of %s: %s", running.parent, exc)
        return []
    removed = [path for path in stale if _remove(path)]
    if removed:
        logger.info("Removed the onefile cache of older releases: %s", ", ".join(map(str, removed)))
    return removed


def cleanup_stale_onefile_caches() -> None:
    """Startup hook; a no-op outside the compiled binary."""
    if is_frozen():
        remove_stale_extractions(_extraction_dir())
