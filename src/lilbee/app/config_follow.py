"""Take what another process wrote to config.toml before a long-running process answers."""

from __future__ import annotations

import asyncio
import logging

from lilbee.app.settings import (
    ConfigReading,
    SettingsUpdateResult,
    adopt_external_changes,
    read_config_state,
)
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.core.config.keys import PROVIDER_SWITCHING_KEYS

log = logging.getLogger(__name__)

_NO_FAILURE = object()


class ConfigFileFollower:
    """Keeps a running process current with config.toml, one check per request.

    The check is two stat() calls against the last reading, off the loop and one at a
    time. A change goes through the same settle step as a write through the settings
    API, so this process's own writes are not applied twice. A provider switch waits for
    a restart, as the settings API refuses it here. A check that fails leaves the
    settings in memory and the request runs; the error is logged once per file state.
    """

    def __init__(self) -> None:
        self._reading: ConfigReading = read_config_state()
        self._lock = asyncio.Lock()
        # The file state of the last failed check, so the error is logged once per state.
        self._failed_stamp: object = _NO_FAILURE

    async def _adopt(self) -> SettingsUpdateResult | None:
        """Take the outside changes; on an error, keep the settings in memory."""
        async with self._lock:
            try:
                self._reading, result = await asyncio.to_thread(
                    adopt_external_changes, self._reading, hold=PROVIDER_SWITCHING_KEYS
                )
            except Exception:
                stamp = persistent_settings.overlay_stamp(cfg.data_root)
                if stamp != self._failed_stamp:
                    self._failed_stamp = stamp
                    log.exception(
                        "Could not take the changes in config.toml; "
                        "the server keeps its current settings."
                    )
                return None
            self._failed_stamp = _NO_FAILURE
            return result

    async def take(self) -> None:
        """Take the changes another process made to config.toml, and log what was taken."""
        result = await self._adopt()
        if result is None:
            return
        log.info("Took %s from config.toml.", ", ".join(result.updated))
        if result.reindex_required:
            log.warning(
                "config.toml changed a setting the library was built with; run lilbee rebuild."
            )
        for warning in result.warnings:
            log.warning("%s", warning)
