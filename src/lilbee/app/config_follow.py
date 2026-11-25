"""Take what another process wrote to config.toml while this process runs."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

from lilbee.app.settings import (
    ConfigReading,
    SettingsUpdateResult,
    adopt_external_changes,
    reset_for_provider_switch,
    start_config_reading,
)
from lilbee.config_meta import REINDEX_FIELDS
from lilbee.core import settings as persistent_settings
from lilbee.core.config import Config, cfg
from lilbee.providers.reload_keys import FLEET_RELOAD_KEYS

log = logging.getLogger(__name__)

_NO_FAILURE = object()

ENGINE_ONLY_KEYS: frozenset[str] = FLEET_RELOAD_KEYS - REINDEX_FIELDS
"""The keys that make the fleet owner rebuild and leave the library as it is."""

ONE_OFF_HOLD: frozenset[str] = frozenset(Config.model_fields) - ENGINE_ONLY_KEYS
"""The keys a one-off command keeps at their start value for its whole run.

It takes only the engine-only keys, so its retry binds an owner's fleet that was
planned on config.toml, and its output stays consistent with the library settings
it started on. It also holds the keys its command-line flags set.
"""


class ConfigFileFollower:
    """Keeps a process current with config.toml, one check at a time.

    The check is two stat() calls against the last reading, under one thread lock that
    a request's check and the fleet's retry check share. A change goes through the same
    settle step as a write through the settings API, so this process's own writes are
    not applied twice. A key in *hold* is never taken. A provider switch stops the
    fleet after the lock is free. A check that fails leaves the settings in memory;
    the error is logged once per file state.
    """

    def __init__(self, hold: frozenset[str], *, long_running: bool) -> None:
        self.hold = hold
        # Whether the process runs until it is stopped (serve, the stdio MCP server, the TUI).
        self.long_running = long_running
        self._reading: ConfigReading = start_config_reading()
        self._lock = threading.Lock()
        # The file state of the last failed check, so the error is logged once per state.
        self._failed_stamp: object = _NO_FAILURE
        self._on_taken: Callable[[SettingsUpdateResult], None] | None = None

    def report_to(self, on_taken: Callable[[SettingsUpdateResult], None]) -> None:
        """Give each take that ``take_now`` makes to *on_taken*, such as a screen that shows it."""
        self._on_taken = on_taken

    def check(self) -> tuple[bool, SettingsUpdateResult | None]:
        """Take the outside changes; return whether the file moved since the last check.

        Also returns what was taken, or None. On an error the settings in memory stay.
        """
        with self._lock:
            before = self._reading
            try:
                self._reading, result = adopt_external_changes(
                    before, hold=self.hold, reset_providers=False
                )
            except Exception:
                stamp = persistent_settings.overlay_stamp(cfg.data_root)
                if stamp != self._failed_stamp:
                    self._failed_stamp = stamp
                    log.exception(
                        "Could not take the changes in config.toml; "
                        "this process keeps its current settings."
                    )
                return False, None
            self._failed_stamp = _NO_FAILURE
            changed = self._reading is not before
        reset_for_provider_switch(result)
        return changed, result

    async def take(self) -> None:
        """Take the changes another process made to config.toml, off the event loop."""
        await asyncio.to_thread(self.take_now)

    def take_now(self) -> None:
        """Take the changes another process made to config.toml, and log what was taken."""
        _changed, result = self.check()
        if result is None:
            return
        log.info("Took %s from config.toml.", ", ".join(result.updated))
        if result.reindex_required:
            log.warning(
                "config.toml changed a setting the library was built with; run lilbee rebuild."
            )
        for warning in result.warnings:
            log.warning("%s", warning)
        if self._on_taken is not None:
            self._on_taken(result)
