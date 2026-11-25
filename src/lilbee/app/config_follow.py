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
from lilbee.core import settings as persistent_settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.context import published_config_scope
from lilbee.core.config.model import follow_published_config
from lilbee.providers.reload_keys import ENGINE_TUNING_KEYS

log = logging.getLogger(__name__)

_NO_FAILURE = object()

ONE_OFF_HOLD: frozenset[str] = frozenset(Config.model_fields) - ENGINE_TUNING_KEYS
"""The keys a one-off command keeps at their start value for its whole run.

It takes only the engine tuning keys, so its retry binds an owner's fleet that was
planned on config.toml. Every model role, role gate and library key stays at its
start value, so its output stays consistent for the run. It also holds the keys its
command-line flags set.
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

    def __init__(self, hold: frozenset[str], *, long_running: bool, keeper: str) -> None:
        self.hold = hold
        # What the failure log says keeps its settings: "the server", "the TUI", "the command".
        self._keeper = keeper
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
        The check reads and publishes the published Config, whatever the caller pins.
        """
        with published_config_scope():
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
                            "%s keeps its current settings.",
                            self._keeper,
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
        """Take the changes another process made to config.toml, and log what was taken.

        The check compares against the published Config and publishes a new one. It
        leaves every pin as it is; ``refresh_now`` also moves its caller's pin.
        """
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

    def refresh_now(self) -> None:
        """Take the changes in config.toml, then move the caller's pin to the published Config.

        The engine's retry runs this through the services: its caller (the fleet's last
        retry) asks for the newest settings, also inside a request.
        """
        self.take_now()
        follow_published_config()
