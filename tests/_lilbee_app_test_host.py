"""LilbeeApp test host: skips the heavyweight on_mount setup."""

from __future__ import annotations

from lilbee.cli.tui.app import LilbeeApp


class LilbeeAppHost(LilbeeApp):
    """LilbeeApp subclass that skips the heavyweight on_mount work for tests."""

    _test_skip_auto_init = True
