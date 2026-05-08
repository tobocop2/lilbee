"""Test host for widgets / screens that need a LilbeeApp without the heavy on_mount.

Replaces the historical ``class _PlainApp(App[None])`` pattern that forced
production code to add ``isinstance(self.app, LilbeeApp)`` defensive guards
at every host-attribute read. With ``TestLilbeeApp(LilbeeApp)`` the production
widgets can assume their host is a LilbeeApp and call ``self.app.task_bar``
etc. directly; the test seam is the ``_test_skip_auto_init`` ClassVar that
short-circuits ``LilbeeApp.on_mount`` before its heavyweight setup runs
(canonicalize-persisted-models, ChatScreen install, signal subscriptions,
worker-pool notification wiring, task_bar detect-pending probe).
"""

from __future__ import annotations

from lilbee.cli.tui.app import LilbeeApp


class TestLilbeeApp(LilbeeApp):
    """LilbeeApp subclass that skips the heavyweight on_mount work for tests."""

    _test_skip_auto_init = True
