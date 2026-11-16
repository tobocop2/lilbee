"""The toast record keeps a toast that Textual's own list has already dropped."""

from __future__ import annotations

import time

import pytest
from textual.app import App

from tests._async_wait import wait_until
from tests._toasts import posted_toasts


async def test_a_toast_shown_before_a_long_stall_is_still_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = App[None]()
    async with app.run_test() as pilot:
        app.notify("Exported legal-discovery", timeout=5)
        assert await wait_until(pilot, lambda: bool(posted_toasts()))

        after_stall = time.time() + 30
        monkeypatch.setattr("textual.notifications.time", lambda: after_stall)

        assert [toast.message for toast in posted_toasts()] == ["Exported legal-discovery"]
        assert [toast.timeout for toast in posted_toasts()] == [5]


async def test_the_record_holds_each_toast_in_the_order_shown() -> None:
    app = App[None]()
    async with app.run_test() as pilot:
        app.notify("first")
        app.notify("second", severity="warning")
        assert await wait_until(pilot, lambda: len(posted_toasts()) == 2)

        assert [(toast.message, toast.severity) for toast in posted_toasts()] == [
            ("first", "information"),
            ("second", "warning"),
        ]
