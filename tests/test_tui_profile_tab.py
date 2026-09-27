"""The Profile tab, the profile line, the Apply and Save-as dialogs, and /profile."""

from __future__ import annotations

from enum import StrEnum
from unittest import mock

import pytest
from textual.app import ComposeResult

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmPill
from lilbee.core.config import cfg
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._lilbee_app_test_host import ready_services as _ready_services


@pytest.fixture(autouse=True)
def _gate_releases_at_once():
    with _ready_services():
        yield


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class _Pick(StrEnum):
    ONE = "one"


class _PillApp(LilbeeAppHost):
    def __init__(self) -> None:
        super().__init__()
        self.picked: list[object] = []

    def compose(self) -> ComposeResult:
        yield ConfirmPill("One", pill_id="pick-one", answer=_Pick.ONE)

    def on_confirm_pill_picked(self, event: ConfirmPill.Picked) -> None:
        self.picked.append(event.answer)


async def test_a_pill_carries_its_own_enum_answer() -> None:
    app = _PillApp()
    async with app.run_test() as pilot:
        app.query_one("#pick-one", ConfirmPill).focus()
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: bool(app.picked))
        assert app.picked == [_Pick.ONE]


async def test_publish_settings_tells_subscribers_each_key_and_its_value() -> None:
    app = LilbeeAppHost()
    seen: list[tuple[str, object]] = []
    async with app.run_test() as pilot:
        app.settings_changed_signal.subscribe(app, seen.append)
        cfg.chunk_size = 700
        cfg.top_k = 9
        app.publish_settings(["chunk_size", "top_k"])
        assert await wait_until(pilot, lambda: len(seen) == 2)
    assert seen == [("chunk_size", 700), ("top_k", 9)]


def test_start_rebuild_queues_a_forced_rebuild_on_the_chat_screen() -> None:
    app = LilbeeAppHost()
    chat = mock.MagicMock()
    with mock.patch.object(LilbeeAppHost, "chat_screen", return_value=chat):
        app.start_rebuild()
    chat.run_sync.assert_called_once_with(force_rebuild=True)


def test_start_rebuild_without_a_chat_screen_does_nothing() -> None:
    app = LilbeeAppHost()
    with mock.patch.object(LilbeeAppHost, "chat_screen", return_value=None) as chat_screen:
        app.start_rebuild()
    chat_screen.assert_called_once_with()
