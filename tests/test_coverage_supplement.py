"""Coverage supplement: small targeted tests for branches that the
broader pilot-driven suites did not exercise. These are unit-style
direct calls and minimal app harnesses that drive a specific code
path without spinning up a full TUI session.

Each test names what it covers in its docstring so future readers can
see why it exists.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path: Any) -> Any:
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


class TestStatusHelpers:
    """Branches in `_ocr_label`, `_ocr_pill`, `_data_dir_pill` that the
    other status tests didn't reach (auto / missing-dir paths)."""

    def test_ocr_label_auto_when_none(self) -> None:
        from lilbee.cli.tui.screens.status import _ocr_label

        cfg.enable_ocr = None  # type: ignore[assignment]
        assert _ocr_label() == "auto"

    def test_ocr_pill_auto_when_none(self) -> None:
        from lilbee.cli.tui.screens.status import _ocr_pill

        cfg.enable_ocr = None  # type: ignore[assignment]
        result = _ocr_pill()
        assert "auto" in str(result.plain)

    def test_data_dir_pill_missing_when_dir_absent(self, tmp_path: Any) -> None:
        from lilbee.cli.tui.screens.status import _data_dir_pill

        cfg.data_dir = tmp_path / "nonexistent"
        result = _data_dir_pill()
        assert "missing" in str(result.plain)


class TestSetupPendingDownload:
    """The isinstance-narrowing branches in `_pending_download` and
    `_preselect_recommended` for the unreachable-but-typesafe case
    where a card row is not a LocalCatalogRow."""

    def test_pending_download_returns_none_for_frontier_row(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow, KeyStatus
        from lilbee.cli.tui.screens.setup import _pending_download
        from lilbee.cli.tui.widgets.model_card import ModelCard

        row = FrontierCatalogRow(
            name="claude-test",
            ref="anthropic/claude-test",
            task="chat",
            provider="Anthropic",
            provider_id="anthropic",
            key_status=KeyStatus.READY,
        )
        card = ModelCard(row)
        assert _pending_download(card) is None

    def test_pending_download_returns_none_for_no_card(self) -> None:
        from lilbee.cli.tui.screens.setup import _pending_download

        assert _pending_download(None) is None


class TestSettingsLazyBody:
    """`_LazyGroupBody.populated` property and `_populate_pane` early
    returns for unknown pane ids."""

    def test_populated_property_initially_false(self) -> None:
        from lilbee.cli.tui.screens.settings import _LazyGroupBody

        body = _LazyGroupBody(id="probe-body")
        assert body.populated is False

    async def test_populate_pane_unknown_pane_id_is_noop(self) -> None:
        """`_populate_pane('does-not-exist')` returns immediately when the
        pane id is not in the screen's group map."""
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.settings import SettingsScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(SettingsScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, SettingsScreen)
            # No-op: should not raise even though the id is unknown.
            screen._populate_pane("settings-tab-does-not-exist")


class TestChatInputNewline:
    """`ChatInput.action_newline` inserts a newline character."""

    async def test_action_newline_inserts_newline(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.chat_input import ChatInput

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ChatInput(id="probe-input")

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            inp = pilot.app.query_one("#probe-input", ChatInput)
            inp.value = "abc"
            inp.move_cursor((0, 3))
            inp.action_newline()
            assert "\n" in inp.value


class TestChatInputCheckConsumeKey:
    """`ChatInput.check_consume_key` releases keys that App-level help binds."""

    async def test_question_mark_is_consumed_by_input(self) -> None:
        """``?`` lands as a literal character in the chat input.

        Help opens only via F1 / Ctrl+H while the input is focused, or
        via the non-priority App binding when no input is focused.
        """
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.chat_input import ChatInput

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ChatInput(id="probe-input")

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            inp = pilot.app.query_one("#probe-input", ChatInput)
            assert inp.check_consume_key("question_mark", "?") is True
            assert inp.check_consume_key("a", "a") is True


class TestStatusBarSwitch:
    """`ViewTab.action_activate` calls `_switch`; under a non-LilbeeApp
    test parent the isinstance gate skips ``switch_view`` but still
    executes the binding wiring (line 62)."""

    async def test_action_activate_runs_switch(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.status_bar import ViewTab

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ViewTab(msg.DEFAULT_VIEW)

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tab = pilot.app.query_one(ViewTab)
            # Should not raise; LilbeeApp isinstance gate skips
            # switch_view but ``_switch`` itself runs.
            tab.action_activate()
            tab.on_click()


class TestCatalogUtilsFrontierFromRemote:
    """`frontier_row_from_remote` converts a RemoteModel into a
    FrontierCatalogRow. Direct unit call covers the constructor."""

    def test_converts_remote_to_frontier_row(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import (
            FrontierCatalogRow,
            KeyStatus,
            frontier_row_from_remote,
        )
        from lilbee.modelhub.model_manager import RemoteModel
        from lilbee.modelhub.models import ModelTask

        rm = RemoteModel(
            name="gemini-test",
            provider="Gemini",
            task=ModelTask.CHAT,
            family="gemini",
            parameter_size="--",
        )
        row = frontier_row_from_remote(rm, provider_id="gemini", key_status=KeyStatus.READY)
        assert isinstance(row, FrontierCatalogRow)
        assert row.provider == "Gemini"
        assert row.key_status == KeyStatus.READY
        # ref must be the canonical provider/name form so it round-trips
        # through Config.chat_model's validator without a per-call-site fixup.
        assert row.ref == "gemini/gemini-test"


class TestRowDeleteId:
    """`row_delete_id` returns frontier ``ref`` directly, but for remote
    rows hands back the bare backend name (Ollama keys models by bare
    name, not the canonical ``ollama/<name>`` ref)."""

    def test_frontier_row_returns_canonical_ref(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import (
            FrontierCatalogRow,
            KeyStatus,
            row_delete_id,
        )

        row = FrontierCatalogRow(
            name="claude",
            ref="anthropic/claude",
            task="chat",
            provider="Anthropic",
            provider_id="anthropic",
            key_status=KeyStatus.READY,
        )
        assert row_delete_id(row) == "anthropic/claude"

    def test_ollama_remote_row_returns_bare_backend_name(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import remote_to_row, row_delete_id
        from lilbee.modelhub.model_manager import RemoteModel

        rm = RemoteModel(
            name="qwen3:0.6b",
            provider="Ollama",
            task="chat",
            family="qwen",
            parameter_size="0.6B",
        )
        row = remote_to_row(rm)
        # ref carries the canonical chat_model form; delete uses the bare name.
        assert row.ref == "ollama/qwen3:0.6b"
        assert row_delete_id(row) == "qwen3:0.6b"


class TestCatalogVimNavListView:
    """List-view branches of action_cursor_*, action_page_*, and
    action_jump_* that the existing grid-view tests don't reach."""

    async def test_list_view_cursor_down_with_no_focus(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        with (
            mock.patch("lilbee.cli.tui.screens.catalog.classify_remote_models", return_value=[]),
            mock.patch(
                "lilbee.cli.tui.screens.catalog.get_catalog",
                return_value=mock.MagicMock(models=[], total=0, has_more=False),
            ),
        ):
            async with _Probe().run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = pilot.app.screen
                assert isinstance(screen, CatalogScreen)
                screen._grid_view = False  # switch to list view path
                # Exercises the else branch of cursor_down / cursor_up /
                # page_down / page_up / jump_top / jump_bottom.
                screen.action_cursor_down()
                screen.action_cursor_up()
                screen.action_page_down()
                screen.action_page_up()
                screen.action_jump_top()
                screen.action_jump_bottom()


class TestTaskCenterOnHide:
    """`on_hide` unsubscribes from the queue. The contextlib.suppress
    catches when the queue is no longer accessible (test apps tearing
    down)."""

    async def test_on_hide_unsubscribes_cleanly(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.task_center import TaskCenter
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        class _Probe(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.task_bar = TaskBarController(self)

            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(TaskCenter())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, TaskCenter)
            # on_hide is invoked when the screen is dismissed; calling it
            # directly exercises the unsubscribe path including the
            # contextlib.suppress block.
            screen.on_hide()


class TestSettingsFeatureGating:
    """`_group_settings()` hides API-Keys / Crawling / Wiki groups when
    the corresponding feature is not available."""

    def test_wiki_group_hidden_when_cfg_wiki_off(self) -> None:
        from lilbee.cli.tui.screens.settings import _group_settings

        cfg.wiki = False
        groups = _group_settings()
        assert "Wiki" not in groups

    def test_wiki_group_visible_when_cfg_wiki_on(self) -> None:
        from lilbee.cli.tui.screens.settings import _group_settings

        cfg.wiki = True
        groups = _group_settings()
        assert "Wiki" in groups

    def test_api_keys_group_hidden_without_litellm(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch.object(settings_mod, "_litellm_installed", return_value=False):
            groups = settings_mod._group_settings()
        assert "API-Keys" not in groups

    def test_api_keys_group_visible_with_litellm(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch.object(settings_mod, "_litellm_installed", return_value=True):
            groups = settings_mod._group_settings()
        assert "API-Keys" in groups

    def test_crawling_group_hidden_without_crawler(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch.object(settings_mod, "_crawler_installed", return_value=False):
            groups = settings_mod._group_settings()
        assert "Crawling" not in groups

    def test_crawling_group_visible_with_crawler(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch.object(settings_mod, "_crawler_installed", return_value=True):
            groups = settings_mod._group_settings()
        assert "Crawling" in groups


class TestSettingsTabActivatedEdges:
    """`_on_tab_activated` early-returns when pane / pane.id is None."""

    async def test_tab_activated_with_no_pane_id_is_noop(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.settings import SettingsScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(SettingsScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, SettingsScreen)
            # Construct the TabActivated payload via mocks; the handler
            # only reads .pane (and pane.id), so a mock with pane=None
            # exercises the early-return without round-tripping through
            # Textual's bubble-up dispatch.
            event = mock.MagicMock()
            event.pane = None
            screen._on_tab_activated(event)
            # And a pane whose id is None.
            event2 = mock.MagicMock()
            event2.pane = mock.MagicMock(id=None)
            screen._on_tab_activated(event2)


class TestStatusArchWorkerError:
    """`_fetch_arch_worker` swallows exceptions and returns an empty
    ModelArchInfo (lines 196-198)."""

    def test_arch_worker_swallows_error(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen
        from lilbee.modelhub.model_info import ModelArchInfo

        screen = StatusScreen.__new__(StatusScreen)
        with mock.patch(
            "lilbee.cli.tui.screens.status.get_model_architecture",
            side_effect=RuntimeError("disk error"),
        ):
            # Decorator wraps the method; access the underlying callable.
            result = screen._fetch_arch_worker.__wrapped__(screen)  # type: ignore[attr-defined]
        assert isinstance(result, ModelArchInfo)


class TestAppCanonicalizeFallbackNotice:
    """`LilbeeApp._canonicalize_persisted_models` setattrs a fallback
    when canonicalize returns a different effective ref (lines 210-211)."""

    async def test_fallback_notice_fires_when_effective_differs(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.modelhub.model_manager import (
            CanonicalRef,
            ValidationResult,
        )

        app = LilbeeApp()
        # Stub canonicalize_*_model so canon.original != canon.effective
        # and status != OK -- triggers the setattr branch.
        chat_canon = CanonicalRef(
            original="missing/model",
            effective="fallback/model",
            status=ValidationResult.NOT_INSTALLED,
        )
        embed_canon = CanonicalRef(
            original="missing/embed",
            effective="missing/embed",
            status=ValidationResult.OK,
        )
        notifications: list[Any] = []
        # Patch where the symbols actually live; the app imports them
        # function-locally so we cannot patch on lilbee.cli.tui.app.
        with (
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_chat_model",
                return_value=chat_canon,
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_embedding_model",
                return_value=embed_canon,
            ),
            mock.patch.object(app, "notify", side_effect=lambda *a, **kw: notifications.append(a)),
        ):
            app._canonicalize_persisted_models()
        # The chat canon's effective gets written to cfg; notify was called
        # with the fallback message.
        assert notifications, "expected a fallback notification"


class TestCatalogToggleViewWhileSwitching:
    """`action_toggle_view` early-returns when `_view_switching` is True
    (line 267)."""

    def test_re_entry_during_switch_is_noop(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        screen._view_switching = True
        screen._grid_view = True
        # Should return without raising.
        screen.action_toggle_view()


class TestCatalogSelectFrontierRow:
    """`_select_frontier_row` READY path applies the model; MISSING_KEY
    path notifies and switches to Settings."""

    async def test_select_frontier_ready_applies_model(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow, KeyStatus
        from lilbee.core.config import cfg

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        # No apply_active_model mock: the canonical ref must round-trip
        # through Config.chat_model's validator (a bare ref would raise
        # and regress b3a36798).
        with (
            mock.patch("lilbee.cli.tui.screens.catalog.classify_remote_models", return_value=[]),
            mock.patch(
                "lilbee.cli.tui.screens.catalog.get_catalog",
                return_value=mock.MagicMock(models=[], total=0, has_more=False),
            ),
        ):
            async with _Probe().run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = pilot.app.screen
                assert isinstance(screen, CatalogScreen)
                row = FrontierCatalogRow(
                    name="gemini-2.0-flash",
                    ref="gemini/gemini-2.0-flash",
                    task="chat",
                    provider="Gemini",
                    provider_id="gemini",
                    key_status=KeyStatus.READY,
                )
                screen._select_frontier_row(row)
                assert cfg.chat_model == "gemini/gemini-2.0-flash"

    async def test_select_frontier_missing_key_notifies(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow, KeyStatus

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        with (
            mock.patch("lilbee.cli.tui.screens.catalog.classify_remote_models", return_value=[]),
            mock.patch(
                "lilbee.cli.tui.screens.catalog.get_catalog",
                return_value=mock.MagicMock(models=[], total=0, has_more=False),
            ),
        ):
            async with _Probe().run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = pilot.app.screen
                assert isinstance(screen, CatalogScreen)
                notifications: list[Any] = []
                screen.notify = lambda *a, **kw: notifications.append(a)  # type: ignore[method-assign]
                row = FrontierCatalogRow(
                    name="gpt-4",
                    ref="openai/gpt-4",
                    task="chat",
                    provider="OpenAI",
                    provider_id="openai",
                    key_status=KeyStatus.MISSING_KEY,
                )
                screen._select_frontier_row(row)
                assert notifications, "missing-key path must notify the user"


class TestCatalogProviderAvailabilityDebounce:
    """`_on_provider_availability_changed` (re)arms a debounce timer."""

    async def test_signal_arms_timer(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        with (
            mock.patch("lilbee.cli.tui.screens.catalog.classify_remote_models", return_value=[]),
            mock.patch(
                "lilbee.cli.tui.screens.catalog.get_catalog",
                return_value=mock.MagicMock(models=[], total=0, has_more=False),
            ),
        ):
            async with _Probe().run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = pilot.app.screen
                assert isinstance(screen, CatalogScreen)
                # First call: timer is None, set it.
                screen._on_provider_availability_changed(("openai_api_key", "x"))
                first_timer = screen._frontier_refresh_timer
                # Second call: existing timer is stopped and replaced.
                screen._on_provider_availability_changed(("openai_api_key", "y"))
                second_timer = screen._frontier_refresh_timer
                assert first_timer is not second_timer


class TestSettingsPopulatePaneBodyMissing:
    """`_populate_pane` swallows the NoMatches when the body widget
    isn't mounted yet (lines 392-394)."""

    async def test_pane_with_no_body_swallows_query_error(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.settings import SettingsScreen, _PaneGroup

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(SettingsScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, SettingsScreen)
            # Inject a fake pane group whose body widget id won't resolve.
            screen._pane_groups["settings-tab-fake"] = _PaneGroup(
                pane_id="settings-tab-fake", group_name="Fake", items=[]
            )
            screen._populate_pane("settings-tab-fake")


class TestWorkerNotificationMessages:
    """Helpers for spawn-lifecycle notifications route through messages.py."""

    def test_worker_starting_title_cases_role_name(self) -> None:
        from lilbee.cli.tui.messages import worker_starting

        assert worker_starting("chat") == "Starting Chat worker..."
        assert worker_starting("vision") == "Starting Vision worker..."

    def test_worker_starting_handles_underscored_role(self) -> None:
        from lilbee.cli.tui.messages import worker_starting

        assert worker_starting("vector_embed") == "Starting Vector Embed worker..."

    def test_worker_ready_title_cases_role_name(self) -> None:
        from lilbee.cli.tui.messages import worker_ready

        assert worker_ready("rerank") == "Rerank worker ready"
        assert worker_ready("chat_v2") == "Chat V2 worker ready"


class TestServicesPoolListener:
    """``Services.add_pool_listener`` forwards to the underlying WorkerPool."""

    def test_forwards_both_callbacks_to_pool(self) -> None:
        from tests.conftest import make_mock_services

        seen_spawning: list[str] = []
        seen_spawned: list[str] = []

        class _RecordingPool:
            registered_roles: tuple[str, ...] = ()

            def add_listener(self, *, on_spawning=None, on_spawned=None) -> None:
                # Re-fire with a synthetic role to verify both callbacks routed.
                if on_spawning is not None:
                    on_spawning("embed")
                    seen_spawning.append("embed")
                if on_spawned is not None:
                    on_spawned("embed")
                    seen_spawned.append("embed")

        services = make_mock_services(worker_pool=_RecordingPool())
        services.add_pool_listener(
            on_spawning=lambda _r: None,
            on_spawned=lambda _r: None,
        )
        assert seen_spawning == ["embed"]
        assert seen_spawned == ["embed"]

    def test_either_callback_is_optional(self) -> None:
        from tests.conftest import make_mock_services

        captured: dict[str, object] = {}

        class _CapturingPool:
            registered_roles: tuple[str, ...] = ()

            def add_listener(self, *, on_spawning=None, on_spawned=None) -> None:
                captured["on_spawning"] = on_spawning
                captured["on_spawned"] = on_spawned

        services = make_mock_services(worker_pool=_CapturingPool())
        services.add_pool_listener(on_spawning=lambda _r: None)
        assert captured["on_spawning"] is not None
        assert captured["on_spawned"] is None
