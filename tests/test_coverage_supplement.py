"""Coverage supplement: small targeted tests for branches that the
broader pilot-driven suites did not exercise. These are unit-style
direct calls and minimal app harnesses that drive a specific code
path without spinning up a full TUI session.

Each test names what it covers in its docstring so future readers can
see why it exists.
"""

from __future__ import annotations

import logging
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

        with mock.patch("lilbee.providers.litellm_sdk.litellm_available", return_value=False):
            groups = settings_mod._group_settings()
        assert "API-Keys" not in groups

    def test_api_keys_group_visible_with_litellm(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch("lilbee.providers.litellm_sdk.litellm_available", return_value=True):
            groups = settings_mod._group_settings()
        assert "API-Keys" in groups

    def test_crawling_group_hidden_without_crawler(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch("lilbee.crawler.crawler_available", return_value=False):
            groups = settings_mod._group_settings()
        assert "Crawling" not in groups

    def test_crawling_group_visible_with_crawler(self) -> None:
        from lilbee.cli.tui.screens import settings as settings_mod

        with mock.patch("lilbee.crawler.crawler_available", return_value=True):
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
    when canonicalize returns a different effective ref."""

    async def test_fallback_writes_cfg_and_logs_silently(self, caplog) -> None:
        """Fallback writes cfg.<field> = effective, logs a WARNING, and does not toast the user."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.core.config import cfg
        from lilbee.modelhub.model_manager import (
            CanonicalRef,
            ValidationResult,
        )

        app = LilbeeApp()
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
        snapshot_chat = cfg.chat_model
        try:
            with (
                mock.patch(
                    "lilbee.modelhub.model_manager.canonicalize_chat_model",
                    return_value=chat_canon,
                ),
                mock.patch(
                    "lilbee.modelhub.model_manager.canonicalize_embedding_model",
                    return_value=embed_canon,
                ),
                mock.patch.object(
                    app, "notify", side_effect=lambda *a, **kw: notifications.append(a)
                ),
                caplog.at_level(logging.WARNING, logger="lilbee.cli.tui.app"),
            ):
                app._canonicalize_persisted_models()
            assert cfg.chat_model == "fallback/model"
            assert not notifications, "fallback must not toast the user"
            assert any("fallback/model" in record.getMessage() for record in caplog.records), (
                "fallback must be logged at WARNING for diagnosis"
            )
        finally:
            cfg.chat_model = snapshot_chat


class TestCatalogToggleViewWhileSwitching:
    """`action_toggle_view` early-returns when `_view_switching` is True
    (line 267)."""

    def test_re_entry_during_switch_is_noop(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
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


class TestModelInfoModal:
    """`ModelInfoModal` renders a markdown body with what we know about a row."""

    def _row(self, **kw: Any) -> Any:
        from lilbee.catalog import CatalogModel
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow

        defaults: dict[str, Any] = {
            "name": "Acme 1B",
            "task": "chat",
            "params": "1B",
            "size": "700 MB",
            "quant": "Q8_0",
            "downloads": "42",
            "featured": False,
            "installed": False,
            "sort_downloads": 42,
            "sort_size": 0.7,
            "ref": "acme/acme-1b-gguf",
            "catalog_model": CatalogModel(
                hf_repo="acme/acme-1b-gguf",
                gguf_filename="acme-1b-q8.gguf",
                size_gb=0.7,
                min_ram_gb=2.0,
                description="A small chat model.",
                featured=False,
                downloads=42,
                task="chat",
            ),
        }
        defaults.update(kw)
        return LocalCatalogRow(**defaults)

    async def test_modal_compose_and_markdown_includes_known_fields(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.model_info import ModelInfoModal

        row = self._row(installed=True)
        modal = ModelInfoModal(row)

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(modal)

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            md = modal._build_markdown()
            for needle in (
                "A small chat model.",
                "**Task:** chat",
                "**Parameters:** 1B",
                "**Download size:** 700 MB",
                "**Recommended RAM:** 2 GB",
                "**Quantization:** Q8_0",
                "**Downloads:** 42",
                "**Status:** installed",
                "**GGUF file:** `acme-1b-q8.gguf`",
                "huggingface.co/acme/acme-1b-gguf",
            ):
                assert needle in md, f"missing {needle!r}: {md}"
            await pilot.press("escape")
            await pilot.pause()
            assert pilot.app.screen is not modal, "escape should dismiss modal"

    def test_markdown_drops_optional_lines_when_fields_empty(self) -> None:
        from lilbee.cli.tui.screens.model_info import ModelInfoModal

        row = self._row(
            params="",
            size="",
            quant="",
            downloads="",
            installed=False,
            catalog_model=None,
        )
        md = ModelInfoModal(row)._build_markdown()
        assert "**Task:** chat" in md
        assert "**Parameters:**" not in md
        assert "**Download size:**" not in md
        assert "**Quantization:**" not in md
        assert "**Downloads:**" not in md
        assert "**Status:**" not in md
        assert "**Recommended RAM:**" not in md
        assert "**GGUF file:**" not in md
        assert "huggingface.co/acme/acme-1b-gguf" in md


class TestCatalogActionShowInfoEarlyReturns:
    """`action_show_info` early-return paths (Input-focused, no row)."""

    def test_input_focused_does_nothing(self) -> None:
        """Direct call: when `self.focused` is an Input, action_show_info bails."""
        from textual.widgets import Input

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        fake_input = mock.MagicMock(spec=Input)
        with (
            mock.patch.object(
                CatalogScreen,
                "focused",
                new_callable=mock.PropertyMock,
                return_value=fake_input,
            ),
            mock.patch.object(screen, "_highlighted_row") as mock_row,
            mock.patch.object(screen, "notify") as mock_notify,
        ):
            screen.action_show_info()
        mock_row.assert_not_called()
        mock_notify.assert_not_called()


class TestCatalogHighlightedRow:
    """`_highlighted_row` covers list-view, no-grid, ModelGrid, GridSelect paths."""

    async def test_list_view_with_focused_list_returns_highlighted_row(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            screen._grid_view = False
            sentinel = mock.Mock(name="row")
            with (
                mock.patch.object(
                    type(screen._list_widget),
                    "has_focus",
                    new_callable=mock.PropertyMock,
                    return_value=True,
                ),
                mock.patch.object(screen._list_widget, "highlighted_row", return_value=sentinel),
            ):
                assert screen._highlighted_row() is sentinel

    async def test_grid_view_no_focused_grid_returns_none(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            with mock.patch.object(screen, "_focused_grid", return_value=None):
                assert screen._highlighted_row() is None

    async def test_model_grid_returns_row_at_index(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        row = LocalCatalogRow(
            name="X",
            task="chat",
            params="1B",
            size="500 MB",
            quant="Q4",
            downloads="1",
            featured=False,
            installed=False,
            sort_downloads=1,
            sort_size=0.5,
            ref="x/x",
        )

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            fake_grid = mock.MagicMock(spec=ModelGrid)
            fake_grid.highlighted = 0
            fake_grid.rows = [row]
            with mock.patch.object(screen, "_focused_grid", return_value=fake_grid):
                assert screen._highlighted_row() is row
            # Out-of-bounds index returns None.
            fake_grid.highlighted = 7
            with mock.patch.object(screen, "_focused_grid", return_value=fake_grid):
                assert screen._highlighted_row() is None

    async def test_grid_select_returns_row_when_child_is_model_card(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
        from lilbee.cli.tui.widgets.model_card import ModelCard

        row = LocalCatalogRow(
            name="Y",
            task="chat",
            params="1B",
            size="500 MB",
            quant="Q4",
            downloads="1",
            featured=False,
            installed=False,
            sort_downloads=1,
            sort_size=0.5,
            ref="y/y",
        )

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            fake_card = mock.MagicMock(spec=ModelCard)
            fake_card.row = row
            fake_grid = mock.MagicMock()
            fake_grid.highlighted = 0
            fake_grid.children = [fake_card]
            # Force isinstance(focused_grid, ModelGrid) → False so the
            # GridSelect branch runs.
            with mock.patch.object(screen, "_focused_grid", return_value=fake_grid):
                assert screen._highlighted_row() is row

    async def test_grid_select_non_model_card_returns_none(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            non_card = mock.MagicMock()
            fake_grid = mock.MagicMock()
            fake_grid.highlighted = 0
            fake_grid.children = [non_card]
            with mock.patch.object(screen, "_focused_grid", return_value=fake_grid):
                assert screen._highlighted_row() is None


class TestCatalogActionShowInfoFrontierWarn:
    """`action_show_info` warns when the highlighted row is a frontier row."""

    async def test_frontier_row_emits_warning_toast(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.catalog_utils import (
            FrontierCatalogRow,
            KeyStatus,
        )

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            row = FrontierCatalogRow(
                name="Gemini 2.5 Pro",
                ref="gemini/2.5-pro",
                task="chat",
                provider="Gemini",
                provider_id="gemini",
                key_status=KeyStatus.READY,
            )
            with (
                mock.patch.object(screen, "_highlighted_row", return_value=row),
                mock.patch.object(screen, "notify") as mock_notify,
            ):
                screen.action_show_info()
                mock_notify.assert_called_once()
                assert "downloadable" in mock_notify.call_args[0][0]


class TestSettingsTabNavFallbacks:
    """`_move_focus_within_pane` and `_focus_pane_edge` fallback branches."""

    async def test_move_focus_falls_back_when_body_missing(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer, TabbedContent

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
            # Pretend the active pane id has no body widget.
            tabs = screen.query_one(TabbedContent)
            tabs.active = ""
            with mock.patch.object(screen.app, "action_focus_next") as mock_next:
                screen.action_next_field_or_pane()
                mock_next.assert_called_once()
            with mock.patch.object(screen.app, "action_focus_previous") as mock_prev:
                screen.action_prev_field_or_pane()
                mock_prev.assert_called_once()

    async def test_move_focus_falls_back_when_focused_outside_pane(self) -> None:
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
            screen.populate_all_panes()
            await pilot.pause()
            with mock.patch.object(screen.app, "action_focus_next") as mock_next:
                # focused is None at this point: walks the fallback path.
                screen.action_next_field_or_pane()
                mock_next.assert_called_once()

    async def test_move_focus_returns_when_active_pane_unknown(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer, TabbedContent

        from lilbee.cli.tui.screens.settings import (
            SettingsScreen,
            _LazyGroupBody,
        )

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(SettingsScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, SettingsScreen)
            screen.populate_all_panes()
            await pilot.pause()
            tabs = screen.query_one(TabbedContent)
            active_pane_id = tabs.active
            body = screen.query_one(f"#{active_pane_id}-body", _LazyGroupBody)
            focusables = [w for w in body.query("*") if w.focusable]
            assert focusables
            focusables[-1].focus()
            await pilot.pause()
            # Drop the active pane from the screen's bookkeeping so the
            # boundary path hits the early-return guard.
            screen._pane_groups = {}
            screen.action_next_field_or_pane()
            await pilot.pause()
            # Active pane stays put because the boundary guard returned early.
            assert tabs.active == active_pane_id

    async def test_focus_pane_edge_handles_missing_body(self) -> None:
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
            # Unknown pane id: query_one raises NoMatches, swallowed.
            screen._focus_pane_edge("settings-tab-does-not-exist", direction=1)

    async def test_focus_pane_edge_no_focusables_returns(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.settings import (
            SettingsScreen,
            _LazyGroupBody,
        )

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(SettingsScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, SettingsScreen)
            # Mount a fresh empty body that lives in the DOM but has no
            # focusable descendants, then exercise the edge focus helper.
            empty_body = _LazyGroupBody(id="settings-tab-empty-body")
            screen.mount(empty_body)
            await pilot.pause()
            screen._focus_pane_edge("settings-tab-empty", direction=-1)


class TestSyncSkippedMessageBranches:
    """`sync_skipped_message` returns vision-failed vs no-vision text."""

    def test_returns_vision_failed_when_vision_model_set(self) -> None:
        from lilbee.cli.tui.messages import sync_skipped_message

        cfg.vision_model = "stub/vision"
        assert "vision OCR returned no text" in sync_skipped_message("a.pdf")

    def test_returns_no_vision_when_vision_model_unset(self) -> None:
        from lilbee.cli.tui.messages import sync_skipped_message

        cfg.vision_model = ""
        assert "Configure a vision_model" in sync_skipped_message("a.pdf")


class TestWikiEmptyStateSpacyBranches:
    """Wiki-empty-state messages have a spaCy-unavailable branch."""

    def test_wiki_empty_state_leaf_when_spacy_missing(self) -> None:
        from lilbee.cli.tui import messages as mod
        from lilbee.cli.tui.messages import (
            WIKI_EMPTY_NEEDS_SPACY_LEAF,
            wiki_empty_state_leaf,
        )

        with mock.patch.object(mod, "_spacy_available", return_value=False):
            assert wiki_empty_state_leaf() == WIKI_EMPTY_NEEDS_SPACY_LEAF

    def test_wiki_empty_state_detail_when_spacy_missing(self) -> None:
        from lilbee.cli.tui import messages as mod
        from lilbee.cli.tui.messages import (
            WIKI_EMPTY_NEEDS_SPACY_DETAIL,
            wiki_empty_state_detail,
        )

        with mock.patch.object(mod, "_spacy_available", return_value=False):
            assert wiki_empty_state_detail() == WIKI_EMPTY_NEEDS_SPACY_DETAIL

    def test_spacy_available_returns_false_on_import_error(self) -> None:
        from lilbee.cli.tui.messages import _spacy_available

        with mock.patch(
            "lilbee.retrieval.concepts.nlp.load_spacy_pipeline",
            side_effect=ImportError,
        ):
            assert _spacy_available() is False

    def test_spacy_available_returns_true_on_other_exception(self) -> None:
        from lilbee.cli.tui.messages import _spacy_available

        with mock.patch(
            "lilbee.retrieval.concepts.nlp.load_spacy_pipeline",
            side_effect=RuntimeError,
        ):
            assert _spacy_available() is True

    def test_spacy_available_returns_true_on_success(self) -> None:
        from lilbee.cli.tui.messages import _spacy_available

        with mock.patch(
            "lilbee.retrieval.concepts.nlp.load_spacy_pipeline",
            return_value=mock.MagicMock(),
        ):
            assert _spacy_available() is True

    def test_wiki_empty_state_leaf_when_spacy_present(self) -> None:
        from lilbee.cli.tui import messages as mod
        from lilbee.cli.tui.messages import WIKI_EMPTY_STATE, wiki_empty_state_leaf

        with mock.patch.object(mod, "_spacy_available", return_value=True):
            assert wiki_empty_state_leaf() == WIKI_EMPTY_STATE

    def test_wiki_empty_state_detail_when_spacy_present(self) -> None:
        from lilbee.cli.tui import messages as mod
        from lilbee.cli.tui.messages import WIKI_NO_CONTENT, wiki_empty_state_detail

        with mock.patch.object(mod, "_spacy_available", return_value=True):
            assert wiki_empty_state_detail() == WIKI_NO_CONTENT


class TestChatInputUnconsumedKey:
    """`ChatInput.check_consume_key` releases keys named in _UNCONSUMED_KEYS."""

    async def test_unconsumed_key_returns_false(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.chat_input import ChatInput

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ChatInput(id="probe")

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            inp = pilot.app.query_one("#probe", ChatInput)
            with mock.patch.object(ChatInput, "_UNCONSUMED_KEYS", new=frozenset({"f1"})):
                assert inp.check_consume_key("f1", None) is False
                # Other keys still consume normally.
                assert inp.check_consume_key("a", "a") is True


class TestModelCardTruncate:
    """`_truncate_name` shortens names longer than the visible budget."""

    def test_long_name_is_truncated(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _NAME_MAX_CHARS, _truncate_name

        long_name = "x" * (_NAME_MAX_CHARS + 5)
        out = _truncate_name(long_name)
        assert len(out) == _NAME_MAX_CHARS


class TestModelGridTruncateAndPad:
    """The model_grid module has its own _truncate_name + render padding."""

    def test_grid_truncate_name_long(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _NAME_MAX_CHARS, _truncate_name

        long_name = "x" * (_NAME_MAX_CHARS + 5)
        out = _truncate_name(long_name)
        assert len(out) == _NAME_MAX_CHARS

    def test_render_card_strip_pads_short_body(self) -> None:
        """Force the render path that pads body lines up to _CARD_BODY_HEIGHT."""
        from textual.content import Content

        from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow, KeyStatus
        from lilbee.cli.tui.widgets import model_grid as mg

        row = FrontierCatalogRow(
            name="api",
            ref="acme/api",
            task="chat",
            provider="Acme",
            provider_id="acme",
            key_status=KeyStatus.READY,
        )
        # 1-line body forces the pad loop to run _CARD_BODY_HEIGHT-1 times.
        with mock.patch.object(mg, "_frontier_lines", return_value=[Content("hi")]):
            out = mg._render_card_strip(row, selected=False, width=20, border_style="dim")
        assert out.lines, "expected card lines"

    def test_cell_at_returns_none_in_gutter_row(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
        from lilbee.cli.tui.widgets import model_grid as mg
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        row = LocalCatalogRow(
            name="m",
            task="chat",
            params="",
            size="",
            quant="",
            downloads="",
            featured=False,
            installed=False,
            sort_downloads=0,
            sort_size=0.0,
            ref="m/m",
        )
        grid = ModelGrid(rows=[row])
        grid._cards_per_row = 1
        # _ROW_GUTTER is currently 0; bump _ROW_HEIGHT for this test so a y
        # past the card body lands in a gutter strip and the >=
        # _CARD_HEIGHT guard triggers.
        with mock.patch.object(mg, "_ROW_HEIGHT", new=mg._CARD_HEIGHT + 2):
            assert grid._cell_at(0, mg._CARD_HEIGHT) is None


class TestChatScreenFocusBranches:
    """`ChatScreen` on_show normal-mode focus + chat input focus event."""

    def test_on_show_normal_mode_focuses_chat_log(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = False
        screen._chat_log = mock.MagicMock()
        screen.refresh_model_bar = mock.MagicMock()  # type: ignore[method-assign]
        with mock.patch("lilbee.runtime.splash.dismiss"):
            screen.on_show()
        screen._chat_log.focus.assert_called_once()

    def test_on_show_insert_mode_re_enters_input(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = True
        screen.refresh_model_bar = mock.MagicMock()  # type: ignore[method-assign]
        screen._enter_insert_mode = mock.MagicMock()  # type: ignore[method-assign]
        with mock.patch("lilbee.runtime.splash.dismiss"):
            screen.on_show()
        screen._enter_insert_mode.assert_called_once()

    def test_chat_input_focus_event_flips_to_insert(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = False
        screen._enter_insert_mode = mock.MagicMock()  # type: ignore[method-assign]
        screen._on_chat_input_focused(mock.MagicMock())
        screen._enter_insert_mode.assert_called_once()

    def test_chat_input_click_in_normal_mode_enters_insert(self) -> None:
        from textual import events

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = False
        screen._enter_insert_mode = mock.MagicMock()  # type: ignore[method-assign]
        click = mock.MagicMock(spec=events.Click)
        screen._on_chat_input_clicked(click)
        screen._enter_insert_mode.assert_called_once()
        click.stop.assert_called_once()

    def test_chat_input_click_in_insert_mode_is_noop(self) -> None:
        from textual import events

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = True
        screen._enter_insert_mode = mock.MagicMock()  # type: ignore[method-assign]
        click = mock.MagicMock(spec=events.Click)
        screen._on_chat_input_clicked(click)
        screen._enter_insert_mode.assert_not_called()

    def test_click_outside_chat_input_in_insert_mode_returns_to_normal(self) -> None:
        from textual import events

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = True
        chat_input = mock.MagicMock()
        chat_input.parent = None
        outside = mock.MagicMock()
        outside.parent = None
        with (
            mock.patch.object(
                ChatScreen,
                "_chat_input",
                new_callable=mock.PropertyMock,
                return_value=chat_input,
            ),
            mock.patch.object(screen, "action_enter_normal_mode") as mock_normal,
        ):
            click = mock.MagicMock(spec=events.Click)
            click.widget = outside
            screen.on_click(click)
        mock_normal.assert_called_once()

    def test_click_inside_chat_input_in_insert_mode_stays_insert(self) -> None:
        from textual import events

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = True
        chat_input = mock.MagicMock()
        chat_input.parent = None
        # Click target is a descendant of chat_input.
        descendant = mock.MagicMock()
        descendant.parent = chat_input
        with (
            mock.patch.object(
                ChatScreen,
                "_chat_input",
                new_callable=mock.PropertyMock,
                return_value=chat_input,
            ),
            mock.patch.object(screen, "action_enter_normal_mode") as mock_normal,
        ):
            click = mock.MagicMock(spec=events.Click)
            click.widget = descendant
            screen.on_click(click)
        mock_normal.assert_not_called()

    def test_click_in_normal_mode_is_noop(self) -> None:
        from textual import events

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = False
        with mock.patch.object(screen, "action_enter_normal_mode") as mock_normal:
            click = mock.MagicMock(spec=events.Click)
            click.widget = mock.MagicMock()
            screen.on_click(click)
        mock_normal.assert_not_called()

    def test_click_with_no_widget_target_is_noop(self) -> None:
        from textual import events

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._insert_mode = True
        with mock.patch.object(screen, "action_enter_normal_mode") as mock_normal:
            click = mock.MagicMock(spec=events.Click)
            click.widget = None
            screen.on_click(click)
        mock_normal.assert_not_called()


class TestChatModeToggleAction:
    """`ChatModePill.action_select` switches the parent toggle's mode."""

    async def test_pill_action_select_routes_to_toggle(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_mode = "search"

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ChatModeToggle()

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            toggle = pilot.app.query_one(ChatModeToggle)
            chat_pill = toggle.query_one("#chat-mode-chat")
            chat_pill.action_select()
            await pilot.pause()
            assert cfg.chat_mode == "chat"

    async def test_pill_action_select_returns_when_no_toggle(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.model_bar import ChatModePill

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ChatModePill("Chat", id="chat-mode-chat")

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            pill = pilot.app.query_one("#chat-mode-chat", ChatModePill)
            pill.action_select()


class TestModelBarVisionSidecarPicker:
    """`classify_installed_models_full` drops a vision-sidecar chat model into VISION too."""

    def test_chat_model_with_vision_sidecar_appears_in_vision_bucket(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace

        from lilbee.cli.tui.widgets.model_bar import classify_installed_models_full
        from lilbee.modelhub.models import ModelTask

        manifest = SimpleNamespace(
            ref="acme/chat-with-vision",
            hf_repo="acme/chat-with-vision",
            gguf_filename="model-Q8_0.gguf",
            task=ModelTask.CHAT.value,
        )
        registry = mock.MagicMock()
        registry.list_installed.return_value = [manifest]
        registry.resolve.return_value = Path("/tmp/fake/chat.gguf")
        with (
            mock.patch(
                "lilbee.modelhub.registry.ModelRegistry",
                return_value=registry,
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.discovery.reclassify_by_name",
                return_value=ModelTask.CHAT,
            ),
            mock.patch(
                "pathlib.Path.glob",
                return_value=[Path("/tmp/fake/mmproj-vision.gguf")],
            ),
        ):
            buckets = classify_installed_models_full()
        assert any(opt.ref == manifest.ref for opt in buckets[ModelTask.VISION])


class TestScopeChipPillNoChipReturns:
    """`ScopePill.action_select` returns when no ScopeChip ancestor exists."""

    async def test_orphaned_pill_select_returns_silently(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.scope_chip import ScopePill

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ScopePill("docs", id="scope-pill-both")

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            pill = pilot.app.query_one("#scope-pill-both", ScopePill)
            pill.action_select()


class TestProviderProtocolBranches:
    """Each persistent-pool wrapper raises ProviderError when the worker returns the wrong type."""

    def _provider(self) -> Any:
        from lilbee.providers.llama_cpp.provider import LlamaCppProvider

        return LlamaCppProvider()

    def test_embed_protocol_error(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._provider()
        accessor = mock.MagicMock()
        runtime = mock.MagicMock()
        runtime.run_sync = mock.MagicMock(return_value="not-a-list")
        with (
            mock.patch.object(provider, "_get_pool_accessor", return_value=accessor),
            mock.patch.object(provider, "_pool_runtime", return_value=runtime),
            pytest.raises(ProviderError),
        ):
            provider.embed(["text"])

    def test_rerank_protocol_error(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._provider()
        accessor = mock.MagicMock()
        runtime = mock.MagicMock()
        runtime.run_sync = mock.MagicMock(return_value="not-a-list")
        with (
            mock.patch.object(provider, "_get_pool_accessor", return_value=accessor),
            mock.patch.object(provider, "_pool_runtime", return_value=runtime),
            pytest.raises(ProviderError),
        ):
            provider.rerank("q", ["a", "b"])

    def test_vision_ocr_protocol_error(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._provider()
        accessor = mock.MagicMock()
        runtime = mock.MagicMock()
        runtime.run_sync = mock.MagicMock(return_value=42)
        with (
            mock.patch.object(provider, "_get_pool_accessor", return_value=accessor),
            mock.patch.object(provider, "_pool_runtime", return_value=runtime),
            pytest.raises(ProviderError),
        ):
            provider.vision_ocr(b"png", "ref")

    def test_chat_protocol_error(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._provider()
        accessor = mock.MagicMock()
        runtime = mock.MagicMock()
        runtime.run_sync = mock.MagicMock(return_value=42)
        with (
            mock.patch.object(provider, "_get_pool_accessor", return_value=accessor),
            mock.patch.object(provider, "_pool_runtime", return_value=runtime),
            pytest.raises(ProviderError),
        ):
            provider.chat(messages=[{"role": "user", "content": "hi"}])


class TestCatalogPriorScrollAndPrefetchEdges:
    """Catalog edges around prior_scroll_y and prefetch ValueError/empty grid guards."""

    def test_first_grid_or_none_returns_none_when_query_raises(self) -> None:
        """Direct call covers the NoMatches catch in `_first_grid_or_none`."""
        from textual.css.query import NoMatches

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        with mock.patch.object(screen, "query_one", side_effect=NoMatches("none")):
            assert screen._first_grid_or_none() is None

    def test_scroll_to_end_of_last_grid_when_cursor_on_last_row(self) -> None:
        """Direct call covers the scroll_end branch deterministically."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        target = mock.MagicMock(spec=ModelGrid)
        target.highlighted = 0
        target.columns_per_row = 1
        target.rows = [object()]
        fake_container = mock.MagicMock()
        fake_container.query.return_value = [target]
        with (
            mock.patch.object(screen, "_focused_grid", return_value=target),
            mock.patch.object(CatalogScreen, "_grid_container", new=fake_container),
        ):
            screen._reveal_scroll_hint_at_catalog_end()
        fake_container.scroll_end.assert_called_once()

    def test_reveal_scroll_hint_returns_when_cursor_not_on_last_row(self) -> None:
        """Cursor above the last row leaves the catalog viewport in place."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        target = mock.MagicMock(spec=ModelGrid)
        target.highlighted = 0  # row 0
        target.columns_per_row = 1
        target.rows = [object(), object(), object()]  # last_row = 2
        fake_container = mock.MagicMock()
        fake_container.query.return_value = [target]
        with (
            mock.patch.object(screen, "_focused_grid", return_value=target),
            mock.patch.object(CatalogScreen, "_grid_container", new=fake_container),
        ):
            screen._reveal_scroll_hint_at_catalog_end()
        fake_container.scroll_end.assert_not_called()

    def test_prefetch_returns_when_no_grids(self) -> None:
        """Direct call: when grid container has no ModelGrid, prefetch no-ops."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        screen._grid_view = True
        screen._hf_has_more = True
        screen._loading_more = False
        fake_container = mock.MagicMock()
        fake_container.query.return_value = []
        # Override the textual.getters.query_one descriptor with a plain
        # instance attribute so _grid_container resolves to our stub.
        with mock.patch.object(CatalogScreen, "_grid_container", new=fake_container):
            screen._maybe_prefetch_on_grid_nav()

    async def test_prefetch_swallows_value_error_when_focused_not_in_grids(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            screen._hf_has_more = True
            screen._loading_more = False
            stranger = mock.MagicMock(spec=ModelGrid)
            stranger.highlighted = 0
            stranger.rows = []
            with mock.patch.object(screen, "_focused_grid", return_value=stranger):
                # _grid_container has at least one real ModelGrid; stranger
                # isn't in it, so grids.index(focused) raises ValueError.
                screen._maybe_prefetch_on_grid_nav()

    async def test_prefetch_returns_when_total_rows_zero(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            screen._hf_has_more = True
            screen._loading_more = False
            grids = list(screen.query(ModelGrid))
            if not grids:
                pytest.skip("no grids mounted in this build")
            focused = grids[0]
            with (
                mock.patch.object(screen, "_focused_grid", return_value=focused),
                mock.patch.object(
                    type(focused), "rows", new_callable=mock.PropertyMock, return_value=[]
                ),
            ):
                focused.highlighted = 0
                screen._maybe_prefetch_on_grid_nav()

    async def test_mount_remaining_grid_sections_restores_prior_scroll(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            with (
                mock.patch.object(screen, "_mount_grid_section"),
                mock.patch.object(screen, "_mount_grid_ctas"),
                mock.patch.object(screen._grid_container, "scroll_to") as scroll_to,
            ):
                screen._mount_remaining_grid_sections([], hf_count=0, prior_scroll_y=12.5)
                scroll_to.assert_called_once()

    def test_mount_remaining_returns_when_restore_focused_section_succeeds(self) -> None:
        """`_mount_remaining_grid_sections` returns after a successful restore."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        screen._grid_view = True
        fake_container = mock.MagicMock()
        fake_container.scroll_y = 0
        with (
            mock.patch.object(CatalogScreen, "_grid_container", new=fake_container),
            mock.patch.object(screen, "_mount_grid_section"),
            mock.patch.object(screen, "_mount_grid_ctas"),
            mock.patch.object(screen, "_focused_grid", return_value=None),
            mock.patch.object(screen, "_restore_focused_section", return_value=True),
            mock.patch.object(screen, "_focus_first_grid") as focus_first,
        ):
            screen._mount_remaining_grid_sections(
                [], hf_count=0, focus_anchor=("Chat", 0), prior_scroll_y=0.0
            )
            focus_first.assert_not_called()


class TestModelGridScrollIntoViewParentGuard:
    """`watch_highlighted` returns early when the parent isn't a Widget."""

    def test_returns_when_parent_not_widget(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        row = LocalCatalogRow(
            name="m",
            task="chat",
            params="",
            size="",
            quant="",
            downloads="",
            featured=False,
            installed=False,
            sort_downloads=0,
            sort_size=0.0,
            ref="m/m",
        )
        grid = ModelGrid(rows=[row])
        grid._cards_per_row = 1
        # Stub size so the first early-return guard passes. With no parent
        # mounted, `self.parent` is None and the isinstance(parent, Widget)
        # guard returns at line 190.
        type(grid).size = property(lambda self: mock.Mock(width=80, height=20))  # type: ignore[assignment]
        try:
            grid.watch_highlighted(None, 0)
        finally:
            del type(grid).size  # restore the descriptor inherited from Widget


class TestModelBarVisionSidecarErrors:
    """`_has_vision_sidecar` returns False when registry.resolve raises."""

    def test_has_vision_sidecar_returns_false_on_missing_ref(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _has_vision_sidecar

        registry = mock.MagicMock()
        registry.resolve.side_effect = KeyError
        assert _has_vision_sidecar(registry, "missing/ref") is False

    def test_has_vision_sidecar_returns_false_on_value_error(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _has_vision_sidecar

        registry = mock.MagicMock()
        registry.resolve.side_effect = ValueError
        assert _has_vision_sidecar(registry, "bad/ref") is False


class TestScopeChipPillSelect:
    """`ScopePill.action_select` routes to the parent chip's scope setter."""

    async def test_pill_select_changes_chip_scope(self) -> None:
        from textual.app import App, ComposeResult

        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield ScopeChip()

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            chip = pilot.app.query_one(ScopeChip)
            wiki_pill = chip.query_one("#scope-pill-wiki")
            wiki_pill.action_select()
            await pilot.pause()
            assert chip.scope is SearchScope.WIKI


class TestCatalogSmallEdgeBranches:
    """A handful of small catalog branches that flake out across xdist workers."""

    def test_restore_focused_section_skips_non_matching_grids(self) -> None:
        """`_restore_focused_section` skips grids whose name != target_heading."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        non_match = mock.MagicMock(spec=ModelGrid)
        non_match.name = "OtherName"
        match = mock.MagicMock(spec=ModelGrid)
        match.name = "Chat"
        match.rows = [object(), object()]
        fake_container = mock.MagicMock()
        fake_container.query.return_value = [non_match, match]
        with mock.patch.object(CatalogScreen, "_grid_container", new=fake_container):
            assert screen._restore_focused_section(("Chat", 5)) is True
        match.focus.assert_called_once()
        non_match.focus.assert_not_called()

    def test_restore_focused_section_returns_false_when_no_match(self) -> None:
        """`_restore_focused_section` returns False when no grid matches."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        screen = CatalogScreen.__new__(CatalogScreen)
        # __new__ bypasses __init__; pin minimum fields the screen reads.
        screen._active_tab_id_cache = "chat"
        screen._activation_settled = True
        screen._tab_grid_cache = {}
        screen._tab_list_cache = {}
        screen._grid_cache_keys = {}
        screen._list_cache_keys = {}
        screen._source_modes = {
            "chat": "local",
            "embed": "local",
            "vision": "local",
            "rerank": "local",
        }
        non_match = mock.MagicMock(spec=ModelGrid)
        non_match.name = "Other"
        fake_container = mock.MagicMock()
        fake_container.query.return_value = [non_match]
        with mock.patch.object(CatalogScreen, "_grid_container", new=fake_container):
            assert screen._restore_focused_section(("Missing", None)) is False

    async def test_action_toggle_view_to_list_triggers_first_hf_fetch(self) -> None:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        class _Probe(App[None]):
            def compose(self) -> ComposeResult:
                yield Footer()

            def on_mount(self) -> None:
                self.push_screen(CatalogScreen())

        async with _Probe().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = pilot.app.screen
            assert isinstance(screen, CatalogScreen)
            screen._active_tab_id_cache = "chat"
            screen._activation_settled = True
            screen._hf_fetched = False
            screen._grid_view = True
            with mock.patch.object(screen, "_fetch_all_hf_models") as mock_fetch:
                screen.action_toggle_view()
                assert screen._hf_fetched is True
                mock_fetch.assert_called_once()
