"""The TUI start-up swap and `lilbee ask` show the warnings of a model-role settings update."""

from __future__ import annotations

from unittest import mock

from lilbee.app.settings import SettingsUpdateResult
from lilbee.catalog.types import ModelSource
from lilbee.cli.commands import search_chat
from lilbee.core.config import cfg
from lilbee.modelhub import models as models_hub

NOTICE = "No installed engine loads this model. Install one with `lilbee engine install`."


class TestTuiStartupSwapShowsWarnings:
    async def test_the_swap_toasts_the_settings_warning_once(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult

        app = LilbeeApp()
        chat_canon = CanonicalRef(
            original="missing/model",
            effective="fallback/model",
            status=ValidationResult.NOT_INSTALLED,
            reason="it isn't installed",
        )
        embed_canon = CanonicalRef(
            original="e/m", effective="e/m", status=ValidationResult.OK
        )
        toasts: list[str] = []
        snapshot = cfg.chat_model
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
                mock.patch(
                    "lilbee.cli.tui.app.apply_settings_update",
                    return_value=SettingsUpdateResult(
                        updated=["chat_model"], reindex_required=False, warnings=(NOTICE,)
                    ),
                ),
                mock.patch.object(
                    app, "notify", side_effect=lambda text, **_kw: toasts.append(text)
                ),
                mock.patch(
                    "lilbee.cli.tui.app.call_from_thread",
                    side_effect=lambda _node, fn, *a, **k: fn(*a, **k),
                ),
            ):
                app.canonicalize_persisted_models()
        finally:
            cfg.chat_model = snapshot
        assert toasts.count(NOTICE) == 1

    async def test_an_adopt_into_an_unconfigured_role_still_toasts_the_warning(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult

        app = LilbeeApp()
        chat_canon = CanonicalRef(
            original="", effective="adopted/model", status=ValidationResult.UNKNOWN
        )
        embed_canon = CanonicalRef(
            original="e/m", effective="e/m", status=ValidationResult.OK
        )
        toasts: list[str] = []
        with (
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_chat_model", return_value=chat_canon
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_embedding_model",
                return_value=embed_canon,
            ),
            mock.patch(
                "lilbee.cli.tui.app.apply_settings_update",
                return_value=SettingsUpdateResult(
                    updated=["chat_model"], reindex_required=False, warnings=(NOTICE,)
                ),
            ),
            mock.patch.object(app, "notify", side_effect=lambda text, **_kw: toasts.append(text)),
            mock.patch(
                "lilbee.cli.tui.app.call_from_thread",
                side_effect=lambda _node, fn, *a, **k: fn(*a, **k),
            ),
        ):
            app.canonicalize_persisted_models()
        assert toasts == [NOTICE]


class TestAskShowsWarnings:
    def _prepare(self, warnings: tuple[str, ...]) -> mock.MagicMock:
        ensure = mock.MagicMock(return_value="org/repo/m.gguf")
        with (
            mock.patch("lilbee.modelhub.models.ensure_chat_model", ensure),
            mock.patch(
                "lilbee.app.settings.apply_settings_update",
                return_value=SettingsUpdateResult(
                    updated=["chat_model"], reindex_required=False, warnings=warnings
                ),
            ),
            mock.patch.object(search_chat, "_swap_stale_models_to_installed"),
            mock.patch.object(search_chat, "get_services") as services,
            mock.patch.object(cfg, "auto_sync", False),
        ):
            services.return_value.searcher.search_unavailable.return_value = False
            search_chat._prepare_ask(no_sync=True)
        return ensure

    def test_the_warning_reaches_stderr_once(self, capsys) -> None:
        self._prepare((NOTICE,))
        captured = capsys.readouterr()
        assert captured.err.count(NOTICE) == 1
        assert NOTICE not in captured.out

    def test_the_pull_stays_silent_on_the_notice_the_update_reports(self) -> None:
        ensure = self._prepare(())
        ensure.assert_called_once_with(report_no_engine=False)


class TestFirstRunPullForwardsTheNoEngineSwitch:
    def test_ensure_chat_model_forwards_the_switch_to_the_pull(self) -> None:
        info = mock.MagicMock()
        with (
            mock.patch.object(models_hub, "list_installed_models", return_value=[]),
            mock.patch.object(models_hub, "_refuse_a_model_with_no_engine"),
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch.object(models_hub, "get_system_ram_gb", return_value=8.0),
            mock.patch.object(models_hub, "get_free_disk_gb", return_value=50.0),
            mock.patch.object(models_hub, "prompt_model_choice", return_value=info),
            mock.patch.object(models_hub, "validate_disk_and_pull", return_value="r") as pull,
        ):
            models_hub.ensure_chat_model(report_no_engine=False)
        pull.assert_called_once_with(info, 50.0, report_no_engine=False)

    def test_the_progress_pull_forwards_the_switch_to_the_manager(self) -> None:
        from lilbee.app.services import set_services

        manager = mock.MagicMock()
        set_services(mock.MagicMock(model_manager=manager))
        with mock.patch("lilbee.app.models.ensure_vision_projector", return_value=None):
            models_hub.pull_with_progress("org/repo", report_no_engine=False)
        assert manager.pull.call_args.args[:2] == ("org/repo", ModelSource.NATIVE)
        assert manager.pull.call_args.kwargs["report_no_engine"] is False
