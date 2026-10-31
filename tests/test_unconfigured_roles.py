"""An unconfigured model role is a state, not an error: quiet at boot, loud on use."""

from __future__ import annotations

import logging
from unittest import mock

import pytest

from lilbee.core.config import cfg


class TestCanonicalizeUnconfigured:
    """`LilbeeApp.canonicalize_persisted_models` on empty refs."""

    def _canon(self, original: str, effective: str):
        from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult

        return CanonicalRef(
            original=original,
            effective=effective,
            status=ValidationResult.UNKNOWN,
            reason="it is unavailable",
        )

    async def test_empty_roles_with_nothing_installed_say_nothing(self, caplog) -> None:
        """A fresh install must not log or toast about models nobody chose."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        notifications: list[tuple] = []
        with (
            mock.patch.object(
                LilbeeApp, "notify", side_effect=lambda *a, **k: notifications.append(a)
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_chat_model",
                return_value=self._canon("", ""),
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_embedding_model",
                return_value=self._canon("", ""),
            ),
            mock.patch(
                "lilbee.cli.tui.app.call_from_thread", side_effect=lambda a, fn, *args: None
            ),
            caplog.at_level(logging.INFO, logger="lilbee.cli.tui.app"),
        ):
            app.canonicalize_persisted_models()
        assert not notifications
        assert not caplog.records

    async def test_empty_role_adopts_an_installed_model_quietly(self, caplog) -> None:
        """Models pulled before the TUI ever ran become active with an INFO, no toast."""
        from lilbee.cli.tui.app import LilbeeApp

        snapshot = cfg.chat_model
        app = LilbeeApp()
        notifications: list[tuple] = []
        try:
            with (
                mock.patch.object(
                    LilbeeApp, "notify", side_effect=lambda *a, **k: notifications.append(a)
                ),
                mock.patch(
                    "lilbee.modelhub.model_manager.canonicalize_chat_model",
                    return_value=self._canon("", "installed/chat-GGUF/chat.gguf"),
                ),
                mock.patch(
                    "lilbee.modelhub.model_manager.canonicalize_embedding_model",
                    return_value=self._canon("", ""),
                ),
                mock.patch(
                    "lilbee.cli.tui.app.apply_settings_update",
                ) as apply_update,
                mock.patch(
                    "lilbee.cli.tui.app.call_from_thread", side_effect=lambda a, fn, *args: None
                ),
                caplog.at_level(logging.INFO, logger="lilbee.cli.tui.app"),
            ):
                app.canonicalize_persisted_models()
            apply_update.assert_called_once_with({"chat_model": "installed/chat-GGUF/chat.gguf"})
            assert not notifications, "adopting into an empty role is expected, not a warning"
            infos = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any("installed/chat-GGUF/chat.gguf" in r.getMessage() for r in infos)
            assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            cfg.chat_model = snapshot or ""


class TestEmbedderUnconfigured:
    def test_validate_model_empty_ref_is_one_info_line(self, caplog) -> None:
        from lilbee.retrieval.embedder import Embedder

        config = cfg.model_copy(update={"embedding_model": ""})
        embedder = Embedder(config, mock.MagicMock())
        with caplog.at_level(logging.INFO, logger="lilbee.retrieval.embedder"):
            assert embedder.validate_model() is False
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("No embedding model configured" in r.getMessage() for r in caplog.records)


class TestServeBootUnconfigured:
    def test_log_state_says_nothing_extra_for_empty_ref(self, caplog, monkeypatch) -> None:
        from lilbee.server.app import _log_embedding_model_state

        monkeypatch.setattr(cfg, "embedding_model", "")
        embedder = mock.MagicMock()
        embedder.validate_model.return_value = False
        with caplog.at_level(logging.INFO, logger="lilbee.server.app"):
            _log_embedding_model_state(embedder)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestResolveUnconfigured:
    def test_resolve_model_path_empty_names_the_state_not_a_model(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.engine_params import resolve_model_path

        with pytest.raises(ProviderError, match="No model is configured"):
            resolve_model_path("", registry=mock.MagicMock())


class TestLauncherUnconfigured:
    def test_no_pin_warning_when_chat_model_is_empty(self, monkeypatch) -> None:
        """Served models + an empty chat role is a workable launch, not a warning."""
        import typer

        from lilbee.cli.launchers.launcher import _warn_on_model_pin_gaps

        monkeypatch.setattr(cfg, "chat_model", "")
        with mock.patch.object(typer, "secho") as secho:
            _warn_on_model_pin_gaps(["owner/served-GGUF/served.gguf"])
        secho.assert_not_called()


class TestOneShotSwapUnconfigured:
    def test_swap_notice_names_the_state_for_an_empty_original(self, capsys) -> None:
        from lilbee.cli.commands.search_chat import _swap_stale_models_to_installed
        from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult

        adopted = CanonicalRef(
            original="",
            effective="installed/chat-GGUF/chat.gguf",
            status=ValidationResult.UNKNOWN,
            reason="it is unavailable",
        )
        untouched = CanonicalRef(original="", effective="", status=ValidationResult.UNKNOWN)
        with (
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_chat_model", return_value=adopted
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_embedding_model",
                return_value=untouched,
            ),
            mock.patch("lilbee.app.settings.apply_ephemeral_model_swap") as swap,
        ):
            _swap_stale_models_to_installed()
        swap.assert_called_once_with("chat_model", "installed/chat-GGUF/chat.gguf")
        err = capsys.readouterr().err
        assert "No chat model configured" in err
        assert "''" not in err


class TestSdkProviderUnconfigured:
    def _provider(self):
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(backend=mock.MagicMock())
        provider._initialized = True
        return provider

    def test_chat_names_the_state(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cfg, "chat_model", "")
        with pytest.raises(ProviderError, match="No chat model is configured"):
            self._provider().chat([{"role": "user", "content": "hi"}])

    def test_embed_names_the_state(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cfg, "embedding_model", "")
        with pytest.raises(ProviderError, match="No embedding model is configured"):
            self._provider().embed(["hi"])

    def test_rerank_names_the_state(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cfg, "reranker_model", "")
        with pytest.raises(ProviderError, match="No reranker model is configured"):
            self._provider().rerank("q", ["a"])


class TestIngestUnconfigured:
    def test_refusal_names_the_state_not_an_empty_ref(self, monkeypatch) -> None:
        from lilbee.data.ingest import pipeline

        monkeypatch.setattr(cfg, "embedding_model", "")
        services = mock.MagicMock()
        services.embedder.validate_model.return_value = False
        monkeypatch.setattr(pipeline, "get_services", lambda: services)
        with pytest.raises(RuntimeError, match="None is configured"):
            pipeline._require_embedding_model()


class TestRoutingProviderUnconfigured:
    def test_embed_and_count_tokens_name_the_state(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.routing_provider import RoutingProvider

        monkeypatch.setattr(cfg, "embedding_model", "")
        provider = RoutingProvider()
        with pytest.raises(ProviderError, match="No embedding model is configured"):
            provider.embed(["hi"])
        with pytest.raises(ProviderError, match="No embedding model is configured"):
            provider.count_tokens("hi")

    def test_chat_paths_name_the_state_and_tools_read_false(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.routing_provider import RoutingProvider

        monkeypatch.setattr(cfg, "chat_model", "")
        provider = RoutingProvider()
        with pytest.raises(ProviderError, match="No chat model is configured"):
            provider.chat([{"role": "user", "content": "hi"}])
        with pytest.raises(ProviderError, match="No chat model is configured"):
            provider.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])
        assert provider.supports_tools("") is False
