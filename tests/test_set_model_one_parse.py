"""A ref typed with a #gguf qualifier lands as the plain ref where it enters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from conftest import PICKS_CHAT, install_fake_model
from lilbee.app import engine_choice, engine_service
from lilbee.app import models as app_models
from lilbee.app.engine_service import EngineRefusedError
from lilbee.catalog.refs import canonical_ref
from lilbee.catalog.types import ModelSource
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import ModelManager
from lilbee.modelhub.model_manager.discovery import KnownModelCache
from lilbee.modelhub.role_validator import validate_model_task_assignment
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.model_ref import canonical_model_ref
from lilbee.providers.roles import WorkerRole
from lilbee.server.handlers import models as handlers


class TestCanonicalRef:
    @pytest.mark.parametrize(
        ("typed", "plain"),
        [
            ("org/repo/f.gguf#gguf", "org/repo/f.gguf"),
            ("org/repo#gguf", "org/repo"),
            ("org/repo#mlx", "org/repo#mlx"),
            ("org/repo#nosuch", "org/repo#nosuch"),
        ],
    )
    def test_gguf_qualifier_reads_as_the_plain_ref(self, typed: str, plain: str) -> None:
        assert canonical_ref(typed) == plain


class TestConfigBoundary:
    def test_config_form_drops_the_gguf_qualifier(self) -> None:
        assert canonical_model_ref("org/repo/f.gguf#gguf") == "org/repo/f.gguf"

    def test_provider_named_org_stays_local_once_plain(self) -> None:
        assert canonical_model_ref("mistral/repo/f.gguf#gguf") == "mistral/repo/f.gguf"

    def test_assigning_a_qualified_ref_stores_the_plain_ref(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "embedding_model", "")
        cfg.embedding_model = "org/repo/f.gguf#gguf"
        assert cfg.embedding_model == "org/repo/f.gguf"


class TestSetModelBoundary:
    def test_installed_exact_file_persists_the_plain_ref(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment("chat_model", f"{ref}#gguf", allow_bypass=False)
        assert result == ref

    def test_installed_bare_repo_with_qualifier_resolves_to_its_quant(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment(
            "chat_model", "other/Chat-Model-GGUF#gguf", allow_bypass=False
        )
        assert result == ref

    def test_pick_with_qualifier_resolves_to_the_picks_ref(self) -> None:
        pick = PICKS_CHAT[0]
        result = validate_model_task_assignment(
            "chat_model", f"{pick.hf_repo}#gguf", allow_bypass=False
        )
        assert result == pick.ref

    def test_malformed_qualifier_is_refused_as_not_installed(self) -> None:
        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment("chat_model", "nobody/Repo#nosuch", allow_bypass=False)


class TestRestSetModelBoundary:
    @pytest.fixture
    def available(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        listed = ["org/repo/f.gguf"]
        services = SimpleNamespace(provider=SimpleNamespace(list_models=lambda: listed))
        monkeypatch.setattr(handlers, "get_services", lambda: services)
        return listed

    def test_qualified_file_matches_the_listed_ref(self, available: list[str]) -> None:
        assert handlers._require_model_available("org/repo/f.gguf#gguf") == "org/repo/f.gguf"

    def test_qualified_bare_repo_resolves_to_a_listed_quant(self, available: list[str]) -> None:
        assert handlers._require_model_available("org/repo#gguf") == "org/repo/f.gguf"

    def test_unlisted_qualified_ref_is_refused_by_its_written_name(
        self, available: list[str]
    ) -> None:
        with pytest.raises(ValueError, match=r"Model 'org/other#gguf' is not available"):
            handlers._require_model_available("org/other#gguf")


class TestApiResolveBoundary:
    def test_qualified_ref_resolves_to_the_known_plain_ref(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = KnownModelCache()
        monkeypatch.setattr(cache, "refs", lambda: frozenset({"org/repo/f.gguf"}))
        assert cache.resolve("org/repo/f.gguf#gguf") == "org/repo/f.gguf"

    def test_unknown_qualified_ref_stays_unresolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = KnownModelCache()
        monkeypatch.setattr(cache, "refs", lambda: frozenset({"org/repo/f.gguf"}))
        monkeypatch.setattr(
            "lilbee.modelhub.model_manager.discovery.get_services",
            lambda: SimpleNamespace(registry=MagicMock()),
        )
        assert cache.resolve("org/other/g.gguf#gguf") is None


class TestFleetPerCallModel:
    def test_qualified_spelling_of_the_configured_model_is_accepted(self) -> None:
        provider = FleetProvider()
        provider._require_configured_model(
            "org/repo/f.gguf#gguf", "org/repo/f.gguf", WorkerRole.CHAT
        )

    def test_a_different_model_is_still_refused(self) -> None:
        provider = FleetProvider()
        with pytest.raises(ProviderError):
            provider._require_configured_model(
                "org/other/g.gguf#gguf", "org/repo/f.gguf", WorkerRole.CHAT
            )


class TestEngineUseBoundary:
    def test_qualified_ref_names_the_configured_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chosen: list[tuple[str, str]] = []
        monkeypatch.setattr(
            engine_service, "use", lambda model, name, confirmed=False: chosen.append((model, name))
        )
        config = SimpleNamespace(chat_model="org/repo/f.gguf")
        engine_service.use_for_role(WorkerRole.CHAT, "org/repo/f.gguf#gguf", "auto", config)
        assert chosen == [("org/repo/f.gguf", "auto")]

    def test_a_different_model_is_still_refused(self) -> None:
        config = SimpleNamespace(chat_model="org/repo/f.gguf")
        with pytest.raises(EngineRefusedError, match=r"not org/other/g\.gguf"):
            engine_service.use_for_role(WorkerRole.CHAT, "org/other/g.gguf#gguf", "auto", config)


class _StopAtChoice(Exception):
    """Ends a pull at the engine choice, after the stored choice was read."""


class TestPullEngineChoice:
    @pytest.fixture
    def overrides(self, monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
        seen: list[str | None] = []

        def _choose(**kwargs: object) -> None:
            seen.append(cast(str | None, kwargs["override"]))
            raise _StopAtChoice

        monkeypatch.setattr(engine_choice, "choose_engine", _choose)
        monkeypatch.setattr(engine_choice, "installed_engines", lambda: {})
        monkeypatch.setattr(cfg, "engine_choices", json.dumps({"org/repo/f.gguf": "llama"}))
        return seen

    def test_a_qualified_pull_reads_the_choice_stored_under_the_plain_ref(
        self, overrides: list[str | None], tmp_path: Path
    ) -> None:
        manager = ModelManager(tmp_path / "models", cfg)
        with pytest.raises(_StopAtChoice):
            manager.pull("org/repo/f.gguf#gguf", ModelSource.NATIVE)
        assert overrides == ["llama"]

    async def test_the_rest_preflight_reads_the_choice_stored_under_the_plain_ref(
        self, overrides: list[str | None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        services = SimpleNamespace(model_manager=ModelManager(tmp_path / "models", cfg))
        monkeypatch.setattr(handlers, "get_services", lambda: services)
        with pytest.raises(_StopAtChoice):
            await handlers.enforce_pull_arch_compat("org/repo/f.gguf#gguf")
        assert overrides == ["llama"]


class TestAdoptEmbedder:
    def test_qualified_ref_of_the_active_embedder_is_already_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = MagicMock()
        manager.is_installed.return_value = True
        services = SimpleNamespace(model_manager=manager)
        monkeypatch.setattr(app_models, "get_services", lambda: services)
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/f.gguf")
        update = SimpleNamespace(reindex_required=False, warnings=())
        with patch("lilbee.app.settings.apply_settings_update", return_value=update) as write:
            result = app_models.adopt_embedder("org/repo/f.gguf#gguf")
        assert result.status == app_models.AdoptStatus.ALREADY_ACTIVE
        assert result.model == "org/repo/f.gguf"
        write.assert_called_once_with({"embedding_model": "org/repo/f.gguf"})

    def test_an_installed_embedder_no_engine_loads_carries_the_no_engine_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = MagicMock()
        manager.is_installed.return_value = True
        services = SimpleNamespace(model_manager=manager)
        monkeypatch.setattr(app_models, "get_services", lambda: services)
        notice = "No installed engine loads this model."
        update = SimpleNamespace(reindex_required=False, warnings=(notice,))
        with patch("lilbee.app.settings.apply_settings_update", return_value=update):
            result = app_models.adopt_embedder("org/repo/f.gguf#gguf")
        assert result.warning == notice

    def test_the_settings_notice_follows_the_pull_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = MagicMock()
        manager.is_installed.return_value = False
        services = SimpleNamespace(model_manager=manager)
        monkeypatch.setattr(app_models, "get_services", lambda: services)
        update = SimpleNamespace(reindex_required=False, warnings=("No engine.",))
        pulled = SimpleNamespace(warning="Slow.")
        with (
            patch("lilbee.app.settings.apply_settings_update", return_value=update),
            patch.object(app_models, "pull_model_data", return_value=pulled),
        ):
            result = app_models.adopt_embedder("org/repo/f.gguf#gguf")
        assert result.warning == "Slow. No engine."

    def test_a_missing_non_gguf_embedder_shows_the_no_engine_notice_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A real pull yields the no-engine notice and another one; the update repeats the first."""
        from lilbee.app import settings as app_settings

        ref = "org/repo#safetensors"
        manager = ModelManager(tmp_path / "models", cfg)
        services = SimpleNamespace(model_manager=manager)
        monkeypatch.setattr(app_models, "get_services", lambda: services)
        monkeypatch.setattr(cfg, "embedding_model", ref)
        monkeypatch.setattr(engine_choice, "installed_engines", dict)
        monkeypatch.setattr(app_models, "ensure_vision_projector", lambda *a, **k: None)
        (settings_notice,) = app_settings._no_engine_notices({"embedding_model"})

        def download(model, engine, *, on_notice, **_kw):
            on_notice("The download size is unknown.")
            return tmp_path / "model"

        update = SimpleNamespace(reindex_required=False, warnings=(settings_notice,))
        with (
            patch.object(ModelManager, "is_installed", return_value=False),
            patch.object(ModelManager, "enforce_arch_compat"),
            patch.object(ModelManager, "_pull_native", side_effect=download),
            patch("lilbee.app.settings.apply_settings_update", return_value=update),
        ):
            result = app_models.adopt_embedder(ref)
        assert result.warning == f"The download size is unknown. {settings_notice}"


_NO_ENGINE_SENTENCE = (
    "No installed engine loads the safetensors format of org/repo#safetensors. "
    "Run lilbee engine list to see the installed engines."
)


def _pull_with_no_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: bool
) -> app_models.PullResult:
    """Run a real pull of a non-GGUF model that no engine loads, with the download stubbed."""
    manager = ModelManager(tmp_path / "models", cfg)
    services = SimpleNamespace(model_manager=manager)
    monkeypatch.setattr(app_models, "get_services", lambda: services)
    monkeypatch.setattr(engine_choice, "installed_engines", dict)
    monkeypatch.setattr(app_models, "ensure_vision_projector", lambda *a, **k: None)
    with (
        patch.object(ModelManager, "is_installed", return_value=False),
        patch.object(ModelManager, "enforce_arch_compat"),
        patch.object(ModelManager, "_pull_native", return_value=tmp_path / "model"),
    ):
        return app_models.pull_model_data("org/repo#safetensors", ModelSource.NATIVE, **kwargs)


class TestPullModelDataNoEngineNotice:
    def test_a_pull_with_no_keyword_reports_the_no_engine_notice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The CLI and task bar pulls pass no keyword and rely on this notice."""
        result = _pull_with_no_engine(monkeypatch, tmp_path)
        assert result.warning == _NO_ENGINE_SENTENCE

    def test_a_pull_that_opts_out_leaves_the_no_engine_notice_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = _pull_with_no_engine(monkeypatch, tmp_path, report_no_engine=False)
        assert result.warning is None
