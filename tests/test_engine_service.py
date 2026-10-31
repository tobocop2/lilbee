"""The engine service stores the engine choice, refuses what cannot run, and the book reads it."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from lilbee.app import engine_choice, engine_service
from lilbee.app.engine_choice import EngineBook, engine_for_ref, resolve_engines, stored_choices
from lilbee.catalog.hf_client import HfClient
from lilbee.catalog.types import ModelCompat
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.core.config.model import SERVICE_ONLY_FIELDS
from lilbee.modelhub.model_manager import ModelManager
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    AUTO,
    HostFacts,
    NoEngineError,
)
from lilbee.providers.roles import WorkerRole
from lilbee.server.chat_dispatch import capability
from tests.conformance.fake_adapter import FakeAdapter

MLX_REF = "org/repo#mlx"
GGUF_REF = "org/repo/file.gguf"
NOT_INSTALLED = "Engine {} is not installed. Run lilbee engine list to see the installed engines."


class _GgufEngine(FakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


class _MlxEngine(FakeAdapter):
    name = "mlx"
    formats = frozenset({"mlx"})
    architectures = frozenset({"mlx_arch"})


class _OpenvinoEngine(FakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


GGUF_ENGINE = _GgufEngine()
MLX_ENGINE = _MlxEngine()
OPENVINO_ENGINE = _OpenvinoEngine()


@pytest.fixture(autouse=True)
def _host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_choice, "host_facts", lambda: HostFacts(system="darwin", machine="arm64")
    )


@pytest.fixture
def with_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GGUF_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, MLX_ENGINE)  # type: ignore[arg-type]


@pytest.fixture
def with_openvino(with_mlx: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second engine that loads GGUF, so a choice differs from the host default."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.OPENVINO, OPENVINO_ENGINE)  # type: ignore[arg-type]


def _stored(key: str) -> object:
    return settings.load(cfg.data_root).get(key)


def test_the_fields_are_service_only_and_the_setting_holds_every_engine() -> None:
    assert {"preferred_engine", "engine_choices"} <= SERVICE_ONLY_FIELDS
    assert {s.value for s in EngineSetting} == {AUTO, *(e.value for e in EngineName)}
    config = Config()
    assert config.preferred_engine is EngineSetting.AUTO
    assert config.engine_choices is None


def test_a_toml_table_of_choices_reads_as_text() -> None:
    config = Config(engine_choices={MLX_REF: "mlx"})  # type: ignore[arg-type]
    assert json.loads(config.engine_choices or "") == {MLX_REF: "mlx"}


def test_the_environment_does_not_set_the_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILBEE_PREFERRED_ENGINE", "mlx")
    assert Config().preferred_engine is EngineSetting.AUTO


@pytest.mark.parametrize("key", ["preferred_engine", "engine_choices"])
def test_the_general_settings_path_refuses_both_keys(key: str) -> None:
    from lilbee.app.settings import apply_settings_update

    with pytest.raises(ValueError, match="Unknown or read-only setting"):
        apply_settings_update({key: "mlx"})
    assert _stored(key) is None


def test_prefer_stores_the_name_and_the_book_follows(with_mlx: None) -> None:
    assert engine_service.prefer("mlx") == "mlx"
    assert _stored("preferred_engine") == "mlx"
    config = Config()
    config.chat_model = MLX_REF
    config.preferred_engine = cfg.preferred_engine
    assert EngineBook(config).for_role(WorkerRole.CHAT).adapter is MLX_ENGINE
    assert engine_service.prefer("auto") == "auto"
    assert _stored("preferred_engine") == "auto"


def test_the_preference_beats_the_host_default_and_loses_to_the_model_choice(
    with_openvino: None,
) -> None:
    engine_service.prefer("openvino")
    assert cfg.preferred_engine is EngineSetting.OPENVINO
    config = Config()
    config.preferred_engine = cfg.preferred_engine
    config.chat_model = GGUF_REF
    config.embedding_model = "org/other/embed.gguf"
    config.engine_choices = json.dumps({"org/other/embed.gguf": "llama.cpp"})
    assignment = resolve_engines(config)
    assert assignment.for_role(WorkerRole.CHAT).adapter is OPENVINO_ENGINE
    assert assignment.for_role(WorkerRole.EMBED).adapter is GGUF_ENGINE
    assert engine_for_ref(GGUF_REF, cfg) is OPENVINO_ENGINE


def test_a_pull_follows_the_choice_for_that_model_only(with_openvino: None) -> None:
    engine_service.use(GGUF_REF, "openvino")
    assert cfg.engine_choices == json.dumps({GGUF_REF: "openvino"})
    assert engine_for_ref(GGUF_REF, cfg) is OPENVINO_ENGINE
    assert engine_for_ref("org/other/file.gguf", cfg) is GGUF_ENGINE
    assert ModelManager(cfg.models_dir, cfg).require_engine(GGUF_REF) is OPENVINO_ENGINE


def test_prefer_refuses_an_engine_that_is_not_installed() -> None:
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.prefer("mlx")
    assert str(raised.value) == NOT_INSTALLED.format("mlx")
    assert _stored("preferred_engine") is None
    assert cfg.preferred_engine is EngineSetting.AUTO


def test_prefer_refuses_an_unknown_name() -> None:
    with pytest.raises(engine_service.EngineRefusedError, match="No engine is named 'nope'"):
        engine_service.prefer("nope")


def test_use_stores_the_choice_for_one_model_and_the_other_keeps_its_own(with_mlx: None) -> None:
    engine_service.use(MLX_REF, "mlx")
    engine_service.use(GGUF_REF, "llama.cpp")
    assert json.loads(str(_stored("engine_choices"))) == {MLX_REF: "mlx", GGUF_REF: "llama.cpp"}
    assert engine_for_ref(MLX_REF, cfg) is MLX_ENGINE
    assert engine_for_ref(GGUF_REF, cfg) is GGUF_ENGINE


def test_use_with_auto_removes_only_that_model(with_mlx: None) -> None:
    engine_service.use(MLX_REF, "mlx")
    engine_service.use(GGUF_REF, "llama.cpp")
    engine_service.use(GGUF_REF, "auto")
    assert json.loads(str(_stored("engine_choices"))) == {MLX_REF: "mlx"}
    engine_service.use(MLX_REF, "auto")
    assert _stored("engine_choices") is None
    assert cfg.engine_choices is None


def test_use_refuses_an_engine_that_is_not_installed() -> None:
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.use(MLX_REF, "mlx")
    assert str(raised.value) == NOT_INSTALLED.format("mlx")
    assert _stored("engine_choices") is None


def test_use_refuses_an_engine_that_cannot_load_the_format(with_mlx: None) -> None:
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.use(MLX_REF, "llama.cpp")
    assert str(raised.value) == (
        "Engine llama.cpp cannot load the mlx format of org/repo#mlx. "
        "Choose an engine that loads mlx."
    )
    assert _stored("engine_choices") is None


def test_use_refuses_a_blank_model_and_an_unknown_engine() -> None:
    with pytest.raises(engine_service.EngineRefusedError, match="Give the model"):
        engine_service.use("  ", "llama.cpp")
    with pytest.raises(engine_service.EngineRefusedError, match="No engine is named 'nope'"):
        engine_service.use(GGUF_REF, "nope")


def test_the_override_loses_to_nothing_and_the_other_role_keeps_the_default(
    with_mlx: None,
) -> None:
    config = Config()
    config.chat_model = MLX_REF
    config.embedding_model = GGUF_REF
    config.engine_choices = json.dumps({MLX_REF: "mlx"})
    assignment = resolve_engines(config)
    assert assignment.for_role(WorkerRole.CHAT).adapter is MLX_ENGINE
    assert assignment.for_role(WorkerRole.EMBED).adapter is GGUF_ENGINE


def test_an_unusable_override_falls_back_with_the_notice_in_the_role_view(with_mlx: None) -> None:
    config = Config()
    config.chat_model = GGUF_REF
    config.engine_choices = json.dumps({GGUF_REF: "openvino"})
    chat = next(v for v in engine_service.roles(config) if v.role is WorkerRole.CHAT)
    assert chat.engine == "llama.cpp"
    assert chat.notice == (
        "Engine openvino for chat is not installed. Using llama.cpp. "
        "Run lilbee engine list to see the installed engines."
    )


def test_the_role_view_of_a_role_with_no_engine_names_the_model_and_format() -> None:
    config = Config()
    config.chat_model = MLX_REF
    chat = next(v for v in engine_service.roles(config) if v.role is WorkerRole.CHAT)
    assert (chat.engine, chat.model, chat.format) == (None, MLX_REF, "mlx")
    assert chat.notice == (
        "For chat: No installed engine loads the mlx format of org/repo#mlx. "
        "Run lilbee engine list to see the installed engines."
    )


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"a": 1}'])
def test_unreadable_stored_choices_act_as_no_choices(raw: str) -> None:
    assert stored_choices(raw) == {}


UNREADABLE_NOTICE = (
    "The saved engine choices cannot be read. Using the default engine for each role. "
    "Run lilbee engine use to choose again."
)


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"a": 1}'])
def test_unreadable_stored_choices_show_the_notice_once_for_each_role(
    raw: str, with_mlx: None
) -> None:
    config = Config()
    config.chat_model = GGUF_REF
    config.engine_choices = raw
    views = engine_service.roles(config)
    assert len(views) > 1
    assert [v.notice for v in views if v.model == GGUF_REF] == [UNREADABLE_NOTICE]
    assert all(v.notice and v.notice.count(UNREADABLE_NOTICE) == 1 for v in views)


def test_the_unreadable_notice_comes_before_the_fallback_notice(with_mlx: None) -> None:
    config = Config()
    config.chat_model = GGUF_REF
    config.preferred_engine = EngineSetting.OPENVINO
    config.engine_choices = "not json"
    chat = next(v for v in engine_service.roles(config) if v.role is WorkerRole.CHAT)
    assert chat.notice == (
        f"{UNREADABLE_NOTICE} Engine openvino for chat is not installed. Using llama.cpp. "
        "Run lilbee engine list to see the installed engines."
    )


def test_readable_stored_choices_carry_no_unreadable_notice(with_openvino: None) -> None:
    config = Config()
    config.chat_model = GGUF_REF
    config.engine_choices = json.dumps({GGUF_REF: "openvino"})
    assert all(v.notice is None for v in engine_service.roles(config) if v.model == GGUF_REF)


@pytest.mark.parametrize("raw", ["not json", "[1]"])
def test_unreadable_stored_choices_leave_every_role_its_default_engine(
    raw: str, with_openvino: None
) -> None:
    config = Config()
    config.chat_model = GGUF_REF
    config.engine_choices = raw
    assignment = resolve_engines(config)
    assert assignment.for_role(WorkerRole.CHAT).adapter is GGUF_ENGINE
    assert assignment.for_role(WorkerRole.EMBED).adapter is GGUF_ENGINE
    assert engine_for_ref(GGUF_REF, config) is GGUF_ENGINE


class _LibraryEngine(FakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    architectures = frozenset({"library_arch"})


def _library_config() -> Config:
    """A library container's config: it prefers an engine that the global config does not."""
    config = Config()
    config.chat_model = GGUF_REF
    config.preferred_engine = EngineSetting.OPENVINO
    config.engine_choices = json.dumps({GGUF_REF: "openvino"})
    return config


def test_a_pull_and_a_run_on_one_library_config_name_the_same_engine(
    with_openvino: None,
) -> None:
    library = _library_config()
    assert cfg.preferred_engine is EngineSetting.AUTO
    run = resolve_engines(library).for_role(WorkerRole.CHAT).adapter
    assert run is OPENVINO_ENGINE
    assert engine_for_ref(GGUF_REF, library) is run
    assert engine_choice.engine_for_format(choice.ModelFormat.GGUF, library) is run
    assert ModelManager(cfg.models_dir, library).require_engine(GGUF_REF) is run
    assert EngineBook(library).for_ref(GGUF_REF) is run
    assert ModelManager(cfg.models_dir, cfg).require_engine(GGUF_REF) is GGUF_ENGINE


@pytest.mark.real_hf_client
def test_the_catalog_client_judges_architectures_with_the_engine_of_its_config(
    monkeypatch: pytest.MonkeyPatch, with_mlx: None
) -> None:
    monkeypatch.setitem(choice.ADAPTERS, EngineName.OPENVINO, _LibraryEngine())  # type: ignore[arg-type]
    row = {
        "id": "acme/test-GGUF",
        "siblings": [{"rfilename": "model.gguf", "size": 4_000_000_000}],
        "downloads": 100,
        "pipeline_tag": "text-generation",
        "cardData": {"description": "x"},
        "gguf": {"total": 4_000_000_000, "context_length": 4096, "architecture": "library_arch"},
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, json=[row]))
    assert HfClient(_library_config()).fetch_models().models[0].compat is ModelCompat.SUPPORTED
    assert HfClient(cfg).fetch_models().models[0].compat is ModelCompat.UNSUPPORTED


def test_no_stored_choices_read_as_empty() -> None:
    assert stored_choices(None) == {}


def test_the_preference_picks_the_engine_for_a_pull(with_mlx: None) -> None:
    cfg.preferred_engine = EngineSetting.MLX
    assert engine_for_ref(MLX_REF, cfg) is MLX_ENGINE
    assert engine_choice.engine_for_format(choice.ModelFormat.MLX, cfg) is MLX_ENGINE


def test_the_manager_and_the_client_ask_for_the_engine_of_each_call(with_mlx: None) -> None:
    manager = ModelManager(cfg.models_dir, cfg)
    assert manager.require_engine(GGUF_REF) is GGUF_ENGINE
    assert manager.require_engine(MLX_REF) is MLX_ENGINE
    assert "engine" not in HfClient.__init__.__code__.co_varnames


def test_the_manager_follows_an_engine_that_arrives_after_it_was_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelManager(cfg.models_dir, cfg)
    with pytest.raises(NoEngineError):
        manager.require_engine(MLX_REF)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, MLX_ENGINE)  # type: ignore[arg-type]
    assert manager.require_engine(MLX_REF) is MLX_ENGINE


def test_a_tool_chat_with_no_engine_gets_the_engine_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = mock.Mock()
    provider.supports_tools.side_effect = NoEngineError("For chat: No installed engine loads x.")
    monkeypatch.setattr(capability, "get_services", lambda: SimpleNamespace(provider=provider))
    with pytest.raises(NoEngineError, match="For chat: No installed engine loads x"):
        capability.model_supports_tools("org/repo#mlx")


class _RecordingProvider:
    def __init__(self, failure: Exception | None = None) -> None:
        self.reloads: list[bool] = []
        self.failure = failure

    def reload_placement(self, *, wait: bool = False) -> None:
        self.reloads.append(wait)
        if self.failure is not None:
            raise self.failure


@pytest.fixture
def running(monkeypatch: pytest.MonkeyPatch) -> _RecordingProvider:
    provider = _RecordingProvider()
    monkeypatch.setattr(
        engine_service, "peek_services", lambda: SimpleNamespace(provider=provider)
    )
    return provider


def test_use_and_prefer_reload_a_running_server_and_wait(
    with_mlx: None, running: _RecordingProvider
) -> None:
    engine_service.use(GGUF_REF, "llama.cpp")
    engine_service.prefer("llama.cpp")
    assert running.reloads == [True, True]


def test_a_refused_write_reloads_nothing(running: _RecordingProvider) -> None:
    with pytest.raises(engine_service.EngineRefusedError):
        engine_service.prefer("nonesuch")
    assert running.reloads == []


def test_with_no_services_built_a_write_runs_nothing(
    with_mlx: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_service, "peek_services", lambda: None)
    assert engine_service.prefer("llama.cpp") == "llama.cpp"


def test_a_failed_reload_keeps_the_saved_choice_and_tells_the_user(
    with_mlx: None, running: _RecordingProvider
) -> None:
    from lilbee.providers.base import ProviderError

    running.failure = ProviderError("engine refused", provider="llama-server")
    with pytest.raises(engine_service.EngineNotAppliedError, match="engine refused") as raised:
        engine_service.prefer("llama.cpp")
    assert "keeps the engine it started with" in str(raised.value)
    assert _stored(engine_service.PREFERRED_KEY) == "llama.cpp"
