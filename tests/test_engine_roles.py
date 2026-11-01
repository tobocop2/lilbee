"""Each engine call site asks for the engine of its own role.

Two different adapters are installed. The model of one role names the mlx format, so only that
role runs on the mlx adapter. A site that asks for the wrong role receives the wrong adapter.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lilbee.app import engine_choice, placement
from lilbee.app import services as services_mod
from lilbee.app.engine_choice import EngineBook, engine_for_ref
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.commands import setup as setup_cmd
from lilbee.cli.launchers import server as launcher
from lilbee.core.config import Config, cfg
from lilbee.data import offload
from lilbee.data.ingest import fanout
from lilbee.data.ingest import pipeline as ingest_pipeline
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.roles import WorkerRole
from tests.conformance.fake_adapter import BundledFakeAdapter

MLX_REF = "org/repo#mlx"
GGUF_REF = "org/repo/file.gguf"
_FIELD = {
    WorkerRole.CHAT: "chat_model",
    WorkerRole.EMBED: "embedding_model",
    WorkerRole.VISION: "vision_model",
}


class _GgufEngine(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})
    served_roles = frozenset(WorkerRole)


class _MlxEngine(BundledFakeAdapter):
    name = "mlx"
    formats = frozenset({"mlx"})
    architectures = frozenset({"mlx_arch_a", "mlx_arch_b"})
    served_roles = frozenset(WorkerRole)


GGUF_ENGINE = _GgufEngine()
MLX_ENGINE = _MlxEngine()


@pytest.fixture(autouse=True)
def two_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a gguf engine and an mlx engine on a host that prefers mlx for mlx models."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GGUF_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, MLX_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setattr(
        engine_choice,
        "session_host",
        lambda: collect(system="Darwin", machine="arm64", cpu_info={}, nvml=None),
    )


def _config(mlx_role: WorkerRole | None) -> Config:
    config = Config()
    for role, field in _FIELD.items():
        setattr(config, field, MLX_REF if role is mlx_role else GGUF_REF)
    return config


def _tag(engine: Any) -> str:
    return "mlx" if engine is MLX_ENGINE else "gguf" if engine is GGUF_ENGINE else repr(engine)


def _expected(role: WorkerRole, mlx_role: WorkerRole) -> str:
    return "mlx" if role is mlx_role else "gguf"


ALL_ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION)


def test_the_book_answers_each_role_with_its_own_engine() -> None:
    for mlx_role in ALL_ROLES:
        book = EngineBook(_config(mlx_role))
        for role in ALL_ROLES:
            assert _tag(book.for_role(role).adapter) == _expected(role, mlx_role)


def test_the_book_follows_the_config_without_a_refresh() -> None:
    config = _config(None)
    book = EngineBook(config)
    assert _tag(book.for_role(WorkerRole.CHAT).adapter) == "gguf"
    config.chat_model = MLX_REF
    assert _tag(book.for_role(WorkerRole.CHAT).adapter) == "mlx"
    assert not hasattr(book, "refresh")


def test_placement_and_the_book_give_one_answer_after_a_config_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(None)
    book = EngineBook(config)
    monkeypatch.setattr(placement, "cfg", config)
    config.chat_model = MLX_REF
    assert (
        placement._planning_engines().adapters[WorkerRole.CHAT]
        is book.for_role(WorkerRole.CHAT).adapter
        is MLX_ENGINE
    )


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_placement_sizes_with_the_chat_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    monkeypatch.setattr(placement, "cfg", _config(mlx_role))
    assert _tag(placement._planning_engines().adapters[WorkerRole.CHAT]) == _expected(
        WorkerRole.CHAT, mlx_role
    )


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_provider_kwargs_read_the_chat_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(setup_cmd, "cfg", _config(mlx_role))
    monkeypatch.setattr(
        "lilbee.providers.fleet.planning.chat_cache_type_flag",
        lambda engine: seen.append(_tag(engine)),
    )
    setup_cmd._resolved_provider_kwargs()
    assert seen == [_expected(WorkerRole.CHAT, mlx_role)]


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
@pytest.mark.parametrize("role", (WorkerRole.CHAT, WorkerRole.EMBED))
def test_self_check_server_reads_the_engine_of_the_role_it_checks(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole, role: WorkerRole
) -> None:
    seen: list[str] = []

    def _launch(r: WorkerRole, path: Path, *, engine: Any) -> None:
        seen.append(_tag(engine))
        raise RuntimeError("stop")

    monkeypatch.setattr(setup_cmd, "cfg", _config(mlx_role))
    monkeypatch.setattr("lilbee.providers.fleet.planning.build_single_role_launch", _launch)
    with pytest.raises(RuntimeError, match="stop"):
        setup_cmd._self_check_server(role, Path("/x.gguf"))
    assert seen == [_expected(role, mlx_role)]


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_the_launcher_sizes_the_chat_window_with_the_chat_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    seen: list[str] = []
    config = _config(mlx_role)
    config.chat_model = GGUF_REF if mlx_role is not WorkerRole.CHAT else MLX_REF
    monkeypatch.setattr(launcher, "cfg", config)
    monkeypatch.setattr(
        "lilbee.providers.model_ref.parse_model_ref",
        lambda ref: SimpleNamespace(is_local=True),
    )
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda ref: Path(ref))
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda path: {})
    monkeypatch.setattr(
        "lilbee.providers.fleet.planning.plan_sizing_budget",
        lambda engine: seen.append(_tag(engine)) or 1,
    )
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_chat_ctx", lambda *a, **k: k["available_bytes"]
    )
    launcher.planned_chat_ctx()
    assert seen == [_expected(WorkerRole.CHAT, mlx_role)]


def _services(config: Config) -> SimpleNamespace:
    return SimpleNamespace(
        engines=EngineBook(config),
        provider=SimpleNamespace(vision_slot_capacity=lambda: None),
    )


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_fanout_counts_devices_of_the_embed_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(services_mod._state, "singleton", _services(_config(mlx_role)))
    monkeypatch.setattr(
        "lilbee.providers.fleet.replicas.gpu_device_count",
        lambda engine: seen.append(_tag(engine)) or 1,
    )
    fanout.plan_fanout()
    assert seen == [_expected(WorkerRole.EMBED, mlx_role)]


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_offload_counts_devices_of_the_embed_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(services_mod._state, "singleton", _services(_config(mlx_role)))
    monkeypatch.setattr(
        "lilbee.providers.fleet.replicas.gpu_device_count",
        lambda engine: seen.append(_tag(engine)) or 1,
    )
    offload.embed_inflight_target()
    assert seen == [_expected(WorkerRole.EMBED, mlx_role)]


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_vision_replicas_count_devices_of_the_vision_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    seen: list[str] = []
    services = _services(_config(mlx_role))
    monkeypatch.setattr(ingest_pipeline, "get_services", lambda: services)
    monkeypatch.setattr(
        "lilbee.providers.fleet.replicas.gpu_device_count",
        lambda engine: seen.append(_tag(engine)) or 1,
    )
    monkeypatch.setattr(
        ingest_pipeline, "ocr_backend", lambda: ingest_pipeline.OcrBackendUsed.VISION
    )
    ingest_pipeline._max_concurrent()
    assert seen == [_expected(WorkerRole.VISION, mlx_role)]


@pytest.mark.parametrize("mlx_role", ALL_ROLES)
def test_adaptive_admission_enumerates_devices_of_the_embed_engine(
    monkeypatch: pytest.MonkeyPatch, mlx_role: WorkerRole
) -> None:
    seen: list[str] = []
    services = _services(_config(mlx_role))
    monkeypatch.setattr(ingest_pipeline, "get_services", lambda: services)
    monkeypatch.setattr(ingest_pipeline, "profile_for", lambda model: object())
    monkeypatch.setattr(
        ingest_pipeline,
        "enumerate_fleet_devices",
        lambda engine: seen.append(_tag(engine)) or [],
    )
    ingest_pipeline._build_admission(1, [0])
    assert seen == [_expected(WorkerRole.EMBED, mlx_role)]


def _row(task: ModelTask) -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/embedder-GGUF",
        gguf_filename="model.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=task,
    )


def test_a_pull_uses_the_engine_of_the_format_of_the_model_not_of_a_role() -> None:
    config = _config(WorkerRole.CHAT)
    assert _tag(EngineBook(config).for_role(WorkerRole.CHAT).adapter) == "mlx"
    assert _tag(engine_for_ref("acme/embedder-GGUF", cfg)) == "gguf"
    assert _tag(engine_for_ref("acme/embedder/model.gguf", cfg)) == "gguf"
    assert _tag(engine_for_ref("acme/mlx-model#mlx", cfg)) == "mlx"
    assert _tag(engine_choice.engine_for_format(ModelFormat.GGUF, cfg)) == "gguf"


def test_a_pull_of_a_format_no_engine_loads_raises_the_user_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    with pytest.raises(NoEngineError, match="No installed engine loads the mlx format"):
        engine_for_ref("acme/mlx-model#mlx", cfg)


@pytest.mark.asyncio
async def test_the_mcp_pull_of_an_embedding_model_reports_the_gguf_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.catalog.compat import UnsupportedArchError
    from lilbee.mcp_server import model_pull

    def _raise(*args: object, **kwargs: object) -> object:
        raise UnsupportedArchError("acme/embedder-GGUF", "kimi_k2")

    monkeypatch.setattr("lilbee.app.models.pull_model_data", _raise)
    monkeypatch.setattr(
        "lilbee.mcp_server.get_services",
        lambda: SimpleNamespace(engines=EngineBook(_config(WorkerRole.CHAT))),
    )
    result = await model_pull(model="acme/embedder-GGUF")
    assert result["error"]["supported_examples"] == ["gguf_arch"]
    assert result["error"]["total_supported"] == 1


@pytest.mark.asyncio
async def test_the_models_handler_pull_of_an_embedding_model_reports_the_gguf_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litestar.exceptions import HTTPException

    from lilbee.catalog.compat import UnsupportedArchError
    from lilbee.server import handlers

    manager = SimpleNamespace(
        require_engine=lambda model: None,
        enforce_arch_compat=lambda model: (_ for _ in ()).throw(
            UnsupportedArchError("acme/embedder-GGUF", "kimi_k2")
        ),
    )
    monkeypatch.setattr(
        "lilbee.server.handlers.models.get_services",
        lambda: SimpleNamespace(
            model_manager=manager, engines=EngineBook(_config(WorkerRole.CHAT))
        ),
    )
    with pytest.raises(HTTPException) as excinfo:
        await handlers.enforce_pull_arch_compat("acme/embedder-GGUF", source="native")
    assert excinfo.value.extra["supported_examples"] == ["gguf_arch"]
    assert excinfo.value.extra["total_supported"] == 1


@pytest.mark.asyncio
async def test_the_catalog_install_of_an_embedding_model_resolves_a_file_with_the_gguf_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.app import ComposeResult
    from textual.widgets import Static

    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.task_queue import TaskQueue
    from tests._lilbee_app_test_host import LilbeeAppHost

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield Static("host")

    seen: list[str] = []

    def _resolve(entry: CatalogModel, engine: Any) -> str:
        seen.append(_tag(engine))
        return "model.gguf"

    monkeypatch.setattr("lilbee.cli.tui.screens.catalog.resolve_filename", _resolve)
    monkeypatch.setattr("lilbee.cli.tui.screens.catalog.download_bytes", lambda repo, filename: 1)
    monkeypatch.setattr("lilbee.cli.tui.screens.catalog.disk_shortfall", lambda *_a, **_kw: None)
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        pilot.app.task_bar.queue = TaskQueue()
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: None)
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: None)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.catalog.get_services",
            lambda: SimpleNamespace(engines=EngineBook(_config(WorkerRole.CHAT))),
        )
        screen._install_model(_row(ModelTask.EMBEDDING))
    assert seen == ["gguf"]


@pytest.fixture
def chat_without_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A container built while the chat model names a format no installed engine loads."""
    from lilbee.app.services import build_services
    from lilbee.core.config import cfg

    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "data")
    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "data" / "lancedb")
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "chat_model", MLX_REF)
    monkeypatch.setattr(cfg, "embedding_model", GGUF_REF)
    services = build_services(cfg)
    yield services
    services.store.close()


def test_the_container_builds_when_the_chat_model_has_no_engine(chat_without_engine: Any) -> None:
    assert chat_without_engine.engines.for_role(WorkerRole.EMBED).adapter is GGUF_ENGINE


def test_the_first_use_of_the_chat_role_raises_the_user_text(chat_without_engine: Any) -> None:
    with pytest.raises(NoEngineError, match="For chat: No installed engine loads the mlx format"):
        chat_without_engine.engines.for_role(WorkerRole.CHAT)
    with pytest.raises(NoEngineError, match="For chat: No installed engine loads the mlx format"):
        chat_without_engine.provider.chat([{"role": "user", "content": "hi"}])


def test_the_embed_offload_target_does_not_hide_a_missing_embed_engine(
    chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "embedding_model", MLX_REF)
    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    with pytest.raises(NoEngineError, match="For embed: No installed engine loads the mlx format"):
        offload.embed_inflight_target()


_NO_MLX_TEXT = "No installed engine loads the mlx format"


def test_a_missing_engine_is_a_provider_error() -> None:
    from lilbee.providers.base import ProviderError

    assert issubclass(NoEngineError, ProviderError)


def _assert_chat_is_reported_and_embed_is_planned(skipped: Any, not_installed: Any) -> None:
    assert [(s["role"], s["model"]) for s in skipped] == [("chat", MLX_REF)]
    assert _NO_MLX_TEXT in skipped[0]["notice"]
    assert "Run lilbee engine list" in skipped[0]["notice"]
    assert [(s["role"], s["model"]) for s in not_installed] == [("embed", GGUF_REF)]


def test_the_placement_read_reports_the_role_with_no_engine_and_plans_the_others(
    chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.app import placement

    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    view = placement.get_placement()
    assert [(s.role, s.model) for s in view.skipped_no_engine] == [(WorkerRole.CHAT, MLX_REF)]
    assert _NO_MLX_TEXT in view.skipped_no_engine[0].notice
    assert [(s.role, s.model) for s in view.skipped_not_installed] == [(WorkerRole.EMBED, GGUF_REF)]
    assert WorkerRole.CHAT not in {role.role for role in view.roles}


def test_the_placement_command_prints_the_notice_and_exits_clean(
    chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from lilbee.cli.placement import placement_app

    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    result = CliRunner().invoke(placement_app, ["show"])
    assert result.exit_code == 0
    assert "chat: For chat: No installed engine loads the mlx format of org/repo#mlx" in " ".join(
        result.output.split()
    )
    assert "Run lilbee engine list to see the installed engines." in " ".join(result.output.split())
    assert f"{GGUF_REF} not downloaded" in " ".join(result.output.split())
    assert "Traceback" not in result.output


def test_the_mcp_placement_tool_returns_the_notice_and_plans_the_others(
    chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.mcp_server import get_placement_tool

    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    result = get_placement_tool()
    assert "error" not in result
    _assert_chat_is_reported_and_embed_is_planned(
        result["skipped_no_engine"], result["skipped_not_installed"]
    )


def test_the_placement_route_returns_the_notice_with_status_200(
    chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.server.routes.placement import placement_route
    from tests._litestar_test_client import create_test_client

    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    with create_test_client([placement_route]) as client:
        response = client.get("/api/placement")
    assert response.status_code == 200
    body = response.json()
    _assert_chat_is_reported_and_embed_is_planned(
        body["skipped_no_engine"], body["skipped_not_installed"]
    )


def test_the_gpu_stream_route_starts_while_one_role_has_no_engine(
    chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.server import handlers
    from lilbee.server.routes.placement import gpu_stats_stream_route
    from tests._litestar_test_client import create_test_client

    async def _one_event(devices: Any) -> Any:
        yield 'event: gpu_stats\ndata: {"gpus": []}\n\n'

    monkeypatch.setattr(handlers, "gpu_stats_stream", _one_event)
    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    with create_test_client([gpu_stats_stream_route]) as client:
        response = client.get("/api/gpus/stream")
    assert response.status_code == 200


_NO_ENGINE_ANYWHERE = (
    "No engine is assigned to any role, so there is nothing to plan. "
    "Run lilbee engine list to see the installed engines."
)


@pytest.fixture
def no_role_has_an_engine(chat_without_engine: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The same container, with every configured role on a format no installed engine loads."""
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "embedding_model", MLX_REF)
    monkeypatch.setattr(cfg, "vision_model", MLX_REF)
    monkeypatch.setattr(cfg, "reranker_model", MLX_REF)
    monkeypatch.setattr(services_mod._state, "singleton", chat_without_engine)
    return chat_without_engine


def test_the_placement_command_refuses_when_no_role_has_an_engine(
    no_role_has_an_engine: Any,
) -> None:
    from typer.testing import CliRunner

    from lilbee.cli.placement import placement_app

    result = CliRunner().invoke(placement_app, ["show"])
    assert result.exit_code == 1
    assert _NO_ENGINE_ANYWHERE in " ".join(result.output.split())
    assert "Traceback" not in result.output


def test_the_mcp_placement_tool_refuses_when_no_role_has_an_engine(
    no_role_has_an_engine: Any,
) -> None:
    from lilbee.mcp_server import get_placement_tool

    assert _NO_ENGINE_ANYWHERE in get_placement_tool()["error"]


def test_the_placement_route_answers_503_when_no_role_has_an_engine(
    no_role_has_an_engine: Any,
) -> None:
    from lilbee.server.routes.placement import placement_route
    from tests._litestar_test_client import create_test_client

    with create_test_client([placement_route]) as client:
        response = client.get("/api/placement")
    assert response.status_code == 503
    assert _NO_ENGINE_ANYWHERE in response.text


def test_the_gpu_stream_route_answers_503_when_no_role_has_an_engine(
    no_role_has_an_engine: Any,
) -> None:
    from lilbee.server.routes.placement import gpu_stats_stream_route
    from tests._litestar_test_client import create_test_client

    with create_test_client([gpu_stats_stream_route]) as client:
        response = client.get("/api/gpus/stream")
    assert response.status_code == 503
    assert _NO_ENGINE_ANYWHERE in response.text
    assert "Traceback" not in response.text


@pytest.fixture
def no_mlx_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from lilbee.core.config import cfg

    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    return tmp_path / "models"


@pytest.fixture
def pulls(no_mlx_engine: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A real model manager in the services; every download that starts is recorded."""
    from lilbee.modelhub.model_manager.core import ModelManager

    started: list[str] = []
    manager = ModelManager(no_mlx_engine, cfg)
    services = SimpleNamespace(model_manager=manager)
    monkeypatch.setattr("lilbee.app.models.get_services", lambda: services)
    monkeypatch.setattr("lilbee.server.handlers.models.get_services", lambda: services)
    monkeypatch.setattr("lilbee.catalog.download_model", lambda *a, **k: started.append("download"))
    return started


async def test_the_mcp_pull_of_a_format_nothing_loads_starts_no_download(
    pulls: list[str], no_mlx_engine: Path
) -> None:
    from lilbee.mcp_server import model_pull

    result = await model_pull(MLX_REF, source="native")
    assert _NO_MLX_TEXT in result["error"]
    assert pulls == []
    assert not no_mlx_engine.exists()


def test_the_rest_pull_of_a_format_nothing_loads_is_refused_with_the_text(
    pulls: list[str], no_mlx_engine: Path
) -> None:
    from lilbee.server.routes.models import models_pull_route
    from tests._litestar_test_client import create_test_client

    with create_test_client([models_pull_route]) as client:
        for body in (
            {"model": MLX_REF},
            {"model": MLX_REF, "allow_unsupported": True},
        ):
            response = client.post("/api/models/pull", json=body)
            assert response.status_code == 409
            assert _NO_MLX_TEXT in response.text
    assert pulls == []
    assert not no_mlx_engine.exists()


def test_the_cli_pull_of_a_format_nothing_loads_prints_the_text(
    pulls: list[str], no_mlx_engine: Path
) -> None:
    from typer.testing import CliRunner

    from lilbee.cli import app

    result = CliRunner().invoke(app, ["model", "pull", MLX_REF])
    assert result.exit_code == 1
    assert _NO_MLX_TEXT in result.output
    assert pulls == []
    assert not no_mlx_engine.exists()


def test_the_task_bar_download_of_a_format_nothing_loads_fails_with_the_text(
    pulls: list[str], no_mlx_engine: Path
) -> None:
    from lilbee.cli.tui.widgets.task_bar_controller import _download_target

    with pytest.raises(RuntimeError, match=_NO_MLX_TEXT):
        _download_target(SimpleNamespace(), SimpleNamespace(pull_ref=MLX_REF, display_name="m"))  # type: ignore[arg-type]
    assert pulls == []
    assert not no_mlx_engine.exists()


async def test_the_catalog_install_of_a_format_nothing_loads_enqueues_nothing(
    no_mlx_engine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from tests.test_catalog_already_downloading import _App, _model

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        enqueued: list[str] = []
        notices: list[str] = []
        monkeypatch.setattr(screen, "_enqueue_download", lambda m: enqueued.append(m.hf_repo))
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: notices.append(str(message)))
        screen._install_model(dataclasses.replace(_model(), hf_repo=MLX_REF))
    assert enqueued == []
    assert any(_NO_MLX_TEXT in n for n in notices)
    assert not no_mlx_engine.exists()
