"""The log line and the status fields name the engine that serves each request."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.app import presets
from lilbee.app.status import ServingStatus, chat_engine, serving_roles
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines import LLAMA, EngineName, choice
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.providers.serving import (
    PRESET_VERSION,
    REMOTE_VERSION,
    UNKNOWN_VERSION,
    log_name,
    served_by,
    serving_engine,
)
from lilbee.server import handlers
from tests.conftest import make_mock_services

NATIVE = "org/chat-GGUF/chat.gguf"
PIN = str(LLAMA.pin_record()["version"])


@pytest.fixture()
def preset_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "server_presets", None)
    presets.add_preset("box", "http://127.0.0.1:1", roles=("chat", "embed", "rerank", "vision"))


def _served_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("served by")]


def test_a_native_ref_is_served_by_the_pinned_engine() -> None:
    served = serving_engine(WorkerRole.CHAT, NATIVE, LLAMA)
    assert (served.engine, served.version) == ("llama.cpp", PIN)


def test_a_preset_ref_is_served_by_the_preset(preset_cfg: None) -> None:
    served = serving_engine(WorkerRole.CHAT, "box/model-a", LLAMA)
    assert (served.engine, served.version) == ("box", PRESET_VERSION)


def test_a_hosted_prefix_is_served_by_its_provider() -> None:
    served = serving_engine(WorkerRole.CHAT, "openai/gpt-4o", LLAMA)
    assert (served.engine, served.version) == ("openai", REMOTE_VERSION)


def test_a_mixed_plan_names_a_different_engine_for_each_role(
    preset_cfg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "box/chat-a")
    monkeypatch.setattr(cfg, "embedding_model", NATIVE)
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    assert serving_roles() == [
        ServingStatus(role=WorkerRole.CHAT, model="box/chat-a", engine="box", version="preset"),
        ServingStatus(role=WorkerRole.EMBED, model=NATIVE, engine="llama.cpp", version=PIN),
    ]
    assert chat_engine() == "box"


def test_a_role_with_no_model_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    assert serving_roles() == []
    assert chat_engine() is None


def _status_store() -> MagicMock:
    store = MagicMock()
    store.get_meta.return_value = None
    store.source_summary.return_value = MagicMock(newest=[], kinds={}, document_count=0)
    store.total_chunks = 0
    store.held_records.return_value = {}
    store.health_warnings.return_value = []
    store.upgrade_progress.return_value = None
    return store


async def test_health_and_status_carry_the_engine(
    preset_cfg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.app import services as svc_mod

    monkeypatch.setattr(cfg, "chat_model", "box/chat-a")
    monkeypatch.setattr(cfg, "embedding_model", NATIVE)
    svc_mod.set_services(make_mock_services(store=_status_store()))
    try:
        assert (await handlers.health()).chat_engine == "box"
        status = await handlers.status()
    finally:
        svc_mod.set_services(None)
    by_role = {row.role: (row.engine, row.version) for row in status.serving}
    assert by_role[WorkerRole.CHAT] == ("box", "preset")
    assert by_role[WorkerRole.EMBED] == ("llama.cpp", PIN)


def test_the_cli_status_lines_name_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.app.status import StatusConfig, StatusResult
    from lilbee.cli.helpers import render_status_result

    result = StatusResult(
        config=StatusConfig(documents_dir="d", data_dir="x", chat_model="m", embedding_model="e"),
        sources=[],
        document_count=0,
        total_chunks=0,
        serving=[ServingStatus(role=WorkerRole.CHAT, model="m", engine="box", version="preset")],
    )
    text = " ".join(str(line) for line in render_status_result(result))
    assert "Chat engine" in text
    assert "box preset" in text


def test_the_tui_config_section_names_the_engine(
    preset_cfg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.cli.tui.screens.status import _build_config_content

    monkeypatch.setattr(cfg, "chat_model", "box/chat-a")
    assert "Chat engine" in _build_config_content().plain
    assert "box preset" in _build_config_content().plain


def _provider(monkeypatch: pytest.MonkeyPatch) -> tuple[RoutingProvider, MagicMock, MagicMock]:
    provider = RoutingProvider(engine=lambda: LLAMA)
    local, sdk = MagicMock(), MagicMock()
    monkeypatch.setattr(provider, "_get_local", lambda: local)
    monkeypatch.setattr(provider, "_get_sdk_provider", lambda: sdk)
    return provider, local, sdk


def test_each_request_logs_the_engine_that_serves_it(
    preset_cfg: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _local, _sdk = _provider(monkeypatch)
    monkeypatch.setattr(cfg, "chat_model", NATIVE)
    monkeypatch.setattr(cfg, "embedding_model", "box/embed-a")
    monkeypatch.setattr(cfg, "reranker_model", "box/rerank-a")
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        provider.chat([], model="box/chat-a")
        provider.chat([], model=None)
        provider.chat_with_tools([], tools=[], model="box/chat-a")
        provider.embed(["x"])
        provider.rerank("q", ["a"])
        provider.vision_ocr(b"", "box/vision-a")
    assert _served_lines(caplog) == [
        "served by engine=box version=preset role=chat model=box/chat-a",
        f"served by engine=llama.cpp version={PIN} role=chat model={NATIVE}",
        "served by engine=box version=preset role=chat model=box/chat-a",
        "served by engine=box version=preset role=embed model=box/embed-a",
        "served by engine=box version=preset role=rerank model=box/rerank-a",
        "served by engine=box version=preset role=vision model=box/vision-a",
    ]


def test_a_native_rerank_logs_the_pinned_engine(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _local, _sdk = _provider(monkeypatch)
    monkeypatch.setattr(cfg, "reranker_model", "org/rerank-GGUF/rerank.gguf")
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        provider.rerank("q", ["a"])
    assert _served_lines(caplog) == [
        f"served by engine=llama.cpp version={PIN} role=rerank model=org/rerank-GGUF/rerank.gguf"
    ]


def test_the_placement_response_names_the_engine_of_each_role() -> None:
    from lilbee.app.placement import PlacementView, RolePlacementView
    from lilbee.server.models import PlacementResponse

    view = PlacementView(
        gpus=(),
        roles=(RolePlacementView(WorkerRole.CHAT, NATIVE, (0,), None, 1, "llama.cpp"),),
        unplaceable=(),
        manual=False,
        spec_json=None,
    )
    assert [r.engine for r in PlacementResponse.from_view(view).roles] == ["llama.cpp"]


HOSTED_RERANK = "cohere/rerank-v3"


def test_a_hosted_rerank_ref_with_an_unknown_prefix_gets_one_answer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _local, sdk = _provider(monkeypatch)
    sdk.supports_rerank.return_value = True
    monkeypatch.setattr(cfg, "chat_model", "")
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "reranker_model", HOSTED_RERANK)
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        provider.rerank("q", ["a"])
    assert _served_lines(caplog) == [
        f"served by engine=cohere version={REMOTE_VERSION} role=rerank model={HOSTED_RERANK}"
    ]
    assert serving_roles() == [
        ServingStatus(
            role=WorkerRole.RERANK, model=HOSTED_RERANK, engine="cohere", version=REMOTE_VERSION
        )
    ]


@pytest.mark.parametrize(
    "model",
    ["nosuchengine", HOSTED_RERANK, "ollama/x", "/abs/dir/m.gguf", NATIVE, "//"],
)
@pytest.mark.parametrize("role", list(WorkerRole))
def test_the_engine_of_a_ref_is_defined_for_every_ref(
    role: WorkerRole, model: str, preset_cfg: None
) -> None:
    served = serving_engine(role, model, LLAMA)
    assert served.engine
    assert served.version


def test_a_pin_that_cannot_be_read_is_reported_as_unknown() -> None:
    engine = MagicMock()
    engine.name = "llama.cpp"
    engine.pin_record.side_effect = OSError("gone")
    assert served_by(NATIVE, engine, native=True).version == UNKNOWN_VERSION


def test_chat_answers_when_the_pin_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    engine.name = "llama.cpp"
    engine.pin_record.side_effect = OSError("gone")
    engine.bundled = True
    engine.formats = frozenset({"gguf"})
    engine.served_roles = frozenset(WorkerRole)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, engine)
    provider = RoutingProvider(engine=lambda: engine)
    local = MagicMock()
    local.chat.return_value = "Rome"
    monkeypatch.setattr(provider, "_get_local", lambda: local)
    monkeypatch.setattr(cfg, "chat_model", NATIVE)
    assert provider.chat([], model=None) == "Rome"
    engine.pin_record.assert_called()


def test_a_local_path_is_logged_by_its_file_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _local, _sdk = _provider(monkeypatch)
    secret = "/Users/me/secret/dir/model.gguf"
    monkeypatch.setattr(cfg, "chat_model", secret)
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        provider.chat([], model=None)
    assert _served_lines(caplog) == [
        f"served by engine=llama.cpp version={PIN} role=chat model=model.gguf"
    ]
    assert log_name("C:\\models\\m.gguf") == "m.gguf"
    assert log_name("org/chat-GGUF/chat.gguf") == "org/chat-GGUF/chat.gguf"


async def test_every_surface_names_the_same_engine_for_one_state(
    preset_cfg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee import mcp_server
    from lilbee.app import placement as app_placement
    from lilbee.app import services as svc_mod
    from lilbee.app.status import gather_status
    from lilbee.cli.helpers import render_status_result
    from lilbee.cli.tui.screens.status import _build_config_content
    from lilbee.providers.fleet.devices import FleetDevice
    from lilbee.providers.fleet.placement import InstancePlan
    from lilbee.providers.fleet.planning import ResolvedPlacement

    monkeypatch.setattr(cfg, "chat_model", "box/chat-a")
    monkeypatch.setattr(cfg, "embedding_model", NATIVE)
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    resolved = ResolvedPlacement(
        devices=(FleetDevice("CUDA", 0, "GPU", 1, 1),),
        instances=(InstancePlan(role=WorkerRole.EMBED, devices=(0,), tensor_split=None),),
        unplaceable_roles=(),
        model_refs={WorkerRole.EMBED: NATIVE},
    )
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec, **_kw: resolved)
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    svc_mod.set_services(make_mock_services(store=_status_store()))
    try:
        rest = {r.role: r.engine for r in (await handlers.status()).serving}
        health = (await handlers.health()).chat_engine
        mcp = {r["role"]: r["engine"] for r in mcp_server.status()["serving"]}
        cli = " ".join(str(line) for line in render_status_result(gather_status()))
    finally:
        svc_mod.set_services(None)
    placed = {r.role: r.engine for r in app_placement.get_placement().roles}
    tui = _build_config_content().plain
    assert rest == {WorkerRole.CHAT: "box", WorkerRole.EMBED: "llama.cpp"}
    assert mcp == {"chat": "box", "embed": "llama.cpp"}
    assert health == "box"
    assert "box preset" in cli
    assert "box preset" in tui
    assert placed == {WorkerRole.EMBED: rest[WorkerRole.EMBED]}
