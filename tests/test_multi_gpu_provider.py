"""Tests for FleetProvider and the multi-gpu factory branch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from lilbee.core.config import cfg
from lilbee.core.config.enums import LlmProvider
from lilbee.providers.multi_gpu import provider as prov_mod
from lilbee.providers.multi_gpu.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.multi_gpu.provider import FleetProvider, _least_in_flight
from lilbee.providers.worker.transport import WorkerRole


def _fake_client(in_flight: int = 0) -> MagicMock:
    client = MagicMock()
    client.in_flight = in_flight
    return client


def _provider_with_clients(clients: dict[WorkerRole, list[MagicMock]]) -> FleetProvider:
    p = FleetProvider()
    p._fleet = MagicMock()  # non-None: _server_clients won't try to build
    p._clients = clients  # type: ignore[assignment]
    p._local = MagicMock()
    return p


def test_least_in_flight_picks_minimum() -> None:
    busy, idle = _fake_client(5), _fake_client(1)
    assert _least_in_flight([busy, idle]) is idle


def test_chat_routes_to_least_busy_server() -> None:
    busy, idle = _fake_client(5), _fake_client(1)
    idle.chat.return_value = "from-server"
    p = _provider_with_clients({WorkerRole.CHAT: [busy, idle]})
    assert p.chat([{"role": "user", "content": "hi"}]) == "from-server"
    idle.chat.assert_called_once()
    busy.chat.assert_not_called()


def test_chat_falls_back_to_local_when_no_server() -> None:
    p = _provider_with_clients({})  # no chat server
    p._local.chat.return_value = "from-local"
    assert p.chat([{"role": "user", "content": "hi"}]) == "from-local"
    p._local.chat.assert_called_once()


def test_embed_routes_to_server_when_present() -> None:
    client = _fake_client()
    client.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.EMBED: [client]})
    assert p.embed(["a"]) == [[0.1]]


def test_embed_falls_back_to_local() -> None:
    p = _provider_with_clients({})
    p._local.embed.return_value = [[0.2]]
    assert p.embed(["a"]) == [[0.2]]


def test_rerank_vision_pdf_models_delegate_to_local() -> None:
    p = _provider_with_clients({})
    p.rerank("q", ["a"])
    p._local.rerank.assert_called_once_with("q", ["a"])
    p.supports_rerank()
    p._local.supports_rerank.assert_called_once()
    p.vision_ocr(b"png", "vmodel")
    p._local.vision_ocr.assert_called_once()
    p.list_models()
    p._local.list_models.assert_called_once()
    p.list_chat_models("openai")
    p._local.list_chat_models.assert_called_once_with("openai")
    p.pull_model("m")
    p._local.pull_model.assert_called_once()
    p.show_model("m")
    p._local.show_model.assert_called_once_with("m")
    p.get_capabilities("m")
    p._local.get_capabilities.assert_called_once_with("m")
    p.warm_up_pool()
    p._local.warm_up_pool.assert_called_once()


def test_pdf_ocr_delegates_to_local() -> None:
    p = _provider_with_clients({})
    p.pdf_ocr(Path("/x.pdf"), backend="tesseract")  # type: ignore[arg-type]
    p._local.pdf_ocr.assert_called_once()


def test_shutdown_tears_down_fleet_and_local() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    fleet = p._fleet
    local = p._local
    p.shutdown()
    fleet.shutdown.assert_called_once()
    local.shutdown.assert_called_once()
    assert p._fleet is None and p._clients == {}


def test_invalidate_load_cache_respawns_fleet_and_drops_local_cache() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    fleet = p._fleet
    p.invalidate_load_cache()
    fleet.shutdown.assert_called_once()
    assert p._fleet is None
    p._local.invalidate_load_cache.assert_called_once()


def test_server_clients_builds_fleet_once(monkeypatch) -> None:
    built = {WorkerRole.CHAT: [_fake_client()]}
    calls = {"n": 0}

    def _fake_build() -> tuple[object, dict]:
        calls["n"] += 1
        return MagicMock(), built

    monkeypatch.setattr(prov_mod, "_build_fleet", _fake_build)
    p = FleetProvider()
    assert p._server_clients(WorkerRole.CHAT) == built[WorkerRole.CHAT]
    p._server_clients(WorkerRole.EMBED)  # second call must not rebuild
    assert calls["n"] == 1


def test_local_provider_is_lazy_and_cached(monkeypatch) -> None:
    sentinel = MagicMock()
    monkeypatch.setattr("lilbee.providers.llama_cpp.LlamaCppProvider", lambda: sentinel)
    p = FleetProvider()
    assert p._local_provider() is sentinel
    assert p._local_provider() is sentinel  # cached


def test_factory_returns_fleet_provider_for_multi_gpu() -> None:
    from lilbee.providers.factory import create_provider

    cfg.llm_provider = LlmProvider.MULTI_GPU
    assert isinstance(create_provider(cfg), FleetProvider)


class TestBuildFleetWiring:
    def test_server_model_inputs_covers_chat_and_embed(self, monkeypatch) -> None:
        monkeypatch.setattr(
            prov_mod,
            "_estimate_role",
            lambda role, ref, **_k: ModelPlacementInput(role, 5 * 1024**3),
        )
        inputs, refs = prov_mod._server_model_inputs()
        assert {i.role for i in inputs} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert set(refs) == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_launch_for_builds_instance(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.provider.resolve_model_path",
            lambda ref: Path("/models/chat.gguf"),
        )
        monkeypatch.setattr(prov_mod, "pick_free_port", lambda: 42700)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = prov_mod._launch_for(plan, "ref", Path("/bin/llama-server"), Path("/data"))
        assert launch.role == WorkerRole.CHAT
        assert launch.port == 42700
        assert launch.port_file == Path("/data/llama-server-chat.port")
        assert "--model" in launch.argv

    def test_build_fleet_plans_and_starts(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.gpu_select.enumerate_gpu_vram",
            lambda: [(0, 24 * 1024**3)],
        )
        monkeypatch.setattr(
            prov_mod,
            "_server_model_inputs",
            lambda: ([ModelPlacementInput(WorkerRole.CHAT, 5 * 1024**3)], {WorkerRole.CHAT: "ref"}),
        )
        monkeypatch.setattr(
            prov_mod,
            "plan_placement",
            lambda inputs, devices: Placement(
                instances=(InstancePlan(WorkerRole.CHAT, (0,)),), in_process_roles=()
            ),
        )
        monkeypatch.setattr(
            prov_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(prov_mod, "_launch_for", lambda *a: MagicMock())
        started = {WorkerRole.CHAT: [_fake_client()]}
        monkeypatch.setattr(prov_mod.Fleet, "start", lambda self, launches: started)
        _fleet, clients = prov_mod._build_fleet()
        assert clients is started
