"""Tests for FleetProvider and the multi-gpu factory branch."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import LlmProvider
from lilbee.providers.multi_gpu import provider as prov_mod
from lilbee.providers.multi_gpu.devices import FleetDevice, visible_env
from lilbee.providers.multi_gpu.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.multi_gpu.provider import FleetProvider, _least_in_flight
from lilbee.providers.worker.transport import WorkerRole

_GB = 1024**3


def _fake_client(in_flight: int = 0) -> MagicMock:
    client = MagicMock()
    client.in_flight = in_flight
    return client


def _fake_fleet(clients: dict[WorkerRole, list[MagicMock]]) -> MagicMock:
    fleet = MagicMock()
    fleet.healthy_clients.side_effect = lambda role: clients.get(role, [])
    return fleet


def _provider_with_clients(clients: dict[WorkerRole, list[MagicMock]]) -> FleetProvider:
    p = FleetProvider()
    p._fleet = _fake_fleet(clients)  # non-None: _server_clients won't try to build
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


def test_chat_falls_back_to_local_when_no_healthy_server() -> None:
    p = _provider_with_clients({})  # no healthy chat server
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


def test_concurrent_first_requests_build_fleet_once(monkeypatch) -> None:
    calls = {"n": 0}
    client = _fake_client()
    client.chat.return_value = "ok"

    def _slow_build() -> object:
        calls["n"] += 1
        time.sleep(0.05)  # widen the race window between concurrent first calls
        return _fake_fleet({WorkerRole.CHAT: [client]})

    monkeypatch.setattr(prov_mod, "_build_fleet", _slow_build)
    p = FleetProvider()
    p._local = MagicMock()
    barrier = threading.Barrier(8)

    def _hit() -> None:
        barrier.wait()  # release all threads at once
        p.chat([{"role": "user", "content": "hi"}])

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1  # single-flight: 8 concurrent first-requests build one fleet


def test_rerank_routes_to_fleet() -> None:
    client = _fake_client()
    client.rerank.return_value = [0.9, 0.1]
    p = _provider_with_clients({WorkerRole.RERANK: [client]})
    assert p.rerank("q", ["a", "b"]) == [0.9, 0.1]
    client.rerank.assert_called_once_with("q", ["a", "b"])


def test_rerank_falls_back_to_local_without_server() -> None:
    p = _provider_with_clients({})
    p._local.rerank.return_value = [0.5]
    assert p.rerank("q", ["a"]) == [0.5]


def test_vision_ocr_routes_to_fleet_for_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    client = _fake_client()
    client.chat.return_value = "ocr text"
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p.vision_ocr(b"png", "org/repo/v.gguf") == "ocr text"


def test_vision_ocr_falls_back_to_local_for_model_override(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client()]})
    p._local.vision_ocr.return_value = "local-ocr"
    # override != the server's configured vision model -> in-process
    assert p.vision_ocr(b"png", "org/repo/other.gguf") == "local-ocr"
    p._local.vision_ocr.assert_called_once()


def test_chat_with_model_override_uses_local(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._local.chat.return_value = "local"
    assert p.chat([{"role": "user", "content": "hi"}], model="org/repo/other.gguf") == "local"
    p._local.chat.assert_called_once()


def test_vision_call_returns_text() -> None:
    client = _fake_client()
    client.chat.return_value = "OCR text"
    assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None) == "OCR text"


def test_vision_call_enforces_timeout() -> None:
    client = _fake_client()
    client.chat.return_value = "OCR text"
    assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 5.0) == "OCR text"
    client.chat.assert_called_once()


def test_vision_call_rejects_non_text() -> None:
    from lilbee.providers.base import ProviderError

    client = _fake_client()
    client.chat.return_value = iter(["streamed"])  # not a str
    with pytest.raises(ProviderError, match="expected text"):
        prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None)


def test_vision_mmproj_returns_path_when_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.gguf_meta.find_mmproj_for_model",
        lambda _p: Path("/m/mmproj.gguf"),
    )
    assert prov_mod._vision_mmproj("ref") == Path("/m/mmproj.gguf")


def test_vision_mmproj_returns_none_when_absent(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )

    def _raise(_p: Path) -> Path:
        raise ProviderError("no mmproj")

    monkeypatch.setattr("lilbee.providers.llama_cpp.gguf_meta.find_mmproj_for_model", _raise)
    assert prov_mod._vision_mmproj("ref") is None


def test_chat_local_stream_path() -> None:
    p = _provider_with_clients({})  # no chat server -> local
    p._local.chat.return_value = iter(["a", "b"])
    list(p.chat([{"role": "user", "content": "hi"}], stream=True))
    assert p._local.chat.call_args.kwargs["stream"] is True


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
    assert p._fleet is None


def test_invalidate_load_cache_respawns_fleet_and_drops_local_cache() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    fleet = p._fleet
    p.invalidate_load_cache()
    fleet.shutdown.assert_called_once()
    assert p._fleet is None
    p._local.invalidate_load_cache.assert_called_once()


def test_server_clients_builds_fleet_once(monkeypatch) -> None:
    calls = {"n": 0}
    fleet = _fake_fleet({WorkerRole.CHAT: [_fake_client()]})

    def _fake_build() -> object:
        calls["n"] += 1
        return fleet

    monkeypatch.setattr(prov_mod, "_build_fleet", _fake_build)
    p = FleetProvider()
    assert len(p._server_clients(WorkerRole.CHAT)) == 1
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
    try:
        assert isinstance(create_provider(cfg), FleetProvider)
    finally:
        cfg.llm_provider = LlmProvider.AUTO


class TestBuildFleetWiring:
    def test_server_model_inputs_skips_unconfigured_optional_roles(self, monkeypatch) -> None:
        monkeypatch.setattr(
            prov_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, 5 * _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "")  # unconfigured -> skipped
        monkeypatch.setattr(cfg, "vision_model", "")
        inputs, refs = prov_mod._server_model_inputs()
        assert {i.role for i in inputs} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert set(refs) == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_server_model_inputs_includes_configured_rerank(self, monkeypatch) -> None:
        monkeypatch.setattr(
            prov_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "some/reranker.gguf")
        monkeypatch.setattr(cfg, "vision_model", "")
        _inputs, refs = prov_mod._server_model_inputs()
        assert WorkerRole.RERANK in refs

    def test_server_model_inputs_includes_vision_only_with_mmproj(self, monkeypatch) -> None:
        monkeypatch.setattr(
            prov_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "some/vision.gguf")

        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: None)
        assert WorkerRole.VISION not in prov_mod._server_model_inputs()[1]

        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        assert WorkerRole.VISION in prov_mod._server_model_inputs()[1]

    def test_estimate_role_vision_adds_mmproj_size(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"y" * 500)
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.provider.resolve_model_path", lambda _r: model
        )
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.gguf_meta.read_gguf_metadata", lambda _p: {}
        )
        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: mmproj)
        inp = prov_mod._estimate_role(WorkerRole.VISION, "ref", slots=1, ctx=16)
        assert inp.est_vram_bytes >= 1500  # weights + mmproj counted

    def test_launch_for_vision_passes_mmproj(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.provider.resolve_model_path",
            lambda _r: Path("/m/v.gguf"),
        )
        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        plan = InstancePlan(role=WorkerRole.VISION, devices=(0,))
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        launch = prov_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        assert "--mmproj" in launch.argv
        assert str(Path("/m/mmproj.gguf")) in launch.argv

    def test_estimate_role_reads_weights_and_meta(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.provider.resolve_model_path", lambda _ref: model
        )
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.gguf_meta.read_gguf_metadata",
            lambda _p: {"block_count": "4", "embedding_length": "8"},
        )
        inp = prov_mod._estimate_role(WorkerRole.CHAT, "ref", slots=2, ctx=16)
        assert inp.role == WorkerRole.CHAT
        assert inp.est_vram_bytes > 1000  # weights + kv + overhead

    def test_launch_for_builds_instance_with_pinning(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.provider.resolve_model_path",
            lambda ref: Path("/models/chat.gguf"),
        )
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = prov_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        assert launch.role == WorkerRole.CHAT
        assert launch.env_overrides == visible_env((device,))
        # port file is stamped with the owning pid so reaping is instance-safe
        assert launch.port_file == Path(f"/data/llama-server-chat-{os.getpid()}.port")
        assert "--model" in launch.argv
        assert "--port" not in launch.argv  # claimed at spawn, not here

    def test_build_fleet_resolves_devices_plans_and_starts(self, monkeypatch) -> None:
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(
            prov_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(prov_mod, "probe_devices", lambda _binary: [device])
        monkeypatch.setattr(
            prov_mod,
            "_server_model_inputs",
            lambda: ([ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)], {WorkerRole.CHAT: "ref"}),
        )
        monkeypatch.setattr(
            prov_mod,
            "plan_placement",
            lambda inputs, devices: Placement(
                instances=(InstancePlan(WorkerRole.CHAT, (0,)),), in_process_roles=()
            ),
        )
        monkeypatch.setattr(prov_mod, "_launch_for", lambda *a: MagicMock())
        started = {"n": 0}
        monkeypatch.setattr(
            prov_mod.Fleet, "start", lambda self, launches: started.__setitem__("n", 1)
        )
        fleet = prov_mod._build_fleet()
        assert isinstance(fleet, prov_mod.Fleet)
        assert started["n"] == 1

    def test_build_fleet_falls_back_to_vulkan_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(
            prov_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(prov_mod, "probe_devices", lambda _binary: [])  # binary can't enumerate
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.gpu_select.enumerate_gpu_vram",
            lambda: [(0, 24 * _GB)],
        )
        seen: dict[str, list] = {}
        monkeypatch.setattr(
            prov_mod,
            "_server_model_inputs",
            lambda: ([ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)], {WorkerRole.CHAT: "ref"}),
        )

        def _capture(inputs, devices):
            seen["devices"] = devices
            return Placement(instances=(), in_process_roles=(WorkerRole.CHAT,))

        monkeypatch.setattr(prov_mod, "plan_placement", _capture)
        monkeypatch.setattr(prov_mod.Fleet, "start", lambda self, launches: None)
        prov_mod._build_fleet()
        assert seen["devices"] == [(0, 24 * _GB)]  # synthesized from the Vulkan fallback
