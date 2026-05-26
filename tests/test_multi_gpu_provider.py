"""Tests for FleetProvider and the multi-gpu factory branch."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
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


def test_chat_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})  # no healthy chat server, no in-process fallback
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p.chat([{"role": "user", "content": "hi"}])


def test_chat_translates_options_before_routing_to_server() -> None:
    # The fleet must apply the same option translation as in-process: num_predict
    # -> max_tokens (the server doesn't read num_predict) and drop the load-only
    # num_ctx. A raw passthrough would silently ignore the generation length.
    client = _fake_client(0)
    client.chat.return_value = "ok"
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    p.chat(
        [{"role": "user", "content": "hi"}],
        options={"num_predict": 64, "temperature": 0.2, "num_ctx": 8192},
    )
    sent = client.chat.call_args.kwargs["options"]
    assert sent["max_tokens"] == 64
    assert sent["temperature"] == 0.2
    assert "num_predict" not in sent
    assert "num_ctx" not in sent


def test_embed_routes_to_server_when_present() -> None:
    client = _fake_client()
    client.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.EMBED: [client]})
    assert p.embed(["a"]) == [[0.1]]


def test_embed_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError, match="No embed model server is running"):
        p.embed(["a"])


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


def test_rerank_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError, match="No rerank model server is running"):
        p.rerank("q", ["a"])


def test_vision_ocr_routes_to_fleet_for_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    client = _fake_client()
    client.chat.return_value = "ocr text"
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p.vision_ocr(b"png", "org/repo/v.gguf") == "ocr text"


def test_vision_ocr_model_override_raises(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client()]})
    # override != the server's configured vision model -> hard error (no fallback)
    with pytest.raises(ProviderError, match="serves the configured vision model"):
        p.vision_ocr(b"png", "org/repo/other.gguf")


def test_chat_model_override_raises(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    with pytest.raises(ProviderError, match="serves the configured chat model"):
        p.chat([{"role": "user", "content": "hi"}], model="org/repo/other.gguf")


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
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.find_mmproj_for_model",
        lambda _p: Path("/m/mmproj.gguf"),
    )
    assert prov_mod._vision_mmproj("ref") == Path("/m/mmproj.gguf")


def test_vision_mmproj_returns_none_when_absent(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )

    def _raise(_p: Path) -> Path:
        raise ProviderError("no mmproj")

    monkeypatch.setattr("lilbee.providers.gguf_meta.find_mmproj_for_model", _raise)
    assert prov_mod._vision_mmproj("ref") is None


def test_role_ctx_chat_honors_configured_num_ctx(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "num_ctx", 16384)
    assert prov_mod._role_ctx(WorkerRole.CHAT, Path("/m/c.gguf"), None) == 16384


def test_role_ctx_chat_uses_dynamic_picker_when_unset(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_chat_ctx", lambda _p, _m: 4096)
    assert prov_mod._role_ctx(WorkerRole.CHAT, Path("/m/c.gguf"), None) == 4096


def test_role_ctx_embed_uses_model_training_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.train_ctx_from_meta",
        lambda _meta, *, fallback, model_path: 512,
    )
    assert prov_mod._role_ctx(WorkerRole.EMBED, Path("/m/e.gguf"), {}) == 512


def test_role_gpu_layers_marks_embed_roles(monkeypatch) -> None:
    seen: dict[str, bool] = {}

    def _fake(*, embedding: bool) -> int:
        seen["embedding"] = embedding
        return 7

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_n_gpu_layers", _fake)
    assert prov_mod._role_gpu_layers(WorkerRole.RERANK) == 7
    assert seen["embedding"] is True  # rerank is embedding-class
    assert prov_mod._role_gpu_layers(WorkerRole.CHAT) == 7
    assert seen["embedding"] is False  # chat honors cfg.n_gpu_layers


def test_role_gpu_layers_vision_offloads_all_layers(monkeypatch) -> None:
    # The in-process mtmd loader hardcodes n_gpu_layers=-1; the fleet must too,
    # not honor cfg.n_gpu_layers for vision.
    seen: dict[str, bool] = {}

    def _fake(*, embedding: bool) -> int:
        seen["embedding"] = embedding
        return -1

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_n_gpu_layers", _fake)
    assert prov_mod._role_gpu_layers(WorkerRole.VISION) == -1
    assert seen["embedding"] is True  # vision => all layers


def test_role_ctx_vision_uses_vision_picker(monkeypatch) -> None:
    # Vision must use the vision loader's training-ctx picker, not cfg.num_ctx
    # or the chat-ctx dynamic picker.
    monkeypatch.setattr(cfg, "num_ctx", 16384)  # would be wrong for vision
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_vision_ctx", lambda _p: 4321)
    assert prov_mod._role_ctx(WorkerRole.VISION, Path("/m/v.gguf"), {}) == 4321


def test_flash_attn_flag_on_by_default(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "flash_attention", None)
    assert prov_mod._flash_attn_flag() == "on"
    monkeypatch.setattr(cfg, "flash_attention", True)
    assert prov_mod._flash_attn_flag() == "on"


def test_flash_attn_flag_off_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "flash_attention", False)
    assert prov_mod._flash_attn_flag() == "off"


def test_cache_type_flag_none_for_f16(monkeypatch) -> None:
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.F16)
    assert prov_mod._cache_type_flag() is None


def test_cache_type_flag_uses_enum_value(monkeypatch) -> None:
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
    assert prov_mod._cache_type_flag() == "q8_0"


def test_chat_streams_from_server() -> None:
    client = _fake_client(0)
    client.chat.return_value = iter(["a", "b"])
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    assert list(p.chat([{"role": "user", "content": "hi"}], stream=True)) == ["a", "b"]
    assert client.chat.call_args.kwargs["stream"] is True


def test_supports_rerank_always_true() -> None:
    # llama-server reranks any cross-encoder GGUF via --pooling rank.
    assert FleetProvider().supports_rerank() is True


def test_list_chat_models_empty() -> None:
    # The local engine has no frontier-provider catalog.
    assert FleetProvider().list_chat_models("openai") == []


def test_pull_model_not_supported() -> None:
    with pytest.raises(NotImplementedError, match="cannot pull"):
        FleetProvider().pull_model("org/repo/m.gguf")


def test_list_models_reads_registry(monkeypatch) -> None:
    services = MagicMock()
    manifest_a, manifest_b = MagicMock(), MagicMock()
    manifest_a.ref, manifest_b.ref = "z/repo/b.gguf", "a/repo/a.gguf"
    services.registry.list_installed.return_value = [manifest_a, manifest_b]
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: services)
    assert FleetProvider().list_models() == ["a/repo/a.gguf", "z/repo/b.gguf"]


def test_show_model_reads_gguf_headers(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _m: Path("/m/x.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "llama"}
    )
    assert FleetProvider().show_model("org/repo/x.gguf") == {"architecture": "llama"}


def test_show_model_returns_none_when_unresolved(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    def _raise(_m: str) -> Path:
        raise ProviderError("not installed")

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().show_model("org/repo/x.gguf") is None


def test_get_capabilities_rerank_ref(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: True)
    assert FleetProvider().get_capabilities("org/repo/rerank.gguf") == ["rerank"]


def test_get_capabilities_vision_when_mmproj_present(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _m: Path("/m/v.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.find_mmproj_for_model", lambda _p: Path("/m/mmproj.gguf")
    )
    assert FleetProvider().get_capabilities("org/repo/v.gguf") == ["completion", "vision"]


def test_get_capabilities_completion_only_without_mmproj(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _m: Path("/m/c.gguf")
    )

    def _no_mmproj(_p: Path) -> Path:
        raise ProviderError("no mmproj")

    monkeypatch.setattr("lilbee.providers.gguf_meta.find_mmproj_for_model", _no_mmproj)
    assert FleetProvider().get_capabilities("org/repo/c.gguf") == ["completion"]


def test_pdf_ocr_ocrs_each_page_over_vision_server(monkeypatch) -> None:
    from lilbee.runtime.progress import EventType
    from lilbee.vision import PageText

    client = _fake_client(0)
    client.chat.side_effect = ["page one", "page two"]
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    monkeypatch.setattr(cfg, "vision_model", "")  # empty model arg -> configured
    monkeypatch.setattr("lilbee.vision.pdf_page_count", lambda _p: 2)
    monkeypatch.setattr(
        "lilbee.vision.rasterize_pdf", lambda _p: iter([(0, b"png0"), (1, b"png1")])
    )
    events: list[tuple] = []
    result = p.pdf_ocr(
        Path("doc.pdf"),
        backend="vision",  # type: ignore[arg-type]
        on_progress=lambda etype, evt: events.append((etype, evt.page, evt.total_pages)),
    )
    assert result == [PageText(1, "page one"), PageText(2, "page two")]
    assert events == [(EventType.EXTRACT, 1, 2), (EventType.EXTRACT, 2, 2)]


def test_pdf_ocr_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError, match="No vision model server is running"):
        p.pdf_ocr(Path("doc.pdf"), backend="vision")  # type: ignore[arg-type]


def test_shutdown_tears_down_fleet() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    fleet = p._fleet
    p.shutdown()
    fleet.shutdown.assert_called_once()
    assert p._fleet is None


def test_invalidate_load_cache_respawns_fleet() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    fleet = p._fleet
    p.invalidate_load_cache()
    fleet.shutdown.assert_called_once()
    assert p._fleet is None


def test_warm_up_pool_builds_fleet_once(monkeypatch) -> None:
    calls = {"n": 0}
    fleet = _fake_fleet({})

    def _fake_build() -> object:
        calls["n"] += 1
        return fleet

    monkeypatch.setattr(prov_mod, "_build_fleet", _fake_build)
    p = FleetProvider()
    p.warm_up_pool()
    p.warm_up_pool()  # idempotent: fleet already up
    assert calls["n"] == 1


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


def test_routing_provider_local_engine_is_fleet() -> None:
    # The fleet is the sole local engine now: AUTO routes local refs through
    # RoutingProvider, whose local engine is a FleetProvider (a single machine
    # is a fleet-of-one). Construction is side-effect-free; no fleet is spawned.
    from lilbee.providers.routing_provider import RoutingProvider

    assert isinstance(RoutingProvider()._get_local(), FleetProvider)


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
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: mmproj)
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: 16)
        inp = prov_mod._estimate_role(WorkerRole.VISION, "ref", slots=1)
        assert inp.est_vram_bytes >= 1500  # weights + mmproj counted

    def test_estimate_role_aux_kv_uses_f16_not_configured_type(self, tmp_path, monkeypatch) -> None:
        # Aux roles run f16 KV regardless of cfg.kv_cache_type, so the estimate
        # must use f16 to match runtime (only chat passes --cache-type).
        from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType

        model = tmp_path / "e.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata",
            lambda _p: {"block_count": "8", "embedding_length": "16"},
        )
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: 512)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)  # would be wrong for embed
        inp = prov_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1)
        f16 = KV_CACHE_TYPE_BYTES[KvCacheType.F16]
        expected_kv = 2 * 8 * 16 * 512 * 1 * f16
        assert inp.est_vram_bytes == 1000 + expected_kv + 1024**3  # weights + f16 KV + overhead

    def test_launch_for_vision_passes_mmproj(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _r: Path("/m/v.gguf"),
        )
        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: 4096)
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
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _ref: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata",
            lambda _p: {"block_count": "4", "embedding_length": "8"},
        )
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: 16)
        inp = prov_mod._estimate_role(WorkerRole.CHAT, "ref", slots=2)
        assert inp.role == WorkerRole.CHAT
        assert inp.est_vram_bytes > 1000  # weights + kv + overhead

    def test_launch_for_builds_instance_with_pinning(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda ref: Path("/models/chat.gguf"),
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: 4096)
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

    def _launch_role(self, monkeypatch, role: WorkerRole, ctx: int = 4096) -> list[str]:
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _r: Path("/models/m.gguf"),
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: ctx)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=role, devices=(0,))
        launch = prov_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        return launch.argv

    def test_launch_for_chat_sets_flash_and_cache_type(self, monkeypatch) -> None:
        from lilbee.core.config.enums import KvCacheType

        monkeypatch.setattr(cfg, "flash_attention", None)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
        argv = self._launch_role(monkeypatch, WorkerRole.CHAT)
        assert argv[argv.index("--flash-attn") + 1] == "on"
        assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
        assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
        assert "--batch-size" not in argv  # chat is not an embedding role
        assert "--threads" not in argv

    def test_launch_for_chat_f16_omits_cache_type(self, monkeypatch) -> None:
        from lilbee.core.config.enums import KvCacheType

        monkeypatch.setattr(cfg, "flash_attention", False)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.F16)
        argv = self._launch_role(monkeypatch, WorkerRole.CHAT)
        assert argv[argv.index("--flash-attn") + 1] == "off"
        assert "--cache-type-k" not in argv

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_launch_for_embed_roles_raise_batch_to_ctx(self, monkeypatch, role) -> None:
        argv = self._launch_role(monkeypatch, role, ctx=8192)
        # full-context embeddings: both batch and ubatch raised (server caps at ubatch)
        assert argv[argv.index("--batch-size") + 1] == "8192"
        assert argv[argv.index("--ubatch-size") + 1] == "8192"
        assert "--flash-attn" not in argv  # embedding path applies no flash attn
        assert "--cache-type-k" not in argv

    def _launch_for_role(self, monkeypatch, role: WorkerRole, ctx: int = 4096):
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _r: Path("/models/m.gguf"),
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(prov_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        monkeypatch.setattr(prov_mod, "_role_ctx", lambda _r, _p, _m: ctx)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=role, devices=(0,))
        return prov_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_launch_for_embed_roles_set_token_cap(self, monkeypatch, role) -> None:
        launch = self._launch_for_role(monkeypatch, role, ctx=8192)
        assert launch.token_cap == 8192  # embed/rerank truncate to per-slot ctx

    @pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.VISION])
    def test_launch_for_non_embed_roles_have_no_token_cap(self, monkeypatch, role) -> None:
        launch = self._launch_for_role(monkeypatch, role)
        assert launch.token_cap is None

    def test_launch_for_vision_sets_full_core_threads(self, monkeypatch) -> None:
        monkeypatch.setattr(prov_mod.os, "cpu_count", lambda: 12)
        argv = self._launch_role(monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--threads") + 1] == "12"
        assert argv[argv.index("--threads-batch") + 1] == "12"
        assert "--flash-attn" not in argv  # vision loader applies no flash attn
        assert "--batch-size" not in argv

    def test_launch_for_vision_threads_floor_when_cpu_count_unknown(self, monkeypatch) -> None:
        monkeypatch.setattr(prov_mod.os, "cpu_count", lambda: None)
        argv = self._launch_role(monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--threads") + 1] == str(prov_mod._DEFAULT_THREADS)

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
            "lilbee.providers.multi_gpu.gpu_select.enumerate_gpu_vram",
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
