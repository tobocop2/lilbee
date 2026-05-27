"""Tests for FleetProvider and the fleet factory branch."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.provider import FleetProvider, _least_in_flight
from lilbee.providers.roles import WorkerRole

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
    from lilbee.providers.base import ChatResult, FinishReason

    busy, idle = _fake_client(5), _fake_client(1)
    result = ChatResult(text="from-server", tool_calls=(), finish_reason=FinishReason.STOP)
    idle.chat_result.return_value = result
    p = _provider_with_clients({WorkerRole.CHAT: [busy, idle]})
    assert p.chat([{"role": "user", "content": "hi"}]) == result
    idle.chat_result.assert_called_once()
    busy.chat_result.assert_not_called()


def test_chat_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})  # no healthy chat server, no in-process fallback
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p.chat([{"role": "user", "content": "hi"}])


def test_chat_translates_options_before_routing_to_server() -> None:
    # The fleet must apply the same option translation as in-process: num_predict
    # -> max_tokens (the server doesn't read num_predict) and drop the load-only
    # num_ctx. A raw passthrough would silently ignore the generation length.
    from lilbee.providers.base import ChatResult, FinishReason

    client = _fake_client(0)
    client.chat_result.return_value = ChatResult(
        text="ok", tool_calls=(), finish_reason=FinishReason.STOP
    )
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    p.chat(
        [{"role": "user", "content": "hi"}],
        options={"num_predict": 64, "temperature": 0.2, "num_ctx": 8192},
    )
    sent = client.chat_result.call_args.kwargs["options"]
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

    def _slow_build(*_listeners) -> object:
        calls["n"] += 1
        time.sleep(0.05)  # widen the race window between concurrent first calls
        return _fake_fleet({WorkerRole.CHAT: [client]})

    monkeypatch.setattr(planning_mod, "build_fleet", _slow_build)
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


def test_chat_streams_from_server() -> None:
    client = _fake_client(0)
    client.chat_stream_items.return_value = iter(["a", "b"])
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    assert list(p.chat([{"role": "user", "content": "hi"}], stream=True)) == ["a", "b"]
    client.chat_stream_items.assert_called_once()


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


_TOOL_AWARE_TEMPLATE = "{% if tools %}{{ tool_calls }}{% endif %}"
_PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


@pytest.fixture()
def _clear_tools_cache():
    """``_supports_tools_cached`` is module-level lru_cache; clear it per test so
    a True/False from one case can't leak into the next via a shared path key."""
    prov_mod._supports_tools_cached.cache_clear()
    yield
    prov_mod._supports_tools_cached.cache_clear()


def test_supports_tools_true_for_tool_aware_template(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": _TOOL_AWARE_TEMPLATE},
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is True


def test_supports_tools_false_for_plain_template(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": _PLAIN_TEMPLATE},
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is False


def test_supports_tools_false_when_metadata_unreadable(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: None)
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is False


def test_supports_tools_false_when_template_missing(monkeypatch, tmp_path, _clear_tools_cache):
    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: gguf)
    # Metadata present but no chat_template key -> template is not a str.
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "llama"}
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is False


def test_supports_tools_false_when_model_unresolved(monkeypatch, _clear_tools_cache):
    from lilbee.providers.base import ProviderError

    def _raise(_m: str) -> Path:
        raise ProviderError("not installed")

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().supports_tools("org/repo/missing.gguf") is False


def test_supports_tools_tolerates_unstattable_path(monkeypatch, _clear_tools_cache):
    """A resolved path whose ``stat()`` fails still probes (mtime falls back to 0)."""
    bogus = Path("/does/not/exist/chat.gguf")
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: bogus)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": _TOOL_AWARE_TEMPLATE},
    )
    assert FleetProvider().supports_tools("org/repo/chat.gguf") is True


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


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll *predicate* until true or *timeout*; generous so xdist load can't flake it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_warm_up_pool_builds_fleet_off_thread(monkeypatch) -> None:
    # The eager warm-up at TUI mount must not block the caller; it dispatches a
    # background build thread and returns immediately. The fleet appears once the
    # thread finishes.
    started = threading.Event()
    release = threading.Event()
    fleet = _fake_fleet({})

    def _slow_build(*_listeners) -> object:
        started.set()
        release.wait(timeout=5.0)  # hold the build so the call clearly returns first
        return fleet

    monkeypatch.setattr(planning_mod, "build_fleet", _slow_build)
    p = FleetProvider()
    p.warm_up_pool()
    assert started.wait(timeout=5.0)  # build runs on a background thread
    assert p._fleet is None  # warm_up_pool returned before the build completed
    release.set()
    assert _wait_until(lambda: p._fleet is fleet)


def test_warm_up_pool_single_flight_does_not_double_build(monkeypatch) -> None:
    calls = {"n": 0}
    in_build = threading.Event()
    release = threading.Event()
    fleet = _fake_fleet({})

    def _slow_build(*_listeners) -> object:
        calls["n"] += 1
        in_build.set()
        release.wait(timeout=5.0)
        return fleet

    monkeypatch.setattr(planning_mod, "build_fleet", _slow_build)
    p = FleetProvider()
    p.warm_up_pool()
    assert in_build.wait(timeout=5.0)  # first build is genuinely in flight
    p.warm_up_pool()  # second call while warming: must not start a second build
    release.set()
    assert _wait_until(lambda: p._fleet is fleet)
    assert calls["n"] == 1


def test_warm_up_pool_noop_when_fleet_already_up(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_build(*_listeners) -> object:
        calls["n"] += 1
        return _fake_fleet({})

    monkeypatch.setattr(planning_mod, "build_fleet", _fake_build)
    p = FleetProvider()
    p._fleet = _fake_fleet({})  # already built
    p.warm_up_pool()
    assert calls["n"] == 0  # no build dispatched


def test_warm_up_blocking_logs_and_clears_guard_on_failure(monkeypatch, caplog) -> None:
    def _boom(*_listeners) -> object:
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(planning_mod, "build_fleet", _boom)
    p = FleetProvider()
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.provider"):
        p._warm_up_blocking()  # runs the body synchronously for the assertion
    assert p._fleet is None
    assert p._warming is False  # guard cleared so a later warm-up can retry
    assert "warm-up failed" in caplog.text.lower()


def test_warm_up_blocking_discards_duplicate_when_fleet_raced(monkeypatch) -> None:
    # A concurrent _server_clients built a fleet while warm-up was loading; the
    # warm-up's duplicate must be shut down, not stranded (each holds GPU memory).
    winner = _fake_fleet({})
    duplicate = _fake_fleet({})
    monkeypatch.setattr(planning_mod, "build_fleet", lambda *_a: duplicate)
    p = FleetProvider()
    p._fleet = winner
    p._warm_up_blocking()
    assert p._fleet is winner
    duplicate.shutdown.assert_called_once()


def test_role_ready_false_without_fleet() -> None:
    assert FleetProvider().role_ready(WorkerRole.CHAT) is False


def test_role_ready_reflects_healthy_clients() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    assert p.role_ready(WorkerRole.CHAT) is True
    assert p.role_ready(WorkerRole.EMBED) is False


def test_drop_loaded_models_async_tears_down_off_thread(monkeypatch) -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    fleet = p._fleet
    p.drop_loaded_models_async()
    assert _wait_until(lambda: p._fleet is None)
    fleet.shutdown.assert_called_once()


def test_drop_loaded_models_async_noop_without_fleet() -> None:
    p = FleetProvider()  # _fleet is None
    p.drop_loaded_models_async()  # must not raise or spawn a thread
    assert p._fleet is None


def test_server_clients_builds_fleet_once(monkeypatch) -> None:
    calls = {"n": 0}
    fleet = _fake_fleet({WorkerRole.CHAT: [_fake_client()]})

    def _fake_build(*_listeners) -> object:
        calls["n"] += 1
        return fleet

    monkeypatch.setattr(planning_mod, "build_fleet", _fake_build)
    p = FleetProvider()
    assert len(p._server_clients(WorkerRole.CHAT)) == 1
    p._server_clients(WorkerRole.EMBED)  # second call must not rebuild
    assert calls["n"] == 1


def test_apply_fleet_gpu_env_skips_autoselect(monkeypatch) -> None:
    # The fleet selects devices via placement; the in-process single-device
    # Vulkan autoselect must NOT run here or it would pin one adapter and hide
    # the rest from placement.
    from lilbee.providers.fleet import gpu_env

    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr(
        "lilbee.providers.fleet.gpu_select.autoselect_best_gpu_index",
        lambda: pytest.fail("autoselect must not run for the fleet"),
    )
    gpu_env.apply_fleet_gpu_env()  # no autoselect call -> no failure


def test_apply_fleet_gpu_env_honors_gpu_devices_pin(monkeypatch) -> None:
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", "0")
    gpu_env.apply_fleet_gpu_env()
    for name in _GPU_VISIBLE_ENV_VARS:
        assert os.environ[name] == "0"


def test_routing_provider_local_engine_is_fleet() -> None:
    # The fleet is the sole local engine now: AUTO routes local refs through
    # RoutingProvider, whose local engine is a FleetProvider (a single machine
    # is a fleet-of-one). Construction is side-effect-free; no fleet is spawned.
    from lilbee.providers.routing_provider import RoutingProvider

    assert isinstance(RoutingProvider()._get_local(), FleetProvider)


class TestLifecycleMethods:
    def test_cancel_inference_is_noop(self) -> None:
        # llama-server stops on client disconnect; cancel has nothing to flip.
        assert _provider_with_clients({}).cancel_inference() is None

    def test_reload_role_noop_when_fleet_not_built(self, monkeypatch) -> None:
        spawned = {"thread": False}
        monkeypatch.setattr("threading.Thread", lambda *a, **k: spawned.__setitem__("thread", True))
        p = FleetProvider()  # _fleet is None
        p.reload_role(WorkerRole.EMBED)
        assert spawned["thread"] is False  # no background restart dispatched

    def test_reload_role_dispatches_background_restart(self) -> None:
        done = threading.Event()
        p = FleetProvider()
        p._fleet = _fake_fleet({})  # non-None so reload dispatches
        p._reload_role_blocking = lambda role: done.set()  # type: ignore[method-assign]
        p.reload_role(WorkerRole.EMBED)
        assert done.wait(timeout=2.0)  # the spawned thread ran the blocking restart

    def test_reload_role_blocking_restarts_only_that_role(self, monkeypatch) -> None:
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(
            prov_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(planning_mod, "resolve_devices", lambda _b: [device])
        launch = MagicMock()
        monkeypatch.setattr(planning_mod, "plan_launches", lambda roles, *_a: [launch])
        fleet = MagicMock()
        p = FleetProvider()
        p._fleet = fleet
        p._reload_role_blocking(WorkerRole.EMBED)
        fleet.restart_role.assert_called_once_with(WorkerRole.EMBED, [launch])

    def test_reload_role_blocking_noop_when_fleet_cleared(self, monkeypatch) -> None:
        monkeypatch.setattr(
            prov_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(planning_mod, "resolve_devices", lambda _b: [])
        monkeypatch.setattr(planning_mod, "plan_launches", lambda *_a: [])
        p = FleetProvider()  # _fleet stays None
        p._reload_role_blocking(WorkerRole.EMBED)  # must not raise

    def test_add_spawn_listener_stores_callbacks(self) -> None:
        p = FleetProvider()  # no fleet yet

        def on_spawning(_r: WorkerRole) -> None: ...

        def on_spawned(_r: WorkerRole) -> None: ...

        p.add_spawn_listener(on_spawning=on_spawning, on_spawned=on_spawned)
        assert p._on_spawning is on_spawning
        assert p._on_spawned is on_spawned

    def test_add_spawn_listener_attaches_to_running_fleet(self) -> None:
        fleet = MagicMock()
        p = FleetProvider()
        p._fleet = fleet

        def on_spawning(_r: WorkerRole) -> None: ...

        p.add_spawn_listener(on_spawning=on_spawning, on_spawned=None)
        fleet.set_listener.assert_called_once_with(on_spawning=on_spawning, on_spawned=None)


def test_get_capabilities_unresolved_model_returns_completion(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    def _raise(_m: str):
        raise ProviderError("not found")

    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().get_capabilities("missing/model.gguf") == ["completion"]


class TestChatWithTools:
    def test_routes_to_least_busy_chat_server(self) -> None:
        from lilbee.providers.base import ChatToolResult, ToolCall

        client = _fake_client(0)
        client.chat_tools.return_value = ChatToolResult(
            content="", tool_calls=[ToolCall("c1", "f", "{}")]
        )
        p = _provider_with_clients({WorkerRole.CHAT: [client]})
        result = p.chat_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
        )
        assert result.tool_calls[0].name == "f"
        client.chat_tools.assert_called_once()
        assert client.chat_tools.call_args.kwargs["tool_choice"] == "auto"

    def test_without_server_raises(self) -> None:
        from lilbee.providers.base import ProviderError

        p = _provider_with_clients({})
        with pytest.raises(ProviderError, match="No chat model server is running"):
            p.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])

    def test_model_override_raises(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        with pytest.raises(ProviderError, match="serves the configured chat model"):
            p.chat_with_tools([], tools=[], model="org/repo/other.gguf")
