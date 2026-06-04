"""Tests for FleetProvider routing and llama-swap lifecycle."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.provider import FleetProvider, _least_in_flight
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


def _fake_client(in_flight: int = 0) -> MagicMock:
    client = MagicMock()
    client.in_flight = in_flight
    return client


def _fake_launch(role: WorkerRole, *, slots: int = 1, ctx: int = 0) -> MagicMock:
    launch = MagicMock()
    launch.role = role
    launch.slots = slots
    launch.ctx = ctx
    return launch


class _FakeSwap:
    """A stand-in SwapManager recording lifecycle calls; ready roles are settable."""

    def __init__(self) -> None:
        self.started: list[list] = []
        self.reloads = 0
        self.shutdowns = 0
        self.ready: set[WorkerRole] = set()

    def start(self, launches: list) -> None:
        self.started.append(launches)

    def endpoint(self) -> str:
        return "http://fake-endpoint"

    def role_ready(self, role: WorkerRole) -> bool:
        return role in self.ready

    def reload(self, launches: list) -> None:
        self.reloads += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


def _install_engine(monkeypatch, *, launches: list, swap: _FakeSwap | None = None) -> _FakeSwap:
    """Patch the swap, client, and planner so _ensure_swap builds controllable fakes."""
    swap = swap or _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir: swap)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _endpoint, _model: _fake_client())
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: launches)
    return swap


def _provider_with_clients(clients: dict[WorkerRole, list[MagicMock]]) -> FleetProvider:
    """A provider with a fake swap already up and one client per role (no real start)."""
    p = FleetProvider()
    p._swap = _FakeSwap()  # non-None so _ensure_swap short-circuits
    p._clients = {role: cs[0] for role, cs in clients.items() if cs}
    return p


def test_least_in_flight_picks_minimum() -> None:
    busy, idle = _fake_client(5), _fake_client(1)
    assert _least_in_flight([busy, idle]) is idle


def test_chat_routes_to_chat_server() -> None:
    from lilbee.providers.base import ChatResult, FinishReason

    client = _fake_client()
    result = ChatResult(text="from-server", tool_calls=(), finish_reason=FinishReason.STOP)
    client.chat_result.return_value = result
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    assert p.chat([{"role": "user", "content": "hi"}]) == result
    client.chat_result.assert_called_once()


def test_chat_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})  # no chat client, no in-process fallback
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p.chat([{"role": "user", "content": "hi"}])


def test_chat_translates_options_before_routing_to_server() -> None:
    # The engine must apply the same option translation as in-process: num_predict
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


def test_rerank_routes_to_engine() -> None:
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


def test_vision_ocr_routes_to_engine_for_configured_model(monkeypatch) -> None:
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


def test_vision_call_caps_output_tokens(monkeypatch) -> None:
    # A runaway OCR page can loop to tens of thousands of chars and dominate a
    # scan's time; the call must cap generation at cfg.vision_ocr_max_tokens.
    monkeypatch.setattr(cfg, "vision_ocr_max_tokens", 4096)
    client = _fake_client()
    client.chat.return_value = "OCR text"
    prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None)
    assert client.chat.call_args.kwargs["options"] == {"max_tokens": 4096}


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
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 1)  # sequential: side_effect by call order
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


def test_pdf_ocr_runs_pages_concurrently_and_preserves_order(monkeypatch) -> None:
    # OCR fans pages across the vision server's batching slots; results must still
    # come back in page order, and more than one page must be in flight at once.
    import threading
    import time as _time

    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    n = 8
    monkeypatch.setattr("lilbee.vision.pdf_page_count", lambda _p: n)
    monkeypatch.setattr(
        "lilbee.vision.rasterize_pdf", lambda _p: iter([(i, f"png{i}".encode()) for i in range(n)])
    )
    lock = threading.Lock()
    inflight = {"now": 0, "max": 0}

    def _vision(_client, _messages, _timeout):
        with lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        _time.sleep(0.02)
        with lock:
            inflight["now"] -= 1
        return "ocr"

    monkeypatch.setattr(prov_mod, "_vision_call", _vision)
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client(0)]})
    result = p.pdf_ocr(Path("doc.pdf"), backend="vision")  # type: ignore[arg-type]
    assert [pt.page for pt in result] == list(range(1, n + 1))  # reassembled in order
    assert inflight["max"] >= 2  # pages ran concurrently, not one at a time


def test_pdf_drain_budget_totals_pages_plus_load_grace(monkeypatch) -> None:
    """Budget is one document-wide pool: pages*per_page + load grace, else uncapped."""
    monkeypatch.setattr(cfg, "vision_load_budget_s", 300.0)
    assert prov_mod._pdf_drain_budget(2, 120.0) == 540.0
    assert prov_mod._pdf_drain_budget(5, None) is None
    assert prov_mod._pdf_drain_budget(5, 0.0) is None


def test_pdf_ocr_spends_one_document_budget_across_pages(monkeypatch) -> None:
    """Each page gets the remaining doc budget, not a fixed per-page cap."""
    from lilbee.vision import PageText

    p = _provider_with_clients({WorkerRole.VISION: [_fake_client(0)]})
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "vision_load_budget_s", 300.0)
    monkeypatch.setattr("lilbee.vision.pdf_page_count", lambda _p: 2)
    monkeypatch.setattr(
        "lilbee.vision.rasterize_pdf", lambda _p: iter([(0, b"png0"), (1, b"png1")])
    )
    seen: list[float | None] = []

    def _capture(_client, _messages, timeout):
        seen.append(timeout)
        return "ocr"

    monkeypatch.setattr(prov_mod, "_vision_call", _capture)
    result = p.pdf_ocr(Path("doc.pdf"), backend="vision", per_page_timeout_s=120.0)  # type: ignore[arg-type]
    assert result == [PageText(1, "ocr"), PageText(2, "ocr")]
    # Budget is 2*120 + 300 = 540; both pages draw from it (far above any 120 cap),
    # and the second page sees no more than the first since time only moves forward.
    assert seen[0] == pytest.approx(540.0, abs=1.0)
    assert seen[1] is not None and seen[0] is not None and seen[1] <= seen[0]
    assert all(t is not None and t > 120.0 for t in seen)


def test_pdf_ocr_without_per_page_timeout_runs_uncapped(monkeypatch) -> None:
    """No per-page cap means an uncapped (None) budget on every page."""
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client(0)]})
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr("lilbee.vision.pdf_page_count", lambda _p: 1)
    monkeypatch.setattr("lilbee.vision.rasterize_pdf", lambda _p: iter([(0, b"png0")]))
    seen: list[float | None] = []
    monkeypatch.setattr(prov_mod, "_vision_call", lambda *a: seen.append(a[2]) or "ocr")
    p.pdf_ocr(Path("doc.pdf"), backend="vision", per_page_timeout_s=None)  # type: ignore[arg-type]
    assert seen == [None]


def test_pdf_ocr_without_server_raises() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({})
    with pytest.raises(ProviderError, match="No vision model server is running"):
        p.pdf_ocr(Path("doc.pdf"), backend="vision")  # type: ignore[arg-type]


# --- llama-swap lifecycle ----------------------------------------------------


def test_ensure_swap_starts_once_and_builds_clients(monkeypatch) -> None:
    launches = [_fake_launch(WorkerRole.CHAT, slots=4, ctx=32768), _fake_launch(WorkerRole.EMBED)]
    swap = _install_engine(monkeypatch, launches=launches)
    p = FleetProvider()
    assert p._ensure_swap() is swap
    assert len(swap.started) == 1  # the swap was started with the planned launches
    assert set(p._clients) == {WorkerRole.CHAT, WorkerRole.EMBED}  # one client per placed role
    assert p._chat_slots == 4  # chat capacity / ctx taken from the chat launch
    assert p._chat_ctx == 32768
    p._ensure_swap()  # second call reuses the running swap
    assert len(swap.started) == 1


def test_ensure_swap_defaults_chat_slots_without_chat_launch(monkeypatch) -> None:
    _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.EMBED)])
    p = FleetProvider()
    p._ensure_swap()
    assert p._chat_slots == 1  # no chat launch -> default capacity
    assert p._chat_ctx is None


def test_ensure_swap_spawns_nothing_when_no_models(monkeypatch) -> None:
    # No configured/installed model -> no launches -> no swap process at all
    # (matches the old supervisor, which spawned nothing for an empty launch set).
    started = {"swaps": 0}

    class _CountingSwap(_FakeSwap):
        def start(self, launches: list) -> None:
            started["swaps"] += 1
            super().start(launches)

    _install_engine(monkeypatch, launches=[], swap=_CountingSwap())
    p = FleetProvider()
    assert p._ensure_swap() is None
    assert started["swaps"] == 0  # never started
    assert p._swap is None
    assert p._clients == {}


def test_chat_starts_swap_on_first_use(monkeypatch) -> None:
    from lilbee.providers.base import ChatResult, FinishReason

    captured: dict[str, MagicMock] = {}

    def _make_client(_endpoint, model):
        client = _fake_client()
        client.chat_result.return_value = ChatResult(
            text="ok", tool_calls=(), finish_reason=FinishReason.STOP
        )
        captured[model] = client
        return client

    swap = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d: swap)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", _make_client)
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    assert p.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert len(swap.started) == 1  # routing the first chat started the swap


def test_concurrent_first_requests_start_swap_once(monkeypatch) -> None:
    starts = {"n": 0}

    class _SlowSwap(_FakeSwap):
        def start(self, launches: list) -> None:
            starts["n"] += 1
            time.sleep(0.05)  # widen the race window
            super().start(launches)

    swap = _SlowSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d: swap)

    def _make_client(_endpoint, _model):
        client = _fake_client()
        client.chat_result.return_value = "ok"
        return client

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _make_client)
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    barrier = threading.Barrier(8)

    def _hit() -> None:
        # Bounded wait so a thread that dies under heavy CI load can't deadlock the
        # barrier and hang the test; the single-flight assertion holds regardless.
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=10.0)
        p.chat([{"role": "user", "content": "hi"}])

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    assert starts["n"] == 1  # single-flight: 8 concurrent first-requests start one swap


def test_shutdown_tears_down_swap_and_closes_clients() -> None:
    client = _fake_client()
    p = _provider_with_clients({WorkerRole.CHAT: [client]})
    swap = p._swap
    p.shutdown()
    assert swap.shutdowns == 1
    client.close.assert_called_once()
    assert p._swap is None


def test_invalidate_load_cache_drops_swap() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    swap = p._swap
    p.invalidate_load_cache()
    assert swap.shutdowns == 1
    assert p._swap is None


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll *predicate* until true or *timeout*; generous so xdist load can't flake it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_warm_up_pool_starts_swap_off_thread(monkeypatch) -> None:
    # The eager warm-up at TUI mount must not block the caller; it dispatches a
    # background start thread and returns immediately.
    started = threading.Event()
    release = threading.Event()

    class _SlowSwap(_FakeSwap):
        def start(self, launches: list) -> None:
            started.set()
            release.wait(timeout=5.0)
            super().start(launches)

    swap = _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.CHAT)], swap=_SlowSwap())
    p = FleetProvider()
    p.warm_up_pool()
    assert started.wait(timeout=5.0)  # start runs on a background thread
    assert p._swap is None  # warm_up_pool returned before start completed
    release.set()
    assert _wait_until(lambda: p._swap is swap)


def test_warm_up_pool_single_flight_does_not_double_start(monkeypatch) -> None:
    starts = {"n": 0}
    in_start = threading.Event()
    release = threading.Event()

    class _GatedSwap(_FakeSwap):
        def start(self, launches: list) -> None:
            starts["n"] += 1
            in_start.set()
            release.wait(timeout=5.0)
            super().start(launches)

    swap = _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.CHAT)], swap=_GatedSwap())
    p = FleetProvider()
    p.warm_up_pool()
    assert in_start.wait(timeout=5.0)  # first start genuinely in flight
    p.warm_up_pool()  # second call while warming: must not start a second swap
    release.set()
    assert _wait_until(lambda: p._swap is swap)
    assert starts["n"] == 1


def test_warm_up_pool_noop_when_swap_already_up(monkeypatch) -> None:
    starts = {"n": 0}
    swap = _install_engine(monkeypatch, launches=[])
    monkeypatch.setattr(swap, "start", lambda launches: starts.__setitem__("n", starts["n"] + 1))
    p = FleetProvider()
    p._swap = _FakeSwap()  # already up
    p.warm_up_pool()
    assert starts["n"] == 0  # no start dispatched


def test_warm_up_blocking_logs_and_clears_guard_on_failure(monkeypatch, caplog) -> None:
    def _boom() -> list:
        raise RuntimeError("plan failed")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _boom)
    p = FleetProvider()
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.provider"):
        p._warm_up_blocking()  # runs the body synchronously for the assertion
    assert p._swap is None
    assert p._warming is False  # guard cleared so a later warm-up can retry
    assert "warm-up failed" in caplog.text.lower()


def test_preload_roles_warms_each_role_and_fires_listeners(monkeypatch) -> None:
    chat, embed = _fake_client(), _fake_client()
    embed.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    spawning: list[WorkerRole] = []
    spawned: list[WorkerRole] = []
    p.add_spawn_listener(on_spawning=spawning.append, on_spawned=spawned.append)
    p._preload_roles()
    chat.chat.assert_called_once()  # chat warmed with a minimal completion
    embed.embed.assert_called_once()  # embed warmed
    assert set(spawning) == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert set(spawned) == {WorkerRole.CHAT, WorkerRole.EMBED}


def test_preload_roles_skips_failing_role(monkeypatch) -> None:
    embed = _fake_client()
    embed.embed.side_effect = RuntimeError("not loaded")
    p = _provider_with_clients({WorkerRole.EMBED: [embed]})
    p._preload_roles()  # a per-role warm failure must not raise
    embed.embed.assert_called_once()


def test_warm_role_dispatches_per_role() -> None:
    chat, embed, rerank, vision = (_fake_client() for _ in range(4))
    prov_mod._warm_role(WorkerRole.CHAT, chat)
    prov_mod._warm_role(WorkerRole.EMBED, embed)
    prov_mod._warm_role(WorkerRole.RERANK, rerank)
    prov_mod._warm_role(WorkerRole.VISION, vision)  # vision warms lazily -> no call
    chat.chat.assert_called_once()
    embed.embed.assert_called_once_with([prov_mod._WARM_PROMPT])
    rerank.rerank.assert_called_once()
    vision.chat.assert_not_called()
    vision.embed.assert_not_called()


def test_role_ready_false_without_swap() -> None:
    assert FleetProvider().role_ready(WorkerRole.CHAT) is False


def test_role_ready_reflects_swap_running_state() -> None:
    p = FleetProvider()
    swap = _FakeSwap()
    swap.ready = {WorkerRole.CHAT}
    p._swap = swap
    assert p.role_ready(WorkerRole.CHAT) is True
    assert p.role_ready(WorkerRole.EMBED) is False


def test_drop_loaded_models_async_tears_down_off_thread() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    swap = p._swap
    p.drop_loaded_models_async()
    assert _wait_until(lambda: p._swap is None)
    assert swap.shutdowns == 1


def test_drop_loaded_models_async_noop_without_swap() -> None:
    p = FleetProvider()  # _swap is None
    p.drop_loaded_models_async()  # must not raise or spawn a thread
    assert p._swap is None


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
    # is a fleet-of-one). Construction is side-effect-free; no swap is started.
    from lilbee.providers.routing_provider import RoutingProvider

    assert isinstance(RoutingProvider()._get_local(), FleetProvider)


def test_get_capabilities_unresolved_model_returns_completion(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    def _raise(_m: str):
        raise ProviderError("not found")

    monkeypatch.setattr("lilbee.catalog.is_rerank_ref", lambda _m: False)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _raise)
    assert FleetProvider().get_capabilities("missing/model.gguf") == ["completion"]


class TestLifecycleMethods:
    def test_cancel_inference_is_noop(self) -> None:
        # llama-server stops on client disconnect; cancel has nothing to flip.
        assert _provider_with_clients({}).cancel_inference() is None

    def test_reload_role_noop_when_swap_not_up(self, monkeypatch) -> None:
        spawned = {"thread": False}
        monkeypatch.setattr("threading.Thread", lambda *a, **k: spawned.__setitem__("thread", True))
        p = FleetProvider()  # _swap is None
        p.reload_role(WorkerRole.EMBED)
        assert spawned["thread"] is False  # no background restart dispatched

    def test_reload_role_dispatches_background_restart(self) -> None:
        done = threading.Event()
        p = FleetProvider()
        p._swap = _FakeSwap()  # non-None so reload dispatches
        p._reload_blocking = lambda: done.set()  # type: ignore[method-assign]
        p.reload_role(WorkerRole.EMBED)
        assert done.wait(timeout=2.0)  # the spawned thread ran the blocking restart

    def test_reload_blocking_restarts_swap_and_readopts(self, monkeypatch) -> None:
        launches = [_fake_launch(WorkerRole.CHAT, slots=2, ctx=4096)]
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: launches)
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m: _fake_client())
        p = FleetProvider()
        swap = _FakeSwap()
        p._swap = swap
        p._reload_blocking()
        assert swap.reloads == 1
        assert p._chat_slots == 2  # capacity re-adopted from the new launch set
        assert set(p._clients) == {WorkerRole.CHAT}

    def test_reload_blocking_noop_when_swap_cleared(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [])
        p = FleetProvider()  # _swap stays None
        p._reload_blocking()  # must not raise

    def test_add_spawn_listener_stores_callbacks(self) -> None:
        p = FleetProvider()

        def on_spawning(_r: WorkerRole) -> None: ...

        def on_spawned(_r: WorkerRole) -> None: ...

        p.add_spawn_listener(on_spawning=on_spawning, on_spawned=on_spawned)
        assert p._on_spawning is on_spawning
        assert p._on_spawned is on_spawned


class TestChatWithTools:
    def test_routes_to_chat_server(self) -> None:
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


class TestChatCapacityAndCtxGetters:
    """max_concurrent_chats / served_chat_ctx read the chat launch once the swap is up."""

    def test_max_concurrent_chats_defaults_to_one_before_swap(self) -> None:
        assert FleetProvider().max_concurrent_chats() == 1

    def test_max_concurrent_chats_reads_chat_slots_when_up(self) -> None:
        p = FleetProvider()
        p._swap = _FakeSwap()
        p._chat_slots = 4
        assert p.max_concurrent_chats() == 4

    def test_served_chat_ctx_is_none_before_swap(self) -> None:
        assert FleetProvider().served_chat_ctx() is None

    def test_served_chat_ctx_reads_chat_ctx_when_up(self) -> None:
        p = FleetProvider()
        p._swap = _FakeSwap()
        p._chat_ctx = 32768
        assert p.served_chat_ctx() == 32768
