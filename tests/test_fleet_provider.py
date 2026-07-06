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
from lilbee.providers.roles import RerankMode, WorkerRole

_GB = 1024**3


def _fake_client(in_flight: int = 0) -> MagicMock:
    client = MagicMock()
    client.in_flight = in_flight
    return client


def _fake_launch(
    role: WorkerRole, *, slots: int = 1, ctx: int = 0, weights_bytes: int = 0, replica: int = 0
) -> MagicMock:
    launch = MagicMock()
    launch.role = role
    launch.slots = slots
    launch.ctx = ctx
    launch.weights_bytes = weights_bytes
    launch.replica = replica
    return launch


class _FakeSwap:
    """A stand-in SwapManager recording lifecycle calls; ready roles are settable."""

    def __init__(self) -> None:
        self.started: list[list] = []
        self.reaps = 0
        self.reloads = 0
        self.shutdowns = 0
        self.ready: set[WorkerRole] = set()
        self.running = True

    def reap_stale(self) -> None:
        self.reaps += 1

    def start(self, launches: list) -> None:
        self.started.append(launches)
        self.running = True

    def endpoint(self) -> str:
        return "http://fake-endpoint"

    def is_live(self) -> bool:
        # Default: the fake swap is considered live so existing tests that have
        # an empty client pool still raise ProviderError, not trigger a rebuild.
        return True

    def role_ready(self, role: WorkerRole) -> bool:
        return role in self.ready

    def reload(self, launches: list) -> None:
        self.reloads += 1

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.running = False


@pytest.fixture(autouse=True)
def _no_real_probe(monkeypatch):
    """No test in this module may probe real hardware or resolve real binaries.

    capture_plan_probe resolves the engine binary and spawns device probes; on a
    host without the bundled engine (CI) it raises, and on a dev box it silently
    probes the real GPUs. Tests that exercise the capture lifecycle override
    this stub with their own recorder.
    """
    monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: None)


def _install_engine(monkeypatch, *, launches: list, swap: _FakeSwap | None = None) -> _FakeSwap:
    """Patch the swap, client, and planner so _ensure_fleet builds controllable fakes."""
    swap = swap or _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: swap)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _data_dir: None)
    monkeypatch.setattr(prov_mod, "sweep_owned", lambda _data_dir: None)
    monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: None)
    monkeypatch.setattr(
        prov_mod, "LlamaServerClient", lambda _endpoint, _model, **_kw: _fake_client()
    )
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: launches)
    return swap


def _provider_with_clients(clients: dict[WorkerRole, list[MagicMock]]) -> FleetProvider:
    """A provider with a fake swap already up and a client pool per role (no real start)."""
    p = FleetProvider()
    # Non-empty so _ensure_fleet short-circuits; roles without clients still error.
    p._swaps = {role: _FakeSwap() for role in clients} or {WorkerRole.CHAT: _FakeSwap()}
    p._clients = {role: list(cs) for role, cs in clients.items() if cs}
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


def test_embed_routes_to_least_busy_replica() -> None:
    # Data-parallel replicas: a request goes to the idlest replica in the pool.
    busy, idle = _fake_client(5), _fake_client(1)
    idle.embed.return_value = [[0.2]]
    p = _provider_with_clients({WorkerRole.EMBED: [busy, idle]})
    assert p.embed(["a"]) == [[0.2]]
    idle.embed.assert_called_once()
    busy.embed.assert_not_called()


def test_adopt_role_builds_a_client_per_replica(monkeypatch) -> None:
    launches = [_fake_launch(WorkerRole.EMBED), _fake_launch(WorkerRole.EMBED)]
    _install_engine(monkeypatch, launches=launches)
    p = FleetProvider()
    p._ensure_fleet()
    assert len(p._clients[WorkerRole.EMBED]) == 2  # one client per replica launch


def test_ensure_fleet_refused_after_shutdown(monkeypatch) -> None:
    """bb-dpp source guard: once shut down (and likely discarded by reset_services),
    a lingering warm-up/reload thread's _ensure_fleet must not spawn a new llama-swap
    on the dead provider -- that is exactly the duplicate that leaks on teardown."""
    swap = _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    p._shutdown_swap()  # latches _shut_down (and reaps via a fresh SwapManager)
    assert p._ensure_fleet() is False
    assert swap.started == []  # no swap started after shutdown


def test_adopt_role_retires_old_clients_without_closing(monkeypatch) -> None:
    # Re-adopting (a reload) must not close old clients in place (a
    # reader may still hold one); they are retired for deferred close.
    launch = _fake_launch(WorkerRole.EMBED)
    swap = _install_engine(monkeypatch, launches=[launch])
    p = FleetProvider()
    old = [_fake_client(), _fake_client()]
    p._clients = {WorkerRole.EMBED: old}

    with p._lock:
        p._adopt_role(WorkerRole.EMBED, swap, [launch])

    assert p._retiring_clients == old  # retired, not closed yet
    for client in old:
        client.close.assert_not_called()
    assert p._clients[WorkerRole.EMBED][0] not in old  # fresh client adopted


def test_retire_closes_prior_idle_generation_at_next_reload() -> None:
    # The prior reload's clients are closed at the
    # next reload, by when their readers (in_flight==0) have finished.
    p = FleetProvider()
    prior = [_fake_client(in_flight=0), _fake_client(in_flight=0)]
    p._retiring_clients = list(prior)
    current_old = [_fake_client(in_flight=0)]

    with p._lock:
        p._retire_clients(current_old)

    for client in prior:
        client.close.assert_called_once_with()  # prior idle generation closed
    assert p._retiring_clients == current_old  # this reload's clients now pending


def test_retire_keeps_busy_prior_client_for_a_later_reload() -> None:
    p = FleetProvider()
    busy = _fake_client(in_flight=1)  # a reader is still mid-request on it
    p._retiring_clients = [busy]

    with p._lock:
        p._retire_clients([])

    busy.close.assert_not_called()  # not closed while in flight
    assert busy in p._retiring_clients  # retained for the next reload


def test_drop_swap_refs_closes_retiring_clients() -> None:
    p = FleetProvider()
    retiring = _fake_client(in_flight=1)  # even a busy one is closed at shutdown
    live = _fake_client()
    p._clients = {WorkerRole.EMBED: [live]}
    p._retiring_clients = [retiring]

    p._drop_swap_refs()

    live.close.assert_called_once_with()
    retiring.close.assert_called_once_with()
    assert p._retiring_clients == []


def test_adopt_role_threads_rerank_mode(monkeypatch) -> None:
    launch = _fake_launch(WorkerRole.RERANK)
    launch.rerank_mode = RerankMode.LLM
    captured: dict[str, object] = {}

    def _capture(_endpoint, _model, **kw):
        captured["rerank_mode"] = kw.get("rerank_mode")
        return _fake_client()

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: _FakeSwap())
    monkeypatch.setattr(prov_mod, "LlamaServerClient", _capture)
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [launch])
    FleetProvider()._ensure_fleet()
    assert captured["rerank_mode"] is RerankMode.LLM


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


def test_fit_chat_context_passthrough_when_ctx_unknown() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    msgs = [{"role": "user", "content": "hi"}]
    # _chat_ctx is None until a chat launch is adopted: no windowing, same list.
    assert p._fit_chat_context(msgs, None, None, "m") is msgs


def test_fit_chat_context_windows_overlong_history() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 2048
    msgs: list[dict] = [{"role": "system", "content": "s"}]
    for _ in range(30):
        msgs.append({"role": "user", "content": "x" * 500})
        msgs.append({"role": "assistant", "content": "y" * 500})
    msgs.append({"role": "user", "content": "final"})
    out = p._fit_chat_context(msgs, None, None, "m")
    assert len(out) < len(msgs)
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "final"


def test_fit_chat_context_raises_context_overflow_when_unfixable() -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 64
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 9000}]
    with pytest.raises(ProviderError) as excinfo:
        p._fit_chat_context(msgs, None, None, "qwen")
    assert excinfo.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW


def test_fit_chat_context_reserves_requested_max_tokens() -> None:
    from lilbee.providers.base import ProviderError

    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    p._chat_ctx = 2000
    # a large num_predict shrinks the prompt budget below what a modest history needs
    msgs = [{"role": "user", "content": "x" * 3000}]
    with pytest.raises(ProviderError):
        p._fit_chat_context(msgs, None, {"num_predict": 1900}, "m")


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


def test_chat_model_override_error_is_bad_request_kind(monkeypatch) -> None:
    """The mismatch is a client error; BAD_REQUEST maps it to a 400 envelope, not a 500."""
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr(cfg, "chat_model", "org/repo/configured.gguf")
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    with pytest.raises(ProviderError) as excinfo:
        p.chat([{"role": "user", "content": "hi"}], model="org/repo/other.gguf")
    assert excinfo.value.kind is ProviderErrorKind.BAD_REQUEST


def test_vision_call_returns_text() -> None:
    client = _fake_client()
    client.chat.return_value = "OCR text"
    assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None) == "OCR text"


def test_vision_call_enforces_timeout() -> None:
    client = _fake_client()
    client.chat_bounded.return_value = "OCR text"
    assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 5.0) == "OCR text"
    client.chat_bounded.assert_called_once()
    client.chat.assert_not_called()  # the timed path streams via chat_bounded, not chat


def test_vision_call_caps_output_tokens(monkeypatch) -> None:
    # A runaway OCR page can loop to tens of thousands of chars and dominate a
    # scan's time; the call must cap generation at cfg.vision_ocr_max_tokens.
    monkeypatch.setattr(cfg, "vision_ocr_max_tokens", 4096)
    client = _fake_client()
    client.chat.return_value = "OCR text"
    prov_mod._vision_call(client, [{"role": "user", "content": "x"}], None)
    assert client.chat.call_args.kwargs["options"] == {"max_tokens": 4096}


def test_vision_ocr_retries_busy_then_succeeds(monkeypatch) -> None:
    # A 429 mid-ingest is transient (cold replicas, slot contention); the vision
    # path must back off and retry so the page is ingested, not dropped.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    busy = ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)
    client = _fake_client()
    client.chat.side_effect = [busy, busy, "ocr text"]
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    assert p.vision_ocr(b"png", "org/repo/v.gguf") == "ocr text"
    assert client.chat.call_count == 3


def test_vision_ocr_gives_up_when_persistently_busy(monkeypatch) -> None:
    # The retry budget is bounded; a fleet that never frees a slot fails the page.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    client = _fake_client()
    client.chat.side_effect = ProviderError(
        "busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT
    )
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    with pytest.raises(ProviderError) as excinfo:
        p.vision_ocr(b"png", "org/repo/v.gguf")
    assert excinfo.value.kind is ProviderErrorKind.RATE_LIMIT
    assert client.chat.call_count == prov_mod._VISION_BUSY_RETRIES


def test_vision_ocr_does_not_retry_non_busy_errors(monkeypatch) -> None:
    # A genuine extraction failure must surface immediately, not burn the budget.
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr(cfg, "vision_model", "org/repo/v.gguf")
    client = _fake_client()
    client.chat.side_effect = ProviderError("boom", provider="llama-server")
    p = _provider_with_clients({WorkerRole.VISION: [client]})
    with pytest.raises(ProviderError, match="boom"):
        p.vision_ocr(b"png", "org/repo/v.gguf")
    assert client.chat.call_count == 1


def test_vision_gate_capacity_sums_real_launch_slots(monkeypatch) -> None:
    # The gate caps at the servers' fitted --parallel slots: planning can fit
    # fewer slots than vision_ocr_concurrency asks for, and a cap above the real
    # slot count oversubscribes the servers into a 429 storm.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 16)
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client(), _fake_client()]})
    p._launches[WorkerRole.VISION] = (
        _fake_launch(WorkerRole.VISION, slots=3),
        _fake_launch(WorkerRole.VISION, slots=2, replica=1),
    )
    assert p._vision_gate_capacity() == 5


def test_vision_gate_capacity_falls_back_to_configured_formula(monkeypatch) -> None:
    # Without a launch snapshot (mid-reload) the configured ceiling still bounds it.
    monkeypatch.setattr(prov_mod, "gpu_device_count", lambda: 1)
    monkeypatch.setattr(cfg, "vision_replicas", 3)
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    p = _provider_with_clients({WorkerRole.VISION: [_fake_client()]})
    assert p._vision_gate_capacity() == 12


def test_vision_request_gate_tracks_fleet_capacity(monkeypatch) -> None:
    # The gate must cap in-flight vision requests at the caller-supplied capacity
    # and rebuild to a new capacity while idle.
    monkeypatch.setattr(prov_mod._VISION_GATE, "_semaphore", None)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_capacity", 0)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_in_flight", 0)
    with prov_mod._VISION_GATE.slot(12):
        first = prov_mod._VISION_GATE._semaphore
    assert prov_mod._VISION_GATE._capacity == 12
    with prov_mod._VISION_GATE.slot(4):
        assert prov_mod._VISION_GATE._semaphore is not first  # rebuilt while idle
    assert prov_mod._VISION_GATE._capacity == 4


def test_vision_gate_resize_deferred_while_in_flight(monkeypatch) -> None:
    # Resizing the gate while a request is in flight would build a
    # fresh full-capacity semaphore beside the old holders and momentarily double
    # the real cap. The resize must wait until the gate drains to idle.
    import threading

    monkeypatch.setattr(prov_mod._VISION_GATE, "_semaphore", None)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_capacity", 0)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_in_flight", 0)

    entered = threading.Event()
    release = threading.Event()
    held: list[object] = []

    def _hold() -> None:
        with prov_mod._VISION_GATE.slot(2):
            held.append(prov_mod._VISION_GATE._semaphore)
            entered.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=_hold)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        # Capacity change arrives mid-flight; a checkout must reuse the live
        # semaphore rather than swap, so the cap is never doubled.
        reused = prov_mod._VISION_GATE._checkout(8)
        try:
            assert reused is held[0]
            assert prov_mod._VISION_GATE._capacity == 2
        finally:
            with prov_mod._VISION_GATE._lock:  # balance the raw _checkout increment
                prov_mod._VISION_GATE._in_flight -= 1
    finally:
        release.set()
        worker.join()

    # Once idle again, the next checkout applies the deferred capacity.
    with prov_mod._VISION_GATE.slot(8):
        assert prov_mod._VISION_GATE._capacity == 8


def test_vision_gate_slot_decrements_in_flight_when_acquire_raises(monkeypatch) -> None:
    # If acquire() raises (e.g. an interrupted blocking acquire), the in-flight
    # counter must still be decremented, or a leaked count would pin the gate
    # non-idle and defer every later capacity resize forever.
    monkeypatch.setattr(prov_mod._VISION_GATE, "_semaphore", None)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_capacity", 0)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_in_flight", 0)

    real_checkout = prov_mod._VISION_GATE._checkout

    class _BoomSemaphore:
        def acquire(self) -> None:
            raise KeyboardInterrupt

    def _checkout_returning_boom(capacity):
        real_checkout(capacity)  # increments _in_flight like the real path
        return _BoomSemaphore()

    monkeypatch.setattr(prov_mod._VISION_GATE, "_checkout", _checkout_returning_boom)

    with pytest.raises(KeyboardInterrupt), prov_mod._VISION_GATE.slot(2):
        pass  # pragma: no cover - acquire raises before the body

    assert prov_mod._VISION_GATE._in_flight == 0  # counter not leaked


def test_vision_gate_bounds_concurrency_to_capacity(monkeypatch) -> None:
    # The ingest file fan-out can launch far more concurrent OCR requests than the
    # vision server has slots; the gate (held by every vision entry point) caps
    # concurrent holders at the fleet's real slot capacity so a single-replica
    # server isn't 429-stormed. All callers still run.
    import threading
    import time

    monkeypatch.setattr(prov_mod._VISION_GATE, "_semaphore", None)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_capacity", 0)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_in_flight", 0)

    lock = threading.Lock()
    live = 0
    peak = 0
    ran = 0

    def _hold() -> None:
        nonlocal live, peak, ran
        with prov_mod._VISION_GATE.slot(2):
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lock:
                live -= 1
                ran += 1

    threads = [threading.Thread(target=_hold) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 2  # the cap is both reached and never exceeded
    assert ran == 8  # every caller ran, just serialized through the gate


def test_ocr_pdf_page_skips_when_budget_exhausted(monkeypatch) -> None:
    # A page that waited out the document deadline while queued is skipped (empty
    # text), not run un-timed -- a 0 timeout would disable the per-page cap.
    monkeypatch.setattr(prov_mod._VISION_GATE, "_semaphore", None)
    monkeypatch.setattr("lilbee.vision.build_vision_messages", lambda *_a, **_k: [])
    called: list[object] = []
    monkeypatch.setattr(prov_mod, "_vision_call", lambda *a, **_k: called.append(a) or "ocr")
    idx, text = prov_mod._ocr_pdf_page(
        3,
        b"\x89PNG",
        clients=[_fake_client()],
        ocr_prompt="describe",
        deadline=time.monotonic() - 1.0,  # already past
        page_path=Path("doc.pdf"),
        gate_capacity=2,
    )
    assert (idx, text) == (3, "")
    assert called == []  # _vision_call never ran


def test_ocr_pdf_page_retries_busy_then_keeps_text(monkeypatch) -> None:
    # A 429 on a PDF page is retried like the image path, so the page text lands
    # instead of the page being skipped to empty.
    from lilbee.providers.base import ProviderError, ProviderErrorKind

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr(prov_mod._VISION_GATE, "_semaphore", None)
    monkeypatch.setattr("lilbee.vision.build_vision_messages", lambda *_a, **_k: [])
    busy = ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)
    client = _fake_client()
    client.chat.side_effect = [busy, busy, "page text"]
    idx, text = prov_mod._ocr_pdf_page(
        0,
        b"\x89PNG",
        clients=[client],
        ocr_prompt="describe",
        deadline=None,
        page_path=Path("doc.pdf"),
        gate_capacity=2,
    )
    assert (idx, text) == (0, "page text")
    assert client.chat.call_count == 3


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
    # Auto replicas (vision_replicas left at its 0 default) resolves to one per
    # GPU; pin the probe to a single GPU so the gate admits 1 x 4 = 4 in flight.
    monkeypatch.setattr(prov_mod, "gpu_device_count", lambda: 1)
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
    # Pin the device-count probe so _checkout() does not run a real subprocess
    # and spend ~4s that would bleed into the budget timing assertion.
    monkeypatch.setattr(prov_mod, "gpu_device_count", lambda: 1)
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
    # Budget is 2*120 + 300 = 540; pages run concurrently, so each draws nearly
    # the full remaining budget (far above any 120 cap), in either capture order.
    assert seen[0] == pytest.approx(540.0, abs=1.0)
    assert seen[1] == pytest.approx(540.0, abs=1.0)
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


def test_ensure_fleet_starts_once_and_builds_clients(monkeypatch) -> None:
    launches = [_fake_launch(WorkerRole.CHAT, slots=4, ctx=32768), _fake_launch(WorkerRole.EMBED)]
    swap = _install_engine(monkeypatch, launches=launches)
    p = FleetProvider()
    assert p._ensure_fleet() is True
    assert len(swap.started) == 2  # one start per placed role group
    assert set(p._clients) == {WorkerRole.CHAT, WorkerRole.EMBED}  # one client per placed role
    assert p._chat_slots == 4  # chat capacity / ctx taken from the chat launch
    assert p._chat_ctx == 32768
    p._ensure_fleet()  # second call reuses the running groups
    assert len(swap.started) == 2


def test_ensure_fleet_defaults_chat_slots_without_chat_launch(monkeypatch) -> None:
    _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.EMBED)])
    p = FleetProvider()
    p._ensure_fleet()
    assert p._chat_slots == 1  # no chat launch -> default capacity
    assert p._chat_ctx is None


class _OrderedReapSwap(_FakeSwap):
    """A fake swap appending reap events to a shared order log."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def reap_stale(self) -> None:
        super().reap_stale()
        self._order.append("reap")


def _ordered_planner(order: list[str], launches: list) -> object:
    """A plan_all_launches stand-in appending to the shared order log."""

    def _plan() -> list:
        order.append("plan")
        return launches

    return _plan


def test_ensure_fleet_reaps_stale_swaps_before_planning(monkeypatch) -> None:
    # An OOM-survivor llama-swap holds VRAM; reaping after planning would let
    # the device probe see artificially reduced free memory and misplace.
    order: list[str] = []
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: order.append("reap"))
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", _ordered_planner(order, [_fake_launch(WorkerRole.CHAT)])
    )
    FleetProvider()._ensure_fleet()
    assert order == ["reap", "plan"]


def test_reload_pass_reaps_stale_swaps_before_planning(monkeypatch) -> None:
    order: list[str] = []
    swap = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: swap)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: order.append("reap"))
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod, "plan_all_launches", _ordered_planner(order, [_fake_launch(WorkerRole.CHAT)])
    )
    p = FleetProvider()
    p._swaps = {WorkerRole.CHAT: swap}
    p._reload_pass()
    assert order == ["reap", "plan"]


def test_ensure_fleet_spawns_nothing_when_no_models(monkeypatch) -> None:
    # No configured/installed model -> no launches -> no swap process at all
    # (matches the old supervisor, which spawned nothing for an empty launch set).
    started = {"swaps": 0}

    class _CountingSwap(_FakeSwap):
        def start(self, launches: list) -> None:
            started["swaps"] += 1
            super().start(launches)

    _install_engine(monkeypatch, launches=[], swap=_CountingSwap())
    p = FleetProvider()
    assert p._ensure_fleet() is False
    assert started["swaps"] == 0  # never started
    assert p._swaps == {}
    assert p._clients == {}


def test_ensure_fleet_returns_none_when_engine_binary_unavailable(monkeypatch) -> None:
    """plan_all_launches raising ProviderError (no engine binary) yields no swap."""
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _data_dir, _group: _FakeSwap())

    def _no_binary() -> list:
        raise ProviderError("Engine binary unavailable")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _no_binary)
    p = FleetProvider()
    assert p._ensure_fleet() is False
    assert p._swaps == {}


def _captured_client_kwargs(monkeypatch, launch) -> dict:
    """Build the engine around *launch* and return the client constructor kwargs."""
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [launch])
    captured: list[dict] = []

    def _capture(_endpoint, _model, **kwargs):
        captured.append(kwargs)
        return _fake_client()

    monkeypatch.setattr(prov_mod, "LlamaServerClient", _capture)
    FleetProvider()._ensure_fleet()
    return captured[0]


def test_clients_get_token_cap_and_cold_load_timeout(monkeypatch) -> None:
    # Each role's client carries the launch token_cap (embed/rerank input truncation,
    # the in-process backstop) and a timeout long enough for a cold upstream load,
    # matching the old supervisor's client construction.
    launch = _fake_launch(WorkerRole.EMBED)
    launch.token_cap = 2048
    kwargs = _captured_client_kwargs(monkeypatch, launch)
    assert kwargs["token_cap"] == 2048
    assert kwargs["timeout"] == prov_mod._REQUEST_TIMEOUT_FLOOR_S


def test_small_model_client_keeps_the_floor_timeout(monkeypatch) -> None:
    launch = _fake_launch(WorkerRole.CHAT, weights_bytes=4 * _GB)
    kwargs = _captured_client_kwargs(monkeypatch, launch)
    assert kwargs["timeout"] == prov_mod._REQUEST_TIMEOUT_FLOOR_S


def test_giant_model_client_timeout_covers_its_cold_load(monkeypatch) -> None:
    # A split-GGUF giant loads longer than the fixed floor; the client request
    # timeout must ride out the cold load llama-swap itself is willing to wait
    # for, or the first chat marks the replica unhealthy mid-load.
    from lilbee.providers.fleet import swap_config as swap_config_mod

    launch = _fake_launch(WorkerRole.CHAT, weights_bytes=300 * _GB)
    kwargs = _captured_client_kwargs(monkeypatch, launch)
    health_timeout = swap_config_mod._health_check_timeout_s([launch])
    assert health_timeout > prov_mod._REQUEST_TIMEOUT_FLOOR_S
    assert kwargs["timeout"] >= health_timeout
    assert kwargs["timeout"] == health_timeout + prov_mod._REQUEST_TIMEOUT_GENERATION_MARGIN_S


def test_chat_starts_swap_on_first_use(monkeypatch) -> None:
    from lilbee.providers.base import ChatResult, FinishReason

    captured: dict[str, MagicMock] = {}

    def _make_client(_endpoint, model, **_kw):
        client = _fake_client()
        client.chat_result.return_value = ChatResult(
            text="ok", tool_calls=(), finish_reason=FinishReason.STOP
        )
        captured[model] = client
        return client

    swap = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: swap)
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
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: swap)

    def _make_client(_endpoint, _model, **_kw):
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
    swap = next(iter(p._swaps.values()))
    p.shutdown()
    assert swap.shutdowns == 1
    client.close.assert_called_once()
    assert p._swaps == {}


def test_invalidate_load_cache_drops_swap() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    swap = next(iter(p._swaps.values()))
    p.invalidate_load_cache()
    assert swap.shutdowns == 1
    assert p._swaps == {}


def test_invalidate_load_cache_leaves_provider_reusable(monkeypatch) -> None:
    """A cache drop is not terminal: the next use rebuilds the swap."""
    swap = _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.invalidate_load_cache()
    assert p._swaps == {}
    assert p._ensure_fleet() is True  # rebuilt with current cfg, not refused
    assert p._swaps.get(WorkerRole.CHAT) is swap


def test_drop_loaded_models_async_leaves_provider_reusable(monkeypatch) -> None:
    """The off-thread drop used by settings changes must not latch shutdown.

    app.settings routes num_ctx/kv_cache_type changes here while retaining the
    provider; a latched flag would refuse every later chat/embed/rerank call
    until process restart.
    """
    swap = _install_engine(monkeypatch, launches=[_fake_launch(WorkerRole.CHAT)])
    p = FleetProvider()
    assert p._ensure_fleet() is True
    p.drop_loaded_models_async()
    assert _wait_until(lambda: p._swaps == {})
    assert p._ensure_fleet() is True  # rebuilt with current cfg, not refused
    assert p._swaps.get(WorkerRole.CHAT) is swap


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
    assert p._swaps == {}  # warm_up_pool returned before start completed
    release.set()
    assert _wait_until(lambda: p._swaps.get(WorkerRole.CHAT) is swap)


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
    assert _wait_until(lambda: p._swaps.get(WorkerRole.CHAT) is swap)
    assert starts["n"] == 1


def test_warm_up_pool_noop_when_swap_already_up(monkeypatch) -> None:
    starts = {"n": 0}
    swap = _install_engine(monkeypatch, launches=[])
    monkeypatch.setattr(swap, "start", lambda launches: starts.__setitem__("n", starts["n"] + 1))
    p = FleetProvider()
    p._swaps = {WorkerRole.CHAT: _FakeSwap()}  # already up
    p.warm_up_pool()
    assert starts["n"] == 0  # no start dispatched


def test_warm_up_blocking_logs_and_clears_guard_on_failure(monkeypatch, caplog) -> None:
    def _boom() -> list:
        raise RuntimeError("plan failed")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _boom)
    p = FleetProvider()
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.provider"):
        p._warm_up_blocking()  # runs the body synchronously for the assertion
    assert p._swaps == {}
    assert p._warming is False  # guard cleared so a later warm-up can retry
    assert "warm-up failed" in caplog.text.lower()
    # The handled failure must not carry a traceback: a WARNING with exc_info
    # reads like a crash for a condition the next real call recovers from.
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings and all(r.exc_info is None for r in warnings)


def test_warm_up_blocking_swallows_interpreter_shutdown_race(monkeypatch, caplog) -> None:
    # A fast CLI exit tears down the interpreter mid-warm; the pool submit then
    # raises RuntimeError. During finalization this must be dropped quietly, not
    # logged as a scary WARNING traceback.
    def _shutdown_race() -> list:
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    monkeypatch.setattr(planning_mod, "plan_all_launches", _shutdown_race)
    monkeypatch.setattr("lilbee.providers.fleet.provider.sys.is_finalizing", lambda: True)
    p = FleetProvider()
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.provider"):
        p._warm_up_blocking()
    assert p._warming is False
    assert "warm-up failed" not in caplog.text.lower()  # no WARNING on shutdown


def test_preload_roles_warms_each_role_and_fires_listeners(monkeypatch) -> None:
    chat, embed = _fake_client(), _fake_client()
    embed.embed.return_value = [[0.1]]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    # Keep the unit hermetic: the chat warm path's shard prewarm reads cfg.chat_model
    # off disk, which this test isn't exercising.
    monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
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


def test_preload_roles_warms_roles_concurrently(monkeypatch) -> None:
    # The light roles must not queue behind the chat load: with the chat warm
    # blocked mid-flight, the embed warm still completes.
    chat_started = threading.Event()
    release_chat = threading.Event()
    embed_warmed = threading.Event()

    chat, embed = _fake_client(), _fake_client()

    def _blocked_chat(*args, **kwargs) -> MagicMock:
        chat_started.set()
        release_chat.wait(timeout=5.0)
        return MagicMock()

    chat.chat.side_effect = _blocked_chat
    embed.embed.side_effect = lambda *a, **k: (embed_warmed.set(), [[0.1]])[1]
    p = _provider_with_clients({WorkerRole.CHAT: [chat], WorkerRole.EMBED: [embed]})
    monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
    preload = threading.Thread(target=p._preload_roles, daemon=True)
    preload.start()
    assert chat_started.wait(timeout=5.0)
    assert embed_warmed.wait(timeout=5.0)  # embed warmed while chat still loading
    release_chat.set()
    preload.join(timeout=5.0)
    assert not preload.is_alive()


def _registry_with_shards(monkeypatch, shards: list[Path]) -> None:
    registry = MagicMock()
    registry.shard_paths.return_value = shards
    monkeypatch.setattr(prov_mod, "ModelRegistry", MagicMock(return_value=registry))


def test_prewarm_skips_shards_already_paged_this_boot(monkeypatch, tmp_path) -> None:
    shard = tmp_path / "model.gguf"
    shard.write_bytes(b"x" * 64)
    _registry_with_shards(monkeypatch, [shard])
    monkeypatch.setattr(prov_mod, "_PREWARMED_SHARDS", set())
    p = _provider_with_clients({})
    p._prewarm_chat_weights()  # first pass reads and records the shard

    opens = {"n": 0}
    real_open = Path.open

    def _counting_open(self, *args, **kwargs):
        if self == shard:
            opens["n"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)
    p._prewarm_chat_weights()  # second pass: cache is hot, no re-read
    assert opens["n"] == 0


def test_prewarm_rereads_when_a_shard_changed(monkeypatch, tmp_path) -> None:
    shard = tmp_path / "model.gguf"
    shard.write_bytes(b"x" * 64)
    _registry_with_shards(monkeypatch, [shard])
    monkeypatch.setattr(prov_mod, "_PREWARMED_SHARDS", set())
    p = _provider_with_clients({})
    p._prewarm_chat_weights()
    shard.write_bytes(b"y" * 128)  # new size + mtime -> new prewarm identity

    opens = {"n": 0}
    real_open = Path.open

    def _counting_open(self, *args, **kwargs):
        if self == shard:
            opens["n"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)
    p._prewarm_chat_weights()
    assert opens["n"] == 1  # changed shard is read again


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
    p._swaps = {WorkerRole.CHAT: swap}
    assert p.role_ready(WorkerRole.CHAT) is True
    assert p.role_ready(WorkerRole.EMBED) is False


def test_drop_loaded_models_async_tears_down_off_thread() -> None:
    p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    swap = next(iter(p._swaps.values()))
    p.drop_loaded_models_async()
    # Wait on the actual shutdown rather than ``_swap is None``: the worker clears
    # the ref before it calls swap.shutdown(), so the latter is the later signal.
    assert _wait_until(lambda: swap.shutdowns == 1)
    assert p._swaps == {}


def test_drop_loaded_models_async_noop_without_swap() -> None:
    p = FleetProvider()  # _swap is None
    p.drop_loaded_models_async()  # must not raise or spawn a thread
    assert p._swaps == {}


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
    # apply_fleet_gpu_env writes these vars in place; monkeypatch.delenv does not
    # track app-side additions, so restore the pre-apply snapshot to avoid leaking
    # the pin (CUDA_VISIBLE_DEVICES=0) into later tests that read os.environ.
    snapshot = {name: os.environ.get(name) for name in _GPU_VISIBLE_ENV_VARS}
    try:
        gpu_env.apply_fleet_gpu_env()
        for name in _GPU_VISIBLE_ENV_VARS:
            assert os.environ[name] == "0"
    finally:
        for name, value in snapshot.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_apply_fleet_gpu_env_clears_empty_cuda_visible_devices(monkeypatch) -> None:
    # SkyPilot / Docker GPU setups expose the GPU via NVIDIA_VISIBLE_DEVICES but export an
    # empty CUDA_VISIBLE_DEVICES, which hides the GPU from CUDA ("no CUDA-capable device").
    # With a GPU present, the empty var must be cleared so the engine can enumerate it.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_apply_fleet_gpu_env_clears_empty_non_cuda_visible_devices(monkeypatch) -> None:
    # The clear covers every backend visible-devices var, not just CUDA: an empty
    # GGML_VK_VISIBLE_DEVICES would otherwise hide adapters from the Vulkan VRAM fallback.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ


def test_apply_fleet_gpu_env_keeps_nonempty_cuda_visible_devices(monkeypatch) -> None:
    # An explicit index pin (and the conventional "-1" CPU opt-out) is user intent, not an
    # orchestration artifact: leave it alone.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_apply_fleet_gpu_env_pin_replaces_empty_cuda_visible_devices(monkeypatch) -> None:
    # Clearing the empty var before the pin runs is what lets a cfg.gpu_devices pin take
    # effect: setdefault would otherwise treat the present-but-empty value as already set.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", "0")
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_apply_fleet_gpu_env_keeps_empty_cuda_visible_devices_without_gpu(monkeypatch) -> None:
    # With no GPU present, an empty CUDA_VISIBLE_DEVICES is honored as genuine CPU intent.
    from lilbee.providers.fleet import gpu_env
    from lilbee.providers.fleet.gpu_env import _GPU_VISIBLE_ENV_VARS

    for name in _GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_env.apply_fleet_gpu_env()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


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
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}  # non-None so reload dispatches
        p._reload_blocking = lambda: done.set()  # type: ignore[method-assign]
        p.reload_role(WorkerRole.EMBED)
        assert done.wait(timeout=2.0)  # the spawned thread ran the blocking restart

    def test_reload_blocking_restarts_swap_and_readopts(self, monkeypatch) -> None:
        launches = [_fake_launch(WorkerRole.CHAT, slots=2, ctx=4096)]
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: launches)
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        fresh = _FakeSwap()
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: fresh)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        p = FleetProvider()
        stale = _FakeSwap()
        p._swaps = {WorkerRole.CHAT: stale}
        p._reload_blocking()
        # The chat launches changed (old running set unknown -> differs), so the
        # old group was stopped and a fresh one started and adopted.
        assert stale.shutdowns == 1
        assert len(fresh.started) == 1
        assert p._swaps[WorkerRole.CHAT] is fresh
        assert p._chat_slots == 2  # capacity re-adopted from the new launch set
        assert set(p._clients) == {WorkerRole.CHAT}

    def test_reload_blocking_noop_when_swap_cleared(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [])
        p = FleetProvider()  # _swap stays None
        p._reload_blocking()  # must not raise

    def test_reload_role_wait_runs_synchronously(self, monkeypatch) -> None:
        spawned = {"thread": False}
        monkeypatch.setattr("threading.Thread", lambda *a, **k: spawned.__setitem__("thread", True))
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        ran = {"blocking": False}
        p._reload_blocking = lambda: ran.__setitem__("blocking", True)  # type: ignore[method-assign]
        p.reload_role(WorkerRole.CHAT, wait=True)
        assert spawned["thread"] is False  # no background thread; ran in the caller's
        assert ran["blocking"] is True  # reload ran synchronously before returning

    def test_reload_role_wait_propagates_failure(self) -> None:
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}

        def boom() -> None:
            raise RuntimeError("reload failed")

        p._reload_blocking = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="reload failed"):
            p.reload_role(WorkerRole.CHAT, wait=True)

    def test_reload_role_wait_blocks_until_in_flight_done(self) -> None:
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        with p._lock:
            p._reloading = True  # simulate a reload already in flight
        returned = threading.Event()

        def waiter() -> None:
            p.reload_role(WorkerRole.CHAT, wait=True)  # sets pending, waits on the cond
            returned.set()

        t = threading.Thread(target=waiter)
        t.start()
        assert not returned.wait(timeout=0.3)  # blocked while the in-flight reload runs
        assert p._reload_pending is True  # the waiter queued its pass
        with p._lock:  # the in-flight reload finishes
            p._reloading = False
            p._reload_done.notify_all()
        assert returned.wait(timeout=2.0)  # the waiter woke and returned
        t.join(timeout=2.0)

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
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._chat_slots = 4
        assert p.max_concurrent_chats() == 4

    def test_served_chat_ctx_is_none_before_swap(self) -> None:
        assert FleetProvider().served_chat_ctx() is None

    def test_served_chat_ctx_reads_chat_ctx_when_up(self) -> None:
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._chat_ctx = 32768
        assert p.served_chat_ctx() == 32768


class _FakeReplica:
    """A minimal client double with real health/in-flight state for routing tests."""

    def __init__(self, *, in_flight: int = 0, fail: Exception | None = None) -> None:
        self.in_flight = in_flight
        self.healthy = True
        self.fail = fail
        self.calls = 0

    def mark_unhealthy(self) -> None:
        self.healthy = False

    def mark_healthy(self) -> None:
        self.healthy = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return [[0.1]] * len(texts)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return [0.5] * len(candidates)

    def chat(
        self, messages: object, options: object = None, stream: bool = False, **_kw: object
    ) -> str:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return "ocr text"


class TestReplicaHealthRouting:
    def test_least_in_flight_skips_unhealthy_clients(self) -> None:
        dead, busy_but_alive = _FakeReplica(in_flight=0), _FakeReplica(in_flight=9)
        dead.mark_unhealthy()
        assert prov_mod._least_in_flight([dead, busy_but_alive]) is busy_but_alive

    def test_least_in_flight_falls_back_when_all_unhealthy(self) -> None:
        only = _FakeReplica()
        only.mark_unhealthy()
        assert prov_mod._least_in_flight([only]) is only

    def test_embed_fails_over_once_to_a_healthy_replica(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)  # busier, picked only after failover
        p = _provider_with_clients({WorkerRole.EMBED: [dead, alive]})
        assert p.embed(["a", "b"]) == [[0.1], [0.1]]
        assert dead.healthy is False  # marked out of the pool
        assert alive.calls == 1

    def test_dead_replica_stops_receiving_traffic_after_failover(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)
        p = _provider_with_clients({WorkerRole.EMBED: [dead, alive]})
        p.embed(["a"])
        dead_calls_after_failover = dead.calls
        p.embed(["b"])  # routed straight to the healthy replica now
        assert dead.calls == dead_calls_after_failover

    def test_all_dead_surfaces_provider_error(self) -> None:
        import httpx as _httpx

        from lilbee.providers.base import ProviderError

        only = _FakeReplica(fail=_httpx.ConnectError("refused"))
        p = _provider_with_clients({WorkerRole.EMBED: [only]})
        with pytest.raises(ProviderError, match="no healthy replica"):
            p.embed(["a"])

    def test_vision_ocr_fails_over_to_healthy_replica(self, monkeypatch) -> None:
        """Vision OCR uses the failover path like embed/rerank: a dead replica is
        marked unhealthy and the call retries on a live one, instead of the dead
        replica being re-picked and the error swallowed to empty text (bb-7jg1.5)."""
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)  # busier, picked only after failover
        monkeypatch.setattr(cfg, "vision_model", "org/vis/model.gguf")
        p = _provider_with_clients({WorkerRole.VISION: [dead, alive]})
        assert p.vision_ocr(b"\x89PNG", "") == "ocr text"
        assert dead.healthy is False
        assert alive.calls == 1

    def test_vision_pdf_page_fails_over_to_healthy_replica(self, monkeypatch) -> None:
        """The per-page PDF OCR path fails a dead replica over too, so a page is
        OCR'd by a live replica rather than skipped to empty text (bb-7jg1.5)."""
        import httpx as _httpx

        from lilbee.providers.fleet import provider as prov

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)
        idx, text = prov._ocr_pdf_page(
            0,
            b"\x89PNG",
            clients=[dead, alive],
            ocr_prompt="read it",
            deadline=None,
            page_path=Path("doc.pdf"),
            gate_capacity=2,
        )
        assert (idx, text) == (0, "ocr text")
        assert dead.healthy is False
        assert alive.calls == 1

    def test_successful_call_restores_an_unhealthy_replica(self) -> None:
        only = _FakeReplica()
        only.mark_unhealthy()
        p = _provider_with_clients({WorkerRole.EMBED: [only]})
        assert p.embed(["a"]) == [[0.1]]
        assert only.healthy is True

    def test_model_level_errors_propagate_without_failover(self) -> None:
        from lilbee.providers.base import ProviderError

        broken = _FakeReplica(fail=ProviderError("bad input", provider="llama-server"))
        sibling = _FakeReplica(in_flight=5)
        p = _provider_with_clients({WorkerRole.EMBED: [broken, sibling]})
        with pytest.raises(ProviderError, match="bad input"):
            p.embed(["a"])
        assert broken.healthy is True  # not a connection failure, stays in the pool
        assert sibling.calls == 0

    def test_rerank_fails_over_to_a_healthy_replica(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        alive = _FakeReplica(in_flight=5)
        p = _provider_with_clients({WorkerRole.RERANK: [dead, alive]})
        assert p.rerank("q", ["c"]) == [0.5]
        assert alive.calls == 1

    def test_failover_probes_a_cooling_replica_when_none_healthy(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        cooling = _FakeReplica(in_flight=5)
        cooling.mark_unhealthy()  # the only sibling is mid cool-down
        p = _provider_with_clients({WorkerRole.EMBED: [dead, cooling]})
        assert p.embed(["a"]) == [[0.1]]  # probed instead of "no healthy replica"
        assert cooling.calls == 1
        assert cooling.healthy is True  # the successful probe restored it

    def test_retry_connection_failure_marks_the_second_replica_unhealthy(self) -> None:
        import httpx as _httpx

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        also_dead = _FakeReplica(in_flight=5, fail=_httpx.ConnectError("refused"))
        p = _provider_with_clients({WorkerRole.EMBED: [dead, also_dead]})
        with pytest.raises(_httpx.ConnectError):
            p.embed(["a"])
        assert dead.healthy is False
        assert also_dead.healthy is False  # the retry target is taken out too

    def test_retry_model_error_propagates_without_marking(self) -> None:
        import httpx as _httpx

        from lilbee.providers.base import ProviderError

        dead = _FakeReplica(fail=_httpx.ConnectError("refused"))
        broken = _FakeReplica(in_flight=5, fail=ProviderError("bad input", provider="llama-server"))
        p = _provider_with_clients({WorkerRole.EMBED: [dead, broken]})
        with pytest.raises(ProviderError, match="bad input"):
            p.embed(["a"])
        assert broken.healthy is True  # not a connection failure, stays in the pool

    def test_cooled_down_replica_gets_traffic_again_and_recovers(self, monkeypatch) -> None:
        import httpx as _httpx

        from lilbee.providers.fleet import client as client_mod
        from lilbee.providers.fleet.client import LlamaServerClient

        clock = {"now": 0.0}
        monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])
        calls: dict[str, int] = {"recovered": 0, "sibling": 0}

        def _handler(name: str):
            def handler(_request: _httpx.Request) -> _httpx.Response:
                calls[name] += 1
                return _httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

            return handler

        def _real_client(name: str) -> LlamaServerClient:
            http = _httpx.Client(
                transport=_httpx.MockTransport(_handler(name)), base_url="http://gpu"
            )
            return LlamaServerClient("http://gpu", name, http=http)

        recovered, sibling = _real_client("recovered"), _real_client("sibling")
        recovered.mark_unhealthy()
        p = _provider_with_clients({WorkerRole.EMBED: [recovered, sibling]})
        p.embed(["a"])  # within the cool-down: routed to the sibling only
        assert (calls["recovered"], calls["sibling"]) == (0, 1)
        clock["now"] = client_mod._UNHEALTHY_RETRY_S
        p.embed(["b"])  # cooled down: the replica is routable again (the probe)
        assert calls["recovered"] == 1
        assert recovered.healthy is True  # the successful probe restored it


class TestVisionTimeout:
    def test_deadline_signal_maps_to_vision_timeout_error(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet.client import ChatDeadlineError

        client = _fake_client(0)
        client.chat_bounded.side_effect = ChatDeadlineError("deadline", provider="llama-server")
        with pytest.raises(ProviderError, match="Vision OCR timed out after 12s") as excinfo:
            prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 12.0)
        # A timeout is not a connection failure, so failover must not retry it.
        assert not prov_mod.is_connection_failure(excinfo.value)

    def test_bounded_call_passes_the_deadline_to_chat_bounded(self) -> None:
        client = _fake_client(0)
        client.chat_bounded.return_value = "text"
        assert prov_mod._vision_call(client, [{"role": "user", "content": "x"}], 9.0) == "text"
        assert client.chat_bounded.call_args.kwargs["deadline_s"] == 9.0

    def test_pdf_ocr_one_timed_out_page_does_not_abort_siblings(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.vision import PageText

        monkeypatch.setattr(cfg, "vision_model", "")
        monkeypatch.setattr(cfg, "vision_ocr_concurrency", 1)
        monkeypatch.setattr("lilbee.vision.pdf_page_count", lambda _p: 2)
        monkeypatch.setattr(
            "lilbee.vision.rasterize_pdf", lambda _p: iter([(0, b"png0"), (1, b"png1")])
        )

        failed_once: list[bool] = []

        def _vision(_client, _messages, _timeout) -> str:
            if not failed_once:
                failed_once.append(True)
                raise ProviderError("Vision OCR timed out after 1s.", provider="llama-server")
            return "page two"

        monkeypatch.setattr(prov_mod, "_vision_call", _vision)
        p = _provider_with_clients({WorkerRole.VISION: [_fake_client(0)]})
        result = p.pdf_ocr(Path("doc.pdf"), backend="vision")  # type: ignore[arg-type]
        assert result == [PageText(1, ""), PageText(2, "page two")]


class TestReloadSingleFlight:
    def test_concurrent_reload_calls_dispatch_one_reload(self, monkeypatch) -> None:
        threads: list[object] = []

        class _RecordingThread:
            def __init__(self, *, target, name, daemon) -> None:
                self.target = target
                threads.append(self)

            def start(self) -> None:
                return None  # held un-run so the second call sees the in-flight guard

        monkeypatch.setattr(prov_mod.threading, "Thread", _RecordingThread)
        p = _provider_with_clients({})
        p.reload_role(WorkerRole.CHAT)
        p.reload_role(WorkerRole.EMBED)  # racing call while the first is in flight
        assert len(threads) == 1
        assert p._reload_pending is True  # queued for the in-flight thread, not dropped

    def test_reload_requested_mid_flight_runs_a_second_pass(self, monkeypatch) -> None:
        plans: list[int] = []

        def _plan() -> list:
            plans.append(len(plans))
            return [_fake_launch(WorkerRole.CHAT)]  # fresh object -> differs -> restarts

        monkeypatch.setattr(planning_mod, "plan_all_launches", _plan)
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._reloading = True  # an in-flight reload that already snapshotted its plan
        p.reload_role(WorkerRole.EMBED)  # the second settings change arrives mid-flight
        assert p._reload_pending is True
        p._reload_blocking()  # the in-flight thread runs to completion
        assert len(plans) == 2  # one pass per request: the change was applied
        assert p._reloading is False
        assert p._reload_pending is False

    def test_fresh_reload_clears_a_stale_pending_flag(self, monkeypatch) -> None:
        threads: list[object] = []

        class _RecordingThread:
            def __init__(self, *, target, name, daemon) -> None:
                threads.append(self)

            def start(self) -> None:
                return None

        monkeypatch.setattr(prov_mod.threading, "Thread", _RecordingThread)
        p = _provider_with_clients({})
        p._reload_pending = True  # left over; the fresh pass plans from current cfg
        p.reload_role(WorkerRole.CHAT)
        assert len(threads) == 1
        assert p._reload_pending is False

    def test_reload_pass_failure_clears_guards_and_propagates(self, monkeypatch) -> None:
        class _ExplodingSwap(_FakeSwap):
            def start(self, launches: list) -> None:
                raise RuntimeError("respawn failed")

        plans: list[int] = []

        def _plan() -> list:
            plans.append(len(plans))
            return [_fake_launch(WorkerRole.CHAT)]

        monkeypatch.setattr(planning_mod, "plan_all_launches", _plan)
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _ExplodingSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._reloading = True
        p._reload_pending = True
        with pytest.raises(RuntimeError, match="respawn failed"):
            p._reload_blocking()
        assert len(plans) == 2  # the pending pass still ran before the failure surfaced
        assert p._reloading is False
        assert p._reload_pending is False

    def test_failed_pass_still_applies_the_pending_change(self, monkeypatch) -> None:
        built: list[_FakeSwap] = []

        class _FlakyFirstSwap(_FakeSwap):
            def start(self, launches: list) -> None:
                if len(built) == 1:  # only the first fresh manager fails its spawn
                    raise RuntimeError("first pass failed")
                super().start(launches)

        def _factory(_d: object, _g: object) -> _FakeSwap:
            built.append(_FlakyFirstSwap())
            return built[-1]

        monkeypatch.setattr(prov_mod, "SwapManager", _factory)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        monkeypatch.setattr(
            planning_mod, "plan_all_launches", lambda: [_fake_launch(WorkerRole.CHAT)]
        )
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._reloading = True
        p._reload_pending = True  # a settings change arrived during the failing pass
        p._reload_blocking()  # must not raise: the pending pass succeeded
        assert len(built) == 2
        assert p._swaps.get(WorkerRole.CHAT) is built[-1]  # adopted by the successful pass
        assert p._reloading is False
        assert p._reload_pending is False

    def test_final_pass_failure_drops_the_dead_swap(self, monkeypatch) -> None:
        class _ExplodingSwap(_FakeSwap):
            def start(self, launches: list) -> None:
                self.running = False  # the failed restart tore the process down
                raise RuntimeError("respawn failed")

        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _ExplodingSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        monkeypatch.setattr(
            planning_mod, "plan_all_launches", lambda: [_fake_launch(WorkerRole.CHAT)]
        )
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._reloading = True
        with pytest.raises(RuntimeError, match="respawn failed"):
            p._reload_blocking()
        assert p._swaps == {}  # the next call rebuilds instead of hitting a dead swap

    def test_planning_failure_keeps_a_live_swap(self, monkeypatch) -> None:
        swap = _FakeSwap()
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: swap}
        p._reloading = True

        def _broken_plan() -> list:
            raise RuntimeError("no devices")

        monkeypatch.setattr(planning_mod, "plan_all_launches", _broken_plan)
        with pytest.raises(RuntimeError, match="no devices"):
            p._reload_blocking()
        assert p._swaps.get(WorkerRole.CHAT) is swap  # still running and serving the old config
        assert swap.shutdowns == 0

    def test_reload_clears_the_guard_when_done(self, monkeypatch) -> None:
        swap = _FakeSwap()
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: swap}
        p._reloading = True
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [])
        p._reload_blocking()
        assert swap.shutdowns == 1  # nothing planned -> the running group stops
        assert p._reloading is False
        p.reload_role(WorkerRole.CHAT)  # guard released -> a new reload can dispatch

    def test_reload_blocking_noops_when_swap_already_gone(self, monkeypatch) -> None:
        # Nothing running and nothing planned: the pass replans (a resurrect
        # would start whatever the fresh plan holds), finds nothing, and the
        # guard is still released.
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [])
        p = FleetProvider()
        p._reloading = True
        p._reload_blocking()
        assert p._reloading is False
        assert p._swaps == {}

    def test_reload_racing_shutdown_serializes_and_leaks_nothing(self, monkeypatch) -> None:
        order: list[str] = []
        gate = threading.Event()

        class _OrderedSwap(_FakeSwap):
            def shutdown(self) -> None:
                order.append("shutdown")
                super().shutdown()

        swap = _OrderedSwap()
        p = FleetProvider()
        p._swaps = {WorkerRole.CHAT: swap}
        p._reloading = True

        reload_entered = threading.Event()

        def _slow_plan() -> list:
            reload_entered.set()
            gate.wait(5.0)
            return []

        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
        monkeypatch.setattr(prov_mod, "sweep_owned", lambda _d: None)
        monkeypatch.setattr(planning_mod, "plan_all_launches", _slow_plan)
        reloader = threading.Thread(target=p._reload_blocking)
        reloader.start()
        assert reload_entered.wait(5.0)  # the reload holds the build lock first
        shutter = threading.Thread(target=p._shutdown_swap)
        shutter.start()
        time.sleep(0.05)
        assert order == []  # shutdown is blocked behind the in-flight reload
        gate.set()
        reloader.join(timeout=5.0)
        shutter.join(timeout=5.0)
        # The reload's stop phase ran first (nothing planned -> group stops), then
        # the terminal shutdown's sweep; serialized on the build lock either way.
        assert order == ["shutdown"]
        assert p._swaps == {}  # the shutdown's state cleanup still landed


class TestWarmProgressTracking:
    """The chat-role warm path drives the WarmProgress tracker for launchers."""

    @staticmethod
    def _patch_registry(monkeypatch, registry: MagicMock) -> None:
        # ModelRegistry is imported at the fleet module top, so patch it there.
        monkeypatch.setattr(prov_mod, "ModelRegistry", lambda _dir: registry)

    def test_prewarm_reads_shards_and_reports_full_byte_progress(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        shard_a = tmp_path / "model-00001.gguf"
        shard_b = tmp_path / "model-00002.gguf"
        shard_a.write_bytes(b"a" * 4096)
        shard_b.write_bytes(b"b" * 2048)
        registry = MagicMock()
        registry.shard_paths.return_value = [shard_a, shard_b]
        self._patch_registry(monkeypatch, registry)
        monkeypatch.setattr(cfg, "chat_model", "repo/model-00001.gguf")

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/model-00001.gguf")
        p._prewarm_chat_weights()

        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.READING_WEIGHTS
        assert snap.bytes_done == 4096 + 2048
        assert snap.bytes_total == 4096 + 2048
        assert snap.detail == "shard 2/2"  # multi-shard detail string

    def test_prewarm_single_shard_omits_detail(self, monkeypatch, tmp_path: Path) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        shard = tmp_path / "solo.gguf"
        shard.write_bytes(b"x" * 1024)
        registry = MagicMock()
        registry.shard_paths.return_value = [shard]
        self._patch_registry(monkeypatch, registry)
        monkeypatch.setattr(cfg, "chat_model", "repo/solo.gguf")

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/solo.gguf")
        p._prewarm_chat_weights()
        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.READING_WEIGHTS
        assert snap.bytes_done == 1024
        assert snap.detail is None  # single shard: no "shard N/M" label

    def test_prewarm_skips_unresolvable_ref(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        registry = MagicMock()
        registry.shard_paths.side_effect = KeyError("not registered")
        self._patch_registry(monkeypatch, registry)
        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("ghost/model.gguf")
        p._prewarm_chat_weights()
        # Returned before the read phase: the tracker stays at STARTING.
        assert p._warm_tracker.snapshot().phase is WarmPhase.STARTING

    def test_prewarm_skips_when_shards_are_empty(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        registry = MagicMock()
        registry.shard_paths.return_value = []  # nothing to size -> total 0
        self._patch_registry(monkeypatch, registry)
        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/m.gguf")
        p._prewarm_chat_weights()
        assert p._warm_tracker.snapshot().phase is WarmPhase.STARTING

    def test_prewarm_tolerates_an_unreadable_shard_mid_read(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        good = tmp_path / "good.gguf"
        good.write_bytes(b"a" * 2048)
        # A directory stats fine (so sizing succeeds) but open('rb') raises
        # IsADirectoryError (an OSError), exercising the inner read guard.
        unreadable = tmp_path / "broken.gguf"
        unreadable.mkdir()
        registry = MagicMock()
        registry.shard_paths.return_value = [good, unreadable]
        self._patch_registry(monkeypatch, registry)
        monkeypatch.setattr(cfg, "chat_model", "repo/m.gguf")

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        p._warm_tracker.begin("repo/m.gguf")
        p._prewarm_chat_weights()  # must not raise
        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.READING_WEIGHTS
        assert snap.bytes_done == 2048  # only the readable shard counted

    def test_preload_marks_chat_ready_when_a_warm_request_returns(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
        monkeypatch.setattr(cfg, "chat_model", "repo/m.gguf")
        p._preload_roles()  # the fake client's warm request returns -> ready
        assert p._warm_tracker.snapshot().phase is WarmPhase.READY

    def test_preload_marks_chat_failed_when_every_warm_request_raises(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        client = _fake_client()
        client.chat.side_effect = RuntimeError("engine never came up")
        p = _provider_with_clients({WorkerRole.CHAT: [client]})
        monkeypatch.setattr(p, "_prewarm_chat_weights", lambda: None)
        monkeypatch.setattr(cfg, "chat_model", "repo/m.gguf")
        p._preload_roles()
        snap = p._warm_tracker.snapshot()
        assert snap.phase is WarmPhase.ERROR
        assert snap.error

    def test_warm_progress_snapshot_exposed(self, monkeypatch) -> None:
        from lilbee.providers.warm_progress import WarmPhase

        p = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
        assert p.warm_progress() is None  # nothing warming yet
        p._warm_tracker.begin("repo/m.gguf")
        p._warm_tracker.ready()
        assert p.warm_progress().phase is WarmPhase.READY


def test_require_clients_reprobes_dead_swap(monkeypatch) -> None:
    """Empty pool + dead swap triggers a one-shot rebuild; clients are returned after."""
    from unittest import mock

    p = FleetProvider()
    p._clients = {}
    dead = mock.Mock()
    dead.is_live.return_value = False
    p._swaps = {WorkerRole.CHAT: dead}
    rebuilt = {"called": False}

    def fake_rebuild(role: WorkerRole) -> None:
        rebuilt["called"] = True
        p._clients = {WorkerRole.CHAT: [_fake_client()]}

    monkeypatch.setattr(p, "_rebuild_role", fake_rebuild, raising=False)
    clients = p._require_clients(WorkerRole.CHAT)
    assert rebuilt["called"] is True
    assert len(clients) == 1


def test_require_clients_no_reprobe_when_swap_none(monkeypatch) -> None:
    """Empty pool + no swap at all (unconfigured role) still raises, no rebuild."""
    from lilbee.providers.base import ProviderError

    rebuilt = {"called": False}

    def fake_rebuild(role: WorkerRole) -> None:
        rebuilt["called"] = True

    p = FleetProvider()
    p._swaps = {}
    p._clients = {}
    monkeypatch.setattr(p, "_rebuild_role", fake_rebuild, raising=False)
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p._require_clients(WorkerRole.CHAT)
    assert rebuilt["called"] is False


def test_require_clients_no_reprobe_when_swap_live(monkeypatch) -> None:
    """Empty pool + live swap (real misconfiguration) still raises, no rebuild."""
    from unittest import mock

    from lilbee.providers.base import ProviderError

    rebuilt = {"called": False}

    def fake_rebuild(role: WorkerRole) -> None:
        rebuilt["called"] = True

    p = FleetProvider()
    live = mock.Mock()
    live.is_live.return_value = True
    p._swaps = {WorkerRole.CHAT: live}
    p._clients = {}
    monkeypatch.setattr(p, "_rebuild_role", fake_rebuild, raising=False)
    with pytest.raises(ProviderError, match="No chat model server is running"):
        p._require_clients(WorkerRole.CHAT)
    assert rebuilt["called"] is False


def test_rebuild_role_restarts_only_that_role(monkeypatch) -> None:
    """A dead group's rebuild replaces just that role's swap; the live one stays."""
    fresh = _FakeSwap()
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: fresh)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    chat_launch, embed_launch = _fake_launch(WorkerRole.CHAT), _fake_launch(WorkerRole.EMBED)
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: [chat_launch, embed_launch])
    p = FleetProvider()
    monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
    dead, live = _FakeSwap(), _FakeSwap()
    p._swaps = {WorkerRole.EMBED: dead, WorkerRole.CHAT: live}
    # The running launches match the plan, so nothing restarts on its own; the
    # force set is what replaces the dead embed group.
    p._launches = {
        WorkerRole.EMBED: (embed_launch,),
        WorkerRole.CHAT: (chat_launch,),
    }
    p._rebuild_role(WorkerRole.EMBED)
    assert dead.shutdowns == 1  # the dead group was torn down...
    assert p._swaps[WorkerRole.EMBED] is fresh  # ...and replaced from the fresh plan
    assert p._swaps[WorkerRole.CHAT] is live  # the healthy group was never touched
    assert live.shutdowns == 0


def test_ensure_fleet_partial_failure_tears_down_started_groups(monkeypatch) -> None:
    """A later group failing to start must stop the groups already started, so a
    half-built fleet never leaks past the failure."""
    built: list[_FakeSwap] = []

    class _SecondExplodes(_FakeSwap):
        def start(self, launches: list) -> None:
            if len(built) > 1:
                raise RuntimeError("second group failed")
            super().start(launches)

    def _factory(_d: object, _g: object) -> _FakeSwap:
        built.append(_SecondExplodes())
        return built[-1]

    monkeypatch.setattr(prov_mod, "SwapManager", _factory)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    monkeypatch.setattr(
        planning_mod,
        "plan_all_launches",
        lambda: [_fake_launch(WorkerRole.CHAT), _fake_launch(WorkerRole.EMBED)],
    )
    p = FleetProvider()
    with pytest.raises(RuntimeError, match="second group failed"):
        p._ensure_fleet()
    assert built[0].shutdowns == 1  # the group that did start was torn down
    assert p._swaps == {}


def test_drop_dead_swaps_drops_only_dead_groups() -> None:
    p = FleetProvider()
    dead, live = _FakeSwap(), _FakeSwap()
    dead.running = False
    p._swaps = {WorkerRole.EMBED: dead, WorkerRole.CHAT: live}
    p._drop_dead_swaps()
    assert set(p._swaps) == {WorkerRole.CHAT}  # the live group is untouched


def test_reload_placement_dispatches_the_diff_reload(monkeypatch) -> None:
    passes: list[bool] = []
    p = FleetProvider()
    p._swaps = {WorkerRole.CHAT: _FakeSwap()}
    monkeypatch.setattr(p, "_reload_pass", lambda force=frozenset(): passes.append(True))
    p.reload_placement(wait=True)
    assert passes == [True]


def test_reload_pass_refuses_after_terminal_shutdown(monkeypatch) -> None:
    """A reload queued behind a terminal shutdown must not resurrect the fleet."""
    plans: list[int] = []
    monkeypatch.setattr(planning_mod, "plan_all_launches", lambda: plans.append(1) or [])
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: None)
    p = FleetProvider()
    p._shut_down = True
    p._reload_pass(force=frozenset((WorkerRole.CHAT,)))
    assert plans == []  # returned before planning; nothing can spawn


class TestPlanProbeLifecycle:
    """The provider owns the plan snapshot: captured on clean-box builds only."""

    def _wire(self, monkeypatch, order: list[str]) -> None:
        monkeypatch.setattr(prov_mod, "SwapManager", lambda _d, _g: _FakeSwap())
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d: order.append("reap"))
        monkeypatch.setattr(planning_mod, "capture_plan_probe", lambda: order.append("capture"))
        monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
        monkeypatch.setattr(
            planning_mod,
            "plan_all_launches",
            lambda: order.append("plan") or [_fake_launch(WorkerRole.CHAT)],
        )

    def test_first_build_snapshots_after_reaping(self, monkeypatch) -> None:
        # Capture must follow the reap (a dead owner's servers still hold VRAM
        # before it) and precede planning (the plan sizes against the snapshot).
        order: list[str] = []
        self._wire(monkeypatch, order)
        FleetProvider()._ensure_fleet()
        assert order == ["reap", "capture", "plan"]

    def test_reload_with_a_loaded_fleet_reuses_the_snapshot(self, monkeypatch) -> None:
        # THE #474 follow-up regression guard: re-planning while our own fleet
        # holds VRAM must not re-probe (which would shrink ctx/slots and diff
        # every launch); the boot snapshot stays the sizing basis.
        order: list[str] = []
        self._wire(monkeypatch, order)
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._swaps = {WorkerRole.CHAT: _FakeSwap()}
        p._reload_pass()
        assert "capture" not in order
        assert "plan" in order

    def test_resurrect_reload_recaptures_the_clean_box(self, monkeypatch) -> None:
        order: list[str] = []
        self._wire(monkeypatch, order)
        p = FleetProvider()
        monkeypatch.setattr(p, "_preload_roles", lambda roles=None: None)
        p._reload_pass(force=frozenset((WorkerRole.CHAT,)))  # nothing running
        assert order.index("capture") < order.index("plan")

    def test_full_teardown_clears_the_snapshot(self, monkeypatch) -> None:
        cleared: list[bool] = []
        monkeypatch.setattr(planning_mod, "clear_plan_probe", lambda: cleared.append(True))
        FleetProvider()._drop_swap_refs()
        assert cleared == [True]
