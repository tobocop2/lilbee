"""A warm reads the engine's own report: the device after a load, the reason after a failure."""

from __future__ import annotations

import base64
import functools
import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import client as client_mod
from lilbee.providers.fleet import provider as provider_mod
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName, RequestProfile
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.server.handlers import _chat_status

REPLIES = Path(__file__).parent / "fixtures" / "lemonade"
MODEL_ID = "chat-0"
NOT_FOUND = 404
"""The status of the recorded load refusal; tests/fixtures/lemonade/sources.txt."""
RUNNING = "/running"
HEALTH = f"/upstream/{MODEL_ID}/v1/health"
SYSTEM_INFO = f"/upstream/{MODEL_ID}/v1/system-info"
UNLOAD = f"/api/models/unload/{MODEL_ID}"
VECTOR = base64.b64encode(np.array([0.5, 0.5], dtype=np.float32).tobytes()).decode()
EMBEDDING = {"data": [{"index": 0, "embedding": VECTOR}], "object": "list"}
CHAT_DONE = b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'
ON_NPU = {"all_models_loaded": [{"model_name": "user.lilbee-embed", "device": "npu"}]}
CHAT_ON_NPU = {"all_models_loaded": [{"model_name": "user.lilbee-chat", "device": "npu"}]}
CHAT_ON_CPU = {"all_models_loaded": [{"model_name": "user.lilbee-chat", "device": "cpu"}]}
READY = {"running": [{"model": MODEL_ID, "state": "ready"}]}
NOTHING_RUNNING: dict[str, list[object]] = {"running": []}
NPU_FIX = "Check that the AMD XDNA driver is installed, then restart lilbee."

Reply = object
"""A route's answer: an httpx.Response, raw bytes, a JSON value, or a callable of the request."""


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway retry sleeps between attempts; the tests count the attempts instead."""
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)


def _reply(name: str) -> bytes:
    return (REPLIES / name).read_bytes()


def _answer(reply: Reply, request: httpx.Request) -> httpx.Response:
    if callable(reply):
        reply = reply(request)
    if isinstance(reply, httpx.Response):
        return reply
    if isinstance(reply, bytes):
        return httpx.Response(200, content=reply)
    return httpx.Response(200, json=reply)


def _client(
    role: WorkerRole,
    profile_of: Callable[[WorkerRole], RequestProfile],
    routes: dict[str, Reply],
) -> tuple[LlamaServerClient, list[str]]:
    """A client whose llama-swap answers each path in *routes*; it records every path asked.

    llama-swap lists the model as running and ready unless *routes* answers ``/running``.
    """
    asked: list[str] = []
    routes = {RUNNING: READY, **routes}

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        reply = routes.get(request.url.path, httpx.Response(NOT_FOUND, content=b"no route"))
        return _answer(reply, request)

    client = LlamaServerClient(
        "http://swap.test",
        MODEL_ID,
        profile=profile_of(role),
        load=functools.partial(provider_mod.load_role, role),
    )
    _attach(client, answer)
    return client, asked


def _attach(client: LlamaServerClient, answer: Callable[[httpx.Request], httpx.Response]) -> None:
    """Send every request of *client*, the abortable chat too, to *answer*."""
    transport = httpx.MockTransport(answer)
    client._http = httpx.Client(base_url="http://swap.test", transport=transport)
    client._async_transport = transport


def _replies(*replies: Reply) -> Callable[[httpx.Request], Reply]:
    """Answer each request with the next of *replies*, and with the last one after that."""
    queue = list(replies)

    def next_reply(_request: httpx.Request) -> Reply:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return next_reply


def test_d1_engines_without_a_health_route_skip_the_check() -> None:
    others = [a for name, a in choice.ADAPTERS.items() if name is not EngineName.LEMONADE]
    assert others
    for adapter in others:
        for role in adapter.served_roles:
            assert adapter.request_profile(role).device_check is None, adapter.name
    client, asked = _client(WorkerRole.EMBED, LLAMA.request_profile, {"/v1/embeddings": EMBEDDING})
    provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked == ["/v1/embeddings"]


def test_d1_the_warm_reads_health_through_the_upstream_route() -> None:
    client, asked = _client(
        WorkerRole.EMBED, LEMONADE.request_profile, {"/v1/embeddings": EMBEDDING, HEALTH: ON_NPU}
    )
    provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked == ["/v1/embeddings", RUNNING, HEALTH]


def test_d1_a_warm_whose_model_lemond_lists_on_the_cpu_fails_with_the_fix() -> None:
    client, _ = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {"/v1/chat/completions": CHAT_DONE, HEALTH: CHAT_ON_CPU},
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.CHAT, client)
    assert str(caught.value) == (
        f"Lemonade loaded user.lilbee-chat on the CPU, not the NPU. {NPU_FIX}"
    )


def test_d1_a_reply_that_lists_only_another_model_fails_the_chat_warm() -> None:
    client, _ = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {"/v1/chat/completions": CHAT_DONE, HEALTH: _reply("health-cpu-v2026.44.1.json")},
    )
    with pytest.raises(ProviderError, match=r"does not list user\.lilbee-chat as loaded"):
        provider_mod._warm_role(WorkerRole.CHAT, client)


def test_d1_one_unreadable_health_reply_is_read_again() -> None:
    gateway_down = httpx.Response(502, content=b"upstream restarting")
    client, asked = _client(
        WorkerRole.EMBED,
        LEMONADE.request_profile,
        {"/v1/embeddings": EMBEDDING, HEALTH: _replies(gateway_down, ON_NPU)},
    )
    provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked.count(HEALTH) == 2


@pytest.mark.parametrize(
    "unreadable",
    [httpx.Response(200, content=b"not json"), httpx.Response(503, content=b"down")],
    ids=["not-json", "gateway-503"],
)
def test_d1_a_health_reply_unreadable_until_the_retries_end_fails_the_warm(
    unreadable: httpx.Response,
) -> None:
    client, asked = _client(
        WorkerRole.EMBED,
        LEMONADE.request_profile,
        {"/v1/embeddings": EMBEDDING, HEALTH: unreadable},
    )
    with pytest.raises(ProviderError, match="Lemonade did not report the device of the model"):
        provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked.count(HEALTH) > 1


def test_d1_health_is_never_read_from_an_upstream_that_is_not_running() -> None:
    client, asked = _client(
        WorkerRole.EMBED,
        LEMONADE.request_profile,
        {"/v1/embeddings": EMBEDDING, RUNNING: NOTHING_RUNNING, HEALTH: ON_NPU},
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.EMBED, client)
    assert str(caught.value) == (
        "Lemonade did not report the device of the model: Lemonade is not running the model."
        " Check that the Lemonade server is running, then restart lilbee."
    )
    assert HEALTH not in asked


def test_d1_a_running_list_with_an_error_status_is_not_trusted() -> None:
    client, asked = _client(
        WorkerRole.EMBED,
        LEMONADE.request_profile,
        {
            "/v1/embeddings": EMBEDDING,
            RUNNING: httpx.Response(500, json=READY),
            HEALTH: ON_NPU,
        },
    )
    with pytest.raises(ProviderError, match="Lemonade did not report the device of the model"):
        provider_mod._warm_role(WorkerRole.EMBED, client)
    assert HEALTH not in asked


@pytest.mark.parametrize("listing", [{"running": "x"}, {"running": ["x"]}, []])
def test_d1_a_running_list_of_another_shape_reads_as_unreadable(listing: object) -> None:
    client, asked = _client(
        WorkerRole.EMBED,
        LEMONADE.request_profile,
        {"/v1/embeddings": EMBEDDING, RUNNING: listing, HEALTH: ON_NPU},
    )
    with pytest.raises(ProviderError, match="Lemonade did not report the device of the model"):
        provider_mod._warm_role(WorkerRole.EMBED, client)
    assert HEALTH not in asked


def test_e2_a_refused_load_on_a_host_without_the_npu_gives_the_no_npu_error() -> None:
    client, asked = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": httpx.Response(
                NOT_FOUND, content=_reply("load-refused-no-npu-v2026.44.1.json")
            ),
            SYSTEM_INFO: _reply("system-info-no-npu-v2026.44.1.json"),
        },
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.CHAT, client)
    text = str(caught.value)
    assert text.startswith(
        "This machine has no NPU that Lemonade can use: Requires AMD XDNA 2 AMD NPU. "
        "Choose a CPU or GPU engine for this role. Lemonade said: "
    )
    # The stored warm error is str(exc): it keeps Lemonade's own detail.
    assert "Detected processor: AMD EPYC 7B12" in text
    assert isinstance(caught.value.__cause__, ProviderError)
    assert asked == ["/v1/chat/completions", RUNNING, SYSTEM_INFO]


def test_e2_the_reason_is_never_read_from_an_upstream_that_is_not_running() -> None:
    client, asked = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": httpx.Response(NOT_FOUND, content=b"upstream did not start"),
            RUNNING: NOTHING_RUNNING,
            SYSTEM_INFO: _reply("system-info-no-npu-v2026.44.1.json"),
        },
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.CHAT, client)
    assert caught.value.__cause__ is None
    assert SYSTEM_INFO not in asked


def test_e2_an_unreadable_system_info_keeps_the_engine_failure() -> None:
    refused = _reply("load-refused-no-npu-v2026.44.1.json")
    client, _ = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": httpx.Response(NOT_FOUND, content=refused),
            # An error status with a body that would refuse: the status alone decides.
            SYSTEM_INFO: httpx.Response(500, content=_reply("system-info-no-npu-v2026.44.1.json")),
        },
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.CHAT, client)
    assert caught.value.__cause__ is None
    assert "Requires AMD XDNA 2 AMD NPU Detected processor" in str(caught.value)


def test_e2_a_system_info_that_shows_no_refusal_keeps_the_engine_failure() -> None:
    # Not a recorded reply: the flm reply of a host that runs it is owed by an XDNA2 run.
    runnable = {"recipes": {"flm": {"backends": {"npu": {"state": "installed"}}}}}
    client, _ = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": httpx.Response(
                NOT_FOUND, content=_reply("load-refused-no-npu-v2026.44.1.json")
            ),
            SYSTEM_INFO: runnable,
        },
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.CHAT, client)
    assert caught.value.__cause__ is None
    assert "Requires AMD XDNA 2 AMD NPU Detected processor" in str(caught.value)


def test_e2_an_engine_without_a_failure_route_keeps_its_failure() -> None:
    client, asked = _client(
        WorkerRole.EMBED,
        LLAMA.request_profile,
        {"/v1/embeddings": httpx.Response(NOT_FOUND, content=b'{"error": "no model"}')},
    )
    with pytest.raises(ProviderError):
        provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked == ["/v1/embeddings"]


def test_a_vision_warm_sends_nothing() -> None:
    client, asked = _client(WorkerRole.VISION, LEMONADE.request_profile, {})
    provider_mod._warm_role(WorkerRole.VISION, client)
    assert asked == []


class _Swap:
    """llama-swap as the provider sees it: the chat role is ready while its upstream runs."""

    def __init__(self, health: Reply, *, unload_status: int = 200) -> None:
        self.running: set[str] = set()
        # The last user message of each chat request that reached the upstream.
        self.prompts: list[str] = []
        self._health = health
        self._unload_status = unload_status
        self.log_path = Path("/nonexistent/llama-swap.log")

    def endpoint(self) -> str:
        return "http://swap.test"

    def role_ready(self, role: WorkerRole) -> bool:
        return role is WorkerRole.CHAT and MODEL_ID in self.running

    def is_live(self) -> bool:
        return True

    def record_ready(self) -> None:
        return None

    def routes(self) -> dict[str, Reply]:
        """A chat request starts the upstream; the unload route stops it."""
        return {
            "/v1/chat/completions": self._chat,
            RUNNING: self._listing,
            HEALTH: self._health,
            UNLOAD: self._unload,
        }

    def answer(self, asked: list[str]) -> Callable[[httpx.Request], httpx.Response]:
        routes = self.routes()

        def answer(request: httpx.Request) -> httpx.Response:
            asked.append(request.url.path)
            reply = routes.get(request.url.path, httpx.Response(NOT_FOUND, content=b"no route"))
            return _answer(reply, request)

        return answer

    def _chat(self, request: httpx.Request) -> Reply:
        self.running.add(MODEL_ID)
        self.prompts.append(json.loads(request.content)["messages"][-1]["content"])
        return CHAT_DONE

    def _listing(self, _request: httpx.Request) -> Reply:
        return {"running": [{"model": m, "state": "ready"} for m in sorted(self.running)]}

    def _unload(self, _request: httpx.Request) -> Reply:
        if self._unload_status == 200:
            self.running.discard(MODEL_ID)
        return httpx.Response(self._unload_status, content=b"OK")


QUESTION = "What is in my notes?"


def _chat_fleet(
    monkeypatch: pytest.MonkeyPatch, swap: _Swap
) -> tuple[FleetProvider, LlamaServerClient, list[str]]:
    """A provider with one Lemonade chat replica behind *swap*."""
    client, asked = _client(WorkerRole.CHAT, LEMONADE.request_profile, swap.routes())
    provider = FleetProvider()
    group = SwapGroup(WorkerRole.CHAT.value)
    provider._role_group = {WorkerRole.CHAT: group}
    provider._swaps = {group: swap}  # type: ignore[dict-item]
    provider._clients = {WorkerRole.CHAT: [client]}
    monkeypatch.setattr(provider, "_prewarm_chat_weights", lambda: None)
    return provider, client, asked


def _ask(provider: FleetProvider) -> ProviderError:
    """Send one chat request as a caller does: pick a replica of the role, then send to it."""
    replica = provider_mod._least_in_flight(provider._require_clients(WorkerRole.CHAT))
    with pytest.raises(ProviderError) as caught:
        replica.chat([{"role": "user", "content": QUESTION}], stream=False)
    return caught.value


CPU_REFUSAL = "Lemonade loaded user.lilbee-chat on the CPU, not the NPU."


def test_d1_a_cpu_device_unloads_the_upstream_and_the_role_serves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    swap = _Swap(CHAT_ON_CPU)
    provider, client, asked = _chat_fleet(monkeypatch, swap)
    provider._warm_chat_role([client])
    assert UNLOAD in asked
    assert provider.role_ready(WorkerRole.CHAT) is False
    status, error = _chat_status(provider)
    assert status == "error"
    assert error is not None
    assert CPU_REFUSAL in error
    asked_before = len(asked)
    refusal = _ask(provider)
    assert refusal.kind is ProviderErrorKind.BAD_REQUEST
    assert CPU_REFUSAL in str(refusal)
    assert len(asked) == asked_before
    assert QUESTION not in swap.prompts


def test_d1_after_a_replan_the_refused_role_still_answers_with_the_device_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, client, _ = _chat_fleet(monkeypatch, _Swap(CHAT_ON_CPU))
    provider._warm_chat_role([client])
    # A replan sets the refusals from the new plan (provider.py, the reload and the build).
    provider._refused_roles = {}
    refusal = _ask(provider)
    assert CPU_REFUSAL in str(refusal)
    assert "No chat model server" not in str(refusal)


def test_d1_a_client_built_again_after_a_rediscover_checks_the_device_before_it_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    swap = _Swap(CHAT_ON_CPU)
    provider, client, _ = _chat_fleet(monkeypatch, swap)
    provider._warm_chat_role([client])
    # A rediscover builds every client of the live fleet again; the new one knows nothing.
    rebuilt, asked = _client(WorkerRole.CHAT, LEMONADE.request_profile, swap.routes())
    provider._clients = {WorkerRole.CHAT: [rebuilt]}
    refusal = _ask(provider)
    assert CPU_REFUSAL in str(refusal)
    # The check loads the model with the warm prompt, reads the device, and stops it again;
    # the question never reaches the upstream, and the upstream is left stopped.
    assert swap.prompts == [provider_mod._WARM_PROMPT, provider_mod._WARM_PROMPT]
    assert asked == ["/v1/chat/completions", RUNNING, HEALTH, UNLOAD]
    assert swap.running == set()
    assert CPU_REFUSAL in str(_ask(provider))
    assert asked == ["/v1/chat/completions", RUNNING, HEALTH, UNLOAD]


def test_d1_a_bind_builds_lemonade_clients_that_check_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every bind adopts its group, in this process or in another one bound to the fleet."""
    swap = _Swap(CHAT_ON_CPU)
    launch = SimpleNamespace(
        role=WorkerRole.CHAT,
        engine=EngineName.LEMONADE,
        model_id=MODEL_ID,
        model="gemma4e-flash:e2b",
        token_cap=None,
        rerank_mode=None,
        weights_bytes=0,
        slots=1,
        ctx=0,
        replica=0,
    )
    monkeypatch.setattr(provider_mod, "_output_log_of", lambda _e, _l, log: log)
    monkeypatch.setattr(provider_mod.planning, "warn_when_chat_downsized", lambda _l: None)
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_prewarm_chat_weights", lambda: None)
    group = SwapGroup(WorkerRole.CHAT.value)
    with provider._lock:
        provider._adopt_group(group, swap, [launch])  # type: ignore[arg-type, list-item]
    (client,) = provider._clients[WorkerRole.CHAT]
    asked: list[str] = []
    _attach(client, swap.answer(asked))
    refusal = _ask(provider)
    assert CPU_REFUSAL in str(refusal)
    assert swap.prompts == [provider_mod._WARM_PROMPT]
    assert asked == ["/v1/chat/completions", RUNNING, HEALTH, UNLOAD]


def test_d1_a_refused_unload_is_logged_and_the_client_still_refuses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    swap = _Swap(CHAT_ON_CPU, unload_status=500)
    provider, client, _ = _chat_fleet(monkeypatch, swap)
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.client"):
        provider._warm_chat_role([client])
    assert "Could not stop the upstream of chat-0" in caplog.text
    assert CPU_REFUSAL in str(_ask(provider))
    assert QUESTION not in swap.prompts


def test_a_refused_replica_leaves_its_siblings_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, refused, _ = _chat_fleet(monkeypatch, _Swap(CHAT_ON_CPU))
    sibling, _ = _client(WorkerRole.CHAT, LEMONADE.request_profile, {})
    provider._warm_chat_role([refused])
    # Past the cool-down of a failed request, the refused replica is still not routable.
    monkeypatch.setattr(client_mod, "_UNHEALTHY_RETRY_S", 0.0)
    assert refused.healthy is False
    assert sibling.healthy is True
    assert provider_mod._least_in_flight([refused, sibling]) is sibling


def test_the_device_check_holds_a_reservation_for_its_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload on another thread closes only an idle retired client."""
    held: list[int] = []

    def health(_request: httpx.Request) -> Reply:
        held.append(client.in_flight)
        return CHAT_ON_NPU

    provider, client, _ = _chat_fleet(monkeypatch, _Swap(health))
    provider._warm_chat_role([client])
    assert held == [1]
    assert client.in_flight == 0
    assert client.needs_device_check is False


def test_a_client_without_a_load_request_reads_the_device_of_the_running_model() -> None:
    client = LlamaServerClient(
        "http://swap.test", MODEL_ID, profile=LEMONADE.request_profile(WorkerRole.EMBED)
    )
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return _answer({RUNNING: READY, HEALTH: ON_NPU}[request.url.path], request)

    _attach(client, answer)
    client.ensure_device()
    assert asked == [RUNNING, HEALTH]
    assert client.needs_device_check is False


def _pretend_a_check_is_running(client: LlamaServerClient) -> None:
    """Make *client* look as if another thread's check were running its load."""
    with client._device_cond:
        client._device_checking = True


def _run_waiter(call: Callable[[], object]) -> tuple[threading.Thread, list[BaseException]]:
    """Run *call* on a thread, and keep what it raises; the thread still waits after 0.3 s."""
    outcome: list[BaseException] = []

    def wait() -> None:
        try:
            call()
        except BaseException as exc:
            outcome.append(exc)

    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(0.3)
    assert waiter.is_alive()
    return waiter, outcome


def test_a_request_that_waited_out_another_check_reads_nothing_itself() -> None:
    client, asked = _client(WorkerRole.EMBED, LEMONADE.request_profile, {HEALTH: ON_NPU})
    _pretend_a_check_is_running(client)
    waiter, outcome = _run_waiter(client.ensure_device)
    # The other check ends with a confirmed device while the request waits.
    with client._device_cond:
        client._device_ok = True
        client._device_checking = False
        client._device_cond.notify_all()
    waiter.join(5)
    assert not waiter.is_alive()
    assert outcome == []
    assert asked == []


def test_a_request_that_waits_for_the_check_can_be_cancelled() -> None:
    client, asked = _client(WorkerRole.CHAT, LEMONADE.request_profile, {})
    _pretend_a_check_is_running(client)
    cancel = threading.Event()
    waiter, outcome = _run_waiter(
        lambda: client.chat_abortable([{"role": "user", "content": QUESTION}], cancel=cancel)
    )
    cancel.set()
    waiter.join(5)
    assert not waiter.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], TaskCancelledError)
    assert asked == []
    assert client.in_flight == 0


def test_a_request_that_waits_for_the_check_ends_when_the_client_closes() -> None:
    client, _ = _client(WorkerRole.EMBED, LEMONADE.request_profile, {})
    _pretend_a_check_is_running(client)
    waiter, outcome = _run_waiter(client.ensure_device)
    client.close()
    waiter.join(5)
    assert not waiter.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], TaskCancelledError)


def test_the_check_does_not_hold_its_lock_while_the_load_runs() -> None:
    """A load that fans out to worker threads can take the gate's condition."""
    client, _ = _client(WorkerRole.EMBED, LEMONADE.request_profile, {HEALTH: ON_NPU})
    free_during_load: list[bool] = []

    def take_it() -> None:
        got = client._device_cond.acquire(timeout=1)
        free_during_load.append(got)
        if got:
            client._device_cond.release()

    def load(_client: LlamaServerClient) -> None:
        worker = threading.Thread(target=take_it)
        worker.start()
        worker.join(5)

    client._load = load
    client.ensure_device()
    assert free_during_load == [True]
    assert client.needs_device_check is False


MESSAGES = [{"role": "user", "content": QUESTION}]


def _two_replicas(
    first: Reply, second: Reply
) -> tuple[provider_mod._LivePool[LlamaServerClient], list[LlamaServerClient]]:
    """Two unchecked Lemonade chat replicas, each on the device its own *first*/*second* reports."""
    clients = [
        _client(WorkerRole.CHAT, LEMONADE.request_profile, _Swap(devices).routes())[0]
        for devices in (first, second)
    ]
    pool = provider_mod._LivePool(threading.RLock(), lambda: list(clients), lambda: None)
    return pool, clients


def test_a_refused_unchecked_replica_hands_the_request_to_its_sibling() -> None:
    pool, (bad, good) = _two_replicas(CHAT_ON_CPU, CHAT_ON_NPU)
    assert bad.needs_device_check
    assert provider_mod._call_reserved(pool, lambda c: c.chat(MESSAGES, stream=False)) == "hi"
    assert bad.healthy is False
    assert good.needs_device_check is False
    assert bad.in_flight == good.in_flight == 0


def test_a_refused_unchecked_replica_hands_the_stream_to_its_sibling() -> None:
    pool, (bad, good) = _two_replicas(CHAT_ON_CPU, CHAT_ON_NPU)
    stream = provider_mod._stream_reserved(pool, lambda c: c.chat_stream_items(MESSAGES))
    assert "hi" in "".join(str(getattr(item, "content", item)) for item in stream)
    assert bad.healthy is False
    assert good.needs_device_check is False


def test_with_every_replica_refused_the_request_gets_the_device_error() -> None:
    pool, _ = _two_replicas(CHAT_ON_CPU, CHAT_ON_CPU)
    with pytest.raises(ProviderError, match="not the NPU"):
        provider_mod._call_reserved(pool, lambda c: c.chat(MESSAGES, stream=False))
    pool, _ = _two_replicas(CHAT_ON_CPU, CHAT_ON_CPU)
    with pytest.raises(ProviderError, match="not the NPU"):
        provider_mod._stream_reserved(pool, lambda c: c.chat_stream_items(MESSAGES))


def test_a_lone_refused_replica_gives_the_device_error() -> None:
    for open_it in (
        lambda pool: provider_mod._call_reserved(pool, lambda c: c.chat(MESSAGES, stream=False)),
        lambda pool: provider_mod._stream_reserved(pool, lambda c: c.chat_stream_items(MESSAGES)),
    ):
        pool, (bad, _) = _two_replicas(CHAT_ON_CPU, CHAT_ON_CPU)
        alone = provider_mod._LivePool(threading.RLock(), lambda: [bad], lambda: None)
        with pytest.raises(ProviderError, match="not the NPU"):
            open_it(alone)
        assert bad.in_flight == 0


def test_a_refused_replica_hands_a_failover_call_to_its_sibling() -> None:
    """The embed, count and rerank calls retry on a sibling for a refusal as for a dead replica."""
    pool, (bad, good) = _two_replicas(CHAT_ON_CPU, CHAT_ON_NPU)
    assert provider_mod._call_with_failover(pool, lambda c: c.chat(MESSAGES, stream=False)) == "hi"
    assert bad.healthy is False
    assert good.needs_device_check is False
    pool, _ = _two_replicas(CHAT_ON_CPU, CHAT_ON_CPU)
    with pytest.raises(ProviderError, match="not the NPU"):
        provider_mod._call_with_failover(pool, lambda c: c.chat(MESSAGES, stream=False))


def _vision_pool(clients: list[LlamaServerClient]) -> provider_mod._LivePool:
    return provider_mod._LivePool(
        threading.RLock(),
        lambda: [provider_mod._VisionReplica(c, 1) for c in clients],
        lambda: None,
    )


def test_a_refused_vision_replica_hands_the_request_to_its_sibling() -> None:
    _, (bad, good) = _two_replicas(CHAT_ON_CPU, CHAT_ON_NPU)
    pool = _vision_pool([bad, good])
    assert provider_mod._dispatch_vision(pool, lambda c: c.chat(MESSAGES, stream=False)) == "hi"
    assert bad.healthy is False
    _, refused = _two_replicas(CHAT_ON_CPU, CHAT_ON_CPU)
    with pytest.raises(ProviderError, match="not the NPU"):
        provider_mod._dispatch_vision(
            _vision_pool(refused), lambda c: c.chat(MESSAGES, stream=False)
        )


def test_d1_a_later_good_warm_clears_the_error_of_an_unreadable_health_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = httpx.Response(503, content=b"down")
    health = _replies(*[unreadable] * (client_mod._GATEWAY_RETRIES + 1), CHAT_ON_NPU)
    provider, client, asked = _chat_fleet(monkeypatch, _Swap(health))
    provider._warm_chat_role([client])
    first = provider.warm_progress()
    assert first is not None
    assert first.phase is WarmPhase.ERROR
    assert UNLOAD not in asked
    assert provider._clients[WorkerRole.CHAT] == [client]
    provider._warm_chat_role([client])
    second = provider.warm_progress()
    assert second is not None
    assert second.phase is WarmPhase.READY
    assert provider._warm_errors == {}
    assert _chat_status(provider) == ("ready", None)
