"""A warm reads the engine's own report: the device after a load, the reason after a failure."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

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
NPU_FIX = (
    "Check that the XDNA driver is installed and that `lemonade system-info` lists the npu device."
)

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

    transport = httpx.MockTransport(answer)
    client = LlamaServerClient(
        "http://swap.test",
        MODEL_ID,
        http=httpx.Client(base_url="http://swap.test", transport=transport),
        async_transport=transport,
        profile=profile_of(role),
    )
    return client, asked


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
    with pytest.raises(ProviderError, match="does not list user.lilbee-chat as loaded"):
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

    def __init__(self, running: set[str]) -> None:
        self._running = running

    def role_ready(self, role: WorkerRole) -> bool:
        return role is WorkerRole.CHAT and MODEL_ID in self._running

    def is_live(self) -> bool:
        return True

    def record_ready(self) -> None:
        return None


def _chat_fleet(
    monkeypatch: pytest.MonkeyPatch, health: Reply, *, unload_status: int = 200
) -> tuple[FleetProvider, LlamaServerClient, list[str]]:
    """A provider with one Lemonade chat replica behind a llama-swap that can unload it."""
    running = {MODEL_ID}

    def listing(_request: httpx.Request) -> Reply:
        return {"running": [{"model": m, "state": "ready"} for m in sorted(running)]}

    def unload(_request: httpx.Request) -> Reply:
        if unload_status == 200:
            running.discard(MODEL_ID)
        return httpx.Response(unload_status, content=b"OK")

    client, asked = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {"/v1/chat/completions": CHAT_DONE, RUNNING: listing, HEALTH: health, UNLOAD: unload},
    )
    provider = FleetProvider()
    group = SwapGroup(WorkerRole.CHAT.value)
    provider._role_group = {WorkerRole.CHAT: group}
    provider._swaps = {group: _Swap(running)}  # type: ignore[dict-item]
    provider._clients = {WorkerRole.CHAT: [client]}
    monkeypatch.setattr(provider, "_prewarm_chat_weights", lambda: None)
    return provider, client, asked


def test_d1_a_cpu_device_unloads_the_upstream_and_the_role_serves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, client, asked = _chat_fleet(monkeypatch, CHAT_ON_CPU)
    provider._warm_chat_role([client])
    assert UNLOAD in asked
    assert provider.role_ready(WorkerRole.CHAT) is False
    status, error = _chat_status(provider)
    assert status == "error"
    assert error is not None
    assert "Lemonade loaded user.lilbee-chat on the CPU, not the NPU." in error
    asked_before = len(asked)
    with pytest.raises(ProviderError) as caught:
        provider._require_clients(WorkerRole.CHAT)
    assert caught.value.kind is ProviderErrorKind.BAD_REQUEST
    assert "not the NPU" in str(caught.value)
    assert len(asked) == asked_before


def test_d1_a_refused_unload_is_logged_and_the_replica_still_leaves(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, client, _ = _chat_fleet(monkeypatch, CHAT_ON_CPU, unload_status=500)
    with caplog.at_level("WARNING", logger="lilbee.providers.fleet.client"):
        provider._warm_chat_role([client])
    assert "Could not stop the upstream of chat-0" in caplog.text
    assert provider._clients[WorkerRole.CHAT] == []


def test_a_refused_replica_leaves_its_siblings_serving() -> None:
    provider = FleetProvider()
    held: list[int] = []

    def unload(_request: httpx.Request) -> Reply:
        held.append(refused.in_flight)
        return b"OK"

    refused, asked = _client(WorkerRole.CHAT, LEMONADE.request_profile, {UNLOAD: unload})
    sibling, _ = _client(WorkerRole.CHAT, LEMONADE.request_profile, {})
    provider._clients = {WorkerRole.CHAT: [refused, sibling]}
    provider._remove_refused_replica(WorkerRole.CHAT, refused, "refused")
    assert provider._clients[WorkerRole.CHAT] == [sibling]
    assert WorkerRole.CHAT not in provider._refused_roles
    assert asked == [UNLOAD]
    # A reload on another thread closes only an idle retired client, so the unload
    # holds a reservation, and gives it back after.
    assert held == [1]
    assert refused.in_flight == 0


def test_a_replica_that_a_reload_already_replaced_changes_nothing() -> None:
    provider = FleetProvider()
    client, asked = _client(WorkerRole.CHAT, LEMONADE.request_profile, {UNLOAD: b"OK"})
    provider._remove_refused_replica(WorkerRole.CHAT, client, "refused")
    assert asked == []
    assert WorkerRole.CHAT not in provider._refused_roles
    assert WorkerRole.CHAT not in provider._clients


def test_d1_a_later_good_warm_clears_the_error_of_an_unreadable_health_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = httpx.Response(503, content=b"down")
    health = _replies(*[unreadable] * (client_mod._GATEWAY_RETRIES + 1), CHAT_ON_NPU)
    provider, client, asked = _chat_fleet(monkeypatch, health)
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
