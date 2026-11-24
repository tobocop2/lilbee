"""The warm of a role reads the engine's own report: the device after a load, the reason after a failure."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import provider as provider_mod
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName, RequestProfile
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.roles import WorkerRole

REPLIES = Path(__file__).parent / "fixtures" / "lemonade"
MODEL_ID = "chat-0"
NOT_FOUND = 404
"""The status of the recorded load refusal; tests/fixtures/lemonade/sources.txt."""
EMBEDDING = {"data": [{"index": 0, "embedding": [0.5, 0.5]}], "object": "list"}
CHAT_DONE = b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'
ON_NPU = {"all_models_loaded": [{"model_name": "user.lilbee-embed", "device": "npu"}]}


def _reply(name: str) -> bytes:
    return (REPLIES / name).read_bytes()


def _client(
    role: WorkerRole,
    profile_of: Callable[[WorkerRole], RequestProfile],
    routes: dict[str, object],
) -> tuple[LlamaServerClient, list[str]]:
    """A client whose server answers each path in *routes*; it records every path it is asked."""
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        reply = routes[request.url.path]
        if isinstance(reply, httpx.Response):
            return reply
        if isinstance(reply, bytes):
            return httpx.Response(200, content=reply)
        return httpx.Response(200, json=reply)

    transport = httpx.MockTransport(answer)
    client = LlamaServerClient(
        "http://swap.test",
        MODEL_ID,
        http=httpx.Client(base_url="http://swap.test", transport=transport),
        async_transport=transport,
        profile=profile_of(role),
    )
    return client, asked


def test_d1_engines_without_a_health_route_skip_the_check() -> None:
    others = [a for name, a in choice.ADAPTERS.items() if name is not EngineName.LEMONADE]
    assert others
    for adapter in others:
        for role in adapter.served_roles:
            assert adapter.request_profile(role).device_check is None, adapter.name
    client, asked = _client(
        WorkerRole.EMBED, LLAMA.request_profile, {"/v1/embeddings": EMBEDDING}
    )
    provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked == ["/v1/embeddings"]


def test_d1_the_warm_reads_health_through_the_upstream_route() -> None:
    health = f"/upstream/{MODEL_ID}/v1/health"
    client, asked = _client(
        WorkerRole.EMBED, LEMONADE.request_profile, {"/v1/embeddings": EMBEDDING, health: ON_NPU}
    )
    provider_mod._warm_role(WorkerRole.EMBED, client)
    assert asked == ["/v1/embeddings", health]


def test_d1_a_reply_that_lists_only_another_model_fails_the_chat_warm() -> None:
    client, _ = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": CHAT_DONE,
            f"/upstream/{MODEL_ID}/v1/health": _reply("health-cpu-v2026.44.1.json"),
        },
    )
    with pytest.raises(ProviderError, match="does not list the model as loaded"):
        provider_mod._warm_role(WorkerRole.CHAT, client)


def test_d1_an_unreadable_health_reply_fails_the_warm() -> None:
    client, _ = _client(
        WorkerRole.EMBED,
        LEMONADE.request_profile,
        {
            "/v1/embeddings": EMBEDDING,
            f"/upstream/{MODEL_ID}/v1/health": httpx.Response(200, content=b"not json"),
        },
    )
    with pytest.raises(ProviderError, match="Lemonade did not report the device of the model"):
        provider_mod._warm_role(WorkerRole.EMBED, client)


def test_e2_a_refused_load_on_a_host_without_the_npu_gives_the_no_npu_error() -> None:
    client, asked = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": httpx.Response(
                NOT_FOUND, content=_reply("load-refused-no-npu-v2026.44.1.json")
            ),
            f"/upstream/{MODEL_ID}/v1/system-info": _reply("system-info-no-npu-v2026.44.1.json"),
        },
    )
    with pytest.raises(ProviderError) as caught:
        provider_mod._warm_role(WorkerRole.CHAT, client)
    assert str(caught.value) == (
        "Lemonade cannot run this model: this computer has no AMD XDNA2 NPU that Lemonade "
        "can use. Lemonade reports: Requires AMD XDNA 2 AMD NPU"
    )
    assert isinstance(caught.value.__cause__, ProviderError)
    assert asked == ["/v1/chat/completions", f"/upstream/{MODEL_ID}/v1/system-info"]


def test_e2_an_unreadable_system_info_keeps_the_engine_failure() -> None:
    refused = _reply("load-refused-no-npu-v2026.44.1.json")
    client, _ = _client(
        WorkerRole.CHAT,
        LEMONADE.request_profile,
        {
            "/v1/chat/completions": httpx.Response(NOT_FOUND, content=refused),
            f"/upstream/{MODEL_ID}/v1/system-info": httpx.Response(500, content=b"down"),
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
    assert f"/upstream/{MODEL_ID}/v1/system-info" not in asked


def test_a_vision_warm_sends_nothing() -> None:
    client, asked = _client(WorkerRole.VISION, LEMONADE.request_profile, {})
    provider_mod._warm_role(WorkerRole.VISION, client)
    assert asked == []
