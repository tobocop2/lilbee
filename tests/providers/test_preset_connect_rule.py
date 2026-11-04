"""A server preset reaches liter-llm with the network rule; a refusal reads as a user error."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from lilbee.core import blocked_networks, settings
from lilbee.core.config import Config, cfg
from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import WorkerRole
from lilbee.providers.sdk_backend import CompletionRequest, EmbeddingRequest, RerankRequest
from lilbee.providers.sdk_llm_provider import SdkLLMProvider
from tests import _fake_sdk as fakes

URL = "http://pub.example.test:8080/v1"
MESSAGES = [{"role": "user", "content": "hi"}]
REFUSAL = "resolves to an address that a server preset cannot use"


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Any) -> Iterator[None]:
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.server_presets = None
    settings.set_value(
        cfg.data_root,
        "server_presets",
        json.dumps([{"name": "wired", "url": URL, "roles": ["chat", "embed", "rerank"]}]),
    )
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def sdk() -> Iterator[fakes.FakeSdk]:
    fake = fakes.FakeSdk()
    fake.chat_result = fakes.text_response("hello")
    fake.stream_chunks = [fakes.chunk("hi", finish="stop")]
    fake.embed_result = fakes.embedding_response([(0, [0.5])])
    fake.rerank_result = fakes.rerank_response([(0, 0.5)])
    with fakes.installed(fake):
        yield fake


def _chat(ref: str, base: str | None = URL) -> Callable[[LiterLlmBackend], None]:
    request = CompletionRequest(ref=parse_model_ref(ref), messages=MESSAGES, api_base=base)
    return lambda backend: backend.complete(request) and None


def _stream(ref: str, base: str | None = URL) -> Callable[[LiterLlmBackend], None]:
    request = CompletionRequest(ref=parse_model_ref(ref), messages=MESSAGES, api_base=base)
    return lambda backend: list(backend.complete_stream(request)) and None


def _embed(ref: str, base: str | None = URL) -> Callable[[LiterLlmBackend], None]:
    request = EmbeddingRequest(ref=parse_model_ref(ref), inputs=["x"], api_base=base)
    return lambda backend: backend.embed(request) and None


def _rerank(ref: str, base: str | None = URL) -> Callable[[LiterLlmBackend], None]:
    request = RerankRequest(ref=parse_model_ref(ref), query="q", candidates=["a"], api_base=base)
    return lambda backend: backend.rerank(request) and None


PATHS = [_chat, _stream, _embed, _rerank]
PATH_IDS = ["chat", "stream", "embed", "rerank"]


class TestRuleForPresets:
    @pytest.mark.parametrize("path", PATHS, ids=PATH_IDS)
    def test_every_preset_request_path_passes_the_rule_to_the_client(
        self, sdk: fakes.FakeSdk, path: Callable[..., Callable[[LiterLlmBackend], None]]
    ) -> None:
        path("wired/m")(LiterLlmBackend())
        sent = sdk.clients[-1]["blocked_networks"]
        assert sent == [str(network) for network in blocked_networks.CONNECT_BLOCKED_NETWORKS]

    def test_the_rule_names_the_metadata_range_and_the_tunnel_ranges(
        self, sdk: fakes.FakeSdk
    ) -> None:
        _chat("wired/m")(LiterLlmBackend())
        sent = set(sdk.clients[-1]["blocked_networks"])
        assert {"169.254.0.0/16", "64:ff9b::/96", "64:ff9b:1::/48", "2001::/32"} <= sent

    def test_the_rule_leaves_the_private_ranges_open(self, sdk: fakes.FakeSdk) -> None:
        _chat("wired/m")(LiterLlmBackend())
        sent = set(sdk.clients[-1]["blocked_networks"])
        assert not sent & {"127.0.0.0/8", "10.0.0.0/8", "192.168.0.0/16", "::1/128", "fc00::/7"}

    @pytest.mark.parametrize("path", PATHS, ids=PATH_IDS)
    @pytest.mark.parametrize(
        "ref, base",
        [("ollama/qwen3:0.6b", "http://localhost:11434"), ("openai/gpt-4o", None)],
    )
    def test_a_local_server_or_hosted_ref_gets_no_rule(
        self,
        sdk: fakes.FakeSdk,
        path: Callable[..., Callable[[LiterLlmBackend], None]],
        ref: str,
        base: str | None,
    ) -> None:
        path(ref, base)(LiterLlmBackend())
        assert sdk.clients[-1]["blocked_networks"] is None

    def test_a_preset_and_a_local_server_at_one_address_do_not_share_a_client(
        self, sdk: fakes.FakeSdk
    ) -> None:
        backend = LiterLlmBackend()
        _chat("wired/m", "http://localhost:11434")(backend)
        _chat("ollama/openai/m", "http://localhost:11434")(backend)
        rules = [client["blocked_networks"] for client in sdk.clients]
        assert [rule is None for rule in rules] == [False, True]


class TestNoPreCallLookup:
    def test_a_preset_call_does_not_resolve_the_host_before_the_client_connects(
        self, sdk: fakes.FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def resolves(*_args: Any) -> list[Any]:
            raise AssertionError("lilbee looked the host up before the call")

        monkeypatch.setattr(blocked_networks.socket, "getaddrinfo", resolves)
        monkeypatch.setattr(socket, "getaddrinfo", resolves)
        reply = SdkLLMProvider(LiterLlmBackend()).chat(MESSAGES, model="wired/m")
        assert reply.text == "hello"

    def test_the_provider_still_refuses_a_role_the_preset_does_not_serve(
        self, sdk: fakes.FakeSdk
    ) -> None:
        with pytest.raises(ProviderError, match="does not serve the vision role"):
            SdkLLMProvider(LiterLlmBackend()).chat(
                MESSAGES, model="wired/m", role=WorkerRole.VISION
            )
        assert sdk.calls == []


class TestRefusalMessage:
    @pytest.mark.parametrize("path", PATHS, ids=PATH_IDS)
    def test_a_refused_connect_reads_as_a_bad_request_with_the_fix(
        self, sdk: fakes.FakeSdk, path: Callable[..., Callable[[LiterLlmBackend], None]]
    ) -> None:
        refusal = sdk.error("OutboundForbiddenError", "10.1.2.3 is blocked")
        sdk.chat_error = sdk.embed_error = sdk.rerank_error = sdk.stream_open_error = refusal
        with pytest.raises(ProviderError) as err:
            path("wired/m")(LiterLlmBackend())
        assert err.value.kind is ProviderErrorKind.BAD_REQUEST
        assert REFUSAL in str(err.value)
        assert "wired/m" in str(err.value)
        assert "Fix the URL of the preset" in str(err.value)

    def test_the_message_leaks_no_internal_word_and_no_address(self, sdk: fakes.FakeSdk) -> None:
        sdk.chat_error = sdk.error("OutboundForbiddenError", "10.1.2.3 is blocked")
        with pytest.raises(ProviderError) as err:
            _chat("wired/m")(LiterLlmBackend())
        text = str(err.value)
        assert "10.1.2.3" not in text
        assert "Outbound" not in text
