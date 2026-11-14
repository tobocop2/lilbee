"""A 404 from a server preset names the base URL and the role; a hosted 404 keeps its message."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.sdk_backend import CompletionRequest, EmbeddingRequest, RerankRequest
from tests import _fake_sdk as fakes

URL = "http://infinity.example.test:7997"
MESSAGES = [{"role": "user", "content": "hi"}]
HOSTED = "The provider doesn't offer"


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
    fake.chat_error = fake.embed_error = fake.rerank_error = fake.stream_open_error = fake.error(
        "NotFoundError", '{"detail":"Not Found"}'
    )
    with fakes.installed(fake):
        yield fake


def _chat(ref: str, base: str | None) -> Callable[[LiterLlmBackend], object]:
    request = CompletionRequest(ref=parse_model_ref(ref), messages=MESSAGES, api_base=base)
    return lambda backend: backend.complete(request)


def _stream(ref: str, base: str | None) -> Callable[[LiterLlmBackend], object]:
    request = CompletionRequest(ref=parse_model_ref(ref), messages=MESSAGES, api_base=base)
    return lambda backend: list(backend.complete_stream(request))


def _embed(ref: str, base: str | None) -> Callable[[LiterLlmBackend], object]:
    request = EmbeddingRequest(ref=parse_model_ref(ref), inputs=["x"], api_base=base)
    return lambda backend: backend.embed(request)


def _rerank(ref: str, base: str | None) -> Callable[[LiterLlmBackend], object]:
    request = RerankRequest(ref=parse_model_ref(ref), query="q", candidates=["a"], api_base=base)
    return lambda backend: backend.rerank(request)


PATHS = [
    pytest.param(_chat, "chat", id="chat"),
    pytest.param(_stream, "chat", id="stream"),
    pytest.param(_embed, "embed", id="embed"),
    pytest.param(_rerank, "rerank", id="rerank"),
]


@pytest.mark.parametrize(("path", "role"), PATHS)
def test_a_preset_404_names_the_url_the_role_and_the_missing_route(
    sdk: fakes.FakeSdk, path: Callable[..., Callable[[LiterLlmBackend], object]], role: str
) -> None:
    with pytest.raises(ProviderError) as err:
        path("wired/m", URL)(LiterLlmBackend())
    text = str(err.value)
    assert err.value.kind is ProviderErrorKind.NOT_FOUND
    assert URL in text
    assert f"{role} request" in text
    assert "does not serve that route" in text
    assert "route prefix" in text
    assert HOSTED not in text
    assert "Not Found" not in text


@pytest.mark.parametrize(("path", "role"), PATHS)
def test_a_hosted_404_keeps_the_account_message(
    sdk: fakes.FakeSdk, path: Callable[..., Callable[[LiterLlmBackend], object]], role: str
) -> None:
    with pytest.raises(ProviderError) as err:
        path("openai/gpt-4o", None)(LiterLlmBackend())
    assert err.value.kind is ProviderErrorKind.NOT_FOUND
    assert HOSTED in str(err.value)
    assert "route" not in str(err.value)


def test_a_local_server_404_keeps_the_account_message(sdk: fakes.FakeSdk) -> None:
    with pytest.raises(ProviderError, match=HOSTED):
        _chat("ollama/qwen3:0.6b", "http://localhost:11434")(LiterLlmBackend())


def test_a_preset_404_after_the_first_chunk_names_the_url_and_the_role(
    sdk: fakes.FakeSdk,
) -> None:
    sdk.stream_open_error = None
    sdk.stream_chunks = [fakes.chunk("hi")]
    sdk.stream_error = sdk.error("NotFoundError", "gone")
    with pytest.raises(ProviderError) as err:
        _stream("wired/m", URL)(LiterLlmBackend())
    assert URL in str(err.value)
    assert "chat request" in str(err.value)


def test_a_preset_error_of_another_kind_keeps_its_message(sdk: fakes.FakeSdk) -> None:
    sdk.chat_error = sdk.error("AuthenticationError", "no")
    with pytest.raises(ProviderError, match="rejected your API key"):
        _chat("wired/m", URL)(LiterLlmBackend())
