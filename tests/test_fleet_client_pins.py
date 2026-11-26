"""Pins for the error text a fleet client builds, found loose by a mutation run."""

from __future__ import annotations

import httpx
import pytest

from lilbee.providers.base import ProviderError, WarmTimeoutError
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole

_PROFILE = LLAMA.request_profile(WorkerRole.CHAT)


def _client(model_ref: str | None = None) -> LlamaServerClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    return LlamaServerClient(
        "http://gpu0",
        "m",
        http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0"),
        async_transport=httpx.MockTransport(handler),
        profile=_PROFILE,
        model_ref=model_ref,
    )


def _refusal(resp: httpx.Response) -> str:
    with pytest.raises(ProviderError) as excinfo:
        _client()._raise_for_status(resp)
    return str(excinfo.value)


def test_a_busy_engine_answers_one_plain_sentence() -> None:
    message = _refusal(httpx.Response(429, text="Too many requests"))
    assert message == "The model is busy with too many requests. Try again in a moment."


def test_a_failed_answer_with_no_body_names_only_the_status() -> None:
    assert _refusal(httpx.Response(500, text="")) == f"{_PROFILE.label} returned HTTP 500"


def test_a_failed_answer_keeps_the_first_600_characters_of_its_body() -> None:
    message = _refusal(httpx.Response(500, text="x" * 700))
    assert message == f"{_PROFILE.label} returned HTTP 500: " + "x" * 600


def test_a_warm_timeout_names_the_model_of_the_client() -> None:
    error = _client("org/repo/chat.gguf")._warm_timeout_error(30.0)
    assert isinstance(error, WarmTimeoutError)
    assert str(error) == "The engine for org/repo/chat.gguf did not finish starting in time."
    assert error.wait_s == 30.0


def test_a_warm_timeout_with_no_model_name_says_the_model() -> None:
    error = _client()._warm_timeout_error(None)
    assert str(error) == "The engine for the model did not finish starting in time."
