"""What lilbee does about the engines' gaps. Each row is a strict xfail until its item lands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole
from tests.conformance.fake_engine import PROFILES, FakeEngine

_TURN = [{"role": "user", "content": "Say ok."}]
_OVERSIZE = [{"role": "user", "content": "w " * 60000}]
_TOKEN_CAP = 16
_FAILS_AS_STATED = (AssertionError, ProviderError)


def _error_of(call: Callable[[], Any]) -> Exception | None:
    """The provider error *call* raises, or None when it returns."""
    try:
        call()
    except ProviderError as exc:
        return exc
    return None


def _client(engine: FakeEngine, model: str, **options: Any) -> LlamaServerClient:
    profile = LLAMA.request_profile(WorkerRole.CHAT)
    return LlamaServerClient(engine.url, model, timeout=5.0, profile=profile, **options)


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L1, owner item 1180")
def test_L1_embed_without_tokenize_estimates_the_cut() -> None:
    """A server with no /tokenize or /detokenize still embeds an oversize input."""
    with FakeEngine(PROFILES["lemonade"].behaviour) as engine:
        vectors = _client(engine, "m-embed", token_cap=_TOKEN_CAP).embed(["w " * 500])
    assert len(vectors) == 1


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L2, owner item 1180")
def test_L2_json_404_on_apply_template_falls_back_to_an_estimate() -> None:
    """A JSON error envelope on a missing route is an absent route, so the count estimates."""
    with FakeEngine(PROFILES["lemonade"].behaviour) as engine:
        count = _client(engine, "m-chat").count_chat_prompt_tokens(_TURN)
    assert count > 0


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L3, owner item 1180")
def test_L3_reasoning_field_is_read_as_well_as_reasoning_content() -> None:
    """An engine that names the field `reasoning` still gets its reasoning shown as <think> text."""
    with FakeEngine(PROFILES["mlx_lm"].behaviour) as engine:
        text = _client(engine, "m-chat", inline_reasoning=True).chat(_TURN)
    assert text.startswith("<think>")


def test_L4_sequence_length_overflow_text_is_a_context_overflow() -> None:
    """The text "sequence length is greater than" classifies as context overflow, not unknown."""
    with FakeEngine(PROFILES["mistralrs_cpu"].behaviour) as engine:
        error = _error_of(lambda: _client(engine, "m-chat").chat(_OVERSIZE))
    assert isinstance(error, ProviderError)
    assert error.kind is ProviderErrorKind.CONTEXT_OVERFLOW


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L5, owner item 1180")
def test_L5_stream_with_only_an_error_frame_raises() -> None:
    """A stream whose single data frame is an error object ends in an error, not an empty answer."""
    with FakeEngine(PROFILES["mistralrs_cuda"].behaviour) as engine:
        error = _error_of(lambda: list(_client(engine, "m-chat").chat_stream_items(_TURN)))
    assert isinstance(error, ProviderError)


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L6, owner items 1180 and 1193")
def test_L6_engine_that_claims_rerank_gets_the_rerank_route() -> None:
    """The client asks /v1/rerank of an engine that serves it, and no longer pools embeddings."""
    with FakeEngine(PROFILES["lemonade"].behaviour) as engine:
        _client(engine, "m-rerank").rerank("capital of France", ["Paris is the capital of France"])
        seen = list(engine.seen)
    assert "POST /v1/rerank" in seen


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L7, owner item 1193")
def test_L7_failing_role_is_reported_unavailable_by_model_name() -> None:
    """A chat role whose engine cannot serve chat says so with the model's name."""
    with FakeEngine(PROFILES["mistralrs_cuda"].behaviour) as engine:
        error = _error_of(lambda: _client(engine, "m-chat", model_ref="m-chat").chat(_TURN))
    assert isinstance(error, ProviderError)
    assert "m-chat is unavailable" in str(error)


@pytest.mark.xfail(strict=True, raises=_FAILS_AS_STATED, reason="L8, owner item 1180")
def test_L8_engine_that_cannot_serve_chat_is_not_ready_when_health_answers() -> None:
    """Health 200 with every chat call a 503 is not a ready engine (row 37)."""
    with FakeEngine(PROFILES["mistralrs_cuda"].behaviour) as engine:
        ready = _client(engine, "m-chat").health()
    assert ready is False
