"""liter-llm failures are classified into ``ProviderErrorKind`` by exception type.

liter-llm is an optional extra, so a stand-in module with its exception classes
is injected. The classifier only reads the class name, so the real classes are
not required to verify the table.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.backends.liter_llm.errors import (
    _ERROR_KINDS,
    _KIND_MESSAGES,
    PresetCall,
    _classify_error,
    _provider_error,
)
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import WorkerRole
from lilbee.providers.sdk_backend import CompletionRequest, EmbeddingRequest
from tests import _fake_sdk as fakes

# The kinds this backend can actually raise. Kinds that only a local engine can
# produce are excluded: a message for one here is unreachable text that reads as
# coverage, and the fleet surfaces those in its own words.
_SDK_KINDS = frozenset(
    {
        ProviderErrorKind.AUTH,
        ProviderErrorKind.NOT_FOUND,
        ProviderErrorKind.RATE_LIMIT,
        ProviderErrorKind.CONTEXT_OVERFLOW,
        ProviderErrorKind.BAD_REQUEST,
        ProviderErrorKind.CONNECTION,
        ProviderErrorKind.SERVER,
    }
)


def test_every_actionable_kind_has_a_user_message() -> None:
    # Adding a kind without a message would silently fall back to the raw error.
    for kind in _SDK_KINDS:
        assert kind in _KIND_MESSAGES, f"{kind} has no user-facing message"
        assert "{model}" in _KIND_MESSAGES[kind]
    assert ProviderErrorKind.UNKNOWN not in _KIND_MESSAGES


def test_the_table_carries_nothing_this_backend_cannot_raise() -> None:
    assert set(_KIND_MESSAGES) == set(_SDK_KINDS)


def test_every_table_entry_names_a_class_liter_llm_defines() -> None:
    assert _ERROR_KINDS
    assert set(_ERROR_KINDS) <= set(fakes.ERROR_NAMES)


@pytest.fixture
def sdk() -> Iterator[fakes.FakeSdk]:
    with fakes.installed(fakes.FakeSdk()) as fake:
        yield fake


@pytest.mark.parametrize(
    ("exc_name", "expected"),
    [
        ("AuthenticationError", ProviderErrorKind.AUTH),
        ("NotFoundError", ProviderErrorKind.NOT_FOUND),
        ("RateLimitedError", ProviderErrorKind.RATE_LIMIT),
        ("ProviderQuotaExceededError", ProviderErrorKind.RATE_LIMIT),
        ("ContextWindowExceededError", ProviderErrorKind.CONTEXT_OVERFLOW),
        ("BadRequestError", ProviderErrorKind.BAD_REQUEST),
        ("ContentPolicyError", ProviderErrorKind.BAD_REQUEST),
        ("EndpointNotSupportedError", ProviderErrorKind.BAD_REQUEST),
        ("LiterLlmTimeoutError", ProviderErrorKind.CONNECTION),
        ("NetworkError", ProviderErrorKind.CONNECTION),
        ("StreamingError", ProviderErrorKind.CONNECTION),
        ("ServiceUnavailableError", ProviderErrorKind.SERVER),
        ("ServerError", ProviderErrorKind.SERVER),
        ("InternalError", ProviderErrorKind.UNKNOWN),
    ],
)
def test_classifies_each_exception_type(
    sdk: fakes.FakeSdk, exc_name: str, expected: ProviderErrorKind
) -> None:
    assert _classify_error(sdk.error(exc_name)) == expected


def test_unrecognized_exception_is_unknown() -> None:
    assert _classify_error(RuntimeError("???")) == ProviderErrorKind.UNKNOWN


def test_provider_error_recognized_kind_drops_raw_blob(sdk: fakes.FakeSdk) -> None:
    # The raw SDK blob must not reach the user.
    blob = 'b\'{"error": {"code": 429, ...huge JSON dump...}}\''
    err = _provider_error(
        "Chat failed", sdk.error("RateLimitedError", blob), "gemini/gemini-3.1-pro"
    )
    assert err.kind == ProviderErrorKind.RATE_LIMIT
    assert err.provider == "remote"
    assert "gemini-3.1-pro" in str(err)
    assert "quota" in str(err).lower() or "rate" in str(err).lower()
    assert "lilbee" in str(err).lower()
    assert "huge JSON dump" not in str(err)


def test_provider_error_unknown_keeps_raw_fallback() -> None:
    err = _provider_error("Embedding failed", RuntimeError("nope"), "openai/text-embedding-3-small")
    assert err.kind == ProviderErrorKind.UNKNOWN
    assert str(err) == "Embedding failed: nope"


def test_provider_error_is_a_provider_error(sdk: fakes.FakeSdk) -> None:
    err = _provider_error(
        "Rerank failed", sdk.error("AuthenticationError", "bad key"), "cohere/rerank"
    )
    assert isinstance(err, ProviderError)
    assert err.kind == ProviderErrorKind.AUTH


def test_complete_surfaces_classified_message_end_to_end(sdk: fakes.FakeSdk) -> None:
    # A Gemini quota error driven all the way through backend.complete. The user
    # must see the clean rate-limit message, never the raw blob.
    sdk.chat_error = sdk.error(
        "ProviderQuotaExceededError", 'b\'{"error": {"code": 429, ...blob...}}\''
    )
    req = CompletionRequest(
        ref=parse_model_ref("gemini/gemini-3.1-pro"),
        messages=[{"role": "user", "content": "test"}],
    )
    with pytest.raises(ProviderError) as caught:
        LiterLlmBackend().complete(req)
    assert caught.value.kind == ProviderErrorKind.RATE_LIMIT
    assert "gemini-3.1-pro" in str(caught.value)
    assert "blob" not in str(caught.value)


# The 400 body a local server returned for a prompt past its window, as recorded.
_OVERFLOW_400 = (
    '{"error":{"message":"Prompt sequence length is greater than 40960, '
    'perhaps consider using `truncate_sequence`?",'
    '"type":"invalid_request_error","param":null,"code":null}}'
)


def test_prompt_length_400_through_the_backend_is_a_context_overflow(sdk: fakes.FakeSdk) -> None:
    sdk.chat_error = sdk.error("BadRequestError", _OVERFLOW_400)
    req = CompletionRequest(
        ref=parse_model_ref("gemini/gemini-3.1-pro"),
        messages=[{"role": "user", "content": "test"}],
    )
    with pytest.raises(ProviderError) as caught:
        LiterLlmBackend().complete(req)
    assert caught.value.kind == ProviderErrorKind.CONTEXT_OVERFLOW
    assert "context window" in str(caught.value)


@pytest.mark.parametrize(
    "message",
    [
        '{"error":{"message":"Unknown field `foo`","type":"invalid_request_error"}}',
        '{"error":{"message":"Prompt sequence length is invalid"}}',
    ],
)
def test_unrelated_400_through_the_backend_stays_a_bad_request(
    sdk: fakes.FakeSdk, message: str
) -> None:
    sdk.chat_error = sdk.error("BadRequestError", message)
    req = CompletionRequest(
        ref=parse_model_ref("gemini/gemini-3.1-pro"),
        messages=[{"role": "user", "content": "test"}],
    )
    with pytest.raises(ProviderError) as caught:
        LiterLlmBackend().complete(req)
    assert caught.value.kind == ProviderErrorKind.BAD_REQUEST


def test_prompt_length_text_on_another_error_type_keeps_its_kind(sdk: fakes.FakeSdk) -> None:
    assert _classify_error(sdk.error("ServerError", _OVERFLOW_400)) == ProviderErrorKind.SERVER


def _embed_error(sdk: fakes.FakeSdk, exc: BaseException) -> ProviderError:
    sdk.embed_error = exc
    request = EmbeddingRequest(ref=parse_model_ref("openai/text-embedding-3-small"), inputs=["a"])
    with pytest.raises(ProviderError) as caught:
        LiterLlmBackend().embed(request)
    return caught.value


_PROMPT_TEXT = "private prompt words"
_MODEL = "openai/text-embedding-3-small"
_NO_REASON = (
    f"The server for {_MODEL} refused the request with HTTP 422 and gave no readable reason. "
    "Check the request limits of the server."
)


def _quoted(reason: str) -> str:
    return (
        f'The server for {_MODEL} refused the request with HTTP 422: "{reason}". '
        "Check the request limits of the server."
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            '{"error":"batch size 33 > maximum allowed batch size 32","error_type":"Validation"}',
            _quoted("batch size 33 > maximum allowed batch size 32"),
            id="tei-string-error",
        ),
        pytest.param(
            "too many inputs", _quoted("too many inputs"), id="openai-reason-unwrapped-by-the-sdk"
        ),
        pytest.param(
            '{"error":{"message":"too many inputs","type":"invalid_request_error"}}',
            _quoted("too many inputs"),
            id="nested-error-object",
        ),
        pytest.param(
            '{"detail":[{"type":"too_long","loc":["body","input"],"msg":"List too long",'
            f'"input":["{_PROMPT_TEXT}"]}},"junk",{{"msg":"  "}},{{"msg":"Second reason"}}]}}',
            _quoted("List too long; Second reason"),
            id="fastapi-detail-list-with-input-echo",
        ),
        pytest.param(
            '{"error":"first","detail":"second","message":"third"}',
            _quoted("first"),
            id="error-beats-detail-and-message",
        ),
        pytest.param(
            '{"detail":"second","message":"third"}', _quoted("second"), id="detail-beats-message"
        ),
        pytest.param(
            '{"error":{"code":7},"detail":"second"}',
            _quoted("second"),
            id="error-without-text-falls-to-detail",
        ),
        pytest.param('{"error":"bad \\ud800 value"}', _quoted("bad ? value"), id="lone-surrogate"),
        pytest.param('{"detail":"too many inputs"}', _quoted("too many inputs"), id="detail-text"),
        pytest.param('{"message":"too many inputs"}', _quoted("too many inputs"), id="message"),
        pytest.param(
            '{"error":"line one\\n   line two"}', _quoted("line one line two"), id="whitespace"
        ),
        pytest.param(
            f"<html><body>502 {_PROMPT_TEXT}</body></html>", _NO_REASON, id="html-proxy-body"
        ),
        pytest.param(
            f"\n  <!DOCTYPE html><p>{_PROMPT_TEXT}</p>", _NO_REASON, id="html-after-whitespace"
        ),
        pytest.param(f'{{"input":"{_PROMPT_TEXT}"}}', _NO_REASON, id="unknown-json"),
        pytest.param('{"error":{"code":7},"detail":[]}', _NO_REASON, id="no-text-field"),
        pytest.param(f'["{_PROMPT_TEXT}"]', _NO_REASON, id="json-list"),
        pytest.param("", _NO_REASON, id="empty-body"),
        pytest.param("  \n ", _NO_REASON, id="blank-body"),
    ],
)
def test_a_422_shows_only_the_reason_of_the_reply(
    sdk: fakes.FakeSdk, body: str, expected: str
) -> None:
    error = _embed_error(sdk, sdk.error("BadRequestError", body, status=422))
    assert str(error) == expected
    assert _PROMPT_TEXT not in str(error)
    assert error.kind is ProviderErrorKind.BAD_REQUEST
    str(error).encode("utf-8")


@pytest.mark.parametrize(
    ("length", "shown"),
    [(199, 199), (200, 200), (201, 203)],
    ids=["below-the-cap", "at-the-cap", "above-the-cap"],
)
def test_a_422_reason_is_capped_at_200_characters(
    sdk: fakes.FakeSdk, length: int, shown: int
) -> None:
    error = _embed_error(sdk, sdk.error("BadRequestError", "a" * length, status=422))
    expected = "a" * length if length <= 200 else "a" * 200 + "..."
    assert len(expected) == shown
    assert str(error) == _quoted(expected)


def test_a_422_reason_is_cut_at_a_word_edge_without_a_trailing_blank(sdk: fakes.FakeSdk) -> None:
    body = json.dumps({"error": "word " * 200})
    error = _embed_error(sdk, sdk.error("BadRequestError", body, status=422))
    assert str(error) == _quoted(("word " * 40).rstrip()[:200].rstrip() + "...")


def test_a_reason_that_echoes_the_request_shows_up_to_the_cap(sdk: fakes.FakeSdk) -> None:
    body = json.dumps({"detail": [{"msg": f"Invalid value {_PROMPT_TEXT} " * 20}]})
    error = _embed_error(sdk, sdk.error("BadRequestError", body, status=422))
    reason = str(error).split('"')[1]
    assert reason.startswith(f"Invalid value {_PROMPT_TEXT}")
    assert len(reason) <= 203


def test_a_400_bad_request_still_names_the_model(sdk: fakes.FakeSdk) -> None:
    body = '{"error":"batch size 33 > maximum allowed batch size 32"}'
    error = _embed_error(sdk, sdk.error("BadRequestError", body, status=400))
    assert "Check the model name and your settings" in str(error)
    assert "HTTP 422" not in str(error)


def test_a_422_that_names_the_prompt_length_stays_a_context_overflow(sdk: fakes.FakeSdk) -> None:
    error = _embed_error(sdk, sdk.error("BadRequestError", _OVERFLOW_400, status=422))
    assert error.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert "HTTP 422" not in str(error)


def test_a_content_policy_refusal_keeps_the_bad_request_text(sdk: fakes.FakeSdk) -> None:
    error = _embed_error(sdk, sdk.error("ContentPolicyError", "blocked"))
    assert "Check the model name and your settings" in str(error)


_PRESET = PresetCall("http://box:8080/v1", WorkerRole.EMBED)


@pytest.mark.parametrize(
    ("exc_name", "preset", "expected_kind", "expected_text"),
    [
        pytest.param(
            "NotFoundError",
            _PRESET,
            ProviderErrorKind.NOT_FOUND,
            "The server for m answered 404 to the embed request at http://box:8080/v1. "
            "Check the base URL and the route prefix of the server. "
            "Check that the server has a model loaded for the embed role.",
            id="not-found-on-a-preset-names-role-and-url",
        ),
        pytest.param(
            "NotFoundError",
            None,
            ProviderErrorKind.NOT_FOUND,
            "The provider doesn't offer m on your account. "
            "Pick a different model or check the name.",
            id="not-found-without-a-preset",
        ),
        pytest.param(
            "RateLimitedError",
            _PRESET,
            ProviderErrorKind.RATE_LIMIT,
            _KIND_MESSAGES[ProviderErrorKind.RATE_LIMIT].format(model="m"),
            id="a-kind-message-names-the-model",
        ),
        pytest.param(
            "OutboundForbiddenError",
            None,
            ProviderErrorKind.BAD_REQUEST,
            "The server for m resolves to an address that a server preset cannot use. "
            "Fix the URL of the preset or pick a different model.",
            id="a-type-message-wins-over-the-kind-message",
        ),
        pytest.param(
            "SerializationError",
            None,
            ProviderErrorKind.UNKNOWN,
            "The server for m sent a reply that lilbee cannot read: "
            "a field of the OpenAI reply format is missing. "
            "Check that the server answers in the OpenAI reply format.",
            id="a-reply-the-sdk-cannot-read-gets-a-plain-sentence",
        ),
        pytest.param(
            "InternalError",
            None,
            ProviderErrorKind.UNKNOWN,
            "Embedding failed: boom",
            id="an-unmapped-class-keeps-the-raw-error",
        ),
    ],
)
def test_the_error_text_and_kind_of_each_mapping(
    sdk: fakes.FakeSdk,
    exc_name: str,
    preset: PresetCall | None,
    expected_kind: ProviderErrorKind,
    expected_text: str,
) -> None:
    error = _provider_error("Embedding failed", sdk.error(exc_name), "m", preset=preset)
    assert str(error) == expected_text
    assert error.kind is expected_kind
    assert error.provider == "remote"


def test_a_422_error_names_the_remote_provider_and_the_bad_request_kind(
    sdk: fakes.FakeSdk,
) -> None:
    error = _provider_error(
        "Embedding failed", sdk.error("BadRequestError", "too many", status=422), "m"
    )
    assert error.provider == "remote"
    assert error.kind is ProviderErrorKind.BAD_REQUEST
