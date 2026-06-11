"""Tests for the OpenAI-shaped error envelope."""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.server.chat_completions_api.errors import (
    COMPLETIONS_ERROR_TYPES,
    CompletionsErrorCode,
    classify_provider_error,
    completions_error_body,
)
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
)


class TestCompletionsErrorBody:
    def test_model_not_found_shape(self) -> None:
        body = completions_error_body(CompletionsErrorCode.MODEL_NOT_FOUND, "Model 'foo' not found")
        assert body == {
            "error": {
                "message": "Model 'foo' not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }

    def test_model_does_not_support_tools_is_invalid_request_error(self) -> None:
        body = completions_error_body(
            CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS, "no tools for x"
        )
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "model_does_not_support_tools"

    def test_authentication_error_type(self) -> None:
        body = completions_error_body(CompletionsErrorCode.INVALID_API_KEY, "Bad token")
        assert body["error"]["type"] == "authentication_error"
        assert body["error"]["code"] == "invalid_api_key"

    def test_rate_limit_exceeded_is_rate_limit_error(self) -> None:
        body = completions_error_body(CompletionsErrorCode.RATE_LIMIT_EXCEEDED, "Backend busy")
        assert body["error"]["type"] == "rate_limit_error"

    def test_internal_error_is_api_error(self) -> None:
        body = completions_error_body(CompletionsErrorCode.INTERNAL_ERROR, "boom")
        assert body["error"]["type"] == "api_error"


class TestCompletionsErrorCodeEnum:
    def test_codes_compare_equal_to_their_string_form(self) -> None:
        assert CompletionsErrorCode.MODEL_NOT_FOUND == "model_not_found"
        assert CompletionsErrorCode.INVALID_API_KEY == "invalid_api_key"

    def test_every_code_has_a_type_mapping(self) -> None:
        for code in CompletionsErrorCode:
            assert code in COMPLETIONS_ERROR_TYPES, code


class TestInvalidCodeRejected:
    def test_bare_string_not_in_enum_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            completions_error_body("zzz_unknown", "wat")  # type: ignore[arg-type]


class TestClassifyProviderError:
    def test_model_not_found_error_maps_to_404(self) -> None:
        result = classify_provider_error(ModelNotFoundError("foo"))
        assert result is not None
        assert result.http_status == 404
        assert result.code == CompletionsErrorCode.MODEL_NOT_FOUND
        assert "foo" in result.message

    def test_no_tool_support_error_maps_to_400(self) -> None:
        result = classify_provider_error(ModelDoesNotSupportToolsError("foo"))
        assert result is not None
        assert result.http_status == 400
        assert result.code == CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS

    def test_context_overflow_maps_to_400(self) -> None:
        result = classify_provider_error(
            ProviderError("too long", kind=ProviderErrorKind.CONTEXT_OVERFLOW)
        )
        assert result is not None
        assert result.http_status == 400
        assert result.code == CompletionsErrorCode.CONTEXT_LENGTH_EXCEEDED
        assert result.message == "too long"

    def test_provider_not_found_maps_to_404(self) -> None:
        result = classify_provider_error(ProviderError("missing", kind=ProviderErrorKind.NOT_FOUND))
        assert result is not None
        assert result.http_status == 404
        assert result.code == CompletionsErrorCode.MODEL_NOT_FOUND

    @pytest.mark.parametrize(
        ("kind", "status", "code"),
        [
            (ProviderErrorKind.BAD_REQUEST, 400, CompletionsErrorCode.INVALID_REQUEST),
            (ProviderErrorKind.AUTH, 401, CompletionsErrorCode.INVALID_API_KEY),
            (ProviderErrorKind.RATE_LIMIT, 429, CompletionsErrorCode.RATE_LIMIT_EXCEEDED),
            (ProviderErrorKind.CONNECTION, 503, CompletionsErrorCode.INTERNAL_ERROR),
            (ProviderErrorKind.SERVER, 502, CompletionsErrorCode.INTERNAL_ERROR),
        ],
    )
    def test_classified_provider_kinds_keep_status_code_and_message(
        self, kind, status, code
    ) -> None:
        result = classify_provider_error(ProviderError("the original message", kind=kind))
        assert result is not None
        assert result.http_status == status
        assert result.code == code
        assert result.message == "the original message"

    def test_unknown_provider_kind_returns_none(self) -> None:
        assert classify_provider_error(ProviderError("x", kind=ProviderErrorKind.UNKNOWN)) is None

    def test_unrelated_exception_returns_none(self) -> None:
        assert classify_provider_error(ValueError("nope")) is None
