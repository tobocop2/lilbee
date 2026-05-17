"""Tests for the OpenAI-shaped error envelope."""

from __future__ import annotations

import pytest

from lilbee.server.chat_completions_api.errors import (
    COMPLETIONS_ERROR_TYPES,
    CompletionsErrorCode,
    completions_error_body,
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
