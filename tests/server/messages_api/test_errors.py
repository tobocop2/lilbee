"""Tests for the Anthropic-shape error envelope helper."""

from __future__ import annotations

import pytest

from lilbee.server.messages_api.errors import (
    MessagesErrorType,
    messages_error_body,
    status_for_error_type,
)


class TestErrorBodyShape:
    def test_not_found_error(self) -> None:
        body = messages_error_body(MessagesErrorType.NOT_FOUND, "model not found")
        assert body == {
            "type": "error",
            "error": {"type": "not_found_error", "message": "model not found"},
        }

    def test_invalid_request_error(self) -> None:
        body = messages_error_body(MessagesErrorType.INVALID_REQUEST, "no tools")
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == "no tools"
        assert body["type"] == "error"

    def test_authentication_error(self) -> None:
        body = messages_error_body(MessagesErrorType.AUTHENTICATION, "bad token")
        assert body["error"]["type"] == "authentication_error"

    def test_permission_error(self) -> None:
        body = messages_error_body(MessagesErrorType.PERMISSION, "nope")
        assert body["error"]["type"] == "permission_error"

    def test_rate_limit_error(self) -> None:
        body = messages_error_body(MessagesErrorType.RATE_LIMIT, "slow down")
        assert body["error"]["type"] == "rate_limit_error"

    def test_overloaded_error(self) -> None:
        body = messages_error_body(MessagesErrorType.OVERLOADED, "busy")
        assert body["error"]["type"] == "overloaded_error"

    def test_api_error(self) -> None:
        body = messages_error_body(MessagesErrorType.API, "boom")
        assert body["error"]["type"] == "api_error"


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            (MessagesErrorType.INVALID_REQUEST, 400),
            (MessagesErrorType.AUTHENTICATION, 401),
            (MessagesErrorType.PERMISSION, 403),
            (MessagesErrorType.NOT_FOUND, 404),
            (MessagesErrorType.RATE_LIMIT, 429),
            (MessagesErrorType.API, 500),
            (MessagesErrorType.OVERLOADED, 529),
        ],
    )
    def test_each_type_maps_to_anthropic_status(
        self, error_type: MessagesErrorType, expected: int
    ) -> None:
        assert status_for_error_type(error_type) == expected


class TestEnumValues:
    def test_enum_values_match_anthropic_strings(self) -> None:
        assert MessagesErrorType.INVALID_REQUEST == "invalid_request_error"
        assert MessagesErrorType.AUTHENTICATION == "authentication_error"
        assert MessagesErrorType.PERMISSION == "permission_error"
        assert MessagesErrorType.NOT_FOUND == "not_found_error"
        assert MessagesErrorType.RATE_LIMIT == "rate_limit_error"
        assert MessagesErrorType.API == "api_error"
        assert MessagesErrorType.OVERLOADED == "overloaded_error"
