"""Direct tests for the shared tool-arguments parser."""

from lilbee.server.chat_dispatch.tool_args import parse_tool_arguments


def test_empty_and_whitespace_only_both_parse_to_an_empty_dict():
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments("  \n\t") == {}


def test_malformed_json_falls_back_to_raw():
    assert parse_tool_arguments("not json{") == {"_raw": "not json{"}


def test_non_object_json_falls_back_to_raw():
    assert parse_tool_arguments("[1, 2]") == {"_raw": "[1, 2]"}


def test_object_json_parses():
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
