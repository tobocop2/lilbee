"""Tests for the auto-extraction parsing and orchestration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.data.store import MemoryKind
from lilbee.retrieval.query.memory_extract import (
    ExtractedMemory,
    build_extract_messages,
    extract_memories,
    parse_extraction,
)


class TestParseExtraction:
    def test_parses_well_formed_array(self):
        raw = '[{"text": "the user prefers rust", "kind": "fact"}, '
        raw += '{"text": "wants terse answers", "kind": "preference"}]'
        result = parse_extraction(raw)
        assert result == [
            ExtractedMemory("the user prefers rust", MemoryKind.FACT),
            ExtractedMemory("wants terse answers", MemoryKind.PREFERENCE),
        ]

    def test_extracts_array_wrapped_in_prose(self):
        raw = 'Here is what I found:\n[{"text": "uses a Crown Victoria", "kind": "fact"}]\nDone.'
        assert parse_extraction(raw) == [ExtractedMemory("uses a Crown Victoria", MemoryKind.FACT)]

    def test_empty_array_yields_nothing(self):
        assert parse_extraction("[]") == []

    def test_no_array_yields_nothing(self):
        assert parse_extraction("I could not find anything to remember.") == []

    def test_malformed_json_yields_nothing(self):
        assert parse_extraction("[{text: broken}]") == []

    def test_non_list_json_yields_nothing(self):
        assert parse_extraction('{"text": "x"}') == []

    def test_skips_items_that_are_not_objects(self):
        assert parse_extraction('["just a string", 42]') == []

    def test_skips_missing_or_nonstring_text(self):
        assert parse_extraction('[{"kind": "fact"}, {"text": 5}]') == []

    def test_filters_too_short_text(self):
        assert parse_extraction('[{"text": "hi", "kind": "fact"}]') == []

    def test_filters_too_long_text(self):
        long_text = "x" * 600
        assert parse_extraction(f'[{{"text": "{long_text}", "kind": "fact"}}]') == []

    def test_unknown_kind_defaults_to_fact(self):
        result = parse_extraction('[{"text": "the user lives in Berlin", "kind": "weird"}]')
        assert result == [ExtractedMemory("the user lives in Berlin", MemoryKind.FACT)]

    def test_missing_kind_defaults_to_fact(self):
        result = parse_extraction('[{"text": "the user lives in Berlin"}]')
        assert result[0].kind is MemoryKind.FACT


class TestBuildMessages:
    def test_system_and_user_roles(self):
        messages = build_extract_messages("q", "a")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "q" in messages[1]["content"]
        assert "a" in messages[1]["content"]


class TestExtractMemories:
    def test_calls_chat_and_parses(self):
        chat = MagicMock(return_value='[{"text": "the user prefers rust", "kind": "fact"}]')
        result = extract_memories("I love rust", "Rust is great.", chat)
        assert result == [ExtractedMemory("the user prefers rust", MemoryKind.FACT)]
        chat.assert_called_once()
        assert chat.call_args.kwargs["stream"] is False

    @pytest.mark.parametrize("question,answer", [("", "a"), ("q", "   ")])
    def test_skips_empty_turn(self, question, answer):
        chat = MagicMock()
        assert extract_memories(question, answer, chat) == []
        chat.assert_not_called()

    def test_chat_failure_yields_nothing(self):
        chat = MagicMock(side_effect=RuntimeError("model down"))
        assert extract_memories("q", "a", chat) == []
