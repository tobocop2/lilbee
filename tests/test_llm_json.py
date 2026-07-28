"""The shared first-JSON-value scan every LLM reply parser depends on.

Three call sites previously carried their own copy or a greedy brace/bracket
span; these cases are the ones those copies got wrong.
"""

from __future__ import annotations

import pytest

from lilbee.core.llm_json import first_json_array, first_json_object


class TestFirstJsonObject:
    @pytest.mark.parametrize(
        "text",
        ['{"a": }', "{unclosed", "no json here"],
        ids=["malformed", "unbalanced", "absent"],
    )
    def test_unparseable_is_none(self, text):
        assert first_json_object(text) is None

    def test_braces_inside_string_values_do_not_derail_the_scan(self):
        """A regex pattern or entity text containing a brace is legal JSON;
        a naive brace counter truncates mid-string and loses the object."""
        text = '{"types": [{"name": "x", "pattern": "\\\\}"}]}'
        assert first_json_object(text) == {"types": [{"name": "x", "pattern": "\\}"}]}
        assert first_json_object('{"0": [{"type": "note", "text": "a } b"}]}') == {
            "0": [{"type": "note", "text": "a } b"}]
        }

    def test_object_after_stray_brace_in_prose_is_found(self):
        assert first_json_object('opening { thoughts... {"kind": "x"}') == {"kind": "x"}

    def test_trailing_prose_containing_a_brace_is_ignored(self):
        """A greedy span runs to the LAST brace in the reply and drops the value."""
        assert first_json_object('{"kind": "x"} (note: the } is fine)') == {"kind": "x"}

    def test_first_of_several_wins(self):
        assert first_json_object('{"a": 1} and {"b": 2}') == {"a": 1}

    def test_an_object_nested_in_an_array_is_still_found(self):
        """The scan starts at the first opener of the requested kind, wherever
        it sits, so a reply that wraps its object in a list still parses."""
        assert first_json_object('[{"a": 1}]') == {"a": 1}


class TestFirstJsonArray:
    def test_trailing_prose_containing_a_bracket_is_ignored(self):
        assert first_json_array('[{"a": 1}] see footnote [2]') == [{"a": 1}]

    def test_brackets_inside_string_values_do_not_derail_the_scan(self):
        assert first_json_array('[{"text": "a ] b"}]') == [{"text": "a ] b"}]

    @pytest.mark.parametrize("text", ["[unclosed", "no json here"], ids=["unbalanced", "absent"])
    def test_unparseable_is_none(self, text):
        assert first_json_array(text) is None

    def test_an_object_is_not_an_array(self):
        assert first_json_array('{"a": 1}') is None
