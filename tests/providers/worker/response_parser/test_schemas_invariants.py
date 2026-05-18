"""Schema-shape invariants the streaming layer relies on."""

from __future__ import annotations

import re
from typing import Any

import pytest

from lilbee.providers.worker.response_parser import SCHEMAS
from lilbee.providers.worker.response_parser.streaming import _MARKER_OPENERS

# Markers in shipped schemas look like ``<tool_call>``, ``</tool_call>``,
# ``<|tool_call>``, ``<tool_call|>``, ``<think>``, ``</think>``, ``[TOOL_CALLS]``.
# This pattern matches that literal shape inside any regex string.
_MARKER_PATTERN = re.compile(r"(?:</?\\?\|?[A-Za-z_/!]+\\?\|?>|\[[A-Z_]+\])")


def _walk_regex_strings(node: Any):
    """Yield every regex string value from a response-schema tree."""
    if isinstance(node, dict):
        for key in ("x-regex", "x-regex-iterator", "x-regex-key-value"):
            value = node.get(key)
            if isinstance(value, str):
                yield value
        subs = node.get("x-regex-substitutions") or []
        if isinstance(subs, list):
            for pair in subs:
                if isinstance(pair, list | tuple) and pair and isinstance(pair[0], str):
                    yield pair[0]
        for child in node.values():
            yield from _walk_regex_strings(child)
    elif isinstance(node, list):
        for child in node:
            yield from _walk_regex_strings(child)


@pytest.mark.parametrize("family,schema", list(SCHEMAS.items()))
def test_every_schema_marker_opens_with_known_character(family, schema) -> None:
    """Every literal marker in shipped schemas starts with ``<`` or ``[``.

    The streaming parser's safety margin (``_MARKER_OPENERS``) holds emission
    when the trailing window contains any of those characters. A future
    schema whose markers begin differently would silently break streaming
    correctness; this test forces the schema author to update
    ``_MARKER_OPENERS`` deliberately rather than discover it via a stream
    that leaks partial bytes.
    """
    for pattern in _walk_regex_strings(schema):
        for marker in _MARKER_PATTERN.findall(pattern):
            assert marker[0] in _MARKER_OPENERS, (
                f"{family.value}: marker {marker!r} in schema does not open with "
                f"a character in _MARKER_OPENERS={_MARKER_OPENERS}. Update the "
                "streaming module's _MARKER_OPENERS constant before adding a "
                "schema with a new opener character."
            )
