"""Schema-shape invariants the streaming layer relies on."""

from __future__ import annotations

import re
from typing import Any

import pytest

from lilbee.providers.worker.response_parser import get_schemas
from lilbee.providers.worker.response_parser.families import TemplateFamily
from lilbee.providers.worker.response_parser.streaming import _NATIVE_OPENERS as _MARKER_OPENERS


def test_schemas_registry_covers_every_known_family() -> None:
    """Every ``TemplateFamily`` except ``UNKNOWN`` has a shipped schema.

    Adding a new variant without a schema would silently disable tool
    extraction for that family; failing this invariant forces the schema
    author to either ship the schema or document the omission deliberately.
    """
    expected = {f for f in TemplateFamily if f is not TemplateFamily.UNKNOWN}
    assert set(get_schemas().keys()) == expected, (
        f"get_schemas() missing entries for {expected - set(get_schemas().keys())} or has "
        f"extras for {set(get_schemas().keys()) - expected}."
    )


def test_load_returns_none_for_missing_file(caplog) -> None:
    """A missing schema file degrades to ``None`` instead of crashing import."""
    import logging

    from lilbee.providers.worker.response_parser.schemas import _load

    with caplog.at_level(logging.WARNING, logger="lilbee.providers.worker.response_parser.schemas"):
        result = _load("does_not_exist.json")
    assert result is None
    assert any("does_not_exist" in r.message for r in caplog.records)


def test_load_returns_none_for_invalid_json(tmp_path, monkeypatch, caplog) -> None:
    """A schema file with malformed JSON degrades to ``None`` with a warning."""
    import logging

    from lilbee.providers.worker.response_parser import schemas as schemas_mod

    bad_dir = tmp_path / "schemas"
    bad_dir.mkdir()
    (bad_dir / "broken.json").write_text("not json", encoding="utf-8")

    class _FakeFiles:
        def joinpath(self, *parts):
            return bad_dir / parts[-1]

    monkeypatch.setattr(schemas_mod.resources, "files", lambda _: _FakeFiles())
    with caplog.at_level(logging.WARNING, logger="lilbee.providers.worker.response_parser.schemas"):
        result = schemas_mod._load("broken.json")
    assert result is None
    assert any("broken.json" in r.message for r in caplog.records)


# Markers in shipped schemas look like ``<tool_call>``, ``</tool_call>``,
# ``<|tool_call>``, ``<tool_call|>``, ``<think>``, ``</think>``, ``[TOOL_CALLS]``,
# and the bareword openers ``functools`` (Phi-4) and ``>>>`` (Functionary v3).
# This pattern matches every shape so the safety-opener invariant catches them.
_MARKER_PATTERN = re.compile(
    r"(?:"
    r"</?\\?\|?[A-Za-z_/!]+\\?\|?>"  # angle-wrapped tokens
    r"|\[[A-Z_]+\]"  # bracketed uppercase tokens
    r"|\bfunctools\b"  # Phi-4 marker opens with 'f'
    r"|>>>"  # Functionary v3 marker opens with '>'
    r")"
)


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


@pytest.mark.parametrize("family,schema", list(get_schemas().items()))
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
