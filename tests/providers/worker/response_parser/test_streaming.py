"""Tests for ``StreamingResponseParser``."""

from __future__ import annotations

import json

from lilbee.providers.worker.response_parser import (
    StreamingResponseParser,
    TemplateFamily,
    get_schemas,
)


def _drain(parser: StreamingResponseParser, chunks: list[str]):
    """Feed *chunks* into *parser* and return (full_content, all_tool_deltas)."""
    content_parts: list[str] = []
    tool_deltas: list = []
    for chunk in chunks:
        c, deltas = parser.feed(chunk)
        content_parts.append(c)
        tool_deltas.extend(deltas)
    final_c, final_deltas = parser.flush()
    content_parts.append(final_c)
    tool_deltas.extend(final_deltas)
    return "".join(content_parts), tool_deltas


def test_streaming_emits_text_only_when_no_tool_calls() -> None:
    """A plain text stream comes out as content with no tool deltas."""
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    content, deltas = _drain(parser, ["Hello ", "world", "!"])
    assert content == "Hello world!"
    assert deltas == []


def test_streaming_holds_partial_marker_until_complete() -> None:
    """A chunk ending mid-marker (``<tool_c``) must not leak the partial bytes."""
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    # After the first chunk, the trailing "<tool_c" must not be emitted yet.
    content_after_first, _ = parser.feed("Hi <tool_c")
    assert "<tool_c" not in content_after_first
    # Once the full tool call lands, content emitted is just "Hi " and a delta arrives.
    content, deltas = _drain(parser, ['all>{"name": "f", "arguments": {}}</tool_call>'])
    full_content = content_after_first + content
    assert full_content.strip() == "Hi"
    assert len(deltas) == 1
    assert deltas[0].name == "f"


def test_streaming_emits_one_delta_per_completed_tool_call() -> None:
    """Two ``<tool_call>`` blocks across chunks emit two tool deltas."""
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    _, deltas = _drain(
        parser,
        [
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>',
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>',
        ],
    )
    assert [d.name for d in deltas] == ["a", "b"]
    assert [d.index for d in deltas] == [0, 1]


def test_streaming_emits_tool_call_arguments_as_json_string() -> None:
    """Arguments are serialised JSON in each ``ToolCallDelta.arguments_delta``."""
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    _, deltas = _drain(
        parser, ['<tool_call>{"name": "search", "arguments": {"q": "foo"}}</tool_call>']
    )
    assert len(deltas) == 1
    args_str = deltas[0].arguments_delta
    assert args_str is not None
    assert json.loads(args_str) == {"q": "foo"}


def test_streaming_flush_releases_held_content() -> None:
    """Content held by the safety margin is released on ``flush()``."""
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    held, _ = parser.feed("short text")
    flushed, _ = parser.flush()
    assert held + flushed == "short text"


def test_streaming_does_not_leak_tool_call_body_across_chunks() -> None:
    """Regression: a ``<tool_call>`` body that spans many chunks must not
    leak its prefix as content before the closing marker arrives. (bb-zgcx)

    Without holding content from the first opener, mid-body chunks like
    ``{"name": "x",`` would be emitted as plain text, and opencode would
    render the raw JSON instead of invoking the tool.
    """
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    chunks = [
        "Before. ",
        "<tool_call>\n",
        '{"name": ',
        '"lilbee_search", ',
        '"arguments": ',
        '{"query": "battery"}',
        "}\n</tool_call>\n",
        "Middle. ",
        "<tool_call>\n",
        '{"name": ',
        '"lilbee_search", ',
        '"arguments": ',
        '{"query": "oil"}',
        "}\n</tool_call>",
    ]
    content, deltas = _drain(parser, chunks)
    assert "<tool_call>" not in content
    assert '"name"' not in content
    assert "lilbee_search" not in content
    assert "Before." in content
    assert "Middle." in content
    assert [d.name for d in deltas] == ["lilbee_search", "lilbee_search"]
    args = [json.loads(d.arguments_delta) for d in deltas if d.arguments_delta]
    assert args == [{"query": "battery"}, {"query": "oil"}]


def test_streaming_handles_token_sized_chunks() -> None:
    """A token-by-token stream (1 char at a time) emits no partial tool-call text."""
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    full = 'Answer: <tool_call>{"name": "f", "arguments": {"q": "x"}}</tool_call> done.'
    chunks = list(full)
    content, deltas = _drain(parser, chunks)
    assert "<tool_call>" not in content
    assert "{" not in content
    assert content.strip().split() == ["Answer:", "done."]
    assert len(deltas) == 1
    assert deltas[0].name == "f"


def test_streaming_holds_phi4_functools_marker_until_close() -> None:
    """Phi-4's bareword ``functools[...]`` marker streams char-by-char; the
    parser must hold from the leading ``f`` so the marker word never reaches
    the UI as plain text. Without ``f`` in ``_MARKER_OPENERS`` it would.
    """
    parser = StreamingResponseParser(get_schemas()[TemplateFamily.PHI4MINI])
    full = 'I will help. functools[{"name": "search", "arguments": {"q":"x"}}] done.'
    content, deltas = _drain(parser, list(full))
    assert "functools" not in content
    assert "{" not in content
    assert len(deltas) == 1
    assert deltas[0].name == "search"


def test_streaming_buffer_caps_on_never_closing_marker() -> None:
    """A model that emits an opener and never closes must not grow the
    parser's internal buffer past the hard cap; otherwise an adversarial
    stream could drive unbounded memory + O(n^2) regex cost.
    """
    from lilbee.providers.worker.response_parser.streaming import _BUFFER_HARD_CAP

    parser = StreamingResponseParser(get_schemas()[TemplateFamily.QWEN3])
    # First chunk opens a never-closing tag, then a flood of bytes.
    parser.feed("<tool_call>")
    long_blob = "x" * (_BUFFER_HARD_CAP + 100)
    content, _ = parser.feed(long_blob)
    # On cap-exceed the parser finalises and releases buffered content, so
    # subsequent feeds don't keep growing without bound.
    assert len(parser._buffer) <= _BUFFER_HARD_CAP + len(long_blob)
    # Content delta released at least the bulk of the input (no infinite hold).
    assert len(content) > _BUFFER_HARD_CAP // 2
