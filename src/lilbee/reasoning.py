"""Reasoning token filter — detects <think>...</think> tags in streaming output.

Reasoning models (Qwen3, DeepSeek-R1) wrap their thinking process in
``<think>...</think>`` tags. This module provides a stateful filter that
classifies tokens as reasoning or response content.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


@dataclass
class StreamToken:
    """A classified token from the stream."""

    content: str
    is_reasoning: bool


def filter_reasoning(tokens: Iterator[str], *, show: bool) -> Iterator[StreamToken]:
    """Filter ``<think>...</think>`` tags from a token stream.

    When *show* is True, yields thinking content as ``StreamToken(is_reasoning=True)``.
    When *show* is False, strips thinking content entirely.
    Tokens outside thinking blocks are always yielded as ``is_reasoning=False``.
    """
    buf = ""
    in_thinking = False

    for token in tokens:
        buf += token

        while buf:
            if in_thinking:
                close_idx = buf.find(_CLOSE_TAG)
                if close_idx == -1:
                    # Might be a partial close tag at the end
                    if _could_be_partial(_CLOSE_TAG, buf):
                        break  # wait for more tokens
                    # All buffered content is thinking
                    if show:
                        yield StreamToken(content=buf, is_reasoning=True)
                    buf = ""
                else:
                    # Emit thinking content before the close tag
                    thinking_content = buf[:close_idx]
                    if thinking_content and show:
                        yield StreamToken(content=thinking_content, is_reasoning=True)
                    buf = buf[close_idx + len(_CLOSE_TAG) :]
                    in_thinking = False
            else:
                open_idx = buf.find(_OPEN_TAG)
                if open_idx == -1:
                    # Might be a partial open tag at the end
                    if _could_be_partial(_OPEN_TAG, buf):
                        break  # wait for more tokens
                    # All buffered content is normal response
                    yield StreamToken(content=buf, is_reasoning=False)
                    buf = ""
                else:
                    # Emit response content before the open tag
                    before = buf[:open_idx]
                    if before:
                        yield StreamToken(content=before, is_reasoning=False)
                    buf = buf[open_idx + len(_OPEN_TAG) :]
                    in_thinking = True

    # Flush remaining buffer
    if buf:
        if in_thinking:
            if show:
                yield StreamToken(content=buf, is_reasoning=True)
        else:
            yield StreamToken(content=buf, is_reasoning=False)


def _could_be_partial(tag: str, buf: str) -> bool:
    """Check if the end of buf could be the start of the given tag."""
    return any(buf.endswith(tag[:length]) for length in range(1, len(tag)))
