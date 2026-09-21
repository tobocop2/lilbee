"""Both /v1 translators report the same prompt-cache reuse from one canonical count.

The serialized bodies and SSE frames are pinned by the route tests for each
surface; this file compares the two translators against one fixture.
"""

from __future__ import annotations

import pytest

from lilbee.server.anthropic_api.translate import (
    canonical_stream_to_anthropic_events,
    canonical_to_messages_response,
)
from lilbee.server.chat_completions_api.translate import (
    canonical_stream_to_completions_chunks,
    canonical_to_completions_response,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalUsage,
    MessageDelta,
    MessageStart,
    MessageStop,
    StopReason,
    TextBlock,
)

_MODEL = "m"
_RESPONSE_ID = "msg_parity"
# Measured against llama-server: a second turn over the same prefix reported
# prompt_tokens 423 with prompt_tokens_details.cached_tokens 404.
_PROMPT_TOKENS = 423
_CACHED_TOKENS = 404


def _response(cached: int) -> CanonicalResponse:
    return CanonicalResponse(
        id="x",
        model=_MODEL,
        content=[TextBlock(text="hi")],
        stop_reason=StopReason.END_TURN,
        usage=CanonicalUsage(
            input_tokens=_PROMPT_TOKENS, output_tokens=8, cached_input_tokens=cached
        ),
    )


def _events(cached: int) -> list[CanonicalStreamEvent]:
    return [
        MessageStart(id=_RESPONSE_ID, model=_MODEL),
        MessageDelta(stop_reason=StopReason.END_TURN, usage=_response(cached).usage),
        MessageStop(),
    ]


async def _anthropic_stream_usage(cached: int) -> dict[str, int]:
    async def _aiter():
        for event in _events(cached):
            yield event

    pairs = [
        pair
        async for pair in canonical_stream_to_anthropic_events(
            _aiter(), model=_MODEL, response_id=_RESPONSE_ID
        )
    ]
    return next(payload["usage"] for kind, payload in pairs if kind == "message_delta")


async def _completions_stream_usage(cached: int):
    async def _aiter():
        for event in _events(cached):
            yield event

    chunks = [
        chunk
        async for chunk in canonical_stream_to_completions_chunks(
            _aiter(), model=_MODEL, response_id=_RESPONSE_ID, include_usage=True
        )
    ]
    usage = chunks[-1].usage
    assert usage is not None
    return usage


def test_both_wires_report_the_same_reuse_count() -> None:
    anthropic = canonical_to_messages_response(_response(_CACHED_TOKENS), response_id=_RESPONSE_ID)
    openai = canonical_to_completions_response(_response(_CACHED_TOKENS), response_id=_RESPONSE_ID)
    assert (
        anthropic.usage.cache_read_input_tokens == openai.usage.prompt_tokens_details.cached_tokens
    )
    assert anthropic.usage.cache_read_input_tokens == _CACHED_TOKENS


def test_each_wire_keeps_its_own_prompt_arithmetic() -> None:
    """Both wires still let a client recover the whole prompt, by different sums."""
    anthropic_cold = canonical_to_messages_response(_response(0), response_id=_RESPONSE_ID)
    anthropic_warm = canonical_to_messages_response(
        _response(_CACHED_TOKENS), response_id=_RESPONSE_ID
    )
    openai_cold = canonical_to_completions_response(_response(0), response_id=_RESPONSE_ID)
    openai_warm = canonical_to_completions_response(
        _response(_CACHED_TOKENS), response_id=_RESPONSE_ID
    )
    # Anthropic: the prompt-side counts are disjoint and sum to the prompt.
    assert (
        anthropic_warm.usage.input_tokens
        + anthropic_warm.usage.cache_creation_input_tokens
        + anthropic_warm.usage.cache_read_input_tokens
        == anthropic_cold.usage.input_tokens
    )
    # OpenAI: the cached count is a subset, so prompt_tokens does not move.
    assert openai_warm.usage.prompt_tokens == openai_cold.usage.prompt_tokens
    assert openai_warm.usage.total_tokens == openai_cold.usage.total_tokens


def test_no_reuse_reports_zero_on_both_wires() -> None:
    anthropic = canonical_to_messages_response(_response(0), response_id=_RESPONSE_ID)
    openai = canonical_to_completions_response(_response(0), response_id=_RESPONSE_ID)
    assert anthropic.usage.cache_read_input_tokens == 0
    assert openai.usage.prompt_tokens_details.cached_tokens == 0


@pytest.mark.asyncio
async def test_streaming_agrees_with_non_streaming_on_both_wires() -> None:
    anthropic_stream = await _anthropic_stream_usage(_CACHED_TOKENS)
    openai_stream = await _completions_stream_usage(_CACHED_TOKENS)
    anthropic_body = canonical_to_messages_response(
        _response(_CACHED_TOKENS), response_id=_RESPONSE_ID
    )
    openai_body = canonical_to_completions_response(
        _response(_CACHED_TOKENS), response_id=_RESPONSE_ID
    )
    assert (
        anthropic_stream["cache_read_input_tokens"] == anthropic_body.usage.cache_read_input_tokens
    )
    assert (
        openai_stream.prompt_tokens_details.cached_tokens
        == openai_body.usage.prompt_tokens_details.cached_tokens
    )
    assert anthropic_stream["cache_read_input_tokens"] == (
        openai_stream.prompt_tokens_details.cached_tokens
    )
