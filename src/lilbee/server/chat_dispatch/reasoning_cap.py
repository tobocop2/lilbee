"""Applies ``cfg.max_reasoning_chars`` to the canonical chat surfaces.

Counting and the stop-thinking nudge live in :mod:`lilbee.retrieval.reasoning`.
The cap notice is written into the still-open ``<think>`` block rather than
handled per surface, so every downstream translator presents a capped turn.
Under ``inline`` that block is the answer text, so the notice reads there.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import AsyncGenerator, AsyncIterator

from lilbee.providers.base import THINK_CLOSE_TAG, THINK_OPEN_TAG
from lilbee.retrieval.reasoning import (
    CAP_CONTINUATION_PROMPT,
    CAP_NOTICE_TEMPLATE,
    TagParser,
    split_reasoning,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalUsage,
    ContentBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from lilbee.server.chat_dispatch.dispatch import dispatch_chat, dispatch_chat_stream

CHARS_PER_TOKEN = 4
"""Approximate chars per token, for reading ``budget_tokens`` as a char cap."""


def budget_capped_chars(cap_chars: int, budget_tokens: int | None) -> int:
    """Tighten *cap_chars* with a per-request token budget; never loosen it.

    ``0`` means unlimited on both sides. A budget of zero or less is no budget,
    not a request for unlimited thinking.
    """
    if budget_tokens is None or budget_tokens <= 0:
        return cap_chars
    budget_chars = budget_tokens * CHARS_PER_TOKEN
    if cap_chars <= 0:
        return budget_chars
    return min(cap_chars, budget_chars)


def nudged_request(req: CanonicalChatRequest) -> CanonicalChatRequest:
    """Append the cap-continuation user prompt; every other request field is kept."""
    return dataclasses.replace(
        req,
        messages=[
            *req.messages,
            CanonicalMessage.from_string(role="user", text=CAP_CONTINUATION_PROMPT),
        ],
    )


def _cap_notice(cap_chars: int) -> str:
    """The notice written into the thinking block when the cap fires."""
    return CAP_NOTICE_TEMPLATE.format(chars=cap_chars)


async def _aclose(stream: AsyncIterator[CanonicalStreamEvent]) -> None:
    """Best-effort close; only an async generator has ``aclose``."""
    if isinstance(stream, AsyncGenerator):
        with contextlib.suppress(Exception):
            await stream.aclose()


def _reindexed(event: CanonicalStreamEvent, offset: int) -> CanonicalStreamEvent:
    """Shift a continuation event's block index past the first stream's blocks.

    Without it the continuation reopens index 0 and translators merge it into
    the block the cap just closed.
    """
    if isinstance(event, ContentBlockStart | ContentBlockDelta | ContentBlockStop):
        return dataclasses.replace(event, index=event.index + offset)
    return event


async def cap_aware_chat_stream(
    stream: AsyncIterator[CanonicalStreamEvent],
    req: CanonicalChatRequest,
    *,
    canonical_model: str,
    cap_chars: int,
) -> AsyncIterator[CanonicalStreamEvent]:
    """Forward *stream*, stopping the reasoning at *cap_chars* and forcing an answer.

    On cap-fire: close upstream, write the notice into the open thinking block,
    splice in the continuation. The continuation is not capped again.
    ``cap_chars <= 0`` forwards the stream verbatim.
    """
    parser = TagParser(show=True)
    cap_fired = False
    max_index = -1
    open_index = 0
    spent = None
    try:
        async for event in stream:
            if isinstance(event, MessageDelta) and event.usage is not None:
                spent = event.usage
            if isinstance(event, ContentBlockStart | ContentBlockDelta | ContentBlockStop):
                max_index = max(max_index, event.index)
            if isinstance(event, ContentBlockDelta) and isinstance(event.delta, TextDelta):
                open_index = event.index
                parser.feed(event.delta.text)
            yield event
            # Only cap while still inside <think>; a closed block already answered.
            if cap_chars > 0 and parser.in_thinking and parser.reasoning_chars > cap_chars:
                cap_fired = True
                break
    finally:
        if cap_fired:
            await _aclose(stream)

    if not cap_fired:
        return

    yield ContentBlockDelta(
        index=open_index,
        delta=TextDelta(text=f"{_cap_notice(cap_chars)}{THINK_CLOSE_TAG}"),
    )
    offset = max_index + 1
    async for event in dispatch_chat_stream(nudged_request(req), canonical_model=canonical_model):
        # The message already started; a second prelude would restart it.
        if isinstance(event, MessageStart):
            continue
        yield _reindexed(_with_spent(event, spent), offset)


def _with_spent(event: CanonicalStreamEvent, spent: CanonicalUsage | None) -> CanonicalStreamEvent:
    """Add the capped call's tokens to the continuation's usage.

    The caller paid for both generations, so reporting only the continuation's
    hides the reasoning the cap just stopped. A provider that reports usage
    once at the end has none to add: the cap closes the first stream before
    that event, and those tokens go unreported.
    """
    if spent is None or not isinstance(event, MessageDelta) or event.usage is None:
        return event
    return dataclasses.replace(
        event,
        usage=CanonicalUsage(
            input_tokens=event.usage.input_tokens + spent.input_tokens,
            output_tokens=event.usage.output_tokens + spent.output_tokens,
        ),
    )


def cap_aware_chat(
    req: CanonicalChatRequest, *, canonical_model: str, cap_chars: int
) -> CanonicalResponse:
    """Run a non-streaming chat call, re-issuing a turn that only reasoned.

    One finished result arrives, so the first call's reasoning cannot be stopped
    mid-flight. A turn over the cap with no answer and no tool call is re-issued
    with the nudge; a turn that answered is kept.
    """
    resp = dispatch_chat(req, canonical_model=canonical_model)
    if cap_chars <= 0:
        return resp
    text = "".join(b.text for b in resp.content if isinstance(b, TextBlock))
    reasoning, answer = split_reasoning(text)
    tool_uses = [b for b in resp.content if isinstance(b, ToolUseBlock)]
    if len(reasoning) <= cap_chars or answer.strip() or tool_uses:
        return resp

    continuation = dispatch_chat(nudged_request(req), canonical_model=canonical_model)
    return _merged_response(resp, continuation, cap_chars=cap_chars, reasoning=reasoning)


def _merged_response(
    capped: CanonicalResponse,
    continuation: CanonicalResponse,
    *,
    cap_chars: int,
    reasoning: str,
) -> CanonicalResponse:
    """Fold a capped turn and its continuation into one response.

    Reasoning is truncated to the cap; both calls' token counts are summed.
    """
    kept = f"{THINK_OPEN_TAG}{reasoning[:cap_chars]}{_cap_notice(cap_chars)}{THINK_CLOSE_TAG}"
    tail = "".join(b.text for b in continuation.content if isinstance(b, TextBlock))
    content: list[ContentBlock] = [TextBlock(text=f"{kept}{tail}")]
    content.extend(b for b in continuation.content if isinstance(b, ToolUseBlock))
    return dataclasses.replace(
        continuation,
        content=content,
        usage=CanonicalUsage(
            input_tokens=capped.usage.input_tokens + continuation.usage.input_tokens,
            output_tokens=capped.usage.output_tokens + continuation.usage.output_tokens,
        ),
    )
