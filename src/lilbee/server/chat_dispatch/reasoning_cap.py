"""Reasoning cap for the canonical chat surfaces.

``cfg.max_reasoning_chars`` promises a maximum on a reasoning model's thinking,
"before lilbee forces the model to answer". :mod:`lilbee.retrieval.reasoning`
owns the counting and the nudge; this module applies them to the chat surfaces
(``/v1/chat/completions`` and ``/v1/messages``), which speak canonical stream
events rather than the ``StreamToken`` split the RAG handler consumes.

The wrapper works in lilbee's inline ``<think>`` protocol instead of splitting
reasoning out itself: on cap-fire it writes the cap notice into the still-open
thinking block and closes it. Every downstream translator already parses that
protocol, so all three reasoning modes present the capped turn correctly with
no per-surface handling.
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
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageStart,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from lilbee.server.chat_dispatch.dispatch import dispatch_chat, dispatch_chat_stream

CHARS_PER_TOKEN = 4
"""Chars-per-token used to read a token budget as a character cap.

The cap counts characters and the Anthropic ``budget_tokens`` parameter counts
tokens, so one of the two has to be approximated. Four is the usual English
ratio; the cap is a runaway-loop guard, not an accounting boundary.
"""


def budget_capped_chars(cap_chars: int, budget_tokens: int | None) -> int:
    """Tighten *cap_chars* with a per-request token budget.

    ``0`` means unlimited on both sides. A budget may only tighten: a caller
    asking for more thinking than the operator configured still gets the
    configured cap.
    """
    if budget_tokens is None or budget_tokens < 0:
        return cap_chars
    budget_chars = budget_tokens * CHARS_PER_TOKEN
    if cap_chars <= 0:
        return budget_chars
    return min(cap_chars, budget_chars)


def nudged_request(req: CanonicalChatRequest) -> CanonicalChatRequest:
    """Append the cap-continuation user prompt to *req*'s messages.

    Everything else on the request is kept, so a capped turn can still call
    tools and keeps the caller's sampling settings.
    """
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
    """Best-effort close so a capped stream stops costing generation.

    Only an async generator has ``aclose``; a plain async iterator is left to
    its own cleanup rather than being closed by force.
    """
    if isinstance(stream, AsyncGenerator):
        with contextlib.suppress(Exception):
            await stream.aclose()


def _reindexed(event: CanonicalStreamEvent, offset: int) -> CanonicalStreamEvent:
    """Shift a continuation event's block index past the first stream's blocks.

    Without the shift the continuation reopens index 0 and a surface translator
    merges it into the block the cap just closed.
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

    Events pass through unchanged until the reasoning exceeds the cap. Then the
    upstream stream is closed, the cap notice closes the thinking block, and the
    continuation stream is spliced in. The continuation is not capped again,
    because re-capping would cut the forced answer off.

    ``cap_chars <= 0`` disables the cap and forwards the stream verbatim.
    """
    parser = TagParser(show=True)
    cap_fired = False
    max_index = -1
    open_index = 0
    try:
        async for event in stream:
            if isinstance(event, ContentBlockStart | ContentBlockDelta | ContentBlockStop):
                max_index = max(max_index, event.index)
            if isinstance(event, ContentBlockDelta) and isinstance(event.delta, TextDelta):
                open_index = event.index
                parser.feed(event.delta.text)
            yield event
            # in_thinking gates the cap: reasoning that already closed produced
            # its answer, and cutting the turn off then would discard it.
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
        yield _reindexed(event, offset)


def cap_aware_chat(
    req: CanonicalChatRequest, *, canonical_model: str, cap_chars: int
) -> CanonicalResponse:
    """Run a non-streaming chat call, forcing an answer when reasoning ate the turn.

    The provider returns one finished result here, so the cap cannot stop the
    first call's reasoning the way the streaming path does. It applies the other
    half of the promise: a turn that blew past the cap and produced no answer is
    re-issued with the stop-thinking nudge instead of returning reasoning alone.
    A turn that reasoned long and *did* answer is kept as it is -- re-issuing
    would throw away a good answer and pay for a second call.
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

    The reasoning is truncated to the cap so the response honors the maximum the
    setting names, and both calls' token counts are summed because the caller
    paid for both.
    """
    kept = f"{THINK_OPEN_TAG}{reasoning[:cap_chars]}{_cap_notice(cap_chars)}{THINK_CLOSE_TAG}"
    tail = "".join(b.text for b in continuation.content if isinstance(b, TextBlock))
    content: list = [TextBlock(text=f"{kept}{tail}")]
    content.extend(b for b in continuation.content if isinstance(b, ToolUseBlock))
    return dataclasses.replace(
        continuation,
        content=content,
        usage=CanonicalUsage(
            input_tokens=capped.usage.input_tokens + continuation.usage.input_tokens,
            output_tokens=capped.usage.output_tokens + continuation.usage.output_tokens,
        ),
    )
