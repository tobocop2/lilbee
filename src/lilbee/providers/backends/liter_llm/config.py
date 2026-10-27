"""Map lilbee requests to liter-llm request fields."""

from __future__ import annotations

import base64
import json
from typing import Any

from lilbee.providers.backends.liter_llm.client import _require_sdk
from lilbee.providers.local_servers import detect_local_server, local_server_for_key
from lilbee.providers.model_ref import ProviderModelRef
from lilbee.providers.sdk_backend import CompletionRequest


def _route_model(ref: ProviderModelRef, api_base: str | None) -> str:
    """Format *ref* for liter-llm using the ``provider/model`` convention.

    API and local-server refs already carry their canonical prefix. A bare
    ``local`` ref forced through the SDK (``llm_provider=remote``) gets the
    prefix of whichever local server its ``api_base`` points at.
    """
    if ref.is_api or local_server_for_key(ref.provider) is not None:
        return ref.for_openai_prefix()
    if api_base and (spec := detect_local_server(api_base)) is not None:
        return spec.qualify(ref.name)
    return ref.name


def _wire_base_url(api_base: str | None) -> str | None:
    """Return the ``/v1`` base URL for a local server, or ``None``.

    Both routes accept the ``/v1`` form. liter-llm posts to
    ``{base_url}/chat/completions`` for an OpenAI-compatible server such as
    LM Studio, which serves that route under ``/v1``. For Ollama, liter-llm
    strips one trailing ``/v1`` and appends ``/api/chat``. A base that
    already ends in ``/v1`` is kept as it is.
    """
    if not api_base:
        return None
    trimmed = api_base.rstrip("/").removesuffix("/v1")
    return f"{trimmed}/v1"


EMPTY_MESSAGE_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


def _is_blank_turn(msg: dict[str, Any]) -> bool:
    """Whether *msg* is a user or assistant turn with blank text and no tool calls."""
    content = msg.get("content")
    return (
        msg["role"] in ("user", "assistant")
        and isinstance(content, str)
        and not content.strip()
        and not msg.get("tool_calls")
    )


def _format_messages(
    messages: list[dict[str, Any]], *, fill_blank_turns: bool = False
) -> list[dict[str, Any]]:
    """Convert messages with inline image bytes into OpenAI content parts.

    Messages without ``images`` pass through untouched. With *fill_blank_turns*,
    a blank user or assistant turn carries a placeholder, as Anthropic rejects blank text.
    """
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        if fill_blank_turns and "images" not in msg and _is_blank_turn(msg):
            formatted.append({**msg, "content": EMPTY_MESSAGE_PLACEHOLDER})
        elif "images" in msg:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": msg.get("content", "")}]
            for img in msg["images"]:
                if isinstance(img, bytes):
                    b64 = base64.b64encode(img).decode()
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
            formatted.append({"role": msg["role"], "content": content_parts})
        else:
            formatted.append(msg)
    return formatted


# lilbee option -> chat request field, for the options liter-llm takes as typed fields.
_REQUEST_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "response_format",
        "tool_choice",
    }
)


def _chat_fields(request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
    """Translate a ``CompletionRequest`` into liter-llm chat request fields."""
    liter_llm = _require_sdk()
    fields: dict[str, Any] = {
        "model": _route_model(request.ref, request.api_base),
        "messages": _format_messages(
            request.messages, fill_blank_turns=request.ref.provider == "anthropic"
        ),
        "stream": stream,
    }
    extra_body: dict[str, Any] = {}
    for key, value in request.options.items():
        if key == "num_predict":
            fields["max_tokens"] = value
        elif key == "tools":
            fields["tools"] = [liter_llm.ChatCompletionTool.from_json(json.dumps(t)) for t in value]
        elif key in _REQUEST_FIELDS:
            fields[key] = value
        elif key == "reasoning_effort":
            fields[key] = liter_llm.ReasoningEffort(value)
        else:
            extra_body[key] = value
    if extra_body:
        fields["extra_body"] = extra_body
    return fields
