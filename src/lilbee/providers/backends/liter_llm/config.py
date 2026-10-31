"""Map lilbee requests to liter-llm request fields."""

from __future__ import annotations

import base64
import json
from typing import Any

from lilbee.providers.backends.liter_llm.client import _require_sdk
from lilbee.providers.local_servers import OLLAMA, detect_local_server, local_server_for_key
from lilbee.providers.model_ref import ProviderModelRef
from lilbee.providers.sdk_backend import CompletionRequest
from lilbee.providers.server_registry import RerankWire


def _route_model(ref: ProviderModelRef, api_base: str | None) -> str:
    """Format *ref* for liter-llm using the ``provider/model`` convention.

    API and local-server refs already carry their canonical prefix. A server
    preset speaks the OpenAI wire under its own name, so it goes out as
    ``openai/<model>``. A bare ``local`` ref forced through the SDK
    (``llm_provider=remote``) gets the prefix of whichever local server its
    ``api_base`` points at.
    """
    if ref.is_preset:
        return f"openai/{ref.name}"
    if ref.is_api or local_server_for_key(ref.provider) is not None:
        return ref.for_openai_prefix()
    if api_base and (spec := detect_local_server(api_base)) is not None:
        return spec.qualify(ref.name)
    return ref.name


def _wire_base_url(model: str, api_base: str | None) -> str | None:
    """Return the base URL to give liter-llm for a routed *model*, or ``None``.

    liter-llm drops one trailing ``/v1`` from an Ollama base and appends
    ``/api/...``, so Ollama gets the base as configured. An OpenAI-compatible
    server such as LM Studio serves ``/chat/completions`` under ``/v1``, so it
    gets the base with exactly one ``/v1``.
    """
    if not api_base:
        return None
    trimmed = api_base.rstrip("/")
    if model.startswith(OLLAMA.wire_prefix):
        return trimmed
    return f"{trimmed.removesuffix('/v1')}/v1"


# Rerank wire -> (liter-llm model prefix, whether the base ends in /v1).
_RERANK_WIRES: dict[RerankWire, tuple[str, bool]] = {
    RerankWire.V1: ("openai", True),
    RerankWire.TEI: ("huggingface", False),
}


def rerank_route(
    ref: ProviderModelRef, api_base: str | None, wire: RerankWire
) -> tuple[str, str | None]:
    """The model and the base URL for a rerank call; a preset follows its *wire*.

    The model is both the request model and the client hint, so the SDK reads the wire's
    request shape and sends the plain model name in the body.
    """
    if not ref.is_preset:
        model = _route_model(ref, api_base)
        return model, _wire_base_url(model, api_base)
    prefix, versioned = _RERANK_WIRES[wire]
    base = api_base.rstrip("/").removesuffix("/v1") if api_base else None
    return f"{prefix}/{ref.name}", f"{base}/v1" if base and versioned else base


EMPTY_MESSAGE_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


def _is_blank(text: Any) -> bool:
    """Whether *text* is not a string, or is empty or only whitespace."""
    return not isinstance(text, str) or not text.strip()


def _fill_blank_text(msg: dict[str, Any]) -> dict[str, Any]:
    """Give a blank text of a user or assistant turn the placeholder, as Anthropic needs.

    A string content and each text part of a list content are checked. Other parts and
    other roles pass through.
    """
    if msg.get("role") not in ("user", "assistant"):
        return msg
    content = msg.get("content")
    if isinstance(content, str):
        return {**msg, "content": EMPTY_MESSAGE_PLACEHOLDER} if _is_blank(content) else msg
    if isinstance(content, list):
        parts = [
            {**part, "text": EMPTY_MESSAGE_PLACEHOLDER}
            if isinstance(part, dict) and part.get("type") == "text" and _is_blank(part.get("text"))
            else part
            for part in content
        ]
        return {**msg, "content": parts}
    return msg


def _format_messages(
    messages: list[dict[str, Any]], *, fill_blank_turns: bool = False
) -> list[dict[str, Any]]:
    """Convert messages with inline image bytes into OpenAI content parts.

    Messages without ``images`` pass through untouched. With *fill_blank_turns*,
    blank text in a user or assistant turn carries a placeholder, as Anthropic rejects it.
    """
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        if "images" in msg:
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
    return [_fill_blank_text(msg) for msg in formatted] if fill_blank_turns else formatted


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
