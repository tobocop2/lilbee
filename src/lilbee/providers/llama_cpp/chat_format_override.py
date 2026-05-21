"""Map GGUFs whose embedded chat template strips tool blocks to a llama-cpp preset.

Community quantizers (bartowski et al) frequently ship GGUFs whose
``tokenizer.chat_template`` field is the bare ChatML / Llama-INST form with
the ``{% if tools %}`` blocks deleted. lilbee's ``_supports_tools_cached``
probe greps the embedded template, so without an override the affected
models are reported as not tool-capable even when the base model was
trained for tool calling.

This module names a small, conservative set of models for which lilbee
should bypass the embedded template and use a llama-cpp built-in chat
format. The override is keyed on a regex match against the GGUF's
``general.name`` field, which is stable across quantizers in practice.
The chat_format strings come from
``llama_cpp.llama_chat_format.LlamaChatCompletionHandlerRegistry``; each
entry pairs the chat_format with the response-parser family that matches
the output wire shape that preset produces, so swap-time and
extraction-time stay in sync.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from lilbee.providers.worker.response_parser.families import TemplateFamily


@dataclass(frozen=True)
class _ChatFormatOverride:
    """One (name-pattern -> chat_format) rule.

    ``family`` is the response-parser schema that matches the output shape
    the preset produces; pinning it here keeps prompt-time rendering and
    extraction-time parsing in lockstep, otherwise the model emits one
    wire format and lilbee tries to parse a different one. *reason*
    documents why the override exists so future readers can decide
    whether the entry still applies after llama-cpp / a quantizer ships a
    fix upstream.
    """

    name_pattern: re.Pattern[str]
    chat_format: str
    family: TemplateFamily
    reason: str


# Ordered most-specific-first. The first match wins; downstream rules are
# only consulted when the upstream did not fire.
_OVERRIDES: tuple[_ChatFormatOverride, ...] = (
    _ChatFormatOverride(
        name_pattern=re.compile(r"hermes[\s\-_]?3", re.IGNORECASE),
        chat_format="chatml-function-calling",
        family=TemplateFamily.HERMES,
        reason=(
            "Hermes-3 community GGUFs (bartowski, NousResearch) embed a bare "
            "ChatML template with no tool blocks. The chatml-function-calling "
            "preset emits <tool_call> markers that match the hermes schema "
            "in response_parser/schemas/hermes.json."
        ),
    ),
    _ChatFormatOverride(
        name_pattern=re.compile(r"functionary[\s\-_]?v1", re.IGNORECASE),
        chat_format="functionary-v1",
        family=TemplateFamily.FUNCTIONARY_V3,
        reason="Functionary-v1 community GGUFs commonly drop the template.",
    ),
    _ChatFormatOverride(
        name_pattern=re.compile(r"functionary[\s\-_]?v2", re.IGNORECASE),
        chat_format="functionary-v2",
        family=TemplateFamily.FUNCTIONARY_V3,
        reason="Functionary-v2 community GGUFs commonly drop the template.",
    ),
)


def _match(metadata: Mapping[str, object] | None) -> _ChatFormatOverride | None:
    if not metadata:
        return None
    name = metadata.get("name") or metadata.get("general.name")
    if not isinstance(name, str):
        return None
    for rule in _OVERRIDES:
        if rule.name_pattern.search(name):
            return rule
    return None


def resolve_chat_format_override(metadata: Mapping[str, object] | None) -> str | None:
    """Return a llama-cpp ``chat_format`` preset to override the embedded template.

    Returns ``None`` when the model's embedded template should be used as-is.
    Only consults stable, declarative GGUF metadata keys; never loads the
    model. Safe to call from the ``_supports_tools`` cache key.
    """
    rule = _match(metadata)
    return rule.chat_format if rule is not None else None


def resolve_override_family(metadata: Mapping[str, object] | None) -> TemplateFamily | None:
    """Return the response-parser family that matches the override's output shape.

    When an override applies, the model emits the preset's wire format
    instead of whatever the (stripped) embedded template would have driven.
    The parser has to follow suit or the tool calls leak into content.
    Returns ``None`` when no override applies, so the caller can fall back
    to template-based detection.
    """
    rule = _match(metadata)
    return rule.family if rule is not None else None
