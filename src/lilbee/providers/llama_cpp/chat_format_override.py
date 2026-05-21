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
``llama_cpp.llama_chat_format.LlamaChatCompletionHandlerRegistry``;
each entry includes the lineage that justifies it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class _ChatFormatOverride:
    """One (name-pattern -> chat_format) rule.

    *reason* documents why the override exists so future readers can decide
    whether the entry still applies after llama-cpp / a quantizer ships a
    fix upstream.
    """

    name_pattern: re.Pattern[str]
    chat_format: str
    reason: str


# Ordered most-specific-first. The first match wins; downstream rules are
# only consulted when the upstream did not fire.
_OVERRIDES: tuple[_ChatFormatOverride, ...] = (
    _ChatFormatOverride(
        name_pattern=re.compile(r"hermes[\s\-_]?3", re.IGNORECASE),
        chat_format="chatml-function-calling",
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
        reason="Functionary-v1 community GGUFs commonly drop the template.",
    ),
    _ChatFormatOverride(
        name_pattern=re.compile(r"functionary[\s\-_]?v2", re.IGNORECASE),
        chat_format="functionary-v2",
        reason="Functionary-v2 community GGUFs commonly drop the template.",
    ),
)


def resolve_chat_format_override(metadata: Mapping[str, object] | None) -> str | None:
    """Return a llama-cpp ``chat_format`` preset to override the embedded template.

    Returns ``None`` when the model's embedded template should be used as-is.
    Only consults stable, declarative GGUF metadata keys; never loads the
    model. Safe to call from the ``_supports_tools`` cache key.
    """
    if not metadata:
        return None
    name = metadata.get("name") or metadata.get("general.name")
    if not isinstance(name, str):
        return None
    for rule in _OVERRIDES:
        if rule.name_pattern.search(name):
            return rule.chat_format
    return None
