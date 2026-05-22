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
    """One model-identity-pattern -> chat_format rule.

    ``name_pattern`` matches against the GGUF's ``general.name``; ``ref_pattern``
    matches against the canonical HF-style ``<repo>/<file>.gguf`` ref. Either
    pattern can be ``None`` for rules that key on just one identifier.
    Functionary-style fine-tunes inherit their base model's ``general.name``
    (e.g. "Meta Llama 3.1 8B Instruct") so name-only matching misses them;
    the ref carries ``meetkai/functionary-small-v3.2-GGUF/...`` which
    disambiguates.

    ``family`` is the response-parser schema that matches the output shape
    the preset produces; pinning it here keeps prompt-time rendering and
    extraction-time parsing in lockstep, otherwise the model emits one
    wire format and lilbee tries to parse a different one. *reason*
    documents why the override exists so future readers can decide
    whether the entry still applies after llama-cpp / a quantizer ships a
    fix upstream.
    """

    name_pattern: re.Pattern[str] | None
    chat_format: str
    family: TemplateFamily
    reason: str
    ref_pattern: re.Pattern[str] | None = None


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
        name_pattern=re.compile(r"smollm[\s\-_]?3", re.IGNORECASE),
        chat_format="chatml-function-calling",
        family=TemplateFamily.SMOLLM,
        reason=(
            "SmolLM3 GGUFs embed a chat template that uses the {% generation %}"
            " Jinja tag, which llama-cpp-python's bundled Jinja parser does "
            "not recognize and rejects with TemplateSyntaxError before any "
            "inference runs. chatml-function-calling renders the prompt with "
            "the ChatML markers SmolLM3 was trained on and emits <tool_call> "
            "wrappers compatible with response_parser/schemas/smollm.json."
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
    # NOTE: Functionary v3.x GGUFs (meetkai/functionary-small-v3.2-GGUF) embed
    # the Llama-3.1 chat template natively and don't get an override here.
    # llama-cpp-python's functionary-v2 preset requires hf_tokenizer_path at
    # Llama init for its custom tool tokenization, which lilbee can't supply
    # offline. Without the override, the model loads with its embedded
    # Llama-3.1 template, family detection picks LLAMA3, and the bare-JSON
    # parser arm extracts the tool calls the model emits via the standard
    # Llama-3.1 tools prompt.
)


def _match(metadata: Mapping[str, object] | None, *, ref: str | None) -> _ChatFormatOverride | None:
    name_value = None
    if metadata:
        candidate = metadata.get("name") or metadata.get("general.name")
        if isinstance(candidate, str):
            name_value = candidate
    for rule in _OVERRIDES:
        if (
            rule.name_pattern is not None
            and name_value is not None
            and rule.name_pattern.search(name_value)
        ):
            return rule
        if rule.ref_pattern is not None and ref is not None and rule.ref_pattern.search(ref):
            return rule
    return None


def resolve_chat_format_override(
    metadata: Mapping[str, object] | None, *, ref: str | None = None
) -> str | None:
    """Return a llama-cpp ``chat_format`` preset to override the embedded template.

    Returns ``None`` when the model's embedded template should be used as-is.
    Only consults stable, declarative GGUF metadata keys; never loads the
    model. Safe to call from the ``_supports_tools`` cache key. *ref* is the
    canonical ``<repo>/<file>.gguf`` form, consulted when a rule keys on
    repo path (e.g. Functionary fine-tunes inherit Meta's general.name).
    """
    rule = _match(metadata, ref=ref)
    return rule.chat_format if rule is not None else None


def resolve_override_family(
    metadata: Mapping[str, object] | None, *, ref: str | None = None
) -> TemplateFamily | None:
    """Return the response-parser family that matches the override's output shape.

    When an override applies, the model emits the preset's wire format
    instead of whatever the (stripped) embedded template would have driven.
    The parser has to follow suit or the tool calls leak into content.
    Returns ``None`` when no override applies, so the caller can fall back
    to template-based detection.
    """
    rule = _match(metadata, ref=ref)
    return rule.family if rule is not None else None
