"""``FamilyProfile`` dataclass + the enums that drive its behavior fields."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.providers.worker.response_parser.families import TemplateFamily


class OutputFormat(StrEnum):
    """Wire format the model uses for its tool-call output.

    The parser engine combines the family's native schema regex with a
    shared fallback keyed on this field; ``DUAL`` means accept either the
    family-native shape or bare-JSON-with-name (OpenAI-compatible clients
    routinely elicit the latter from any tool-trained model).
    """

    NATIVE = "native"
    BARE_JSON = "bare_json"
    DUAL = "dual"
    CHATML_TOOL_CALL = "chatml_tool_call"
    HARMONY = "harmony"


class StreamingPolicy(StrEnum):
    """Whether the family's chat_format preset supports streaming with tools.

    ``llama-cpp-python``'s ``chatml-function-calling`` and ``functionary-*``
    presets raise ``"Automatic streaming tool choice is not supported"`` when
    invoked with ``stream=True`` and ``tool_choice="auto"``. lilbee silently
    downgrades to non-streaming for those presets and synthesises a one-shot
    stream. This enum lets each profile declare its policy explicitly so the
    chat worker doesn't carry the per-preset bool table.
    """

    NATIVE = "native"
    DOWNGRADE_AUTO_TOOL_CHOICE = "downgrade_auto_tool_choice"
    NEEDS_SPECIFIC_TOOL_CHOICE = "needs_specific_tool_choice"


@dataclass(frozen=True)
class FamilyProfile:
    """One canonical record of every knob lilbee needs to handle one model family."""

    family: TemplateFamily
    template_markers: tuple[str, ...] = ()
    name_patterns: tuple[re.Pattern[str], ...] = ()
    ref_patterns: tuple[re.Pattern[str], ...] = ()
    architectures: tuple[str, ...] = ()
    chat_format_override: str | None = None
    hf_tokenizer_repo: str | None = None
    streaming_policy: StreamingPolicy = StreamingPolicy.NATIVE
    output_format: OutputFormat = OutputFormat.NATIVE
    sample_output_fixture: str | None = None
    reason: str = ""
    extras: Mapping[str, str] = field(default_factory=dict)

    def matches(self, metadata: Mapping[str, object] | None, ref: str | None) -> bool:
        """Return True if this profile matches the given GGUF identity.

        Match order within a single profile: ref_patterns (most specific) >
        name_patterns > template_markers > architectures. The registry's
        ordered ``match_order`` decides which PROFILE is consulted first
        when multiple could match.
        """
        return (
            _matches_any_pattern(ref, self.ref_patterns)
            or _matches_any_pattern(_gguf_name(metadata), self.name_patterns)
            or _matches_template_markers(
                _gguf_field(metadata, "chat_template"), self.template_markers
            )
            or _matches_architecture(_gguf_field(metadata, "architecture"), self.architectures)
        )


def _matches_any_pattern(value: str | None, patterns: tuple[re.Pattern[str], ...]) -> bool:
    if value is None or not patterns:
        return False
    return any(p.search(value) for p in patterns)


def _matches_template_markers(template: str | None, markers: tuple[str, ...]) -> bool:
    if not template or not markers:
        return False
    return all(marker in template for marker in markers)


def _matches_architecture(arch: str | None, architectures: tuple[str, ...]) -> bool:
    if not arch or not architectures:
        return False
    return arch.lower() in {a.lower() for a in architectures}


def _gguf_name(metadata: Mapping[str, object] | None) -> str | None:
    return _gguf_field(metadata, "name") or _gguf_field(metadata, "general.name")


def _gguf_field(metadata: Mapping[str, object] | None, key: str) -> str | None:
    if not metadata:
        return None
    value = metadata.get(key)
    return value if isinstance(value, str) else None
