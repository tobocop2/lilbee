"""``FamilyProfile`` dataclass plus the enums that drive its behavior fields."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from lilbee.providers.chat_format import LlamaCppChatFormatPreset

if TYPE_CHECKING:
    from lilbee.providers.worker.response_parser.families import TemplateFamily


class OutputFormat(StrEnum):
    """Wire format the model uses for its tool-call output."""

    NATIVE = "native"
    BARE_JSON = "bare_json"
    DUAL = "dual"
    CHATML_TOOL_CALL = "chatml_tool_call"
    HARMONY = "harmony"


class StreamingPolicy(StrEnum):
    """Whether the family's ``chat_format`` preset supports streaming with tools."""

    NATIVE = "native"
    DOWNGRADE_AUTO_TOOL_CHOICE = "downgrade_auto_tool_choice"


@dataclass(frozen=True)
class FamilyProfile:
    """One canonical record of every knob lilbee needs to handle one model family."""

    family: TemplateFamily
    template_markers: tuple[str, ...] = ()
    name_patterns: tuple[re.Pattern[str], ...] = ()
    ref_patterns: tuple[re.Pattern[str], ...] = ()
    architectures: tuple[str, ...] = ()
    chat_format_override: LlamaCppChatFormatPreset | None = None
    hf_tokenizer_repo: str | None = None
    render_with_hf_template: bool = False
    context_length_override: int | None = None
    streaming_policy: StreamingPolicy = StreamingPolicy.NATIVE
    output_format: OutputFormat = OutputFormat.NATIVE

    def matches(self, metadata: Mapping[str, object] | None, ref: str | None) -> bool:
        """True if any of the profile's identity hints fires on this GGUF.

        Hints are ORed: any single hit wins. Profile precedence when several
        could match a model is decided by the package-level ``ALL_PROFILES``
        ordering, not by any rank inside the profile.
        """
        return (
            _matches_any_pattern(ref, self.ref_patterns)
            or _matches_any_pattern(_gguf_field(metadata, "name"), self.name_patterns)
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


def _gguf_field(metadata: Mapping[str, object] | None, key: str) -> str | None:
    if not metadata:
        return None
    value = metadata.get(key)
    return value if isinstance(value, str) else None
