"""Kimi K2: native ``tool_call_begin/end`` with ``tool_call_argument_begin`` separator."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.KIMI_K2,
    template_markers=("<|tool_calls_section_begin|>", "<|tool_call_argument_begin|>"),
    output_format=OutputFormat.NATIVE,
)
