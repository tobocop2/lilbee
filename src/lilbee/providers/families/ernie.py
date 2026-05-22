"""Baidu ERNIE-4.5 family profile."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.ERNIE,
    template_markers=("<|begin_of_sentence|>", "<|end_of_sentence|>"),
    output_format=OutputFormat.NATIVE,
    reason="ERNIE-4.5 native <tool_call>{json}</tool_call> wrapper inside its own sentence frame.",
)
