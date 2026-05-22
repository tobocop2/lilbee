"""DeepSeek-V3.1 family profile."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

# Full-width pipes are part of DeepSeek's wire format; substituting ASCII | breaks
# extraction. ruff allowed-confusables already permits these in the lilbee repo.
PROFILE = FamilyProfile(
    family=TemplateFamily.DEEPSEEK_V31,
    template_markers=("<｜tool▁calls▁begin｜>",),
    output_format=OutputFormat.NATIVE,
    reason="DeepSeek-V3.1 native tool_call_begin / tool_sep / tool_call_end (full-width pipes).",
)
