"""OpenAI gpt-oss family profile."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.GPT_OSS,
    template_markers=("<|channel|>", "<|call|>"),
    output_format=OutputFormat.HARMONY,
    reason="OpenAI gpt-oss Harmony format: <|channel|>commentary to=functions...<|call|>.",
)
