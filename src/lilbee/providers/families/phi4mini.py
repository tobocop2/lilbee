"""Microsoft Phi-4-mini: native ``functools[...]`` array wrapper."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.PHI4MINI,
    template_markers=("<|tool|>", "<|/tool|>"),
    output_format=OutputFormat.NATIVE,
)
