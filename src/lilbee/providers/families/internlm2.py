"""InternLM2: minimal embedded template, detected via GGUF architecture only."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.INTERNLM2,
    architectures=("internlm2", "internlm"),
    output_format=OutputFormat.NATIVE,
)
