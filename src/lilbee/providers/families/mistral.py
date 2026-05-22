"""Mistral (Nemo, Small, etc.): native ``[TOOL_CALLS]`` + JSON-array wrapper."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.MISTRAL,
    template_markers=("[TOOL_CALLS]",),
    output_format=OutputFormat.NATIVE,
)
