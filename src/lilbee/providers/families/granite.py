"""IBM Granite: native ``<|tool_call|>[json-array]`` and bare-JSON fallback."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.GRANITE,
    template_markers=("<|start_of_role|>",),
    output_format=OutputFormat.DUAL,
)
