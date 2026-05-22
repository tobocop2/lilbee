"""GLM-4.5 / 4.6 family profile."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.GLM46,
    template_markers=("<arg_key>", "<arg_value>"),
    output_format=OutputFormat.NATIVE,
    reason="GLM 4.5/4.6 <tool_call>NAME plus <arg_key>K</arg_key><arg_value>V</arg_value>.",
)
