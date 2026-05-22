"""GLM-4.7: GLM-4.6 minus the newline after the function name."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.GLM47,
    template_markers=("<tool_call>{function-name}<arg_key>",),
    output_format=OutputFormat.NATIVE,
)
