"""AI2 OLMo-3 family profile."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.OLMO3,
    template_markers=("<function_calls>",),
    output_format=OutputFormat.NATIVE,
    reason="OLMo-3 wraps tool calls in <function_calls> with name(key=value) syntax.",
)
