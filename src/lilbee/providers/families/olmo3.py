"""AI2 OLMo-3: ``<function_calls>`` wrapper with ``name(key=value)`` syntax."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.OLMO3,
    template_markers=("<function_calls>",),
    output_format=OutputFormat.NATIVE,
)
