"""Cohere Command-R: native ``<|START_ACTION|>...<|END_ACTION|>`` JSON wrapper."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.COHERE,
    template_markers=("<|START_ACTION|>",),
    output_format=OutputFormat.NATIVE,
)
