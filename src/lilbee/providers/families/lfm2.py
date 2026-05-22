"""Liquid Foundation Models v2: ``tool_call_start/end`` wrappers, kwarg-style args."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.LFM2,
    template_markers=("<|tool_list_start|>",),
    output_format=OutputFormat.NATIVE,
)
