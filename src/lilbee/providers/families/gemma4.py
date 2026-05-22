"""Gemma-4 family profile (gemma-4-E2B-it via unsloth GGUFs)."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.GEMMA4,
    template_markers=('<|"|>',),
    name_patterns=(re.compile(r"gemma[\s\-_]?4", re.IGNORECASE),),
    output_format=OutputFormat.NATIVE,
    reason='Gemma-4 uses <|"|> tool-call quote marker, no override needed.',
)
