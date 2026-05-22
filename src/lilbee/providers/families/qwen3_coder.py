"""Qwen3-Coder (sparse-MoE coder variant): native ``<function=>`` plus ``<parameter=>`` XML."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.QWEN3_CODER,
    template_markers=("<function=", "<parameter="),
    name_patterns=(re.compile(r"qwen[\s\-_]?3[\s\-_]?coder", re.IGNORECASE),),
    output_format=OutputFormat.NATIVE,
)
