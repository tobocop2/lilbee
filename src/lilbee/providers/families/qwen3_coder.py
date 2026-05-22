"""Qwen3-Coder family profile (sparse MoE coder variant)."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.QWEN3_CODER,
    template_markers=("<function=", "<parameter="),
    name_patterns=(re.compile(r"qwen[\s\-_]?3[\s\-_]?coder", re.IGNORECASE),),
    output_format=OutputFormat.NATIVE,
    reason="Qwen3-Coder native <tool_call><function=...><parameter=...> XML wrapper.",
)
