"""Qwen3 (Qwen org GGUFs, template-tools intact): ChatML + native ``<tool_call>`` plus bare-JSON."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.QWEN3,
    template_markers=("<tool_call>", "</tool_call>"),
    name_patterns=(re.compile(r"qwen[\s\-_]?3", re.IGNORECASE),),
    output_format=OutputFormat.DUAL,
)
