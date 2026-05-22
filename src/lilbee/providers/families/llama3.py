"""Llama-3.1 family profile."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.LLAMA3,
    template_markers=("<|python_tag|>",),
    output_format=OutputFormat.DUAL,
    reason=(
        "Llama-3 native <|python_tag|>{json} wrapper; accept bare JSON too since "
        "OpenAI-compatible clients prompt llama-3.1 via the tools parameter without "
        "invoking the python-tag hint."
    ),
)
