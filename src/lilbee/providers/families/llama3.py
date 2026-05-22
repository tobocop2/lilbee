"""Llama-3 / Llama-3.1: native ``<|python_tag|>{json}`` plus bare-JSON fallback.

OpenAI-compatible clients prompt llama-3.1 via the standard ``tools`` parameter
without invoking the python-tag hint, so the model emits bare ``{"name": ...,
"arguments": ...}`` JSON in that path. ``DUAL`` accepts either shape.
"""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.LLAMA3,
    template_markers=("<|python_tag|>",),
    output_format=OutputFormat.DUAL,
)
