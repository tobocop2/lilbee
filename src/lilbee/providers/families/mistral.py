"""Mistral (Nemo, Small, etc.): native ``[TOOL_CALLS]`` + JSON-array wrapper."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.MISTRAL,
    template_markers=("[TOOL_CALLS]",),
    # Mistral-Nemo (and some quants) emit the tool call as a bare JSON array
    # ``[{"name": ..., "parameters": ...}]`` with no ``[TOOL_CALLS]`` prefix,
    # so the native marker regex misses it. DUAL runs the bare-JSON fallback
    # after the native pass to catch that shape.
    output_format=OutputFormat.DUAL,
)
