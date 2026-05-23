"""Liquid Foundation Models v2: ``tool_call_start/end`` wrappers, kwarg-style args."""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.LFM2,
    # Some LFM2 GGUF quants (QuantFactory/LFM2-1.2B-Tool-GGUF) ship an empty
    # embedded chat_template, so the marker probe never fires; match the
    # ``lfm2`` architecture too so detection + tool-capability still land.
    template_markers=("<|tool_list_start|>",),
    architectures=("lfm2",),
    output_format=OutputFormat.NATIVE,
)
