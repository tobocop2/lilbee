"""MeetKai Functionary v3 family profile."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.FUNCTIONARY_V3,
    template_markers=(">>>all",),
    ref_patterns=(re.compile(r"functionary[^/]*v3", re.IGNORECASE),),
    chat_format_override="functionary-v2",
    hf_tokenizer_repo="meetkai/functionary-small-v3.2",
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.NATIVE,
    reason=(
        "Functionary v3 GGUFs inherit Llama-3.1's general.name; match by repo. "
        "The functionary-v2 preset needs the HF tokenizer for its special tool "
        "tokens, and the model emits >>>name\\n{json} wire shape."
    ),
)
